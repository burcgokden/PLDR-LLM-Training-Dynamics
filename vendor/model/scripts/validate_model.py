#!/usr/bin/env python
"""Native loop equivalence, blocked tangent composition, and complete G freezing."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from model_rg.native import NativeModel
from model_rg.provenance import environment, sha256, source_manifest, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--data',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0')
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    m=NativeModel(args.model,args.device);d=m.model.decoder
    tokens=np.load(Path(args.data)/'tokens.npy');shape=(5,14);start=time.time()
    rng=np.random.default_rng(2009);direction=torch.tensor(rng.normal(size=shape),dtype=torch.float32,device=args.device)
    zero=torch.zeros(shape,device=args.device)
    checks=[]
    for row in tokens[1024:1028,:128]:
        ids=torch.tensor(row,dtype=torch.long,device=args.device)[None]
        native=m.logits(ids).detach()
        s=ids.shape[1];positions=torch.arange(s,device=args.device)[None]
        mask=torch.triu(torch.full((s,s),torch.finfo(torch.float32).min,device=args.device),diagonal=1)[None,None]
        x0=d.layernorm1(d.embedding(ids)*torch.sqrt(torch.tensor(d.d_model,dtype=torch.float32,device=args.device)))
        def stages(x,eta,a,b):
            m.eta=eta;m.capture=False
            for l in range(a,b):
                x=d.dec_layers[l](x,mask,position_embeddings=None,position_ids=positions,use_cache=False,
                                  past_G_values=d.past_G_values,past_G_values_status=d.past_G_values_status)[0]
            return x
        def whole(eta):return m.model.final_layer(stages(x0,eta,0,5)[:,-1])
        primal,jwhole=torch.autograd.functional.jvp(whole,zero,direction)
        for split in [2,4]:
            xmid,jmid=torch.autograd.functional.jvp(lambda eta:stages(x0,eta,0,split),zero,direction)
            _,jblocked=torch.autograd.functional.jvp(lambda x,eta:m.model.final_layer(stages(x,eta,split,5)[:,-1]),
                                                     (xmid,zero),(jmid,direction))
            prob=native[0].double().softmax(-1)
            delta=(jwhole-jblocked)[0].double();base=jwhole[0].double()
            delta-=prob@delta;base-=prob@base
            checks.append({'split':split,'primal_max_abs_error':float((primal-native).abs().max()),
                           'tangent_relative_fisher_error':float(((prob@delta.square())/(prob@base.square())).sqrt())})
    # One calibration context defines every frozen deductive tensor.
    refids=torch.tensor(tokens[:1,:128],dtype=torch.long,device=args.device)
    with torch.no_grad():
        ref=m.forward(refids,capture=True)
        fixed=torch.stack([torch.stack([att[0],att[1],att[5]],dim=0) for att in ref.pldr_attentions])
        selected=tokens[512:768]
        baseline=[]
        for i in range(0,len(selected),16):
            ids=torch.tensor(selected[i:i+16,:128],dtype=torch.long,device=args.device)
            baseline.append(m.logits(ids).cpu())
        baseline=torch.cat(baseline).double()
        d.past_G_values=fixed;d.past_G_values_status=torch.ones(5,dtype=torch.bool,device=args.device)
        d.custom_G_type='external'
        for layer in d.dec_layers:
            layer.mha1.custom_G_type='external';layer.mha1.plgatt_layer.custom_G_type='external'
        frozen=[]
        for i in range(0,len(selected),16):
            ids=torch.tensor(selected[i:i+16,:128],dtype=torch.long,device=args.device)
            frozen.append(m.logits(ids).cpu())
        frozen=torch.cat(frozen).double()
        delta=frozen-baseline;prob=baseline.softmax(-1)
        kl=(prob*(baseline.log_softmax(-1)-frozen.log_softmax(-1))).sum(-1)
        target=torch.tensor(selected[:,128],dtype=torch.long)
        nll_delta=(-frozen.log_softmax(-1)+baseline.log_softmax(-1)).gather(1,target[:,None]).squeeze(1)
    np.savez_compressed(out/'native-checks.npz',baseline_logits=baseline.numpy(),frozen_logits=frozen.numpy(),
                         target=target.numpy(),frozen_deductive_tensors=fixed.cpu().numpy())
    write_json(out/'result.json',{'schema':'model-rg-native-checks-v1','environment':environment(),
        'arguments':vars(args),'source_files':source_manifest(),'composition_checks':checks,
        'frozen_operator':{'calibration_document':0,'confirmation_documents':256,
           'max_abs_logit_change':float(delta.abs().max()),'mean_kl':float(kl.mean()),'max_kl':float(kl.max()),
           'minimum_kl_roundoff':float(kl.min()),'argmax_changes':int((frozen.argmax(-1)!=baseline.argmax(-1)).sum()),
           'mean_nll_change':float(nll_delta.mean()),'max_abs_nll_change':float(nll_delta.abs().max())},
        'model_sha256':sha256(Path(args.model)/'model.safetensors'),'seconds':time.time()-start,
        'raw_sha256':sha256(out/'native-checks.npz')})
    print('Completed native validation',time.time()-start,flush=True)


if __name__=='__main__':main()
