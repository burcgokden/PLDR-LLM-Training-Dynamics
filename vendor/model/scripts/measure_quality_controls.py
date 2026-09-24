#!/usr/bin/env python
"""Paired context-destruction and operator-replacement measurements on fixed weights."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from model_rg.native import NativeModel
from model_rg.training import TrainingModel
from model_rg.controlled import bind_run, device_name, stable_kl
from model_rg.provenance import sha256, write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--run-id',required=True)
    ap.add_argument('--device',default='cuda:0');a=ap.parse_args();a.device=device_name(a.device)
    root=Path(a.root);study=root/'controlled-study-20260905';data=study/'data/short'
    out=study/'runs'/('quality-'+a.run_id);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.run_id.startswith('soc'):
        source=root/f'assets/PLDR-LLM-v51-SOC-110M-{a.run_id[-1]}'
        weights=source/'model.safetensors';m=NativeModel(source,a.device)
    else:
        weights=study/'runs'/a.run_id/'final-training-state.pt'
        ck=torch.load(weights,map_location='cpu',weights_only=True);ca=ck['arguments']
        source=root/'assets/PLDR-LLM-v51-SOC-110M-1'
        m=TrainingModel(source,ca['heads'],ca['seed'],a.device);m.model.load_state_dict(ck['model'])
        del ck;m.model.eval().requires_grad_(False)
    bind_run(out,[weights,data/'tokens.npy',data/'offsets.npy',source/'modeling_pldrllm.py',study/'quality-protocol.json'],vars(a))
    t=np.load(data/'tokens.npy');offset=np.load(data/'offsets.npy')
    crops=t[np.arange(len(t))[:,None],offset[:,None]+np.arange(65)];crops=crops[512:2048]
    swapped=np.roll(crops[:,:64],1,axis=0);last=swapped.copy();last[:,-1]=crops[:,63]
    records={};start=time.time();base=None
    with torch.no_grad():
        ref=m.forward(torch.tensor(t[:1,offset[0]:offset[0]+64],dtype=torch.long,device=a.device),capture=True)
        fixed=torch.stack([torch.stack([att[0],att[1],att[5]],0) for att in ref.pldr_attentions])
        for mode,inputs in [('intact',crops[:,:64]),('other_document',swapped),('other_document_keep_last',last),('frozen_operator',crops[:,:64])]:
            if mode=='frozen_operator':
                d=m.model.decoder;d.past_G_values=fixed;d.past_G_values_status=torch.ones(5,dtype=torch.bool,device=a.device);d.custom_G_type='external'
                for layer in d.dec_layers:
                    layer.mha1.custom_G_type='external';layer.mha1.plgatt_layer.custom_G_type='external'
            zz=[]
            for j in range(0,len(crops),32):
                zz.append(m.logits(torch.tensor(inputs[j:j+32],dtype=torch.long,device=a.device)).cpu().double())
            z=torch.cat(zz);lp=z.log_softmax(-1);p=lp.exp();target=torch.tensor(crops[:,64],dtype=torch.long)
            if base is None:base=z
            records[mode+'_nll']=(-lp.gather(1,target[:,None]).squeeze(1)).numpy()
            records[mode+'_entropy']=(-(p*lp).sum(-1)).numpy()
            records[mode+'_kl']=stable_kl(base,z).numpy()
            records[mode+'_max_logit_difference']=(base-z).abs().max(-1).values.numpy()
            records[mode+'_argmax_changed']=(base.argmax(-1)!=z.argmax(-1)).numpy()
    np.savez_compressed(out/'quality.npz',**records)
    write_json(out/'manifest.json',dict(schema='quality-controls-v1',arguments=vars(a),evaluation_rows=[512,2048],anchor_row=0,
        modes=['intact','other_document','other_document_keep_last','frozen_operator'],
        raw_sha256=sha256(out/'quality.npz'),binding_sha256=sha256(out/'binding.json'),seconds=time.time()-start,
        peak_cuda_gb=torch.cuda.max_memory_allocated(a.device)/2**30))
    print(a.run_id,'context effect',float(np.mean(records['other_document_nll']-records['intact_nll'])),
          'beyond last',float(np.mean(records['other_document_keep_last_nll']-records['intact_nll'])),flush=True)


if __name__=='__main__':main()
