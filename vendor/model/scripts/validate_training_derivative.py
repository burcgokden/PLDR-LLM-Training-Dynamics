#!/usr/bin/env python
"""Independent reverse-mode check of the saved full-parameter forward derivative."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from model_rg.training import TrainingModel
from model_rg.provenance import environment, sha256, source_manifest, write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True)
    ap.add_argument('--response',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--device',default='cuda:0');args=ap.parse_args()
    run=Path(args.run);response=Path(args.response);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    start=time.time();sources=source_manifest()
    ck=torch.load(run/'final-training-state.pt',map_location='cpu',weights_only=True)
    a=ck['arguments'];m=TrainingModel(a['source'],a['heads'],a['seed'],args.device)
    m.model.load_state_dict(ck['model'])
    optimizer=torch.optim.AdamW(m.model.parameters(),lr=3e-4,betas=(.9,.95),eps=1e-8,
                               weight_decay=.01,foreach=False)
    optimizer.load_state_dict(ck['optimizer'])
    tokens=np.load(Path(a['data'])/'tokens.npy');sampling=np.load(run/'sampling.npz')
    k=ck['step'];rows=sampling['rows'][k];offsets=sampling['offsets'][k]
    batch=torch.tensor(tokens[rows[:,None],offsets[:,None]+np.arange(65)[None]],dtype=torch.long,device=args.device)
    m.model.train();optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(m.logits(batch[:,:64]),batch[:,64]).backward()
    torch.nn.utils.clip_grad_norm_(m.model.parameters(),1.,error_if_nonfinite=True)
    theta={n:p.detach().clone() for n,p in m.model.named_parameters()};optimizer.step()
    delta={n:p.detach()-theta[n] for n,p in m.model.named_parameters()}
    del optimizer,ck
    m.model.zero_grad(set_to_none=True);m.model.eval();m.capture=False;m.eta=None
    raw=np.load(response/'response.npz')
    ids=torch.tensor(tokens[raw['probe_rows'],:64],dtype=torch.long,device=args.device)
    def f(params):
        return torch.func.functional_call(m.model,params,(ids,),
               {'use_cache':False,'logits_to_keep':1}).logits[:,-1]
    z,vjp=torch.func.vjp(f,theta)
    dz=torch.tensor(raw['logit_derivative'],device=args.device)
    p=torch.tensor(raw['baseline_logits'],device=args.device).softmax(-1)
    rng=np.random.default_rng(90971);left=[];right=[]
    for _ in range(3):
        w=torch.tensor(rng.normal(size=tuple(z.shape)),device=args.device)*p.sqrt()
        w-=p*w.sum(-1,keepdim=True)
        weight=w.float()
        gradient=vjp(weight)[0]
        left.append(float((weight.double()*dz).sum()))
        right.append(float(sum((gradient[n].double()*delta[n].double()).sum() for n in theta)))
        del gradient
    denom=float(np.sqrt(raw['curvature'].sum()))
    residual=np.abs(np.array(left)-np.array(right))/denom
    np.savez_compressed(out/'duality.npz',left=left,right=right,normalized_residual=residual)
    write_json(out/'manifest.json',{'schema':'model-rg-training-derivative-v1','source_files':sources,
        'arguments':vars(args),'environment':environment(),'seconds':time.time()-start,
        'training_manifest_sha256':sha256(run/'manifest.json'),
        'response_manifest_sha256':sha256(response/'manifest.json'),
        'raw_sha256':sha256(out/'duality.npz'),'test_directions':3,'random_seed':90971,
        'baseline_max_abs_error':float((z.detach().double()-torch.tensor(raw['baseline_logits'],device=args.device)).abs().max()),
        'max_normalized_duality_error':float(residual.max())})
    print(a['heads'],a['seed'],'duality error',float(residual.max()),flush=True)


if __name__=='__main__':main()
