#!/usr/bin/env python
"""Predict inference changes along one actual full-parameter AdamW update."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from model_rg.training import TrainingModel
from model_rg.provenance import environment, sha256, source_manifest, write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args();run=Path(args.run);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source_files=source_manifest();start=time.time()
    # The training state is produced locally by this repository and is not an untrusted download.
    ck=torch.load(run/'final-training-state.pt',map_location='cpu',weights_only=True)
    a=ck['arguments'];m=TrainingModel(a['source'],a['heads'],a['seed'],args.device)
    m.model.load_state_dict(ck['model'])
    optimizer=torch.optim.AdamW(m.model.parameters(),lr=3e-4,betas=(.9,.95),eps=1e-8,
                               weight_decay=.01,foreach=False)
    optimizer.load_state_dict(ck['optimizer'])
    tokens=np.load(Path(a['data'])/'tokens.npy');sampling=np.load(run/'sampling.npz')
    index=ck['step'];rows=sampling['rows'][index];offsets=sampling['offsets'][index]
    batch=tokens[rows[:,None],offsets[:,None]+np.arange(65)[None,:]]
    ids=torch.tensor(batch[:,:64],dtype=torch.long,device=args.device)
    target=torch.tensor(batch[:,64],dtype=torch.long,device=args.device)
    m.model.train();optimizer.zero_grad(set_to_none=True)
    loss=torch.nn.functional.cross_entropy(m.logits(ids),target);loss.backward()
    gradnorm=torch.nn.utils.clip_grad_norm_(m.model.parameters(),1.,error_if_nonfinite=True)
    theta={n:p.detach().clone() for n,p in m.model.named_parameters()}
    optimizer.step()
    delta={n:p.detach()-theta[n] for n,p in m.model.named_parameters()}
    del optimizer,ck
    m.model.zero_grad(set_to_none=True);m.model.eval()
    m.eta=None;m.capture=False
    probe_rows=np.arange(4096,4112)
    ids=torch.tensor(tokens[probe_rows,:64],dtype=torch.long,device=args.device)
    def logits(parameters):
        return torch.func.functional_call(m.model,parameters,(ids,),
               {'use_cache':False,'logits_to_keep':1}).logits[:,-1]
    z,dz=torch.func.jvp(logits,(theta,),(delta,))
    z=z.detach().double();dz=dz.detach().double()
    prob=z.softmax(-1);centered=dz-(prob*dz).sum(-1,keepdim=True)
    curvature=(prob*centered.square()).sum(-1)
    epsilons=np.array([.03,.1,.3,1.]);kl=[];perturbed=[]
    with torch.no_grad():
        for epsilon in epsilons:
            zz=logits({n:p+epsilon*delta[n] for n,p in theta.items()}).double()
            kl.append((prob*(z.log_softmax(-1)-zz.log_softmax(-1))).sum(-1).cpu().numpy())
            perturbed.append(zz.cpu().numpy())
        h=.05
        zp=logits({n:p+h*delta[n] for n,p in theta.items()}).double()
        zm=logits({n:p-h*delta[n] for n,p in theta.items()}).double()
        finite=(zp-zm)/(2*h);error=finite-dz
        error-= (prob*error).sum(-1,keepdim=True)
        derivative_error=((prob*error.square()).sum(-1)/curvature).sqrt()
    raw=out/'response.npz'
    np.savez_compressed(raw,probe_rows=probe_rows,epsilon=epsilons,kl=np.stack(kl),
                        curvature=curvature.cpu().numpy(),baseline_logits=z.cpu().numpy(),
                        logit_derivative=dz.cpu().numpy(),perturbed_logits=np.stack(perturbed),
                        central_derivative=finite.cpu().numpy())
    relative=np.abs(np.stack(kl)/(.5*epsilons[:,None]**2*curvature.cpu().numpy()[None])-1)
    write_json(out/'manifest.json',{'schema':'model-rg-training-response-v1','arguments':vars(args),
        'training_manifest_sha256':sha256(run/'manifest.json'),'source_files':source_files,
        'raw_sha256':sha256(raw),'environment':environment(),'seconds':time.time()-start,
        'peak_cuda_gb':torch.cuda.max_memory_allocated(args.device)/2**30,
        'parameters':sum(p.numel() for p in m.model.parameters()),'head_count':a['heads'],
        'training_seed':a['seed'],'update_loss':float(loss.detach()),'gradient_norm':float(gradnorm),
        'median_relative_kl_error':np.median(relative,axis=1).tolist(),
        'max_directional_derivative_fisher_error':float(derivative_error.max()),
        'update_norm':float(sum(d.double().square().sum() for d in delta.values()).sqrt())})
    print('Completed full training response',a['heads'],a['seed'],np.median(relative,axis=1),flush=True)


if __name__=='__main__':main()
