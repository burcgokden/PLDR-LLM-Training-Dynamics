#!/usr/bin/env python
"""Predict held-out optimizer-update responses from a conditional batch ensemble."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from model_rg.training import TrainingModel
from model_rg.controlled import bind_run, device_name, stable_kl
from model_rg.provenance import sha256, write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--run-id',required=True)
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--batches',type=int,default=16);ap.add_argument('--label',default='smooth')
    a=ap.parse_args();a.device=device_name(a.device);root=Path(a.root);study=root/'controlled-study-20260905'
    run=study/'runs'/a.run_id;out=study/'runs'/('drift-'+a.label+'-'+a.run_id);out.mkdir(parents=True,exist_ok=False)
    data=root/'data/refinedweb-4608';probe=study/'data/short';source=root/'assets/PLDR-LLM-v51-SOC-110M-1'
    bind_run(out,[run/'manifest.json',run/'final-training-state.pt',data/'tokens.npy',probe/'tokens.npy',
                  probe/'offsets.npy',source/'modeling_pldrllm.py',study/'protocol.json',study/'smooth-response-protocol.json'],vars(a))
    torch.set_num_threads(3);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ck=torch.load(run/'final-training-state.pt',map_location='cpu',weights_only=True);ca=ck['arguments']
    m=TrainingModel(source,ca['heads'],ca['seed'],a.device);m.model.load_state_dict(ck['model'])
    md=TrainingModel(source,ca['heads'],ca['seed'],a.device);md.model.load_state_dict(ck['model'])
    md.model.double().eval().requires_grad_(False)
    def attention(module,query,key,value,attention_mask,scaling,dropout=0.,**kwargs):
        weights=torch.matmul(query,key.transpose(2,3))*scaling
        if attention_mask is not None:weights=weights+attention_mask
        weights=weights.softmax(-1)
        return torch.matmul(weights,value).transpose(1,2).contiguous(),weights
    original_iswiglu=md.module.iSwiGLU
    original_forwards=[]
    md.module.iSwiGLU=lambda x:x*x*torch.sigmoid(x)
    md._attention=attention
    # Equivalent smooth arithmetic supports nested forward AD without fused backward kernels.
    for module in md.model.modules():
        if module.__class__.__name__ in ['SiLUActivation','SiLU']:
            original_forwards.append((module,module.forward))
            module.forward=lambda x:x*torch.sigmoid(x)
        elif isinstance(module,torch.nn.LayerNorm):
            original_forwards.append((module,module.forward))
            def layernorm(x,mod=module):
                axes=tuple(range(x.ndim-len(mod.normalized_shape),x.ndim))
                centered=x-x.mean(axes,keepdim=True)
                result=centered*torch.rsqrt(centered.square().mean(axes,keepdim=True)+mod.eps)
                if mod.weight is not None:result=result*mod.weight
                if mod.bias is not None:result=result+mod.bias
                return result
            module.forward=layernorm
        elif module.__class__.__name__=='RotaryPositionalEmbeddings':
            original_forwards.append((module,module.forward))
            def rotary(x,*,input_pos=None,mod=module):
                seq=x.size(1);cache=mod.cache[:seq] if input_pos is None else mod.cache[input_pos]
                shaped=x.reshape(*x.shape[:-1],-1,2)
                cache=cache.view(-1,seq,1,shaped.size(3),2).to(x.dtype)
                y=torch.stack([shaped[...,0]*cache[...,0]-shaped[...,1]*cache[...,1],
                               shaped[...,1]*cache[...,0]+shaped[...,0]*cache[...,1]],-1)
                return y.flatten(3)
            module.forward=rotary
    explicit_forwards=[(module,module.forward) for module,fn in original_forwards]
    explicit_iswiglu=md.module.iSwiGLU
    checks={}
    theta={n:p.detach().clone() for n,p in m.model.named_parameters()}
    td={n:p.double() for n,p in theta.items()}
    tokens=np.load(data/'tokens.npy');pt=np.load(probe/'tokens.npy');po=np.load(probe/'offsets.npy')
    pr=np.arange(768,784);pb=pt[pr[:,None],po[pr,None]+np.arange(65)]
    ids=torch.tensor(pb[:,:64],dtype=torch.long,device=a.device);target=torch.tensor(pb[:,64],dtype=torch.long,device=a.device)
    rng=np.random.default_rng(630091+ca['stream']);rows=rng.integers(0,3072,size=(a.batches,32));offs=rng.integers(0,449,size=rows.shape)
    batches=tokens[rows[:,:,None],offs[:,:,None]+np.arange(65)]
    eps=np.array([1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,.001,.003,.01,.03,.1,1.]);records=[];start=time.time()
    def logits(par):
        md.eta=None;md.capture=False
        return torch.func.functional_call(md.model,par,(ids,),{'use_cache':False,'logits_to_keep':1}).logits[:,-1]
    group_count=len(ck['optimizer']['param_groups'])
    if group_count==1:
        parameter_groups=list(m.model.parameters())
    elif group_count==2:
        generator_names=set(json.loads((run/'manifest.json').read_text())['generator_parameters'])
        parameter_groups=[{'params':[p for n,p in m.model.named_parameters() if n in generator_names]},
                          {'params':[p for n,p in m.model.named_parameters() if n not in generator_names]}]
    else:
        raise ValueError(f'Unsupported saved optimizer group count: {group_count}')
    for k in range(a.batches):
        m.model.load_state_dict(ck['model']);opt=torch.optim.AdamW(parameter_groups,lr=3e-4,
            betas=(.9,.95),eps=1e-8,weight_decay=.01,foreach=False);opt.load_state_dict(copy.deepcopy(ck['optimizer']))
        b=torch.tensor(batches[k],dtype=torch.long,device=a.device)
        opt.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(m.logits(b[:,:64]),b[:,64]);loss.backward()
        torch.nn.utils.clip_grad_norm_(m.model.parameters(),1.,error_if_nonfinite=True);opt.step()
        delta={n:(p.detach()-theta[n]).double() for n,p in m.model.named_parameters()}
        del opt;m.model.zero_grad(set_to_none=True)
        with torch.no_grad():
            z,v=torch.func.jvp(logits,(td,),(delta,))
            _,w=torch.func.jvp(lambda par:torch.func.jvp(logits,(par,),(delta,))[1],(td,),(delta,))
            if k==0:
                for module,fn in original_forwards:module.forward=fn
                md.module.iSwiGLU=original_iswiglu
                reference=logits(td)
                for module,fn in explicit_forwards:module.forward=fn
                md.module.iSwiGLU=explicit_iswiglu
                h=1e-5
                vp=torch.func.jvp(logits,({n:t+h*delta[n] for n,t in td.items()},),(delta,))[1]
                vm=torch.func.jvp(logits,({n:t-h*delta[n] for n,t in td.items()},),(delta,))[1]
                wf=(vp-vm)/(2*h);prob=z.softmax(-1)
                wc0=w-(prob*w).sum(-1,keepdim=True);dw=wf-w;dw-= (prob*dw).sum(-1,keepdim=True)
                wn=(prob*wc0.square()).sum(-1).sqrt();en=(prob*dw.square()).sum(-1).sqrt()
                checks=dict(primal_max_abs_error=float((reference-z).abs().max()),second_derivative_secant_step=h,
                    second_derivative_fisher_absolute_error=en.cpu().tolist(),second_derivative_fisher_norm=wn.cpu().tolist(),
                    second_derivative_relative_error=(en/wn.clamp_min(1e-12)).cpu().tolist())
            p=z.softmax(-1);vc=v-(p*v).sum(-1,keepdim=True);wc=w-(p*w).sum(-1,keepdim=True)
            q=(p*vc.square()).sum(-1)
            cubic=(p*vc**3).sum(-1)+3*(p*vc*wc).sum(-1)
            first=(p*v).sum(-1)-v.gather(1,target[:,None]).squeeze(1)
            second=q+(p*w).sum(-1)-w.gather(1,target[:,None]).squeeze(1)
            base=-z.log_softmax(-1).gather(1,target[:,None]).squeeze(1)
            kl=[];dnll=[]
            for e in eps:
                zz=logits({n:t+e*delta[n] for n,t in td.items()})
                kl.append(stable_kl(z,zz).cpu().numpy())
                dnll.append((-zz.log_softmax(-1).gather(1,target[:,None]).squeeze(1)-base).cpu().numpy())
            records.append(dict(q=q.cpu().numpy(),cubic=cubic.cpu().numpy(),first=first.cpu().numpy(),
                                second=second.cpu().numpy(),kl=np.stack(kl),dnll=np.stack(dnll)))
        print(a.run_id,k+1,'seconds',round(time.time()-start,1),flush=True)
    raw=dict(epsilon=eps,probe_rows=pr,batch_rows=rows,batch_offsets=offs)
    raw.update({key:np.stack([r[key] for r in records]) for key in records[0]})
    if not all(np.isfinite(v).all() for v in raw.values()):raise RuntimeError('Nonfinite drift record')
    np.savez_compressed(out/'drift.npz',**raw)
    write_json(out/'manifest.json',dict(schema='conditional-drift-v1',arguments=vars(a),
        precision='float32 native optimizer displacement; float64 response with dtype-preserving softmax and rotary input arithmetic, explicit SiLU/LayerNorm formulas; native rotary cache constants',
        role=('calibration batch ensemble 0..7 predicts independent batch ensemble 8..15 at the same full training state'
              if a.batches==16 else 'Numerical and optimizer restoration control; no calibrated independent-ensemble inference'),
        numerical_checks=checks,raw_sha256=sha256(out/'drift.npz'),binding_sha256=sha256(out/'binding.json'),seconds=time.time()-start,
        peak_cuda_gb=torch.cuda.max_memory_allocated(a.device)/2**30 if a.device.startswith('cuda') else 0))


if __name__=='__main__':main()
