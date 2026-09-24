#!/usr/bin/env python
"""Full PLDR training, common crop law, and paired optimizer/generator branches."""
import argparse
import copy
import math
import time
from pathlib import Path
import numpy as np
import torch
from model_rg.training import TrainingModel
from model_rg.controlled import bind_run, collectives, collective_names, device_name
from model_rg.provenance import sha256, write_json


def generator(name):
    return any(s in name for s in ['reslayerAs', 'plgatt_layer', 'layernormA'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True)
    ap.add_argument('--heads',type=int,required=True);ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--stream',type=int,required=True);ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--steps',type=int,default=2048);ap.add_argument('--run-id',required=True)
    ap.add_argument('--normalization',choices=['native','fan_in'],default='native')
    a=ap.parse_args();a.device=device_name(a.device);root=Path(a.root)
    study=root/'controlled-study-20260905';out=study/'runs'/a.run_id
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=root/'assets/PLDR-LLM-v51-SOC-110M-1';data=root/'data/refinedweb-4608'
    probe=study/'data/short'
    bind_run(out,[data/'manifest.json',data/'tokens.npy',probe/'manifest.json',probe/'tokens.npy',
                  source/'modeling_pldrllm.py',source/'configuration_pldrllm.py',study/'protocol.json',
                  *([study/'normalization-protocol.json'] if a.normalization=='fan_in' else []),
                  *([study/'seed-confirmation-protocol.json'] if a.run_id.startswith('normalized-confirmation') else [])],vars(a))
    tokens=np.load(data/'tokens.npy');ptokens=np.load(probe/'tokens.npy');poff=np.load(probe/'offsets.npy')
    probes=ptokens[np.arange(len(ptokens))[:,None],poff[:,None]+np.arange(65)]
    rng=np.random.default_rng(a.stream)
    rows=rng.integers(0,3072,size=(a.steps+16,32));offsets=rng.integers(0,449,size=rows.shape)
    batches=tokens[rows[:,:,None],offsets[:,:,None]+np.arange(65)]
    np.savez_compressed(out/'sampling.npz',rows=rows,offsets=offsets,probe_offsets=poff)
    m=TrainingModel(source,a.heads,a.seed,a.device)
    if a.normalization=='fan_in':
        width=64*a.heads
        with torch.no_grad():
            for name,module in m.model.named_modules():
                if isinstance(module,torch.nn.Linear) and not generator(name):
                    reference=128 if module.in_features==width else (128*8)//3
                    if module.in_features in [width,(width*8)//3]:
                        module.weight.mul_(math.sqrt(reference/module.in_features))
        groups_for_optimizer=[{'params':[p for n,p in m.model.named_parameters() if generator(n)],'lr':3e-4},
                              {'params':[p for n,p in m.model.named_parameters() if not generator(n)],'lr':3e-4*128/width}]
    else:groups_for_optimizer=m.model.parameters()
    optimizer=torch.optim.AdamW(groups_for_optimizer,lr=3e-4,betas=(.9,.95),eps=1e-8,weight_decay=.01,foreach=False)
    named=list(m.model.named_parameters());groups=[int(generator(n)) for n,p in named]
    m.config.to_json_file(str(out/'config.json'))
    start=time.time();losses=[];gradnorms=[];path=[];opstats=[];path_steps=[]
    evaluations={};matrix_dispersion={};counts=np.zeros(m.config.vocab_size,dtype=np.int64)
    pool=np.bincount(tokens[:3072].ravel(),minlength=m.config.vocab_size)
    poollog=np.log((pool+.5)/(pool.sum()+.5*len(pool)))
    milestones=set([0,128,512,1024,a.steps])

    def optstats():
        result=[]
        with torch.no_grad():
            for group in [0,1]:
                sums=torch.zeros(4,device=a.device,dtype=torch.float64);n=0
                for (name,p),g in zip(named,groups):
                    if g != group:continue
                    n+=p.numel();sums[0]+=p.double().square().sum()
                    state=optimizer.state.get(p,{})
                    if 'exp_avg' in state:
                        mm=state['exp_avg'].double();vv=state['exp_avg_sq'].double()
                        sums[1]+=mm.square().sum();sums[2]+=vv.sum()
                        sums[3]+=(mm*p.double()).sum()
                result.extend((sums/max(n,1)).cpu().tolist())
        return result

    def evaluate(selected, shuffled=False, matrices=False):
        m.model.eval();ff=[];gg=[]
        order=np.random.default_rng(630071).permutation(63)
        with torch.no_grad():
            for part in np.array_split(selected,max(1,(len(selected)+31)//32)):
                batch=torch.tensor(probes[part],dtype=torch.long,device=a.device)
                ids=batch[:,:64].clone()
                if shuffled:ids[:,:63]=ids[:,order]
                f,g=collectives(m,ids,batch[:,64],matrices=matrices)
                if not torch.isfinite(f).all():raise RuntimeError('Nonfinite collective field')
                ff.append(f.cpu().numpy())
                if matrices:gg.append(torch.stack(g,1).cpu().numpy())
        return np.concatenate(ff), np.concatenate(gg) if gg else None

    def snapshot(t):
        f,g=evaluate(np.arange(64),matrices=t in milestones)
        path_steps.append(t);path.append(f.mean(0));opstats.append(optstats())
        if t in milestones:
            ef,_=evaluate(np.arange(512,2048));sf,_=evaluate(np.arange(512,2048),shuffled=True)
            matchedlog=np.log((counts+.5)/(counts.sum()+.5*len(counts)))
            y=probes[512:2048,64]
            evaluations[str(t)]=dict(features=ef,shuffled_features=sf,
                matched_unigram_nll=-matchedlog[y],pool_unigram_nll=-poollog[y])
            gm=g.mean(0);disp=np.sqrt(np.mean((g-gm[None])**2,axis=(0,3,4))/np.maximum(np.mean(gm**2,axis=(2,3)),1e-30))
            matrix_dispersion[str(t)]=disp
            print(a.run_id,t,'nll',float(ef[:,25].mean()),'shuffle_delta',float((sf[:,25]-ef[:,25]).mean()),
                  'Gdisp',float(np.median(disp)),'seconds',round(time.time()-start,1),flush=True)

    def step(index,freeze_generator=False):
        m.model.train();optimizer.zero_grad(set_to_none=True)
        batch=torch.tensor(batches[index],dtype=torch.long,device=a.device)
        loss=torch.nn.functional.cross_entropy(m.logits(batch[:,:64]),batch[:,64])
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        loss.backward()
        # Match clipping across paired branches; suppress generator update afterwards.
        norm=torch.nn.utils.clip_grad_norm_(m.model.parameters(),1.,error_if_nonfinite=True)
        if freeze_generator:
            for (name,p),group in zip(named,groups):
                if group:p.grad=None
        optimizer.step()
        return float(loss.detach()),float(norm)

    snapshot(0)
    for t in range(a.steps):
        loss,norm=step(t);losses.append(loss);gradnorms.append(norm)
        counts+=np.bincount(batches[t,:,64],minlength=len(counts))
        if (t+1)%32==0 or t+1 in milestones:snapshot(t+1)
    # Locally produced checkpoint includes the augmented optimizer state.
    before={n:p.detach().cpu().clone() for n,p in m.model.state_dict().items()}
    opt_before=copy.deepcopy(optimizer.state_dict())
    saved={'model':before,'optimizer':opt_before,'step':a.steps,'arguments':vars(a)}
    torch.save(saved,out/'final-training-state.pt')
    branches={};branch_rows=np.arange(512,768)
    for mode in ['continue','reset_moments','freeze_generator','reset_and_freeze']:
        m.model.load_state_dict(before);optimizer.load_state_dict(copy.deepcopy(opt_before))
        if 'reset' in mode:
            # Reset both moments while retaining step counters and schedule.
            for state in optimizer.state.values():
                state['exp_avg'].zero_();state['exp_avg_sq'].zero_()
        branch=[]
        f,_=evaluate(branch_rows);branch.append(f)
        for k in range(16):
            step(a.steps+k,freeze_generator='freeze' in mode)
            if k+1 in [1,2,4,8,16]:
                f,_=evaluate(branch_rows);branch.append(f)
        branches[mode]=np.stack(branch)
    raw={'steps':np.array(path_steps),'mean_fields':np.array(path),'optimizer_statistics':np.array(opstats),
         'losses':np.array(losses),'gradient_norms':np.array(gradnorms),'supervised_target_counts':counts,
         'branch_steps':np.array([0,1,2,4,8,16])}
    raw.update({f'evaluation_{t}_{k}':v for t,values in evaluations.items() for k,v in values.items()})
    raw.update({f'matrix_dispersion_{t}':v for t,v in matrix_dispersion.items()})
    raw.update({f'branch_{k}':v for k,v in branches.items()})
    np.savez_compressed(out/'measurements.npz',**raw)
    write_json(out/'manifest.json',dict(schema='controlled-training-v1',arguments=vars(a),
        parameters=sum(p.numel() for n,p in named),feature_names=collective_names(),
        optimizer_group_rates=[float(g['lr']) for g in optimizer.param_groups],
        generator_parameters=[n for (n,p),g in zip(named,groups) if g],
        optimizer=dict(name='AdamW',lr=3e-4,betas=[.9,.95],eps=1e-8,weight_decay=.01,clip_norm=1.),
        train_rows=[0,3072],path_probe_rows=[0,64],evaluation_rows=[512,2048],branch_rows=[512,768],
        context_control='Permute positions 0..62, preserve last context token and external target; same permutation for every document',
        sampling_sha256=sha256(out/'sampling.npz'),raw_sha256=sha256(out/'measurements.npz'),
        checkpoint_sha256=sha256(out/'final-training-state.pt'),binding_sha256=sha256(out/'binding.json'),
        seconds=time.time()-start,peak_cuda_gb=torch.cuda.max_memory_allocated(a.device)/2**30))
    print('Completed',a.run_id,round(time.time()-start,1),flush=True)


if __name__=='__main__':main()
