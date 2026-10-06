#!/usr/bin/env python3
"""Dense potential observations on exact single-pass pretraining continuations."""
from companion_paths import configured_path
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch
from model_rg.training import TrainingModel
from model_rg.onepass_regimes import sample_batches
from model_rg.schedules import optimizer_and_scheduler,loss,clip
from model_rg.provenance import sha256,write_json,environment
from train_potential_avalanches import observe,FIELDS
from model_rg.deductive_activity import DeductiveActivity,TENSOR_NAMES,STAT_NAMES
ROOT=Path(configured_path('data:model'))
STUDY=ROOT/'potential-avalanche-20260913'
REPO=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--device',required=True);a=p.parse_args()
    protocol=STUDY/'protocols/continuations.json';spec=json.loads(protocol.read_text())
    cases=[j for j in spec['cases'] if j['name']==a.case]
    if len(cases)!=1:raise ValueError('One frozen branch required')
    case=cases[0]
    for f,d in spec['producer_sources'].items():
        if sha256(REPO/f)!=d:raise ValueError('Producer changed: '+f)
    parent=Path(case['checkpoint'])
    if sha256(parent)!=case['checkpoint_sha256']:raise ValueError('Changed incoming full state')
    meta=json.loads(Path(case['parent_manifest']).read_text())
    if sha256(case['parent_manifest'])!=case['parent_manifest_sha256'] or meta['status']!='complete':raise ValueError('Changed parent record')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    saved=torch.load(parent,map_location='cpu',weights_only=True)
    job=saved['arguments'];profile=saved['recipe'];start=saved['step']
    if start!=65536 or job['stream_seed']!=640001 or job['heads']!=14:raise ValueError('Unexpected native continuation')
    model=TrainingModel(ROOT/'assets/PLDR-LLM-v51-SOC-110M-1',14,job['seed'],a.device)
    model.model.load_state_dict(saved['model'])
    opt,sched=optimizer_and_scheduler(model,profile)
    opt.load_state_dict(saved['optimizer']);sched.load_state_dict(saved['scheduler']);del saved
    if sched.last_epoch!=start:raise ValueError('Scheduler phase lost')
    initial_rates=[g['lr'] for g in opt.param_groups]
    allrows,alloffsets,_=sample_batches(np.load(ROOT/'data/refinedweb-onepass-524288/tokens.npy',mmap_mode='r'),job['stream_seed'],start+case['steps'])
    history=np.load(Path(case['parent_manifest']).parent/'sampling.npz')
    if not np.array_equal(history['rows'],allrows[:start]) or not np.array_equal(history['offsets'],alloffsets[:start]):raise ValueError('Stream continuation mismatch')
    ids=allrows*8+alloffsets//64
    if np.unique(ids).size!=ids.size:raise ValueError('Training prefix repeats a block')
    rows=allrows[start:];offsets=alloffsets[start:]
    token=np.load(ROOT/'data/refinedweb-onepass-524288/tokens.npy',mmap_mode='r')
    batches=token[rows[:,:,None],offsets[:,:,None]+np.arange(65)]
    sel=np.load(STUDY/'selection.npz');probes=torch.as_tensor(sel['probes'],dtype=torch.long,device=a.device)
    coords=torch.as_tensor(sel['coordinates'],dtype=torch.long,device=a.device)
    out=STUDY/'continuations'/case['name'];out.mkdir(parents=True,exist_ok=False)
    write_json(out/'binding.json',dict(case=case,environment=environment(),protocol_sha256=sha256(protocol),
        stream_prefix_equal=True,unique_prefix_blocks=int(ids.size),producer_sources=spec['producer_sources']))
    fields=[];bases=[];powers=[];nlls=[];losses=[];rates=[];state=None;t0=time.time()
    deductive=DeductiveActivity(model,probes[:2],coords)
    def snapshot():
        nonlocal state
        f,b,p,n,state=observe(model,probes[:2],coords,state)
        fields.append(f);bases.append(b);powers.append(p);nlls.append(n)
        deductive.snapshot()
    snapshot()
    for k in range(case['steps']):
        if case['drive']=='anneal':
            mul=.1+.45*(1+math.cos(math.pi*k/(case['steps']-1)))
            for g,rate in zip(opt.param_groups,initial_rates):g['lr']=rate*mul
        # "native" continues its original 250000-update schedule without reset.
        model.model.train();opt.zero_grad(set_to_none=True)
        batch=torch.as_tensor(batches[k],dtype=torch.long,device=a.device)
        value=loss(model,batch,profile)
        if not torch.isfinite(value):raise FloatingPointError('Nonfinite continuation loss')
        value.backward();clip(model,profile)
        rates.append([g['lr'] for g in opt.param_groups]);opt.step()
        if case['drive']=='native':sched.step()
        losses.append(float(value.detach()));snapshot()
        if (k+1)%256==0:
            write_json(out/'progress.json',dict(step=k+1,seconds=time.time()-t0))
            print(case['name'],k+1,round(time.time()-t0,1),flush=True)
    raw=dict(fields=np.asarray(fields),logbase=np.asarray(bases),power=np.asarray(powers),probe_nll=np.asarray(nlls),
        loss=losses,lr=np.asarray(rates),rows=rows,offsets=offsets,**deductive.arrays())
    if not all(np.isfinite(x).all() for x in raw.values()):raise FloatingPointError('Nonfinite raw continuation')
    np.savez_compressed(out/'activity.npz',**raw)
    torch.save(dict(model=model.model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),
        start_step=start,additional_steps=case['steps'],profile=profile,case=case),out/'final-state.pt')
    write_json(out/'manifest.json',dict(status='complete',case=case,producer_sources=spec['producer_sources'],
        protocol_sha256=sha256(protocol),field_names=FIELDS,tensor_names=TENSOR_NAMES,tensor_stat_names=STAT_NAMES,parent_objective=profile['objective'],
        unique_prefix_blocks=int(ids.size),no_repetition_including_parent=True,
        runtime_seconds=time.time()-t0,max_cuda_memory_bytes=torch.cuda.max_memory_allocated(a.device),
        artifacts={f:sha256(out/f) for f in ['activity.npz','final-state.pt']}))

if __name__=='__main__':main()
