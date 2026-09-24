#!/usr/bin/env python
"""Bounded source selection from the released 110M PLDR checkpoint.

One released model supplies four paired fine-tuning paths. A fresh AdamW
origin is explicit because its pretraining optimizer state is unavailable.
This is an invariant-regime reference, not an additional width replicate.
"""
from companion_paths import legacy_path
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from datetime import datetime,timezone
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
from model_rg.native import NativeModel
from model_rg.onepass_regimes import optimizer_and_scheduler
from model_rg.schedules import loss,clip
from model_rg.finetuning import mixture_stream
from model_rg.provenance import sha256,write_json
from finetuning_observation import observe
from measure_law_closure import digest_state

ARMS=[dict(name='narrative',rho=0.),dict(name='mixture',rho=.5),dict(name='technical',rho=1.),dict(name='general',rho=None)]


def sources():
    names=['scripts/finetuning_reference.py','scripts/finetuning_observation.py','scripts/measure_law_closure.py']
    names += [str(p.relative_to(REPO)) for p in (REPO/'src/model_rg').glob('*.py')]
    return {n:sha256(REPO/n) for n in sorted(names)}


def runtime():
    return dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,cuda=torch.version.cuda,
        devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],threads=4,dtype='float32',tf32=False)


def validate(study,full=False):
    study=Path(study);s=json.loads((study/'protocol.json').read_text());stage=s['stage']
    if s['schema']!='pldr-reference-finetuning-v1' or stage not in ['qualification','assessment']:raise ValueError('Wrong reference protocol')
    horizon=2 if stage=='qualification' else 512
    if s['horizon']!=horizon or s['times']!=([0,1,2] if horizon==2 else [0,16,64,128,256,512]) or s['arms']!=ARMS:raise ValueError('Changed reference design')
    if s['sources']!=sources() or s['runtime']!=runtime() or s['rate']!=.00012:raise ValueError('Changed producer/platform/rate')
    if s['optimizer_origin']!='fresh_zero_moments_and_counters':raise ValueError('Unknown optimizer origin')
    for p,h in s['inputs'].items():
        if full or not p.endswith('tokens.npy'):
            if sha256(p)!=h:raise ValueError('Changed reference input: '+p)
    if stage=='assessment':
        q=Path(s['qualification']['path'])
        if sha256(q)!=s['qualification']['sha256']:raise ValueError('Qualification changed')
        v=json.loads(q.read_text());qs=json.loads((q.parent/'protocol.json').read_text())
        if v['status']!='passed' or v['protocol_sha256']!=sha256(q.parent/'protocol.json'):raise ValueError('Unpassed reference qualification')
        if qs['sources']!=s['sources'] or qs['runtime']!=s['runtime'] or qs['arms']!=ARMS:raise ValueError('Unqualified reference implementation')
    return s


def prepare(study,parent,stage,qualification):
    study=Path(study).resolve();parent=Path(parent).resolve();p=json.loads((parent/'protocol.json').read_text());root=Path(p['root']);data=Path(p['data'])
    if study.exists():raise FileExistsError(study)
    if not study.is_relative_to(root):raise ValueError('Unauthorized destination')
    info=Path(legacy_path('/pldr-assets/refinedweb/datasets/huggingface_datasets/tiiuae___falcon-refinedweb/default/0.0.0/c735840575b629292b41da8dde11dcd523d4f91c/dataset_info.json'))
    lengths=json.loads(info.read_text())['splits']['train']['shard_lengths'];offset=np.r_[0,np.cumsum(lengths)]
    records_path=root/'data/refinedweb-onepass-524288/records.json';records=json.loads(records_path.read_text())
    positions=np.array([offset[int(re.search(r'train-(\d+)-of-',r['shard']).group(1))]+r['row'] for r in records],np.int64)
    if positions.min()<32000000:raise ValueError('Source overlaps the reported public pretraining interval')
    assets=root/'assets/PLDR-LLM-v51-SOC-110M-1';inputs={str(data/n):sha256(data/n) for n in ['streams.npz','evaluation.npz','manifest.json','technical-blocks.npy','narrative-blocks.npy','general-blocks.npy']}
    for f in assets.iterdir():
        if f.is_file() and f.suffix in ['.py','.json','.model','.safetensors']:inputs[str(f)]=sha256(f)
    inputs[str(info)]=sha256(info);inputs[str(records_path)]=sha256(records_path)
    tokens=root/'data/refinedweb-onepass-524288/tokens.npy';inputs[str(tokens)]=p['inputs'][str(tokens)]
    case=next(c for c in p['cases'] if c['heads']==14 and c['seed']==640101)
    horizon=2 if stage=='qualification' else 512
    spec=dict(schema='pldr-reference-finetuning-v1',stage=stage,created_at=datetime.now(timezone.utc).isoformat(),root=str(root),data=str(data),assets=str(assets),horizon=horizon,times=[0,1,2] if horizon==2 else [0,16,64,128,256,512],arms=ARMS,
        sources=sources(),inputs=inputs,runtime=runtime(),rate=.00012,optimizer_origin='fresh_zero_moments_and_counters',recipe=case['profile'],
        source_position_scope='Cached global document order, conditional on the released model description of pretraining on RefinedWeb document interval [16000000,32000000). This verifies position separation under that reported ordering; it does not recover unavailable pretraining optimizer history.',
        global_document_min=int(positions.min()),global_document_max=int(positions.max()),reported_pretraining_interval=[16000000,32000000],
        hypothesis='Test source-response visibility and operator invariance in the released strongly row-concentrated model. One width and one released initialization cannot identify native thermodynamic critical exponents.',
        law='Same fixed lexical reservoirs and uniforms as the controlled source study, without replacement. Every evaluation document is excluded. All eligible documents lie outside the reported public pretraining interval under the bound cached ordering.',
        analysis='Retain every source arm, time and proper-prefix risk. Measure paired A/G discrepancy, row fraction and incoming-Fisher two-source Gram. No pooling with the controlled four-body-initialization width family.')
    if stage=='assessment':
        if qualification is None:raise ValueError('Reference qualification required')
        q=Path(qualification).resolve();v=json.loads(q.read_text());qs=json.loads((q.parent/'protocol.json').read_text())
        if v['status']!='passed' or qs['sources']!=spec['sources'] or qs['runtime']!=spec['runtime']:raise ValueError('Unqualified reference study')
        spec['qualification']=dict(path=str(q),sha256=sha256(q))
    study.mkdir(parents=True);np.save(study/'global-document-positions.npy',positions);spec['inputs'][str(study/'global-document-positions.npy')]=sha256(study/'global-document-positions.npy')
    for n in spec['sources']:
        dest=study/'executed-source'/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((REPO/n).read_bytes())
    write_json(study/'protocol.json',spec);validate(study,full=True);print('Frozen released-reference',stage,'global rows',int(positions.min()),int(positions.max()),flush=True)


def panel(stage,t,horizon):
    ids=np.arange(480) if stage=='assessment' and t in [0,horizon] else np.array([j+k for j in [0,80,160,240,320,400] for k in range(16)])
    return ids,np.array([0,80,160,240,320,400])


def worker(study,name,device):
    study=Path(study).resolve();s=validate(study)
    if device not in ['cuda:0','cuda:1']:raise ValueError('One GPU per worker')
    replay=name=='general-replay';unobserved=name=='general-unobserved';base='general' if replay or unobserved else name
    arm=next(x for x in ARMS if x['name']==base)
    if unobserved and s['stage']!='qualification':raise ValueError('Only qualification has unobserved control')
    dest=study/name;dest.mkdir(exist_ok=False);torch.set_num_threads(4);torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=NativeModel(s['assets'],device);model.model.requires_grad_(True)
    optimizer,_=optimizer_and_scheduler(model,s['recipe'])
    for g in optimizer.param_groups:g['lr']=s['rate'];g['initial_lr']=s['rate']
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    if optimizer.state:raise ValueError('Reference optimizer is not fresh')
    with np.load(Path(s['data'])/'streams.npz') as f:
        blocks=f['general'][:s['horizon']*32].reshape(-1,32) if arm['rho'] is None else mixture_stream(f['technical'],f['narrative'],f['uniforms'][:s['horizon']],arm['rho'])
    horizon=min(16,s['horizon']) if replay else s['horizon'];blocks=blocks[:horizon];np.save(dest/'blocks.npy',blocks)
    with np.load(Path(s['data'])/'evaluation.npz') as f:crops=f['crops'];evdocs=f['document_ids']
    if len(np.unique(blocks))!=blocks.size or np.isin(blocks//8,evdocs).any():raise ValueError('Reference source/evaluation collision')
    corpus=np.load(Path(s['root'])/'data/refinedweb-onepass-524288/tokens.npy',mmap_mode='r')
    losses=[];norms=[];rates=[];observations=[];digests={};begin=time.perf_counter();torch.cuda.reset_peak_memory_stats(device)
    def capture(t):
        ids,raw=panel(s['stage'],t,s['horizon']);v=observe(model,crops,ids,raw);p=dest/f'observation-{t:04d}.npz';np.savez(p,**v)
        observations.append(dict(time=t,file=str(p),sha256=sha256(p)));digests[str(t)]=digest_state(model,optimizer,scheduler)
    capture(0)
    for t in range(1,horizon+1):
        model.model.train();optimizer.zero_grad(set_to_none=True);b=blocks[t-1]
        x=torch.as_tensor(corpus[(b//8)[:,None],(64*(b%8))[:,None]+np.arange(65)],dtype=torch.long,device=device)
        value=loss(model,x,s['recipe'])
        if not torch.isfinite(value):raise FloatingPointError('Nonfinite reference training loss')
        value.backward();clip(model,s['recipe']);norm=sum(p.grad.detach().double().square().sum() for p in model.model.parameters()).sqrt()
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite reference gradient')
        losses.append(float(value.detach()));norms.append(float(norm));rates.append([g['lr'] for g in optimizer.param_groups]);optimizer.step();scheduler.step()
        if (t in s['times'] and not unobserved) or t==horizon:capture(t);print(name,t,'loss',round(losses[-1],6),flush=True)
    np.savez(dest/'training.npz',losses=np.array(losses),gradient_norms=np.array(norms),rates=np.array(rates))
    checkpoint=None
    if s['stage']=='assessment' and not replay:
        checkpoint=dest/'final-state.pt';torch.save(dict(model=model.model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),finetuning_step=horizon,optimizer_origin=s['optimizer_origin'],blocks=blocks),checkpoint)
    write_json(dest/'result.json',dict(status='complete',name=name,arm=arm,role='replay' if replay else 'unobserved replay' if unobserved else 'scientific',steps=horizon,observations=observations,state_digests=digests,
        protocol_sha256=sha256(study/'protocol.json'),blocks_sha256=sha256(dest/'blocks.npy'),training_sha256=sha256(dest/'training.npz'),checkpoint=str(checkpoint) if checkpoint else None,checkpoint_sha256=sha256(checkpoint) if checkpoint else None,
        seconds=time.perf_counter()-begin,peak_cuda_bytes=torch.cuda.max_memory_allocated(device)))


def run(study):
    study=Path(study).resolve();s=validate(study,full=True)
    if (study/'launcher.json').exists():raise FileExistsError('Reference launcher already exists')
    jobs=[x['name'] for x in ARMS]+['general-replay']
    if s['stage']=='qualification':jobs.append('general-unobserved')
    active={};done=[];begin=time.perf_counter()
    try:
        while jobs or active:
            for gpu in [0,1]:
                if gpu not in active and jobs:
                    name=jobs.pop(0);log=(study/(name+'.log')).open('w')
                    command=[sys.executable,str(Path(__file__).resolve()),'worker','--study',str(study),'--arm',name,'--device',f'cuda:{gpu}']
                    active[gpu]=(subprocess.Popen(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT),log,name)
            for gpu,(p,log,name) in list(active.items()):
                if p.poll() is not None:
                    log.close();del active[gpu]
                    if p.returncode:raise RuntimeError('Reference worker failed: '+name)
                    done.append(name)
            time.sleep(1)
        write_json(study/'launcher.json',dict(status='complete',arms=done,protocol_sha256=sha256(study/'protocol.json'),seconds=time.perf_counter()-begin))
    except BaseException:
        for p,log,name in active.values():p.terminate();p.wait();log.close()
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','worker','validate']);p.add_argument('--study',required=True);p.add_argument('--parent-study');p.add_argument('--stage',choices=['qualification','assessment']);p.add_argument('--qualification');p.add_argument('--arm');p.add_argument('--device');a=p.parse_args()
    if a.action=='prepare':prepare(a.study,a.parent_study,a.stage,a.qualification)
    elif a.action=='run':run(a.study)
    elif a.action=='worker':worker(a.study,a.arm,a.device)
    else:validate(a.study,full=True)
