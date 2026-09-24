#!/usr/bin/env python
"""Matched fresh-corpus continuations of adapted native PLDR states.

Tests whether a source-response span persists when every branch returns to
the same remaining general-source law. All input positions are unused on
every parent path; no repeated corpus is introduced.
"""
import argparse
import copy
from datetime import datetime,timezone
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
from model_rg.training import TrainingModel
from model_rg.onepass_regimes import optimizer_and_scheduler
from model_rg.schedules import loss,clip
from model_rg.provenance import sha256,write_json
from measure_law_closure import digest_state
from finetuning_observation import observe

PARENTS=['mix0000','mix1000','general']


def sources():
    names=['scripts/finetuning_return.py','scripts/finetuning_observation.py','scripts/measure_law_closure.py']
    names += [str(p.relative_to(REPO)) for p in (REPO/'src/model_rg').glob('*.py')]
    return {n:sha256(REPO/n) for n in sorted(names)}


def runtime():
    return dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,cuda=torch.version.cuda,
        devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],threads=4,tf32=False,dtype='float32')


def validate(study,full=False,case=None):
    study=Path(study);s=json.loads((study/'protocol.json').read_text())
    if s['schema']!='pldr-source-return-v1' or s['stage'] not in ['qualification','assessment']:raise ValueError('Invalid protocol')
    horizon=2 if s['stage']=='qualification' else 256
    if s['horizon']!=horizon or s['times']!=([0,2] if horizon==2 else [0,16,64,128,256]):raise ValueError('Unqualified horizon')
    seeds=[640101] if horizon==2 else [640101,640102,640103,640104]
    if [(c['heads'],c['seed']) for c in s['cases']]!=[(h,k) for h in [4,8,14] for k in seeds]:raise ValueError('Changed case inventory')
    if s['parents']!=PARENTS or s['runtime']!=runtime():raise ValueError('Changed intervention/platform')
    if s['sources']!=sources():raise ValueError('Changed producer')
    for p,h in s['inputs'].items():
        if full or not p.endswith('tokens.npy'):
            if sha256(p)!=h:raise ValueError('Changed input: '+p)
    for c in s['cases'] if full else ([case] if case else []):
        for row in c['parents']:
            for key in ['checkpoint','result','observation']:
                if sha256(row[key])!=row[key+'_sha256']:raise ValueError('Changed parent '+key)
    if s['stage']=='assessment':
        q=Path(s['qualification']['path'])
        if sha256(q)!=s['qualification']['sha256']:raise ValueError('Qualification changed')
        v=json.loads(q.read_text());qs=json.loads((q.parent/'protocol.json').read_text())
        if v['status']!='passed' or v['protocol_sha256']!=sha256(q.parent/'protocol.json'):raise ValueError('Unpassed qualification')
        if qs['sources']!=s['sources'] or qs['runtime']!=s['runtime'] or qs['parents']!=s['parents']:raise ValueError('Unqualified producer/intervention')
    return s


def prepare(study,parent,stage,qualification):
    study=Path(study).resolve();parent=Path(parent).resolve();ps=json.loads((parent/'protocol.json').read_text())
    if study.exists():raise FileExistsError(study)
    root=Path(ps['root']);data=Path(ps['data'])
    if not study.is_relative_to(root):raise ValueError('Unauthorized destination')
    seeds=[640101] if stage=='qualification' else [640101,640102,640103,640104]
    cases=[]
    for h in [4,8,14]:
        for seed in seeds:
            name=f'h{h}-s{seed}';p=parent/name;r=json.loads((p/'results.json').read_text())
            if r['status']!='complete':raise ValueError('Incomplete parent case')
            rows=[]
            for arm in PARENTS:
                record=next(x for x in r['records'] if x['arm']['name']==arm)
                if record['steps']!=512 or not record['checkpoint']:raise ValueError('Missing complete adapted state')
                obs=next(o for o in record['observations'] if o['time']==512)
                row=dict(arm=arm,checkpoint=record['checkpoint'],result=str(p/arm/'result.json'),observation=obs['file'],expected_state_digest=record['state_digests']['512'])
                for key in ['checkpoint','result','observation']:row[key+'_sha256']=sha256(row[key])
                if row['checkpoint_sha256']!=record['checkpoint_sha256'] or row['observation_sha256']!=obs['sha256']:raise ValueError('Parent hash mismatch')
                rows.append(row)
            cases.append(dict(name=name,heads=h,seed=seed,parents=rows))
    with np.load(data/'streams.npz') as f:excluded=np.unique(np.concatenate([f[n][:16384] for n in ['technical','narrative','general']]))
    pool=np.load(data/'general-blocks.npy');pool=pool[~np.isin(pool,excluded)]
    draw=np.random.default_rng(915003).permutation(pool)[:8192].reshape(256,32)
    horizon=2 if stage=='qualification' else 256
    inputs={str(data/n):sha256(data/n) for n in ['streams.npz','general-blocks.npy','evaluation.npz','manifest.json']}
    inputs[str(root/'data/refinedweb-onepass-524288/tokens.npy')]=ps['inputs'][str(root/'data/refinedweb-onepass-524288/tokens.npy')]
    for p,h in ps['inputs'].items():
        if '/assets/' in p:inputs[p]=h
    spec=dict(schema='pldr-source-return-v1',stage=stage,created_at=datetime.now(timezone.utc).isoformat(),root=str(root),data=str(data),parent_study=str(parent),parent_protocol_sha256=sha256(parent/'protocol.json'),
        parents=PARENTS,cases=cases,horizon=horizon,times=[0,2] if horizon==2 else [0,16,64,128,256],runtime=runtime(),sources=sources(),inputs=inputs,
        law='Uniform permutation of the eligible general-source pool after exclusion of every parent source block. Identical fresh blocks for all adapted states. All optimizer moments, counters and schedule phases retained. Full native updates.',
        hypothesis='Finite predictive source spans may persist or rotate under a common subsequent data law. Fit a two-source transport and an expanded seven-mixture-contrast transport on calibration documents and score held-out documents. All seven contrasts are fixed from the parent study before return outcomes. Compare the two-source transport residual with the weaker incoming response singular scale; this is a finite inheritance diagnostic, not an asymptotic exponent proof.',
        budget=dict(scientific_updates=9216,max_worker_seconds=5400),sampling_seed=915003,
        analysis='Retain all 36 source-return paths. Observe fixed-prefix predictive Fisher contrasts against the continuing general parent, source Gram spectra, canonical angles and complete residual. Compare two endpoint contrasts against the nested dictionary of those endpoints plus all five interior mixture contrasts. No critical threshold selected from this study.')
    if stage=='assessment':
        if qualification is None:raise ValueError('Qualification required')
        q=Path(qualification).resolve();v=json.loads(q.read_text());qs=json.loads((q.parent/'protocol.json').read_text())
        if v['status']!='passed' or qs['sources']!=spec['sources'] or qs['runtime']!=spec['runtime']:raise ValueError('Unqualified return study')
        spec['qualification']=dict(path=str(q),sha256=sha256(q))
    study.mkdir(parents=True);np.save(study/'blocks.npy',draw[:horizon]);spec['inputs'][str(study/'blocks.npy')]=sha256(study/'blocks.npy')
    for n in spec['sources']:
        p=study/'executed-source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((REPO/n).read_bytes())
    write_json(study/'protocol.json',spec);validate(study,full=True)
    print('Frozen source return',stage,len(cases),'cases',horizon,'updates each',flush=True)


def indices(stage,t,horizon):
    ids=np.arange(480) if stage=='assessment' and t in [0,horizon] else np.array([j+k for j in [0,80,160,240,320,400] for k in range(16)])
    return ids,np.array([0,80,160,240,320,400])


def worker(study,name,device):
    study=Path(study).resolve();s0=json.loads((study/'protocol.json').read_text());case=next(c for c in s0['cases'] if c['name']==name)
    spec=validate(study,case=case)
    if device not in ['cuda:0','cuda:1']:raise ValueError('One GPU per worker')
    dest=study/name;dest.mkdir(exist_ok=False);torch.cuda.set_device(device);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(spec['root']);data=Path(spec['data']);blocks=np.load(study/'blocks.npy')
    corpus=np.load(root/'data/refinedweb-onepass-524288/tokens.npy',mmap_mode='r')
    with np.load(data/'evaluation.npz') as f:crops=f['crops']
    model=TrainingModel(root/'assets/PLDR-LLM-v51-SOC-110M-1',case['heads'],case['seed'],device)
    records=[];started=time.perf_counter();reference={};torch.cuda.reset_peak_memory_stats(device)
    jobs=[dict(p,role='scientific',name=p['arm']) for p in case['parents']]
    general=next(p for p in case['parents'] if p['arm']=='general')
    jobs.append(dict(general,role='replay',name='general-replay'))
    if spec['stage']=='qualification':jobs.append(dict(general,role='unobserved replay',name='general-unobserved'))
    for job in jobs:
        if time.perf_counter()-started>spec['budget']['max_worker_seconds']:raise TimeoutError('Return worker budget')
        state=torch.load(job['checkpoint'],map_location='cpu',weights_only=False)
        if state['pretraining_step']!=40960 or state['finetuning_step']!=512 or state['scheduler']['last_epoch']!=41472:raise ValueError('Parent age differs')
        if np.intersect1d(state['consumed_finetuning_blocks'],blocks).size:raise ValueError('Repeated parent block')
        model.model.load_state_dict(state['model']);model.model.requires_grad_(True);model.capture=False;model.head_outputs={};model.eta=None
        optimizer,scheduler=optimizer_and_scheduler(model,state['recipe'])
        optimizer.load_state_dict(copy.deepcopy(state['optimizer']));scheduler.load_state_dict(copy.deepcopy(state['scheduler']))
        optimizer.zero_grad(set_to_none=True)
        folder=dest/job['name'];folder.mkdir();observations=[];digests={};losses=[];norms=[];rates=[]
        horizon=min(16,spec['horizon']) if job['role']=='replay' else spec['horizon']
        ids,raw=indices(spec['stage'],0,spec['horizon']);measured=observe(model,crops,ids,raw)
        # Parent reproduction is checked array by array before any new update.
        with np.load(job['observation']) as p:
            for k in measured:
                if k in ['raw_tensors','raw_indices']:expected=p[k]
                elif k=='indices':expected=ids
                else:expected=p[k][ids]
                if not np.array_equal(measured[k],expected):raise ValueError('Parent emission not reproduced: '+k)
        np.savez(folder/'observation-0000.npz',**measured)
        observations.append(dict(time=0,file=str(folder/'observation-0000.npz'),sha256=sha256(folder/'observation-0000.npz')))
        digests['0']=digest_state(model,optimizer,scheduler)
        if digests['0']!=job['expected_state_digest']:raise ValueError('Parent full state not reproduced')
        del measured
        start=time.perf_counter()
        for t in range(1,horizon+1):
            model.model.train();optimizer.zero_grad(set_to_none=True)
            selected=blocks[t-1];rows=selected//8;offsets=64*(selected%8)
            x=torch.as_tensor(corpus[rows[:,None],offsets[:,None]+np.arange(65)],dtype=torch.long,device=device)
            value=loss(model,x,state['recipe'])
            if not torch.isfinite(value):raise FloatingPointError('Nonfinite return loss')
            value.backward();clip(model,state['recipe']);norm=sum(p.grad.detach().double().square().sum() for p in model.model.parameters()).sqrt()
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
            losses.append(float(value.detach()));norms.append(float(norm));rates.append([g['lr'] for g in optimizer.param_groups])
            optimizer.step();scheduler.step()
            if (t in spec['times'] and job['role']!='unobserved replay') or t==horizon:
                ids,raw=indices(spec['stage'],t,spec['horizon']);measured=observe(model,crops,ids,raw)
                path=folder/f'observation-{t:04d}.npz';np.savez(path,**measured);digest=digest_state(model,optimizer,scheduler)
                observations.append(dict(time=t,file=str(path),sha256=sha256(path)));digests[str(t)]=digest
                if job['name']=='general':reference[t]=dict(path=path,digest=digest)
                if job['role'] in ['replay','unobserved replay']:
                    if digest!=reference[t]['digest']:raise ValueError('Return replay state differs')
                    with np.load(reference[t]['path']) as p:
                        if any(not np.array_equal(measured[k],p[k]) for k in measured):raise ValueError('Return replay differs')
                del measured
                print(name,job['name'],t,'loss',round(losses[-1],6),'seconds',round(time.perf_counter()-start,1),flush=True)
        np.savez(folder/'training.npz',losses=np.array(losses),gradient_norms=np.array(norms),rates=np.array(rates))
        record=dict(name=job['name'],parent=job['arm'],role=job['role'],steps=horizon,observations=observations,state_digests=digests,
                    training_sha256=sha256(folder/'training.npz'),seconds=time.perf_counter()-start)
        write_json(folder/'result.json',record);records.append(record)
        del state,optimizer,scheduler;gc.collect();torch.cuda.empty_cache()
    write_json(dest/'results.json',dict(status='complete',case=name,protocol_sha256=sha256(study/'protocol.json'),records=records,
        parent_reproduction_bitwise=True,replay_bitwise=True,seconds=time.perf_counter()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated(device)))


def run(study):
    study=Path(study).resolve();spec=validate(study,full=True)
    if (study/'launcher.json').exists():raise FileExistsError('Launcher already exists')
    queue=sorted(spec['cases'],key=lambda c:-c['heads']);running={};completed=[];begin=time.perf_counter()
    try:
        while queue or running:
            for gpu in [0,1]:
                if gpu not in running and queue:
                    c=queue.pop(0);log=(study/(c['name']+'.log')).open('w')
                    cmd=[sys.executable,str(Path(__file__).resolve()),'worker','--study',str(study),'--name',c['name'],'--device',f'cuda:{gpu}']
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
                    running[gpu]=(subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT),log,c)
                    print('started',c['name'],gpu,flush=True)
            for gpu,(p,log,c) in list(running.items()):
                if p.poll() is not None:
                    log.close();del running[gpu]
                    if p.returncode:raise RuntimeError('Return worker failed: '+c['name'])
                    completed.append(c['name']);print('completed',c['name'],flush=True)
            time.sleep(2)
        write_json(study/'launcher.json',dict(status='complete',cases=completed,seconds=time.perf_counter()-begin,protocol_sha256=sha256(study/'protocol.json')))
    except BaseException:
        for p,log,c in running.values():p.terminate();p.wait();log.close()
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','worker','validate']);p.add_argument('--study',required=True)
    p.add_argument('--parent-study');p.add_argument('--stage',choices=['qualification','assessment']);p.add_argument('--qualification');p.add_argument('--name');p.add_argument('--device')
    a=p.parse_args()
    if a.action=='prepare':prepare(a.study,a.parent_study,a.stage,a.qualification)
    elif a.action=='run':run(a.study)
    elif a.action=='worker':worker(a.study,a.name,a.device)
    else:validate(a.study)
