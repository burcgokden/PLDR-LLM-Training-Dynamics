#!/usr/bin/env python
"""Frozen, bounded native fine-tuning with distinct remaining corpus blocks."""
from companion_paths import legacy_path
import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
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
from model_rg.finetuning import mixture_stream
from model_rg.onepass_regimes import optimizer_and_scheduler
from model_rg.provenance import sha256, write_json
from model_rg.schedules import loss as native_loss, clip as native_clip
from model_rg.training import TrainingModel
from measure_law_closure import digest_state
from finetuning_observation import observe
from finetuning_design import check_design, check_qualification, expected_arms, canonical

RHOS=[0.,.125,.25,.5,.75,.875,1.]


def runtime():
    return dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
                devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                threads=4,tf32=False,dtype='float32',microbatch=32)


def source_files():
    names=['scripts/finetuning_study.py','scripts/finetuning_observation.py','scripts/measure_law_closure.py',
           'scripts/finetuning_design.py', 'scripts/verify_finetuning.py',
           'scripts/finetuning_verification_design.py']
    names += [str(p.relative_to(REPO)) for p in (REPO/'src/model_rg').glob('*.py')]
    return {n:sha256(REPO/n) for n in sorted(names)}


def arms():
    return expected_arms()


def stream(spec, arm):
    with np.load(Path(spec['data'])/'streams.npz') as f:
        if arm['rho'] is None: return f['general'][:spec['horizon']*32].reshape(-1,32)
        return mixture_stream(f['technical'],f['narrative'],f['uniforms'][:spec['horizon']],arm['rho'])


def validate(study, case=None, full=False):
    study=Path(study); spec=json.loads((study/'protocol.json').read_text())
    validate_spec(spec, case=case, full=full)
    return spec


def validate_spec(spec, case=None, full=False):
    check_design(spec)
    if runtime()!=spec['runtime']: raise ValueError('Execution platform changed')
    if source_files()!=spec['sources']: raise ValueError('Producer source inventory changed')
    report_path=Path(spec['data'])/'manifest.json'
    report=json.loads(report_path.read_text())
    if report['status']!='complete': raise ValueError('Incomplete data')
    required={str(Path(spec['data'])/n) for n in list(report['files'])+['manifest.json','streams.npz']}
    required.add(str(Path(spec['root'])/'data/refinedweb-onepass-524288/tokens.npy'))
    required.update(str(p) for p in (Path(spec['root'])/'assets/PLDR-LLM-v51-SOC-110M-1').iterdir()
                    if p.is_file() and p.suffix in ['.py','.json','.model'])
    if set(spec['inputs'])!=required: raise ValueError('Scientific input inventory changed')
    for n,h in report['files'].items():
        if spec['inputs'][str(Path(spec['data'])/n)]!=h: raise ValueError('Data manifest/input mismatch')
    for p,h in spec['inputs'].items():
        if full or not p.endswith('tokens.npy'):
            if sha256(p)!=h: raise ValueError('Input changed: '+p)
    if case is not None and case not in spec['cases']: raise ValueError('Unregistered worker case')
    cases=spec['cases'] if full else ([case] if case is not None else [])
    for c in cases:
        if sha256(c['state'])!=c['state_sha256']: raise ValueError('Incoming state changed')
    check_qualification(spec, REPO)
    return spec


def prepare(study, root, stage, qualification):
    if stage not in ('qualification','assessment'): raise ValueError('Unknown preparation stage')
    study=Path(study).resolve(); root=Path(root).resolve(); data=study.parent/'data'
    if not study.is_relative_to(root): raise ValueError('Unauthorized experiment destination')
    if study.exists(): raise FileExistsError(study)
    if not (data/'manifest.json').exists(): raise ValueError('Prepare data first')
    report=json.loads((data/'manifest.json').read_text())
    if report['status']!='complete': raise ValueError('Incomplete data')
    for n,h in report['files'].items():
        if sha256(data/n)!=h: raise ValueError('Changed data selection')
    source=source_files(); cases=[]
    for h in [4,8,14]:
        for seed in ([640101] if stage=='qualification' else [640101,640102,640103,640104]):
            folder=root/f'scheduled-training-feasible-20260908/runs/compact-reference1-h{h}-s{seed}'
            manifest=json.loads((folder/'manifest.json').read_text())
            if manifest['status']!='complete' or manifest['completed_step']!=40960:
                raise ValueError('Incomplete pretraining')
            if manifest['arguments']['stream_seed']!=640001 or manifest['data_law']['consumed_blocks']!=1310720:
                raise ValueError('Nonmatching incoming data law')
            cases.append(dict(name=f'h{h}-s{seed}',heads=h,seed=seed,
                state=str(folder/'final-training-state.pt'),state_sha256=sha256(folder/'final-training-state.pt'),
                manifest=str(folder/'manifest.json'),manifest_sha256=sha256(folder/'manifest.json'),
                profile=manifest['recipe']))
    horizon=2 if stage=='qualification' else 512
    if not (data/'streams.npz').exists(): raise ValueError('Prepare frozen streams before preparing the study')
    inputs={str(data/n):sha256(data/n) for n in list(report['files'])+['manifest.json','streams.npz']}
    inputs[str(root/'data/refinedweb-onepass-524288/tokens.npy')]=report['input_files'][str(root/'data/refinedweb-onepass-524288/tokens.npy')]
    assets=root/'assets/PLDR-LLM-v51-SOC-110M-1'
    for p in assets.iterdir():
        if p.is_file() and p.suffix in ['.py','.json','.model']: inputs[str(p)]=sha256(p)
    spec=dict(schema='pldr-finetuning-study-v1',stage=stage,created_at=datetime.now(timezone.utc).isoformat(),
        cases=cases,arms=arms(),horizon=horizon,times=[0,1,2] if horizon==2 else [0,16,64,128,256,512],
        batch_size=32,runtime=runtime(),sources=source,inputs=inputs,data=str(data),root=str(root),
        evaluation='Qualification: 18 reserved contexts, six per lexical stratum. Assessment: 96 at intermediate times and all 480 at initial/final time. Each stratum has fixed calibration and held-out document halves. Prefix32 versus prefix64 operator discrepancy is an explicitly different coupling from stochastic generation in the SOC study.',
        law='Each token block independently selects the technical reservoir with probability rho using shared uniforms, otherwise narrative. Reservoirs are separately uniformly permuted, exhausted only without replacement; all arms exclude pretraining and reserved documents. General control samples the unstratified remaining population. Data draws are shared across initialization conditions, not extra independent model seeds.',
        intervention='Full native all-nonpadding-target learning at the incoming floor rate. Component controls set the other group rate to zero while retaining its moment recursion. Reset controls zero both moments and their counters but retain the incoming scheduler phase.',
        statistical_units='Three widths, four body initializations per width, one shared initial metric learner, one common pretraining corpus and stream. Conditional on that environment. Controls use body seed 640101 only.',
        hypotheses='Domain selection may expose multiple predictive response sectors. A critical interpretation additionally needs stable sector identity, control-window and time/size scaling. A raw covariance slope, a small global tensor discrepancy or a finite response rank does not meet that criterion.',
        analysis='Report every cell. Fit affine and midpoint-quadratic mixture responses at rho=0,0.5,1 and score rho=0.125,0.25,0.75,0.875 on held-out documents. Inspect connected across-initialization covariance at fixed input, proper-prefix predictive Fisher response, and cross-layer sectors. Finite size slopes and peak positions are descriptive unless a coherent scaling law independently qualifies.',
        budgets=dict(max_worker_seconds=10800,assessment_updates=59904,primary_paths=96,control_paths=21),
        numeric=dict(signal_floor=5e-7,replay='Every width replays 16 general updates, or all qualification updates; compare matching observations and complete state hash.'))
    if stage=='assessment':
        if qualification is None: raise ValueError('Assessment requires qualification')
        q=Path(qualification).resolve(); result=json.loads(q.read_text())
        if result.get('status')!='passed': raise ValueError('Qualification did not pass')
        qs=json.loads((q.parent/'protocol.json').read_text())
        if qs['sources']!=source or qs['arms']!=spec['arms'] or qs['runtime']!=spec['runtime']:
            raise ValueError('Qualification/source/design mismatch')
        spec['qualification']=dict(path=str(q),sha256=sha256(q))
    validate_spec(spec,full=True)
    study.mkdir(parents=True)
    for n in source:
        p=study/'executed-source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((REPO/n).read_bytes())
    write_json(study/'protocol.json',spec)
    print('Frozen',stage,'cases',len(cases),'horizon',horizon,flush=True)


def observation_indices(stage, timepoint, horizon):
    if stage=='qualification':
        indices=np.array([j+k for j in [0,80,160,240,320,400] for k in range(3)])
    elif timepoint in [0,horizon]: indices=np.arange(480)
    else: indices=np.array([j+k for j in [0,80,160,240,320,400] for k in range(16)])
    return indices,np.array([0,80,160,240,320,400])


def reset(model, optimizer, scheduler, state, arm):
    model.model.requires_grad_(True)
    optimizer.zero_grad(set_to_none=True);model.capture=False;model.eta=None;model.head_outputs={}
    model.model.load_state_dict(state['model'])
    optimizer.load_state_dict(copy.deepcopy(state['optimizer']));scheduler.load_state_dict(copy.deepcopy(state['scheduler']))
    if arm.get('reset'):
        for s in optimizer.state.values():
            s['exp_avg'].zero_();s['exp_avg_sq'].zero_();s['step'].zero_()
    if arm.get('group')=='generator':optimizer.param_groups[1]['lr']=0.
    if arm.get('group')=='body':optimizer.param_groups[0]['lr']=0.


def save_state(path, model, optimizer, scheduler, state, arm, blocks):
    value=dict(model={n:p.detach().cpu() for n,p in model.model.state_dict().items()},
               optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),recipe=state['recipe'],
               pretraining_step=40960,finetuning_step=len(blocks),arm=arm,consumed_finetuning_blocks=blocks)
    torch.save(value,path)


def worker(study, name, device):
    study=Path(study).resolve(); initial=json.loads((study/'protocol.json').read_text())
    check_design(initial)
    matches=[c for c in initial['cases'] if c['name']==name]
    if len(matches)!=1: raise ValueError('Unregistered or duplicate worker case')
    case=matches[0]
    spec=validate(study,case=case)
    if device not in ['cuda:0','cuda:1']: raise ValueError('One physical GPU per worker')
    dest=study/name;dest.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(spec['root']);data=Path(spec['data'])
    corpus=np.load(root/'data/refinedweb-onepass-524288/tokens.npy',mmap_mode='r')
    with np.load(data/'evaluation.npz') as f:crops=f['crops'];evdocs=f['document_ids']
    state=torch.load(case['state'],map_location='cpu',weights_only=False)
    if state['step']!=40960 or state['scheduler']['last_epoch']!=40960: raise ValueError('Incorrect incoming age')
    if canonical(state['recipe'])!=canonical(case['profile']): raise ValueError('Loaded recipe/profile mismatch')
    for moment in state['optimizer']['state'].values():
        if int(moment['step'])!=40960 or not all(k in moment for k in ['exp_avg','exp_avg_sq']):
            raise ValueError('Incomplete incoming optimizer memory')
    model=TrainingModel(root/'assets/PLDR-LLM-v51-SOC-110M-1',case['heads'],case['seed'],device)
    optimizer,scheduler=optimizer_and_scheduler(model,state['recipe'])
    zero=dict(group='full',reset=False)
    reset(model,optimizer,scheduler,state,zero);incoming_digest=digest_state(model,optimizer,scheduler)
    indices,raw=observation_indices(spec['stage'],0,spec['horizon'])
    baseline=observe(model,crops,indices,raw);np.savez(dest/'initial-observation.npz',**baseline)
    jobs=[a for a in spec['arms'] if a['role']=='primary' or case['seed']==640101]
    general=next(a for a in jobs if a['name']=='general')
    jobs.append(dict(general,name='general-replay',role='replay'))
    if spec['stage']=='qualification':jobs.append(dict(general,name='general-unobserved',role='unobserved replay'))
    records=[];begin=time.perf_counter();reference={};torch.cuda.reset_peak_memory_stats(device)
    for arm in jobs:
        if time.perf_counter()-begin>spec['budgets']['max_worker_seconds']: raise TimeoutError('Worker budget exceeded')
        reset(model,optimizer,scheduler,state,zero)
        if digest_state(model,optimizer,scheduler)!=incoming_digest: raise ValueError('Incomplete state restore')
        reset(model,optimizer,scheduler,state,arm)
        folder=dest/arm['name'];folder.mkdir()
        blocks=stream(spec,arm)
        horizon=min(16,spec['horizon']) if arm['role']=='replay' else spec['horizon']
        blocks=blocks[:horizon]
        if np.isin(blocks//8,evdocs).any() or len(np.unique(blocks))!=blocks.size:
            raise ValueError('Source/evaluation collision or repeated block')
        losses=[];gradient_norms=[];observations=[];digests={};rates=[];start=time.perf_counter()
        np.save(folder/'blocks.npy',blocks)
        for t in range(1,horizon+1):
            model.model.train();optimizer.zero_grad(set_to_none=True)
            rows=blocks[t-1]//8;offsets=64*(blocks[t-1]%8)
            batch=corpus[rows[:,None],offsets[:,None]+np.arange(65)]
            x=torch.as_tensor(batch,dtype=torch.long,device=device)
            value=native_loss(model,x,state['recipe'])
            if not torch.isfinite(value): raise FloatingPointError('Nonfinite training objective')
            value.backward();native_clip(model,state['recipe'])
            norm=sum(p.grad.detach().double().square().sum() for p in model.model.parameters()).sqrt()
            if not torch.isfinite(norm): raise FloatingPointError('Nonfinite clipped gradient')
            losses.append(float(value.detach()));gradient_norms.append(float(norm));rates.append([g['lr'] for g in optimizer.param_groups])
            optimizer.step();scheduler.step()
            if arm['group']=='generator':optimizer.param_groups[1]['lr']=0.
            if arm['group']=='body':optimizer.param_groups[0]['lr']=0.
            if (t in spec['times'] and arm['role']!='unobserved replay') or t==horizon:
                ids,raw=observation_indices(spec['stage'],t,spec['horizon'])
                measured=observe(model,crops,ids,raw)
                path=folder/f'observation-{t:04d}.npz';np.savez(path,**measured)
                digest=digest_state(model,optimizer,scheduler);digests[str(t)]=digest
                observations.append(dict(time=t,file=str(path),sha256=sha256(path)))
                if arm['name']=='general':reference[t]=dict(path=path,digest=digest)
                if arm['role'] in ['replay','unobserved replay']:
                    ref=reference[t]
                    if digest!=ref['digest']: raise ValueError('Replay full state mismatch')
                    with np.load(ref['path']) as saved:
                        if any(not np.array_equal(v,saved[k]) for k,v in measured.items()):
                            raise ValueError('Replay observation mismatch')
                del measured
                print(name,arm['name'],t,'loss',round(losses[-1],6),'seconds',round(time.perf_counter()-start,1),flush=True)
        np.savez(folder/'training.npz',losses=losses,gradient_norms=gradient_norms,rates=rates)
        checkpoint=None
        if spec['stage']=='assessment' and arm['role']=='primary' and arm['rho'] in [None,0.,.5,1.]:
            checkpoint=folder/'final-state.pt';save_state(checkpoint,model,optimizer,scheduler,state,arm,blocks)
        record=dict(arm=arm,steps=horizon,seconds=time.perf_counter()-start,observations=observations,
                    blocks_sha256=sha256(folder/'blocks.npy'),training_sha256=sha256(folder/'training.npz'),
                    state_digests=digests,checkpoint=str(checkpoint) if checkpoint else None,
                    checkpoint_sha256=sha256(checkpoint) if checkpoint else None)
        write_json(folder/'result.json',dict(status='complete',**record));records.append(record)
    write_json(dest/'results.json',dict(status='complete',case=case,records=records,
        protocol_sha256=sha256(study/'protocol.json'),incoming_digest=incoming_digest,
        initial_observation_sha256=sha256(dest/'initial-observation.npz'),
        seconds=time.perf_counter()-begin,peak_cuda_bytes=torch.cuda.max_memory_allocated(device),
        parameters=sum(p.numel() for p in model.model.parameters()),replay_bitwise=True))
    print('COMPLETE',name,flush=True)


def run(study):
    spec=validate(study,full=True);study=Path(study)
    pending=sorted(spec['cases'],key=lambda c:-c['heads']);active={};results=[]
    while pending or active:
        for gpu in [0,1]:
            if gpu in active or not pending:continue
            case=pending.pop(0);log=(study/(case['name']+'.log')).open('x')
            command=[sys.executable,str(Path(__file__).resolve()),'worker','--study',str(study),
                     '--case',case['name'],'--device',f'cuda:{gpu}']
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            active[gpu]=(process,log,case['name'])
        for gpu,(p,log,name) in list(active.items()):
            if p.poll() is not None:
                log.close();results.append(dict(case=name,returncode=p.returncode));del active[gpu]
                if p.returncode:
                    for other,_,_ in active.values():other.terminate()
                    for other,otherlog,_ in active.values():other.wait();otherlog.close()
                    write_json(study/'launcher.json',dict(status='failed',results=results))
                    raise RuntimeError('Worker failed: '+name)
        time.sleep(.25)
    write_json(study/'launcher.json',dict(status='complete',results=results,protocol_sha256=sha256(study/'protocol.json')))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run','worker','validate'])
    ap.add_argument('--study',required=True);ap.add_argument('--root',default=legacy_path('/pldr-data/model'))
    ap.add_argument('--stage',choices=['qualification','assessment']);ap.add_argument('--qualification')
    ap.add_argument('--case');ap.add_argument('--device');a=ap.parse_args()
    if a.action=='prepare':
        if a.stage is None:ap.error('--stage is required')
        prepare(a.study,a.root,a.stage,a.qualification)
    elif a.action=='run':run(a.study)
    elif a.action=='worker':worker(a.study,a.case,a.device)
    else:validate(a.study,full=True);print('admitted')
