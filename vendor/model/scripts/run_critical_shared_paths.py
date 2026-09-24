#!/usr/bin/env python3
"""Run a frozen independent refinement with passive shared-parameter recording.

The successful native numerical program is unchanged. Qualification compares
an uninstrumented execution against the instrumented execution, including
model, Adam, RNG, losses and all original observations, bit for bit.
"""
from companion_paths import child_pythonpath, dispatch_worker, validate_worker_cli
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import torch
from model_rg.provenance import sha256, write_json
from model_rg.replay_equality import require_replay, require_observations, contract
from model_rg.run_outcome import save_artifact, inventory
import run_critical_onepass as native
from prepare_critical_refinement import prepare_refinement

REPO=native.REPO
ROOT=native.ROOT
SELF='scripts/run_critical_shared_paths.py'


def prepare(study, design, selection):
    prepare_refinement(study, design, selection)
    p=json.loads((study/'protocol.json').read_text())
    horizon=p['design']['steps']
    p['observations']['milestone_steps']=sorted(set(p['observations']['milestone_steps']) | set(range(0,horizon+1,512)) | {horizon})
    p['shared_paths']=dict(schema='native-shared-parameter-path-v1',parameter_selector='reslayerAs',
        cadence=256,include_first_update=True,first_raw_and_clipped_gradient=True,
        arithmetic='Detached float32 copies, no changes to tensors, gradients, random draws or optimizer return values.',
        uncertainty='Variation across independently initialized native bodies at a fixed source order and initial shared state; block-increment covariance is not a minibatch innovation covariance.')
    p['qualification']['uninstrumented_comparison']=True
    p['source_sha256'][SELF]=sha256(__file__)
    write_json(study/'protocol.json',p)
    target=study/'executed-source'/SELF
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(REPO/SELF,target)
    if sha256(target)!=p['source_sha256'][SELF]:raise ValueError('The observer changed while preparing it')
    print('Shared-path protocol frozen',sha256(study/'protocol.json'),flush=True)


def train(study,job,device,role,destination):
    p=native.admit(study,scientific=role=='scientific')
    if p.get('shared_paths',{}).get('schema')!='native-shared-parameter-path-v1':
        raise ValueError('Missing shared-path declaration')
    original_optimizer=native.optimizer_for
    original_clip=torch.nn.utils.clip_grad_norm_
    shared=[];values=[];times=[];first={};metadata={};completed=0
    def flatten(gradient=False):
        if not shared or (gradient and any(v.grad is None for _,v in shared)):
            raise ValueError('The shared parameter/gradient selection is incomplete')
        return torch.cat([(v.grad if gradient else v).detach().reshape(-1) for _,v in shared]).cpu().numpy().copy()
    def record(step):
        value=flatten();times.append(step);values.append(value)
    def instrument_optimizer(model,multiplier):
        nonlocal shared
        optimizer=original_optimizer(model,multiplier)
        shared=[(name,v) for name,v in model.model.named_parameters() if 'reslayerAs' in name]
        selected={id(v) for _,v in shared}
        groups=[group for group in optimizer.param_groups if any(id(v) in selected for v in group['params'])]
        if len(groups)!=1 or not selected <= {id(v) for v in groups[0]['params']}:
            raise ValueError('Shared coordinates span different optimizer groups')
        group=groups[0]
        metadata.update(names=[name for name,_ in shared],shapes=[list(v.shape) for _,v in shared],
            rate=group['lr'],weight_decay=group['weight_decay'],epsilon=group['eps'],betas=list(group['betas']))
        record(0)
        return optimizer
    def after_update(step):
        nonlocal completed
        completed=step
        if step==1 or step%p['shared_paths']['cadence']==0 or step==job['steps']:
            record(step)
    def clip(*args,**kwargs):
        if completed==0:first['raw_gradient']=flatten(True)
        result=original_clip(*args,**kwargs)
        if completed==0:first['clipped_gradient']=flatten(True)
        return result
    native.optimizer_for=instrument_optimizer
    torch.nn.utils.clip_grad_norm_=clip
    try:
        m=native.train(study,job,device,role,destination,after_update=after_update)
    finally:
        native.optimizer_for=original_optimizer
        torch.nn.utils.clip_grad_norm_=original_clip
    if m['completed_steps']!=completed:
        raise ValueError('The shared observer counted different optimizer updates')
    # Do not let an empty recorder or an I/O error erase a base failure.
    m.setdefault('artifact_save_errors',[]);m.setdefault('original_failure',None)
    original_stage=m.get('stage','complete')
    m['stage']='recorder_save'
    def save_values():
        if values and times[-1]!=completed:record(completed)
        width=sum(int(np.prod(shape)) for shape in metadata.get('shapes',[]))
        np.save(destination/'shared-parameters.npy',np.stack(values) if values else np.empty((0,width),dtype=np.float32),allow_pickle=False)
    writers={
        'shared-parameters.npy':save_values,
        'shared-steps.npy':lambda:np.save(destination/'shared-steps.npy',np.array(times,dtype=np.int64),allow_pickle=False),
        'first-gradient.npz':lambda:np.savez_compressed(destination/'first-gradient.npz',**first),
        'shared-metadata.json':lambda:write_json(destination/'shared-metadata.json',dict(
            schema='native-shared-parameter-path-v1',protocol_sha256=sha256(study/'protocol.json'),role=role,job=job,
            shared_parameter_count=len(values[0]) if values else 0,steps=times,**metadata))}
    pending=None
    for name,writer in writers.items():
        exc=save_artifact(destination,m,name,writer)
        if exc is not None and m['status']=='program_error' and pending is None:pending=exc
    m['shared_path_recorded']=all(name in m['artifacts'] for name in writers)
    if m['status']!='complete':m['checkpoint_resumable']=False
    m['stage']=m['original_failure']['stage'] if m['original_failure'] else original_stage
    write_json(destination/'manifest.json',m)
    if pending is not None:raise pending
    return m


def qualify(study,device):
    p=native.admit(study);q=p['qualification']
    job=dict(p['jobs'][0],run_id='widest-qualification',heads=q['heads'],control=q['control'],
             steps=q['steps'],save_optimizer=True)
    paths=[study/'qualification'/name for name in ['native','replay']]
    a=native.train(study,job,device,'qualification',paths[0])
    b=train(study,job,device,'qualification',paths[1])
    if a['status']!='complete' or b['status']!='complete':raise ValueError('Native qualification failed')
    states=[torch.load(path/'final-state.pt',map_location='cpu',weights_only=False) for path in paths]
    require_replay(*states)
    with np.load(paths[0]/'observations.npz') as x,np.load(paths[1]/'observations.npz') as y:
        require_observations(x,y)
    meta=json.loads((paths[1]/'shared-metadata.json').read_text())
    expected=np.concatenate([states[1]['model'][name].reshape(-1).numpy() for name in meta['names']])
    require_replay(np.asarray(np.load(paths[1]/'shared-parameters.npy',mmap_mode='r')[-1]),expected)
    checked={str(path/name):sha256(path/name) for path,m in zip(paths,[a,b])
             for name in ['manifest.json',*m['artifacts']]}
    write_json(study/'qualification/verification.json',dict(status='passed',
        protocol_sha256=sha256(study/'protocol.json'),checked_sha256=checked,
        replay_contract=contract(),full_state_bitwise_replay=True,uninstrumented_vs_instrumented_bitwise=True,
        shared_endpoint_matches_state=True,scientific_updates=0,qualification_updates=2*q['steps'],
        maximum_cuda_memory_bytes=max(a['maximum_cuda_memory_bytes'],b['maximum_cuda_memory_bytes']),
        runtime_seconds=sum(m['runtime_seconds'] for m in [a,b])))
    print('Uninstrumented/instrumented full-state qualification passed',flush=True)


def run(study):
    p=native.admit(study,scientific=True)
    def queue(device,jobs):
        for job in jobs:
            path=study/'runs'/job['run_id']
            if path.exists():
                m=json.loads((path/'manifest.json').read_text())
                if m['job']!=job or m['protocol_sha256']!=sha256(study/'protocol.json') or (m['status']=='complete' and not m.get('shared_path_recorded')):
                    raise ValueError('Unfinished or foreign existing run')
                for name,digest in m['artifacts'].items():
                    if sha256(path/name)!=digest:raise ValueError('Changed existing artifact')
                print('RETAIN',job['run_id'],m['status'],flush=True)
                continue
            logs=study/'logs';logs.mkdir(exist_ok=True)
            with (logs/(job['run_id']+'.log')).open('x') as log:
                dispatch_worker([sys.executable,str(REPO/SELF),'worker','--study',str(study),
                    '--run-id',job['run_id'],'--device',device],cwd=REPO,
                    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),
                    stdout=log,stderr=subprocess.STDOUT,check=True)
            print('COMPLETE',job['run_id'],flush=True)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(queue,f'cuda:{i}',p['jobs'][i::2]) for i in range(2)]
            for f in futures:f.result()
    finally:
        outcomes=inventory(study,p)
    write_json(study/'execution-complete.json',outcomes)



if __name__ == '__main__':
    validate_worker_cli(__file__)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','qualify','run','worker'])
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--design',type=Path);parser.add_argument('--selection',type=Path)
    parser.add_argument('--run-id');parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();study=args.study.resolve()
    if study==ROOT or not study.is_relative_to(ROOT):raise ValueError('Use the authorized experiment workspace')
    if args.action=='prepare':prepare(study,args.design.resolve(),args.selection.resolve())
    elif args.action=='qualify':qualify(study,args.device)
    elif args.action=='run':run(study)
    else:
        protocol=native.admit(study,scientific=True)
        selected=[j for j in protocol['jobs'] if j['run_id']==args.run_id]
        if len(selected)!=1:raise ValueError('Expected one frozen scientific job')
        train(study,selected[0],args.device,'scientific',study/'runs'/args.run_id)
