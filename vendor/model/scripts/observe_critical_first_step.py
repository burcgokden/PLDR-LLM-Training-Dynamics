#!/usr/bin/env python3
"""Acquire the first shared Adam force while replaying a recorded native prefix.

Only the fixed-dimensional residual metric networks are observed. Their
optimizer group also contains head-specific PLGA parameters; global clipping
uses the entire native gradient. The observer returns every intercepted
value unchanged and verifies the full recorded 64-update prefix afterward.
"""
from companion_paths import child_pythonpath
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from model_rg.provenance import sha256, write_json
import run_critical_onepass as native

REPO=Path(__file__).resolve().parents[1]
ROOT=native.ROOT
SELF='scripts/observe_critical_first_step.py'


def prepare(study, destination):
    if destination.exists() or destination==ROOT or not destination.is_relative_to(ROOT):
        raise ValueError('A fresh experiment-workspace destination is required')
    parent=native.admit(study)
    jobs=[job for job in parent['jobs'] if job['control']==1.]
    expected=len(parent['design']['heads'])*len(parent['design']['seeds'])*len(parent['design']['environments'])
    if len(jobs)!=expected:raise ValueError('The complete unit-control initialization family is required')
    checked={str(study/'protocol.json'):sha256(study/'protocol.json')}
    for job in jobs:
        folder=study/'runs'/job['run_id'];manifest=folder/'manifest.json';m=json.loads(manifest.read_text())
        if m['status']!='complete' or m['job']!=job:raise ValueError('The unit-control parent must be complete')
        checked[str(manifest)]=sha256(manifest)
        checked[str(folder/'observations.npz')]=m['artifacts']['observations.npz']
        if sha256(folder/'observations.npz')!=m['artifacts']['observations.npz']:raise ValueError('Changed parent observations')
    destination.mkdir(parents=True)
    write_json(destination/'protocol.json',dict(schema='critical-first-step-replay-v1',role='qualification',
        study=str(study),jobs=jobs,replay_steps=64,scientific_updates=0,
        replay_updates=64*len(jobs),source_sha256={**parent['source_sha256'],SELF:sha256(__file__)},
        input_sha256=checked,observed_parameters='Only names containing reslayerAs, fixed across widths and fixed at initialization.',
        optimizer_group='Rate 0.0003*g for residual metric networks and head-specific PLGA parameters; global clipping uses all parameters.',
        primary='Per-coordinate shared-gradient, normalized-force and first-step covariance, with identical initial shared parameters.',
        scope='Reconstruction of already counted native prefixes, not independent training replicas or additional scientific trajectories.'))
    print(destination/'protocol.json',flush=True)


def admit(root):
    p=json.loads((root/'protocol.json').read_text())
    if p['schema']!='critical-first-step-replay-v1' or p['role']!='qualification' or p['scientific_updates']!=0:
        raise ValueError('Wrong replay role')
    for name,digest in p['source_sha256'].items():
        if sha256(REPO/name)!=digest:raise ValueError('Changed first-step observer source')
    for name,digest in p['input_sha256'].items():
        if sha256(name)!=digest:raise ValueError('Changed replay input')
    return p


def worker(root, run_id, device):
    p=admit(root);job=next(j for j in p['jobs'] if j['run_id']==run_id)
    output=root/'runs'/run_id
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    parent=Path(p['study']);saved={};shared=[];step_count=0
    original_optimizer=native.optimizer_for
    original_clip=torch.nn.utils.clip_grad_norm_
    def flatten(gradient=False):
        if gradient and any(parameter.grad is None for _,parameter in shared):
            raise ValueError('A selected shared parameter has no gradient')
        return torch.cat([(parameter.grad if gradient else parameter).detach().reshape(-1)
                          for _,parameter in shared]).cpu().numpy().copy()
    def instrument_optimizer(model, multiplier):
        nonlocal shared,step_count
        optimizer=original_optimizer(model,multiplier)
        shared=[(name,parameter) for name,parameter in model.model.named_parameters() if 'reslayerAs' in name]
        saved['names']=[name for name,_ in shared]
        saved['shapes']=[list(parameter.shape) for _,parameter in shared]
        saved['rate']=optimizer.param_groups[0]['lr'];saved['decay']=optimizer.param_groups[0]['weight_decay']
        saved['epsilon']=optimizer.param_groups[0]['eps']
        original_step=optimizer.step
        def step(*args,**kwargs):
            nonlocal step_count
            if step_count==0:
                saved['initial_shared']=flatten()
                saved['clipped_gradient']=flatten(True)
            result=original_step(*args,**kwargs)
            if step_count==0:saved['after_shared']=flatten()
            step_count+=1
            return result
        optimizer.step=step
        return optimizer
    def clip(*args,**kwargs):
        if step_count==0:saved['raw_gradient']=flatten(True)
        result=original_clip(*args,**kwargs)
        if step_count==0:saved['global_gradient_norm']=float(result)
        return result
    native.optimizer_for=instrument_optimizer
    torch.nn.utils.clip_grad_norm_=clip
    replay_job=dict(job,steps=p['replay_steps'])
    try:
        m=native.train(parent,replay_job,device,'qualification',output/'native-replay')
    finally:
        native.optimizer_for=original_optimizer
        torch.nn.utils.clip_grad_norm_=original_clip
    if m['status']!='complete' or step_count!=64 or m['scientific_updates']!=0:
        raise ValueError('Incomplete native prefix replay')
    original_manifest=json.loads((parent/'runs'/run_id/'manifest.json').read_text())
    if m['initial_parameter_sha256']!=original_manifest['initial_parameter_sha256']:
        raise ValueError('Initialization differs from the recorded trajectory')
    with np.load(parent/'runs'/run_id/'observations.npz') as old,np.load(output/'native-replay/observations.npz') as replay:
        for key in ['training_loss','gradient_norm','blocks']:
            np.testing.assert_array_equal(replay[key],old[key][:64])
        for key in ['heads','nll_path']:
            indices=[int(np.flatnonzero(old['steps']==t)[0]) for t in replay['steps']]
            np.testing.assert_array_equal(replay[key],old[key][indices])
        np.testing.assert_array_equal(replay['logits_0'],old['logits_0'])
        if saved['global_gradient_norm']!=old['gradient_norm'][0]:raise ValueError('First-step gradient norm differs')
    raw=output/'first-step.npz'
    np.savez_compressed(raw,**{name:saved[name] for name in ['initial_shared','after_shared','raw_gradient','clipped_gradient']})
    metadata={name:value for name,value in saved.items() if not isinstance(value,np.ndarray)}
    write_json(output/'manifest.json',dict(schema='critical-first-step-acquisition-v1',status='complete',role='qualification',
        job=job,protocol_sha256=sha256(root/'protocol.json'),scientific_updates=0,replay_updates=64,
        recorded_prefix_bitwise=True,raw_sha256=sha256(raw),native_manifest_sha256=sha256(output/'native-replay/manifest.json'),
        first_step=metadata,maximum_cuda_memory_bytes=m['maximum_cuda_memory_bytes'],runtime_seconds=m['runtime_seconds']))
    print('FIRST STEP RECONSTRUCTED',run_id,flush=True)


def run(root):
    p=admit(root)
    def queue(device,jobs):
        for job in jobs:
            path=root/'runs'/job['run_id']
            if path.exists():
                m=json.loads((path/'manifest.json').read_text())
                if m['status']!='complete' or m['job']!=job or m['protocol_sha256']!=sha256(root/'protocol.json'):
                    raise ValueError('Unfinished or unbound replay collision')
                if sha256(path/'first-step.npz')!=m['raw_sha256']:raise ValueError('Changed replay acquisition')
                continue
            with (root/(job['run_id']+'.log')).open('x') as log:
                subprocess.run([sys.executable,str(REPO/SELF),'worker','--destination',str(root),
                                '--run-id',job['run_id'],'--device',device],cwd=REPO,
                               env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),OPENBLAS_NUM_THREADS='2'),
                               stdout=log,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(queue,'cuda:'+str(device),p['jobs'][device::2]) for device in range(2)]
        for future in futures:future.result()
    write_json(root/'execution-complete.json',dict(status='complete',scientific_updates=0,
        replay_updates=p['replay_updates'],paths=len(p['jobs']),protocol_sha256=sha256(root/'protocol.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','worker'])
    parser.add_argument('--study',type=Path);parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--run-id');parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();destination=args.destination.resolve()
    if args.action=='prepare':prepare(args.study.resolve(),destination)
    elif args.action=='run':run(destination)
    else:worker(destination,args.run_id,args.device)
