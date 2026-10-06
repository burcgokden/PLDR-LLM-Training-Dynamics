#!/usr/bin/env python3
"""Measure complete common-centroid and operator projections at saved native states."""
from companion_paths import child_pythonpath
from companion_paths import configured_path
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from model_rg.training import TrainingModel
from model_rg.criticality import parameter_digest
from model_rg.provenance import sha256,write_json

REPO=Path(__file__).resolve().parents[1]
ROOT=Path(configured_path('data:model'))
NATIVE=ROOT/'assets/PLDR-LLM-v51-SOC-110M-1'
SOURCES=['scripts/observe_critical_collectives.py','src/model_rg/training.py','src/model_rg/native.py',
         'src/model_rg/criticality.py','src/model_rg/provenance.py']


def prepare(study,destination,qualification_parent=None):
    if destination.exists():raise FileExistsError(destination)
    protocol=json.loads((study/'protocol.json').read_text())
    jobs=[];excluded=[];checked={str(study/'protocol.json'):sha256(study/'protocol.json'),
                    str(study/'selection.npz'):sha256(study/'selection.npz')}
    if qualification_parent is not None and qualification_parent not in [j['run_id'] for j in protocol['jobs']]:
        raise ValueError('Unlisted qualification parent')
    for job in protocol['jobs']:
        if qualification_parent is not None and job['run_id']!=qualification_parent:continue
        root=study/'runs'/job['run_id']
        m=json.loads((root/'manifest.json').read_text())
        checked[str(root/'manifest.json')]=sha256(root/'manifest.json')
        if m['status']!='complete':
            excluded.append(dict(run_id=job['run_id'],status=m['status'],error=m.get('error')))
            continue
        for name in ['final-state.pt','observations.npz']:
            digest=sha256(root/name)
            if digest!=m['artifacts'][name]:raise ValueError('Changed native artifact')
            checked[str(root/name)]=digest
        checked[str(root/'manifest.json')]=sha256(root/'manifest.json')
        jobs.append(job)
    for name,digest in protocol['input_sha256'].items():
        if sha256(name)!=digest:raise ValueError('Changed parent input')
        checked[name]=digest
    destination.mkdir(parents=True)
    write_json(destination/'protocol.json',dict(schema='critical-collective-observation-v1',
        role='observation',qualification_parent=qualification_parent,study=str(study),jobs=jobs,excluded=excluded,input_sha256=checked,
        producer_sha256={name:sha256(REPO/name) for name in SOURCES},
        source_contexts=64,full_vocabulary_contexts=8,
        observable='Complete 64-coordinate native A common centroid per layer/head, full head-mean G, eight fixed per-head operator projections, and predictive logits.',
        arithmetic='Native float32 GPU forward; float64 reductions; the primary training observations are independently replayed.',
        projection_seed=9152580,operator_projections=8,training_updates=0))


@torch.no_grad()
def measure(destination,run_id,device):
    p=json.loads((destination/'protocol.json').read_text())
    for name,digest in p['producer_sha256'].items():
        if sha256(REPO/name)!=digest:raise ValueError('Changed observation source')
    jobs=[j for j in p['jobs'] if j['run_id']==run_id]
    if len(jobs)!=1:raise ValueError('Unlisted observation')
    job=jobs[0];study=Path(p['study']);root=study/'runs'/run_id
    for path in [study/'selection.npz',root/'manifest.json',root/'final-state.pt',root/'observations.npz',NATIVE/'modeling_pldrllm.py',NATIVE/'configuration_pldrllm.py']:
        if sha256(path)!=p['input_sha256'][str(path)]:raise ValueError('Changed observation input')
    out=destination/'runs'/run_id;out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_device(device);torch.cuda.init();torch.cuda.reset_peak_memory_stats(device)
    state=torch.load(root/'final-state.pt',map_location='cpu',weights_only=False)
    model=TrainingModel(NATIVE,job['heads'],job['seed'],device)
    model.model.load_state_dict(state['model']);model.model.eval();del state
    native=json.loads((root/'manifest.json').read_text())
    if parameter_digest(model)!=native['final_parameter_sha256']:raise ValueError('Loaded parameter identity differs')
    selection=np.load(study/'selection.npz')
    probes=torch.tensor(selection['probes'][:64],device=device,dtype=torch.long)
    projection=np.random.default_rng(p['projection_seed']).normal(size=(4096,p['operator_projections']))
    projection=np.linalg.qr(projection)[0]
    proj=torch.as_tensor(projection,device=device,dtype=torch.float64)
    centroids,operators,mean_operators,fields,losses=[],[],[],[],[];logits=None
    for start in range(0,64,16):
        ids=probes[start:start+16]
        result=model.forward(ids[:,:64],capture=True)
        z=result.logits[:,-1].double()
        if start==0:logits=z[:8].float().cpu().numpy()
        losses.extend(torch.nn.functional.cross_entropy(z,ids[:,64],reduction='none').cpu().tolist())
        cc,gg,gm,ff=[],[],[],[]
        for a,_,_,_,_,g,weights in result.pldr_attentions:
            a=a.double();g=g.double();c=a.mean(-2)
            attention=weights[:,:,-1].double()
            entropy=-(attention*attention.clamp_min(1e-300).log()).sum(-1)/np.log(64)
            energy=a.square().mean((-2,-1))
            row=(a-a.mean(-2,keepdim=True)).square().mean((-2,-1))/energy.clamp_min(1e-30)
            cc.append(c.cpu().numpy());gg.append((g.flatten(-2)@proj).cpu().numpy())
            gm.append(g.mean(1).cpu().numpy())
            ff.append(torch.stack([row,entropy,g.square().mean((-2,-1)).sqrt(),energy],-1).cpu().numpy())
        centroids.append(np.stack(cc,axis=1));operators.append(np.stack(gg,axis=1))
        mean_operators.append(np.stack(gm,axis=1));fields.append(np.stack(ff,axis=1))
    fields=np.concatenate(fields)
    with np.load(root/'observations.npz') as old:
        expected=old['heads'][-1]
        max_replay=float(np.max(np.abs(fields-expected)))
        normalized_replay=float(np.max(np.abs(fields-expected)/(1+np.abs(expected))))
        logit_replay=float(np.max(np.abs(logits-old[f'logits_{job["steps"]}'])))
        if normalized_replay>1e-10 or logit_replay>1e-6:raise ValueError('Independent native observation replay differs')
    arrays=dict(common_centroids=np.concatenate(centroids),operator_projections=np.concatenate(operators),
                mean_operators=np.concatenate(mean_operators),
                operator_basis=projection,head_fields=fields,logits=logits,nll=np.array(losses))
    if not all(np.isfinite(x).all() for x in arrays.values()):raise ValueError('Nonfinite collective readout')
    np.savez_compressed(out/'collectives.npz',**arrays)
    write_json(out/'manifest.json',dict(status='complete',schema=p['schema'],role='observation',job=job,
        protocol_sha256=sha256(destination/'protocol.json'),producer_sha256=p['producer_sha256'],
        checkpoint_sha256=native['artifacts']['final-state.pt'],training_updates=0,
        observations_sha256=sha256(out/'collectives.npz'),maximum_head_replay_error=max_replay,
        maximum_normalized_head_replay_error=normalized_replay,maximum_logit_replay_error=logit_replay,
        runtime_seconds=time.monotonic()-started,maximum_cuda_memory_bytes=torch.cuda.max_memory_allocated(device)))
    print(run_id,'observed',flush=True)


def run(destination):
    p=json.loads((destination/'protocol.json').read_text())
    def queue(device,jobs):
        for job in jobs:
            out=destination/'runs'/job['run_id']
            if out.exists():
                m=json.loads((out/'manifest.json').read_text())
                if m['status']!='complete' or m['protocol_sha256']!=sha256(destination/'protocol.json') or sha256(out/'collectives.npz')!=m['observations_sha256']:
                    raise ValueError('Changed completed observation')
                continue
            logs=destination/'logs';logs.mkdir(exist_ok=True)
            with (logs/(job['run_id']+'.log')).open('x') as log:
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'worker','--destination',str(destination),
                    '--run-id',job['run_id'],'--device',device],stdout=log,stderr=subprocess.STDOUT,check=True,
                    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),OPENBLAS_NUM_THREADS='2'))
            print('COMPLETE '+job['run_id'],flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(queue,f'cuda:{i}',p['jobs'][i::2]) for i in range(2)]
        for f in futures:f.result()


def main():
    p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest='action',required=True)
    for action in ['prepare','run','worker']:
        q=subs.add_parser(action);q.add_argument('--destination',type=Path,required=True)
        if action=='prepare':
            q.add_argument('--study',type=Path,required=True)
            q.add_argument('--qualification-parent')
        if action=='worker':q.add_argument('--run-id',required=True);q.add_argument('--device',required=True)
    a=p.parse_args();destination=a.destination.resolve()
    if not destination.is_relative_to(ROOT) or destination==ROOT:raise ValueError('Use an authorized observation destination')
    if a.action=='prepare':prepare(a.study.resolve(),destination,a.qualification_parent)
    elif a.action=='run':run(destination)
    else:measure(destination,a.run_id,a.device)

if __name__=='__main__':main()
