#!/usr/bin/env python3
"""Frozen single-step row-response checks on the completed matched-clock endpoints."""
from companion_paths import child_pythonpath, dispatch_worker, validate_worker_cli
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from model_rg.provenance import sha256,write_json
from run_matched_clock import ROOT,REPO,CORPUS,NATIVE,read,initialize,configure,update,SOURCES as BASE

PARENT=ROOT/'matched-clock-onepass-20260916'
SOURCES=['scripts/run_collective_row_clock.py']+BASE
JOBS=[dict(heads=n,control=g,seed=9163401,run_id=f'h{n}-g{g:g}-s9163401',steps=128*n) for n in [4,8,14,24] for g in [1.5,2.]]


def prepare(study):
    if study.exists():raise FileExistsError(study)
    p=read(PARENT/'protocol.json');files=[PARENT/'protocol.json',PARENT/'selection.npz',PARENT/'reservation.json']+[Path(n) for n in p['input_sha256']]
    for j in JOBS:
        folder=PARENT/'runs'/j['run_id'];m=read(folder/'manifest.json')
        if m['status']!='complete' or m['job']!=j:raise ValueError('Incomplete parent')
        checkpoint=folder/'final-state.pt'
        if sha256(checkpoint)!=m['artifacts']['final-state.pt']:raise ValueError('Changed parent state')
        files += [folder/'manifest.json',checkpoint,folder/f"age-{j['steps']}.npz"]
    study.mkdir();protocol=dict(schema='collective-row-clock-v1',status='frozen_before_acquisition',jobs=JOBS,
        source_sha256={n:sha256(REPO/n) for n in SOURCES},input_sha256={str(f):sha256(f) for f in files},
        panel_reuse=dict(parent=str(PARENT),reservation_sha256=sha256(PARENT/'reservation.json'),assessment_indices=list(range(8)),independent_panel=False),
        source_rule='Exactly the next 32 previously unused documents of each parent trajectory; no repeated supervised block.',
        parameter_fractions=[-.5,0.,.5,1.],observable='Mean normalized squared row contrast over eight fixed contexts, five decoders and all heads.',
        prediction='Autograd at the incoming state dotted with the actual next-update parameter displacement; compare its central directional secant and finite-step residuals.',
        scope='Eight local native transport checks, one retained initialization per width/control; no new population fit or collective closure claim.',
        peak_bytes_limit=22*1024**3,seconds_per_job_limit=240,scientific_updates=8,native_forwards=40)
    write_json(study/'protocol.json',protocol)
    for n in SOURCES:
        dst=study/'executed-source'/n;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((REPO/n).read_bytes())
    print(dict(status='prepared',jobs=8),flush=True)


def admit(study,index=None):
    p=read(study/'protocol.json')
    if p['schema']!='collective-row-clock-v1' or p['jobs']!=JOBS or set(p['source_sha256'])!=set(SOURCES):raise ValueError('Foreign design')
    if p['parameter_fractions']!=[-.5,0.,.5,1.] or p['scientific_updates']!=8 or p['native_forwards']!=40 or p['panel_reuse']['assessment_indices']!=list(range(8)):raise ValueError('Changed transport geometry')
    for n,h in p['source_sha256'].items():
        if sha256(REPO/n)!=h or sha256(study/'executed-source'/n)!=h:raise ValueError('Changed executing source')
    parent=read(PARENT/'protocol.json')
    expected={str(PARENT/n) for n in ['protocol.json','selection.npz','reservation.json']}|set(parent['input_sha256'])
    for j in JOBS:expected|={str(PARENT/'runs'/j['run_id']/n) for n in ['manifest.json','final-state.pt',f"age-{j['steps']}.npz"]}
    if set(p['input_sha256'])!=expected:raise ValueError('Missing external role')
    for n,h in p['input_sha256'].items():
        if Path(n).name!='final-state.pt' and sha256(n)!=h:raise ValueError('Changed external input')
    if index is not None:
        if type(index) is not int or not 0<=index<len(JOBS):raise ValueError('Invalid worker')
        path=PARENT/'runs'/JOBS[index]['run_id']/'final-state.pt'
        if sha256(path)!=p['input_sha256'][str(path)]:raise ValueError('Changed selected payload')
    return p


def fields(model,ids):
    output=model.forward(ids,capture=True);matrices=[];rows=[];powers=[]
    for layer in output.pldr_attentions:
        a=layer[0].double();power=a.square().mean((-2,-1));row=(a-a.mean(-2,keepdim=True)).square().mean((-2,-1))/power.clamp_min(1e-30)
        matrices.append(layer[0].detach().cpu().numpy());rows.append(row);powers.append(power)
    return torch.stack(rows,1),np.stack(matrices,1),min(float(x.min()) for x in powers)


def worker(study,index,device):
    if device not in ['cuda:0','cuda:1']:raise ValueError('Invalid device')
    p=admit(study,index);j=JOBS[index];out=study/'runs'/j['run_id'];out.mkdir(parents=True,exist_ok=False)
    m=dict(status='started',job=j,protocol_sha256=sha256(study/'protocol.json'),native_forwards=0,scientific_updates=0)
    write_json(out/'manifest.json',m);configure(device);start=time.monotonic()
    try:
        parent=read(PARENT/'protocol.json');model,opt=initialize(j['heads'],j['control'],j['seed'],device,parent)
        path=PARENT/'runs'/j['run_id']/'final-state.pt'
        if sha256(path)!=p['input_sha256'][str(path)]:raise ValueError('Payload changed before loading')
        snapshot=torch.load(path,map_location='cpu',weights_only=False);before=snapshot['model'];model.model.load_state_dict(before);opt.load_state_dict(snapshot['optimizer'])
        if snapshot['consumed_documents']!=32*j['steps'] or snapshot['remaining_order']!=32*j['steps']:raise ValueError('Wrong source cursor')
        if any(int(s['step'])!=j['steps'] for s in opt.state.values()):raise ValueError('Wrong optimizer clock')
        torch.set_rng_state(snapshot['cpu_rng']);torch.cuda.set_rng_state(snapshot['cuda_rng'],device);del snapshot
        with np.load(PARENT/'selection.npz') as z:
            ids=torch.as_tensor(z['probes'][16:24,:64],device=device,dtype=torch.long);next_rows=z['order'][32*j['steps']:32*(j['steps']+1)].copy()
            if len(next_rows)!=32 or np.intersect1d(next_rows,z['order'][:32*j['steps']]).size:raise ValueError('Source repetition')
        model.model.eval();row,A,power=fields(model,ids);m['native_forwards']+=1
        named=list(model.model.named_parameters());grads=torch.autograd.grad(row.mean(),[x for _,x in named],allow_unused=True)
        gradients={n:g.detach().cpu().double() for (n,_),g in zip(named,grads) if g is not None}
        observed=[row.detach().cpu().numpy()];matrices=[A];powers=[power];fractions=[0.];predictions=[float(row.mean())];dots=[0.];norms=[0.]
        with np.load(PARENT/'runs'/j['run_id']/f"age-{j['steps']}.npz") as z:replay=float(abs(observed[0]-z['heads'][:8,:,:,0]).max())
        if replay>1e-9:raise ValueError('Incoming row replay differs')
        del row,grads;model.head_outputs={};gc.collect()
        loss,gnorm=update(model,opt,np.load(CORPUS/'tokens.npy',mmap_mode='r'),next_rows,device);m['native_forwards']+=1;m['scientific_updates']+=1
        after={n:x.detach().cpu().clone() for n,x in model.model.state_dict().items()}
        for fraction in [-.5,.5,1.]:
            state=after if fraction==1 else {n:(x.double()+fraction*(after[n].double()-x.double())).to(x.dtype) if x.dtype.is_floating_point else x for n,x in before.items()}
            dot=sum(float((g*(state[n].double()-before[n].double())).sum()) for n,g in gradients.items())
            norm=sum(float((state[n].double()-before[n].double()).square().sum()) for n,_ in named)**.5
            model.model.load_state_dict(state);model.model.eval()
            with torch.no_grad():r,A,power=fields(model,ids)
            m['native_forwards']+=1;observed.append(r.cpu().numpy());matrices.append(A);powers.append(power);fractions.append(fraction);dots.append(dot);norms.append(norm);predictions.append(float(observed[0].mean())+dot)
            del state,r,A
        if any(not torch.equal(x.cpu(),after[n]) for n,x in model.model.state_dict().items()):raise ValueError('Endpoint restoration differs')
        np.savez_compressed(out/'observations.npz',fractions=fractions,row=np.stack(observed),A=np.stack(matrices),minimum_power=powers,gradient_dot=dots,displacement_norm=norms,prediction=predictions,next_document_rows=next_rows)
        peak=torch.cuda.max_memory_allocated(device);elapsed=time.monotonic()-start
        if peak>=p['peak_bytes_limit'] or elapsed>=p['seconds_per_job_limit']:raise ValueError('Resource ceiling')
        m.update(status='complete',incoming_row_replay_error=replay,endpoint_restored=True,training_loss=loss,training_gradient_norm=gnorm,artifacts={'observations.npz':sha256(out/'observations.npz')})
    except Exception as e:m.update(status='failed',error=repr(e));raise
    finally:
        m.update(elapsed_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(device));write_json(out/'manifest.json',m)
    print(m,flush=True)


def run(study):
    p=admit(study);logs=study/'logs';logs.mkdir(exist_ok=True)
    def launch(index,device):
        j=JOBS[index];folder=study/'runs'/j['run_id']
        if folder.exists():
            m=read(folder/'manifest.json')
            if m['status']!='complete' or m['protocol_sha256']!=sha256(study/'protocol.json') or sha256(folder/'observations.npz')!=m['artifacts']['observations.npz']:raise ValueError('Failed retained run')
            return
        with (logs/(j['run_id']+'.log')).open('x') as log:dispatch_worker([sys.executable,'-B',__file__,'worker','--study',str(study),'--index',str(index),'--device',device],cwd=REPO,env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
        print('COMPLETE',j['run_id'],flush=True)
    # The first, largest declared point qualifies differentiable row replay and
    # resource use before the remaining seven frozen points. It is counted once.
    launch(7,'cuda:0')
    def queue(indices,device):
        for index in indices:launch(index,device)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(queue,list(range(7))[d::2],f'cuda:{d}') for d in range(2)]
        for f in futures:f.result()

if __name__ == '__main__':
    validate_worker_cli(__file__)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run','worker']);a.add_argument('--study',type=Path,required=True);a.add_argument('--index',type=int);a.add_argument('--device',default='cuda:0');v=a.parse_args();study=v.study.resolve()
    if study==ROOT or not study.is_relative_to(ROOT):raise ValueError('Authorized study required')
    if v.action=='prepare':prepare(study)
    elif v.action=='run':run(study)
    else:worker(study,v.index,v.device)
