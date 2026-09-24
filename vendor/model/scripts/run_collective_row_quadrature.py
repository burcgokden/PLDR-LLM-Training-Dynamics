#!/usr/bin/env python3
"""Finite parameter-path integration for eight saved one-step row transports."""
from companion_paths import child_pythonpath, dispatch_worker, validate_worker_cli
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from model_rg.provenance import sha256,write_json
from run_collective_row_clock import ROOT,REPO,CORPUS,PARENT,JOBS,read,initialize,configure,update,fields,admit as admit_parent,SOURCES as BASE
ROW=ROOT/'collective-row-clock-20260916'
SOURCES=['scripts/run_collective_row_quadrature.py']+BASE
INTERVALS=32


def prepare(study):
    if study.exists():raise FileExistsError(study)
    parent=admit_parent(ROW);inputs=dict(parent['input_sha256']);inputs[str(ROW/'protocol.json')]=sha256(ROW/'protocol.json')
    for j in JOBS:
        folder=ROW/'runs'/j['run_id'];m=read(folder/'manifest.json')
        if m['status']!='complete':raise ValueError('Incomplete diagnostic parent')
        for name in ['manifest.json','observations.npz']:inputs[str(folder/name)]=sha256(folder/name)
    study.mkdir();p=dict(schema='collective-row-quadrature-v1',status='frozen_before_acquisition',jobs=JOBS,
        intervals=INTERVALS,fractions=np.linspace(0,1,INTERVALS+1).tolist(),row_absolute_target=1e-5,
        source_sha256={n:sha256(REPO/n) for n in SOURCES},input_sha256=inputs,
        parent_transport=str(ROW),panel_reuse='The same registered eight paired contexts and incoming states; no new independent contexts or initializations.',
        source_rule='Replay each recorded next-unused-block optimizer update, without adding a scientific trajectory.',
        integrand='Autograd row derivative at each interpolated parameter state dotted with the complete recorded update displacement.',
        reconstruction='Nested composite trapezoids and Richardson/Simpson comparisons against the directly measured endpoint difference.',
        peak_bytes_limit=22*1024**3,seconds_per_job_limit=600,optimizer_replay_updates=8,native_forwards=8*(INTERVALS+2))
    write_json(study/'protocol.json',p)
    for n in SOURCES:
        dst=study/'executed-source'/n;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((REPO/n).read_bytes())
    print(dict(status='prepared',jobs=8,forwards=p['native_forwards']),flush=True)


def admit(study,index=None):
    p=read(study/'protocol.json');parent=admit_parent(ROW,index)
    if p['schema']!='collective-row-quadrature-v1' or p['jobs']!=JOBS or p['intervals']!=INTERVALS or p['fractions']!=np.linspace(0,1,INTERVALS+1).tolist():raise ValueError('Wrong frozen quadrature')
    if set(p['source_sha256'])!=set(SOURCES):raise ValueError('Missing sources')
    for n,h in p['source_sha256'].items():
        if sha256(REPO/n)!=h or sha256(study/'executed-source'/n)!=h:raise ValueError('Changed source')
    expected=dict(parent['input_sha256']);expected[str(ROW/'protocol.json')]=sha256(ROW/'protocol.json')
    for j in JOBS:
        for name in ['manifest.json','observations.npz']:
            path=ROW/'runs'/j['run_id']/name;expected[str(path)]=sha256(path)
    if p['input_sha256']!=expected:raise ValueError('Changed external identities')
    return p


def worker(study,index,device):
    if type(index) is not int or not 0<=index<8 or device not in ['cuda:0','cuda:1']:raise ValueError('Invalid worker')
    p=admit(study,index);j=JOBS[index];out=study/'runs'/j['run_id'];out.mkdir(parents=True,exist_ok=False);configure(device);start=time.monotonic()
    m=dict(status='started',job=j,protocol_sha256=sha256(study/'protocol.json'),native_forwards=0,optimizer_replay_updates=0);write_json(out/'manifest.json',m)
    try:
        model,opt=initialize(j['heads'],j['control'],j['seed'],device,read(PARENT/'protocol.json'))
        path=PARENT/'runs'/j['run_id']/'final-state.pt'
        if sha256(path)!=p['input_sha256'][str(path)]:raise ValueError('Changed payload before load')
        snapshot=torch.load(path,map_location='cpu',weights_only=False);before=snapshot['model'];model.model.load_state_dict(before);opt.load_state_dict(snapshot['optimizer'])
        if snapshot['consumed_documents']!=32*j['steps'] or any(int(s['step'])!=j['steps'] for s in opt.state.values()):raise ValueError('Wrong incoming clock')
        torch.set_rng_state(snapshot['cpu_rng']);torch.cuda.set_rng_state(snapshot['cuda_rng'],device);del snapshot
        with np.load(PARENT/'selection.npz') as z:
            ids=torch.as_tensor(z['probes'][16:24,:64],device=device,dtype=torch.long);next_rows=z['order'][32*j['steps']:32*(j['steps']+1)].copy()
        with np.load(ROW/'runs'/j['run_id']/'observations.npz') as z:
            reference_A=z['A'].copy();reference_rows=z['row'].copy()
            if not np.array_equal(next_rows,z['next_document_rows']):raise ValueError('Changed replay source')
        update(model,opt,np.load(CORPUS/'tokens.npy',mmap_mode='r'),next_rows,device);m['native_forwards']+=1;m['optimizer_replay_updates']+=1
        after={n:x.detach().cpu().clone() for n,x in model.model.state_dict().items()};named=list(model.model.named_parameters())
        delta={n:after[n].double()-before[n].double() for n,_ in named};rows=[];dots=[];powers=[];replay=0.
        for k,fraction in enumerate(p['fractions']):
            state=before if k==0 else after if k==INTERVALS else {n:(x.double()+fraction*(after[n].double()-x.double())).to(x.dtype) if x.dtype.is_floating_point else x for n,x in before.items()}
            model.model.load_state_dict(state);model.model.eval();row,A,power=fields(model,ids);m['native_forwards']+=1
            grad=torch.autograd.grad(row.mean(),[x for _,x in named],allow_unused=True)
            dot=sum(float((g.detach().cpu().double()*delta[n]).sum()) for (n,_),g in zip(named,grad) if g is not None)
            rows.append(row.detach().cpu().numpy());dots.append(dot);powers.append(power)
            if k in [0,INTERVALS//2,INTERVALS]:
                old={0:0,INTERVALS//2:2,INTERVALS:3}[k];replay=max(replay,float(abs(A.astype(float)-reference_A[old].astype(float)).max()),float(abs(rows[-1]-reference_rows[old]).max()))
            del row,A,grad,state;model.head_outputs={};gc.collect()
            if time.monotonic()-start>p['seconds_per_job_limit'] or torch.cuda.max_memory_allocated(device)>=p['peak_bytes_limit']:raise ValueError('Resource limit')
        if replay>1e-9:raise ValueError('Recorded transport replay differs')
        np.savez_compressed(out/'quadrature.npz',fractions=p['fractions'],row=np.stack(rows),derivative=dots,minimum_power=powers,next_document_rows=next_rows)
        m.update(status='complete',transport_replay_error=replay,artifacts={'quadrature.npz':sha256(out/'quadrature.npz')})
    except Exception as e:m.update(status='failed',error=repr(e));raise
    finally:m.update(elapsed_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(device));write_json(out/'manifest.json',m)
    print(m,flush=True)


def run(study):
    p=admit(study);logs=study/'logs';logs.mkdir(exist_ok=True)
    def launch(index,device):
        j=JOBS[index];folder=study/'runs'/j['run_id']
        if folder.exists():
            m=read(folder/'manifest.json')
            if m['status']!='complete' or m['protocol_sha256']!=sha256(study/'protocol.json') or sha256(folder/'quadrature.npz')!=m['artifacts']['quadrature.npz']:raise ValueError('Failed retained point')
            return
        with (logs/(j['run_id']+'.log')).open('x') as log:dispatch_worker([sys.executable,'-B',__file__,'worker','--study',str(study),'--index',str(index),'--device',device],cwd=REPO,env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=660)
        print('COMPLETE',j['run_id'],flush=True)
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
