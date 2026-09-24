#!/usr/bin/env python3
"""Frozen native single-pass family with a matched physical clock and finite resource."""
from companion_paths import child_pythonpath, dispatch_worker, validate_worker_cli
from companion_paths import legacy_path
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import threading
import numpy as np
import torch
from model_rg.training import TrainingModel
from model_rg.criticality import fix_shared_generator, generator_parameter, parameter_digest
from model_rg.variance_family import normalize_variance_initialization
from model_rg.inference_interventions import fixed_operators, operator_cache
from model_rg.provenance import sha256, write_json
from numerical_validation import load_json_strict
from context_reservations import reserve, validate_receipt
from matched_clock_contract import CONTRACT, validate_design, validate_qualification

REPO=Path(__file__).resolve().parents[1]
ROOT=Path(legacy_path('/pldr-data/model'))
CORPUS=ROOT/'data/refinedweb-onepass-524288'
PROBES=ROOT/'controlled-study-20260905/data/short'
NATIVE=ROOT/'assets/PLDR-LLM-v51-SOC-110M-1'
SOURCES=['scripts/run_matched_clock.py','scripts/matched_clock_contract.py','scripts/context_reservations.py','scripts/numerical_validation.py',
 'src/model_rg/training.py','src/model_rg/native.py','src/model_rg/criticality.py','src/model_rg/controlled.py',
 'src/model_rg/variance_family.py','src/model_rg/inference_interventions.py','src/model_rg/provenance.py']
HEADS=[4,8,14,24]; CONTROLS=[0.,1.,1.5,2.]; SEEDS=[9163401,9163402,9163403]

def read(path):return load_json_strict(Path(path).read_text())
def jobs():return [dict(run_id=f'h{n}-g{g:g}-s{s}',heads=n,control=g,seed=s,steps=128*n) for n in HEADS for g in CONTROLS for s in SEEDS]
def optimizer(model,n,g):
    named=list(model.model.named_parameters())
    groups=[dict(params=[p for name,p in named if generator_parameter(name)],lr=3e-4*g*14/n),
            dict(params=[p for name,p in named if not generator_parameter(name)],lr=6e-4/n)]
    return torch.optim.AdamW(groups,betas=(.9**(14/n),.95**(14/n)),eps=1e-8,weight_decay=.01,foreach=False)

def prepare(study,reproduce=None):
    if study.exists():raise FileExistsError(study)
    receipt=reserve(study,80,reproduce=reproduce)
    rows=np.array(receipt['context_rows'])
    # One block per distinct master document. A common random reservoir order couples widths.
    order=np.random.default_rng(9163400).permutation(524288)
    records=read(CORPUS/'records.json')
    if len({r['content_sha256'] for r in records})!=524288:raise ValueError('Master corpus repeats documents')
    tokens=np.load(CORPUS/'tokens.npy',mmap_mode='r')
    if tokens.shape!=(524288,513) or tokens.dtype!=np.int32:raise ValueError('Wrong master token geometry')
    panels=np.array(np.load(PROBES/'tokens.npy',mmap_mode='r')[rows,:65])
    study.mkdir(parents=True)
    write_json(study/'reservation.json',receipt)
    np.savez_compressed(study/'selection.npz',order=order,probes=panels,rows=rows)
    paths=[CORPUS/'tokens.npy',CORPUS/'records.json',CORPUS/'manifest.json',PROBES/'tokens.npy',PROBES/'records.json',
           NATIVE/'modeling_pldrllm.py',NATIVE/'configuration_pldrllm.py']
    p=dict(schema='matched-clock-onepass-v1',admission_contract=CONTRACT,status='frozen_before_acquisition',jobs=jobs(),
        document_hashes=receipt['document_hashes'],context_rows=receipt['context_rows'],reservation_sha256=sha256(study/'reservation.json'),
        selection_sha256=sha256(study/'selection.npz'),source_sha256={n:sha256(REPO/n) for n in SOURCES},
        input_sha256={str(p):sha256(p) for p in paths},shared_seed=9163499,
        family=dict(heads=HEADS,controls=CONTROLS,seeds=SEEDS,depth=5,head_dimension=64,
          horizon='128*N',physical_step='1/(128*N)',resource_documents='16384*N',batch=32,
          consumed_fraction=.25,generator_lr='0.0003*g*14/N',other_lr='0.0006/N',
          betas='(0.9**(14/N),0.95**(14/N))',epsilon=1e-8,weight_decay=.01,clip_norm=1.,
          source='Nested prefix of a frozen permutation of the master; first 65 tokens per document, token 64 is the sole training target; distinct documents within every trajectory.'),
        ages=[0,.25,.5,1.],calibration_contexts=16,contexts=64,prefix_length=64,vocabulary=32000,
        cache_conditions=[[8,0.],[8,1.5],[24,0.],[24,1.5]],
        targets=dict(centered_rms=.25,mean_kl=.03),
        hypotheses=['Matched bias-corrected adaptive forces obey their finite-history envelope.',
          'A finite consuming source requires age-conditioned row, common-operator and predictive laws.',
          'State-specific operator calibration is tested at all four ages; initial-cache transport is retained as a separate policy.'],
        interpretation='Three new wide initialization identities conditional on fixed corpus/order and shared generator; controls, ages and contexts are paired, not extra independent training replicas. No critical exponent or stationarity assumption.',
        worker_hour_cap=6,memory_ceiling_bytes=22*1024**3,disk_budget_bytes=200*1024**3,
        scientific_updates=sum(j['steps'] for j in jobs()),qualification_updates=3,
        output_policy='Keep all finite outcomes and failed runs; no replacement seeds; final full states retained outside the manuscript bundle.')
    write_json(study/'protocol.json',p)
    for name in SOURCES:
        dst=study/'executed-source'/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((REPO/name).read_bytes())
    print(dict(study=str(study),jobs=len(jobs()),updates=p['scientific_updates']),flush=True)

def admit(study,scientific=False):
    p=read(study/'protocol.json')
    if p['schema']!='matched-clock-onepass-v1' or p['jobs']!=jobs():raise ValueError('Wrong frozen design')
    if set(p['source_sha256'])!=set(SOURCES):raise ValueError('Incomplete sources')
    for n,h in p['source_sha256'].items():
        if sha256(REPO/n)!=h or sha256(study/'executed-source'/n)!=h:raise ValueError('Changed source or snapshot '+n)
    expected={str(x) for x in [CORPUS/'tokens.npy',CORPUS/'records.json',CORPUS/'manifest.json',PROBES/'tokens.npy',PROBES/'records.json',NATIVE/'modeling_pldrllm.py',NATIVE/'configuration_pldrllm.py']}
    if set(p['input_sha256'])!=expected:raise ValueError('Incomplete external input roles')
    for n,h in p['input_sha256'].items():
        if sha256(n)!=h:raise ValueError('Changed input '+n)
    if sha256(study/'selection.npz')!=p['selection_sha256']:raise ValueError('Changed selection')
    validate_receipt(study,p)
    validate_design(study,p,CORPUS,PROBES)
    with np.load(study/'selection.npz') as a:
        order=a['order']; blocks=a['probes']
        if order.shape!=(524288,) or not np.array_equal(np.sort(order),np.arange(524288)):raise ValueError('Invalid source permutation')
        if blocks.shape!=(80,65) or blocks.dtype.kind not in 'iu' or blocks.min()<0 or blocks.max()>=32000:raise ValueError('Invalid context geometry')
    if scientific:
        q=read(study/'qualification/manifest.json')
        validate_qualification(q,sha256(study/'protocol.json'))
    return p

def configure(device):
    torch.set_num_threads(2);torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.reset_peak_memory_stats(device)

def initialize(n,g,seed,device,p):
    model=TrainingModel(NATIVE,n,seed,device);normalize_variance_initialization(model)
    fix_shared_generator(model,p['shared_seed'])
    return model,optimizer(model,n,g)

def update(model,opt,tokens,rows,device):
    batch=torch.as_tensor(np.array(tokens[rows,:65]),device=device,dtype=torch.long)
    model.model.train();opt.zero_grad(set_to_none=True)
    loss=torch.nn.functional.cross_entropy(model.forward(batch[:,:64]).logits[:,-1],batch[:,64])
    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.model.parameters(),1.,error_if_nonfinite=True)
    opt.step()
    return loss.item(),float(norm)

def to_cpu(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:to_cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [to_cpu(v) for v in value]
    return copy.deepcopy(value)

def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def qualify(study,device):
    p=admit(study);out=study/'qualification';out.mkdir(exist_ok=False)
    configure(device);beg=time.monotonic();m=dict(status='started',protocol_sha256=sha256(study/'protocol.json'),optimizer_updates=0)
    write_json(out/'manifest.json',m)
    try:
        model,opt=initialize(24,2.,9163498,device,p)
        tokens=np.load(CORPUS/'tokens.npy',mmap_mode='r')
        with np.load(study/'selection.npz') as a:rows=a['order'][-64:];probes=a['probes']
        update(model,opt,tokens,rows[:32],device);m['optimizer_updates']+=1
        state=to_cpu(model.model.state_dict());osnap=to_cpu(opt.state_dict());rng=torch.get_rng_state();crng=torch.cuda.get_rng_state(device)
        loss1=update(model,opt,tokens,rows[32:],device);m['optimizer_updates']+=1
        endpoint=to_cpu(model.model.state_dict());ostate=to_cpu(opt.state_dict())
        model.model.load_state_dict(state);opt.load_state_dict(osnap);torch.set_rng_state(rng);torch.cuda.set_rng_state(crng,device)
        loss2=update(model,opt,tokens,rows[32:],device);m['optimizer_updates']+=1
        if loss1!=loss2 or not equal(endpoint,model.model.state_dict()) or not equal(ostate,opt.state_dict()):raise ValueError('Optimizer replay differs')
        del state,osnap,endpoint,ostate
        model.model.eval()
        with torch.no_grad():
            x=torch.as_tensor(probes[:8,:64],device=device,dtype=torch.long)
            y=model.forward(x,capture=True);cache=operator_cache(y);z=y.logits[:,-1].clone();del y
            with fixed_operators(model,cache):other=model.forward(x).logits[:,-1]
            if not torch.equal(z,other):raise ValueError('Native operator replay differs')
            if not torch.equal(z,model.forward(x).logits[:,-1]):raise ValueError('Restoration differs')
        if torch.cuda.max_memory_allocated(device)>=p['memory_ceiling_bytes']:raise ValueError('Qualification exceeds memory')
        m.update(status='complete',optimizer_replay_exact=True,operator_replay_exact=True,restoration_exact=True,forwards=6)
    except Exception as e:m.update(status='failed',error=repr(e));raise
    finally:
        m.update(elapsed_seconds=time.monotonic()-beg,peak_allocated_bytes=torch.cuda.max_memory_allocated(device));write_json(out/'manifest.json',m)
    print(m,flush=True)

@torch.no_grad()
def observe(model,opt,probes,step,job,out,initial_cache,counts):
    model.model.eval();device=model.device
    cache_enabled=[job['heads'],job['control']] in [[8,0.],[8,1.5],[24,0.],[24,1.5]]
    cache=None
    def forward(x,capture=False):counts['observation']+=1;return model.forward(x,capture=capture)
    if cache_enabled:
        pieces=[]
        for i in [0,8]:
            y=forward(probes[i:i+8,:64],True);pieces.append(operator_cache(y).double().sum(2,keepdim=True));del y
        cache=((pieces[0]+pieces[1])/16).float();del pieces
        if step==0:initial_cache=cache.clone()
    native=[];heads=[];operators=[]
    for i in range(16,80,8):
        y=forward(probes[i:i+8,:64],True);native.append(y.logits[:,-1].cpu().numpy());layer=[]
        for a,_,_,_,_,g,w in y.pldr_attentions:
            a=a.double();g=g.double();power=a.square().mean((-2,-1))
            row=(a-a.mean(-2,keepdim=True)).square().mean((-2,-1))/power.clamp_min(1e-30)
            weights=w[:,:,-1].double();entropy=-(weights*weights.clamp_min(1e-300).log()).sum(-1)/math.log(64)
            layer.append(torch.stack([row,entropy,g.square().mean((-2,-1)).sqrt(),power],-1))
        heads.append(torch.stack(layer,1).cpu().numpy())
        if cache_enabled:operators.append(operator_cache(y)[:,2].cpu())
        del y
    data=dict(native=np.concatenate(native),heads=np.concatenate(heads),targets=probes[16:,64].cpu().numpy())
    if cache_enabled:
        for policy,c in [('state',cache),('initial',initial_cache)]:
            z=[]
            with fixed_operators(model,c):
                for i in range(16,80,8):z.append(forward(probes[i:i+8,:64]).logits[:,-1].cpu().numpy())
            data[policy]=np.concatenate(z)
        g=torch.cat(operators,1).double();mean=g.mean(1)
        data['operator_mean']=mean.numpy();data['operator_scatter']=((g-mean[:,None])**2).mean((1,2,3,4)).numpy()
        for policy,c in [('state',cache),('initial',initial_cache)]:
            cg=c[:,2,0].cpu().double();data[policy+'_cache']=cg.numpy()
            data[policy+'_risk']=((g-cg[:,None])**2).mean((1,2,3,4)).numpy()
            data[policy+'_displacement']=((mean-cg)**2).mean((1,2,3)).numpy()
    maximum=0.
    if step:
        b1,b2=opt.param_groups[0]['betas']
        for state in opt.state.values():
            m=state['exp_avg']/(1-b1**step);v=state['exp_avg_sq']/(1-b2**step)
            maximum=max(maximum,(m.abs()/(v.sqrt()+1e-8)).max().item())
        bound=(1-b1)/math.sqrt((1-b2)*(1-b1*b1/b2))
        if maximum>bound*(1+2e-5):raise ValueError('Adaptive-force envelope fails')
    data['maximum_adaptive_force']=np.array(maximum)
    if any(not np.isfinite(v).all() for v in data.values()):raise ValueError('Nonfinite observation')
    np.savez_compressed(out/f'age-{step}.npz',**data)
    return initial_cache

def worker(study,index,device):
    if type(index) is not int or not 0<=index<len(jobs()) or device not in ['cuda:0','cuda:1']:raise ValueError('Invalid worker')
    p=admit(study,True);job=p['jobs'][index];out=study/'runs'/job['run_id'];out.mkdir(parents=True,exist_ok=False)
    beg=time.monotonic();counts=dict(training=0,observation=0);m=dict(status='started',job=job,protocol_sha256=sha256(study/'protocol.json'),updates=0,device=device)
    write_json(out/'manifest.json',m);configure(device)
    try:
        model,opt=initialize(job['heads'],job['control'],job['seed'],device,p)
        m['initial_parameter_sha256']=parameter_digest(model)
        with np.load(study/'selection.npz') as a:
            rows=a['order'][:32*job['steps']].reshape(-1,32);probes=torch.as_tensor(a['probes'],device=device,dtype=torch.long)
        tokens=np.load(CORPUS/'tokens.npy',mmap_mode='r');ages=[0,job['steps']//4,job['steps']//2,job['steps']]
        initial_cache=observe(model,opt,probes,0,job,out,None,counts);losses=[];norms=[]
        for k,batch_rows in enumerate(rows):
            loss,norm=update(model,opt,tokens,batch_rows,device);counts['training']+=1;m['updates']=k+1
            losses.append(loss);norms.append(norm)
            if k+1 in ages:initial_cache=observe(model,opt,probes,k+1,job,out,initial_cache,counts)
            if torch.cuda.max_memory_allocated(device)>=p['memory_ceiling_bytes']:raise RuntimeError('Memory cap')
            if time.monotonic()-beg>1800:raise RuntimeError('Per-worker time cap')
            if (k+1)%256==0:print(job['run_id'],k+1,flush=True)
        np.savez_compressed(out/'training.npz',loss=losses,gradient_norm=norms,document_rows=rows)
        torch.save(dict(model=model.model.state_dict(),optimizer=opt.state_dict(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device),consumed_documents=32*job['steps'],remaining_order=rows.size),out/'final-state.pt')
        m.update(status='complete',artifacts={f.name:sha256(f) for f in out.iterdir() if f.name!='manifest.json'})
    except Exception as e:m.update(status='failed',error=repr(e));raise
    finally:
        m.update(calls=counts,elapsed_seconds=time.monotonic()-beg,peak_allocated_bytes=torch.cuda.max_memory_allocated(device));write_json(out/'manifest.json',m)
    print(dict(run=job['run_id'],seconds=m['elapsed_seconds'],updates=m['updates']),flush=True)

def run(study):
    p=admit(study,True);logs=study/'logs';logs.mkdir(exist_ok=True)
    pending=[]
    for index,job in enumerate(p['jobs']):
        out=study/'runs'/job['run_id']
        if out.exists():
            m=read(out/'manifest.json')
            if m['status']!='complete' or m['protocol_sha256']!=sha256(study/'protocol.json'):raise ValueError('Failed or foreign retained run')
            for n,h in m['artifacts'].items():
                if sha256(out/n)!=h:raise ValueError('Changed retained artifact')
        else:pending.append(index)
    pending.sort(key=lambda i:p['jobs'][i]['heads'],reverse=True)
    lock=threading.Lock()
    def queue(device):
        while True:
            with lock:
                if not pending:return
                index=pending.pop(0);job=p['jobs'][index]
                elapsed=sum(read(f).get('elapsed_seconds',0) for f in (study/'runs').glob('*/manifest.json'))
                if elapsed+3600>p['worker_hour_cap']*3600:raise RuntimeError('Worker-hour cap')
                if sum(f.stat().st_size for f in study.rglob('*') if f.is_file())>p['disk_budget_bytes']:raise RuntimeError('Disk cap')
                logfile=logs/(job['run_id']+'.log')
                if logfile.exists():raise ValueError('Uncompleted claimed job')
            with logfile.open('x') as log:
                dispatch_worker([sys.executable,'-B',__file__,'worker','--study',str(study),'--index',str(index),'--device',device],
                    cwd=REPO,env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1900)
            print('COMPLETE '+job['run_id'],flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(queue,f'cuda:{d}') for d in range(2)]
        for f in futures:f.result()


if __name__ == '__main__':
    validate_worker_cli(__file__)

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('action',choices=['prepare','qualify','run','worker']);a.add_argument('--study',type=Path,required=True);a.add_argument('--index',type=int);a.add_argument('--reproduce-panel',type=Path);a.add_argument('--device',default='cuda:0');args=a.parse_args();study=args.study.resolve()
    if study==ROOT or not study.is_relative_to(ROOT):raise ValueError('Authorized experiment destination required')
    if args.action=='prepare':prepare(study,args.reproduce_panel)
    elif args.action=='qualify':qualify(study,args.device)
    elif args.action=='run':run(study)
    else:worker(study,args.index,args.device)
