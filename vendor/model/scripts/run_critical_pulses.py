#!/usr/bin/env python3
"""Paired native state pulses on unconsumed single-pass suffixes."""
from companion_paths import child_pythonpath
from companion_paths import configured_path
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from model_rg.training import TrainingModel
from model_rg.criticality import generator_parameter, optimizer_for, parameter_digest
from model_rg.provenance import sha256,write_json

REPO=Path(__file__).resolve().parents[1]
ROOT=Path(configured_path('data:model'))
NATIVE=ROOT/'assets/PLDR-LLM-v51-SOC-110M-1'
CORPUS=ROOT/'data/refinedweb-onepass-524288'
SOURCES=['scripts/run_critical_pulses.py','src/model_rg/training.py','src/model_rg/native.py',
         'src/model_rg/criticality.py','src/model_rg/provenance.py']


def prepare(study,design_path,out):
    if out.exists():raise FileExistsError(out)
    d=json.loads(design_path.read_text())
    if set(d)!={'parents','directions','steps','radii','role'} or not d['parents'] or len(set(d['parents']))!=len(d['parents']):
        raise ValueError('Expected unique parent IDs, directions, steps, radii and role')
    if d['role'] not in ['scientific','qualification']:raise ValueError('Unsupported preparation role')
    if set(d['directions'])-{'row','common'} or len(d['directions'])!=len(set(d['directions'])) or not d['directions']:
        raise ValueError('Unsupported pulse direction')
    if d['steps']<64 or d['steps']>2048 or d['steps']%16:
        raise ValueError('Pulse horizon must be a multiple of 16 in 64..2048')
    if len(d['radii'])!=2 or not 0<d['radii'][1]<d['radii'][0]<=.01 or not np.isclose(d['radii'][0],2*d['radii'][1]):
        raise ValueError('Use two small relative radii related by halving')
    p=json.loads((study/'protocol.json').read_text())
    jobs={j['run_id']:j for j in p['jobs']};cases=[]
    checked={str(study/'protocol.json'):sha256(study/'protocol.json'),str(study/'selection.npz'):sha256(study/'selection.npz')}
    selections=np.load(study/'selection.npz');suffixes={}
    for run_id in d['parents']:
        job=jobs[run_id];root=study/'runs'/run_id
        m=json.loads((root/'manifest.json').read_text())
        if m['status']!='complete' or not m['optimizer_saved'] or m['job']!=job:
            raise ValueError('A complete native Adam parent is required')
        for name in ['final-state.pt','observations.npz']:
            if sha256(root/name)!=m['artifacts'][name]:raise ValueError('Changed parent artifact')
            checked[str(root/name)]=m['artifacts'][name]
        checked[str(root/'manifest.json')]=sha256(root/'manifest.json')
        population=selections[f'population_{job["environment"]}']
        local=np.random.default_rng(9152502+job['environment']).permutation(len(population)*8)
        blocks=(8*population[local//8]+local%8).reshape(-1,32)
        end=job['steps']+d['steps']
        if end>len(blocks):raise ValueError('Exhausted one-pass resource')
        np.testing.assert_array_equal(blocks[:job['steps']],selections[f'blocks_{job["environment"]}'])
        suffixes[run_id]=blocks[job['steps']:end]
        for direction in d['directions']:
            cases.append(dict(case_id=run_id+'-'+direction,parent=run_id,job=job,direction=direction))
    out.mkdir(parents=True)
    np.savez_compressed(out/'selection.npz',probes=selections['probes'],**suffixes)
    for name,digest in p['input_sha256'].items():
        if sha256(name)!=digest:raise ValueError('Changed native input')
        checked[name]=digest
    write_json(out/'protocol.json',dict(schema='critical-native-pulses-v1',study=str(study),design=d,cases=cases,
        producer_sha256={name:sha256(REPO/name) for name in SOURCES},input_sha256=checked,
        selection_sha256=sha256(out/'selection.npz'),design_sha256=sha256(design_path),
        source='Every branch restores the parent parameters and both Adam moments, then consumes the identical previously unused source suffix.',
        direction='Unit L2 gradient of the indicated observable on the first eight fixed contexts, restricted to the complete generator parameter group.',
        amplitude='Signed relative radius times incoming generator parameter L2 norm; both radii halve the same fixed direction.',
        common_projection_seed=9152581,observation_cadence=16,
        scientific_arms=['baseline','plus_large','minus_large','plus_small','minus_small'] if d['role']=='scientific' else [],
        qualification_arms=['baseline_replay'] if d['role']=='scientific' else ['baseline','plus_large','minus_large','plus_small','minus_small','baseline_replay'],
        inference='Eight complete vocabulary logits at every observation; four scalar fields on sixteen fixed contexts.',
        interpretation='Finite full-state native response. A temporal eigenmode or critical exponent needs additional amplitude, stationarity and closure evidence.'))


def projection(device):
    v=np.random.default_rng(9152581).choice([-1.,1.],size=64)/8
    return torch.as_tensor(v,device=device,dtype=torch.float64)


def field(out,kind,v):
    values=[]
    for a,_,_,_,_,_,_ in out.pldr_attentions:
        a=a.double()
        if kind=='row':
            q=(a-a.mean(-2,keepdim=True)).square().mean((-2,-1))/a.square().mean((-2,-1)).clamp_min(1e-30)
        else:q=a.mean(-2)@v
        values.append(q.mean())
    return torch.stack(values).mean()


@torch.no_grad()
def observe(model,probes,v):
    model.model.eval();out=model.forward(probes[:16,:64],capture=True)
    z=out.logits[:,-1].double();fields=[]
    for a,_,_,_,_,g,w in out.pldr_attentions:
        a=a.double();g=g.double();p=w[:,:,-1].double()
        r=(a-a.mean(-2,keepdim=True)).square().mean((-2,-1))/a.square().mean((-2,-1)).clamp_min(1e-30)
        h=-(p*p.clamp_min(1e-300).log()).sum(-1)/np.log(64)
        c=a.mean(-2)@v;o=g.square().mean((-2,-1)).sqrt()
        fields.append(torch.stack([r,h,c,o],-1).mean(1))
    f=torch.stack(fields,1).mean(1)
    nll=torch.nn.functional.cross_entropy(z,probes[:16,64],reduction='none')
    if not torch.isfinite(f).all() or not torch.isfinite(z).all():raise FloatingPointError('Nonfinite pulse readout')
    return f.cpu().numpy(),z[:8].float().cpu().numpy(),nll.cpu().numpy()


def state_digest(model,optimizer,device):
    digest=hashlib.sha256()
    for name,param in model.model.named_parameters():
        digest.update(name.encode());digest.update(param.detach().cpu().contiguous().numpy().tobytes())
        for key,value in sorted(optimizer.state[param].items()):
            digest.update(key.encode())
            if torch.is_tensor(value):digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            else:digest.update(str(value).encode())
    digest.update(torch.get_rng_state().numpy().tobytes())
    digest.update(torch.cuda.get_rng_state(device).cpu().numpy().tobytes())
    return digest.hexdigest()


def execute(destination,case_id,device):
    p=json.loads((destination/'protocol.json').read_text())
    for name,digest in p['producer_sha256'].items():
        if sha256(REPO/name)!=digest:raise ValueError('Changed pulse source')
    cases=[c for c in p['cases'] if c['case_id']==case_id]
    if len(cases)!=1:raise ValueError('Expected one frozen case')
    case=cases[0];job=case['job'];root=Path(p['study'])/'runs'/case['parent']
    for path in [root/'manifest.json',root/'final-state.pt',CORPUS/'tokens.npy',NATIVE/'modeling_pldrllm.py',NATIVE/'configuration_pldrllm.py']:
        if sha256(path)!=p['input_sha256'][str(path)]:raise ValueError('Changed pulse input')
    if sha256(destination/'selection.npz')!=p['selection_sha256']:raise ValueError('Changed suffix')
    out=destination/'cases'/case_id;out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_device(device);torch.cuda.init();torch.cuda.reset_peak_memory_stats(device)
    parent=torch.load(root/'final-state.pt',map_location='cpu',weights_only=False)
    selected=np.load(destination/'selection.npz')
    probes=torch.as_tensor(selected['probes'],device=device,dtype=torch.long)
    blocks=selected[case['parent']]
    token=np.load(CORPUS/'tokens.npy',mmap_mode='r');v=projection(device)
    model=TrainingModel(NATIVE,job['heads'],job['seed'],device);model.model.load_state_dict(parent['model']);model.model.eval()
    named=[(n,x) for n,x in model.model.named_parameters() if generator_parameter(n)]
    response=field(model.forward(probes[:8,:64],capture=True),case['direction'],v)
    gradients=torch.autograd.grad(response,[x for _,x in named],allow_unused=True)
    gradients=[torch.zeros_like(x) if g is None else g for (_,x),g in zip(named,gradients)]
    grad_norm=float(torch.sqrt(sum(g.double().square().sum() for g in gradients)))
    weight_norm=float(torch.sqrt(sum(x.double().square().sum() for _,x in named)))
    if not np.isfinite(grad_norm) or grad_norm<=1e-20:
        write_json(out/'manifest.json',dict(status='unresolved_direction',case=case,scientific_updates=0,
            qualification_updates=0,reason='The selected observable has no numerically resolved generator gradient.',
            protocol_sha256=sha256(destination/'protocol.json'),gradient_norm=grad_norm))
        return
    directions={n:(g/grad_norm).detach().cpu() for (n,_),g in zip(named,gradients)}
    torch.save(directions,out/'direction.pt')
    del model,gradients,named,response
    torch.cuda.empty_cache()
    large,small=p['design']['radii']
    role=p['design']['role']
    arms=[('baseline',0.,role),('plus_large',large,role),('minus_large',-large,role),
          ('plus_small',small,role),('minus_small',-small,role),('baseline_replay',0.,'qualification')]
    records=[]
    for label,radius,role in arms:
        model=TrainingModel(NATIVE,job['heads'],job['seed'],device);model.model.load_state_dict(parent['model'])
        optimizer=optimizer_for(model,job['control']);optimizer.load_state_dict(copy.deepcopy(parent['optimizer']))
        torch.set_rng_state(parent['cpu_rng']);torch.cuda.set_rng_state(parent['cuda_rng'],device)
        with torch.no_grad():
            for name,param in model.model.named_parameters():
                if name in directions:param.add_(directions[name].to(device),alpha=radius*weight_norm)
        params=list(model.model.parameters());readouts=[];logits=[];losses=[];fields=[];times=[];training=[]
        def snapshot(step):
            f,z,nll=observe(model,probes,v);fields.append(f);logits.append(z);losses.append(nll);times.append(step)
        snapshot(0)
        for k,b in enumerate(blocks):
            model.model.train();optimizer.zero_grad(set_to_none=True)
            ids=token[b[:,None]//8,64*(b[:,None]%8)+np.arange(65)]
            batch=torch.as_tensor(ids,device=device,dtype=torch.long)
            z=model.forward(batch[:,:64]).logits[:,-1]
            loss=torch.nn.functional.cross_entropy(z,batch[:,64])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite native pulse loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(params,1.,foreach=False,error_if_nonfinite=True);optimizer.step()
            training.append(float(loss.detach()))
            if (k+1)%16==0:snapshot(k+1)
        fp=out/(label+'.npz')
        np.savez_compressed(fp,steps=times,fields=fields,logits=logits,nll=losses,training_loss=training,blocks=blocks)
        record=dict(arm=label,relative_radius=radius,role=role,updates=len(blocks),
                    full_final_state_sha256=state_digest(model,optimizer,device),artifact_sha256=sha256(fp))
        records.append(record);write_json(out/(label+'-manifest.json'),record)
        del model,optimizer,params;torch.cuda.empty_cache()
        print(case_id,label,'complete',flush=True)
    if records[0]['full_final_state_sha256']!=records[-1]['full_final_state_sha256']:
        raise ValueError('Restored native continuation failed full-state replay')
    with np.load(out/'baseline.npz') as a,np.load(out/'baseline_replay.npz') as b:
        if a.files!=b.files or not all(np.array_equal(a[n],b[n]) for n in a.files):raise ValueError('Baseline replay observations differ')
    write_json(out/'manifest.json',dict(schema=p['schema'],status='complete',case=case,
        protocol_sha256=sha256(destination/'protocol.json'),producer_sha256=p['producer_sha256'],
        parent_checkpoint_sha256=p['input_sha256'][str(root/'final-state.pt')],
        gradient_norm=grad_norm,generator_weight_norm=weight_norm,direction_sha256=sha256(out/'direction.pt'),
        arms=records,scientific_updates=sum(r['updates'] for r in records if r['role']=='scientific'),
        qualification_updates=sum(r['updates'] for r in records if r['role']=='qualification'),
        full_state_baseline_replay=True,runtime_seconds=time.monotonic()-started,
        maximum_cuda_memory_bytes=torch.cuda.max_memory_allocated(device)))


def run(destination):
    p=json.loads((destination/'protocol.json').read_text())
    def queue(device,cases):
        for c in cases:
            logs=destination/'logs';logs.mkdir(exist_ok=True)
            with (logs/(c['case_id']+'.log')).open('x') as log:
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'worker','--destination',str(destination),
                    '--case',c['case_id'],'--device',device],stdout=log,stderr=subprocess.STDOUT,check=True,
                    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),OPENBLAS_NUM_THREADS='2'))
            print('COMPLETE '+c['case_id'],flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(queue,f'cuda:{i}',p['cases'][i::2]) for i in range(2)]
        for f in futures:f.result()


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    for action in ['prepare','run','worker']:
        c=sub.add_parser(action);c.add_argument('--destination',type=Path,required=True)
        if action=='prepare':c.add_argument('--study',type=Path,required=True);c.add_argument('--design',type=Path,required=True)
        if action=='worker':c.add_argument('--case',required=True);c.add_argument('--device',required=True)
    a=p.parse_args();destination=a.destination.resolve()
    if not destination.is_relative_to(ROOT) or destination==ROOT:raise ValueError('Use an authorized pulse destination')
    if a.action=='prepare':prepare(a.study.resolve(),a.design.resolve(),destination)
    elif a.action=='run':run(destination)
    else:execute(destination,a.case,a.device)

if __name__=='__main__':main()
