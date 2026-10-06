#!/usr/bin/env python
"""Frozen, single-pass physical-corpus fine-tuning of native PLDR models."""
from companion_paths import configured_path
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import sentencepiece as spm
import torch

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from model_rg.training import TrainingModel
from model_rg.physical_native import context_tokens, spin_tokens, prefix_batch, selected_forward, generate
from model_rg.lattice import summarize
from model_rg.provenance import sha256, write_json

ROOT=Path(configured_path('data:model'))
from physical_design import (sources, stage_design, OPTIMIZER, SCHEDULE, runtime,
    assets, validate_spec, admit, schedule_factor)


def parameter_digest(model, generator):
    result=hashlib.sha256()
    for name,p in model.named_parameters():
        if is_generator(name)==generator:
            result.update(name.encode());result.update(p.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def prepare(study,stage,steps,lr):
    legal_steps, checkpoints, grid = stage_design(stage)
    if type(steps) is not int or steps != legal_steps or lr != .0003:
        raise ValueError('Noncanonical physical horizon or learning rate')
    study=Path(study).resolve();dest=study/stage
    if not study.is_relative_to(ROOT):raise ValueError('Unauthorized destination')
    if dest.exists():raise FileExistsError(dest)
    data=study/('development-data' if stage=='development' else 'training-data')
    data_spec=json.loads((data/'manifest.json').read_text())
    if data_spec['status']!='complete':raise ValueError('Data incomplete')
    cases=[dict(name=f'h{h}-s{seed}-{arm}',heads=h,seed=seed,arm=arm)
           for h,seed,arm in grid]
    for case in cases:
        p=ROOT/f"scheduled-training-feasible-20260908/runs/compact-reference1-h{case['heads']}-s{case['seed']}/final-training-state.pt"
        case.update(state=str(p),state_sha256=sha256(p))
    qualifications={}
    for h in [4,8]:
        p=study/(f'parity-h{h}.json' if stage=='qualification' else f'branch-h{h}.json');q=json.loads(p.read_text())
        if q['status']!='passed':raise ValueError('Native qualification missing')
        for n,signature in q['sources'].items():
            if sha256(REPO/n)!=signature:raise ValueError('Qualification source changed')
        qualifications[str(h)]={'path':str(p),'sha256':sha256(p)}
    rng=np.random.default_rng(1740000 if stage=='development' else 1750000)
    # Each update uses one uniformly selected physical cell and one uniformly
    # selected site. This is an unbiased per-site proper-prefix objective.
    cell_ids=rng.integers(len(data_spec['cells']),size=steps,dtype=np.int64)
    sites=np.array([rng.integers(data_spec['cells'][i]['L']**2) for i in cell_ids])
    indices=np.zeros((steps,32),dtype=np.int64)
    for cell in data_spec['cells']:
        where=np.flatnonzero(cell_ids==cell['id'])
        available=cell['samples_per_chain']*(6 if stage=='development' else cell['chains'])
        if len(where)*32>available:raise ValueError('Insufficient distinct configurations')
        indices[where]=rng.permutation(available)[:len(where)*32].reshape(-1,32)
    source=sources()
    protocol=dict(schema='physical-native-study-v2',stage=stage,status='frozen',steps=steps,
        created_at=datetime.now(timezone.utc).isoformat(),cases=cases,batch_size=32,learning_rate=lr,
        optimizer=OPTIMIZER,schedule=SCHEDULE,
        training_data=str(data),data_manifest_sha256=sha256(data/'manifest.json'),
        sources=source,qualifications=qualifications,
        native_assets=assets(),development_data=str(study/'development-data'),
        development_manifest_sha256=sha256(study/'development-data/manifest.json'),
        checkpoints=checkpoints,
        objective='Uniform cell, uniform site, conditional cross entropy on existing spin token IDs; only metadata and strict preceding spin tokens are provided.',
        exposure='Every (cell,chain,sample) identity is used at most once per path. Paired model seeds and controls share the frozen stream.',
        controls='Frozen-generator stops gradients to the residual metric network. Shuffled independently permutes sites inside each fresh configuration and preserves its histogram.',
        pretraining='Single pass over distinct RefinedWeb blocks. The physical corpus is a separate, explicitly conditioned adaptation distribution.',
        runtime=runtime())
    draws=dict(cell=cell_ids,site=sites,index=indices)
    validate_spec(protocol,draws)
    dest.mkdir();np.savez(dest/'draws.npz',**draws)
    protocol['draws_sha256']=sha256(dest/'draws.npz')
    for n in source:
        p=dest/'executed-source'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/n,p)
    write_json(dest/'protocol.json',protocol);print(dest,flush=True)


def load_model(case,device):
    source=ROOT/'assets/PLDR-LLM-v51-SOC-110M-1'
    state=torch.load(case['state'],map_location='cpu',weights_only=False)
    if state['step']!=40960:raise ValueError('Wrong incoming pretraining age')
    adapter=TrainingModel(source,case['heads'],case['seed'],device)
    adapter.model.load_state_dict(state['model']);del state;gc.collect()
    return adapter


def is_generator(name):
    return '.reslayerAs.' in name


@torch.no_grad()
def observe(adapter,processor,alphabet,data_spec,dest,timepoint,development):
    adapter.model.eval();results=[];arrays={}
    selected=[c for c in data_spec['cells'] if c['temperature_ratio']==1.]
    for c in selected:
        raw=np.memmap(c['path'],mode='r',dtype=np.uint8,shape=tuple(c['shape']))
        # The last two independent development chains are never used in training.
        x=np.asarray(raw[-2:,:16]).reshape(-1,c['L'],c['L'])
        meta=context_tokens(processor,c['q'],c['L'],c['temperature_ratio']);letters=alphabet[:c['q']]
        for fraction in [.25,.5,.875]:
            site=int(c['L']**2*fraction)
            inputs=prefix_batch(x,site,meta,letters,adapter.device)
            logits,out=selected_forward(adapter,inputs,letters,capture=True)
            target=torch.as_tensor(x.reshape(len(x),-1)[:,site].astype('int64'),device=adapter.device)
            losses=torch.nn.functional.cross_entropy(logits,target,reduction='none')
            key=f'q{c["q"]}-L{c["L"]}-j{site}'
            arrays[key+'-logits']=logits.cpu().numpy();arrays[key+'-targets']=target.cpu().numpy()
            arrays[key+'-hidden']=torch.stack([v[:,-1] for v in out.hidden_states],1).cpu().numpy()
            fields=[]
            for layer,att in enumerate(out.pldr_attentions):
                a=att[0].double();g=att[5].double();common=a.mean(-2)
                fields.append(torch.stack([common.square().mean(-1).sqrt(),
                    (a-common.unsqueeze(-2)).square().mean((-2,-1)),g.square().mean((-2,-1)).sqrt(),
                    adapter.head_outputs[layer].double()],-1))
            arrays[key+'-collectives']=torch.stack(fields,1).cpu().numpy()
            results.append(dict(q=c['q'],L=c['L'],site=site,nll=float(losses.mean()),
                                sample_count=len(x),cell=c['id']))
    adapter.capture=False;adapter.head_outputs={}
    np.savez_compressed(dest/f'observation-{timepoint:05d}.npz',**arrays)
    write_json(dest/f'observation-{timepoint:05d}.json',dict(time=timepoint,results=results,
        source='Independent final two development chains; used for development, not assessment.'))
    return results


def worker(study,name,device):
    study=Path(study)
    validated=admit(study,name)
    spec,case,ds,dev_ds=validated.spec,validated.case,validated.data,validated.development
    out=study/name;out.mkdir(exist_ok=False)
    torch.set_num_threads(spec['runtime']['threads']);torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    adapter=load_model(case,device)
    generator_names=[n for n,p in adapter.model.named_parameters() if is_generator(n)]
    if not generator_names:raise ValueError('Could not identify native metric generator')
    if case['arm']=='frozen-generator':
        for n,p in adapter.model.named_parameters():
            if n in generator_names:p.requires_grad_(False)
    parameters=[p for p in adapter.model.parameters() if p.requires_grad]
    opt=spec['optimizer']
    optimizer=torch.optim.AdamW(parameters,lr=spec['learning_rate'],betas=tuple(opt['betas']),
                              eps=opt['epsilon'],weight_decay=opt['weight_decay'])
    initial_generator=parameter_digest(adapter.model,True)
    initial_body=parameter_digest(adapter.model,False)
    shuffle_checks=0
    processor=spm.SentencePieceProcessor(model_file=str(ROOT/'assets/PLDR-LLM-v51-SOC-110M-1/tokenizer.model'))
    alphabet=spin_tokens(processor)
    raw=[np.memmap(c['path'],mode='r',dtype=np.uint8,shape=(c['chains']*c['samples_per_chain'],c['L'],c['L'])) for c in ds['cells']]
    metadata=[context_tokens(processor,c['q'],c['L'],c['temperature_ratio']) for c in ds['cells']]
    draws=validated.draws;rng=np.random.default_rng(1760000)
    traces=[];checkpoints=[];start=time.perf_counter()
    observations=observe(adapter,processor,alphabet,dev_ds,out,0,True)
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(spec['steps']):
        c=ds['cells'][int(draws['cell'][step])];site=int(draws['site'][step]);ids=draws['index'][step]
        configurations=np.asarray(raw[c['id']][ids]).copy()
        if case['arm']=='shuffled':
            flat=configurations.reshape(len(configurations),-1)
            before=np.sort(flat,axis=1)
            for row in flat:rng.shuffle(row)
            if not np.array_equal(before,np.sort(flat,axis=1)):
                raise ValueError('Shuffle changed histogram')
            shuffle_checks+=len(flat)
        inputs=prefix_batch(configurations,site,metadata[c['id']],alphabet[:c['q']],device)
        target=torch.as_tensor(configurations.reshape(32,-1)[:,site].astype('int64'),device=device)
        factor=schedule_factor(step,spec['steps'])
        optimizer.param_groups[0]['lr']=spec['learning_rate']*factor
        adapter.model.train();optimizer.zero_grad(set_to_none=True)
        logits,_=selected_forward(adapter,inputs,alphabet[:c['q']])
        loss=torch.nn.functional.cross_entropy(logits,target)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,opt['gradient_clip_norm'])
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
        optimizer.step()
        traces.append([step+1,c['id'],site,float(loss.detach()),float(norm),optimizer.param_groups[0]['lr']])
        if (step+1)%128==0:
            np.save(out/'training-trace.npy',np.asarray(traces))
            print(name,'step',step+1,'mean_nll',float(np.mean(np.asarray(traces)[-128:,3])),
                  'seconds',round(time.perf_counter()-start,1),flush=True)
        if step+1 in spec['checkpoints']:
            observations=observe(adapter,processor,alphabet,dev_ds,out,step+1,True)
            state_path=out/f'state-{step+1:05d}.pt'
            torch.save(dict(model={n:p.detach().cpu() for n,p in adapter.model.state_dict().items()},
                optimizer=optimizer.state_dict(),step=step+1,case=case,
                protocol_sha256=sha256(study/'protocol.json'),draws_sha256=spec['draws_sha256'],
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device),
                numpy_rng=rng.bit_generator.state),state_path)
            checkpoints.append(dict(step=step+1,path=str(state_path),sha256=sha256(state_path)))
    np.save(out/'training-trace.npy',np.asarray(traces))
    # A development-only generative check. Assessment uses a separate script and
    # frozen, independent reference chains after the training protocol is fixed.
    if spec['stage'].startswith('development'):
        generated=[]
        for c in dev_ds['cells']:
            if c['temperature_ratio']!=1.:continue
            x=generate(adapter,context_tokens(processor,c['q'],c['L'],1.),alphabet[:c['q']],
                       c['L'],128,1770000+c['id'])
            p=out/f'generated-q{c["q"]}-L{c["L"]}.npy';np.save(p,x)
            r=np.memmap(c['path'],mode='r',dtype=np.uint8,shape=tuple(c['shape']))[-2:].reshape(-1,c['L'],c['L'])
            generated.append(dict(q=c['q'],L=c['L'],model=summarize(x,c['q']),reference=summarize(r,c['q'])))
        write_json(out/'development-generation.json',generated)
    final_generator=parameter_digest(adapter.model,True)
    final_body=parameter_digest(adapter.model,False)
    if case['arm']=='frozen-generator' and final_generator!=initial_generator:
        raise ValueError('Frozen metric generator changed')
    if final_body==initial_body or (case['arm']!='frozen-generator' and final_generator==initial_generator):
        raise ValueError('Expected trainable parameter update absent')
    files={str(p.relative_to(out)):sha256(p) for p in out.iterdir() if p.is_file()}
    write_json(out/'manifest.json',dict(status='complete',case=case,completed_updates=spec['steps'],
        consumed_configurations=spec['steps']*32,checkpoints=checkpoints,files=files,
        branch_checks=dict(initial_generator=initial_generator,final_generator=final_generator,
            initial_body=initial_body,final_body=final_body,shuffle_histograms_checked=shuffle_checks),
        generator_parameter_names=generator_names,trainable_parameters=sum(p.numel() for p in parameters),
        protocol_sha256=sha256(study/'protocol.json'),runtime_seconds=time.perf_counter()-start,
        peak_gib=torch.cuda.max_memory_allocated(device)/2**30))
    print('complete',name,flush=True)


def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--study',required=True);p.add_argument('--stage',required=True)
    p.add_argument('--steps',type=int,default=2048);p.add_argument('--lr',type=float,default=.0003)
    p=sub.add_parser('worker');p.add_argument('--study',required=True);p.add_argument('--name',required=True);p.add_argument('--device',required=True)
    a=ap.parse_args()
    if a.command=='prepare':prepare(a.study,a.stage,a.steps,a.lr)
    else:worker(a.study,a.name,a.device)


if __name__=='__main__':main()
