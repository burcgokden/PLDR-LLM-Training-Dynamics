#!/usr/bin/env python
"""Retain full-window and every proper-prefix prediction on a fixed cohort."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from model_rg.controlled import bind_run
from model_rg.provenance import sha256,write_json
from model_rg.training import TrainingModel


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-feasible-20260908');p.add_argument('--case',required=True)
    a=p.parse_args();root=Path(a.root).resolve();study=root/a.study;repo=Path(__file__).resolve().parents[1]
    selection=study/'protocols/prefix-risk-comparison.json';spec=json.loads(selection.read_text())
    chosen=[c for c in [spec['qualification'],*spec['cases']] if c['name']==a.case]
    if len(chosen)!=1:raise AssertionError('One selected causal-risk state is required')
    case=chosen[0]
    for path,digest in spec['inputs_sha256'].items():
        if sha256(path)!=digest:raise AssertionError('A causal-risk input changed')
    for name,digest in spec['producer_sources'].items():
        if sha256(repo/name)!=digest:raise AssertionError('A causal-risk producer changed')
    parent=Path(case['parent_manifest']);meta=json.loads(parent.read_text());state=Path(case['state'])
    if meta['status']!='complete':raise AssertionError('A complete saved state or immutable copy is required')
    digest=sha256(state)
    if digest!=meta['saved_states'][str(case['step'])]['sha256'] or digest!=case.get('state_sha256',digest):
        raise AssertionError('The causal-risk checkpoint changed')
    inputs=[selection,parent,state]
    if case.get('training_verification'):
        proof=Path(case['training_verification'])
        if json.loads(proof.read_text())['status']!='complete':raise AssertionError('Native verification is required')
        inputs.append(proof)
    out=study/'measurements/prefix-risk'/a.case;out.mkdir(parents=True,exist_ok=False)
    bind_run(out,[*inputs,*map(Path,spec['inputs_sha256'])],vars(a));before=time.time()
    torch.set_num_threads(spec['threads'])
    saved=torch.load(state,map_location='cpu',mmap=True,weights_only=True)
    if saved['step']!=case['step']:raise AssertionError('Causal-risk time changed')
    for name in ['heads','seed','shared_seed','stream_seed']:
        if saved['arguments'][name]!=case[name]:raise AssertionError('Causal-risk condition changed')
    model=TrainingModel(spec['source'],case['heads'],case['seed'],'cpu')
    model.model.load_state_dict(saved['model']);del saved;model.model.eval().requires_grad_(False)
    probe=root/'controlled-study-20260905/data/short';tokens=np.load(probe/'tokens.npy',mmap_mode='r')
    offsets=np.load(probe/'offsets.npy');rows=np.asarray(spec['rows'])
    crops=np.asarray(tokens[rows[:,None],offsets[rows,None]+np.arange(65)])
    batch=torch.tensor(crops,dtype=torch.long)
    with torch.no_grad():
        full=model.model(batch[:,:64],use_cache=False,logits_to_keep=0).logits
        if full.shape!=(32,64,32000) or not torch.isfinite(full).all():raise AssertionError('Invalid full-window logits')
        np.save(out/'full-logits.npy',full.numpy())
        native_full=torch.nn.functional.cross_entropy(full.transpose(1,2),batch[:,1:],reduction='none').numpy()
        full_values=full.numpy();del full
        proper=np.lib.format.open_memmap(out/'proper-logits.npy',mode='w+',dtype=np.float32,shape=(32,64,32000))
        native_proper=np.empty((32,64),dtype=np.float32)
        for k in spec['prefix_lengths']:
            value=model.model(batch[:,:k],use_cache=False,logits_to_keep=0).logits[:,-1].contiguous()
            if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite proper-prefix logit')
            proper[:,k-1]=value.numpy()
            native_proper[:,k-1]=torch.nn.functional.cross_entropy(value,batch[:,k],reduction='none').numpy()
            del value
        proper.flush()
    if full_values[:,-1].tobytes()!=np.ascontiguousarray(proper[:,-1]).tobytes():
        raise AssertionError('The identical full-length forward did not replay bytewise')
    targets=crops[:,1:];mask=targets!=0
    raw=dict(rows=rows,crops=crops,mask=mask,native_full_nll=native_full,native_proper_nll=native_proper)
    for key in ['full_nll64','proper_nll64','oscillation','forward_kl','score_error']:
        raw[key]=np.empty((32,64),dtype=np.float64)
    for j in range(64):
        z=full_values[:,j].astype(float);w=proper[:,j].astype(float)
        lz=z-z.max(-1,keepdims=True);lz-=np.log(np.exp(lz).sum(-1,keepdims=True))
        lw=w-w.max(-1,keepdims=True);lw-=np.log(np.exp(lw).sum(-1,keepdims=True))
        probability=np.exp(lz);d=w-z;omega=np.ptp(d,axis=-1)
        centered=d-np.sum(probability*d,axis=-1,keepdims=True)
        # Center before expm1 so small native effects retain their quadratic KL.
        mean=np.sum(probability*centered,axis=-1)
        if np.max(centered)>100:
            weighted=centered+lz;maximum=weighted.max(-1)
            kl=maximum+np.log(np.exp(weighted-maximum[:,None]).sum(-1))-mean
        else:
            residual=np.where(np.abs(centered)<1e-3,
                centered**2*(.5+centered*(1/6+centered*(1/24+centered*(1/120+centered/720)))),
                np.expm1(centered)-centered)
            kl=np.log1p(mean+np.sum(probability*residual,axis=-1))-mean
        target=targets[:,j]
        raw['full_nll64'][:,j]=-lz[np.arange(32),target]
        raw['proper_nll64'][:,j]=-lw[np.arange(32),target]
        raw['oscillation'][:,j]=omega;raw['forward_kl'][:,j]=kl
        raw['score_error'][:,j]=lz[np.arange(32),target]-lw[np.arange(32),target]
    if any(not np.isfinite(value).all() for value in raw.values()):raise FloatingPointError('Nonfinite prefix risk reduction')
    denominator=int(mask.sum())
    average=lambda value:float(np.sum(value*mask)/denominator)
    result=dict(status='complete',case=case,valid_targets=denominator,contexts=32,positions=64,
        native_full_risk=average(native_full.astype(float)),native_proper_risk=average(native_proper.astype(float)),
        full_risk64=average(raw['full_nll64']),proper_risk64=average(raw['proper_nll64']),
        proper_minus_full_risk64=average(raw['score_error']),mean_oscillation_bound=average(raw['oscillation']),
        mean_forward_kl=average(raw['forward_kl']),maximum_forward_kl=float(raw['forward_kl'].max()),
        maximum_oscillation=float(raw['oscillation'].max()),
        maximum_full_native_reduction_difference=float(np.max(np.abs(native_full-raw['full_nll64']))),
        maximum_proper_native_reduction_difference=float(np.max(np.abs(native_proper-raw['proper_nll64']))),
        full_length_replay_bytewise=True,scope=spec['scope'])
    np.savez_compressed(out/'measurements.npz',**raw);write_json(out/'results.json',result)
    write_json(out/'manifest.json',dict(schema='complete-position-prefix-risk-v1',status='complete',case=case,
        binding_sha256=sha256(out/'binding.json'),raw_sha256=sha256(out/'measurements.npz'),
        results_sha256=sha256(out/'results.json'),logit_sidecars={name:sha256(out/name) for name in
        ['full-logits.npy','proper-logits.npy']},seconds=time.time()-before,additional_independent_training_identities=0))
    print(case['name'],result['full_risk64'],result['proper_risk64'],flush=True)


if __name__=='__main__':main()
