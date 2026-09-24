#!/usr/bin/env python
"""Validate passive update observations, then freeze their future run selection."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from model_rg.controlled import bind_run
from model_rg.criticality import optimizer_for, sample_batches
from model_rg.provenance import sha256, write_json
from model_rg.training import TrainingModel
from model_rg.update_records import SharedUpdateRecorder


def state_digest(value):
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            y=x.detach().cpu().contiguous()
            h.update(str((str(y.dtype),tuple(y.shape))).encode())
            h.update(y.numpy().tobytes())
        elif isinstance(x,dict):
            for key in sorted(x,key=str):
                h.update(repr(key).encode());visit(x[key])
        elif isinstance(x,(tuple,list)):
            for item in x:visit(item)
        else:h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--study',default='critical-scaling-20260906')
    a=p.parse_args()
    root=Path(a.root);study=root/a.study
    out=study/'qa/shared-update-native-control'
    out.mkdir(parents=True,exist_ok=False)
    parent=root/'criticality-dynamics-20260906/runs/long-h2-g1-s640101/final-training-state.pt'
    source=root/'assets/PLDR-LLM-v51-SOC-110M-1'
    data=root/'data/refinedweb-4608/tokens.npy'
    protocols=[study/'protocols'/f'{x}.json' for x in
        ['clock-holdout','diffusive-size-map','environment-factorial','zero-horizon']]
    target=study/'protocols/shared-update-observation.json'
    if target.exists():raise FileExistsError(target)
    torch.set_num_threads(4)
    saved=torch.load(parent,map_location='cpu',weights_only=True)
    model=TrainingModel(source,2,640101,'cpu')
    names=[n for n,p in model.model.named_parameters() if 'reslayerAs' in n]
    count=sum(p.numel() for n,p in model.model.named_parameters() if n in names)
    projection=study/'data/shared-update-signs.npy'
    projection.parent.mkdir(parents=True,exist_ok=True)
    if projection.exists():raise FileExistsError(projection)
    signs=np.random.default_rng(650171).integers(0,2,size=(count,16),dtype=np.int8)*2-1
    np.save(projection,signs)
    bind_run(out,[parent,data,projection,*protocols,source/'modeling_pldrllm.py',
                  source/'configuration_pldrllm.py'],vars(a))
    tokens=np.load(data)
    start=saved['step'];steps=8
    rows,offsets,batches=sample_batches(tokens,640001,start+steps)
    np.savez_compressed(out/'sampling.npz',rows=rows[start:],offsets=offsets[start:])
    records=[];direct=[];observed=None
    for enabled in [False,True]:
        model.model.load_state_dict(saved['model'])
        optimizer=optimizer_for(model,1)
        optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
        recorder=SharedUpdateRecorder(model.model,optimizer,projection,names) if enabled else None
        losses=[];norms=[]
        t0=time.time()
        for i in range(start,start+steps):
            model.model.train();optimizer.zero_grad(set_to_none=True)
            batch=torch.tensor(batches[i],dtype=torch.long)
            loss=torch.nn.functional.cross_entropy(model.logits(batch[:,:64]),batch[:,64])
            (loss*1).backward()
            norm=torch.nn.utils.clip_grad_norm_(model.model.parameters(),1.,error_if_nonfinite=True)
            if not enabled:
                before=np.concatenate([p.detach().numpy().ravel().astype(np.float64)
                    for n,p in model.model.named_parameters() if n in names])
                gradient=np.concatenate([p.grad.detach().numpy().ravel().astype(np.float64)
                    for n,p in model.model.named_parameters() if n in names])
            optimizer.step()
            if not enabled:
                after=np.concatenate([p.detach().numpy().ravel().astype(np.float64)
                    for n,p in model.model.named_parameters() if n in names])
                delta=after-before
                # Independent NumPy accumulation uses the unnormalized integer signs.
                direct.append(dict(update=delta@signs/np.sqrt(count),gradient=gradient@signs/np.sqrt(count),
                    moments=np.array([np.mean(delta**2),np.mean(before**2),np.mean(delta*before)])))
            losses.append(float(loss.detach()));norms.append(float(norm))
        record=dict(enabled=enabled,seconds=time.time()-t0,losses=losses,norms=norms,
                    model_sha256=state_digest(model.model.state_dict()),
                    optimizer_sha256=state_digest(optimizer.state_dict()))
        records.append(record)
        if recorder:
            observed=recorder.arrays();recorder.close()
    exact={key:records[0][key]==records[1][key] for key in
        ['model_sha256','optimizer_sha256']}
    for key in ['losses','norms']:
        left=np.asarray(records[0][key],dtype=np.float64)
        right=np.asarray(records[1][key],dtype=np.float64)
        exact[key]=left.shape==right.shape and left.tobytes()==right.tobytes()
    direct_arrays={f'direct_{name}':np.stack([r[name] for r in direct]) for name in ['update','gradient','moments']}
    comparisons={}
    for obs,ref in [('shared_update_projection','direct_update'),
                    ('shared_clipped_gradient_projection','direct_gradient'),
                    ('shared_update_moments','direct_moments')]:
        x=observed[obs];y=direct_arrays[ref]
        comparisons[obs]=dict(max_absolute_error=float(np.max(np.abs(x-y))),
            relative_l2_error=float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-300)),
            passed=bool(np.allclose(x,y,rtol=1e-11,atol=1e-17)))
    np.savez_compressed(out/'measurements.npz',**observed,**direct_arrays)
    accepted=all(exact.values()) and all(x['passed'] for x in comparisons.values())
    result=dict(schema='passive-shared-update-control-v1',status='complete' if accepted else 'failed',
        platform='CPU float32 native forward, gradient clipping and AdamW; float64 read-only projections',
        parent=str(parent),start_step=start,steps=steps,shared_parameters=count,projection_columns=16,
        records=records,bitwise_checks=exact,projection_comparisons=comparisons,
        raw_sha256=sha256(out/'measurements.npz'),binding_sha256=sha256(out/'binding.json'),
        claim='This paired CPU control checks passive instrumentation, not a GPU trajectory equivalence theorem.')
    write_json(out/'manifest.json',result)
    if not accepted:raise AssertionError('Passive observation changed the native control or its projections')
    runs=[]
    for protocol in protocols:
        runs.extend(x['run_id'] for x in json.loads(protocol.read_text())['jobs'])
    runs += [f'extended-h{n}-g1-s{s}' for s in range(640101,640105) for n in [4,14]]
    if len(runs)!=60 or len(set(runs))!=60:raise AssertionError('Unexpected future selection')
    for run in runs:
        if (study/'runs'/run).exists():raise AssertionError('An observation target already started')
    write_json(target,dict(schema='passive-shared-update-observation-v1',
        frozen_at=datetime.now(timezone.utc).isoformat(),run_ids=runs,
        projection=str(projection),projection_sha256=sha256(projection),projection_seed=650171,
        projection_columns=16,coordinate_count=count,parameter_names=names,
        units='Each signed projection is divided by sqrt(coordinate_count); coordinates are native shared reslayerAs parameters in the listed order.',
        observation='Actual float32 parameter differences reduced in float64, clipped incoming gradient projections, and mean squared update, mean squared pre-update weight, mean update times pre-update weight.',
        timing='Every completed optimizer update, including all optimizer-memory effects; pre- and post-step hooks read tensors without writing parameters, gradients, moments, or RNG state.',
        limitations='Sixteen projections do not recover full covariance. Finite temporal correlations and spectra do not establish criticality or stationarity.',
        qualification=str(out/'manifest.json'),qualification_sha256=sha256(out/'manifest.json'),
        training_protocol_sha256={str(x):sha256(x) for x in protocols},
        role='Auxiliary observations only. Existing training protocols and frozen prediction targets remain fixed.'))
    print(json.dumps(dict(control=result['status'],bitwise=exact,comparisons=comparisons,
        selected=len(runs),protocol_sha256=sha256(target)),indent=2),flush=True)


if __name__=='__main__':main()
