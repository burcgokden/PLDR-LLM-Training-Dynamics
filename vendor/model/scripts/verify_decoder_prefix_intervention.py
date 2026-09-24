#!/usr/bin/env python
"""Independently reconstruct every full-window/proper-prefix risk and bound."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model_rg.provenance import sha256,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-feasible-20260908');p.add_argument('--case',required=True)
    a=p.parse_args();root=Path(a.root).resolve();study=root/a.study;repo=Path(__file__).resolve().parents[1]
    selection=study/'protocols/decoder-prefix-intervention-selection.json';spec=json.loads(selection.read_text())
    chosen=[c for c in spec['cases'] if c['name']==a.case]
    if len(chosen)!=1:raise AssertionError('One selected prefix-risk observation is required')
    case=chosen[0];out=study/'verification/decoder-prefix-risk'/(a.case+'.json')
    if out.exists():raise FileExistsError(out)
    checked={}
    def check(path,digest=None):
        path=Path(path).resolve();key=str(path)
        if key not in checked:checked[key]=sha256(path)
        if digest is not None and checked[key]!=digest:raise AssertionError('Changed prefix-risk evidence: '+key)
    check(selection)
    for path,digest in spec['inputs_sha256'].items():check(path,digest)
    for name,digest in spec['producer_sources'].items():check(repo/name,digest)
    folder=study/'measurements/decoder-prefix-risk'/a.case;check(folder/'manifest.json')
    meta=json.loads((folder/'manifest.json').read_text())
    if meta['status']!='complete' or meta['case']!=case:raise AssertionError('A selected causal-risk result is incomplete')
    for name,key in [('binding.json','binding_sha256'),('measurements.npz','raw_sha256'),('results.json','results_sha256')]:
        check(folder/name,meta[key])
    binding=json.loads((folder/'binding.json').read_text())
    for path,digest in binding['inputs'].items():check(path,digest)
    for name,digest in binding['source_files'].items():check(folder/'source'/name,digest)
    arrays=[]
    for name in ['full-logits.npy','proper-logits.npy']:
        check(folder/name,meta['logit_sidecars'][name]);value=np.load(folder/name,mmap_mode='r')
        if value.shape!=(32,64,32000) or value.dtype!=np.float32 or not np.isfinite(value).all():
            raise AssertionError('Incomplete causal vocabulary-logit record')
        arrays.append(value)
    full,proper=arrays
    if full[:,-1].tobytes()!=proper[:,-1].tobytes():raise AssertionError('Identical full-length forwards differ')
    probe=root/'controlled-study-20260905/data/short';tokens=np.load(probe/'tokens.npy',mmap_mode='r')
    offsets=np.load(probe/'offsets.npy');rows=np.arange(512,544)
    crops=tokens[rows[:,None],offsets[rows,None]+np.arange(65)];target=crops[:,1:];mask=target!=0
    torch.set_num_threads(spec['threads']);native_values=0;maximum_error=0.;score_excess=-np.inf;kl_excess=-np.inf
    native_full=torch.nn.functional.cross_entropy(torch.from_numpy(np.array(full)).transpose(1,2),
        torch.tensor(target,dtype=torch.long),reduction='none').numpy()
    native_proper=np.empty((32,64),dtype=np.float32)
    mathematical={name:np.empty((32,64),dtype=float) for name in
        ['full_nll64','proper_nll64','oscillation','forward_kl','score_error']}
    with np.load(folder/'measurements.npz') as raw:
        for key,value in [('rows',rows),('crops',crops),('mask',mask)]:np.testing.assert_array_equal(raw[key],value)
        for j in range(64):
            z=full[:,j].astype(np.longdouble);w=proper[:,j].astype(np.longdouble)
            z-=z.max(axis=1,keepdims=True);w-=w.max(axis=1,keepdims=True)
            lz=z-np.log(np.exp(z).sum(axis=1,keepdims=True))
            lw=w-np.log(np.exp(w).sum(axis=1,keepdims=True))
            probability=np.exp(lz);displacement=proper[:,j].astype(float)-full[:,j].astype(float)
            omega=displacement.max(axis=1)-displacement.min(axis=1)
            # Direct extended-precision KL, independent of the producer's
            # centered expm1 and sixth-order small-displacement formula.
            kl=np.sum(probability*(lz-lw),axis=1).astype(float)
            full_nll=-lz[np.arange(32),target[:,j]].astype(float)
            proper_nll=-lw[np.arange(32),target[:,j]].astype(float)
            mathematical['full_nll64'][:,j]=full_nll;mathematical['proper_nll64'][:,j]=proper_nll
            mathematical['oscillation'][:,j]=omega;mathematical['forward_kl'][:,j]=kl
            mathematical['score_error'][:,j]=proper_nll-full_nll
            native_proper[:,j]=torch.nn.functional.cross_entropy(
                torch.from_numpy(np.array(proper[:,j],order='C')),torch.tensor(target[:,j],dtype=torch.long),
                reduction='none').numpy()
            score_excess=max(score_excess,float(np.max(np.abs(proper_nll-full_nll)-omega)))
            kl_excess=max(kl_excess,float(np.max(kl-np.minimum(omega,omega**2/8))))
            if np.min(kl)<-2e-12:raise AssertionError('Negative extended-precision causal KL')
        for key,value in [('native_full_nll',native_full),('native_proper_nll',native_proper)]:
            if raw[key].dtype!=value.dtype or raw[key].shape!=value.shape or raw[key].tobytes()!=value.tobytes():
                raise AssertionError('Native causal-risk reduction failed bytewise replay')
            native_values+=value.size
        for key,value in mathematical.items():
            error=float(np.max(np.abs(value-raw[key])));maximum_error=max(maximum_error,error)
            if not np.allclose(value,raw[key],rtol=2e-12,atol=2e-10):
                raise AssertionError('Independent causal-risk reduction differs: '+key)
        if score_excess>2e-10 or kl_excess>2e-10:
            raise AssertionError('A gauge-invariant score or KL certificate failed')
        valid=int(mask.sum());avg=lambda value:float(np.sum(value*mask)/valid)
        expected=dict(valid_targets=valid,contexts=32,positions=64,native_full_risk=avg(native_full.astype(float)),
            native_proper_risk=avg(native_proper.astype(float)),full_risk64=avg(raw['full_nll64']),
            proper_risk64=avg(raw['proper_nll64']),proper_minus_full_risk64=avg(raw['score_error']),
            mean_oscillation_bound=avg(raw['oscillation']),mean_forward_kl=avg(raw['forward_kl']),
            maximum_forward_kl=float(raw['forward_kl'].max()),maximum_oscillation=float(raw['oscillation'].max()),
            maximum_full_native_reduction_difference=float(np.max(np.abs(native_full-raw['full_nll64']))),
            maximum_proper_native_reduction_difference=float(np.max(np.abs(native_proper-raw['proper_nll64']))),
            full_length_replay_bytewise=True)
    report=json.loads((folder/'results.json').read_text())
    baseline=np.load(case['baseline_full_logits'],mmap_mode='r')
    with np.load(case['calibration_raw']) as z:calibration=z['calibration_cache'].astype(float).mean(2,keepdims=True).astype(np.float32)
    for name,digest in meta['logit_sidecars'].items():check(folder/name,digest)
    observed=np.load(folder/'calibration-operators.npy')
    if observed.dtype!=calibration.dtype or observed.shape!=calibration.shape or observed.tobytes()!=calibration.tobytes():
        raise AssertionError('The selected calibration operator changed')
    expected_sidecars={'full-logits.npy','proper-logits.npy','calibration-operators.npy','restored-logits.npy'}
    controls=['restored-logits.npy']
    if case['operation']=='fixed':controls.append('own-G-logits.npy');expected_sidecars.add('own-G-logits.npy')
    if case['layers']==list(range(5)):controls.append('global-helper-logits.npy');expected_sidecars.add('global-helper-logits.npy')
    if set(meta['logit_sidecars'])!=expected_sidecars:raise AssertionError('A native intervention control was omitted')
    for name in controls:
        value=np.load(folder/name,mmap_mode='r');target=full if name=='global-helper-logits.npy' else baseline
        if value.dtype!=target.dtype or value.shape!=target.shape or value.tobytes()!=target.tobytes():
            raise AssertionError('An independently retained native control differs bytewise: '+name)
    if report['native_restoration_bytewise'] is not True or report['selected_own_G_replay_bytewise']!=(True if case['operation']=='fixed' else None) or report['global_helper_replay_bytewise']!=(True if case['layers']==list(range(5)) else None):
        raise AssertionError('Native intervention qualification was omitted')
    if report['case']!=case or report['scope']!=spec['scope'] or report['status']!='complete':
        raise AssertionError('Causal-risk report identity changed')
    if any(report[key]!=value for key,value in expected.items()):raise AssertionError('Causal-risk aggregate changed')
    if native_values!=4096:raise AssertionError('The complete position panel is missing')
    write_json(out,dict(status='passed',case=case,checked_sha256=checked,verifier_sha256=sha256(__file__),
        logit_elements_reconstructed=131072000,native_risks_replayed_bytewise=native_values,
        mathematical_values_reconstructed=10240,maximum_extended_precision_difference=maximum_error,
        maximum_score_bound_excess=score_excess,maximum_kl_bound_excess=kl_excess,summary=expected,
        arithmetic_allowance=dict(absolute=2e-10,relative=2e-12),additional_independent_training_identities=0,
        native_control_elements_compared=len(controls)*65536000,
        scope='All full-window and proper-prefix logits, native per-target losses and mathematical score/KL '
        'bounds were reconstructed. Extended-precision discrepancies are reported separately from bytewise '
        'native replay. These finite causal-risk observations do not infer a critical exponent.'))
    print('All 2,048 causal target positions verified',a.case,flush=True)


if __name__=='__main__':main()
