#!/usr/bin/env python
"""Independently check the matched sampler and full row-adjoint arrays."""
import argparse
import json
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256, write_json


def verify_adjoint(study,bound,include_source_control=False):
    study=Path(study);repo=Path(__file__).resolve().parents[1];spec_cases=[]
    for name in ['row-adjoint']+(['row-adjoint-source-control'] if include_source_control else []):
        protocol=study/'protocols'/(name+'.json');spec=json.loads(protocol.read_text());ledger=json.loads((study/(name+'.json')).read_text())
        bound(repo/'internal/dynamics-protocols'/(name+'.json'),sha256(protocol));bound(protocol,ledger['protocol_sha256'])
        if ledger['status']!='complete' or len(ledger['records'])!=24 or any(r['returncode'] for r in ledger['records']):raise AssertionError('Incomplete adjoint ledger')
        if [r['name'] for r in ledger['records']]!=[c['name'] for c in spec['cases']]:raise AssertionError('Adjoint execution identities changed')
        spec_cases.extend(spec['cases'])
    tokens=np.load(study.parent/'data/refinedweb-4608/tokens.npy');units=0;cases=[];matched_crops={}
    for case in spec_cases:
        folder=study/'analysis'/case['name'];meta=json.loads((folder/'manifest.json').read_text());result=json.loads((folder/'results.json').read_text())
        if meta['status']!='complete' or result['case']!=case:raise AssertionError('Adjoint source identity changed')
        for name,key in [('results.json','results_sha256'),('measurements.npz','raw_sha256'),('binding.json','binding_sha256')]:bound(folder/name,meta[key])
        binding=json.loads((folder/'binding.json').read_text())
        for p,h in binding['inputs'].items():bound(p,h)
        for p,h in binding['source_files'].items():bound(folder/'source'/p,h)
        with np.load(folder/'measurements.npz') as raw:
            step=case.get('batch_step',case['step']);rows=np.random.default_rng(640001+1000).integers(0,3072,size=(step+1,32))[-1]
            offsets=np.random.default_rng(640001+1001000).integers(0,449,size=(step+1,32))[-1]
            np.testing.assert_array_equal(raw['rows'],rows);np.testing.assert_array_equal(raw['offsets'],offsets)
            np.testing.assert_array_equal(raw['crops'],tokens[rows[:,None],offsets[:,None]+np.arange(65)])
            if step in matched_crops:np.testing.assert_array_equal(raw['crops'],matched_crops[step])
            else:matched_crops[step]=raw['crops'].copy()
            if len(result['records'])!=40 or {r['unit'] for r in result['records']}!={f'L{l}_U{u}' for l in range(5) for u in range(8)}:raise AssertionError('Adjoint unit coverage changed')
            for record in result['records']:
                key=record['unit'];a=raw[key+'_input_adjoint'];b=raw[key+'_output_adjoint'];reference=raw[key+'_smooth_input_adjoint']
                if not all(np.isfinite(v).all() for v in [a,b,reference]):raise AssertionError('Nonfinite adjoint array')
                vin=np.sum(a*a,axis=(-1,-2));vout=np.sum(b*b,axis=(-1,-2));error=np.sqrt(np.sum((reference-a)**2,axis=(-1,-2)))
                np.testing.assert_allclose([vin.sum(),vout.sum()],[record['input_adjoint_energy'],record['output_adjoint_energy']],rtol=1e-11,atol=1e-15)
                if vout.sum()>0:np.testing.assert_allclose(np.sqrt(vin.sum()/vout.sum()),record['aggregate_adjoint_gain'],rtol=1e-11,atol=1e-14)
                mask=vin>1e-40
                if mask.any():np.testing.assert_allclose(np.max(error[mask]/np.sqrt(vin[mask])),record['maximum_local_vjp_relative_error'],rtol=1e-9,atol=1e-12)
                layer,index=[int(s[1:]) for s in key.split('_')]
                if index>0:np.testing.assert_array_equal(a,raw[f'L{layer}_U{index-1}_output_adjoint'])
                indices=raw[key+'_retained4_indices']
                if indices.shape!=(*a.shape[:-2],4) or indices.dtype!=np.int64 or indices.min()<0 or indices.max()>=64 or np.any(np.diff(np.sort(indices,axis=-1),axis=-1)==0):raise AssertionError('Invalid retained adjoint indices')
                selected=np.zeros(a.shape[:-1],dtype=bool);np.put_along_axis(selected,indices,True,axis=-1)
                for label,adjoint,energy in [('input',a,vin),('output',b,vout)]:
                    if energy.sum()>0:np.testing.assert_allclose(np.sum(np.sum(adjoint**2,axis=-1)*selected)/energy.sum(),record['retained4_'+label+'_adjoint_energy_fraction'],rtol=1e-11,atol=1e-14)
                units+=1
        np.testing.assert_allclose(result['clipping_factor'],min(1.,1/(result['total_gradient_norm']+1e-6)),rtol=1e-13,atol=1e-15)
        cases.append(dict(name=case['name'],manifest_sha256=sha256(folder/'manifest.json')))
    expected=48 if include_source_control else 24
    if units!=expected*40:raise AssertionError('Adjoint unit coverage changed')
    if include_source_control and {(c['heads'],c['seed'],c['step'],c.get('batch_step',c['step'])) for c in spec_cases}!={(h,s,t,b) for h in [4,8,14] for s in range(640101,640105) for t in [2048,8192] for b in [2048,8192]}:raise AssertionError('Unbalanced matched-batch design')
    return dict(cases=cases,training_adjoint_cases=expected,verified_metric_units=units,sampler_rows=expected*32,distinct_matched_minibatches=len(matched_crops),include_source_control=include_source_control)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--output',required=True);parser.add_argument('--include-source-control',action='store_true');args=parser.parse_args();cache={}
    def bound(path,digest):
        path=str(Path(path).resolve())
        if path not in cache:cache[path]=sha256(path)
        if cache[path]!=digest:raise AssertionError('Changed adjoint evidence: '+path)
    report=verify_adjoint(Path(args.root)/'criticality-dynamics-20260906',bound,args.include_source_control)
    write_json(args.output,dict(status='passed',verifier_sha256=sha256(__file__),hashed_files=len(cache),**report));print('Verified',report['training_adjoint_cases'],'matched training-adjoint cases')


if __name__=='__main__':main()
