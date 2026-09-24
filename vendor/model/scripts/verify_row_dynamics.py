#!/usr/bin/env python
"""Independent array-level checks for metric transport and full-graph row reductions."""
import argparse
import json
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256, write_json


def prediction_kl(reference, changed):
    reference=np.asarray(reference,dtype=float);changed=np.asarray(changed,dtype=float)
    probability=np.exp(reference-reference.max(-1,keepdims=True));probability/=probability.sum(-1,keepdims=True)
    delta=changed-reference;delta-=(probability*delta).sum(-1,keepdims=True)
    return np.log1p((probability*np.expm1(delta)).sum(-1))-(probability*delta).sum(-1)


def pair_covariance(values):
    seeds,contexts,layers,heads=values.shape;q=values.astype(float).mean(-1)
    pairs=np.stack([q[i]-q[j] for i in range(seeds) for j in range(i)])
    return heads*np.einsum('pxl,pxm->lm',pairs,pairs)/(seeds*(seeds-1)*contexts)


def verify_rows(study, bound):
    study=Path(study);repo=Path(__file__).resolve().parents[1];transport=0;projection=0;matrix_checks=0;kl_checks=0
    maximum_covariance_error=0.;maximum_kl_error=0.;exact_logit_cases=0;maximum_implementation_kl=0.
    for protocol_name in ['row-transport','row-transport-transfer','row-projection']:
        path=study/'protocols'/(protocol_name+'.json');spec=json.loads(path.read_text());ledger=json.loads((study/(protocol_name+'.json')).read_text())
        bound(repo/'internal/dynamics-protocols'/(protocol_name+'.json'),sha256(path))
        if ledger['status']!='complete' or len(ledger['records'])!=len(spec['cases']) or any(r['returncode'] for r in ledger['records']):raise AssertionError('Row diagnostic inventory incomplete')
        if ledger.get('protocol_sha256'):bound(path,ledger['protocol_sha256'])
        for case in spec['cases']:
            name=case.get('output_name',case['name']);folder=study/'analysis'/name
            meta=json.loads((folder/'manifest.json').read_text());result=json.loads((folder/'results.json').read_text());raw=np.load(folder/'measurements.npz')
            if meta['status']!='complete' or result['case']!=case:raise AssertionError('Row diagnostic condition changed')
            for filename,key in [('results.json','results_sha256'),('measurements.npz','raw_sha256'),('binding.json','binding_sha256')]:bound(folder/filename,meta[key])
            binding=json.loads((folder/'binding.json').read_text())
            for path,digest in binding['inputs'].items():bound(path,digest)
            for path,digest in binding['source_files'].items():bound(folder/'source'/path,digest)
            for key in raw.files:
                if not np.isfinite(raw[key]).all():raise AssertionError('Nonfinite row diagnostic '+name+'/'+key)
            np.testing.assert_array_equal(raw['cohort'],np.arange(*spec['rows']))
            if protocol_name!='row-projection':
                np.testing.assert_array_equal(raw['radii'],spec['radii'])
                if case['step']==0 and result['initial_parameter_identity'] is not True:raise AssertionError('Initial row state identity not checked')
                for record in result['records']:
                    key=record['unit'];x=raw[key+'_native_input'].astype(float);y=raw[key+'_smooth_output'];j=raw[key+'_jacobian']
                    dx=x-x.mean(-2,keepdims=True);dy=y-y.mean(-2,keepdims=True);linear=np.einsum('...ij,...kj->...ik',dx,j);residual=dy-linear
                    np.testing.assert_allclose(linear,raw[key+'_linear_centered'],rtol=1e-11,atol=1e-13)
                    np.testing.assert_allclose(residual,raw[key+'_centered_residual'],rtol=1e-11,atol=1e-13)
                    covariance=dy.swapaxes(-1,-2)@dy/64
                    transported=linear.swapaxes(-1,-2)@linear/64
                    cross=linear.swapaxes(-1,-2)@residual/64
                    residual_cov=residual.swapaxes(-1,-2)@residual/64
                    reconstructed=transported+cross+cross.swapaxes(-1,-2)+residual_cov
                    error=float(np.max(np.abs(reconstructed-covariance)));maximum_covariance_error=max(error,maximum_covariance_error)
                    np.testing.assert_allclose(covariance,reconstructed,rtol=1e-8,atol=1e-11)
                    for label,values in [('input',dx),('output',dy),('linear',linear),('residual',residual)]:
                        np.testing.assert_allclose(np.mean(values**2,axis=(-2,-1)),raw[key+'_variance_'+label],rtol=1e-11,atol=1e-13)
                    np.testing.assert_allclose(2*np.mean(linear*residual,axis=(-2,-1)),raw[key+'_variance_cross'],rtol=1e-10,atol=1e-13)
                    norms=np.linalg.svd(j,compute_uv=False)[...,0]
                    np.testing.assert_allclose(norms,raw[key+'_spectral_norm'],rtol=1e-10,atol=1e-12)
                    native_gap=np.sqrt(np.mean((y-raw[key+'_native_output'].astype(float))**2,axis=(-2,-1)))
                    np.testing.assert_allclose(native_gap,raw[key+'_native_smooth_rms_difference'],rtol=1e-10,atol=1e-14)
                    matrix_checks+=int(np.prod(x.shape[:-2]))
                if len(result['records'])!=40:raise AssertionError('Missing metric unit')
                transport+=1
            else:
                if [r['variant'] for r in result['records']]!=spec['variants']:raise AssertionError('An included row reduction changed')
                for record in result['records']:
                    variant=record['variant'];logits=raw[variant+'_logits'];fields=raw[variant+'_fields'];heads=raw[variant+'_heads']
                    kl=prediction_kl(raw['base_logits'],logits)
                    maximum_kl_error=max(maximum_kl_error,float(np.max(np.abs(kl-raw[variant+'_kl']))))
                    np.testing.assert_allclose(kl,raw[variant+'_kl'],rtol=1e-8,atol=2e-13)
                    np.testing.assert_allclose(kl.mean(),record['mean_kl'],rtol=1e-8,atol=2e-13)
                    np.testing.assert_allclose(kl.max(),record['maximum_kl'],rtol=1e-8,atol=2e-13)
                    mean=heads.mean(-2)
                    np.testing.assert_allclose(mean[...,0]*np.log(64),fields[:,:5],rtol=3e-6,atol=2e-6)
                    for index,section in [(1,slice(5,10)),(2,slice(26,31)),(3,slice(31,36))]:np.testing.assert_allclose(mean[...,index],fields[:,section],rtol=3e-6,atol=2e-6)
                    if variant.startswith('terminal_'):
                        count={'terminal_mean':0,'terminal_keep1':1,'terminal_keep4':4}[variant]
                        for layer in range(5):
                            indices=raw[f'{variant}_L{layer}_kept_indices']
                            if indices.shape!=(*heads.shape[:1],heads.shape[-2],count) or indices.dtype!=np.int64:raise AssertionError('Missing retained projection indices')
                            if count and (indices.min()<0 or indices.max()>=64 or np.any(np.diff(np.sort(indices,axis=-1),axis=-1)==0)):raise AssertionError('Invalid exception selection')
                            if (raw[f'{variant}_L{layer}_row_projection_mse']<0).any():raise AssertionError('Negative projection defect')
                    if variant=='base':np.testing.assert_array_equal(kl,0)
                    kl_checks+=len(kl)
                comparison=prediction_kl(raw['initial_mean_broadcast_logits'],raw['initial_mean_compressed_logits'])
                np.testing.assert_allclose(comparison,raw['compressed_broadcast_kl'],rtol=1e-8,atol=2e-13)
                exact_logit_cases+=int(np.array_equal(raw['initial_mean_broadcast_logits'],raw['initial_mean_compressed_logits']))
                maximum_implementation_kl=max(maximum_implementation_kl,float(comparison.max()))
                projection+=1
            raw.close()
    if (transport,projection)!=(32,34):raise AssertionError('Required row diagnostic count changed')
    return dict(transport_checkpoints=transport,projection_checkpoints=projection,projection_variants=6,
        recomputed_row_covariances=matrix_checks,recomputed_predictive_kls=kl_checks,
        exact_broadcast_compressed_logit_cases=exact_logit_cases,maximum_broadcast_compressed_kl=maximum_implementation_kl,
        maximum_row_covariance_identity_error=maximum_covariance_error,maximum_independent_kl_discrepancy=maximum_kl_error)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    cache={}
    def bound(path,signature):
        path=str(Path(path).resolve())
        if path not in cache:cache[path]=sha256(path)
        if cache[path]!=signature:raise AssertionError('Changed row evidence: '+path)
    report=verify_rows(Path(args.root)/'criticality-dynamics-20260906',bound)
    write_json(args.output,dict(status='passed',verifier_sha256=sha256(__file__),hashed_files=len(cache),**report));print(report)


if __name__=='__main__':main()
