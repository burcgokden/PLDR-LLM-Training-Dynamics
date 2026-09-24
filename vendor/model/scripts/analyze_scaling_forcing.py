#!/usr/bin/env python
"""Compare native conditional forcing geometry with its predictive tangent emission."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp

from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


def gram_summary(gram, cross, mean_squared):
    eigenvalues,eigenvectors=np.linalg.eigh((gram+gram.T)/2)
    tolerance=max(float(np.max(np.abs(eigenvalues))),1e-300)*1e-10
    if eigenvalues.min() < -tolerance:raise AssertionError('Conditional covariance lost positivity')
    eigenvalues=np.maximum(eigenvalues,0)[::-1];eigenvectors=eigenvectors[:,::-1]
    trace=float(eigenvalues.sum());square=float(np.sum(eigenvalues**2))
    active=eigenvalues>tolerance
    projections=np.zeros_like(eigenvalues)
    projections[active]=(eigenvectors[:,active].T@cross)**2/eigenvalues[active]
    fractions=projections/mean_squared if mean_squared>0 else np.zeros_like(projections)
    if fractions.sum()>1+1e-7:raise AssertionError('Mean projection exceeds its norm')
    return dict(sample_covariance_trace=trace,sample_covariance_eigenvalues=eigenvalues.tolist(),
        participation_rank=trace*trace/square if square>0 else 0.,
        top_eigenvalue_fraction=float(eigenvalues[0]/trace) if trace>0 else None,
        squared_sample_mean=float(mean_squared),
        drift_fraction_in_sample_noise_span=float(fractions.sum()),
        drift_fraction_in_top_noise_direction=float(fractions[0]),
        sample_rank_bound=len(eigenvalues)-1,
        unbiased_squared_drift=float(mean_squared-trace/len(eigenvalues)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='critical-scaling-20260906');p.add_argument('--output',required=True)
    p.add_argument('--wait',action='store_true');a=p.parse_args();study=Path(a.root)/a.study
    launcher=study/'launcher-conditional-noise-jvp.json'
    while True:
        status=json.loads(launcher.read_text())['status']
        if status=='complete':break
        if not a.wait or status!='running':raise RuntimeError('Conditional forcing producer: '+status)
        time.sleep(30)
    protocol=study/'protocols/conditional-noise-jvp.json';spec=json.loads(protocol.read_text())
    paths=[study/'measurements'/c['name'] for c in spec['cases']]
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    inputs=[launcher,protocol]
    for path in paths:inputs.extend(path/name for name in ['manifest.json','results.json','measurements.npz','generator-velocities.npy'])
    bind_run(out,inputs,vars(a));records=[];raw={}
    for path in paths:
        meta=json.loads((path/'manifest.json').read_text());result=json.loads((path/'results.json').read_text())
        for filename,key in [('results.json','results_sha256'),('measurements.npz','raw_sha256'),
                             ('generator-velocities.npy','velocities_sha256')]:
            if sha256(path/filename)!=meta[key]:raise AssertionError('Conditional forcing source changed')
        velocity=np.load(path/'generator-velocities.npy',mmap_mode='r')
        if velocity.shape[0]!=32:raise AssertionError('Conditional forcing sample count changed')
        record=dict(case=meta['case'],sectors={})
        for group in ['shared_metric_network','plga_affine']:
            gram=np.zeros((32,32));cross=np.zeros(32);mean_squared=0.;coordinates=0
            for name,(lo,hi) in zip(result['parameter_names'],result['parameter_slices'],strict=True):
                if ('reslayerAs' in name)!=(group=='shared_metric_network'):continue
                v=np.asarray(velocity[:,lo:hi]);mean=v.mean(0);centered=(v-mean)/np.sqrt(31)
                gram+=centered@centered.T;cross+=centered@mean
                mean_squared+=float(mean@mean);coordinates+=hi-lo
            summary=gram_summary(gram,cross,mean_squared);summary['coordinates']=coordinates
            reference=next(r for r in result['groups'] if r['group']==group)
            np.testing.assert_allclose(summary['sample_covariance_trace'],reference['unbiased_noise_trace'],rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(summary['squared_sample_mean'],reference['squared_sample_mean'],rtol=1e-12,atol=1e-12)
            record['sectors'][group]=summary;raw[path.name+'_'+group+'_gram']=gram
        del velocity
        with np.load(path/'measurements.npz') as z:
            base=z['base_logits'];probability=np.exp(base-logsumexp(base,axis=-1,keepdims=True))
            v=np.stack([z[f'b{k}_jvp_logits'] for k in range(8)])
            v-=np.sum(v*probability,axis=-1,keepdims=True)
            v=(v*np.sqrt(probability/len(base))).reshape(8,-1)
            mean=v.mean(0);centered=(v-mean)/np.sqrt(7)
            gram=centered@centered.T;cross=centered@mean
            summary=gram_summary(gram,cross,float(mean@mean))
            summary['coordinates']=int(v.shape[1]);record['sectors']['predictive_fisher_tangent']=summary
            raw[path.name+'_predictive_fisher_gram']=gram
        records.append(record);print(path.name,'forcing geometry complete',flush=True)
    output=dict(schema='native-conditional-forcing-geometry-v1',status='complete',states=records,
        parameter_units='Real-arithmetic Adam generator velocity divided by its native learning rate, from 32 IID batches conditional on the entire saved incoming state.',
        predictive_units='Directional derivative along one nominal native generator-only displacement, with the full-graph Fisher inner product averaged over 16 fixed contexts. Eight paired batches.',
        scope='Ranks are participation ratios of finite sample covariance matrices, bounded by 31 and 7 respectively. They do not estimate a population rank above those bounds, identify a white-noise process, or validate finite-step Taylor extrapolation. Parameter and predictive sectors have different units.')
    write_json(out/'results.json',output);np.savez_compressed(out/'measurements.npz',**raw)
    write_json(out/'manifest.json',dict(status='complete',binding_sha256=sha256(out/'binding.json'),
        results_sha256=sha256(out/'results.json'),raw_sha256=sha256(out/'measurements.npz'),states=len(records)))


if __name__=='__main__':main()
