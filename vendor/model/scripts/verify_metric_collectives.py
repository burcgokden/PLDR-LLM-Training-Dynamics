#!/usr/bin/env python
"""Check common-row covariance by pair differences and native-array reconstruction."""
from companion_paths import required_input
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from model_rg.provenance import sha256, write_json


def verify_metric_collectives(study,folder,bound):
    study=Path(study);folder=Path(folder);repo=Path(__file__).resolve().parents[1]
    protocol=study/'protocols/metric-collectives.json';spec=json.loads(protocol.read_text())
    bound(Path(required_input('dynamics-protocols')) / 'metric-collectives.json',sha256(protocol))
    bound(study/'protocols/row-gradient-projection.json',spec['source_protocol_sha256'])
    meta=json.loads((folder/'manifest.json').read_text());result=json.loads((folder/'results.json').read_text());binding=json.loads((folder/'binding.json').read_text())
    if meta['status']!='complete':raise AssertionError('Incomplete metric collective analysis')
    for name,key in [('binding.json','binding_sha256'),('results.json','results_sha256'),('measurements.npz','raw_sha256')]:bound(folder/name,meta[key])
    for name,signature in binding['inputs'].items():bound(name,signature)
    for name,signature in binding['source_files'].items():bound(folder/'source'/name,signature)
    keys=[(r['heads'],r['step']) for r in result['conditions']]
    if len(keys)!=6 or set(keys)!={(n,t) for n in [4,8,14] for t in [2048,8192]}:raise AssertionError('Metric collective inventory changed')
    lookup={(c['heads'],c['step'],c['seed'],c['batch_step']):c for c in spec['cases']};raw=np.load(folder/'measurements.npz');matrices=0
    torch.set_num_threads(2)
    for record in result['conditions']:
        n=record['heads'];t=record['step'];key=f'N{n}_t{t}';a=raw[key+'_mean_head_matrix'];e=raw[key+'_head_row_energy'];c2=raw[key+'_head_centroid_square'];a2=raw[key+'_head_matrix_square'];names=[]
        if a.shape!=(4,64,5,64,64) or e.shape!=(4,64,5,n):raise AssertionError('Metric observation shape changed')
        if record['seeds']!=list(range(640101,640105)) or record['batch_steps']!=[2048,8192]:raise AssertionError('Metric source order changed')
        for si,seed in enumerate(record['seeds']):
            for bi,batch in enumerate(record['batch_steps']):
                case=lookup[n,t,seed,batch];names.append(case['name']);source=np.load(study/'analysis'/case['name']/'measurements.npz');section=slice(32*bi,32*(bi+1))
                np.testing.assert_array_equal(source['crops'],raw[key+'_crops'][section])
                for layer in range(5):
                    value=torch.from_numpy(source[f'base_L{layer}_terminal_rows']).double();mu=value.mean(-2,keepdim=True)
                    np.testing.assert_allclose(value.mean(1).numpy(),a[si,section,layer],rtol=1e-12,atol=1e-15)
                    np.testing.assert_allclose((value-mu).square().mean((-1,-2)).numpy(),e[si,section,layer],rtol=1e-10,atol=1e-22)
                    np.testing.assert_allclose(mu.square().mean((-1,-2)).numpy(),c2[si,section,layer],rtol=1e-12,atol=1e-15)
                    np.testing.assert_allclose(value.square().mean((-1,-2)).numpy(),a2[si,section,layer],rtol=1e-12,atol=1e-15)
                    matrices+=32
                source.close()
        if names!=record['source_cases']:raise AssertionError('Metric analysis used different sources')
        total=np.zeros(64);common=np.zeros(64);contrast=np.zeros(64);covariance=np.zeros((320,320));energy_chi=0.
        for i in range(4):
            for j in range(i):
                difference=a[i]-a[j];mean=difference.mean(-2);normal=difference-mean[...,None,:]
                total+=n*np.sum(difference*difference,axis=(1,2,3))/(12*5*4096)
                common+=n*np.sum(mean*mean,axis=(1,2))/(12*5*64)
                contrast+=n*np.sum(normal*normal,axis=(1,2,3))/(12*5*4096)
                flat=mean.reshape(64,320);covariance+=n*(flat.T@flat)/(12*64*320)
                de=e[i].mean(-1)-e[j].mean(-1);energy_chi+=n*np.mean(de*de)/12
        for label,values in [('total',total),('common',common),('normal',contrast)]:np.testing.assert_allclose(values,raw[key+'_'+label+'_by_context'],rtol=1e-9,atol=1e-20)
        np.testing.assert_allclose(total,common+contrast,rtol=1e-12,atol=1e-18)
        np.testing.assert_allclose(covariance,raw[key+'_common_covariance'],rtol=1e-9,atol=1e-18)
        values=[e.mean()*4096,e.mean(),c2.mean(),a2.mean(),total.mean(),common.mean(),contrast.mean(),common.mean()/total.mean(),energy_chi]
        fields=['mean_absolute_row_energy','mean_row_energy_per_entry','mean_centroid_square','mean_matrix_square','metric_susceptibility','common_row_susceptibility','contrast_susceptibility','common_fraction','row_energy_susceptibility']
        np.testing.assert_allclose(values,[record[k] for k in fields],rtol=1e-9,atol=1e-20)
        spectrum=torch.linalg.eigvalsh(torch.from_numpy(covariance)).numpy()
        np.testing.assert_allclose(spectrum,record['common_covariance_eigenvalues'],rtol=1e-7,atol=1e-15)
        np.testing.assert_allclose([np.trace(covariance)**2/np.linalg.norm(covariance,'fro')**2,spectrum[-1]/np.trace(covariance)],
                                   [record['common_effective_rank'],record['common_leading_fraction']],rtol=1e-9,atol=1e-14)
        print('Verified metric collective',n,t,flush=True)
    raw.close()
    return dict(conditions=6,source_checkpoint_batches=48,reconstructed_head_mean_matrices=matrices,pairwise_metric_comparisons=6*6*64*5,common_covariance_matrices=6)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--analysis',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    signature=sha256(__file__);cache={}
    def bound(path,h):
        path=Path(path).resolve()
        if path not in cache:cache[path]=sha256(path)
        if cache[path]!=h:raise AssertionError('Metric collective input changed: '+str(path))
    checks=verify_metric_collectives(Path(args.root)/'criticality-dynamics-20260906',Path(args.analysis),bound)
    if signature!=sha256(__file__):raise AssertionError('Metric collective verifier changed during execution')
    write_json(args.output,dict(status='passed',verifier_sha256=signature,hashed_files=len(cache),checks=checks))


if __name__=='__main__':main()
