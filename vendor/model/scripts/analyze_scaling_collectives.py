#!/usr/bin/env python
"""Separate common, contrast, quadratic and predictive collective observations."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from analyze_size_time import whole_seed_statistics
from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


def collective_fields(data, count, n):
    f,h,c,e,m=[data[name] for name in ['fields','heads','centroids','energies','mean_metric']]
    if c.shape!=(count,512,5,n,64) or m.shape!=(count,64,5,64,64):
        raise AssertionError('The declared context or coordinate axes changed')
    np.testing.assert_allclose(e[...,1],e[...,0]+(c*c).mean(-1),rtol=1e-12,atol=1e-14)
    np.testing.assert_allclose((c*c).mean((-1,-2)),(c.mean(-2)**2).mean(-1)+
        ((c-c.mean(-2,keepdims=True))**2).mean((-1,-2)),rtol=1e-12,atol=1e-14)
    fields=dict(row_native=h[...,2].mean(-1),row_reduced64=(e[...,0]/np.maximum(e[...,1],1e-30)).mean(-1),
        absolute_row_energy=e[...,0].mean(-1),total_energy=e[...,1].mean(-1),
        centroid_energy=(c*c).mean((-1,-2)),common_centroid=c.mean(-2),
        head_common_energy=(c.mean(-2)**2).mean(-1),
        head_residual_energy=((c-c.mean(-2,keepdims=True))**2).mean((-1,-2)),
        head_alignment=(c.mean(-2)**2).mean(-1)/np.maximum((c*c).mean((-1,-2)),1e-30),
        mean_metric_total=m,mean_metric_common=m.mean(-2),
        mean_metric_contrast=m-m.mean(-2,keepdims=True),
        attention=h[...,0].mean(-1),operator_rms=h[...,3].mean(-1),
        prediction_entropy=f[...,24],nll=f[...,25],logit_projection=f[...,16:24])
    if 'evaluation_mean_metric' in data:
        matrix=data['evaluation_mean_metric']
        if matrix.shape!=(count,512,5,64,64):raise AssertionError('Evaluation matrix cohort changed')
        fields.update(evaluation_mean_metric_total=matrix,
            evaluation_mean_metric_common=matrix.mean(-2),
            evaluation_mean_metric_contrast=matrix-matrix.mean(-2,keepdims=True))
        np.testing.assert_allclose(matrix.mean(-2),c.mean(-2),rtol=1e-12,atol=1e-14)
    return fields


def arithmetic_comparison(left, right, n):
    rows={}
    for name in sorted(set(left)&set(right)):
        x,y=left[name],right[name]
        error=x-y
        cx=float(n*np.var(x,axis=0,ddof=1).mean())
        cy=float(n*np.var(y,axis=0,ddof=1).mean())
        ce=float(n*np.var(error,axis=0,ddof=1).mean())
        bound=2*np.sqrt(cy*ce)+ce
        difference=abs(cx-cy)
        if difference>bound*(1+1e-7)+1e-20:
            raise AssertionError('Paired arithmetic variance bound failed')
        rows[name]=dict(float32_susceptibility=cx,float64_susceptibility=cy,
            absolute_difference=difference,centered_error_susceptibility=ce,
            susceptibility_error_bound=float(bound),rms_field_difference=float(np.sqrt(np.mean(error**2))),
            max_field_difference=float(np.max(np.abs(error))),
            relative_bound=float(bound/cy) if cy>0 else None,
            meets_one_percent_or_1e_minus8_absolute=bool(bound<=max(.01*cy,1e-8)))
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--study',default='critical-scaling-20260906')
    p.add_argument('--output',required=True)
    p.add_argument('--prefix',default='cpu-')
    p.add_argument('--precisions',nargs='+',default=['float32'],choices=['float32','float64'])
    a=p.parse_args()
    study=Path(a.root)/a.study
    groups=defaultdict(dict)
    inputs=[]
    for path in sorted((study/'measurements').glob(a.prefix+'*/manifest.json')):
        meta=json.loads(path.read_text())
        if meta['status']!='complete':continue
        c=meta['case']
        if sha256(path.parent/'measurements.npz')!=meta['raw_sha256']:
            raise AssertionError('Frozen collective output changed')
        key=(c['heads'],c['multiplier'],c['step'],c['shared_seed'],c['stream_seed'])
        if c['seed'] in groups[key]:raise AssertionError('Duplicate seed condition')
        groups[key][c['seed']]=path
        inputs.extend([path,path.parent/'measurements.npz'])
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    bind_run(out,inputs,vars(a))
    records=[];raw={};paired=[]
    for (n,g,t,shared,stream),lookup in sorted(groups.items()):
        for count in [4,8,16]:
            seeds=list(range(640101,640101+count))
            if not all(seed in lookup for seed in seeds):continue
            available={precision:all(precision in [r['precision'] for r in
                json.loads(lookup[seed].read_text())['precision_records']] for seed in seeds)
                for precision in a.precisions}
            precision_fields={}
            for precision in a.precisions:
                if not available[precision]:continue
                data=defaultdict(list)
                for seed in seeds:
                    with np.load(lookup[seed].parent/'measurements.npz') as z:
                        names=['fields','heads','centroids','energies','mean_metric','context_kl']
                        if precision+'_evaluation_mean_metric' in z.files:names.append('evaluation_mean_metric')
                        for name in names:data[name].append(z[precision+'_'+name].astype(float))
                data={name:np.stack(values) for name,values in data.items() if len(values)==count}
                fields=collective_fields(data,count,n)
                precision_fields[precision]=fields
                c=data['centroids']
                record=dict(precision=precision,heads=n,multiplier=g,step=t,seeds=seeds,seed_count=count,
                            shared_seed=shared,stream_seed=stream,sources=[str(lookup[seed]) for seed in seeds],observables={})
                for name,q in fields.items():
                    stats,bootstrap=whole_seed_statistics(q,n)
                    record['observables'][name]=stats
                    label=f'{precision}_N{n}_g{g}_t{t}_s{count}_c{shared}_b{stream}_{name}'
                    raw[label+'_bootstrap']=bootstrap
                    raw[label+'_cohort']=q.reshape(count,-1).mean(-1)
                o=record['observables']
                total=o['mean_metric_total']['susceptibility']
                common=o['mean_metric_common']['susceptibility']
                contrast=o['mean_metric_contrast']['susceptibility']
                np.testing.assert_allclose(total,common+contrast,rtol=1e-11,atol=1e-20)
                record['calibration_common_fraction']=common/total if total>0 else None
                one_head=float(np.var(c,axis=0,ddof=1).mean())
                collective=o['common_centroid']['susceptibility']
                record['centroid_one_head_variance']=one_head
                record['centroid_cross_head_covariance']=(collective-one_head)/(n-1)
                record['centroid_effective_head_count']=n*one_head/collective if collective>0 else None
                if 'evaluation_mean_metric_total' in o:
                    whole=o['evaluation_mean_metric_total']['susceptibility']
                    ec=o['evaluation_mean_metric_common']['susceptibility']
                    er=o['evaluation_mean_metric_contrast']['susceptibility']
                    np.testing.assert_allclose(whole,ec+er,rtol=1e-11,atol=1e-20)
                    record['evaluation_common_fraction']=ec/whole if whole>0 else None
                record['mean_context_kl']=float(data['context_kl'].mean())
                records.append(record)
                print(precision,n,g,t,count,'common',o['common_centroid']['susceptibility'],
                      'contrast',contrast,'fraction',record['calibration_common_fraction'],flush=True)
            if 'float32' in precision_fields and 'float64' in precision_fields:
                paired.append(dict(heads=n,multiplier=g,step=t,seeds=seeds,seed_count=count,
                    shared_seed=shared,stream_seed=stream,
                    observables=arithmetic_comparison(precision_fields['float32'],precision_fields['float64'],n)))
    result=dict(schema='native-collective-scaling-analysis-v2',status='complete_for_declared_snapshot',conditions=records,paired_arithmetic=paired,
        conditional_contexts=512,calibration_mean_matrix_contexts=64,evaluation_mean_matrix_contexts=512,
        arithmetic_scope='Paired float32/float64 forward programs on exactly the same saved float32 parameters. The empirical variance-difference bound does not certify error relative to exact real arithmetic.',
        matrix_scope='Calibration and evaluation matrix laws have separate names. Evaluation matrices are included only where every selected identity was measured.', 
        normalization='Fixed native column coordinates and mean-square-entry matrix units; all seed variances use the unbiased divisor.',
        interpretation='Linear contrast, quadratic row energy, common-row covariance, and predictive fields are distinct observables. Their scaling exponents need not agree. Common covariance alone does not identify intrinsic criticality.',
        uncertainty='Whole-initialization empirical resampling and delete-one diagnostics, conditioning on fixed shared initialization, complete batch history and contexts.')
    write_json(out/'results.json',result)
    np.savez_compressed(out/'measurements.npz',**raw)
    write_json(out/'manifest.json',dict(status='complete',binding_sha256=sha256(out/'binding.json'),
        results_sha256=sha256(out/'results.json'),raw_sha256=sha256(out/'measurements.npz'),conditions=len(records)))


if __name__=='__main__':main()
