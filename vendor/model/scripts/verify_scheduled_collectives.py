#!/usr/bin/env python
"""Independently reconstruct all scheduled collective and arithmetic statistics."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from model_rg.provenance import sha256,write_json


DRAW = np.array(list(itertools.product(range(4),repeat=4)),dtype=int)


def statistics(q, heads):
    x=np.asarray(q,dtype=float).reshape(4,-1);center=x-x.mean(0)
    m2=float(np.mean(center*center));m4=float(np.mean(center**4))
    # Direct differences of independent seed pairs, without a centered Gram.
    distance=np.zeros((4,4))
    for i in range(4):
        for j in range(i):distance[i,j]=distance[j,i]=float(np.mean((x[i]-x[j])**2))
    boot=np.zeros(len(DRAW))
    for i in range(4):
        for j in range(i):boot+=distance[DRAW[:,i],DRAW[:,j]]*heads/12
    means=x.mean(1);center_means=means-means.mean();a=float(np.mean(center_means**2));b=float(np.mean(center_means**4))
    leave=np.array([heads*np.var(np.delete(x,i,axis=0),axis=0,ddof=1).mean() for i in range(4)])
    report=dict(mean=float(x.mean()),susceptibility=float(heads*np.var(x,axis=0,ddof=1).mean()),
        central_second=m2,central_fourth=m4,fourth_ratio=m4/m2**2 if m2 else None,
        independent_seeds=4,contexts=q.shape[1],empirical_percentiles=np.quantile(boot,[.025,.975]).tolist(),
        jackknife_se=float(np.sqrt(3/4*np.sum((leave-leave.mean())**2))),leave_one_out=leave.tolist(),
        bootstrap_samples=256,bootstrap_zero_fraction=float(np.mean(boot==0)),cohort_means=means.tolist(),
        cohort_susceptibility=float(heads*np.var(means,ddof=1)),cohort_fourth_ratio=b/a**2 if a else None,
        cohort_binder=1-b/(3*a**2) if a else None,finite_seed_gaussian_binder_mean=.4)
    return report,boot,means


def bootstrap_roundoff_allowance(q, heads):
    """Absolute comparison allowance for centered-Gram versus direct-pair sums.

    The operation-count gamma factor covers scalar accumulation of each
    centered dot product; the factor 16 covers distance and four-seed
    covariance assembly. This is a finite floating-point cross-check,
    not a certificate for the native forward relative to real arithmetic.
    """
    x=np.asarray(q,dtype=float).reshape(4,-1)
    centered=x-x.mean(0)
    count=x.shape[1]+32
    epsilon=np.finfo(np.float64).eps
    gamma=count*epsilon/(1-count*epsilon)
    scale=float(np.max(np.mean(centered*centered,axis=1)))
    return 16*heads*gamma*scale


def fields(data):
    f=data['fields'];h=data['heads'];c=data['centroids'];e=data['energies'];m=data['mean_metric'];v=data['evaluation_mean_metric']
    head_common=c.mean(-2);head_total=np.mean(c*c,axis=(-1,-2));head_shared=np.mean(head_common*head_common,axis=-1)
    answer=dict(row_native=h[...,2].mean(-1),row_reduced64=(e[...,0]/np.maximum(e[...,1],1e-30)).mean(-1),
        absolute_row_energy=e[...,0].mean(-1),total_energy=e[...,1].mean(-1),centroid_energy=head_total,
        common_centroid=head_common,head_common_energy=head_shared,
        head_residual_energy=np.mean((c-head_common[...,None,:])**2,axis=(-1,-2)),
        head_alignment=head_shared/np.maximum(head_total,1e-30),attention=h[...,0].mean(-1),operator_rms=h[...,3].mean(-1),
        prediction_entropy=f[...,24],nll=f[...,25],logit_projection=f[...,16:24])
    for name,matrix in [('mean_metric',m),('evaluation_mean_metric',v)]:
        answer[name+'_total']=matrix;answer[name+'_common']=matrix.mean(-2)
        answer[name+'_contrast']=matrix-matrix.mean(-2,keepdims=True)
    return answer


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-20260908');a=p.parse_args()
    study=Path(a.root).resolve()/a.study;folder=study/'analysis/collectives';output=study/'verification/collective-statistics.json'
    if output.exists():raise FileExistsError(output)
    checked={};cache={}

    def check(path,expected=None):
        path=Path(path).resolve();stat=path.stat();key=(str(path),stat.st_size,stat.st_mtime_ns)
        if key not in cache:cache[key]=sha256(path)
        digest=cache[key]
        if expected is not None and expected!=digest:raise AssertionError('Changed scheduled collective evidence: '+str(path))
        checked[str(path)]=digest;return digest

    def load(path):
        check(path);return json.loads(Path(path).read_text())

    def close(left,right,atol=1e-24):
        if right is None:
            if left is not None:raise AssertionError('An undefined collective statistic changed')
        else:np.testing.assert_allclose(left,right,rtol=4e-10,atol=atol)

    meta=load(folder/'manifest.json')
    if meta['status']!='complete':raise AssertionError('The complete collective analysis is required')
    for name,key in [('binding.json','binding_sha256'),('results.json','results_sha256'),('measurements.npz','raw_sha256')]:check(folder/name,meta[key])
    binding=load(folder/'binding.json')
    for name,digest in binding['inputs'].items():check(name,digest)
    for name,digest in binding['source_files'].items():check(folder/'source'/name,digest)
    selection=load(study/'protocols/regime-observation-selection.json')
    cases={(c['recipe'],c['heads'],c['seed'],c['step']):c for c in selection['cases']}
    result=load(folder/'results.json');conditions={(r['recipe'],r['heads'],r['step'],r['precision']):r for r in result['conditions']}
    arithmetic={(r['recipe'],r['heads'],r['step']):r for r in result['paired_arithmetic']}
    selected_groups=sorted({(c['recipe'],c['heads'],c['step']) for c in cases.values()})
    if len(cases)!=320 or len(conditions)!=160 or len(arithmetic)!=80 or len(selected_groups)!=80:
        raise AssertionError('A selected scheduled collective condition was omitted')
    comparisons=0;resamples=0
    with np.load(folder/'measurements.npz') as stored:
        for recipe,n,step in selected_groups:
            observed={}
            for precision in ['float32','float64']:
                group=conditions[recipe,n,step,precision]
                if group['seeds']!=list(range(640101,640105)) or (group['shared_seed'],group['stream_seed'])!=(640011,640001):
                    raise AssertionError('The conditioned collective identities changed')
                data={name:[] for name in ['fields','heads','centroids','energies','mean_metric','evaluation_mean_metric','context_kl']}
                for seed in range(640101,640105):
                    case=cases[recipe,n,seed,step];parent=study/'measurements'/case['name']
                    proof=load(study/'verification/observations'/(case['name']+'.json'))
                    if proof['status']!='passed' or proof['case']!=case:raise AssertionError('Underlying collective reconstruction is required')
                    check(parent/'measurements.npz',proof['checked_sha256'][str(parent/'measurements.npz')])
                    with np.load(parent/'measurements.npz') as z:
                        for field in data:data[field].append(z[precision+'_'+field].astype(float))
                data={k:np.stack(v) for k,v in data.items()};values=fields(data);observed[precision]=values
                if set(group['observables'])!=set(values):raise AssertionError('A collective field was omitted')
                for name,q in values.items():
                    expected,boot,cohort=statistics(q,n);reported=group['observables'][name]
                    if set(expected)!=set(reported):raise AssertionError('A collective moment or diagnostic was omitted')
                    for field,value in expected.items():close(reported[field],value,atol=max(1e-24,bootstrap_roundoff_allowance(q,n)) if field=='empirical_percentiles' else 1e-24)
                    label=f'{recipe}_N{n}_t{step}_{precision}_{name}'
                    close(stored[label+'_bootstrap'],boot,atol=max(1e-24,bootstrap_roundoff_allowance(q,n)));close(stored[label+'_cohort'],cohort)
                    comparisons+=1;resamples+=256
                c=data['centroids'];one=float(np.var(c,axis=0,ddof=1).mean())
                common=float(n*np.var(values['common_centroid'],axis=0,ddof=1).mean())
                close(group['centroid_one_head_variance'],one);close(group['centroid_cross_head_covariance'],(common-one)/(n-1))
                close(group['centroid_effective_head_count'],n*one/common if common else None)
                close(group['mean_context_kl'],float(data['context_kl'].mean()))
                for name in ['mean_metric','evaluation_mean_metric']:
                    whole=float(n*np.var(values[name+'_total'],axis=0,ddof=1).mean())
                    shared=float(n*np.var(values[name+'_common'],axis=0,ddof=1).mean())
                    contrast=float(n*np.var(values[name+'_contrast'],axis=0,ddof=1).mean())
                    close(whole,shared+contrast);close(group[name+'_common_fraction'],shared/whole if whole else None)
                del data,c,values
            for name,x in observed['float32'].items():
                y=observed['float64'][name];error=x-y
                cx=float(n*np.var(x,axis=0,ddof=1).mean());cy=float(n*np.var(y,axis=0,ddof=1).mean())
                ce=float(n*np.var(error,axis=0,ddof=1).mean());bound=float(2*np.sqrt(cy*ce)+ce)
                expected=dict(float32_susceptibility=cx,float64_susceptibility=cy,absolute_difference=abs(cx-cy),
                    centered_error_susceptibility=ce,susceptibility_error_bound=bound,
                    rms_field_difference=float(np.sqrt(np.mean(error*error))),max_field_difference=float(np.max(np.abs(error))),
                    relative_bound=bound/cy if cy else None,meets_one_percent_or_1e_minus8_absolute=bool(bound<=max(.01*cy,1e-8)))
                reported=arithmetic[recipe,n,step]['observables'][name]
                for field,value in expected.items():close(reported[field],value)
            del observed
            print('Verified scheduled collective statistics',recipe,n,step,flush=True)
    write_json(output,dict(status='passed',selected_states=320,precision_conditions=160,arithmetic_groups=80,
        field_conditions=comparisons,whole_initialization_resamples=resamples,checked_sha256=checked,verifier_sha256=sha256(__file__),
        scope='Independent coordinate, centered-moment, direct-seed-pair bootstrap, delete-one, common-head covariance and paired-arithmetic reconstruction for every selected condition and field. No statistical producer is imported. Empirical intervals do not establish population coverage or a critical scaling law.'))


if __name__=='__main__':main()
