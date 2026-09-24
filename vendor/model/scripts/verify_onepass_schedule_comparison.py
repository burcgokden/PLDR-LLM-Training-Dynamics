#!/usr/bin/env python
"""Independently reduce the matched schedule contrasts of checked upstream fields."""
import argparse
from fractions import Fraction as F
import itertools
import json
from numerical_claims import finite_greater
from numerical_validation import load_json_strict
import math
from pathlib import Path

import numpy as np

from model_rg.provenance import sha256, write_json


KEYS = ('schedule_horizon','recipe','heads','step','shared_seed','stream_seed')


def average(values):
    values = list(values)
    return math.fsum(values)/len(values)


def percentiles(values):
    ordered = sorted(values)
    answer = []
    for probability in (F(1,40),F(39,40)):
        position = probability*(len(ordered)-1)
        lower = position.numerator//position.denominator
        fraction = position-lower
        answer.append(float((1-fraction)*F.from_float(float(ordered[lower]))
                            +fraction*F.from_float(float(ordered[min(lower+1,len(ordered)-1)]))))
    return answer


def mean_reference(left, right):
    if len(left)!=4 or len(right)!=4:
        raise AssertionError('Four paired identities are required')
    delta = [x-y for x,y in zip(left,right,strict=True)]
    boot = [average(delta[i] for i in draw) for draw in itertools.product(range(4),repeat=4)]
    return dict(scheduled_values=left,constant_values=right,scheduled_mean=average(left),constant_mean=average(right),
        difference=dict(seed_values=delta,selected_seeds=4,defined_seeds=4,mean=average(delta),
            empirical_percentiles=percentiles(boot),leave_one_out=[average(delta[j] for j in range(4) if j!=i) for i in range(4)]),
        negative_differences=sum(v<0 for v in delta),positive_differences=sum(v>0 for v in delta),exact_ties=sum(v==0 for v in delta))


def susceptibility_reference(left,right,left_boot,right_boot):
    if len(left_boot)!=256 or len(right_boot)!=256:
        raise AssertionError('Ordered resample coverage changed')
    delta = [float(x)-float(y) for x,y in zip(left_boot,right_boot,strict=True)]
    leave = [x-y for x,y in zip(left['leave_one_out'],right['leave_one_out'],strict=True)]
    if len(leave)!=4:
        raise AssertionError('A leave-one-identity contrast is missing')
    center = average(leave)
    return dict(scheduled=left['susceptibility'],constant=right['susceptibility'],
        difference=left['susceptibility']-right['susceptibility'],empirical_percentiles=percentiles(delta),
        negative_resample_fraction=sum(v<0 for v in delta)/256,
        positive_resample_fraction=sum(v>0 for v in delta)/256,
        exact_zero_resample_fraction=sum(v==0 for v in delta)/256,
        leave_one_out=leave,jackknife_se=math.sqrt(.75*math.fsum((v-center)**2 for v in leave))),delta


def equal(actual, expected, scale, name='root'):
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or set(actual)!=set(expected):
            raise AssertionError('Changed statistic fields: '+name)
        for key,value in expected.items():
            equal(actual[key],value,scale,name+'.'+key)
    elif isinstance(expected,list):
        if not isinstance(actual,list) or len(actual)!=len(expected):
            raise AssertionError('Changed selected identities: '+name)
        for i,(x,y) in enumerate(zip(actual,expected,strict=True)):
            equal(x,y,scale,name+'.'+str(i))
    elif isinstance(expected,(int,str,bool)) or expected is None:
        if actual!=expected:
            raise AssertionError('Changed identity or count: '+name)
    elif not math.isfinite(actual) or not math.isfinite(expected) or finite_greater(abs(actual-expected), 1e-11*max(abs(expected),scale,1e-280), 'scripts/verify_onepass_schedule_comparison.py:76'):
        raise AssertionError('Independent paired reduction differs: '+name)


def qualify(output, repo):
    from analyze_onepass_schedule_comparison import paired_mean, paired_susceptibility
    for left,right in [([0.,0.,0.,1.],[0.,0.,0.,1.]),([1.,2.,3.,8.],[2.,4.,1.,5.]),
                       ([1e-20,3e-20,5e-20,7e-20],[2e-20,2e-20,4e-20,6e-20])]:
        equal(paired_mean(left,right),mean_reference(left,right),max(abs(x-y) for x,y in zip(left,right)))
    def variance(values):
        center=average(values)
        return 4*math.fsum((v-center)**2 for v in values)/(len(values)-1)
    def summaries(values):
        return dict(susceptibility=variance(values),
            leave_one_out=[variance([v for j,v in enumerate(values) if j!=i]) for i in range(4)]),[
                variance([values[i] for i in draw]) for draw in itertools.product(range(4),repeat=4)]
    for left,right in [([0.,0.,0.,1.],[0.,0.,0.,1.]),([0.,1.,4.,9.],[1.,-1.,3.,9.])]:
        a,x=summaries(left);b,y=summaries(right)
        expected,delta=susceptibility_reference(a,b,x,y)
        actual,raw=paired_susceptibility(a,b,x,y)
        equal(actual,expected,max(map(abs,delta)))
        np.testing.assert_array_equal(raw,np.asarray(delta))
    values=[0.,1.,4.,9.]
    correct=mean_reference(values,values);wrong=mean_reference(values,list(reversed(values)))
    if correct['difference']['empirical_percentiles']==wrong['difference']['empirical_percentiles']:
        raise AssertionError('Discarded mean pairing was not detected')
    a,x=summaries(values);b,y=summaries(list(reversed(values)))
    correct,_=susceptibility_reference(a,a,x,x);wrong,_=susceptibility_reference(a,b,x,y)
    if correct['empirical_percentiles']==wrong['empirical_percentiles']:
        raise AssertionError('Discarded covariance pairing was not detected')
    write_json(output,dict(status='passed',fixtures=5,pairing_distinctions=2,
        sources={name:sha256(repo/name) for name in ['scripts/analyze_onepass_schedule_comparison.py',
            'scripts/verify_onepass_schedule_comparison.py','scripts/onepass_analysis.py']},
        scope='Finite four-identity examples compare independent complete resample enumeration and detect loss of mean or susceptibility pairing. They are not native experiments or population-coverage tests.'))
    print('Paired schedule qualification passed: five fixtures and two pairing distinctions',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-feasible-20260908')
    p.add_argument('--output',required=True);p.add_argument('--qualify',action='store_true')
    args=p.parse_args();repo=Path(__file__).resolve().parents[1];study=Path(args.root).resolve()/args.study
    if Path(args.output).exists():raise FileExistsError(args.output)
    if args.qualify:
        qualify(args.output,repo);return
    checked={}
    def check(path,expected=None):
        path=Path(path).resolve();digest=sha256(path);checked[str(path)]=digest
        if expected is not None and digest!=expected:raise AssertionError('Changed schedule-comparison input: '+str(path))
    def load(path):
        check(path);return load_json_strict(Path(path).read_text())
    protocol=study/'protocols/onepass-schedule-comparison-selection.json';spec=load(protocol)
    if (spec['conditions'],spec['native_pairs'],spec['trajectory_pairs'],spec['precisions'])!=(4,16,8,['float32','float64']):
        raise AssertionError('The complete matched schedule selection is required')
    for name,digest in spec['sources'].items():check(repo/name,digest)
    for name,digest in spec['inputs_sha256'].items():check(name,digest)
    folder=study/'analysis/onepass-schedule-comparison';meta=load(folder/'manifest.json')
    if meta['status']!='complete':raise AssertionError('The comparison is incomplete')
    for name,key in [('binding.json','binding_sha256'),('results.json','results_sha256'),('measurements.npz','raw_sha256')]:
        check(folder/name,meta[key])
    binding=load(folder/'binding.json')
    for name,digest in binding['inputs'].items():check(name,digest)
    for name,digest in binding['source_files'].items():check(folder/'source'/name,digest)
    results=load(folder/'results.json')
    if (results['status'],results['conditions'],results['native_pairs'],results['trajectory_pairs'],results['precision_conditions'],results['additional_training_updates'])!=('complete',4,16,8,8,0):
        raise AssertionError('The reported inventory changed')
    upstream={}
    for name,filename in [('collectives','onepass-collective-statistics.json'),('predictions','onepass-prediction-statistics.json')]:
        proof=load(study/'verification'/filename)
        if proof['status']!='passed':raise AssertionError('An upstream reduction lacks its independent check')
        path=study/'analysis'/('onepass-'+name)/'results.json'
        check(path,proof['checked_sha256'][str(path)]);upstream[name]=load_json_strict(path.read_text())
    states={(v['case']['run_id'],v['case']['step']):v for v in upstream['predictions']['states']}
    collective={(tuple(v[k] for k in KEYS),v['precision']):v for v in upstream['collectives']['conditions']}
    if len(results['risk'])!=4 or len(results['collectives'])!=8:raise AssertionError('An output condition is missing')
    with np.load(study/'analysis/onepass-collectives/measurements.npz') as upstream_raw, np.load(folder/'measurements.npz') as raw:
        expected_names=set()
        for index,group in enumerate(spec['groups']):
            risk=results['risk'][index]
            if risk['group']!=group or set(risk['observables'])!=set(spec['risk_fields']):
                raise AssertionError('Risk pair coverage changed')
            for field in spec['risk_fields']:
                sides={}
                for side,key_name in [('scheduled','scheduled_key'),('constant','constant_key')]:
                    values=[]
                    for pair in group['pairs']:
                        state=states[pair[side],group['step']]
                        if tuple(state['case'][k] for k in KEYS)!=tuple(group[key_name]) or state['case']['seed']!=pair['seed']:
                            raise AssertionError('A risk source is not the matched checkpoint')
                        values.append(state['scalars'][field])
                    sides[side]=values
                expected=mean_reference(sides['scheduled'],sides['constant'])
                scale=max(abs(x) for x in expected['difference']['seed_values'])
                equal(risk['observables'][field],expected,scale)
            for precision_index,precision in enumerate(spec['precisions']):
                actual=results['collectives'][2*index+precision_index]
                if actual['group']!=group or actual['precision']!=precision or set(actual['observables'])!=set(spec['collective_fields']):
                    raise AssertionError('Collective pair coverage changed')
                a,b=[collective[tuple(group[k]),precision] for k in ['scheduled_key','constant_key']]
                if a['seeds']!=group['seeds'] or b['seeds']!=group['seeds']:raise AssertionError('Collective seed order changed')
                for field in spec['collective_fields']:
                    left,right=a['observables'][field],b['observables'][field]
                    names=[]
                    for k in [group['scheduled_key'],group['constant_key']]:
                        names.append(f'H{k[0]}_{k[1]}_N{k[2]}_t{k[3]}_c{k[4]}_b{k[5]}_{precision}_{field}_bootstrap')
                    expected,delta=susceptibility_reference(left,right,upstream_raw[names[0]],upstream_raw[names[1]])
                    scale=max(map(abs,delta));equal(actual['observables'][field]['susceptibility'],expected,scale)
                    mean=mean_reference(left['cohort_means'],right['cohort_means'])
                    equal(actual['observables'][field]['mean'],mean,max(map(abs,mean['difference']['seed_values'])))
                    raw_name=f'N{group["heads"]}_t{group["step"]}_{precision}_{field}_paired_bootstrap'
                    expected_names.add(raw_name);np.testing.assert_array_equal(raw[raw_name],np.asarray(delta))
                expected=dict(scheduled=a['mean_context_kl'],constant=b['mean_context_kl'],difference=a['mean_context_kl']-b['mean_context_kl'])
                equal(actual['context_dispersion'],expected,abs(expected['difference']))
        if set(raw.files)!=expected_names:raise AssertionError('A paired bootstrap array is missing or extra')
    write_json(args.output,dict(status='passed',conditions=4,native_state_pairs=16,trajectory_pairs=8,
        precision_conditions=8,risk_fields=5,collective_fields=3,paired_bootstrap_arrays=24,
        draws_per_bootstrap_array=256,additional_training_updates=0,checked_sha256=checked,
        verifier_sha256=sha256(__file__),scope='Independent paired reductions of the already independently checked full single-pass analyses. Risk means are reconstructed from their per-state records, collective differences from checked ordered resamples. This does not replay native optimization or supply independent training identities.'))
    print('Paired schedule comparison passed: four conditions, sixteen checkpoint pairs, twenty-four bootstrap arrays',flush=True)


if __name__=='__main__':main()
