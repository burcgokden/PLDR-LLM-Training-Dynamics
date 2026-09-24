#!/usr/bin/env python
"""Qualify finite-ensemble and drive-window analysis before full-panel execution."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import onepass_analysis as producer
from verify_onepass_collectives import statistics, allowance, close
from verify_onepass_predictions import summary, equal, tasks
from analyze_onepass_predictions import task_summary
from analyze_scaling_updates import covariance_diagnostics
from verify_onepass_paths import diagnostics, compare
from model_rg.provenance import sha256, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-feasible-20260908');a=p.parse_args()
    study=Path(a.root).resolve()/a.study;out=study/'qualification/onepass-analysis'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);repo=Path(__file__).resolve().parents[1]
    rng=np.random.default_rng(651791);fixtures=[];mutations=0
    for count in [1,2,4]:
        samples=[rng.normal(size=(count,17,5)),np.full((count,17,5),3.),
            3+1e-10*rng.normal(size=(count,17,5)),1e-14*rng.normal(size=(count,17,5))]
        for index,q in enumerate(samples):
            reported,boot,cohort=producer.statistics(q,14);expected,other,means=statistics(q,14)
            if set(reported)!=set(expected):raise AssertionError('Collective diagnostic coverage differs')
            for field,value in expected.items():
                close(reported[field],value,atol=max(1e-24,allowance(q,14)) if field in ['empirical_percentiles','susceptibility'] else 1e-24)
            close(boot,other,atol=max(1e-24,allowance(q,14)));close(cohort,means)
            fixtures.append(dict(kind='finite_ensemble',seeds=count,case=index))
        for values in [[0.]*count,[float(v) for v in rng.normal(size=count)], [None]+[1.]*(count-1)]:
            equal(producer.empirical_mean(values),summary(values))
            fixtures.append(dict(kind='mean_or_undefined',seeds=count,values=values))
    # Each corruption must be rejected, including an invented one-seed variance.
    q=rng.normal(size=(4,17,5));expected,_,_=statistics(q,14)
    for field in ['mean','susceptibility','central_fourth']:
        report=deepcopy(expected);report[field]+=1
        try:close(report[field],expected[field])
        except AssertionError:mutations+=1
        else:raise AssertionError('A deliberately corrupted statistic passed')
    try:close(0.,None)
    except AssertionError:mutations+=1
    else:raise AssertionError('An invented single-seed covariance passed')
    inputs=[];temporal=0
    qualification=study/'runs/qualification-reference1-h14-d768-continuous/measurements.npz'
    with np.load(qualification) as raw:
        arrays=[raw['shared_update_projection'],raw['shared_clipped_gradient_projection'],raw['shared_update_moments']]
        arrays.extend([np.zeros((64,3)),np.ones((64,3)),np.arange(64,dtype=float)[:,None]*np.ones((1,3))])
        for x in arrays:
            compare(covariance_diagnostics(x,1),diagnostics(x,1),'temporal')
            temporal+=1
    inputs.append(qualification)
    training=study/'protocols/onepass-training-selection.json';jobs=json.loads(training.read_text())['jobs'];windows=0
    for job in jobs:
        phases=producer.phases(job)
        if phases[0][1]!=0 or phases[-1][2]!=job['steps'] or any(x[2]!=y[1] for x,y in zip(phases,phases[1:])):
            raise AssertionError('The observed drive phases do not partition the run')
        for window in producer.windows(job,[2048,8192,32768]):
            begin,end=window['begin'],window['end'];grid=np.arange(((begin+63)//64)*64,end,64)
            if len(grid)<4:raise AssertionError('A selected temporal window has insufficient probes')
            if not any(begin>=a and end<=b and window['phase']==name for name,a,b in phases):
                raise AssertionError('A selected window crosses drive phases')
            windows+=1
    inputs.append(training)
    mechanism=study/'observer-qualification/protocols/reasoning-mechanism-selection.json'
    spec=json.loads(mechanism.read_text());cohort_path=Path(spec['cohort'])
    definitions={r['id']:r for r in json.loads(cohort_path.read_text())['examples']};panels=0
    for case in spec['cases']:
        path=study/'observer-qualification/measurements'/case['name']/'results.json'
        proof=study/'observer-qualification/verification/reasoning'/(case['name']+'.json')
        verification=json.loads(proof.read_text())
        if verification['status']!='passed' or verification['checked_sha256'][str(path)]!=sha256(path):
            raise AssertionError('The qualification mechanism result requires its independent raw proof')
        data=json.loads(path.read_text());modes={r['mode']:r for r in data['modes']}
        equal(task_summary(modes,definitions),tasks(modes,definitions),'tasks');panels+=1
        inputs.extend([path,proof])
    inputs.extend([mechanism,cohort_path])
    sources=['onepass_analysis','analyze_onepass_collectives','verify_onepass_collectives','analyze_onepass_predictions',
        'verify_onepass_predictions','analyze_onepass_paths','verify_onepass_paths','analyze_scaling_updates',
        'verify_scheduled_collectives','check_onepass_analysis']
    write_json(out/'verification.json',dict(status='passed',fixtures=fixtures,mutations_rejected=mutations,
        native_and_exact_temporal_cases=temporal,selected_windows=windows,qualification_task_panels=panels,
        inputs_sha256={str(p):sha256(p) for p in inputs},
        source_sha256={'scripts/'+n+'.py':sha256(repo/'scripts'/(n+'.py')) for n in sources},
        scope='Development qualification of independent finite-ensemble statistics, missing-value policy, temporal reductions and complete selected drive windows. It does not establish a scientific training outcome.'))
    print('Qualified',len(fixtures),'ensemble fixtures,',mutations,'mutations,',temporal,'temporal cases,',windows,'selected windows and',panels,'task panels',flush=True)


if __name__=='__main__':main()
