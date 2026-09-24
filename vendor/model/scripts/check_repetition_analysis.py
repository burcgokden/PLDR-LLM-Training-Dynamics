#!/usr/bin/env python
"""Qualify two independent paired-statistic programs before fresh-data scoring."""
import argparse
import copy
from pathlib import Path

import numpy as np

from analyze_repetition_comparison import summary,coordinates
from verify_repetition_comparison import independent_summary,compare_metric
from model_rg.provenance import sha256,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-feasible-20260908');a=p.parse_args()
    root=Path(a.root).resolve();study=root/a.study;repo=Path(__file__).resolve().parents[1]
    out=study/'qualification/repetition-analysis/verification.json'
    if out.exists():raise FileExistsError(out)
    rng=np.random.default_rng(652681);fixtures=[];inputs={};records=[]
    for s in [1,2,4]:
        x=rng.normal(size=(s,17,5));fixtures.append((f'random-s{s}',x,1.2*x+.4,14))
    x=np.ones((4,31));fixtures.append(('constant-shift',x,2*x,14))
    x=1e-100*rng.normal(size=(4,31));fixtures.append(('tiny-variance',x,2*x,4))
    x=1+1e-8*rng.normal(size=(4,512));fixtures.append(('nearly-identical',x,x[::-1].copy(),14))
    for n in [4,8,14]:
        values=[]
        for seed in range(640101,640105):
            path=root/'critical-scaling-20260906/measurements'/f'cpu-h{n}-g1-t32768-s{seed}'/'manifest.json'
            inputs[str(path)]=sha256(path);inputs[str(path.parent/'measurements.npz')]=sha256(path.parent/'measurements.npz')
            values.append(coordinates(path,'float32'))
        for name in values[0]:
            x=np.stack([q[name] for q in values]);fixtures.append((f'native-n{n}-{name}',x,x[::-1].copy()*1.01,n))
    for name,x,y,n in fixtures:
        actual=summary(x,y,n);expected=independent_summary(x,y,n)
        records.append(dict(name=name,**compare_metric(actual,expected,x,y)))
    x=fixtures[2][1];y=fixtures[2][2];n=fixtures[2][3]
    correct=summary(x,y,n);expected=independent_summary(x,y,n);mutations=[]
    for key in ['onepass_susceptibility','paired_mean','susceptibility_difference_percentiles']:
        changed=copy.deepcopy(correct)
        if isinstance(changed[key],list):changed[key][0]+=1
        else:changed[key]+=1
        try:compare_metric(changed,expected,x,y)
        except AssertionError:mutations.append(key)
        else:raise AssertionError('A corrupted paired statistic was accepted')
    if len(records)!=30 or len(mutations)!=3:raise AssertionError('Incomplete paired-statistic qualification')
    names=['scripts/check_repetition_analysis.py','scripts/analyze_repetition_comparison.py',
           'scripts/verify_repetition_comparison.py']
    write_json(out,dict(status='passed',fixtures=len(records),records=records,rejected_mutations=mutations,
        inputs_sha256=inputs,source_files={name:sha256(repo/name) for name in names},
        scope='Synthetic and existing repeated-data fields qualify paired statistics, degenerate bootstrap resamples '
        'and mutation rejection before scoring the new single-pass comparisons. No scientific target fit is changed.'))
    print('Paired-statistic qualification passed:',len(records),'fixtures and',len(mutations),'rejected mutations',flush=True)


if __name__=='__main__':main()
