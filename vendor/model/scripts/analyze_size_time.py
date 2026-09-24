#!/usr/bin/env python
"""Reconstruct whole-seed size-time laws from completed native trajectories."""
import argparse
from collections import defaultdict
import itertools
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json
from model_rg.scaling import collective_moments


@lru_cache(maxsize=8)
def empirical_weights(count, samples=20000, seed=650061):
    draws = (np.array(list(itertools.product(range(count), repeat=count)),dtype=int) if count == 4
             else np.random.default_rng(seed).integers(count,size=(samples,count)))
    return np.array([(row[:,None] == np.arange(count)).sum(0) for row in draws],dtype=float)


def whole_seed_statistics(values, heads):
    q = np.asarray(values,dtype=float)
    count = len(q)
    if count < 2:
        raise ValueError('At least two initialization identities are required')
    flat = q.reshape(count,-1)
    # Center first to avoid cancellation when the mean is large and variance small.
    centered = flat-flat.mean(0)
    gram = centered@centered.T/flat.shape[1]
    distance = np.maximum(np.diag(gram)[:,None]+np.diag(gram)[None,:]-2*gram,0)
    np.fill_diagonal(distance,0)
    stats = collective_moments(q,heads)
    weights = empirical_weights(count)
    bootstrap = heads*np.einsum('bi,ij,bj->b',weights,distance,weights)/(2*count*(count-1))
    direct = heads*np.var(flat,axis=0,ddof=1).mean()
    if not np.isclose(direct,stats['susceptibility'],rtol=1e-12,atol=1e-30):
        raise AssertionError('Variance reconstruction changed')
    if count > 2:
        leave = np.array([heads*np.var(np.delete(flat,i,axis=0),axis=0,ddof=1).mean() for i in range(count)])
        jackknife = float(np.sqrt((count-1)/count*np.sum((leave-leave.mean())**2)))
    else:
        leave, jackknife = np.array([]), None
    cohort = flat.mean(-1)
    d = cohort-cohort.mean()
    m2, m4 = np.mean(d*d), np.mean(d**4)
    stats.update(empirical_percentiles=np.quantile(bootstrap,[.025,.975]).tolist(),
                 jackknife_se=jackknife,leave_one_out=leave.tolist(),
                 bootstrap_samples=len(weights), bootstrap_zero_fraction=float(np.mean(bootstrap==0)),
                 cohort_means=cohort.tolist(),cohort_susceptibility=float(heads*np.var(cohort,ddof=1)),
                 cohort_fourth_ratio=float(m4/m2**2) if m2>0 else None,
                 cohort_binder=float(1-m4/(3*m2**2)) if m2>0 else None,
                 finite_seed_gaussian_binder_mean=2/(count+1))
    return stats, bootstrap


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--study',default='critical-scaling-20260906')
    p.add_argument('--output',required=True)
    p.add_argument('--include-current',action='store_true')
    a = p.parse_args()
    root = Path(a.root)
    studies = ['criticality-study-20260905','criticality-dynamics-20260906','observation-closure-20260906']
    if a.include_current:
        studies.append(a.study)
    candidates = defaultdict(list)
    inspected = []
    for study in studies:
        for path in sorted((root/study/'runs').glob('*/manifest.json')):
            meta = json.loads(path.read_text())
            if meta.get('schema') not in ['criticality-training-v1','criticality-dynamics-training-v1','native-size-time-training-v1']:
                continue
            condition = meta['arguments']
            if (meta['status'] != 'complete' or condition['seed'] not in range(640101,640117)
                or condition.get('stream_seed') != 640001 or condition.get('shared_seed') != 640011
                or (condition.get('normalization','fan_in') != 'variance' and condition['heads'] != 2)):
                continue
            inspected.append(path)
            for step in meta['milestones']:
                key = (condition['heads'],condition['multiplier'],int(step),condition['seed'])
                candidates[key].append((meta['completed_step'],str(path.parent),path))
    selected = {key:sorted(values,reverse=True)[0][2] for key,values in candidates.items()}
    groups = defaultdict(dict)
    for (n,g,t,seed),path in selected.items():
        groups[n,g,t][seed] = path
    inputs = list(dict.fromkeys([p for p in selected.values()]+[p.parent/'measurements.npz' for p in selected.values()]))
    out = Path(a.output)
    out.mkdir(parents=True,exist_ok=False)
    bind_run(out,inputs,vars(a))
    records, raw = [], {}
    for (n,g,t),seed_paths in sorted(groups.items()):
        for count in [4,8,16]:
            seeds = list(range(640101,640101+count))
            if not all(seed in seed_paths for seed in seeds):
                continue
            heads, fields, sources = [], [], []
            for seed in seeds:
                path = seed_paths[seed]
                with np.load(path.parent/'measurements.npz') as z:
                    h,f = z[f'heads_{t}'].astype(float), z[f'fields_{t}'].astype(float)
                if h.shape != (512,5,n,4) or f.shape != (512,36):
                    raise AssertionError('Evaluation cohort changed')
                heads.append(h)
                fields.append(f)
                sources.append(str(path))
            h,f = np.stack(heads),np.stack(fields)
            observations = dict(row=h[...,2].mean(-1),attention=h[...,0].mean(-1),
                operator_rms=h[...,3].mean(-1),prediction_entropy=f[...,24],nll=f[...,25],
                logit_projection=f[...,16:24])
            row = dict(heads=n,multiplier=g,step=t,seeds=seeds,seed_count=count,sources=sources,observables={})
            for name,q in observations.items():
                stats,bootstrap = whole_seed_statistics(q,n)
                row['observables'][name] = stats
                raw[f'N{n}_g{g}_t{t}_s{count}_{name}_bootstrap'] = bootstrap
                raw[f'N{n}_g{g}_t{t}_s{count}_{name}_cohort'] = q.reshape(count,-1).mean(-1)
            records.append(row)
            print(n,g,t,count,'row chi',row['observables']['row']['susceptibility'],flush=True)
    # Width slopes are explicitly finite diagnostics; no critical control is assumed.
    by_condition = defaultdict(list)
    for row in records:
        by_condition[row['multiplier'],row['step'],row['seed_count']].append(row)
    slopes = []
    for (g,t,count),items in sorted(by_condition.items()):
        if len(items)<3:
            continue
        for name in ['row','attention','operator_rms','prediction_entropy','nll']:
            for cutoff in [2,4,8]:
                rows = [r for r in items if r['heads']>=cutoff and r['observables'][name]['susceptibility']>0]
                if len(rows)<3:
                    continue
                x=np.log([r['heads'] for r in rows])
                y=np.log([r['observables'][name]['susceptibility'] for r in rows])
                slope,intercept=np.polyfit(x,y,1)
                slopes.append(dict(multiplier=g,step=t,seed_count=count,observable=name,minimum_heads=cutoff,
                    heads=[r['heads'] for r in rows],slope=float(slope),log_rmse=float(np.sqrt(np.mean((y-intercept-slope*x)**2)))))
    results = dict(schema='whole-seed-size-time-analysis-v1',status='complete_for_declared_snapshot',
        arguments=vars(a),conditions=records,width_slope_diagnostics=slopes,
        source_selection='Largest completed horizon, then lexical run path, for duplicate observations of one identity and step.',
        statistical_scope='Conditional fixed-cohort whole-seed empirical moments. Finite empirical percentiles have no guaranteed population coverage.',
        moment_scope='Primary susceptibility averages conditional seed variances at fixed context and coordinate. Cohort Binder uses one fixed cohort mean per seed; its Gaussian finite-seed bias is displayed.',
        inference_scope='Predictive fields and native metric coordinates have their declared units. A growing N times predictive variance need not identify intrinsic criticality.',
        exponent_scope='Width slopes are diagnostics only: no independently identified critical surface or controlled asymptotic scaling has been assumed.')
    write_json(out/'results.json',results)
    np.savez_compressed(out/'measurements.npz',**raw)
    write_json(out/'manifest.json',dict(status='complete',binding_sha256=sha256(out/'binding.json'),
        raw_sha256=sha256(out/'measurements.npz'),results_sha256=sha256(out/'results.json'),conditions=len(records)))


if __name__ == '__main__':
    main()
