#!/usr/bin/env python
"""Analyze the complete finite 2-by-2 training-environment factorial."""
import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
import time

import numpy as np

from analyze_scaling_collectives import collective_fields
from analyze_size_time import empirical_weights, whole_seed_statistics
from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


def factorial_components(values):
    """Orthogonal functional ANOVA on a uniform finite product of three indices."""
    q=np.asarray(values,dtype=float)
    if q.shape[:3]!=(2,2,4):raise ValueError('Expected shared, batch and initialization axes')
    effects={};contributions={}
    for size in range(4):
        for subset in itertools.combinations(range(3),size):
            averaged=tuple(i for i in range(3) if i not in subset)
            effect=q.mean(axis=averaged,keepdims=True) if averaged else q.copy()
            for smaller in effects:
                if set(smaller)<set(subset):effect=effect-effects[smaller]
            effects[subset]=effect
            if subset:contributions[','.join(['shared','batch','initialization'][i] for i in subset)]=float(np.mean(effect**2))
    centered=q-effects[()]
    total=float(np.mean(centered**2))
    np.testing.assert_allclose(sum(contributions.values()),total,rtol=1e-11,atol=1e-20)
    reconstructed=sum(effects.values())
    np.testing.assert_allclose(reconstructed,q,rtol=1e-11,atol=1e-14)
    return dict(total_conditional_population_variance=total,components=contributions,
                fractions={k:v/total if total>0 else None for k,v in contributions.items()})


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='critical-scaling-20260906');p.add_argument('--output',required=True)
    p.add_argument('--wait',action='store_true');a=p.parse_args();study=Path(a.root)/a.study
    protocol=study/'protocols/environment-factorial.json';spec=json.loads(protocol.read_text())
    lookup={}
    for job in spec['jobs']:
        shared=job.get('shared_seed',spec['shared_seed']);stream=job.get('stream_seed',spec['stream_seed'])
        path=study/'measurements'/f"cpu-{job['run_id']}-t8192"/'manifest.json'
        lookup[job['heads'],shared,stream,job['seed']]=path
    for n in [4,14]:
        for seed in range(640101,640105):
            lookup[n,640011,640001,seed]=study/'measurements'/f'cpu-h{n}-g1-t8192-s{seed}'/'manifest.json'
    if len(lookup)!=32:raise AssertionError('Finite factorial inventory changed')
    while not all(path.exists() for path in lookup.values()):
        if not a.wait:raise RuntimeError('Factorial frozen observations incomplete')
        time.sleep(30)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    inputs=[protocol]
    for path in lookup.values():inputs.extend([path,path.parent/'measurements.npz'])
    bind_run(out,inputs,vars(a))
    records=[];raw={}
    for n in [4,14]:
        cells=[];cell_records=[]
        for shared in [640011,650011]:
            row=[]
            for stream in [640001,650001]:
                data=defaultdict(list)
                for seed in range(640101,640105):
                    path=lookup[n,shared,stream,seed];meta=json.loads(path.read_text())
                    if meta['status']!='complete' or meta['case']['step']!=8192:
                        raise AssertionError('A factorial parent is incomplete or has changed horizon')
                    if sha256(path.parent/'measurements.npz')!=meta['raw_sha256']:raise AssertionError('A factorial field changed')
                    for key,value in dict(heads=n,shared_seed=shared,stream_seed=stream,seed=seed,multiplier=1).items():
                        if meta['case'][key]!=value:raise AssertionError('A factorial condition changed')
                    with np.load(path.parent/'measurements.npz') as z:
                        for name in ['fields','heads','centroids','energies','mean_metric','context_kl']:
                            data[name].append(z['float32_'+name].astype(float))
                data={key:np.stack(value) for key,value in data.items()}
                fields=collective_fields(data,4,n);row.append(fields)
                cell=dict(shared_seed=shared,stream_seed=stream,observables={},mean_context_kl=float(data['context_kl'].mean()))
                for name,q in fields.items():
                    stats,bootstrap=whole_seed_statistics(q,n);cell['observables'][name]=stats
                    raw[f'h{n}_c{shared}_b{stream}_{name}_bootstrap']=bootstrap
                cell_records.append(cell)
            cells.append(row)
        record=dict(heads=n,step=8192,cells=cell_records,observables={})
        for name in cells[0][0]:
            q=np.stack([np.stack([cell[name] for cell in row]) for row in cells])
            anova=factorial_components(q)
            within=float(np.var(q,axis=2,ddof=1).mean())
            from_components=sum(value for key,value in anova['components'].items() if 'initialization' in key)
            np.testing.assert_allclose(within,from_components*4/3,rtol=1e-11,atol=1e-20)
            effects=dict(shared=(q[1]-q[0]).mean(0),batch=(q[:,1]-q[:,0]).mean(0),
                         shared_batch_interaction=q[1,1]-q[1,0]-q[0,1]+q[0,0])
            comparisons={}
            for effect,value in effects.items():
                seed_means=value.reshape(4,-1).mean(-1)
                means=empirical_weights(4)@seed_means/4
                comparisons[effect]=dict(mean=float(seed_means.mean()),seed_means=seed_means.tolist(),
                    mean_empirical_percentiles=np.quantile(means,[.025,.975]).tolist(),
                    rms_fixed_field=float(np.sqrt(np.mean(value**2))))
            record['observables'][name]=dict(**anova,
                average_conditional_seed_susceptibility=n*within,paired_environment_effects=comparisons)
            raw[f'h{n}_{name}_factorial']=q
        records.append(record)
        print('Factorial',n,'complete',flush=True)
    result=dict(schema='finite-environment-factorial-v1',status='complete',conditions=records,
        axes=dict(shared_initializations=[640011,650011],batch_histories=[640001,650001],initializations=list(range(640101,640105))),
        uncertainty='Empirical intervals resample the four initialization identities together across both fixed environment axes. The two shared states and two batch histories are finite crossed choices, not four IID environments.',
        variance_scope='Functional ANOVA uses population divisors on the finite 2-by-2-by-4 product law at each fixed context and coordinate, then averages. Conditional initialization susceptibilities use divisor three and differ by the explicit factor four-thirds.',
        platform='All fields use the native CPU float32 forward and float64 matrix reductions, with 512 held-out contexts and separate 64-context calibration matrices.')
    write_json(out/'results.json',result);np.savez_compressed(out/'measurements.npz',**raw)
    write_json(out/'manifest.json',dict(status='complete',binding_sha256=sha256(out/'binding.json'),
        results_sha256=sha256(out/'results.json'),raw_sha256=sha256(out/'measurements.npz'),conditions=2,trained_paths=32))


if __name__=='__main__':main()
