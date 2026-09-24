#!/usr/bin/env python3
"""Reconstruct every context variance from a portable complete observation package."""
import argparse
from pathlib import Path
import time
import numpy as np
from model_rg.provenance import sha256, write_json
from numerical_validation import load_json_strict, discrepancy, array_discrepancy
from observed_package_contract import inspect_package
from analyze_context_risk import pairs, context_metrics


def reproduce(package, output):
    if output.exists():raise FileExistsError(output)
    started=time.perf_counter();info=inspect_package(package)
    if not info['runnable']:raise ValueError('A complete observation package is required')
    index=load_json_strict((package/'INDEX.json').read_text())
    cases=index.get('context_risk',[])
    if not cases:raise ValueError('The complete package has no context-risk record')
    if len({x['family'] for x in cases})!=len(cases):raise ValueError('Duplicate context-risk family')
    if any(x['expected'] not in index['files'] for x in cases):raise ValueError('Context-risk record absent from the contained package index')
    with np.load(package/'reference.npz') as data:reference=data['reference'].mean(0).astype(float)
    if not np.all(np.isfinite(reference)) or not np.all(reference>0):raise ValueError('Invalid reference')
    order=np.argsort(-reference,kind='stable');cells=contexts=0;error=0.
    for item in cases:
        expected=load_json_strict((package/item['expected']).read_text())
        lookup={(r['heads'],r['control'],r['retained_tokens']):r for r in expected['rows']}
        keys={(n,g,k) for n in [8,24] for g in [0,1.5] for k in index['sizes']}
        if len(expected['rows'])!=20 or set(lookup)!=keys:raise ValueError('Risk-cell inventory differs')
        members=[x for x in index['observations'] if x['family']==item['family']]
        if len(members)!=24:raise ValueError('Context checkpoint inventory differs')
        for n,g in [(8,0),(8,1.5),(24,0),(24,1.5)]:
            group=sorted([x for x in members if x['heads']==n and x['control']==g],key=lambda x:x['seed'])
            if len(group)!=6 or len({x['seed'] for x in group})!=6:raise ValueError('Replica identities differ')
            probabilities=[]
            for member in group:
                with np.load(package/member['path']) as raw:z=raw['logits'].astype(float)
                if z.shape!=(item['contexts'],len(reference)) or not np.isfinite(z).all():raise ValueError('Invalid logits')
                u=np.exp(z-z.max(-1,keepdims=True));probabilities.append(u/u.sum(-1,keepdims=True))
            p=np.stack(probabilities);native=2*np.sqrt(p);vp=pairs(native)
            for k in index['sizes']:
                q=p.copy();tail=order[k:]
                q[:,:,tail]=p[:,:,tail].sum(-1,keepdims=True)*reference[tail]/reference[tail].sum()
                lifted=2*np.sqrt(q);ve=pairs(native-lifted);vq=pairs(lifted)
                actual=context_metrics(vp,ve,.25);row=lookup[n,g,k]
                if row['contexts']!=item['contexts'] or row['replicas']!=6:raise ValueError('Risk normalization differs')
                for key,values in [('native_variance',vp),('residual_variance',ve),('per_context_rms',actual['per_context_rms'])]:
                    error=max(error,array_discrepancy(values,row[key],key,atol=3e-12))
                for key in ['aggregate_rms','contexts_over_target','weighted_mass_over_target','weighted_mass_markov_bound']:
                    error=max(error,discrepancy(actual[key],row[key],key,atol=3e-12))
                error=max(error,discrepancy(float(vq.sum()/vp.sum()),row['aggregate_retention'],'retained variance',atol=3e-12))
                cells+=1;contexts+=item['contexts']
    write_json(output,dict(schema='portable-context-risk-v1',status='passed',cells=cells,
        context_resolution_cells=contexts,maximum_absolute_error=error,
        input_index_sha256=sha256(package/'INDEX.json'),source_sha256=sha256(__file__),
        native_forward_calls=0,training_updates=0,elapsed_seconds=time.perf_counter()-started,
        scope='Relative package members only; every per-context variance and risk is reconstructed from logits.'))
    print(cells,'context-risk cells and',contexts,'context/resolution values reproduced; maximum error',error)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();reproduce(a.package.resolve(),a.output)
