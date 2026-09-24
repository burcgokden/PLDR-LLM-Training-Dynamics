#!/usr/bin/env python3
"""Reproduce categorical results from a relocatable observed-data package."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from analyze_categorical_scales import probabilities,reduce_panel
from model_rg.provenance import sha256,write_json
from numerical_validation import load_json_strict,discrepancy
from observed_package_contract import inspect_package
from categorical_summary_contract import summaries, compare_summaries


def reproduce(root,output):
    if output.exists():raise FileExistsError(output)
    kind=inspect_package(root)
    if not kind['runnable']:raise ValueError('Metadata only: use the complete observed-data package root')
    index=load_json_strict((root/'INDEX.json').read_text())
    for name,digest in index['files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path)!=digest:raise ValueError('Missing or changed package member '+name)
    with np.load(root/'reference.npz') as a:original=a['reference'];order=a['order']
    development=[]
    for row in index['observations']:
        if row['family']=='coarse' and row['heads']==14 and row['control']==0 and row['environment']==0:
            with np.load(root/row['path']) as a:development.append(probabilities(a['logits_2048']))
    if len(development)!=6 or np.max(np.abs(np.mean(development,axis=0)-original))>3e-12:
        raise ValueError('Development dictionary reconstruction differs')
    maximum=0.;cells=0;chains=0;series={}
    families=index.get('analysis_families',[dict(family='refinement',times=[2048,3072,4096],reference='per_context',expected='refinement-expected.json'),dict(family='context',times=[4096],reference='shared',expected='context-expected.json')])
    for item in families:
        family,times=item['family'],item['times']
        expected=load_json_strict((root/item['expected']).read_text())
        claims={(r['heads'],r['control'],r['time'],r['retained_tokens']):r for r in expected['cells']}
        groups=defaultdict(list)
        for row in index['observations']:
            if row['family']==family:groups[row['heads'],row['control']].append(row)
        seen=set();outputs=[]
        reference=original if item['reference']=='per_context' else original.mean(0)
        for (n,g),jobs in sorted(groups.items()):
            jobs.sort(key=lambda j:j['seed']);saved={t:[] for t in times}
            for job in jobs:
                with np.load(root/job['path']) as a:
                    for t in times:saved[t].append(probabilities(a['logits' if family.startswith('context') else f'logits_{t}']))
            for t in times:
                cs,es=reduce_panel(np.stack(saved[t]),reference,order,index['sizes'])
                for row in cs:
                    key=n,g,t,row['retained_tokens'];seen.add(key);expected_row=claims[key]
                    for field in ['native_variance','coarse_variance','centered_remainder_variance','relative_centered_rms','mean_emission_kl','uncentered_energy','mean_bias_energy']:
                        maximum=max(maximum,discrepancy(row[field],expected_row[field],family+' '+field,atol=3e-12))
                    outputs.append(dict(heads=n,control=g,time=t,**row));cells+=1
                chains+=sum(r['path_context_chains'] for r in es)
        if seen!=set(claims):raise ValueError('Incomplete portable reconstruction')
        compare_summaries(expected['summary'],summaries(outputs,dict(times=times,sizes=index['sizes'],target=.25)),atol=3e-12)
        series[family]=outputs
    if 'expected_scale_cells' in index and (cells!=index['expected_scale_cells'] or chains!=index['expected_kl_chains']):raise ValueError('Package coverage differs')
    write_json(output,dict(status='passed',schema='portable-categorical-reproduction-v1',
        input_index_sha256=sha256(root/'INDEX.json'),package_files=len(index['files']),
        observation_files=len(index['observations']),cells=cells,path_context_kl_chains=chains,
        maximum_absolute_error=maximum,training_updates=0,native_forward_calls=0,
        source_sha256=sha256(__file__),results=series))
    print(json.dumps(dict(status='passed',cells=cells,path_context_kl_chains=chains,maximum_absolute_error=maximum)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();reproduce(a.package.resolve(),a.output.resolve())

