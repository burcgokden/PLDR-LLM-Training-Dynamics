#!/usr/bin/env python
"""Check the complete executed evidence graph without modifying its inputs."""
import argparse
import json
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256, write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();root=Path(a.root);study=root/'controlled-study-20260905';cache={};checks=[]
    def digest(p):
        p=Path(p).resolve()
        if p not in cache:cache[p]=sha256(p)
        return cache[p]
    def bound(p,h):
        if digest(p)!=h:raise AssertionError(f'Hash mismatch: {p}')
    original=json.loads((study/'input-integrity.json').read_text())
    for name,h in original.items():bound(name,h)
    seen=set(r['content_sha256'] for r in json.loads((root/'data/refinedweb-4608/records.json').read_text()))
    cohorts={}
    for name,count,length in [('short',2048,513),('long',1024,4097)]:
        p=study/'data'/name;m=json.loads((p/'manifest.json').read_text())
        for fn,key in [('tokens.npy','tokens_sha256'),('offsets.npy','offsets_sha256'),('records.json','records_sha256')]:bound(p/fn,m[key])
        t=np.load(p/'tokens.npy');rows=json.loads((p/'records.json').read_text())
        if t.shape!=(count,length) or t.min()<0 or t.max()>=32000:raise AssertionError('Invalid token shape or vocabulary')
        hashes={r['content_sha256'] for r in rows}
        if len(hashes)!=count or hashes&seen:raise AssertionError('Cross-cohort document overlap')
        seen|=hashes
        for split,(start,end) in m['split_rows'].items():
            counts={}
            for r in rows[start:end]:counts[r['shard']]=counts.get(r['shard'],0)+1
            if len(counts)!=16 or len(set(counts.values()))!=1:raise AssertionError('Shard imbalance in split')
        cohorts[name]=dict(documents=count,length=length,unique=True,split_rows=m['split_rows'])
    native=[f'family-h{h}-s{s}' for h in [2,4,8,14] for s in [630101,630102,630103]]
    normalized=[f'normalized-h{h}-s{s}' for h in [4,8,14] for s in [630101,630102,630103]]
    fresh=[f'normalized-confirmation-h14-s{s}' for s in [630111,630112,630113]]
    expected=set(native+normalized+fresh+['long-soc1','long-soc2']+
        ['quality-'+n for n in native+normalized+fresh+['soc1','soc2']]+['drift-smooth-'+n for n in native])
    missing=[n for n in expected if not (study/'runs'/n/'manifest.json').is_file()]
    if missing:raise AssertionError(f'Missing required measurements: {missing}')
    inventory=[]
    for run in sorted((study/'runs').iterdir()):
        binding=run/'binding.json';manifest=run/'manifest.json'
        item=dict(run_id=run.name,required_for_release=run.name in expected,status='completed' if manifest.exists() else 'failed_or_incomplete_attempt')
        if binding.exists():
            b=json.loads(binding.read_text())
            for name,h in b['inputs'].items():bound(name,h)
            for name,h in b['source_files'].items():bound(run/'source'/name,h)
            item['binding_sha256']=digest(binding)
        if manifest.exists():
            m=json.loads(manifest.read_text());bound(binding,m['binding_sha256'])
            rawfile={'controlled-training-v1':'measurements.npz','conditional-drift-v1':'drift.npz',
                     'quality-controls-v1':'quality.npz','long-segments-v1':'segments.npz'}[m['schema']]
            bound(run/rawfile,m['raw_sha256']);item['manifest_sha256']=digest(manifest);item['raw_sha256']=m['raw_sha256']
            with np.load(run/rawfile) as arrays:
                for key in arrays.files:
                    if not np.isfinite(arrays[key]).all():raise AssertionError(f'Nonfinite {run.name}/{key}')
                if m['schema']=='controlled-training-v1':
                    bound(run/'final-training-state.pt',m['checkpoint_sha256']);bound(run/'sampling.npz',m['sampling_sha256'])
                    if len(m['feature_names'])!=36 or arrays['mean_fields'].shape[1]!=36:raise AssertionError('Common field mismatch')
                    sampling=np.load(run/'sampling.npz')
                    if sampling['rows'].min()<0 or sampling['rows'].max()>=3072 or sampling['offsets'].max()>448:raise AssertionError('Invalid train sampling')
                    for mode in ['reset_moments','freeze_generator','reset_and_freeze']:
                        if not np.array_equal(arrays['branch_'+mode][0],arrays['branch_continue'][0]):raise AssertionError('Intervention baseline weights differ')
                elif m['schema']=='conditional-drift-v1' and run.name in expected:
                    if arrays['q'].shape!=(16,16) or np.any(arrays['q']<=0):raise AssertionError('Missing curvature/batch/probe coverage')
                    if arrays['kl'].min() < -1e-14:raise AssertionError('Negative KL beyond resolution')
        inventory.append(item)
    for folder in ['supporting','training','long','drift-smooth','normalization','seed-confirmation','quality-all']:
        p=study/'analysis'/folder;b=json.loads((p/'binding.json').read_text());m=json.loads((p/'manifest.json').read_text())
        bound(p/'results.json',m['results_sha256']);bound(p/'binding.json',m['binding_sha256'])
        for name,h in m.get('figures',{}).items():bound(p/name,h)
        for name,h in b['inputs'].items():bound(name,h)
        for name,h in b['source_files'].items():bound(p/'source'/name,h)
        checks.append(dict(analysis=folder,results_sha256=m['results_sha256']))
    formal_dir=study/'formal-integrated'
    formal=json.loads((formal_dir/'verification.json').read_text())
    repo=Path(__file__).resolve().parents[1]
    for name,h in formal['module_sources'].items():bound(repo/name,h)
    bound(repo/'scripts/formal/Gate.lean',formal['gate_sha256'])
    bound(repo/'lake-manifest.json',formal['lake_manifest_sha256'])
    for item in formal['checks']:bound(formal_dir/(item['name']+'.log'),item['log_sha256'])
    if formal['status']!='passed' or not all(c['expectation_met'] for c in formal['checks']):raise AssertionError('Formal gate incomplete')
    report=dict(schema='controlled-release-verification-v1',status='passed',protected_inputs_verified=len(original),
        unique_fresh_cohorts=cohorts,required_scientific_runs=len(expected),inventory=inventory,analyses=checks,
        formal_explicit_theorems=formal['explicit_theorems'],hashed_files=len(cache))
    write_json(a.output,report);print(json.dumps({k:v for k,v in report.items() if k not in ['inventory','analyses']},indent=2))


if __name__=='__main__':main()
