#!/usr/bin/env python
"""Independent raw-corpus, exact-equilibrium and single-pass identity checks."""
import argparse
import itertools
import json
from numerical_claims import finite_greater
from numerical_validation import load_json_strict
import math
from pathlib import Path
import sys
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from model_rg.provenance import sha256,write_json


def statistics(x,q):
    fractions=np.stack([np.count_nonzero(x==color,axis=(1,2))/x.shape[-1]**2 for color in range(q)],1)
    m2=np.maximum(0,q/(q-1)*np.square(fractions-1/q).sum(1))
    equal_right=(x==np.roll(x,-1,axis=2)).sum((1,2))
    equal_down=(x==np.roll(x,-1,axis=1)).sum((1,2))
    energy=-equal_right.astype(float)-equal_down
    return dict(m=np.sqrt(m2),m2=m2,m4=m2*m2,energy=energy,
        field=(q*fractions[:,0]-1)/(q-1),corr1=(q*(equal_right+equal_down)/(2*x.shape[-1]**2)-1)/(q-1))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',required=True);a=ap.parse_args()
    base=Path(a.study);output=base/'data-verification.json'
    if output.exists():raise FileExistsError(output)
    checked={}
    def check(path,expected=None):
        path=Path(path);digest=sha256(path);checked[str(path.resolve())]=digest
        if expected is not None and digest!=expected:raise ValueError('Changed data input: '+str(path))
        return digest
    qualification=load_json_strict((base/'data-qualification/verification.json').read_text())
    check(base/'data-qualification/verification.json')
    check(REPO/'scripts/potts_mc.cpp',qualification['sampler_source_sha256'])
    check(base/'sampler/potts_mc',qualification['sampler_binary_sha256'])
    checks=0;exact_cache={}
    for r in qualification['cases']:
        c=r['spec'];check(c['path'],c['sha256'])
        key=(c['q'],c['L'])
        if key not in exact_cache:
            x=np.asarray(list(itertools.product(range(c['q']),repeat=c['L']**2)),dtype='uint8').reshape(-1,c['L'],c['L'])
            exact_cache[key]=statistics(x,c['q'])
        exact=exact_cache[key];weights=np.exp(-(exact['energy']-exact['energy'].min())/c['temperature']);weights/=weights.sum()
        raw=np.memmap(c['path'],mode='r',dtype=np.uint8,shape=tuple(c['shape']))
        observed=statistics(raw.reshape(-1,c['L'],c['L']),c['q'])
        for k,record in r['checks'].items():
            means=observed[k].reshape(c['chains'],-1).mean(1)
            expected=float(weights@exact[k]);point=float(means.mean());se=float(means.std(ddof=1)/math.sqrt(len(means)))
            tolerance=max(7*se,.025 if k=='energy' else .002)
            if finite_greater(abs(expected-record['exact']), 1e-10, 'scripts/verify_physical_data.py:51') or finite_greater(abs(point-record['observed']), 1e-10, 'scripts/verify_physical_data.py:51'):
                raise ValueError('Equilibrium reconstruction differs')
            if finite_greater(abs(expected-point), tolerance, 'scripts/verify_physical_data.py:53'):raise ValueError('Equilibrium qualification failed')
            checks+=1
    counts={};seeds=set()
    for folder in ['development-data','training-data','assessment-data','precision-reference']:
        p=base/folder/'manifest.json';check(p);m=load_json_strict(p.read_text())
        if m['status']!='complete':raise ValueError('Incomplete physical corpus')
        counts[folder]=0
        for c in m['cells']:
            check(c['path'],c['sha256'])
            if c['seed'] in seeds:raise ValueError('Source cohorts reuse a random seed')
            seeds.add(c['seed'])
            if Path(c['path']).stat().st_size!=int(np.prod(c['shape'])):raise ValueError('Configuration count mismatch')
            if finite_greater(abs(c['temperature']*math.log1p(math.sqrt(c['q']))-c['temperature_ratio']), 1e-14, 'scripts/verify_physical_data.py:65'):
                raise ValueError('Temperature normalization mismatch')
            counts[folder]+=c['chains']*c['samples_per_chain']
    primary=load_json_strict((base/'confirmation/protocol.json').read_text());check(base/'confirmation/protocol.json')
    check(base/'confirmation/draws.npz',primary['draws_sha256']);draws=np.load(base/'confirmation/draws.npz')
    ds=load_json_strict((base/'training-data/manifest.json').read_text())
    if draws['index'].shape!=(16384,32) or len(draws['site'])!=16384:raise ValueError('Wrong primary exposure')
    used={}
    for c in ds['cells']:
        mask=draws['cell']==c['id'];ids=draws['index'][mask].ravel();used[c['id']]=set(ids.tolist())
        if len(set(ids.tolist()))!=len(ids):raise ValueError('Repeated primary configuration identity')
        if ids.min()<0 or ids.max()>=c['chains']*c['samples_per_chain']:raise ValueError('Invalid physical sample index')
        if np.any((draws['site'][mask]<0)|(draws['site'][mask]>=c['L']**2)):raise ValueError('Invalid proper-prefix target')
    check(base/'mechanisms/protocol.json');p=load_json_strict((base/'mechanisms/protocol.json').read_text())
    check(base/'mechanisms/draws.npz',p['draws_sha256']);future=np.load(base/'mechanisms/draws.npz')
    for c in ds['cells']:
        ids=future['index'][future['cell']==c['id']].ravel()
        if len(set(ids.tolist()))!=len(ids) or used[c['id']]&set(ids.tolist()):
            raise ValueError('Finite response repeats consumed physical identities')
    validation=load_json_strict((base/'collective-validation/protocol.json').read_text());check(base/'collective-validation/protocol.json')
    for c in validation['cells']:
        if c['seed'] in seeds:raise ValueError('Collective validation reuses a source seed')
        seeds.add(c['seed']);check(c['path'],c['sha256'])
    counts['collective-validation']=sum(c['chains']*c['samples_per_chain'] for c in validation['cells'])
    # Reconstruct critical source chain moments from raw configurations with an
    # independently expressed magnetization and right/down bond statistic.
    reference=load_json_strict((base/'precision-reference/manifest.json').read_text())
    analysis=load_json_strict((base/'precision-analysis/analysis.json').read_text());check(base/'precision-analysis/analysis.json')
    max_difference=0.
    for c in reference['cells']:
        raw=np.memmap(c['path'],mode='r',dtype=np.uint8,shape=tuple(c['shape']))
        o=statistics(raw.reshape(-1,c['L'],c['L']),c['q'])
        values=np.stack([o['m'],o['m2'],o['m4'],o['energy'],o['m2']*o['energy'],o['m4']*o['energy']],1)
        means=values.reshape(c['chains'],c['samples_per_chain'],6).mean(1)
        r=next(r for r in analysis['rows'] if r['id']==c['id'])
        diff=float(np.max(np.abs(means-np.asarray(r['chain_means']))));max_difference=max(diff,max_difference)
        if not np.allclose(means,r['chain_means'],rtol=2e-12,atol=2e-12):raise ValueError('Critical chain moments differ')
    write_json(output,dict(status='passed',schema='physical-data-verification-v1',
        verifier_sha256=sha256(__file__),checked_sha256=checked,exact_moment_checks=checks,
        configuration_counts=counts,primary_distinct_configurations_per_path=16384*32,
        response_fresh_configurations_per_branch=128*32,
        maximum_reference_chain_moment_difference=max_difference,
        source_sampling_units='Independent seeded chains; physical configuration identities are never deduplicated by spin pattern.'))
    print('passed',checks,'exact moment checks and all single-pass identities',flush=True)


if __name__=='__main__':main()
