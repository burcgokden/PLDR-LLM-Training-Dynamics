#!/usr/bin/env python
"""Check every reported mean, bootstrap gate, cost and calibration summary."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);a=p.parse_args();study=Path(a.study).resolve()
    dest=study/'summary-verification.json'
    if dest.exists():
        raise FileExistsError(dest)
    spec=json.loads((study/'protocol.json').read_text());data=json.loads((study/'analysis.json').read_text())
    checked={};paths={};metas={};sources={};tubes={}
    def read(p):
        checked[str(p)]=sha256(p);return json.loads(p.read_text())
    read(study/'protocol.json');read(study/'analysis.json');read(study/'verification.json')
    archive=read(study/'archive-forecast.json')['fits']
    fresh_sequences=0
    for c in spec['cases']:
        sample=study/'sampling'/(c['name']+'.npz');checked[str(sample)]=sha256(sample)
        with np.load(sample) as f:
            samples={k:f[k] for k in f.files}
        order=np.random.default_rng(c['stream_seed']).permutation(524288)
        remaining=order[65536:]
        seed=115110000+c['heads']*1000+c['corpus']*100+c['identity']
        if spec['qualification']:
            seed+=10000000
        rng=np.random.default_rng(seed)
        step=rng.choice(remaining,32*spec['horizon'],replace=False).reshape(-1,32)
        assert np.array_equal(step,samples['successor'])
        later=remaining[~np.isin(remaining,step.ravel())]
        for name,pool in [('parent',remaining),('later',later)]:
            expected=np.stack([rng.choice(pool,32*spec['horizon'],replace=False).reshape(-1,32)
                               for _ in range(48+spec['assessment'])])
            assert np.array_equal(expected,samples[name])
        signatures=[hashlib.sha256(x.tobytes()).hexdigest()
                    for name in ['parent','later'] for x in samples[name]]
        assert len(signatures)==len(set(signatures))
        old=Path(spec['parent_study'])/'runs'/c['name']/'sampling.npz'
        checked[str(old)]=sha256(old)
        with np.load(old) as f:
            inspected={hashlib.sha256(x.tobytes()).hexdigest() for x in f['branch_blocks']}
        assert not (set(signatures)&inspected)
        fresh_sequences+=len(signatures)
    for c in spec['cases']:
        for state in ['parent','successor']:
            f=study/'runs'/c['name']/state
            path=f/'paths.npy';checked[str(path)]=sha256(path);paths[c['name'],state]=np.load(path)
            metas[c['name'],state]=read(f/'results.json')
            for rep in range(2):
                sources[c['name'],state,rep]=read(f/f'fit-{rep}.json')['fits']
                tubes[c['name'],state,rep]=read(f/f'calibration-{rep}.json')['tubes']
    max_error=0.
    for row in data['means']:
        c=next(c for c in spec['cases'] if c['name']==row['case'])
        p=paths[row['case'],row['state']][48:];y=p[:,1:,0]-p[:,0,None,0]
        if row['source'] in ['current','parent_reuse']:
            state=row['state'] if row['source']=='current' else 'parent'
            mu=np.cumsum(sources[row['case'],state,row['replicate']][str(row['budget'])]['risk']['mean'])
        elif row['source']=='no_change':
            mu=np.zeros(4)
        else:
            mu=np.cumsum(archive[str(c['heads'])]['risk']['mean'])
        assert np.allclose(mu,row['prediction'],atol=1e-12,rtol=0)
        assert np.allclose(y.mean(0),row['assessment_mean'],atol=1e-12,rtol=0)
        error=float(np.max(np.abs(y.mean(0)-mu)));max_error=max(max_error,abs(error-row['maximum_error']))
        assert abs(error-row['maximum_error'])<1e-12
        ci=spec['cases'].index(c);si=['parent','successor'].index(row['state'])
        idx=np.random.default_rng(spec['bootstrap_seed']+2*ci+si).integers(len(y),size=(spec['bootstrap_samples'],len(y)))
        interval=np.quantile(np.max(np.abs(y[idx].mean(1)-mu),axis=1),[.025,.975])
        assert np.allclose(interval,row['bootstrap_interval'],atol=1e-12,rtol=0)
    for c in data['costs']:
        meta=metas[c['case'],c['state']];rep=c['replicate'];m=c['budget']
        assert abs(c['adaptation_seconds']-sum(r['seconds'] for r in meta['records'][rep*24:rep*24+m]))<1e-9
        assert abs(c['calibration_path_seconds']-sum(r['seconds'] for r in meta['records'][rep*24+16:rep*24+24]))<1e-9
        assert c['fit_seconds']==meta['fit_seconds'][str(rep)][str(m)]
        assert c['adaptation_updates']==spec['horizon']*m
        assert c['calibration_updates']==spec['horizon']*8
    for g in data['gates']:
        key=f"current_r{g['replicate']}_m{g['budget']}";candidate=g['candidate']
        mean=next(r for r in data['means'] if r['case']==g['case'] and r['state']==g['state'] and r['key']==key)
        score=next(r for r in data['comparisons'] if r['case']==g['case'] and r['state']==g['state'] and r['target']=='risk'
                   and r['scale']=='fine' and r['kind']=='budget' and r['first']==key+'__'+candidate)
        costs=[]
        for budget in [g['budget'],16]:
            c=next(c for c in data['costs'] if c['case']==g['case'] and c['state']==g['state']
                   and c['replicate']==g['replicate'] and c['budget']==budget)
            query=next(r['query_seconds'] for r in data['records'] if r['case']==g['case'] and r['state']==g['state']
                       and r['source']=='current' and r['replicate']==g['replicate'] and r['budget']==budget
                       and r['candidate']==candidate and r['target']=='risk' and r['scale']=='fine')
            costs.append(c['adaptation_seconds']+c['fit_seconds']+query)
        cheap=costs[0]<costs[1]
        assert g['cheaper']==cheap
        assert g['observed_pass']==(mean['maximum_error']<=.01 and score['mean']<=.05 and cheap)
        assert g['upper_bound_pass']==(mean['bootstrap_interval'][1]<=.01 and score['bootstrap_interval'][1]<=.05 and cheap)
    for s in data['summary']:
        means=[r for r in data['means'] if r['state']==s['state'] and r['source']=='current' and r['budget']==s['budget']]
        covered=[r for r in data['coverage'] if r['state']==s['state'] and r['source']=='current' and r['budget']==s['budget']]
        assert s['fit_cells']==len(means) and s['mean_tolerance_count']==sum(r['maximum_error']<=.01 for r in means)
        assert s['covered']==sum(r['covered'] for r in covered) and s['assessment_events']==sum(r['assessment'] for r in covered)
        assert s['mean_error_range']==[min(r['maximum_error'] for r in means),max(r['maximum_error'] for r in means)]
        assert s['endpoint_half_width_range']==[min(r['half_widths'][-1] for r in covered),max(r['half_widths'][-1] for r in covered)]
    # Every fresh calibration set and reused set has all prescribed scale enclosures.
    for r in data['coverage']:
        p=paths[r['case'],r['state']][48:];y=p[:,1:,0]-p[:,0,None,0]
        src=r['state'] if r['source']=='current' else 'parent'
        t=tubes[r['case'],src,r['replicate']][str(r['budget'])]
        diff=np.eye(4)-np.eye(4,k=-1)
        for scale,b in [('fine',np.eye(4)),('paired',np.array([[1,1,0,0],[0,0,1,1]])),('endpoint',np.ones((1,4)))]:
            b=b@diff;width=t['radius']*(abs(b)@np.array(t['scale']))
            flags=np.all(abs((y-t['mean'])@b.T)<=width+1e-12,axis=1)
            assert int(flags.sum())==r['scale_enclosures'][scale]['covered']
            assert np.allclose(width,r['scale_enclosures'][scale]['half_widths'],atol=1e-12,rtol=0)
    write_json(dest,dict(status='passed',regenerated_fresh_sequences=fresh_sequences,inspected_sequence_reuse=False,mean_predictions=len(data['means']),budget_gates=len(data['gates']),
        costs=len(data['costs']),calibrated_sets=len(data['coverage']),maximum_mean_error_discrepancy=max_error,
        inputs_sha256=checked,verifier_sha256=sha256(__file__)))
    print('Passed all mean, cost, budget and scale-image summaries',flush=True)


if __name__=='__main__':
    main()
