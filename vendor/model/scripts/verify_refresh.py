#!/usr/bin/env python
"""Independent NumPy reconstruction from vocabulary logits to terminal scores.

Does not import project fitting, moment, calibration or scoring functions.
Native bitwise replays are separately counted in each producer record.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256, write_json


def verify(study):
    dest=study/'verification.json'
    if dest.exists():
        raise FileExistsError(dest)
    spec=json.loads((study/'protocol.json').read_text());analysis=json.loads((study/'analysis.json').read_text())
    checked={}
    def check(p,expected=None):
        p=Path(p);v=sha256(p);checked[str(p)]=v
        assert expected is None or v==expected,str(p)
        return v
    check(study/'protocol.json',analysis['protocol_sha256'])
    check(study/'scores.npz',analysis['scores_sha256']);check(study/'analysis.json')
    for path,digest in spec['inputs'].items():
        check(path,digest)
    repo=Path(__file__).resolve().parents[1]
    for path,digest in spec['sources'].items():
        check(repo/path,digest);check(study/'executed-source'/path,digest)
    for path,digest in analysis['inputs_sha256'].items():
        check(path,digest)
    old=Path(spec['parent_study'])
    with np.load(old/'data/panels.npz') as f:
        targets=f['evaluation'][:,64]
    archive=json.loads((study/'archive-forecast.json').read_text())['fits']
    raw_paths={};fit_store={};tube_store={};total_branches=0;vocabulary_coordinates=0;max_logit_error=0.
    native_updates=0;qualification_updates=0;replay_updates=0
    for case in spec['cases']:
        folder=study/'runs'/case['name'];meta=json.loads((folder/'results.json').read_text())
        check(folder/'results.json');assert meta['status']=='complete'
        assert spec['frozen_at']<meta['completed_at']
        check(folder/'successor-state.pt',meta['successor_state_sha256'])
        check(folder/'successor-transition.npz',meta['transition_sha256'])
        native_updates+=meta['scientific_updates'];qualification_updates+=meta['qualification_updates'];replay_updates+=meta['replay_updates']
        with np.load(study/'sampling'/(case['name']+'.npz')) as f:
            samples={k:f[k] for k in f.files}
        expected=np.random.default_rng(case['stream_seed']).permutation(524288)[:65536]
        assert np.array_equal(expected,samples['primary'])
        assert len(np.unique(samples['successor']))==32*spec['successor_updates']
        assert not np.isin(samples['successor'],samples['primary']).any()
        for state,sample_key in [('parent','parent'),('successor','later')]:
            f=folder/state;info=json.loads((f/'results.json').read_text())
            assert info['replay_bitwise'] and len(info['records'])==48+spec['assessment']
            check(f/'results.json',next(r['result_sha256'] for r in meta['states'] if r['state']==state))
            check(f/'paths.npy',info['paths_sha256'])
            observed=[];origin=None
            forbidden=samples['primary'] if state=='parent' else np.r_[samples['primary'],samples['successor'].ravel()]
            assert samples[sample_key].shape==(48+spec['assessment'],spec['horizon'],32)
            for r,ids in zip(info['records'],samples[sample_key]):
                assert len(np.unique(ids))==spec['horizon']*32 and not np.isin(ids,forbidden).any()
                assert np.all((ids>=0)&(ids<524288))
                b=r['branch'];expected_role=('adaptation' if b%24<16 else 'calibration') if b<48 else 'assessment'
                assert r['role']==expected_role
                if b>=48:
                    for rep in range(2):
                        for kind in ['fit','calibration']:
                            assert json.loads((f/f'{kind}-{rep}.json').read_text())['frozen_at']<r['started_at']
                check(f/r['raw'],r['sha256'])
                with np.load(f/r['raw']) as raw:
                    assert np.array_equal(raw['horizons'],spec['horizons'])
                    logits=raw['logits'].astype(float)
                    assert logits.shape==(5,32,32000)
                    z=logits-logits.max(-1,keepdims=True);lp=z-np.log(np.exp(z).sum(-1,keepdims=True))
                    risk=-lp[:,np.arange(32),targets];entropy=-np.sum(np.exp(lp)*lp,axis=-1)
                    error=max(float(np.max(np.abs(raw['nll']-risk))),float(np.max(np.abs(raw['entropy']-entropy))))
                    max_logit_error=max(max_logit_error,error);assert error<1e-10
                    assert np.isfinite(raw['losses']).all() and raw['losses'].shape==(spec['horizon'],)
                    if origin is None:
                        origin=logits[0].copy()
                    assert np.array_equal(logits[0],origin)
                    observed.append(np.stack([risk.mean(1),entropy.mean(1)],axis=1))
                    vocabulary_coordinates+=logits.size
                total_branches+=1
            paths=np.asarray(observed)
            assert np.max(np.abs(paths-np.load(f/'paths.npy')))<1e-10
            raw_paths[case['name'],state]=paths
            for rep in range(2):
                saved=json.loads((f/f'fit-{rep}.json').read_text());cal=json.loads((f/f'calibration-{rep}.json').read_text())
                assert saved['inputs']=={r['raw']:r['sha256'] for r in info['records'][rep*24:rep*24+16]}
                assert cal['inputs']=={r['raw']:r['sha256'] for r in info['records'][rep*24+16:rep*24+24]}
                check(f/f'fit-{rep}.json',cal['fit_sha256'])
                for budget in spec['budgets']:
                    fitted={}
                    for count,target in [(1,'risk'),(2,'risk_entropy')]:
                        p=paths[rep*24:rep*24+budget,:,:count]
                        x=(p[:,1:]-p[:,:-1]).transpose(0,2,1).reshape(budget,4*count)
                        mu=x.mean(0);z=x-mu;c=z.T@z/(budget-1)
                        scale=np.maximum(np.sqrt(np.diag(c)),1e-6)
                        corr=np.asarray(archive[str(case['heads'])][target]['correlation']);ridge=.001*np.diag(scale*scale)
                        covs=dict(local=.75*c+.25*np.diag(np.diag(c))+ridge,
                                  diagonal=np.diag(np.diag(c))+ridge,
                                  transferred=(.75*corr+.25*np.diag(np.diag(corr)))*np.outer(scale,scale)+ridge)
                        s=saved['fits'][str(budget)][target]
                        assert np.max(np.abs(mu-s['mean']))<1e-10
                        assert all(np.max(np.abs(cov-np.asarray(s['covariances'][key])))<1e-10 for key,cov in covs.items())
                        fitted[target]=dict(mean=mu,covariances=covs)
                    fit_store[case['name'],state,rep,budget]=fitted
                    b=np.tril(np.ones((4,4)));m=b@fitted['risk']['mean']
                    a=np.sqrt(np.maximum(np.diag(b@fitted['risk']['covariances']['local']@b.T),1e-12))
                    p=paths[rep*24+16:rep*24+24];y=p[:,1:,0]-p[:,0,None,0]
                    radius=np.max(np.abs(y-m)/a);t=cal['tubes'][str(budget)]
                    assert max(np.max(np.abs(a-t['scale'])),np.max(np.abs(m-t['mean'])),abs(radius-t['radius']))<1e-8
                    tube_store[case['name'],state,rep,budget]=(m,a,radius)
    independent_scores={};max_score_error=0.;score_coordinates=0
    saved_scores=np.load(study/'scores.npz')
    for r in analysis['records']:
        case=r['case'];state=r['state'];paths=raw_paths[case,state][48:]
        count=1 if r['target']=='risk' else 2
        x=(paths[:,1:,:count]-paths[:,:-1,:count]).transpose(0,2,1).reshape(len(paths),4*count)
        base=np.eye(4) if r['scale']=='fine' else np.array([[1,1,0,0],[0,0,1,1]]) if r['scale']=='paired' else np.ones((1,4))
        b=np.kron(np.eye(count),base)
        if r['source'] in ['current','parent_reuse']:
            src=state if r['source']=='current' else 'parent'
            fit=fit_store[case,src,r['replicate'],r['budget']][r['target']]
            mu=fit['mean'];cov=fit['covariances'][r['candidate']]
        else:
            heads=next(c['heads'] for c in spec['cases'] if c['name']==case)
            a=archive[str(heads)][r['target']];c=np.asarray(a['archive_covariance'])
            d=np.maximum(np.sqrt(np.diag(c)),1e-6);cov=.75*c+.25*np.diag(np.diag(c))+.001*np.diag(d*d)
            mu=np.zeros(len(c)) if r['source']=='no_change' else np.asarray(a['mean'])
        c=b@cov@b.T;d=x@b.T-b@mu
        values=np.linalg.slogdet(c)[1]+np.einsum('ni,ij,nj->n',d,np.linalg.inv(c),d)
        error=float(np.max(np.abs(values-saved_scores[r['score_key']])))
        max_score_error=max(max_score_error,error);assert error<2e-6
        assert abs(values.mean()-r['mean_score'])<2e-6
        independent_scores[case,state,r['target'],r['scale'],r['key']+'__'+r['candidate']]=values
        score_coordinates+=len(values)
    for r in analysis['comparisons']:
        prefix=(r['case'],r['state'],r['target'],r['scale'])
        d=independent_scores[*prefix,r['first']]-independent_scores[*prefix,r['second']]
        assert abs(d.mean()-r['mean'])<2e-6 and abs(d.std(ddof=1)/np.sqrt(len(d))-r['mcse'])<2e-6
        ci=next(i for i,c in enumerate(spec['cases']) if c['name']==r['case']);si=['parent','successor'].index(r['state'])
        inds=np.random.default_rng(spec['bootstrap_seed']+ci*2+si).integers(len(d),size=(spec['bootstrap_samples'],len(d)))
        assert np.max(np.abs(np.quantile(d[inds].mean(1),[.025,.975])-r['bootstrap_interval']))<2e-6
    for r in analysis['coverage']:
        p=raw_paths[r['case'],r['state']][48:];y=p[:,1:,0]-p[:,0,None,0]
        src=r['state'] if r['source']=='current' else 'parent'
        m,a,q=tube_store[r['case'],src,r['replicate'],r['budget']]
        flags=np.max(np.abs(y-m)/a,axis=1)<=q
        assert flags.tolist()==r['indicators'] and int(flags.sum())==r['covered']
        assert np.max(np.abs(q*a-r['half_widths']))<1e-8
    assert native_updates==spec['scientific_updates']==analysis['scientific_updates']
    assert total_branches==analysis['conditional_paths']
    write_json(dest,dict(status='passed',schema='onepass-refresh-verification-v1',qualification=spec['qualification'],
        branches=total_branches,vocabulary_coordinates=vocabulary_coordinates,score_coordinates=score_coordinates,
        comparisons=len(analysis['comparisons']),maximum_vocabulary_reduction_error=max_logit_error,
        maximum_score_error=max_score_error,scientific_updates=native_updates,qualification_updates=qualification_updates,
        replay_updates=replay_updates,checked_sha256=checked,verifier_sha256=sha256(__file__),
        scope='Every sampled path is disjoint from its consumed prefix; every stored full-vocabulary panel, fit, tube, individual score and paired score/MCSE/bootstrap interval is independently reconstructed. Native final-state replays are producer checks, not new observations.'))
    print('Passed',total_branches,'paths;',score_coordinates,'scores;',native_updates,'scientific updates',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True)
    verify(Path(p.parse_args().study).resolve())
