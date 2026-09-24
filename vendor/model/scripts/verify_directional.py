#!/usr/bin/env python
"""Independent raw-logit, source-law, pulse and finite-sum verification.

This reducer imports neither the analyzer nor its covariance helper.
Native path integrity is additionally checked by the producer's full zero replay.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from model_rg.provenance import sha256, write_json


def verify(study,repo):
    study=Path(study);spec=json.loads((study/'protocol.json').read_text())
    result=json.loads((study/'analysis.json').read_text());status=json.loads((study/'run-status.json').read_text())
    assert status['status']=='complete' and result['protocol_sha256']==sha256(study/'protocol.json')
    checked={}
    def check(path,digest=None):
        name=str(path);actual=sha256(path)
        if digest is not None:assert actual==digest,name
        checked[name]=actual
    for name,digest in spec['sources'].items():
        check(repo/name,digest);check(study/'executed-source'/name,digest)
    for name,digest in spec['inputs'].items():check(Path(name),digest)
    old=Path(spec['parent_study'])
    with np.load(old/'data/panels.npz') as f:targets=f['evaluation'][:,64]
    assert spec['source_replicates']=={'qualification':2,'development':1,'confirmation':4}[spec['kind']]
    assert len(spec['arms'])==9 and len({a['name'] for a in spec['arms']})==9
    max_logit_error=0.;max_identity_error=0.;logit_coordinates=0;pulses=0
    primary={(r['case'],r['family'],r['source']):r for r in result['primary']}
    assert len(primary)==len(spec['cases'])*2*spec['source_replicates']
    chronological={(r['case'],r['arm']):r for r in result['chronological']}
    transport={(r['case'],r['arm']):r for r in result['signed_transport']}
    def finite_cov(x,y):
        mx=np.mean(x,0);my=np.mean(y,0)
        out=np.zeros((x.shape[1],y.shape[1]))
        for i in range(len(x)):
            out+=np.outer(x[i]-mx,y[i]-my)
        return out/(len(x)-1)
    for case in spec['cases']:
        folder=study/'runs'/case['name'];meta=json.loads((folder/'results.json').read_text())
        assert meta['status']=='complete' and meta['restoration_bitwise'] and meta['zero_replay_bitwise']
        assert meta['protocol_sha256']==sha256(study/'protocol.json')
        assert meta['started_from']==json.loads((old/'runs'/case['name']/'results.json').read_text())['incoming_digest']
        assert len(meta['records'])==9*spec['source_replicates']
        with np.load(study/'sampling'/(case['name']+'.npz')) as f:
            used=f['primary'];blocks=f['blocks'];seed=int(f['seed'])
        order=np.random.default_rng(case['stream_seed']).permutation(524288)
        assert np.array_equal(used,order[:65536])
        rng=np.random.default_rng(seed)
        assert np.array_equal(blocks,np.stack([rng.choice(order[65536:],32*spec['horizon'],replace=False).reshape(-1,32) for _ in range(spec['source_replicates'])]))
        for b in blocks:
            assert len(np.unique(b))==b.size and not np.isin(b,used).any()
        q={a['name']:[] for a in spec['arms']}
        for r in meta['records']:
            file=folder/r['file'];check(file,r['sha256']);check(file,result['inputs_sha256'][str(file)])
            arm=next(a for a in spec['arms'] if a['name']==r['arm'])
            with np.load(file) as f:
                assert np.array_equal(f['blocks'],blocks[r['source']]) and np.array_equal(f['times'],spec['times'])
                assert f['losses'].shape==(spec['horizon'],) and np.isfinite(f['losses']).all()
                logits=f['logits'].astype(np.float64);logit_coordinates+=logits.size
                lp=logits-logsumexp(logits,axis=-1,keepdims=True)
                nll=-np.take_along_axis(lp,targets[None,:,None],axis=-1)[...,0]
                entropy=-np.sum(np.exp(lp)*lp,axis=-1)
                for key,value in [('nll',nll),('entropy',entropy)]:
                    error=float(np.max(np.abs(value-f[key])));max_logit_error=max(error,max_logit_error)
                    assert error<1e-10,(file,key,error)
                common=f['common_ms'];transverse=f['transverse_ms'];total=f['total_ms']
                assert np.allclose(common+transverse,total,rtol=2e-13,atol=1e-12)
                assert np.min(common)>=0 and np.min(transverse)>=0
                vector=np.concatenate((nll.mean(1)[:,None],entropy.mean(1)[:,None],
                    np.log1p(np.sqrt(common.mean((2,3)))),np.log1p(np.sqrt(transverse.mean((2,3))))),axis=1)
                assert np.allclose(vector,f['q'],rtol=1e-12,atol=1e-12)
                q[arm['name']].append(vector)
            if arm['family']:
                pulses+=1
                expected=abs(arm['multiplier'])*spec['amplitude']
                assert abs(r['displacement']['relative_norm']/expected-1)<.01
                assert r['displacement']['relative_displacement_error']<.05
        q={k:np.array(v) for k,v in q.items()}
        for family in ['generator','body']:
            large=(q[family+'-plus']-q[family+'-minus'])/2
            small=(q[family+'-halfplus']-q[family+'-halfminus'])/2
            for s in range(spec['source_replicates']):
                norm=np.sqrt(np.sum((2*small[s])**2))
                error=np.sqrt(np.sum((large[s]-2*small[s])**2))/max(norm,1e-30)
                signal=np.sqrt(np.sum(small[s]**2)/small[s].size)
                r=primary[(case['name'],family,s)]
                assert abs(r['discrepancy']-error)<1e-6 and abs(r['half_signal_rms']-signal)<1e-10
                g=spec['gate'];passed=error<=g['relative_tolerance'] and signal>g['signal_multiple']*g['absolute_floor']
                assert r['passes']==bool(passed)
        if spec['source_replicates']>1:
            x=q['zero'][:,-1];cx=finite_cov(x,x)
            for arm in spec['arms'][1:]:
                y=q[arm['name']][:,-1];e=y-x
                cy=finite_cov(y,y);ce=finite_cov(e,e);cross=finite_cov(x,e)
                err=np.max(np.abs(cy-cx-cross-cross.T-ce));max_identity_error=max(max_identity_error,float(err))
                assert err<1e-12
                r=transport[(case['name'],arm['name'])]
                assert abs(r['signed_cross_trace']-2*np.trace(cross))<1e-10
                assert abs(r['error_trace']-np.trace(ce))<1e-10
                assert abs(r['covariance_change_trace']-np.trace(cy-cx))<1e-10
                assert np.linalg.norm(cy-cx,2)<=r['bound']+1e-10
            for arm in spec['arms']:
                path=q[arm['name']][:,:,0];increments=path[:,1:]-path[:,:-1]
                c=finite_cov(increments,increments)
                endpoint=finite_cov(path[:,-1,None],path[:,-1,None])[0,0]
                assert abs(c.sum()-endpoint)<1e-12
                r=chronological[(case['name'],arm['name'])]
                assert abs(r['complete_variance']-c.sum())<1e-10
                assert abs(r['diagonal_variance']-np.trace(c))<1e-10
        for p in ['results.json','directions.json']:check(folder/p)
        print('Verified',case['name'],flush=True)
    for p in ['protocol.json','analysis.json','run-status.json','executed-source/manifest.json']:check(study/p)
    expected=len(spec['cases'])*9*spec['source_replicates']*spec['horizon']
    assert spec['scientific_updates']+spec['qualification_updates']==expected
    assert status['scientific_updates']==spec['scientific_updates'] and status['replay_updates']==len(spec['cases'])*spec['horizon']
    output=study/'verification.json'
    if output.exists():raise FileExistsError(output)
    write_json(output,dict(status='passed',kind=spec['kind'],verifier_sha256=sha256(__file__),
        cases=len(spec['cases']),arms_per_source=9,primary_cells=len(primary),pulses=pulses,
        scientific_updates=spec['scientific_updates'],qualification_updates=spec['qualification_updates'],
        replay_updates=spec['replay_updates'],logit_coordinates=logit_coordinates,
        max_logit_reduction_error=max_logit_error,max_covariance_identity_error=max_identity_error,
        checked_sha256=checked,scope='Every raw logit, observation reduction, source index, pulse norm and paired primary cell; finite covariance sums reconstructed independently. Producer zero replay checks complete native state. No assertion of a population covariance or complete formalized native derivative.'))
    print('Passed complete directional verification',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);a=p.parse_args()
    verify(a.study,Path(__file__).resolve().parents[1])

if __name__=='__main__':main()
