#!/usr/bin/env python
"""Independently reconstruct sampling, vocabulary moments, fits and all scores.

The verifier does not import the producer, fitting routine, or moment scorer.
Its score uses an eigensystem; the analysis uses a Cholesky factorization.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp
from model_rg.provenance import sha256,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();study=Path(a.study).resolve();out=Path(a.output).resolve();repo=Path(__file__).resolve().parents[1]
    if out.exists():raise FileExistsError(out)
    start=time.time();checked={};maxima={};logits_count=0;primary_logits_count=0
    def bound(path,digest=None):
        path=Path(path).resolve();key=str(path)
        if key not in checked:checked[key]=sha256(path)
        assert digest is None or checked[key]==digest, key
        return checked[key]
    def read(path):bound(path);return json.loads(Path(path).read_text())
    def close(key,x,y,tol=1e-8):
        x=np.asarray(x);y=np.asarray(y);assert x.shape==y.shape,key
        err=float(np.max(abs(x-y))) if x.size else 0.;maxima[key]=max(maxima.get(key,0.),err)
        assert np.isfinite(x).all() and np.isfinite(y).all() and err<=tol,(key,err)
    spec=read(study/'protocol.json');analysis=read(study/'analysis.json');archive=read(study/'archive-forecast.json')
    assert spec['primary_steps']==2048 and spec['branches']==dict(adaptation=16,calibration=8,assessment=32)
    assert analysis['status']=='complete' and analysis['assessment_paths']==256
    for name,digest in spec['sources'].items():bound(repo/name,digest)
    for name,digest in spec['inputs'].items():bound(name,digest)
    for name,digest in analysis['inputs_sha256'].items():bound(name,digest)
    bound(repo/'scripts/analyze_outer_transfer.py',analysis['analyzer_sha256'])
    bound(study/'scores.npz',analysis['scores_sha256'])
    scores_file=np.load(study/'scores.npz')
    data=read(study/'data/manifest.json')
    for name,digest in data['files'].items():bound(study/'data'/name,digest)
    panels=read(study/'data/panel-records.json')
    eval_hashes={r['content_sha256'] for r in panels['evaluation']};donor_hashes={r['content_sha256'] for r in panels['donors']}
    assert len(eval_hashes)==len(donor_hashes)==32 and not eval_hashes&donor_hashes
    corpus_sets=[]
    for c in range(2):
        records=read(study/'data'/f'records-{c}.json');content={r['content_sha256'] for r in records}
        assert len(records)==len(content)==65536 and not content&(eval_hashes|donor_hashes)
        assert all(sum(r['stratum']==s for r in records)==4096 for s in range(16))
        corpus_sets.append(content)
        arr=np.load(study/'data'/f'corpus-{c}.npy',mmap_mode='r')
        assert arr.shape==(65536,513) and arr.dtype==np.int32
    assert len(corpus_sets[0]&corpus_sets[1])==data['cross_corpus_overlap']
    reference=read(next(Path(k) for k in spec['inputs'] if k.endswith('moment-transport-20260911/frozen-predictions.json')))
    for heads in [4,14]:
        states=[v for k,v in reference['fits'].items() if k.startswith(f'h{heads}-')]
        for target in ['risk','risk_entropy']:
            cs=[np.array(s['targets'][target]['covariance']) for s in states]
            corr=np.mean([c/np.sqrt(np.outer(np.diag(c),np.diag(c))) for c in cs],axis=0)
            close('archive_correlation',corr,archive['fits'][str(heads)][target]['correlation'],1e-12)
            close('archive_mean',np.mean([s['targets'][target]['mean'] for s in states],axis=0),archive['fits'][str(heads)][target]['mean'],1e-12)
    qualifications=read(study/'qualification-launcher.json');assert qualifications['status']=='complete'
    for rec in qualifications['records']:
        bound(rec['result'],rec['sha256']);result=read(rec['result'])
        assert result['gradient_qualification']['all_gradients_bitwise'] and result['gradient_qualification']['loss_bitwise']
        assert result['restoration_bitwise'] and result['replay_bitwise']
        assert result['qualification_updates']==264 and result['replay_updates']==4
    launcher=read(study/'launcher.json');assert launcher['status']=='complete' and len(launcher['records'])==8
    components={};case_paths=[]
    for case in spec['cases']:
        folder=study/'runs'/case['name'];meta=read(folder/'results.json')
        for name,digest in meta['files'].items():bound(folder/name,digest)
        launch=next(x for x in launcher['records'] if x['case']==case['name']);bound(folder/'results.json',launch['sha256'])
        assert meta['case']==case and meta['scientific_primary_updates']==2048 and meta['scientific_branch_updates']==3584
        assert meta['replay_bitwise'] and meta['restoration_bitwise'] and meta['replay_updates']==64
        components[case['name']]=meta['initial_component_sha256']
        fit=read(folder/'adapted-forecast.json');tube=read(folder/'calibration.json')
        assert len(meta['records'])==56
        assert fit['inputs']=={r['raw']:r['sha256'] for r in meta['records'][:16]}
        assert tube['inputs']=={r['raw']:r['sha256'] for r in meta['records'][16:24]}
        assert tube['fit_sha256']==bound(folder/'adapted-forecast.json')
        assert fit['protocol_sha256']==bound(study/'protocol.json')
        assert datetime.fromisoformat(spec['frozen_at'])<datetime.fromisoformat(meta['started_at'])<datetime.fromisoformat(fit['frozen_at'])<datetime.fromisoformat(tube['frozen_at'])<datetime.fromisoformat(meta['completed_at'])
        with np.load(folder/'sampling.npz') as f:sample={k:f[k] for k in f.files}
        order=np.random.default_rng(case['stream_seed']).permutation(524288)
        assert np.array_equal(sample['primary_blocks'],order[:65536].reshape(2048,32))
        remaining=order[65536:];rng=np.random.default_rng(case['branch_seed'])
        assert sample['branch_blocks'].shape==(56,64,32)
        used=set(sample['primary_blocks'].ravel().tolist())
        with np.load(study/'data/panels.npz') as f:assert np.array_equal(sample['evaluation_crops'],f['evaluation'])
        transition=next(r for r in analysis['primary_transitions'] if r['case']==case['name'])
        with np.load(folder/'primary.npz') as raw:
            assert np.array_equal(raw['blocks'],sample['primary_blocks'])
            assert raw['losses'].shape==(2048,) and np.isfinite(raw['losses']).all()
            for boundary in ['initial','incoming']:
                logits=raw[boundary+'_logits'].astype(float)
                lp=logits-logsumexp(logits,axis=-1,keepdims=True)
                nll=-lp[np.arange(32),sample['evaluation_crops'][:,64]]
                entropy=-(np.exp(lp)*lp).sum(-1)
                close('primary_nll',nll,raw[boundary+'_nll'],1e-11)
                close('primary_entropy',entropy,raw[boundary+'_entropy'],1e-11)
                close('primary_mean_risk',np.array(nll.mean()),np.array(transition[boundary+'_risk']),1e-11)
                close('primary_mean_entropy',np.array(entropy.mean()),np.array(transition[boundary+'_entropy']),1e-11)
                close('primary_row_field',raw[boundary+'_rows'],transition[boundary+'_rows'],1e-12)
                assert raw[boundary+'_rows'].shape==(5,) and np.all((raw[boundary+'_rows']>=0)&(raw[boundary+'_rows']<=1+1e-7))
                primary_logits_count+=logits.size
        paths=[]
        for b,rec in enumerate(meta['records']):
            assert rec['branch']==b and rec['role']==('adaptation' if b<16 else 'calibration' if b<24 else 'assessment')
            draws=rng.choice(remaining,2048,replace=False).reshape(64,32)
            assert np.array_equal(sample['branch_blocks'][b],draws)
            assert len(set(draws.ravel()))==2048 and not used&set(draws.ravel())
            raw=folder/rec['raw'];bound(raw,rec['sha256'])
            if b>=24:assert raw.stat().st_mtime>=datetime.fromisoformat(tube['frozen_at']).timestamp()
            with np.load(raw,allow_pickle=False) as f:
                assert f['logits'].shape==(5,32,32000) and f['logits'].dtype==np.float32
                assert f['losses'].shape==(64,) and np.isfinite(f['losses']).all()
                assert list(f['horizons'])==[0,1,4,16,64]
                z=f['logits'].astype(float);lp=z-logsumexp(z,axis=-1,keepdims=True)
                target=sample['evaluation_crops'][:,64]
                nll=-np.take_along_axis(lp,np.broadcast_to(target[None,:,None],(5,32,1)),axis=-1)[...,0]
                entropy=-(np.exp(lp)*lp).sum(-1)
                close('nll',nll,f['nll'],1e-11);close('entropy',entropy,f['entropy'],1e-11)
                paths.append(np.stack([f['nll'].mean(1),f['entropy'].mean(1)],axis=1))
                logits_count+=z.size
        paths=np.array(paths);assert np.all(paths[:,0]==paths[0,0]);case_paths.append(paths[24:])
        for target,count in [('risk',1),('risk_entropy',2)]:
            inc=np.concatenate([np.diff(paths[:,:,j],axis=1) for j in range(count)],axis=1)
            design=inc[:16];mu=design.mean(0);z=design-mu;cov=z.T@z/15
            variance=np.maximum(np.diag(cov),1e-12);std=np.sqrt(variance)
            local=.75*cov+.25*np.diag(np.diag(cov))+.001*np.diag(variance)
            diagonal=np.diag(np.diag(cov))+.001*np.diag(variance)
            corr=np.array(archive['fits'][str(case['heads'])][target]['correlation'])
            transferred=(.75*corr+.25*np.diag(np.diag(corr))+.001*np.eye(len(corr)))*np.outer(std,std)
            fit_target=fit['fits'][target];close('fit_mean',mu,fit_target['mean'],1e-12)
            for key,c in [('local',local),('diagonal',diagonal),('transferred',transferred)]:close('fit_covariance',c,fit_target['covariances'][key],1e-12)
            for scale,base in [('fine',np.eye(4)),('paired',np.array([[1.,1.,0.,0.],[0.,0.,1.,1.]])),('endpoint',np.ones((1,4)))]:
                b=np.kron(np.eye(count),base);x=inc[24:]@b.T;mean=b@mu;scored={}
                for name,c in [('local',local),('diagonal',diagonal),('transferred',transferred)]:
                    matrix=b@c@b.T;lam,vec=np.linalg.eigh(matrix);assert lam.min()>0
                    e=(x-mean)@vec;values=np.log(lam).sum()+(e*e/lam).sum(1);scored[name]=values
                    close('individual_score',values,scores_file[case['name']+'__'+target+'__'+scale+'__'+name],1e-7)
                record=next(r for r in analysis['records'] if (r['case'],r['target'],r['scale'])==(case['name'],target,scale))
                observed_mean=x.mean(0);centered=x-observed_mean;population_covariance=centered.T@centered/32
                close('empirical_mean',observed_mean,record['empirical_mean'],1e-12)
                close('empirical_covariance',population_covariance,record['empirical_covariance'],1e-12)
                for name,values in scored.items():close('absolute_score_mean',np.array(values.mean()),np.array(record['mean_scores'][name]),1e-7)
                physical={'local':local,'diagonal':diagonal,'transferred':transferred}
                for first,second in [('local','diagonal'),('transferred','diagonal'),('transferred','local')]:
                    delta=scored[first]-scored[second];r=record['comparisons'][first+'__'+second]
                    close('score_mean',np.array(delta.mean()),np.array(r['mean']),1e-7)
                    close('score_mcse',np.array(delta.std(ddof=1)/np.sqrt(32)),np.array(r['mcse']),1e-7)
                    first_cov=b@physical[first]@b.T;second_cov=b@physical[second]@b.T
                    precision=np.linalg.solve(first_cov,np.eye(len(mean)))-np.linalg.solve(second_cov,np.eye(len(mean)))
                    mean_error=observed_mean-mean
                    parts={'logdet':np.linalg.slogdet(first_cov)[1]-np.linalg.slogdet(second_cov)[1],
                           'covariance':np.sum(precision*population_covariance.T),'mean_error':mean_error@precision@mean_error}
                    for key,value in parts.items():close('score_components',np.array(value),np.array(r['components'][key]),1e-7)
                    close('score_component_sum',np.array(sum(parts.values())),np.array(delta.mean()),1e-7)
        lower=np.tril(np.ones((4,4)));cov=np.array(fit['fits']['risk']['covariances']['local'])
        scale=np.sqrt(np.maximum(np.diag(lower@cov@lower.T),1e-12));mean=np.cumsum(fit['fits']['risk']['mean'])
        close('tube_scale',scale,tube['scale'],1e-12);close('tube_mean',mean,tube['mean'],1e-12)
        changes=paths[:,1:,0]-paths[:,0,None,0]
        cal=np.max(abs(changes[16:24]-mean)/scale,axis=1);close('calibration_scores',cal,tube['scores'],1e-10)
        close('calibration_radius',np.array(cal.max()),np.array(tube['radius']),1e-10)
        covered=np.max(abs(changes[24:]-mean)/scale,axis=1)<=cal.max()
        assert int(covered.sum())==next(r for r in analysis['coverage'] if r['case']==case['name'])['covered']
        row=next(r for r in analysis['coverage'] if r['case']==case['name'])
        assert np.array_equal(covered,np.array(row['indicators']))
        close('assessment_tube_scores',np.max(abs(changes[24:]-mean)/scale,axis=1),row['scores'],1e-10)
        close('tube_half_widths',cal.max()*scale,row['half_widths'],1e-12)
        diff=np.eye(4)-np.eye(4,k=-1)
        for key,base in [('fine',np.eye(4)),('paired',np.array([[1,1,0,0],[0,0,1,1]])),('endpoint',np.ones((1,4)))]:
            b=base@diff;width=cal.max()*abs(b)@scale
            flags=np.all(abs((changes[24:]-mean)@b.T)<=width+1e-12,axis=1)
            assert np.all(flags[covered]) and int(flags.sum())==row['scale_enclosures'][key]['covered']
            close('scale_enclosure_width',width,row['scale_enclosures'][key]['half_widths'],1e-12)
        observed=changes[24:].mean(0);mr=next(r for r in analysis['mean_predictions'] if r['case']==case['name'])
        close('incoming_risk',np.array(paths[0,0,0]),np.array(mr['incoming_risk']),1e-12)
        close('mean_change',observed,mr['observed_mean_change'],1e-12)
        close('mean_change_mcse',changes[24:].std(0,ddof=1)/np.sqrt(32),mr['mcse'],1e-12)
        for name,pred in [('identity',np.zeros(4)),('archive',np.cumsum(archive['fits'][str(case['heads'])]['risk']['mean'])),('adapted',mean)]:
            entry=mr['candidates'][name]
            close('mean_prediction',pred,entry['prediction'],1e-12)
            close('mean_error',abs(pred-observed),entry['absolute_error'],1e-11)
            close('mean_maximum_error',np.array(abs(pred-observed).max()),np.array(entry['maximum_absolute_error']),1e-11)
            close('mean_endpoint_error',np.array(abs(pred[-1]-observed[-1])),np.array(entry['endpoint_absolute_error']),1e-11)
        for prefix,entry in meta['inference'].items():
            raw=folder/entry['raw'];bound(raw,entry['sha256'])
            with np.load(raw) as f:
                z=f['native'].astype(float);lp=z-logsumexp(z,axis=-1,keepdims=True);target=f['targets'];idx=np.arange(32)
                for operation in ['fixed','projected']:
                    w=f[operation].astype(float);lq=w-logsumexp(w,axis=-1,keepdims=True)
                    row=next(r for r in analysis['inference'] if (r['case'],r['prefix'],r['operation'])==(case['name'],int(prefix),operation))
                    close('inference_risk',np.array((lp[idx,target]-lq[idx,target]).mean()),np.array(row['risk_change']),1e-11)
                    close('inference_kl',np.array((np.exp(lp)*(lp-lq)).sum(-1).mean()),np.array(row['mean_forward_kl']),1e-11)
    for heads in [4,14]:
        selected=[components[c['name']] for c in spec['cases'] if c['heads']==heads]
        for key in ['body','generator','metric']:assert len({c[key] for c in selected})==4
    for c in range(2):
        for i in range(2):assert components[f'h4-c{c}-i{i}']['metric']==components[f'h14-c{c}-i{i}']['metric']
    paths=np.array(case_paths)
    for h in [4,14]:
        ix=[i for i,c in enumerate(spec['cases']) if c['heads']==h];x=paths[ix,:,-1,0].reshape(2,2,32)
        row=next(r for r in analysis['hierarchy'] if r['heads']==h)
        expected=dict(within=np.var(x,axis=2).mean(),initialization=np.var(x.mean(2),axis=1).mean(),corpus=np.var(x.mean((1,2))),total=np.var(x))
        for key,value in expected.items():close('hierarchy',np.array(value),np.array(row[key]),1e-12)
    assert primary_logits_count==16384000
    assert logits_count==2293760000 and len(analysis['records'])==48
    assert analysis['total_covered']==sum(r['covered'] for r in analysis['coverage'])
    for target in ['risk','risk_entropy']:
        for scale in ['fine','paired','endpoint']:
            cells=[r for r in analysis['records'] if r['target']==target and r['scale']==scale]
            assert len(cells)==8
            for comparison in ['local__diagonal','transferred__diagonal','transferred__local']:
                differences=np.array([r['comparisons'][comparison]['mean'] for r in cells])
                row=analysis['summaries'][target+'__'+scale][comparison]
                assert row['states']==8 and row['favorable_states']==int(np.sum(differences<0))
                close('equal_state_score',np.array(differences.mean()),np.array(row['equal_state_mean']),1e-12)
    write_json(out,dict(status='passed',schema='outer-transfer-verification-v1',primary_paths=8,primary_updates=16384,
        branches=448,branch_updates=28672,assessment_paths=256,score_cells=48,comparisons_per_cell=3,
        branch_logit_coordinates=logits_count,primary_logit_coordinates=primary_logits_count,source_position_reuse=0,distinct_corpus_draws=2,nested_initializations=4,
        verification_scope='Complete stored branch and source-index reconstruction, independent logit/moment/score reduction, data content identity exclusion, initialization hash separation, and source-bound qualification records. Native replay and token-to-Arrow checks are separate reports.',
        maximum_errors=maxima,checked_sha256=checked,verifier_sha256=sha256(__file__),seconds=time.time()-start))
    print('Passed',logits_count,'logit coordinates and all 48 score cells.',flush=True)


if __name__=='__main__':main()
