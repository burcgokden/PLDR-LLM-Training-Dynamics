#!/usr/bin/env python
"""Independent source-return verification; imports no experiment or analysis code."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def verify(study,output):
    study=Path(study).resolve();spec=json.loads((study/'protocol.json').read_text());files={};errors={};coordinates=dict(logits=0,raw_tensors=0);counts={}
    def check(p,h=None):
        value=sha(p)
        if h is not None and h!=value:raise ValueError('Changed source-return artifact: '+str(p))
        files[str(Path(p).resolve())]=value
    def equal(tag,a,b,atol=2e-12,rtol=2e-10):
        errors[tag]=max(errors.get(tag,0.),float(np.max(np.abs(a-b))))
        if not np.allclose(a,b,atol=atol,rtol=rtol):raise ValueError('Independent reduction failed: '+tag)
    stage=spec['stage'];horizon=2 if stage=='qualification' else 256
    if stage not in ['qualification','assessment'] or spec['horizon']!=horizon:raise ValueError('Invalid return stage')
    expected_cases=[(h,s) for h in [4,8,14] for s in ([640101] if stage=='qualification' else [640101,640102,640103,640104])]
    if [(c['heads'],c['seed']) for c in spec['cases']]!=expected_cases:raise ValueError('Incomplete return family')
    check(study/'protocol.json')
    for p,h in spec['inputs'].items():check(p,h)
    for n,h in spec['sources'].items():check(study/'executed-source'/n,h)
    data=Path(spec['data']);parent=Path(spec['parent_study']);check(parent/'protocol.json',spec['parent_protocol_sha256'])
    with np.load(data/'evaluation.npz') as f:crops=f['crops'];evdocs=f['document_ids']
    with np.load(data/'streams.npz') as f:excluded=np.unique(np.concatenate([f[n][:16384] for n in ['technical','narrative','general']]))
    pool=np.load(data/'general-blocks.npy');pool=pool[~np.isin(pool,excluded)]
    expected=np.random.default_rng(915003).permutation(pool)[:8192].reshape(256,32)[:horizon]
    blocks=np.load(study/'blocks.npy')
    consumed=np.zeros(4194304,bool);consumed[np.random.default_rng(641001).permutation(4194304)[:1310720]]=True
    if not np.array_equal(blocks,expected) or consumed[blocks].any() or np.isin(blocks//8,evdocs).any() or len(np.unique(blocks))!=blocks.size:raise ValueError('Return source-resource rule failed')
    outcomes=[]
    for case in spec['cases']:
        for row in case['parents']:
            for key in ['checkpoint','result','observation']:check(row[key],row[key+'_sha256'])
            check(parent/case['name']/row['arm']/'blocks.npy')
            if np.intersect1d(blocks,np.load(parent/case['name']/row['arm']/'blocks.npy')).size:raise ValueError('Parent block reused')
        folder=study/case['name'];result=json.loads((folder/'results.json').read_text());check(folder/'results.json')
        if result['status']!='complete' or result['protocol_sha256']!=sha(study/'protocol.json'):raise ValueError('Incomplete return case')
        expected_names={'mix0000','mix1000','general','general-replay'}
        if stage=='qualification':expected_names.add('general-unobserved')
        if {r['name'] for r in result['records']}!=expected_names:raise ValueError('Return arm set differs')
        if not result['parent_reproduction_bitwise'] or not result['replay_bitwise']:raise ValueError('Missing exact return replay')
        general=next(r for r in result['records'] if r['name']=='general')
        for record in result['records']:
            arm=folder/record['name'];check(arm/'result.json');counts[record['role']]=counts.get(record['role'],0)+record['steps']
            expected_steps=min(16,horizon) if record['role']=='replay' else horizon
            if record['steps']!=expected_steps:raise ValueError('Wrong return path length')
            check(arm/'training.npz',record['training_sha256'])
            with np.load(arm/'training.npz') as f:
                if f['losses'].shape!=(record['steps'],) or not all(np.isfinite(f[k]).all() for k in f.files):raise ValueError('Invalid return training trace')
                equal('rates',f['rates'],np.full((record['steps'],2),.00012),atol=1e-18,rtol=1e-14)
            parent_row=next(r for r in case['parents'] if r['arm']==record['parent'])
            if record['state_digests']['0']!=parent_row['expected_state_digest']:raise ValueError('Parent state digest differs')
            times={0}|{t for t in spec['times'] if 0<t<=record['steps']}
            if record['role']=='unobserved replay':times={0,record['steps']}
            if {o['time'] for o in record['observations']}!=times:raise ValueError('Missing return observation')
            for o in record['observations']:
                check(o['file'],o['sha256'])
                with np.load(o['file']) as f:
                    if any(not np.isfinite(f[k]).all() for k in f.files):raise ValueError('Nonfinite return observation')
                    ids=f['indices'];raw=f['raw_tensors'].astype(np.float64);rawids=f['raw_indices']
                    expected_ids=np.arange(480) if stage=='assessment' and o['time'] in [0,horizon] else np.array([j+k for j in [0,80,160,240,320,400] for k in range(16)])
                    if not np.array_equal(ids,expected_ids):raise ValueError('Incorrect return panel')
                    if not np.array_equal(rawids,[0,80,160,240,320,400]) or raw.shape!=(6,5,case['heads'],2,2,64,64):raise ValueError('Return raw axes differ')
                    pos=np.searchsorted(ids,rawids);short=raw[:,:,:,0];full=raw[:,:,:,1]
                    equal('paired_sse',f['paired_sse'][pos],np.square(full-short).sum((-2,-1)))
                    equal('reference_sum',f['reference_sum'][pos],short.sum((-2,-1)))
                    equal('reference_sumsq',f['reference_sumsq'][pos],np.square(short).sum((-2,-1)))
                    residual=full[:,:,:,0];mu=residual.mean(-2)
                    equal('common',f['common'][pos],mu);equal('energy',f['energy'][pos],np.square(residual-mu[...,None,:]).mean((-2,-1)))
                    equal('total',f['total'][pos],np.square(residual).mean((-2,-1)))
                    coordinates['raw_tensors']+=raw.size
                    risks=[]
                    for prefix in [32,64]:
                        z=f['logits'+str(prefix)].astype(np.float64)
                        if z.shape!=(len(ids),32000):raise ValueError('Return vocabulary differs')
                        z=z-z.max(-1,keepdims=True);lp=z-np.log(np.exp(z).sum(-1,keepdims=True))
                        risks.append(float(-lp[np.arange(len(ids)),crops[ids,prefix]].mean()));coordinates['logits']+=z.size
                    if o['time']==0:
                        with np.load(parent_row['observation']) as p:
                            for k in f.files:
                                target=ids if k=='indices' else p[k] if k in ['raw_tensors','raw_indices'] else p[k][ids]
                                if not np.array_equal(f[k],target):raise ValueError('Parent emission differs independently')
                    if record['role'] in ['replay','unobserved replay']:
                        base=next(x for x in general['observations'] if x['time']==o['time'])
                        if record['state_digests'][str(o['time'])]!=general['state_digests'][str(o['time'])]:raise ValueError('Return replay state mismatch')
                        with np.load(base['file']) as p:
                            if f.files!=p.files or any(not np.array_equal(f[k],p[k]) for k in f.files):raise ValueError('Return replay arrays mismatch')
                outcomes.append(dict(case=case['name'],parent=record['parent'],role=record['role'],time=o['time'],risk32=risks[0],risk64=risks[1]))
        print('verified return',case['name'],flush=True)
    check(study/'launcher.json')
    if json.loads((study/'launcher.json').read_text())['status']!='complete':raise ValueError('Return launcher incomplete')
    report=dict(schema='pldr-source-return-verification-v1',status='passed',stage=stage,protocol_sha256=sha(study/'protocol.json'),verifier_sha256=sha(__file__),
                verified_files=files,maximum_reconstruction_errors=errors,coordinates=coordinates,update_counts=counts,complete_outcomes=outcomes,
                scope='Independent reconstruction of all saved vocabulary logits and raw tensors; every parent observation and complete-state digest reproduced, replay arrays compared, and source positions regenerated without project helpers.')
    Path(output).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='passed',updates=counts,coordinates=coordinates,errors=errors),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--output',required=True);a=p.parse_args();verify(a.study,a.output)
