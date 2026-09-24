#!/usr/bin/env python
"""Independent NumPy reduction of native fine-tuning artifacts.

Does not import the producer, observer, analyzer or model helper package.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from finetuning_verification_design import terminal_inventory, observation_panel, trace_shapes


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();study=Path(a.study).resolve();spec=json.loads((study/'protocol.json').read_text())
    checked=terminal_inventory(study,spec);errors={};coordinates=dict(logits=0,raw_tensors=0);counts={}
    def check(path,expected=None):
        path=Path(path);value=sha(path)
        if expected is not None and value!=expected:raise ValueError('Hash mismatch: '+str(path))
        checked[str(path)]=value
    check(study/'protocol.json')
    for p,h in spec['inputs'].items():check(p,h)
    repo=Path(__file__).resolve().parents[1]
    for n,h in spec['sources'].items():check(study/'executed-source'/n,h)
    data=Path(spec['data'])
    with np.load(data/'evaluation.npz') as f:
        crops=f['crops'];evdocs=f['document_ids'];labels=f['labels'];split=f['split']
    with np.load(data/'lexical-labels.npz') as f:document_labels=f['labels'];scores=f['scores']
    expected_labels=np.full(524288,-1,np.int8)
    expected_labels[(scores[:,0]>=2)&(scores[:,0]-scores[:,1]>=2)]=1
    expected_labels[(scores[:,1]>=2)&(scores[:,1]-scores[:,0]>=2)]=0
    if not np.array_equal(expected_labels,document_labels):raise ValueError('Lexical rule mismatch')
    order=np.random.default_rng(641001).permutation(4194304)
    consumed=np.zeros(4194304,bool);consumed[order[:1310720]]=True
    if consumed.reshape(-1,8)[evdocs].any():raise ValueError('Evaluation document previously consumed')
    untouched=~consumed.reshape(-1,8).any(1)
    rng_eval=np.random.default_rng(915001)
    selected=np.concatenate([rng_eval.choice(np.flatnonzero(untouched & (document_labels==k)),160,replace=False)
                             for k in [1,0,-1]])
    if not np.array_equal(evdocs,selected) or not np.array_equal(labels,np.repeat([1,0,-1],160)):
        raise ValueError('Evaluation document/stratum identity changed')
    if not np.array_equal(split,np.tile(np.r_[np.zeros(80,dtype=np.int8),np.ones(80,dtype=np.int8)],3)):
        raise ValueError('Calibration/assessment split changed')
    tokens=np.load(Path(spec['root'])/'data/refinedweb-onepass-524288/tokens.npy',mmap_mode='r')
    if not np.array_equal(crops,tokens[evdocs,:65]):raise ValueError('Evaluation source mismatch')
    reserved=np.zeros(524288,bool);reserved[evdocs]=True
    remaining=order[1310720:];remaining=remaining[~reserved[remaining//8]]
    pools={'technical':remaining[document_labels[remaining//8]==1],
           'narrative':remaining[document_labels[remaining//8]==0],'general':remaining}
    for k,v in pools.items():
        if not np.array_equal(v,np.load(data/(k+'-blocks.npy'))):raise ValueError('Pool mismatch: '+k)
    rng=np.random.default_rng(915002)
    draws={k:rng.permutation(pools[k])[:32768] for k in ['technical','narrative','general']}
    uniforms=rng.uniform(size=(1024,32))
    with np.load(data/'streams.npz') as f:
        for k,v in {**draws,'uniforms':uniforms}.items():
            if not np.array_equal(v,f[k]):raise ValueError('Frozen stream mismatch')
    def compare(name,actual,expected,rtol=2e-10,atol=2e-12):
        error=float(np.max(np.abs(actual-expected)))
        errors[name]=max(errors.get(name,0.),error)
        if not np.allclose(actual,expected,rtol=rtol,atol=atol):raise ValueError('Reduction failed: '+name)
    compact=[]
    def observation(path,heads,timepoint):
        with np.load(path) as f:
            observation_panel(f,spec,timepoint)
            if any(not np.isfinite(f[k]).all() for k in f.files):raise ValueError('Nonfinite saved observation')
            idx=f['indices'];raw_idx=f['raw_indices'];raw=f['raw_tensors'].astype(np.float64)
            expected=(len(raw_idx),5,heads,2,2,64,64)
            if raw.shape!=expected:raise ValueError('Unexpected raw tensor axes')
            positions=np.array([int(np.flatnonzero(idx==j)[0]) for j in raw_idx])
            short=raw[:,:,:,0];full=raw[:,:,:,1]
            compare('paired_sse',f['paired_sse'][positions],np.square(full-short).sum((-2,-1)))
            compare('reference_sum',f['reference_sum'][positions],short.sum((-2,-1)))
            compare('reference_sumsq',f['reference_sumsq'][positions],np.square(short).sum((-2,-1)))
            matrix=full[:,:,:,0];center=matrix.mean(-2)
            compare('common',f['common'][positions],center)
            compare('energy',f['energy'][positions],np.square(matrix-center[...,None,:]).mean((-2,-1)))
            compare('total',f['total'][positions],np.square(matrix).mean((-2,-1)))
            risks=[];entropies=[]
            for prefix in [32,64]:
                z=f['logits'+str(prefix)].astype(np.float64)
                if z.shape!=(len(idx),32000):raise ValueError('Vocabulary/panel mismatch')
                shift=z-z.max(-1,keepdims=True);lp=shift-np.log(np.exp(shift).sum(-1,keepdims=True))
                risk=-lp[np.arange(len(idx)),crops[idx,prefix]];entropy=-(np.exp(lp)*lp).sum(-1)
                if not np.isfinite(risk).all() or not np.isfinite(entropy).all():raise ValueError('Bad predictive reduction')
                risks.append(risk);entropies.append(entropy);coordinates['logits']+=z.size
            coordinates['raw_tensors']+=raw.size
            entries=len(idx)*5*heads*4096
            sse=f['paired_sse'].sum((0,1,2));rs=f['reference_sum'].sum((0,1,2));rss=f['reference_sumsq'].sum((0,1,2))
            rms=np.sqrt(sse/entries);mean=rs/entries
            values=dict(contexts=len(idx),risk32=float(risks[0].mean()),risk64=float(risks[1].mean()),
                        entropy64=float(entropies[1].mean()),paired_rmse=rms.tolist(),reference_mean=mean.tolist(),
                        order_mean=[float(r/abs(m)) if m!=0 else None for r,m in zip(rms,mean)],
                        order_rms=[float(np.sqrt(e/r)) if r>0 else None for e,r in zip(sse,rss)])
        return values
    for case in spec['cases']:
        check(case['state'],case['state_sha256']);check(case['manifest'],case['manifest_sha256'])
        folder=study/case['name'];result=json.loads((folder/'results.json').read_text())
        check(folder/'results.json')
        if result['status']!='complete' or result['protocol_sha256']!=sha(study/'protocol.json'):
            raise ValueError('Incomplete case')
        if not result['replay_bitwise']:raise ValueError('No qualified replay')
        initial=folder/'initial-observation.npz';check(initial,result['initial_observation_sha256'])
        initial_summary=observation(initial,case['heads'],0)
        expected_arms={arm['name'] for arm in spec['arms'] if arm['role']=='primary' or case['seed']==640101}
        expected_arms.add('general-replay')
        if spec['stage']=='qualification':expected_arms.add('general-unobserved')
        if {r['arm']['name'] for r in result['records']}!=expected_arms:raise ValueError('Arm set mismatch')
        for record in result['records']:
            arm=record['arm'];p=folder/arm['name'];check(p/'result.json')
            role=arm['role'];counts[role]=counts.get(role,0)+record['steps']
            check(p/'blocks.npy',record['blocks_sha256']);blocks=np.load(p/'blocks.npy')
            if consumed[blocks].any() or reserved[blocks//8].any() or len(np.unique(blocks))!=blocks.size:
                raise ValueError('Source resource violation')
            horizon=record['steps']
            if arm['rho'] is None:expected=draws['general'][:32*horizon].reshape(-1,32)
            else:
                choose=(uniforms[:horizon]<arm['rho']).reshape(-1)
                expected=np.empty(choose.size,np.int64)
                expected[choose]=draws['technical'][:choose.sum()]
                expected[~choose]=draws['narrative'][:(~choose).sum()]
                expected=expected.reshape(horizon,32)
            if not np.array_equal(blocks,expected):raise ValueError('Mixture draw changed')
            check(p/'training.npz',record['training_sha256'])
            with np.load(p/'training.npz') as f:
                trace_shapes(blocks,f,horizon)
                if f['losses'].shape!=(horizon,) or not all(np.isfinite(f[k]).all() for k in f.files):
                    raise ValueError('Invalid training trace')
                rates=np.full((horizon,2),.00012)
                if arm['group']=='generator':rates[:,1]=0
                if arm['group']=='body':rates[:,0]=0
                compare('learning_rate',f['rates'],rates,rtol=1e-14,atol=1e-18)
            if record['checkpoint']:check(record['checkpoint'],record['checkpoint_sha256'])
            steps={o['time'] for o in record['observations']}
            target={t for t in spec['times'] if 0<t<=horizon}
            if role=='unobserved replay':target={horizon}
            if steps!=target:raise ValueError('Missing observation time')
            rows=[]
            for o in record['observations']:
                check(o['file'],o['sha256']);summary=observation(o['file'],case['heads'],o['time']);rows.append(dict(time=o['time'],**summary))
                if role in ['replay','unobserved replay']:
                    base=next(r for r in result['records'] if r['arm']['name']=='general')
                    bo=next(r for r in base['observations'] if r['time']==o['time'])
                    if record['state_digests'][str(o['time'])]!=base['state_digests'][str(o['time'])]:raise ValueError('Replay digest mismatch')
                    with np.load(o['file']) as x,np.load(bo['file']) as y:
                        if x.files!=y.files or any(not np.array_equal(x[k],y[k]) for k in x.files):raise ValueError('Replay arrays differ')
            compact.append(dict(case=case['name'],heads=case['heads'],seed=case['seed'],arm=arm,steps=horizon,
                                incoming=initial_summary,observations=rows))
        print('verified',case['name'],flush=True)
    check(study/'launcher.json')
    if json.loads((study/'launcher.json').read_text())['status']!='complete':raise ValueError('Launcher incomplete')
    value=dict(schema='pldr-finetuning-verification-v1',status='passed',stage=spec['stage'],
               protocol_sha256=sha(study/'protocol.json'),verifier_sha256=sha(__file__),
               design_verifier_sha256=sha(Path(__file__).with_name('finetuning_verification_design.py')),
               verified_files=checked,maximum_reconstruction_errors=errors,coordinates=coordinates,
               update_counts=counts,complete_outcomes=compact,
               scope='All vocabulary logits and every saved raw tensor reconstructed independently. Full-cohort tensor sums and complete-state hashes are producer reductions; their saved raw six-context panel and replay arrays are checked independently. All source positions and registered arms/times checked.')
    output=Path(a.output)
    if output.exists():raise FileExistsError('Preserve existing verification: '+str(output))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='passed',coordinates=coordinates,updates=counts,errors=errors),indent=2),flush=True)


if __name__=='__main__':main()
