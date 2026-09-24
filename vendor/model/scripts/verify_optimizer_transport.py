#!/usr/bin/env python3
"""Independent NumPy reduction of raw optimizer-transport observations.

Imports no producer, analysis or model code. Full tensor prediction residuals
are producer records; this check independently validates all saved sampled
coordinates, vocabulary observations, source indices and replay archives.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def require(test, message):
    if not test: raise ValueError(message)


def verify(study):
    study=Path(study).resolve(); protocol=study/'protocol.json'; spec=json.loads(protocol.read_text())
    files={str(protocol):sha(protocol)}
    for p,h in {**spec['sources'],**spec['inputs'],**spec['sampling']}.items():
        require(sha(p)==h,'Bound source or input changed: '+p); files[p]=h
    panels=Path(spec['data_root'])/'outer-transfer-20260911/data/panels.npz'
    with np.load(panels) as f: targets=f['evaluation'][:,64]
    vocab=0; samplecoords=0; reduction_error=0.; maxunits=np.zeros(3); prediction_records=0; signals=[]; replay=0
    for case in spec['cases']:
        rp=study/case['name']/'results.json'; result=json.loads(rp.read_text()); files[str(rp)]=sha(rp)
        require(result['status']=='complete' and result['protocol_sha256']==sha(protocol),'Incomplete or mismatched worker')
        order=np.random.default_rng(case['stream_seed']).permutation(524288)
        rng=np.random.default_rng(spec['source_seed']+case['heads'])
        expected=np.stack([rng.choice(order[65536:],32*spec['horizon'],replace=False).reshape(-1,32) for _ in range(spec['replicates'])])
        require(np.array_equal(np.load(study/(case['name']+'-blocks.npy')),expected),'Source law differs')
        recs={(r['mode'],r['source'],r['arm']):r for r in result['records']}
        keys={(m,s,a['name']) for m in spec['modes'] for s in range(spec['replicates']) for a in spec['arms']}
        keys|={(m,0,'zero-replay') for m in spec['modes']}
        require(set(recs)==keys and len(recs)==len(result['records']),'Missing or duplicate outcome')
        for mode in spec['modes']:
            metadata=study/case['name']/mode/'metadata.json'; files[str(metadata)]=sha(metadata)
            dtype=np.float32 if mode=='native32' else np.float64
            eps=np.finfo(dtype).eps; tiny=np.finfo(dtype).tiny
            for source in range(spec['replicates']):
                saved={}
                armnames=[a['name'] for a in spec['arms']]+(['zero-replay'] if source==0 else [])
                for armname in armnames:
                    r=recs[(mode,source,armname)]; path=Path(r['file'])
                    require(sha(path)==r['sha256'],'Raw archive changed'); files[str(path)]=r['sha256']
                    with np.load(path) as f: a={k:f[k] for k in f.files}
                    require(np.array_equal(a['blocks'],expected[source]) and np.array_equal(a['times'],spec['times']),'Unmatched sources or times')
                    require(len(np.unique(a['blocks']))==32*spec['horizon'],'Repeated source block')
                    require(not np.isin(a['blocks'],order[:65536]).any(),'Previously consumed source block')
                    logits=a['logits']; vocab+=logits.size
                    shifted=logits-logits.max(-1,keepdims=True)
                    lp=shifted-np.log(np.exp(shifted).sum(-1,keepdims=True))
                    nll=-np.take_along_axis(lp,np.broadcast_to(targets[None,:,None],lp.shape[:-1]+(1,)),-1)[...,0]
                    entropy=-(np.exp(lp)*lp).sum(-1)
                    common=np.log1p(np.sqrt(a['common_ms'].mean(axis=(2,3))))
                    transverse=np.log1p(np.sqrt(a['transverse_ms'].mean(axis=(2,3))))
                    q=np.column_stack([nll.mean(-1),entropy.mean(-1),common,transverse])
                    err=max(float(np.max(np.abs(q-a['q']))),float(np.max(np.abs(nll-a['nll']))),float(np.max(np.abs(entropy-a['entropy']))))
                    reduction_error=max(reduction_error,err); require(err<1e-10,'Observation reduction mismatch')
                    before=a['first_before']; after=a['first_after']; incoming=a['incoming_sample']; pulsed=a['pulsed_sample']
                    require(np.array_equal(incoming[:,1],pulsed[:,1]),'Moment pulse changed parameters')
                    b1,b2,k,rate,offset,decay=[before[:,j] for j in range(5,11)]
                    p,m,v,g=[before[:,j] for j in range(1,5)]
                    require(np.all(v>=0) and np.all(k==2048),'Bad moment or counter')
                    require(np.array_equal(before[:,1:4],pulsed[:,1:4]),'State changed before the predicted first update')
                    mp=b1*m+(1-b1)*g; vp=b2*v+(1-b2)*g*g
                    update=(rate/(1-b1**(k+1)))*mp/(np.sqrt(vp/(1-b2**(k+1)))+offset)
                    predicted=np.column_stack([(1-rate*decay)*p-update,mp,vp])
                    scales=np.column_stack([np.abs(p)+rate*(b1*np.abs(m)+(1-b1)*np.abs(g))/(1-b1**(k+1))/(np.sqrt(vp/(1-b2**(k+1)))+offset),b1*np.abs(m)+(1-b1)*np.abs(g),b2*v+(1-b2)*g*g])
                    units=np.max(np.abs(after[:,1:4]-predicted)/(eps*scales+tiny),axis=0)
                    maxunits=np.maximum(maxunits,units); samplecoords+=len(p)
                    require(np.all(units<=spec['prediction_tolerance_units']),'Saved-coordinate prediction exceeded tolerance')
                    require(np.all(np.array(r['full_prediction']['max_roundoff_units'])<=spec['prediction_tolerance_units']),'Full-coordinate producer prediction exceeded tolerance')
                    prediction_records+=1
                    require(np.all(after[:,7]==before[:,7]+1),'Optimizer counter mismatch')
                    if armname!='zero-replay':
                        arm=next(x for x in spec['arms'] if x['name']==armname)
                        factor=1+arm['h'] if arm['family']=='first' else np.exp(arm['h'])
                        targetcol=2 if arm['family']=='first' else 3
                        if arm['family']:
                            expected_pulse=(incoming[:,targetcol].astype(dtype)*dtype(factor)).astype(dtype).astype(np.float64)
                            require(np.array_equal(expected_pulse,pulsed[:,targetcol]),'Pulse convention differs')
                            other=3 if targetcol==2 else 2
                            require(np.array_equal(incoming[:,other],pulsed[:,other]),'Unselected moment changed')
                    saved[armname]=a
                    base=saved['zero']; rb=recs[(mode,source,'zero')]
                    require(np.array_equal(a['logits'][0],base['logits'][0]),'Initial emission differs')
                    require(np.array_equal(before[:,4],base['first_before'][:,4]) and r['first_gradient_sha256']==rb['first_gradient_sha256'],'First clipped gradient differs')
                    require(r['zero_second_moments']==rb['zero_second_moments'],'Zero-moment support differs')
                    if armname=='zero-replay':
                        require(all(np.array_equal(a[key],base[key]) for key in a),'Replay arrays differ')
                        require(r['final_digest']==rb['final_digest'],'Final complete state replay differs'); replay+=1
                for family in ['first','logsecond']:
                    z=saved['zero']['q']; p=saved[family+'-plus']['q']; m=saved[family+'-minus']['q']
                    odd=(p-m)/2; even=(p+m)/2-z
                    require(np.max(np.abs(z+even+odd-p))<1e-12,'Finite pulse reconstruction differs')
                    signals.append(dict(case=case['name'],mode=mode,source=source,family=family,
                        first_odd_rms=float(np.sqrt(np.mean(odd[1]**2))),endpoint_odd_rms=float(np.sqrt(np.mean(odd[-1]**2)))))
    npaths=len(spec['cases'])*spec['replicates']*len(spec['arms'])
    return dict(status='passed',stage=spec['stage'],protocol_sha256=sha(protocol),verified_files=files,
        verifier_sha256=sha(__file__),raw_archives=prediction_records,vocabulary_coordinates=vocab,
        sampled_parameter_coordinates=samplecoords,max_sample_prediction_units=maxunits.tolist(),
        max_observation_error=reduction_error,replays=replay,signals=signals,
        native_updates=npaths*spec['horizon'],arithmetic_updates=npaths*spec['horizon'],replay_updates=replay*spec['horizon'],
        scope='Independent full-vocabulary reduction, saved first-step coordinate predictions, sampling, incoming emissions and replay arrays. Whole model prediction norms, all-gradient identities and final complete-state equality are bound producer digests/reductions. Full metric tensors are not independently reconstructed.')


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--study',required=True); ap.add_argument('--output',required=True)
    a=ap.parse_args(); out=Path(a.output)
    if out.exists(): raise FileExistsError(out)
    result=verify(a.study); out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['verified_files','signals']},indent=2))
