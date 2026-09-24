#!/usr/bin/env python
"""Independent raw reconstruction of the released-model source experiment.

No project producer or analysis imports. Includes all proper-prefix risks,
paired operator orders, and explicitly centered full-vocabulary Fisher Grams.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def verify(study,output):
    study=Path(study).resolve();s=json.loads((study/'protocol.json').read_text());files={};errors={};counts={};outcomes=[];grams=[];coordinates=dict(logits=0,raw_tensors=0)
    def check(p,h=None):
        v=sha(p)
        if h is not None and v!=h:raise ValueError('Changed reference input: '+str(p))
        files[str(Path(p).resolve())]=v
    def equal(tag,a,b):
        errors[tag]=max(errors.get(tag,0.),float(np.max(np.abs(a-b))))
        if not np.allclose(a,b,atol=2e-12,rtol=2e-10):raise ValueError('Reference reduction differs: '+tag)
    stage=s['stage'];horizon=2 if stage=='qualification' else 512
    arms=[dict(name='narrative',rho=0.),dict(name='mixture',rho=.5),dict(name='technical',rho=1.),dict(name='general',rho=None)]
    if stage not in ['qualification','assessment'] or s['horizon']!=horizon or s['arms']!=arms:raise ValueError('Incomplete reference design')
    if s['rate']!=.00012 or s['optimizer_origin']!='fresh_zero_moments_and_counters':raise ValueError('Wrong reference optimizer law')
    check(study/'protocol.json')
    for p,h in s['inputs'].items():check(p,h)
    for p,h in s['sources'].items():check(study/'executed-source'/p,h)
    info=next(p for p in s['inputs'] if p.endswith('dataset_info.json'));record_path=next(p for p in s['inputs'] if p.endswith('records.json'))
    lengths=json.loads(Path(info).read_text())['splits']['train']['shard_lengths'];offset=np.r_[0,np.cumsum(lengths)]
    positions=np.array([offset[int(re.search(r'train-(\d+)-of-',r['shard']).group(1))]+r['row'] for r in json.loads(Path(record_path).read_text())])
    if not np.array_equal(positions,np.load(study/'global-document-positions.npy')) or positions.min()<32000000:raise ValueError('Public source-position separation failed')
    with np.load(Path(s['data'])/'evaluation.npz') as f:crops=f['crops'];labels=f['labels'];split=f['split'];evdocs=f['document_ids']
    with np.load(Path(s['data'])/'streams.npz') as f:streams={k:f[k] for k in f.files}
    consumed=np.zeros(4194304,bool);consumed[np.random.default_rng(641001).permutation(4194304)[:1310720]]=True
    def masks(ids):
        a={'all':np.ones(len(ids),bool),'heldout':split[ids]==1,'calibration':split[ids]==0}
        for label,name in [(1,'technical'),(0,'narrative'),(-1,'unassigned')]:a[name]=labels[ids]==label;a[name+'-heldout']=(labels[ids]==label)&(split[ids]==1)
        for name,mask in list(a.items()):a[name+'-fixed']=mask & ((ids%80)<16)
        return a
    names=[a['name'] for a in arms]+['general-replay']+(['general-unobserved'] if stage=='qualification' else [])
    records={name:json.loads((study/name/'result.json').read_text()) for name in names}
    general=records['general'];initial_digest=general['state_digests']['0']
    for name,r in records.items():
        folder=study/name;check(folder/'result.json');n=min(16,horizon) if name=='general-replay' else horizon
        if r['status']!='complete' or r['protocol_sha256']!=sha(study/'protocol.json') or r['steps']!=n:raise ValueError('Incomplete reference path')
        counts[r['role']]=counts.get(r['role'],0)+n
        if r['state_digests']['0']!=initial_digest:raise ValueError('Different incoming reference states')
        arm=next(a for a in arms if a['name']==('general' if name.startswith('general-') else name));rho=arm['rho']
        if rho is None:expected=streams['general'][:n*32].reshape(n,32)
        else:
            choose=streams['uniforms'][:n].ravel()<rho;flat=np.empty(n*32,np.int64)
            flat[choose]=streams['technical'][:int(choose.sum())];flat[~choose]=streams['narrative'][:int((~choose).sum())];expected=flat.reshape(n,32)
        check(folder/'blocks.npy',r['blocks_sha256']);blocks=np.load(folder/'blocks.npy')
        if not np.array_equal(blocks,expected) or len(np.unique(blocks))!=blocks.size or consumed[blocks].any() or np.isin(blocks//8,evdocs).any():raise ValueError('Reference single-pass rule failed')
        check(folder/'training.npz',r['training_sha256'])
        with np.load(folder/'training.npz') as f:
            if f['losses'].shape!=(n,) or any(not np.isfinite(f[k]).all() for k in f.files):raise ValueError('Invalid reference trace')
            equal('rates',f['rates'],np.full((n,2),.00012))
        times={0}|{t for t in s['times'] if 0<t<=n}
        if name=='general-unobserved':times={0,n}
        if {o['time'] for o in r['observations']}!=times:raise ValueError('Missing reference emission')
        for o in r['observations']:
            check(o['file'],o['sha256'])
            with np.load(o['file']) as f:
                if any(not np.isfinite(f[k]).all() for k in f.files):raise ValueError('Nonfinite reference observation')
                ids=f['indices'];expectedids=np.arange(480) if stage=='assessment' and o['time'] in [0,horizon] else np.array([j+k for j in [0,80,160,240,320,400] for k in range(16)])
                if not np.array_equal(ids,expectedids) or not np.array_equal(f['raw_indices'],[0,80,160,240,320,400]):raise ValueError('Wrong reference panel')
                raw=f['raw_tensors'].astype(np.float64)
                if raw.shape!=(6,5,14,2,2,64,64):raise ValueError('Wrong reference tensor axes')
                pos=np.searchsorted(ids,f['raw_indices']);short=raw[:,:,:,0];full=raw[:,:,:,1];a=full[:,:,:,0];mu=a.mean(-2)
                equal('paired_sse',f['paired_sse'][pos],np.square(full-short).sum((-2,-1)));equal('reference_sum',f['reference_sum'][pos],short.sum((-2,-1)));equal('reference_sumsq',f['reference_sumsq'][pos],np.square(short).sum((-2,-1)))
                equal('common',f['common'][pos],mu);equal('energy',f['energy'][pos],np.square(a-mu[...,None,:]).mean((-2,-1)));equal('total',f['total'][pos],np.square(a).mean((-2,-1)));coordinates['raw_tensors']+=raw.size
                risk={}
                for prefix in [32,64]:
                    z=f['logits'+str(prefix)].astype(np.float64)
                    if z.shape!=(len(ids),32000):raise ValueError('Reference vocabulary differs')
                    z-=z.max(-1,keepdims=True);lp=z-np.log(np.exp(z).sum(-1,keepdims=True));risk[prefix]=-lp[np.arange(len(ids)),crops[ids,prefix]];coordinates['logits']+=z.size
                if r['role']=='scientific':
                    for cohort,mask in masks(ids).items():
                        entries=int(mask.sum())*5*14*4096;sse=f['paired_sse'][mask].sum((0,1,2));ref=f['reference_sum'][mask].sum((0,1,2));refsq=f['reference_sumsq'][mask].sum((0,1,2));mean=ref/entries;rmse=np.sqrt(sse/entries)
                        outcomes.append(dict(arm=name,time=o['time'],cohort=cohort,documents=int(mask.sum()),risk32=float(risk[32][mask].mean()),risk64=float(risk[64][mask].mean()),reference_mean=mean.tolist(),paired_rmse=rmse.tolist(),order_mean=[float(x/abs(y)) if y!=0 else None for x,y in zip(rmse,mean)],order_rms=[float(np.sqrt(x/y)) if y>0 else None for x,y in zip(sse,refsq)],row_fraction=float(f['energy'][mask].sum()/f['total'][mask].sum())))
                if o['time']==0 or r['role']!='scientific':
                    base=next(x for x in general['observations'] if x['time']==o['time'])
                    if r['state_digests'][str(o['time'])]!=general['state_digests'][str(o['time'])]:raise ValueError('Reference state replay failed')
                    with np.load(base['file']) as b:
                        if f.files!=b.files or any(not np.array_equal(f[k],b[k]) for k in f.files):raise ValueError('Reference replay failed')
        if stage=='assessment' and r['role']=='scientific':
            check(r['checkpoint'],r['checkpoint_sha256'])
            import torch
            state=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)
            if state['finetuning_step']!=512 or state['scheduler']['last_epoch']!=512 or state['optimizer_origin']!=s['optimizer_origin']:raise ValueError('Reference final clock differs')
            if not np.array_equal(state['blocks'],blocks) or any(float(v['step'])!=512 for v in state['optimizer']['state'].values()):raise ValueError('Reference optimizer count differs')
            del state
        print('verified reference',name,flush=True)
    for t in s['times'][1:]:
        arrays={}
        for name in ['narrative','technical','general']:
            with np.load(next(o['file'] for o in records[name]['observations'] if o['time']==t)) as f:arrays[name]=f['logits64'].astype(np.float64);ids=f['indices']
        with np.load(general['observations'][0]['file']) as f:
            pos=np.searchsorted(f['indices'],ids);z=f['logits64'][pos].astype(np.float64);z-=z.max(-1,keepdims=True);p=np.exp(z);p/=p.sum(-1,keepdims=True)
        fields=[]
        for name in ['narrative','technical']:
            d=arrays[name]-arrays['general'];fields.append(np.sqrt(p)*(d-(p*d).sum(-1,keepdims=True)))
        per=np.einsum('icv,jcv->cij',np.stack(fields),np.stack(fields))
        for cohort,mask in masks(ids).items():
            g=per[mask].mean(0);eig=np.linalg.eigvalsh(g)[::-1]
            grams.append(dict(time=t,cohort=cohort,gram=g.tolist(),eigenvalues=eig.tolist(),second_to_first=float(eig[1]/eig[0]) if eig[0]>0 else None,cosine=float(g[0,1]/np.sqrt(g[0,0]*g[1,1])) if g[0,0]*g[1,1]>0 else None))
    check(study/'launcher.json');launch=json.loads((study/'launcher.json').read_text())
    if launch['status']!='complete' or set(launch['arms'])!=set(names):raise ValueError('Incomplete reference launcher')
    report=dict(schema='pldr-reference-finetuning-verification-v1',status='passed',stage=stage,protocol_sha256=sha(study/'protocol.json'),verifier_sha256=sha(__file__),verified_files=files,maximum_reconstruction_errors=errors,coordinates=coordinates,update_counts=counts,outcomes=outcomes,responses=grams,global_document_min=int(positions.min()),global_document_max=int(positions.max()),scope='One released initialization at one width with a fresh optimizer. All retained raw emissions independently reduced; full-vocabulary response Gram explicitly centered; no exponent inference or pooling with controlled width family.')
    Path(output).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('Reference verification passed',counts)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--output',required=True);a=p.parse_args();verify(a.study,a.output)
