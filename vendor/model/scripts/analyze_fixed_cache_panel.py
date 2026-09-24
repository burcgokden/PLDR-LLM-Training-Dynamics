#!/usr/bin/env python3
"""CPU-only reconstruction of all fixed-cache context-transfer policies."""
import argparse
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from model_rg.provenance import sha256,write_json
from numerical_validation import load_json_strict

def read(p):return load_json_strict(Path(p).read_text())
def analyze(study):
    p=read(study/'protocol.json');ph=sha256(study/'protocol.json')
    if p['contexts']!=16 or p['states']!=['initial','g0','g1.5'] or len(p['jobs'])!=12:raise ValueError('Wrong complete panel')
    if p['selection_sha256']!=sha256(study/'inputs.npz'):raise ValueError('Input binding fails')
    with np.load(study/'inputs.npz') as a:targets=a['blocks'][:,64]
    coverage={'protocol.json':ph,'inputs.npz':sha256(study/'inputs.npz')};calls={'assessment':0,'qualification':0};seconds=0.;peak=0;arrays={}
    for j in p['jobs']:
        folder=study/'runs'/j['run_id'];m=read(folder/'manifest.json');name='runs/'+j['run_id']+'/'
        if m['status']!='complete' or m['job']!=j or m['protocol_sha256']!=ph or m['calls']!={'assessment':16,'qualification':6} or m['training_updates']!=0:raise ValueError('Manifest scope differs')
        expected={'logits.npz':sha256(folder/'logits.npz')}
        if m['artifacts']!=expected:raise ValueError('Observation binding fails')
        with np.load(folder/'logits.npz') as a:
            if set(a.files)!={s+'-'+r for s in p['states'] for r in (['native','state_cache'] if s=='initial' else ['native','state_cache','initial_cache'])}:raise ValueError('Incomplete observation roles')
            arrays[j['run_id']]={k:a[k].astype(float) for k in a.files}
        for v in arrays[j['run_id']].values():
            if v.shape!=(16,32000) or not np.isfinite(v).all():raise ValueError('Invalid logits')
        for n,h in expected.items():coverage[name+n]=h
        coverage[name+'manifest.json']=sha256(folder/'manifest.json')
        for k in calls:calls[k]+=m['calls'][k]
        seconds+=m['seconds'];peak=max(peak,m['peak_allocated_bytes'])
    cells=[];maximum=0.
    for n in [8,24]:
        jobs=sorted([j for j in p['jobs'] if j['heads']==n],key=lambda j:j['seed'])
        if len(jobs)!=6:raise ValueError('Missing independent initialization')
        for state in p['states']:
            for policy in (['state_cache'] if state=='initial' else ['state_cache','initial_cache']):
                x=np.stack([arrays[j['run_id']][state+'-native'] for j in jobs]);y=np.stack([arrays[j['run_id']][state+'-'+policy] for j in jobs])
                lp=x-logsumexp(x,axis=-1,keepdims=True);lq=y-logsumexp(y,axis=-1,keepdims=True)
                a=2*np.exp(lp/2);b=2*np.exp(lq/2);e=a-b
                def centered(v):return ((v-v.mean(0))**2).sum((0,2))/5
                def pair(v):return sum(((v[i]-v[j])**2).sum(-1) for i in range(6) for j in range(i))/30
                va,vb,ve=[centered(v) for v in [a,b,e]]
                err=max(np.max(abs(centered(v)-pair(v))) for v in [a,b,e]);maximum=max(maximum,float(err))
                if err>5e-12 or np.any(va<=0):raise ValueError('Pair identity or positive variance fails')
                rc=np.sqrt(ve/va);r=float(np.sqrt(ve.sum()/va.sum()));kl=float((np.exp(lp)*(lp-lq)).sum(-1).mean())
                nll=float(-lp[:,np.arange(16),targets].mean());cached_nll=float(-lq[:,np.arange(16),targets].mean())
                cells.append(dict(heads=n,state=state,policy=policy,seeds=[j['seed'] for j in jobs],relative_centered_rms=r,mean_kl=kl,
                    native_variance=float(va.mean()),cached_variance=float(vb.mean()),residual_variance=float(ve.mean()),
                    native_nll=nll,cached_nll=cached_nll,mean_nll_change=cached_nll-nll,
                    maximum_context_rms=float(rc.max()),contexts_over_target=int((rc>.25).sum()),per_context_rms=rc.tolist(),
                    uncentered_error=float((e*e).sum(-1).mean()),mean_bias=float((e.mean(0)**2).sum(-1).mean()),
                    aggregate_targets_met=bool(r<=.25 and kl<=.03)))
    return dict(status='passed',schema='fixed-cache-panel-analysis-v1',protocol_sha256=ph,cells=cells,checked_sha256=coverage,calls=calls,
        native_forward_calls=sum(calls.values()),training_updates=0,worker_seconds=seconds,peak_allocated_bytes=peak,
        maximum_pair_identity_error=maximum,analysis_source_sha256=sha256(__file__))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    r=analyze(a.study);write_json(a.output,r);print(dict(status=r['status'],cells=len(r['cells']),calls=r['native_forward_calls']))
