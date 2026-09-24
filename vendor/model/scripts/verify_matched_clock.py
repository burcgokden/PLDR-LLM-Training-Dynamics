#!/usr/bin/env python3
"""Independent paired-distance and raw-moment reconstruction of matched-clock results."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import log_softmax
from numerical_validation import load_json_strict,array_discrepancy
from matched_clock_coverage import admitted_protocol, inventory, validate_analysis, canonical, source_hashes, read as read_coverage


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def verify(study,analysis):
    p=admitted_protocol(study);a=read_coverage(analysis);err=0.
    coverage=validate_analysis(a,p)
    _,observed=inventory(study)
    if observed!=a['checked_sha256']:raise ValueError('Observation census differs from raw inventory')
    if a['status']!='passed' or a['protocol_sha256']!=digest(study/'protocol.json') or len(a['fields'])!=64 or len(a['cache_cells'])!=26:raise ValueError('Incomplete result grid')
    for name,h in a['checked_sha256'].items():
        if digest(study/name)!=h:raise ValueError('Changed observation '+name)
    with np.load(study/'selection.npz') as z:targets=z['probes'][16:,64]
    def compare(x,y):
        nonlocal err
        error=array_discrepancy(np.asarray(x),np.asarray(y),'matched-clock result',atol=3e-10);err=max(err,error)
    def pair(values):
        return sum(((values[i]-values[j])**2).sum(-1) for i in range(3) for j in range(i))/6
    lookup={}
    for c in a['fields']:
        n,g,age=c['heads'],c['control'],c['age'];arrays=[]
        for seed in [9163401,9163402,9163403]:
            with np.load(study/'runs'/f'h{n}-g{g:g}-s{seed}'/f'age-{int(age*128*n)}.npz') as z:arrays.append({k:z[k].astype(np.float64) for k in z.files})
        lookup[n,g,age]=arrays
        lp=np.stack([log_softmax(v['native'],axis=1) for v in arrays]);u=2*np.exp(lp/2)
        rows=np.stack([v['heads'][...,0].mean((1,2)) for v in arrays]);ops=np.stack([v['heads'][...,2].mean((1,2)) for v in arrays])
        rv=sum((rows[i]-rows[j])**2 for i in range(3) for j in range(i))/6
        ov=sum((ops[i]-ops[j])**2 for i in range(3) for j in range(i))/6
        compare(c['row_mean'],rows.mean());compare(c['row_susceptibility'],n*rv.mean());compare(c['operator_susceptibility'],n*ov.mean())
        compare(c['predictive_susceptibility'],n*pair(u).mean());compare(c['mean_nll'],-lp[:,np.arange(64),targets].mean())
        compare(c['operator_amplitude'],ops.mean());compare(c['row_seed_means'],rows.mean(1))
        compare(c['seed_nll'],-lp[:,np.arange(64),targets].mean(1))
        compare(c['maximum_adaptive_force'],max(float(v['maximum_adaptive_force']) for v in arrays))
    for c in a['cache_cells']:
        n,g,age=c['heads'],0. if c['control'] is None else c['control'],c['age'];arrays=lookup[n,g,age];policy=c['policy']
        lp=np.stack([log_softmax(v['native'],axis=1) for v in arrays]);lq=np.stack([log_softmax(v[policy],axis=1) for v in arrays]);u=2*np.exp(lp/2);v=2*np.exp(lq/2)
        reference=pair(u);residual=pair(u-v);rc=np.sqrt(residual/reference);R=float(np.sqrt(residual.sum()/reference.sum()));kl=float((np.exp(lp)*(lp-lq)).sum(-1).mean())
        compare(c['relative_centered_rms'],R);compare(c['per_context_rms'],rc);compare(c['mean_kl'],kl)
        compare(c['mean_nll_change'],(lp-lq)[:,np.arange(64),targets].mean())
        if c['contexts_over_target']!=int(np.count_nonzero(rc>.25)) or c['aggregate_targets_met']!=bool(R<=.25 and kl<=.03):raise ValueError('Decision differs')
        compare(c['native_nll'],-lp[:,np.arange(64),targets].mean());compare(c['cached_nll'],-lq[:,np.arange(64),targets].mean())
        compare(c['maximum_context_rms'],rc.max());compare(c['native_variance'],reference.mean());compare(c['residual_variance'],residual.mean())
        by_seed={b['seed']:b for b in c['operator_budget']}
        for index,seed in enumerate(p['family']['seeds']):
            v,b=arrays[index],by_seed[seed]
            mean=v['operator_mean'];cache=v[policy+'_cache'];displacement=np.einsum('lhij,lhij->l',mean-cache,mean-cache)/(n*4096)
            compare(b['displacement'],displacement.mean());compare(b['risk'],(v['operator_scatter']+displacement).mean())
            compare(b['scatter'],v['operator_scatter'].mean())
            initial=lookup[n,g,0][index];drift=mean-initial['operator_mean'];offset=initial['operator_mean']-initial['initial_cache']
            compare(b['mean_drift'],np.einsum('lhij,lhij->',drift,drift)/(5*n*4096))
            compare(b['initial_displacement'],np.einsum('lhij,lhij->',offset,offset)/(5*n*4096))
            compare(b['signed_cross'],2*np.einsum('lhij,lhij->',drift,offset)/(5*n*4096))
    # Check the closed-form finite-population transport of native frozen fields.
    # This is a permutation reference law over observed contexts, not adaptive gradients.
    reference=[]
    for n in [8,24]:
        for g in [0.,1.5]:
            for age in [0.,1.]:
                v=lookup[n,g,age][0]['heads'];x=np.stack([v[...,0].mean((1,2)),v[...,2].mean((1,2))],1);x-=x.mean(0)
                C=x.T@x/64;B=8;T=4
                D=np.array([[[1.+k/4.,(-1.)**k*.3],[.2*k,.7]] for k in range(T)])/T
                S=D.sum(0);compact=(64/B*sum(d@C@d.T for d in D)-S@C@S.T)/63
                weights=np.concatenate([np.repeat(d[None]/B,B,axis=0) for d in D],0)
                direct=np.zeros((2,2))
                for i,w in enumerate(weights):
                    for j,z in enumerate(weights):direct+=w@(C if i==j else -C/63)@z.T
                difference=float(abs(compact-direct).max());compare(compact,direct)
                reference.append(dict(heads=n,control=g,age=age,maximum_difference=difference,scaled_covariance=compact.tolist()))
    return dict(status='passed',schema='matched-clock-independent-v2',coverage=coverage,coverage_sha256=canonical(coverage),coverage_sources_sha256=source_hashes(),fields=64,cache_cells=26,maximum_difference=err,
        analysis_sha256=digest(analysis),protocol_sha256=digest(study/'protocol.json'),verifier_sha256=digest(__file__),
        native_frozen_population_controls=reference,scope='Raw-logit pair-distance and saved-operator-moment reconstruction. Finite-population controls condition on the measured fixed context pool; they do not identify adaptive gradient forcing.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    r=verify(a.study,a.analysis);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['native_frozen_population_controls']}))
