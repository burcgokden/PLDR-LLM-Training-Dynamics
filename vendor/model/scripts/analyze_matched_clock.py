#!/usr/bin/env python3
"""Complete, CPU-only reduction of the matched physical-clock family."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from model_rg.provenance import sha256,write_json
from numerical_validation import load_json_strict
from matched_clock_coverage import inventory, seed_binding, source_hashes, validate_analysis


def read(path):return load_json_strict(Path(path).read_text())
def probability(z):
    lp=z-logsumexp(z,axis=-1,keepdims=True)
    return lp,2*np.exp(lp/2)

def analyze(study):
    p, inventory_hashes = inventory(study)
    ph=sha256(study/'protocol.json')
    if p['schema']!='matched-clock-onepass-v1' or len(p['jobs'])!=48:raise ValueError('Wrong matched-clock design')
    expected={(n,g,s) for n in [4,8,14,24] for g in [0.,1.,1.5,2.] for s in [9163401,9163402,9163403]}
    if {(j['heads'],j['control'],j['seed']) for j in p['jobs']}!=expected:raise ValueError('Incomplete trajectory grid')
    if sha256(study/'selection.npz')!=p['selection_sha256']:raise ValueError('Changed selection')
    with np.load(study/'selection.npz') as a:order=a['order'];targets=a['probes'][16:,64]
    checked={'protocol.json':ph,'selection.npz':sha256(study/'selection.npz'),'reservation.json':sha256(study/'reservation.json')};counts=dict(training=0,observation=0);seconds=0.;peak=0;ledger=[]
    data={}
    for j in p['jobs']:
        folder=study/'runs'/j['run_id'];m=read(folder/'manifest.json');prefix='runs/'+j['run_id']+'/'
        if m['status']!='complete' or m['job']!=j or m['protocol_sha256']!=ph or m['updates']!=j['steps']:raise ValueError('Incomplete or foreign trajectory '+j['run_id'])
        ages=[0,j['steps']//4,j['steps']//2,j['steps']]
        if set(m['artifacts'])!={f'age-{k}.npz' for k in ages}|{'training.npz','final-state.pt'}:raise ValueError('Incomplete artifact inventory')
        checked[prefix+'manifest.json']=sha256(folder/'manifest.json')
        # CPU reduction uses observations, not the large native replay state.
        for name in [f'age-{k}.npz' for k in ages]+['training.npz']:
            if sha256(folder/name)!=m['artifacts'][name]:raise ValueError('Changed acquisition '+name)
            checked[prefix+name]=m['artifacts'][name]
        with np.load(folder/'training.npz') as a:
            rows=a['document_rows'];loss=a['loss'];norm=a['gradient_norm']
            if rows.shape!=(j['steps'],32) or not np.array_equal(rows.ravel(),order[:rows.size]) or len(np.unique(rows))!=rows.size:raise ValueError('Single-pass source identity failed')
            if loss.shape!=(j['steps'],) or norm.shape!=loss.shape or not np.isfinite(loss).all() or not np.isfinite(norm).all():raise ValueError('Invalid training series')
            clipped=float((norm>1).mean())
        cache_enabled=[j['heads'],j['control']] in p['cache_conditions']
        if m['calls']!=dict(training=j['steps'],observation=104 if cache_enabled else 32):raise ValueError('Call ledger differs')
        for age,k in zip(p['ages'],ages,strict=True):
            with np.load(folder/f'age-{k}.npz') as a:v={n:a[n].astype(float) for n in a.files}
            if v['native'].shape!=(64,32000) or v['heads'].shape!=(64,5,j['heads'],4) or not np.array_equal(v['targets'],targets) or any(not np.isfinite(x).all() for x in v.values()):raise ValueError('Invalid observation geometry/domain')
            data[j['heads'],j['control'],j['seed'],age]=v
        ledger.append(dict(**j,initial_parameter_sha256=m['initial_parameter_sha256'],clipped_fraction=clipped,seconds=m['elapsed_seconds'],peak_allocated_bytes=m['peak_allocated_bytes']))
        for key in counts:counts[key]+=m['calls'][key]
        seconds+=m['elapsed_seconds'];peak=max(peak,m['peak_allocated_bytes'])
    if counts['training']!=76800 or counts['observation']!=2400:raise ValueError('Incomplete acquisition count')
    for n in [4,8,14,24]:
        for seed in [9163401,9163402,9163403]:
            group=[j for j in ledger if j['heads']==n and j['seed']==seed]
            if len({j['initial_parameter_sha256'] for j in group})!=1:raise ValueError('Controls are not initially paired')
            for g in [1.,1.5,2.]:
                if not np.array_equal(data[n,g,seed,0]['native'],data[n,0.,seed,0]['native']):raise ValueError('Initial prediction pairing differs')
    fields=[];cache=[];max_identity=0.;native_population=[]
    for n in [4,8,14,24]:
        for g in [0.,1.,1.5,2.]:
            for age in p['ages']:
                ds=[data[n,g,s,age] for s in [9163401,9163402,9163403]]
                z=np.stack([v['native'] for v in ds]);lp,u=probability(z)
                h=np.stack([v['heads'] for v in ds]);rows=h[...,0].mean((2,3));ops=h[...,2].mean((2,3))
                nll=-lp[:,np.arange(64),targets].mean(1)
                fields.append(dict(heads=n,control=g,age=age,update=int(age*128*n),seed_bindings=[seed_binding(p,inventory_hashes,n,g,age,s) for s in p['family']['seeds']],row_mean=float(rows.mean()),
                    row_seed_means=rows.mean(1).tolist(),row_susceptibility=float(n*np.var(rows,axis=0,ddof=1).mean()),
                    operator_amplitude=float(ops.mean()),operator_susceptibility=float(n*np.var(ops,axis=0,ddof=1).mean()),
                    predictive_susceptibility=float(n*np.var(u,axis=0,ddof=1).sum(-1).mean()),
                    mean_nll=float(nll.mean()),seed_nll=nll.tolist(),maximum_adaptive_force=max(float(v['maximum_adaptive_force']) for v in ds)))
                # Keep all three initialization identities, paired ages/controls. At
                # initialization the g=0 record represents the shared baseline once.
                if [n,g] not in p['cache_conditions'] or (age==0 and g!=0):continue
                for policy in (['state'] if age==0 else ['state','initial']):
                    y=np.stack([v[policy] for v in ds]);lq,v=probability(y);e=u-v
                    var=np.var(u,axis=0,ddof=1).sum(-1);res=np.var(e,axis=0,ddof=1).sum(-1)
                    pair=sum(np.sum((e[i]-e[j])**2,axis=-1) for i in range(3) for j in range(i))/6
                    if np.any(var<=0):raise ValueError('Nonpositive reference variance')
                    max_identity=max(max_identity,float(np.max(abs(res-pair))))
                    r=np.sqrt(res/var);R=float(np.sqrt(res.sum()/var.sum()));kl=float(np.mean(np.sum(np.exp(lp)*(lp-lq),axis=-1)))
                    budget=[]
                    for seed,d in zip(p['family']['seeds'],ds,strict=True):
                        d0=data[n,g,seed,0];mean=d['operator_mean'];c=d[policy+'_cache'];scatter=d['operator_scatter'];risk=d[policy+'_risk'];disp=d[policy+'_displacement']
                        delta=mean-d0['operator_mean'];a=d0['operator_mean']-d0['initial_cache'];drift=np.mean(delta**2,axis=(1,2,3));initial=np.mean(a*a,axis=(1,2,3));cross=2*np.mean(delta*a,axis=(1,2,3))
                        max_identity=max(max_identity,float(abs(risk-scatter-disp).max()))
                        if policy=='initial':max_identity=max(max_identity,float(abs(disp-drift-initial-cross).max()))
                        budget.append(dict(**seed_binding(p,inventory_hashes,n,g,age,seed),risk=float(risk.mean()),scatter=float(scatter.mean()),displacement=float(disp.mean()),mean_drift=float(drift.mean()),initial_displacement=float(initial.mean()),signed_cross=float(cross.mean())))
                    cache.append(dict(heads=n,control=None if age==0 else g,age=age,policy=policy,relative_centered_rms=R,mean_kl=kl,
                        native_nll=float(-lp[:,np.arange(64),targets].mean()),cached_nll=float(-lq[:,np.arange(64),targets].mean()),
                        mean_nll_change=float((lp-lq)[:,np.arange(64),targets].mean()),contexts_over_target=int((r>.25).sum()),maximum_context_rms=float(r.max()),
                        per_context_rms=r.tolist(),native_variance=float(var.mean()),residual_variance=float(res.mean()),
                        aggregate_targets_met=bool(R<=.25 and kl<=.03),operator_budget=budget))
    if len(cache)!=26 or max_identity>2e-10:raise ValueError('Incomplete cache cells or identity disagreement')
    q=read(study/'qualification/manifest.json');checked['qualification/manifest.json']=sha256(study/'qualification/manifest.json')
    if q['status']!='complete' or q['optimizer_updates']!=3 or q['forwards']!=6:raise ValueError('Qualification counts differ')
    if checked != inventory_hashes:raise ValueError('Observation census changed during reduction')
    result = dict(status='passed',schema='matched-clock-analysis-v2',protocol_sha256=ph,checked_sha256=checked,
        scientific_updates=counts['training'],scientific_observation_forwards=counts['observation'],qualification_updates=3,qualification_forwards=6,
        native_forward_calls=sum(counts.values())+6,worker_seconds=seconds,peak_allocated_bytes=peak,trajectories=ledger,
        fields=fields,cache_cells=cache,maximum_identity_error=max_identity,analysis_source_sha256=sha256(__file__),coverage_sources_sha256=source_hashes(),
        scope='Finite conditional family with three independent wide initializations per cell, paired controls and fixed corpus/order; no fitted critical exponents or population target coverage.')

    validate_analysis(result,p)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    r=analyze(a.study);write_json(a.output,r);print(dict(status=r['status'],trajectories=len(r['trajectories']),cache_cells=len(r['cache_cells'])))
