#!/usr/bin/env python3
"""Independent pair-distance and raw-moment reconstruction of cache state transfer."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from verify_cache_risk_study import digest, strict_json, finite, strict_difference
from cache_state_contract import (read, validate_protocol, inventory, validate_arrays,
    validate_analysis, analysis_roles, digest as coverage_digest, CONTRACT, DESIGN, VERIFIER_SOURCES, cells)

REPO=Path(__file__).resolve().parents[1]


def predictive(x,y,targets):
    x=finite(x,'native logit array'); y=finite(y,'cached logit array',x.shape)
    lp=x-logsumexp(x,axis=-1,keepdims=True);lq=y-logsumexp(y,axis=-1,keepdims=True)
    a=2*np.exp(lp/2);b=2*np.exp(lq/2);e=a-b;s=len(a)
    def variance(z):
        return sum(np.sum((z[i]-z[j])**2,axis=-1) for i in range(s) for j in range(i))/(s*(s-1))
    va,vb,ve=variance(a),variance(b),variance(e)
    if np.any(va<=0):raise ValueError('Undefined relative variance')
    kl=np.sum(np.exp(lp)*(lp-lq),axis=-1)
    nll=-lp[:,np.arange(len(targets)),targets];nllq=-lq[:,np.arange(len(targets)),targets]
    rr=np.sqrt(ve/va)
    uncentered=np.mean(np.sum(e*e,axis=-1),axis=0)
    bias=np.sum(e.mean(0)**2,axis=-1)
    return dict(uncentered_energy=uncentered.mean(),mean_bias_energy=bias.mean(),
        relative_uncentered_rms=np.sqrt(uncentered.mean()/va.mean()),
        weighted_mass_over_target=va[rr>.25].sum()/va.sum(),
        per_seed_nll_change=(nllq-nll).mean(1),
        relative_centered_rms=np.sqrt(ve.sum()/va.sum()),mean_kl=kl.mean(),maximum_kl=kl.max(),
        native_variance=va.mean(),cached_variance=vb.mean(),residual_variance=ve.mean(),
        signed_cross_covariance=((va-vb-ve)/2).mean(),retained_variance_fraction=vb.sum()/va.sum(),
        native_nll=nll.mean(),cached_nll=nllq.mean(),mean_nll_change=(nllq-nll).mean(),
        contexts_over_target=(rr>.25).sum(),maximum_context_rms=rr.max(),per_context_rms=rr,
        per_seed_mean_kl=kl.mean(1),per_seed_context_kl=kl,per_seed_context_nll_change=nllq-nll,
        per_context_native_variance=va,per_context_residual_variance=ve)


def verify(study,analysis,output):
    if output.exists():raise FileExistsError(output)
    p=validate_protocol(study);a=read(analysis)
    analysis_roles(a,p)
    bound, admitted_counts, admitted_seconds, admitted_peak=inventory(study,p)
    validate_arrays(study,p)
    validate_analysis(a,p,bound)
    strict_difference(admitted_seconds,a['worker_seconds'],'worker time',atol=0,rtol=0)
    strict_difference(admitted_peak,a['peak_allocated_bytes'],'memory',atol=0,rtol=0)
    if a['status']!='passed' or a['protocol_sha256']!=digest(study/'protocol.json'):raise ValueError('Invalid analysis')
    for name,h in a['analysis_sources_sha256'].items():
        if digest(REPO/name)!=h:raise ValueError('Stale analysis source')
    for name,h in a['checked_sha256'].items():
        path=(study/name).resolve()
        if not path.is_relative_to(study) or digest(path)!=h:raise ValueError('Changed observation')
        bound[name]=h
    with np.load(study/'inputs.npz',allow_pickle=False) as z:blocks=z['blocks']
    if blocks.shape!=(80,129):raise ValueError('Wrong input shape')
    expected={(n,s,l,c) for n in [8,24] for s in ['initial','g0','g1.5'] for l in [32,64,128] for c in (['recalibrated'] if s=='initial' else ['recalibrated','initial_cache'])}
    if {(c['heads'],c['state'],c['length'],c['policy']) for c in a['cells']}!=expected or len(a['cells'])!=30:raise ValueError('Incomplete outcomes')
    counts={k:0 for k in p['expected_calls']}
    for job in p['jobs']:
        m=strict_json((study/'runs'/job['run_id']/'manifest.json').read_text())
        if m['status']!='complete' or m['protocol_sha256']!=a['protocol_sha256'] or m['job']!=job:raise ValueError('Foreign acquisition')
        for k in counts:counts[k]+=m['calls'][k]
        with np.load(study/'runs'/job['run_id']/'caches.npz',allow_pickle=False) as ca:
            if set(ca.files)!={f'{s}-L{l}' for s in p['states'] for l in p['prefix_lengths']}:raise ValueError('Wrong cache members')
            for key in ca.files:finite(ca[key],key,(5,3,1,job['heads'],64,64))
    if counts!=p['expected_calls'] or counts!=a['calls']:raise ValueError('Forward counts differ')
    maxp=maxo=0.
    for c in a['cells']:
        jobs=sorted([j for j in p['jobs'] if j['heads']==c['heads']],key=lambda j:j['seed'])
        if [j['seed'] for j in jobs]!=c['seeds'] or len(jobs)!=6:raise ValueError('Seed identity differs')
        xs=[];ys=[];ops={k:[] for k in c['operator_by_seed_layer']};energies=[]
        for job in jobs:
            folder=study/'runs'/job['run_id'];s=c['state'];l=c['length'];policy=c['policy'];shape=(5,job['heads'],64,64)
            x=finite(np.load(folder/f'{s}-L{l}-native.npy',allow_pickle=False),'native',(64,32000))
            y=finite(np.load(folder/f'{s}-L{l}-{policy}.npy',allow_pickle=False),'cached',(64,32000))
            xs.append(x);ys.append(y);dz=x-y;dz-=dz.mean(-1,keepdims=True);energies.append(np.mean(np.sum(dz*dz,axis=-1)))
            with np.load(folder/f'{s}-L{l}-operators.npz',allow_pickle=False) as z,np.load(folder/f'initial-L{l}-operators.npz',allow_pickle=False) as ini,np.load(folder/'caches.npz',allow_pickle=False) as ca:
                mean=finite(z['mean'],'mean',shape);m0=finite(ini['mean'],'initial mean',shape)
                powers=finite(z['powers'],'powers',(5,64))
                if np.any(powers<0):raise ValueError('Negative powers')
                c0=ca[f'initial-L{l}'][:,2,0].astype(float);ct=ca[f'{s}-L{l}'][:,2,0].astype(float);cache=ct if policy=='recalibrated' else c0
                def avg(v):return np.mean(v,axis=(1,2,3))
                scatter=powers.mean(1)-avg(mean*mean)
                displacement=avg(mean*mean)-2*avg(mean*cache)+avg(cache*cache)
                risk=powers.mean(1)-2*avg(mean*cache)+avg(cache*cache)
                drift=avg(mean*mean)-2*avg(mean*m0)+avg(m0*m0)
                initial=avg(m0*m0)-2*avg(m0*c0)+avg(c0*c0)
                cross=2*avg(mean*m0-mean*c0-m0*m0+m0*c0)
                values=dict(scatter=scatter,displacement=displacement,risk=risk,drift=drift,initial_displacement=initial,cross=cross,recalibrated_displacement=avg((mean-ct)**2))
                for key,value in values.items():ops[key].append(value)
                for key,value in [('scatter',scatter),(policy+'-risk',risk),(policy+'-displacement',displacement)]:
                    maxo=max(maxo,strict_difference(value,z[key],key,atol=4e-12,rtol=1e-12))
                if policy=='initial_cache':maxo=max(maxo,strict_difference(displacement,drift+initial+cross,'transport identity',atol=4e-12,rtol=1e-12))
        values=predictive(np.stack(xs),np.stack(ys),blocks[16:,c['length']]);del xs,ys
        for key,value in values.items():maxp=max(maxp,strict_difference(value,c[key],key,atol=4e-12,rtol=2e-10))
        for key,value in ops.items():
            maxo=max(maxo,strict_difference(value,c['operator_by_seed_layer'][key],key,atol=4e-12,rtol=1e-12))
            maxo=max(maxo,strict_difference(np.mean(value),c['operator_means'][key],key,atol=4e-12,rtol=1e-12))
        maxp=max(maxp,strict_difference(energies,c['centered_logit_energy_by_seed'],'logit energy',atol=4e-12,rtol=1e-12))
        if c['targets_met']!=bool(values['relative_centered_rms']<=.25 and values['mean_kl']<=.03):raise ValueError('Target status differs')
    result=dict(status='passed',schema='cache-state-transfer-independent-v2',contract=CONTRACT,design_id=DESIGN,
        coverage_sha256=coverage_digest(bound),external_sha256=p['input_sha256'],
        verification_sources_sha256={n:digest(REPO/n) for n in VERIFIER_SOURCES},
        cell_identities=[list(x) for x in cells()],tolerances=dict(predictive_atol=4e-12,predictive_rtol=2e-10,operator_atol=4e-12,operator_rtol=1e-12),
        cells=30,calls=counts,training_updates=0,
        maximum_predictive_difference=maxp,maximum_operator_difference=maxo,analysis_sha256=digest(analysis),
        protocol_sha256=digest(study/'protocol.json'),verifier_sha256=digest(__file__),helper_sha256=digest(REPO/'scripts/verify_cache_risk_study.py'),checked_sha256=bound)
    payload=json.dumps(result,indent=2,allow_nan=False)+'\n';tmp=output.with_suffix('.tmp');tmp.write_text(payload);tmp.replace(output)
    print({k:v for k,v in result.items() if k!='checked_sha256'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();verify(a.study.resolve(),a.analysis.resolve(),a.output.resolve())
