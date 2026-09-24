#!/usr/bin/env python3
"""Reconstruct every paired cache cell, covariance budget, context risk and timing."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256,write_json
from numerical_validation import load_json_strict,finite_array,discrepancy

REPO=Path(__file__).resolve().parents[1]

def log_probabilities(z):
    z=finite_array(z,'logits').astype(float);z=z-z.max(-1,keepdims=True)
    return z-np.log(np.exp(z).sum(-1,keepdims=True))

def paired_metrics(native,cached,targets):
    if native.shape!=cached.shape or native.ndim!=3 or native.shape[0]<2 or native.shape[1]<1:
        raise ValueError('Matched replica/context/vocabulary arrays required')
    s,c,v=native.shape
    targets=np.asarray(targets)
    if targets.shape!=(c,) or targets.dtype.kind not in 'iu' or np.any(targets<0) or np.any(targets>=v):
        raise ValueError('External target shape/domain differs')
    lp=log_probabilities(native);lq=log_probabilities(cached);p=np.exp(lp);q=np.exp(lq)
    x=2*np.sqrt(p);y=2*np.sqrt(q);e=x-y
    def var(z):return np.sum((z-z.mean(0))**2,axis=(0,2))/(s-1)
    vp,vq,ve=var(x),var(y),var(e)
    if np.any(vp<=0):raise ValueError('Relative error undefined at zero native variance')
    cross=np.sum((y-y.mean(0))*(e-e.mean(0)),axis=(0,2))/(s-1)
    uncentered=np.mean(np.sum(e*e,axis=-1),axis=0);bias=np.sum(e.mean(0)**2,axis=-1)
    kl=np.sum(p*(lp-lq),axis=-1)
    pair=sum(np.sum((x[i]-x[j])**2,axis=-1) for i in range(s) for j in range(i))/(s*(s-1))
    errors=[float(np.max(np.abs(vp-pair))),float(np.max(np.abs(vp-vq-ve-2*cross))),
            float(np.max(np.abs(ve-s/(s-1)*(uncentered-bias))))]
    if max(errors)>3e-12 or np.min(kl)<-3e-12 or np.any(uncentered>4*kl.mean(0)+3e-12):
        raise ValueError('Independent pair, covariance, bias or KL identity failed')
    risk=np.sqrt(ve/vp);weights=vp/vp.sum();rms=float(np.sqrt(ve.sum()/vp.sum()))
    discrepancy(float(np.dot(weights,risk**2)),rms*rms,'context-weighted risk',atol=3e-12)
    nllp=-np.take_along_axis(lp,np.broadcast_to(targets[None,:,None],(s,c,1)),axis=2)[...,0]
    nllq=-np.take_along_axis(lq,np.broadcast_to(targets[None,:,None],(s,c,1)),axis=2)[...,0]
    return dict(native_variance=float(vp.mean()),cached_variance=float(vq.mean()),residual_variance=float(ve.mean()),
        retained_variance_fraction=float(vq.sum()/vp.sum()),relative_centered_rms=rms,
        signed_cross_covariance=float(cross.mean()),uncentered_energy=float(uncentered.mean()),
        mean_bias_energy=float(bias.mean()),relative_uncentered_rms=float(np.sqrt(uncentered.mean()/vp.mean())),
        mean_kl=float(kl.mean()),maximum_kl=float(kl.max()),native_nll=float(nllp.mean()),cached_nll=float(nllq.mean()),
        mean_nll_change=float((nllq-nllp).mean()),per_seed_mean_kl=kl.mean(1).tolist(),
        per_seed_nll_change=(nllq-nllp).mean(1).tolist(),per_context_rms=risk.tolist(),
        maximum_context_rms=float(risk.max()),contexts_over_target=int(np.sum(risk>.25)),
        weighted_mass_over_target=float(weights[risk>.25].sum()),
        per_context_native_variance=vp.tolist(),per_context_residual_variance=ve.tolist(),
        per_seed_context_kl=kl.tolist(),per_seed_context_nll_change=(nllq-nllp).tolist(),
        maximum_algebra_error=max(errors))

def analyze(study,output):
    if output.exists():raise FileExistsError(output)
    checked={}
    def bind(path,expected=None):
        h=sha256(path)
        if expected is not None and h!=expected:raise ValueError('Changed input '+str(path))
        checked[str(path)]=h
        return h
    p=load_json_strict((study/'protocol.json').read_text());ph=bind(study/'protocol.json')
    if p['schema']!='operator-cache-v1' or p['status']!='frozen_before_acquisition':raise ValueError('Wrong frozen protocol')
    for name,digest in p['source_sha256'].items():bind(REPO/name,digest)
    bind(study/'inputs.npz',p['selection_sha256'])
    if len(set(p['document_hashes']))!=144:raise ValueError('Source overlap')
    with np.load(study/'inputs.npz') as a:blocks=a['blocks']
    if blocks.shape!=(144,65):raise ValueError('Unexpected prefix/target panel')
    targets=blocks[16:,64];groups=defaultdict(list)
    for job in p['jobs']:groups[job['heads'],job['control']].append(job)
    if set(groups)!={(8,0),(8,1.5),(24,0),(24,1.5)}:raise ValueError('Incomplete condition grid')
    cells=[];paths=[];counts=dict(calibration=0,assessment=0,timing=0,qualification=0);seconds=0
    for (heads,control),jobs in sorted(groups.items()):
        if len(jobs)!=6 or len({j['seed'] for j in jobs})!=6:raise ValueError('Incomplete initialization block')
        native=[];cached=[];timing=[]
        for job in sorted(jobs,key=lambda j:j['seed']):
            folder=study/'runs'/job['run_id'];mp=folder/'manifest.json';m=load_json_strict(mp.read_text());bind(mp)
            if m['status']!='complete' or m['job']!=job or m['protocol_sha256']!=ph or m['training_updates']!=0:
                raise ValueError('Foreign/incomplete observation')
            for name,digest in m['artifacts'].items():bind(folder/name,digest)
            if m['qualification']!=dict(exact_own_operator_replay=True,exact_native_restoration=True,native_generator_calls=40,cached_generator_calls=0):
                raise ValueError('Missing native qualification')
            with np.load(folder/'observations.npz') as a:
                x=a['native'];y=a['cached']
                if x.shape!=(128,p['vocabulary']) or y.shape!=x.shape:raise ValueError('Unexpected output shape')
                native.append(x);cached.append(y)
            with np.load(folder/'timing.npz') as a:
                tn=finite_array(a['native'],'native timings');tc=finite_array(a['cached'],'cached timings')
                if tn.shape!=(30,) or tc.shape!=(30,) or np.any(tn<=0) or np.any(tc<=0):raise ValueError('Invalid timing pairs')
                n=float(np.median(tn));c=float(np.median(tc));saving=n-c
                t=dict(run_id=job['run_id'],heads=heads,control=control,seed=job['seed'],
                    native_median_seconds=n,cached_median_seconds=c,latency_reduction=1-c/n,
                    paired_latency_reduction=(1-tc/tn).tolist(),native_seconds=tn.tolist(),cached_seconds=tc.tolist(),
                    native_peak_bytes=int(a['native_peak_bytes'].max()),cached_peak_bytes=int(a['cached_peak_bytes'].max()),
                    cache_build_seconds=m['cache_build_seconds'],cache_bytes=m['cache_bytes'],
                    amortization_requests=m['cache_build_seconds']/saving if saving>0 else None)
            paths.append(t);timing.append(t);seconds+=m['elapsed_seconds']
            for key in counts:counts[key]+=m['calls'][key]
        r=paired_metrics(np.stack(native),np.stack(cached),targets)
        r.update(heads=heads,control=control,replicas=6,contexts=128,
            median_latency_reduction=float(np.median([t['latency_reduction'] for t in timing])),
            minimum_latency_reduction=min(t['latency_reduction'] for t in timing),
            median_native_ms=1000*float(np.median([t['native_median_seconds'] for t in timing])),
            median_cached_ms=1000*float(np.median([t['cached_median_seconds'] for t in timing])))
        limits=p['primary_targets'];r['targets_met']=bool(r['relative_centered_rms']<=limits['centered_rms'] and r['mean_kl']<=limits['mean_kl_nats'] and r['median_latency_reduction']>=limits['median_latency_reduction'])
        cells.append(r)
    if counts!=p['expected_calls']:raise ValueError('Call accounting differs')
    result=dict(status='passed',schema='operator-cache-analysis-v1',study=str(study),protocol_sha256=ph,
        cells=cells,paths=paths,calls=counts,worker_seconds=seconds,new_training_updates=0,new_training_replicas=0,
        scientific_target_cells=sum(r['targets_met'] for r in cells),checked_sha256=checked,
        source_sha256=sha256(__file__),scope='Fixed proper-prefix panel and six paired initialization replicas per condition. Model-call timings at batch eight, no autoregressive decoding throughput claim.')
    write_json(output,result)
    print({k:result[k] for k in ['status','scientific_target_cells','calls','worker_seconds']},flush=True)
    for r in cells:print({k:r[k] for k in ['heads','control','relative_centered_rms','mean_kl','retained_variance_fraction','mean_nll_change','median_latency_reduction','maximum_context_rms','targets_met']},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.study.resolve(),a.output.resolve())
