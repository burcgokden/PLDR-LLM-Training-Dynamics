#!/usr/bin/env python3
"""Independent pair-distance and log-sum-exp reconstruction of saved cache outputs."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256,write_json
from numerical_validation import load_json_strict,discrepancy,finite_array

REPO=Path(__file__).resolve().parents[1]

def verify(study,analysis,output):
    if output.exists():raise FileExistsError(output)
    a=load_json_strict(analysis.read_text());p=load_json_strict((study/'protocol.json').read_text())
    if a['schema']!='operator-cache-analysis-v1' or a['status']!='passed':raise ValueError('Unverified analysis')
    if a['protocol_sha256']!=sha256(study/'protocol.json') or a['source_sha256']!=sha256(REPO/'scripts/analyze_operator_cache.py'):
        raise ValueError('Changed analysis provenance')
    checked={str(analysis):sha256(analysis)}
    for path,digest in a['checked_sha256'].items():
        # Resolve packaged native records relative to the selected study.
        path=Path(path)
        original=Path(a['study'])
        if path.is_relative_to(original):path=study/path.relative_to(original)
        if sha256(path)!=digest:raise ValueError('Changed verified input '+str(path))
        checked[str(path)]=digest
    with np.load(study/'inputs.npz') as z:targets=z['blocks'][16:,64]
    grouped=defaultdict(list)
    for job in p['jobs']:grouped[job['heads'],job['control']].append(job)
    official={(r['heads'],r['control']):r for r in a['cells']}
    if set(official)!=set(grouped) or len(a['cells'])!=4:raise ValueError('Incomplete aggregate inventory')
    maxerr=0.;count=0;reports=[];worker_seconds=0.
    counts=dict(calibration=0,assessment=0,timing=0,qualification=0)
    paths={r['run_id']:r for r in a['paths']}
    if len(paths)!=24:raise ValueError('Incomplete timing inventory')
    for key,jobs in sorted(grouped.items()):
        lp=[];lq=[];speeds=[];native_ms=[];cached_ms=[]
        for job in sorted(jobs,key=lambda j:j['seed']):
            folder=study/'runs'/job['run_id']
            with np.load(folder/'observations.npz') as z:
                for name,dest in [('native',lp),('cached',lq)]:
                    raw=finite_array(z[name],name).astype(float)
                    # NumPy's pairwise logaddexp reduction is independent of the
                    # production max/subtract/exponentiate implementation.
                    dest.append(raw-np.logaddexp.reduce(raw,axis=-1)[:,None])
            m=load_json_strict((folder/'manifest.json').read_text());r=paths[job['run_id']]
            if m['status']!='complete' or m['job']!=job or m['protocol_sha256']!=sha256(study/'protocol.json') or m['training_updates']!=0:
                raise ValueError('Incomplete or foreign native manifest')
            worker_seconds+=m['elapsed_seconds']
            for category in counts:counts[category]+=m['calls'][category]
            with np.load(folder/'cache.npz') as cache_record:
                cache=finite_array(cache_record['cache'],'operator cache')
                if cache.shape!=(5,3,1,job['heads'],64,64) or cache.dtype!=np.float32 or cache.nbytes!=m['cache_bytes']:
                    raise ValueError('Cache storage metadata differs')
            with np.load(folder/'timing.npz') as t:
                n=float(np.median(t['native']));c=float(np.median(t['cached']));speed=1-c/n
                expected=dict(native_median_seconds=n,cached_median_seconds=c,latency_reduction=speed,
                    native_peak_bytes=int(t['native_peak_bytes'].max()),cached_peak_bytes=int(t['cached_peak_bytes'].max()),
                    cache_build_seconds=m['cache_build_seconds'],cache_bytes=m['cache_bytes'],
                    amortization_requests=m['cache_build_seconds']/(n-c) if n>c else None)
                for name,value in expected.items():
                    if value is None:
                        if r[name] is not None:raise ValueError('False amortization')
                    else:maxerr=max(maxerr,discrepancy(value,r[name],name,atol=3e-12))
                for name,value in [('native_seconds',t['native']),('cached_seconds',t['cached']),('paired_latency_reduction',1-t['cached']/t['native'])]:
                    if not np.array_equal(np.asarray(r[name]),value):raise ValueError('Timing distribution differs')
                speeds.append(speed);native_ms.append(1000*n);cached_ms.append(1000*c)
        lp=np.stack(lp);lq=np.stack(lq);x=2*np.exp(lp/2);y=2*np.exp(lq/2);e=x-y;s=len(x)
        def pairs(z):return sum(np.sum((z[i]-z[j])**2,-1) for i in range(s) for j in range(i))/(s*(s-1))
        vp,vq,ve=pairs(x),pairs(y),pairs(e)
        cross=sum(np.sum((y[i]-y[j])*(e[i]-e[j]),-1) for i in range(s) for j in range(i))/(s*(s-1))
        u=np.mean(np.sum(e*e,-1),0);bias=np.sum(np.mean(e,0)**2,-1)
        kl=np.sum(np.exp(lp)*(lp-lq),-1);delta=(lp-lq)[:,np.arange(128),targets]
        risk=np.sqrt(ve/vp);w=vp/vp.sum();o=official[key]
        expected=dict(native_variance=float(vp.mean()),cached_variance=float(vq.mean()),residual_variance=float(ve.mean()),
            retained_variance_fraction=float(vq.sum()/vp.sum()),relative_centered_rms=float(np.sqrt(ve.sum()/vp.sum())),
            signed_cross_covariance=float(cross.mean()),uncentered_energy=float(u.mean()),mean_bias_energy=float(bias.mean()),
            relative_uncentered_rms=float(np.sqrt(u.mean()/vp.mean())),mean_kl=float(kl.mean()),maximum_kl=float(kl.max()),
            native_nll=float(-lp[:,np.arange(128),targets].mean()),cached_nll=float(-lq[:,np.arange(128),targets].mean()),
            mean_nll_change=float(delta.mean()),maximum_context_rms=float(risk.max()),contexts_over_target=int(np.sum(risk>.25)),
            weighted_mass_over_target=float(w[risk>.25].sum()),median_latency_reduction=float(np.median(speeds)),
            minimum_latency_reduction=min(speeds),median_native_ms=float(np.median(native_ms)),median_cached_ms=float(np.median(cached_ms)))
        for name,value in expected.items():maxerr=max(maxerr,discrepancy(value,o[name],name,atol=3e-12))
        arrays=dict(per_context_rms=risk,per_context_native_variance=vp,per_context_residual_variance=ve,
            per_seed_mean_kl=kl.mean(1),per_seed_nll_change=delta.mean(1),per_seed_context_kl=kl,per_seed_context_nll_change=delta)
        for name,value in arrays.items():
            saved=finite_array(o[name],name)
            if saved.shape!=value.shape:raise ValueError('Array shape differs')
            error=float(np.max(np.abs(saved-value)));maxerr=max(maxerr,error)
            if error>3e-12:raise ValueError('Saved distribution differs: '+name)
        met=bool(expected['relative_centered_rms']<=.25 and expected['mean_kl']<=.03 and expected['median_latency_reduction']>=.1)
        if o['targets_met'] is not met:raise ValueError('Scientific target differs')
        count+=int(met);reports.append(dict(heads=key[0],control=key[1],targets_met=met))
    if a['scientific_target_cells']!=count or a['calls']!=p['expected_calls'] or counts!=p['expected_calls']:
        raise ValueError('Aggregate count differs')
    discrepancy(worker_seconds,a['worker_seconds'],'worker seconds',atol=3e-12)
    write_json(output,dict(status='passed',schema='operator-cache-verification-v1',cells=4,context_cells=512,
        paired_context_predictions=3072,timing_pairs=720,scientific_target_cells=count,maximum_absolute_error=maxerr,
        checked_sha256=checked,verifier_sha256=sha256(__file__),analysis_sha256=sha256(analysis),
        scope='Independent saved-output probabilities, pairwise covariances, target risks, context distributions and timing statistics. No new native calls or training updates.'))
    print('Verified four cells, 512 context cells, 3072 prediction pairs and 720 timing pairs; maximum discrepancy',maxerr)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify(a.study.resolve(),a.analysis.resolve(),a.output.resolve())
