#!/usr/bin/env python3
"""Exact empirical endpoint profiles and an independent centered-sum check."""
from companion_paths import legacy_path
import argparse
import json
from pathlib import Path
import numpy as np
from model_rg.critical_resampling import replica_distances,resampled_variance,replica_counts,exact_replica_counts,weighted_inverse_cdf
from model_rg.provenance import sha256,write_json
from analyze_critical_joint import peak_draws
ROOT=Path(legacy_path('/pldr-data/model'))
STUDY=ROOT/'critical-onepass-refinement-20260914'


def analyze(output,uncertainty):
    output.mkdir(parents=True,exist_ok=False)
    p=json.loads((STUDY/'protocol.json').read_text());seeds=sorted(p['design']['seeds']);controls=sorted(p['design']['controls'])
    counts,mass=exact_replica_counts(len(seeds));mc=replica_counts(len(seeds))
    joint=json.loads((uncertainty/'refinement/joint-analysis.json').read_text())
    checked={str(STUDY/'protocol.json'):sha256(STUDY/'protocol.json'),str(uncertainty/'refinement/joint-analysis.json'):sha256(uncertainty/'refinement/joint-analysis.json')}
    vectors={};max_error=0.
    for job in p['jobs']:
        root=STUDY/'runs'/job['run_id'];m=json.loads((root/'manifest.json').read_text());path=root/'observations.npz'
        if m['status']!='complete' or m['job']!=job or sha256(path)!=m['artifacts']['observations.npz']:raise ValueError('Unbound endpoint')
        checked[str(path)]=m['artifacts']['observations.npz']
        with np.load(path) as a:
            for t in [2048,4096]:
                k=np.flatnonzero(a['steps']==t)
                if len(k)!=1:raise ValueError('Missing endpoint')
                vectors[(job['heads'],job['control'],job['seed'],t)]=a['heads'][k[0],...,0].mean(-1).reshape(-1)
    profiles=[]
    for n in sorted(p['design']['heads']):
        for t in [2048,4096]:
            exact=[];sampled=[];original=[]
            for g in controls:
                x=np.stack([vectors[n,g,s,t] for s in seeds]);dist=replica_distances(x)
                exact.append(n*resampled_variance(dist,counts));sampled.append(n*resampled_variance(dist,mc))
                original.append(float(n*np.var(x,axis=0,ddof=1).mean()))
                # Weighted centered sums independently check every count vector.
                for count,value in zip(counts,exact[-1]):
                    mean=count@x/len(seeds)
                    ref=n*np.sum(count[:,None]*(x-mean)**2)/((len(seeds)-1)*x.shape[1])
                    max_error=max(max_error,abs(ref-value))
                    if abs(ref-value)>2e-12:raise ValueError('Independent count identity failed')
            peak=peak_draws(controls,np.stack(exact,axis=1));monte=peak_draws(controls,np.stack(sampled,axis=1))
            regular=peak_draws(controls,np.asarray(original)[None])
            zero=peak['peak']==0
            if int(mass[zero].sum())!=6:raise ValueError('Unexpected degenerate resampling mass')
            resolved=np.isfinite(peak['width'])
            if int(mass[~resolved].sum())!=6:raise ValueError('Additional unresolved exact profile')
            found=[r for r in joint['profiles'] if r['heads']==n and r['step']==t and r['field']=='row']
            if len(found)!=1 or found[0]['half_width_resolved_fraction']!=float(np.isfinite(monte['width']).mean()):raise ValueError('Joint caller mismatch')
            profiles.append(dict(heads=n,step=t,controls=controls,susceptibility=original,
                intensive_variance=(np.asarray(original)/n).tolist(),sampled_peak_control=float(regular['controls'][0]),
                width=float(regular['width'][0]),exact_resolved_probability=float(mass[resolved].sum()/mass.sum()),
                exact_width_percentiles=weighted_inverse_cdf(peak['width'],mass),
                monte_carlo_resolved_fraction=found[0]['half_width_resolved_fraction'],
                monte_carlo_width_percentiles=found[0]['resolved_half_width_percentile_95']))
    write_json(output/'analysis.json',dict(status='complete',profiles=profiles,checked_sha256=checked,
        observed_paths=len(p['jobs']),count_vectors=len(counts),ordered_draws=int(mass.sum()),
        unresolved_mass_numerator=6,maximum_centered_identity_error=max_error,scientific_updates=0,
        source_sha256=sha256(__file__),units='N times per-context/decoder row covariance after head averaging.',
        exact_quantile='Weighted inverse CDF conditional on resolved width; no population coverage assertion.'))
    print('Eight exact profiles completed; 216 observation files checked.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--uncertainty',type=Path,required=True)
    a=p.parse_args();analyze(a.output,a.uncertainty)
