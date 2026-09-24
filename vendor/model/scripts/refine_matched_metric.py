#!/usr/bin/env python3
"""Controlled one-percent refinement of all fresh full-horizon logit pairs.

This is a complete follow-up observation calculation, not a new training cohort.
The segment count follows the proved oscillation rule at fixed tolerance 0.01.
"""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.special import logsumexp
from model_rg.provenance import sha256, write_json


def main(study, output):
    if output.exists(): raise FileExistsError(output)
    start=time.monotonic();spec=json.loads((study/'protocol.json').read_text())
    checked={str(study/'protocol.json'):sha256(study/'protocol.json')};records=[]
    tolerance=.01
    for job in spec['jobs']:
        folder=study/'runs'/job['run_id'];mp=folder/'manifest.json';m=json.loads(mp.read_text())
        raw=folder/'observations.npz'
        if m['status']!='complete' or m['job']!=job or sha256(raw)!=m['artifacts']['observations.npz']:
            raise ValueError('Changed or incomplete scientific input')
        checked[str(mp)]=sha256(mp);checked[str(raw)]=sha256(raw)
        with np.load(raw,allow_pickle=False) as z: endpoints=z['logits'][[0,-1]].astype(np.float64)
        a,b=endpoints;d=b-a;radius=np.ptp(d,axis=-1)
        # One common segment count per job, valid for all eight fixed probes.
        count=max(1,math.ceil(float(radius.max())/(2*math.log1p(tolerance))))
        approx=np.zeros(8)
        for j in range(count):
            s=(j+.5)/count;ell=a+s*d
            p=np.exp(ell-logsumexp(ell,axis=-1,keepdims=True))
            centered=d-np.sum(p*d,axis=-1,keepdims=True)
            approx+=(1-s)/count*np.sum(p*centered**2,axis=-1)
        la=a-logsumexp(a,axis=-1,keepdims=True);lb=b-logsumexp(b,axis=-1,keepdims=True)
        kl=np.sum(np.exp(la)*(la-lb),axis=-1)
        envelope=np.exp(radius/(2*count));lower=approx/envelope;upper=approx*envelope
        if np.any(kl<=1e-12) or not np.isfinite(approx).all() or np.any(envelope>1.01+1e-14):
            raise ValueError('Invalid one-percent domain')
        if np.any(lower>kl+2e-12*(1+kl)) or np.any(upper<kl-2e-12*(1+kl)):
            raise ValueError('Refined interval violation')
        for i in range(8):
            records.append(dict(run_id=job['run_id'],probe=i+1,intervals=count,
                oscillation=float(radius[i]),kl=float(kl[i]),midpoint=float(approx[i]),
                lower=float(lower[i]),upper=float(upper[i]),
                relative_error=float(abs(approx[i]/kl[i]-1)),relative_envelope=float(envelope[i]-1)))
        print(job['run_id'],count,'segments',flush=True)
    if len(records)!=192: raise ValueError('Incomplete full-horizon cohort')
    write_json(output,dict(status='passed',schema='matched-metric-refinement-v1',
        records=records,endpoint_pairs=192,scientific_training_updates=0,
        maximum_relative_error=max(r['relative_error'] for r in records),
        maximum_relative_envelope=max(r['relative_envelope'] for r in records),
        intervals_range=[min(r['intervals'] for r in records),max(r['intervals'] for r in records)],
        tolerance=tolerance,checked_sha256=checked,script_sha256=sha256(__file__),
        seconds=time.monotonic()-start,
        scope='Complete follow-up analysis of acquired 1024-update endpoints. Proved real-arithmetic one-percent budget evaluated in float64; neither an interval-arithmetic certificate nor an economical autonomous forecast.'))
    print('PASS 192 one-percent endpoint enclosures',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.study,a.output)
