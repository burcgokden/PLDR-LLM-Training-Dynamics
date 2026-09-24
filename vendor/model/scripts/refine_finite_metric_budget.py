"""Nested midpoint bounds for finite categorical metric integration.

The grid (1, 4, 16, 64 intervals) is fixed before this follow-up run. All
eight partial-row-state contrasts on both suffixes/probes are retained.
This is a refinement of an observation calculation, not native simulation.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import logsumexp
if __package__:
    from .analyze_finite_metric_budget import ROOT, digest, budget
else:
    from analyze_finite_metric_budget import ROOT, digest, budget

def midpoint_bounds(a, intervals):
    origin=a[:-1];d=np.diff(a,axis=0);radius=np.ptp(d,axis=-1)
    total=np.zeros(len(d))
    for j in range(intervals):
        s=(j+.5)/intervals
        logits=origin+s*d
        p=np.exp(logits-logsumexp(logits,axis=-1,keepdims=True))
        mu=np.sum(p*d,axis=-1,keepdims=True)
        variance=np.sum(p*(d-mu)**2,axis=-1)
        total+=(1-s)/intervals*variance
    envelope=np.exp(radius/(2*intervals))
    return total/envelope,total*envelope

def main(output):
    if output.exists():raise FileExistsError(output)
    start=time.monotonic();records=[];identities={}
    for suffix,dirname in [('A','potential-factorial-20260913'),('B','potential-factorial-disjoint-20260914')]:
        folder=ROOT/dirname/'runs/subcritical1'
        mf=folder/'manifest.json';manifest=json.loads(mf.read_text());identities[str(mf)]=digest(mf)
        for moment in ['keep','reset']:
            series=[]
            for offset in ['native','raised']:
                branch=offset+'_'+moment;raw=folder/(branch+'.npz');identities[str(raw)]=digest(raw)
                if identities[str(raw)]!=manifest['branches'][branch]['artifact_sha256']:
                    raise ValueError('Changed saved branch')
                with np.load(raw,allow_pickle=False) as z:series.append(z['step_logits'].astype(np.float64))
            for probe in range(2):
                a,b=(z[:,probe] for z in series)
                va=budget(a[:-1],a[1:]);vb=budget(b[:-1],b[1:])
                delta=float(vb['kl'].sum()-va['kl'].sum())
                lower=float(vb['lower'].sum()-va['upper'].sum())
                upper=float(vb['upper'].sum()-va['lower'].sum())
                levels=[]
                for intervals in [1,4,16,64]:
                    la,ua=midpoint_bounds(a,intervals);lb,ub=midpoint_bounds(b,intervals)
                    # Intersect valid enclosures, guaranteeing nested accepted intervals.
                    lower=max(lower,float(lb.sum()-ua.sum()))
                    upper=min(upper,float(ub.sum()-la.sum()))
                    if delta < lower-4e-10 or delta > upper+4e-10:
                        raise ValueError('Reference outside refinement')
                    levels.append(dict(intervals=intervals,lower=lower,upper=upper,
                        width=upper-lower,sign_resolved_at_1e_minus_8=bool(lower>1e-8 or upper< -1e-8)))
                records.append(dict(suffix=suffix,moment=moment,probe=probe+1,delta_kl=delta,levels=levels))
                print(suffix,moment,probe+1,delta,levels[-1],flush=True)
    result=dict(status='passed',schema='finite-metric-refinement-v1',records=records,comparisons=len(records),
        levels=[1,4,16,64],resolved_by_level={str(n):sum(next(x for x in r['levels'] if x['intervals']==n)['sign_resolved_at_1e_minus_8'] for r in records) for n in [1,4,16,64]},
        checked_sha256=identities,script_sha256=digest(__file__),seconds=time.monotonic()-start,
        new_training_updates=0,scope='All eight fixed partial-row-state contrasts; finite observation refinement in float64, not rigorous floating-point interval arithmetic or prediction of unobserved logits.')
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','checked_sha256']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)
