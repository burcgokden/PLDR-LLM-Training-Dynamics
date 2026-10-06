"""Independent finite logit-oscillation bounds on retained native predictions.

No project reducer or quadrature is imported. This is a new analysis of retained
trajectories, not new training or independent model replication.
"""
from companion_paths import configured_path
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.special import logsumexp

ROOT = Path(configured_path('data:model'))
def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()

def factor(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-3
    out[small] = sum(2*x[small]**k/math.factorial(k+2) for k in range(8))
    out[~small] = 2*(np.expm1(x[~small])-x[~small])/x[~small]**2
    return out

def budget(a, b):
    d = b-a
    la = a-logsumexp(a, axis=-1, keepdims=True)
    lb = b-logsumexp(b, axis=-1, keepdims=True)
    p = np.exp(la)
    mu = np.sum(p*d, axis=-1, keepdims=True)
    q = .5*np.sum(p*(d-mu)**2, axis=-1)
    kl = np.sum(p*(la-lb), axis=-1)
    radius = np.ptp(d, axis=-1)
    lower, upper = factor(-radius)*q, factor(radius)*q
    return dict(kl=kl, quadratic=q, oscillation=radius, lower=lower, upper=upper)

def main(out):
    if out.exists():
        raise FileExistsError(out)
    start = time.monotonic()
    cells, identities = [], {}
    maximum_violation = 0.
    maximum_gauge = 0.
    for suffix, dirname in [('A', 'potential-factorial-20260913'), ('B', 'potential-factorial-disjoint-20260914')]:
        study = ROOT/dirname
        protocol = study/'protocol.json'
        identities[str(protocol)] = digest(protocol)
        spec = json.loads(protocol.read_text())
        for case in spec['cases']:
            path = study/'runs'/case['name']
            mf = path/'manifest.json'
            identities[str(mf)] = digest(mf)
            manifest = json.loads(mf.read_text())
            for branch in ['native_keep', 'raised_keep', 'native_reset', 'raised_reset']:
                raw = path/(branch+'.npz')
                identities[str(raw)] = digest(raw)
                if identities[str(raw)] != manifest['branches'][branch]['artifact_sha256']:
                    raise ValueError('Changed scientific input')
                with np.load(raw, allow_pickle=False) as z:
                    logits = z['step_logits'].astype(np.float64)
                if logits.shape != (129, 2, 32000):
                    raise ValueError(('Unexpected saved logit domain', logits.shape))
                for probe in range(2):
                    series = logits[:, probe]
                    for block in [1, 2, 4, 8, 16, 32, 64, 128]:
                        values = budget(series[:-block:block], series[block::block])
                        tol = 2e-12 + 2e-12*np.abs(values['kl'])
                        violation = np.maximum(values['lower']-values['kl'], values['kl']-values['upper'])
                        maximum_violation = max(maximum_violation, float(np.max(violation)))
                        if not np.all(np.isfinite(np.array(list(values.values())))) or np.any(violation > tol):
                            raise ValueError(('Finite budget failed', suffix, case['name'], branch, probe, block))
                        cells.append(dict(suffix=suffix, case=case['name'], branch=branch, probe=probe+1,
                                          block=block, count=len(values['kl']), **{k: v.tolist() for k,v in values.items()}))
                    # Distinct endpoint gauges on two chosen native spans.
                    base = budget(series[[0,64]], series[[1,128]])
                    shifted = budget(series[[0,64]]+np.array([[3.0],[-2.0]]), series[[1,128]]+np.array([[-4.0],[5.0]]))
                    maximum_gauge = max(maximum_gauge, max(float(np.max(np.abs(base[k]-shifted[k]))) for k in base))
                print(suffix, case['name'], branch, 'checked both probes and eight block scales', flush=True)
    summaries = []
    for probe in [1,2]:
        for block in [1,2,4,8,16,32,64,128]:
            rows = [r for r in cells if r['probe']==probe and r['block']==block]
            cat = lambda name: np.concatenate([np.array(r[name]) for r in rows])
            kl,q,radius = cat('kl'),cat('quadratic'),cat('oscillation')
            active = q>1e-12
            relative = np.abs(kl[active]/q[active]-1)
            bound_relative = np.maximum(factor(radius)-1,1-factor(-radius))
            summaries.append(dict(probe=probe, block=block, spans=len(kl), active_spans=int(active.sum()),
                maximum_absolute_error=float(np.max(np.abs(kl-q))),
                median_relative_error=float(np.median(relative)), maximum_relative_error=float(np.max(relative)),
                maximum_oscillation=float(radius.max()), median_oscillation=float(np.median(radius)),
                guaranteed_one_percent=int((bound_relative<=.01).sum()), observed_one_percent=int((relative<=.01).sum())))
    contrasts = []
    for suffix in ['A','B']:
        for case in spec['cases']:
            for probe in [1,2]:
                for moment in ['keep','reset']:
                    rows = {r['branch']:r for r in cells if r['suffix']==suffix and r['case']==case['name'] and r['probe']==probe and r['block']==1}
                    native,raised = rows['native_'+moment],rows['raised_'+moment]
                    lo = sum(raised['lower'])-sum(native['upper'])
                    hi = sum(raised['upper'])-sum(native['lower'])
                    actual = sum(raised['kl'])-sum(native['kl'])
                    if actual < lo-4e-10 or actual > hi+4e-10:
                        raise ValueError('Contrast outside summed bounds')
                    contrasts.append(dict(suffix=suffix,case=case['name'],probe=probe,moment=moment,delta_kl=actual,
                        quadratic_delta=sum(raised['quadratic'])-sum(native['quadratic']),lower=lo,upper=hi,
                        sign_resolved_at_1e_minus_8=bool(lo>1e-8 or hi< -1e-8)))
    result = dict(status='passed',schema='finite-metric-oscillation-budget-v1',cells=cells,summaries=summaries,
        contrasts=contrasts,scientific_training_updates=0,new_native_updates=0,
        inherited_scientific_branches=32,incoming_cases=4,source_suffixes=2,fixed_probes=2,
        scale_cells=len(cells),endpoint_spans=sum(r['count'] for r in cells),
        maximum_bound_violation=max(0.,maximum_violation),maximum_gauge_discrepancy=maximum_gauge,
        checked_sha256=identities,script_sha256=digest(__file__),seconds=time.monotonic()-start,
        scope='Post hoc held-observation analysis of saved scientific logits; neither an autonomous forecast nor independent training evidence. Inequalities are proved in real arithmetic; float64 checks are not interval certificates.')
    with out.open('x') as f:
        json.dump(result,f,indent=2)
        f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['cells','checked_sha256','contrasts']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)
