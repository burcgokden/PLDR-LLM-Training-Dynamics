#!/usr/bin/env python
"""Freeze out-of-sample rate and joint size-clock forecasts from inherited data."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr

from analyze_size_time import empirical_weights, whole_seed_statistics
from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


def crossing_interval(times, values, threshold):
    """First observed downcrossing, retaining the actual observation interval."""
    hit = np.flatnonzero(values <= threshold)
    if not len(hit):
        return float(times[-1]), None
    i = int(hit[0])
    return float(times[i-1]) if i else 0., float(times[i])


def log_interval_probability(lower, upper):
    # Reflect positive-tail intervals to avoid subtracting nearly unit CDFs.
    lo, hi = np.asarray(lower), np.asarray(upper)
    reflect = lo > 0
    a = np.where(reflect, -hi, lo)
    b = np.where(reflect, -lo, hi)
    lb, la = log_ndtr(b), log_ndtr(a)
    return lb + np.log(-np.expm1(np.minimum(la-lb, -1e-15)))


def fit_clock(records, fixed=None, weights=None):
    widths = sorted({r['heads'] for r in records})
    seeds = sorted({r['seed'] for r in records})
    x = np.array([[float(r['heads']==n) for n in widths] +
                 [float(r['seed']==s) for s in seeds[1:]] for r in records])
    logg = np.log([r['multiplier'] for r in records])
    lower = np.log([max(r['lower'], 1e-30) for r in records])
    upper = np.array([np.log(r['upper']) if r['upper'] is not None else np.inf for r in records])
    w = np.ones(len(records)) if weights is None else np.array([weights[seeds.index(r['seed'])] for r in records])
    def objective(parameters):
        coefficient = parameters[:x.shape[1]]
        power = parameters[-2] if fixed is None else fixed
        sigma = np.exp(parameters[-1])
        mean = x@coefficient-power*logg
        return -float(w@log_interval_probability((lower-mean)/sigma, (upper-mean)/sigma))
    initial = np.r_[np.full(len(widths), np.log(7000)), np.zeros(len(seeds)-1),
                    [2.] if fixed is None else [], np.log(.3)]
    bounds = [(0., 20.)]*len(widths)+[(-5., 5.)]*(len(seeds)-1)
    bounds += ([(-2., 5.)] if fixed is None else []) + [(np.log(.02), np.log(3.))]
    result = minimize(objective, initial, method='L-BFGS-B', bounds=bounds,
                      options=dict(maxiter=1200, ftol=1e-10))
    return dict(power=float(result.x[-2]) if fixed is None else float(fixed),
                negative_log_working_likelihood=float(result.fun),
                scale=float(np.exp(result.x[-1])), success=bool(result.success),
                message=str(result.message), parameters=result.x.tolist(), widths=widths,
                seeds=seeds, records=len(records), censored=sum(r['upper'] is None for r in records))


def full_observations(path, step):
    with np.load(path.parent/'measurements.npz') as z:
        h, f = z[f'heads_{step}'].astype(float), z[f'fields_{step}'].astype(float)
    return dict(row=h[..., 2].mean(-1), attention=h[..., 0].mean(-1),
                operator_rms=h[..., 3].mean(-1), prediction_entropy=f[..., 24],
                nll=f[..., 25], logit_projection=f[..., 16:24])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--study', default='critical-scaling-20260906')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root, out = Path(a.root), Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    protocols = [root/a.study/'protocols'/n for n in ['clock-holdout.json', 'diffusive-size-map.json']]
    for path in protocols:
        for job in json.loads(path.read_text())['jobs']:
            if (root/a.study/'runs'/job['run_id']).exists():
                raise RuntimeError('Forecast must precede every target execution')
    paths, metadata = [], {}
    for study in ['criticality-study-20260905', 'criticality-dynamics-20260906', 'observation-closure-20260906']:
        for path in sorted((root/study/'runs').glob('*/manifest.json')):
            meta = json.loads(path.read_text())
            c = meta.get('arguments', {})
            if (meta.get('schema') in ['criticality-training-v1', 'criticality-dynamics-training-v1']
                and meta['status']=='complete' and c.get('seed') in range(640101, 640105)
                and c.get('shared_seed')==640011 and c.get('stream_seed')==640001
                and (c.get('normalization')=='variance' or c.get('heads')==2)
                and c.get('multiplier') in [1., 1.5, 2., 3., 4.]):
                paths.append(path)
                metadata[path] = meta
    bind_run(out, paths+[p.parent/'measurements.npz' for p in paths]+protocols, vars(a))
    dense, milestones, duplicates = defaultdict(dict), defaultdict(dict), []
    # Later completed parents take priority. All duplicate point discrepancies remain visible.
    for path in sorted(paths, key=lambda p:(metadata[p]['completed_step'], str(p))):
        meta = metadata[path]
        c = meta['arguments']
        key = (c['heads'], c['multiplier'], c['seed'])
        with np.load(path.parent/'measurements.npz') as z:
            points = {}
            if 'dense_heads' in z:
                points.update({int(t):float(h[..., 2].mean(dtype=float)) for t,h in zip(z['steps'], z['dense_heads'])})
            for t in meta['milestones']:
                points[int(t)] = float(z[f'heads_{t}'][:16, ..., 2].mean(dtype=float))
                milestones[key][int(t)] = path
            for t,v in points.items():
                if t in dense[key]:
                    previous = dense[key][t]
                    duplicates.append(dict(heads=key[0], multiplier=key[1], seed=key[2], step=t,
                                           difference=abs(previous[0]-v), first=previous[1], second=str(path)))
                dense[key][t] = (v, str(path))
    crossings, curves = [], {}
    for (n,g,s), series in sorted(dense.items()):
        if n not in [2,4,8,14]:
            continue
        t = np.array(sorted(series))
        v = np.array([series[x][0] for x in t])
        curves[f'h{n}_g{g}_s{s}_time'] = t
        curves[f'h{n}_g{g}_s{s}_row'] = v
        for threshold in [.8,.5,.2,.1]:
            lo,hi = crossing_interval(t,v,threshold)
            crossings.append(dict(heads=n,multiplier=g,seed=s,threshold=threshold,lower=lo,upper=hi))
    fits = []
    for threshold in [.8,.5,.2,.1]:
        selected = [r for r in crossings if r['threshold']==threshold]
        for fixed in [1.,2.,None]:
            fit = fit_clock(selected, fixed)
            fits.append(dict(threshold=threshold,model='free' if fixed is None else str(int(fixed)),**fit))
            print('clock',threshold,fixed,fit['power'],fit['success'],flush=True)
    primary = next(r for r in fits if r['threshold']==.5 and r['model']=='free')
    if not all(r['success'] for r in fits):
        raise RuntimeError('A declared clock fit did not converge; preserve this diagnostic output')
    primary_records = [r for r in crossings if r['threshold']==.5]
    boot = [fit_clock(primary_records,weights=w) for w in empirical_weights(4)]
    if not all(r['success'] for r in boot):
        write_json(out/'bootstrap-failures.json',boot)
        raise RuntimeError('Bootstrap clock convergence needs resolution before freezing')
    curves['clock_power_bootstrap'] = np.array([r['power'] for r in boot])
    primary['empirical_power_percentiles'] = np.quantile(curves['clock_power_bootstrap'],[.025,.975]).tolist()
    width_fits = [dict(heads=n,**fit_clock([r for r in primary_records if r['heads']==n])) for n in [2,4,8,14]]
    forecasts, raw = [], {}
    def interpolate(n,seed,u):
        series = milestones[n,1.,seed]
        times = sorted(series)
        if not times[0]<=u<=times[-1]:
            return None
        lo = max(t for t in times if t<=u)
        hi = min(t for t in times if t>=u)
        weight = (np.log1p(u)-np.log1p(lo))/(np.log1p(hi)-np.log1p(lo)) if hi!=lo else 0.
        left,right = full_observations(series[lo],lo),full_observations(series[hi],hi)
        return {k:(1-weight)*left[k]+weight*right[k] for k in left}
    for protocol in protocols:
        spec = json.loads(protocol.read_text())
        conditions = sorted({(j['heads'],j['multiplier'],j['steps']) for j in spec['jobs']})
        for n,g,horizon in conditions:
            targets = sorted({t for t in [8192,12288,16384,24576,horizon] if t<=horizon})
            for t in targets:
                for name,power in [('linear',1.),('quadratic',2.),('fitted',primary['power'])]:
                    u = g**power*t
                    values = [interpolate(n,s,u) for s in range(640101,640105)]
                    item = dict(protocol=protocol.name,heads=n,multiplier=g,step=t,model=name,
                                power=power,reference_step=u,observables={})
                    if any(v is None for v in values):
                        item['status'] = 'outside_reference_horizon'
                    else:
                        item['status'] = 'frozen_prediction'
                        for field in values[0]:
                            q = np.stack([v[field] for v in values])
                            stats,_ = whole_seed_statistics(q,n)
                            item['observables'][field] = stats
                            raw[f'{protocol.stem}_h{n}_t{t}_{name}_{field}'] = q
                    forecasts.append(item)
    # A second, joint-clock extrapolation uses only the two already completed reference sizes.
    joint = []
    references = []
    for n,g,t in [(2,2.,2048),(8,1.,8192)]:
        arrays = [full_observations(milestones[n,g,s][t],t) for s in range(640101,640105)]
        references.append({field:np.stack([r[field] for r in arrays]) for field in arrays[0]})
    for n in [4,14,24]:
        row = dict(heads=n,step=1024*n,multiplier=float(np.sqrt(8/n)),observables={})
        for field in references[0]:
            mean = [float(r[field].mean()) for r in references]
            variance = [float(np.var(r[field],axis=0,ddof=1).mean()) for r in references]
            x = (1/n-1/8)/(1/2-1/8)
            row['observables'][field] = dict(mean=mean[1]+x*(mean[0]-mean[1]),
                susceptibility=n*(variance[1]+x*(variance[0]-variance[1])),
                reference_means=mean,reference_variances=variance)
        joint.append(row)
    results = dict(schema='frozen-native-scaling-forecasts-v1',status='frozen_before_target_execution',
        frozen_at=datetime.now(timezone.utc).isoformat(),arguments=vars(a),clock_fits=fits,
        width_clock_fits=width_fits,crossing_intervals=crossings,forecasts=forecasts,
        joint_clock_forecasts=joint,duplicate_dense_discrepancies=duplicates,
        protocol_sha256={p.name:sha256(p) for p in protocols},
        clock_method='Interval-censored log-normal accelerated first-passage working model with width and initialization fixed effects. First threshold crossing on one fixed 16-context cohort. Exact four-identity empirical resampling preserves each full block across widths and rates.',
        clock_scope='Clock powers describe finite training time. They are not critical exponents. Working likelihood independence is not assumed for population uncertainty; four-block empirical intervals have no guaranteed coverage.',
        path_forecast='Interpolate the complete fixed-unit, 512-context, g=1 seed fields linearly in log(1+t), then compute conditional susceptibility. Compare every declared alternative without refitting to target data.',
        joint_forecast='Affine in inverse head count for the mean and conditional seed variance, calibrated only at (N,g,t)=(2,2,2048),(8,1,8192). Both have g^2*t=8192 and 2*t/N=2048. A finite extrapolation test, with no assumed critical surface.',
        evaluation='Paired seed endpoint differences and whole-identity empirical errors; report all horizons, observables and alternatives. No power-law or SOC conclusion follows solely from forecast agreement.')
    write_json(out/'results.json',results)
    np.savez_compressed(out/'measurements.npz',**curves,**raw)
    write_json(out/'manifest.json',dict(status='complete',results_sha256=sha256(out/'results.json'),
        raw_sha256=sha256(out/'measurements.npz'),binding_sha256=sha256(out/'binding.json'),
        forecasts=len(forecasts),joint_forecasts=len(joint)))


if __name__=='__main__':
    main()
