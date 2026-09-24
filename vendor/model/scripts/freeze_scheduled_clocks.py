#!/usr/bin/env python
"""Freeze constant-to-scheduled clock predictions before controlled target runs."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np

from analyze_size_time import whole_seed_statistics
from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


def schedule_clock(horizon, power, warmup=2000):
    values = []
    for update in range(horizon):
        phase = min(update, 250000)
        factor = phase/warmup if phase <= warmup else .1+.45*(1+math.cos(math.pi*(phase-warmup)/(250000-warmup)))
        values.append(factor**power)
    return np.r_[0., np.cumsum(values, dtype=np.float64)]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True)
    p.add_argument('--study', default='scheduled-training-20260908'); p.add_argument('--wait', action='store_true')
    a = p.parse_args(); root = Path(a.root).resolve(); study = root/a.study
    constant = root/'critical-scaling-20260906'
    selection = study/'protocols/regime-training-selection.json'; selected = json.loads(selection.read_text())
    jobs = [j for j in selected['jobs'] if j['recipe'] == 'controlled']
    if len(jobs) != 8: raise AssertionError('Exactly eight schedule-only target identities are required')
    times = [8192,16384,32768,65536,98304,131072]; sources = {}
    for n in [4,14]:
        for seed in range(640101,640105):
            for step in times:
                name = f'cpu-h{n}-g1-t{step}-s{seed}' if step <= 32768 else f'cpu-extended-h{n}-g1-s{seed}-t{step}'
                sources[n,seed,step] = constant/'measurements'/name
    waiting = False
    while True:
        if any((study/'runs'/j['run_id']).exists() for j in jobs):
            raise AssertionError('Clock predictions must precede all eight controlled scheduled target executions')
        missing = [str(path/'manifest.json') for path in sources.values() if not (path/'manifest.json').exists()]
        if not missing: break
        if not a.wait: raise FileNotFoundError('The constant-rate reference panel is incomplete: '+str(missing))
        if not waiting: print('Waiting for',len(missing),'constant-rate CPU reference states before freezing',flush=True)
        waiting = True; time.sleep(30)
    out = study/'analysis/frozen-clocks'; out.mkdir(parents=True,exist_ok=False)
    fitted = constant/'analysis/frozen-forecasts'; previous = json.loads((fitted/'results.json').read_text())
    inputs = [selection, fitted/'manifest.json', fitted/'results.json']
    for (n,seed,step), path in sources.items():
        meta = json.loads((path/'manifest.json').read_text()); condition = meta['case']
        if meta['status'] != 'complete' or sha256(path/'measurements.npz') != meta['raw_sha256']:
            raise AssertionError('An immutable constant-rate reference is incomplete or changed')
        if tuple(condition[k] for k in ['heads','seed','step','multiplier','shared_seed','stream_seed']) != (n,seed,step,1.,640011,640001):
            raise AssertionError('The constant-rate reference condition changed')
        inputs += [path/'manifest.json',path/'measurements.npz']
    binding = bind_run(out, inputs, vars(a)); frozen_at = datetime.now(timezone.utc).isoformat()
    if any((study/'runs'/j['run_id']).exists() for j in jobs): raise AssertionError('A controlled target began during forecast preparation')
    power = next(r['power'] for r in previous['clock_fits'] if r['threshold']==.5 and r['model']=='free')
    models = dict(linear=1.,quadratic=2.,fitted=power); clocks = {name:schedule_clock(262144,p) for name,p in models.items()}
    raw = {name+'_cumulative_clock':values for name,values in clocks.items()}; forecasts = []
    for n in [4,14]:
        observations = {}
        for seed in range(640101,640105):
            for step in times:
                with np.load(sources[n,seed,step]/'measurements.npz') as z:
                    h = z['float32_heads'].astype(float); f = z['float32_fields'].astype(float)
                    c = z['float32_centroids']; e = z['float32_energies']
                    observations[seed,step] = dict(row=h[...,2].mean(-1),common_centroid=c.mean(-2),
                        absolute_row_energy=e[...,0].mean(-1),attention=h[...,0].mean(-1),
                        operator_rms=h[...,3].mean(-1),nll=f[...,25],prediction_entropy=f[...,24],logit_projection=f[...,16:24])
        target_steps = sorted({int(t) for j in jobs if j['heads']==n for t in j['save_steps'].split(',')})
        for step in target_steps:
            for name, exponent in models.items():
                u = float(clocks[name][step]); record = dict(heads=n,step=step,model=name,power=exponent,reference_clock=u)
                if not times[0] <= u <= times[-1]:
                    record['status'] = 'outside_reference_horizon'; forecasts.append(record); continue
                lower = max(t for t in times if t <= u); upper = min(t for t in times if t >= u)
                weight = (math.log1p(u)-math.log1p(lower))/(math.log1p(upper)-math.log1p(lower)) if upper!=lower else 0.
                record.update(status='frozen_prediction',interpolation=[lower,upper,weight],observables={})
                for field in observations[640101,lower]:
                    q = np.stack([(1-weight)*observations[seed,lower][field]+weight*observations[seed,upper][field] for seed in range(640101,640105)])
                    stats, bootstrap = whole_seed_statistics(q,n); record['observables'][field] = stats
                    label = f'h{n}_t{step}_{name}_{field}'; raw[label] = q; raw[label+'_bootstrap'] = bootstrap
                forecasts.append(record)
    # The inherited interval-censored working fits have width and fixed seed effects.
    # Their scheduled medians are inverse cumulative-clock predictions, not refits.
    first_passages = []
    for fit in previous['clock_fits']:
        label = {'1':'linear','2':'quadratic','free':'fitted'}[fit['model']]
        clock = schedule_clock(262144,fit['power'])
        raw[f'first_passage_{fit["threshold"]}_{label}_clock'] = clock
        for n in [4,14]:
            for seed in range(640101,640105):
                log_median = fit['parameters'][fit['widths'].index(n)]
                seed_index = fit['seeds'].index(seed)
                if seed_index: log_median += fit['parameters'][len(fit['widths'])+seed_index-1]
                median_clock = math.exp(log_median); index = int(np.searchsorted(clock,median_clock))
                first_passages.append(dict(heads=n,seed=seed,threshold=fit['threshold'],model=label,power=fit['power'],
                    median_reference_clock=median_clock,working_log_scale=fit['scale'],
                    median_scheduled_update=index if index < len(clock) else None,
                    status='frozen_prediction' if index < len(clock) else 'right_censored_median_prediction'))
    np.savez_compressed(out/'measurements.npz',**raw)
    write_json(out/'results.json',dict(status='frozen_before_controlled_targets',frozen_at=frozen_at,
        schema='scheduled-clock-forecasts-v1',controlled_target_ids=[j['run_id'] for j in jobs],
        seeds=list(range(640101,640105)),fitted_path_clock_power=power,forecasts=forecasts,first_passages=first_passages,
        source_times=times,reference_precision='CPU float32 forwards; fixed float64 matrix coordinates',
        path_rule='Use u_p(t)=sum_{k=0}^{t-1}s(k)^p with p=1,2 and the previously fitted primary constant-rate clock power. Interpolate whole paired seed fields linearly in log(1+u), using only the complete declared reference horizon. Retain every unavailable extrapolation explicitly.',
        passage_rule='Transform each inherited interval-censored first-passage working model through its cumulative scheduled clock. Every threshold and all three original powers are retained. Width and initialization coefficients are frozen; no scheduled outcome is used for fitting.',
        scope='Predictions concern only the eight schedule-only controlled trajectories. The four reference-labelled recipes also change objective and optimizer settings, and are not asserted to follow this scalar transport. Cumulative scalar clocks need not close the joint optimizer/body/generator law. These finite time powers are not critical exponents.',
        uncertainty='Whole-initialization empirical ranges condition on the fixed reference histories and contexts; no population coverage or thermodynamic criticality claim.'))
    write_json(out/'manifest.json',dict(status='complete',binding_sha256=sha256(out/'binding.json'),
        results_sha256=sha256(out/'results.json'),raw_sha256=sha256(out/'measurements.npz'),
        target_trajectories=8,forecasts=len(forecasts),first_passage_predictions=len(first_passages)))
    print('Frozen scheduled clock predictions:',len(forecasts),'path conditions and',len(first_passages),'first-passage conditions',flush=True)


if __name__=='__main__':main()
