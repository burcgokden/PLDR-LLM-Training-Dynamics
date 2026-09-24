#!/usr/bin/env python
"""Score every frozen finite-width variance alternative on new initializations."""
import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np

from analyze_scaling_collectives import collective_fields
from analyze_size_time import whole_seed_statistics
from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


def main():
    p = argparse.ArgumentParser();p.add_argument('--root', required=True)
    p.add_argument('--study', default='critical-scaling-20260906');p.add_argument('--output', required=True)
    p.add_argument('--wait', action='store_true');a = p.parse_args();study = Path(a.root)/a.study
    forecast = study/'analysis/frozen-variance'
    meta = json.loads((forecast/'manifest.json').read_text())
    if meta['status'] != 'complete' or sha256(forecast/'results.json') != meta['results_sha256']:
        raise AssertionError('Frozen variance predictions changed')
    frozen = json.loads((forecast/'results.json').read_text())
    jobs = {j['run_id']:j for j in frozen['targets']}
    paths = {(j['heads'], j['seed']):study/'measurements'/f"cpu-h{j['heads']}-g1-t16384-s{j['seed']}"/'manifest.json'
             for j in jobs.values()}
    while not all(p.exists() for p in paths.values()):
        if not a.wait:raise RuntimeError('Fresh-initialization observations incomplete')
        time.sleep(30)
    out = Path(a.output);out.mkdir(parents=True, exist_ok=False)
    inputs = [forecast/'manifest.json', forecast/'results.json', forecast/'measurements.npz']
    for path in paths.values():inputs.extend([path, path.parent/'measurements.npz'])
    for name in jobs:inputs.append(study/'runs'/name/'binding.json')
    bind_run(out, inputs, vars(a))
    for name in jobs:
        binding = json.loads((study/'runs'/name/'binding.json').read_text())
        if datetime.fromisoformat(frozen['frozen_at']) >= datetime.fromisoformat(binding['environment']['utc']):
            raise AssertionError('A target started before predictions were frozen')
    observed, bootstraps, records, raw = {}, {}, [], {}
    for n in frozen['target_widths']:
        data = defaultdict(list)
        for seed in frozen['target_initializations']:
            path = paths[n, seed];m = json.loads(path.read_text());case = m['case']
            if m['status'] != 'complete' or case['step'] != 16384:
                raise AssertionError('Incomplete target measurement')
            for key, expected in dict(heads=n, seed=seed, multiplier=1, shared_seed=640011, stream_seed=640001).items():
                if case[key] != expected:raise AssertionError('Confirmation condition changed')
            if sha256(path.parent/'measurements.npz') != m['raw_sha256']:
                raise AssertionError('Confirmation raw array changed')
            with np.load(path.parent/'measurements.npz') as z:
                for name in ['heads', 'fields', 'centroids', 'energies', 'mean_metric']:
                    data[name].append(z['float32_'+name].astype(float))
        fields = collective_fields({k:np.stack(v) for k,v in data.items()}, 4, n)
        observed[n], bootstraps[n] = {}, {}
        for f in frozen['forecasts']:
            name = f['observable']
            observed[n][name], bootstraps[n][name] = whole_seed_statistics(fields[name], n)
            raw[f'N{n}_{name}_target_bootstrap'] = bootstraps[n][name]
    with np.load(forecast/'measurements.npz') as bootstrap:
        for f in frozen['forecasts']:
            name = f['observable']
            for model, predictions in f['models'].items():
                for i, n in enumerate(frozen['target_widths']):
                    prediction = predictions['prediction'][i]
                    actual = observed[n][name]['susceptibility']
                    # Distinct seed panels: take the product of their finite
                    # empirical resampling laws, without pairing different seeds.
                    predicted_bootstrap = bootstrap[name+'_'+model+'_bootstrap'][:, i]
                    error = bootstraps[n][name][:, None]-predicted_bootstrap[None, :]
                    records.append(dict(observable=name, model=model, heads=n, step=16384,
                        prediction=prediction, actual=actual, signed_error=actual-prediction,
                        relative_error=(actual-prediction)/prediction if prediction > 0 else None,
                        log_ratio=float(np.log(actual/prediction)) if actual > 0 and prediction > 0 else None,
                        product_resampling_error_percentiles=np.quantile(error, [.025, .975]).tolist(),
                        calibration_resamples=len(predicted_bootstrap), target_resamples=len(bootstraps[n][name]),
                        target_statistics=observed[n][name]))
    aggregate = []
    for f in frozen['forecasts']:
        for model in f['models']:
            cells = [r for r in records if r['observable'] == f['observable'] and r['model'] == model]
            errors = [r['relative_error'] for r in cells]
            aggregate.append(dict(observable=f['observable'], model=model,
                relative_rmse=float(np.sqrt(np.mean(np.square(errors)))),
                maximum_absolute_relative_error=float(np.max(np.abs(errors))),
                widths=[r['heads'] for r in cells]))
    result = dict(schema='frozen-finite-variance-scores-v1', status='complete', forecasts=records,
        aggregate=aggregate, prediction_sha256=sha256(forecast/'results.json'),
        independent_unit='The four new initialization identities 640105..640108 at each width. Their labels are paired across widths, but distinct from the four calibration identities.',
        uncertainty='The product of exact finite calibration and confirmation resampling distributions is reported as a diagnostic, not a coverage guarantee. The fitted-model calibration distribution excludes its explicitly counted zero-variance resamples; the same-width baseline retains all 256.',
        scope='All 144 scalar predictions are scored without refitting. Count-power agreement does not establish a critical surface, a singular limit, or self-organized attraction.')
    write_json(out/'results.json', result);np.savez_compressed(out/'measurements.npz', **raw)
    write_json(out/'manifest.json', dict(status='complete', binding_sha256=sha256(out/'binding.json'),
        results_sha256=sha256(out/'results.json'), raw_sha256=sha256(out/'measurements.npz'), predictions=len(records)))
    print('Scored all', len(records), 'frozen variance predictions', flush=True)


if __name__ == '__main__':main()
