#!/usr/bin/env python
"""Quantify the numerical observation convention in the already frozen clock fit."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from freeze_scaling_forecasts import crossing_interval, fit_clock
from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


def main():
    p = argparse.ArgumentParser();p.add_argument('--root', required=True)
    p.add_argument('--study', default='critical-scaling-20260906');p.add_argument('--output', required=True)
    a = p.parse_args();study = Path(a.root)/a.study;frozen = study/'analysis/frozen-forecasts'
    forecast = json.loads((frozen/'results.json').read_text())
    binding = json.loads((frozen/'binding.json').read_text())
    metadata = {}
    for name, signature in binding['inputs'].items():
        path = Path(name)
        if path.name == 'manifest.json' and path.parent.parent.name == 'runs':
            if sha256(path) != signature:raise AssertionError('Frozen calibration parent changed')
            metadata[path] = json.loads(path.read_text())
    out = Path(a.output);out.mkdir(parents=True, exist_ok=False)
    inputs = [frozen/'results.json', frozen/'binding.json', *metadata]
    inputs += [p.parent/'measurements.npz' for p in metadata]
    bind_run(out, inputs, vars(a))
    curves = {name:defaultdict(dict) for name in ['milestone_preferred', 'dense_preferred']}
    differences = [];raw = {}
    for path in sorted(metadata, key=lambda p:(metadata[p]['completed_step'], str(p))):
        meta = metadata[path];c = meta['arguments'];key = (c['heads'], c['multiplier'], c['seed'])
        with np.load(path.parent/'measurements.npz') as z:
            dense = ({int(t):float(h[...,2].mean(dtype=float)) for t,h in zip(z['steps'], z['dense_heads'], strict=True)}
                     if 'dense_heads' in z else {})
            milestones = {int(t):float(z[f'heads_{t}'][:16,...,2].mean(dtype=float)) for t in meta['milestones']}
            for t in set(dense)&set(milestones):
                differences.append(dict(heads=key[0], multiplier=key[1], seed=key[2], step=t,
                    source=str(path), row_mean_difference=dense[t]-milestones[t]))
            curves['milestone_preferred'][key].update({**dense, **milestones})
            curves['dense_preferred'][key].update({**milestones, **dense})
    records, fits = {}, []
    for name, lookup in curves.items():
        records[name] = []
        for (n,g,seed), series in sorted(lookup.items()):
            if n not in [2,4,8,14]:continue
            times = np.array(sorted(series));values = np.array([series[t] for t in times])
            raw[f'{name}_h{n}_g{g}_s{seed}_time'] = times
            raw[f'{name}_h{n}_g{g}_s{seed}_row'] = values
            for threshold in [.8,.5,.2,.1]:
                lower, upper = crossing_interval(times, values, threshold)
                records[name].append(dict(heads=n, multiplier=g, seed=seed, threshold=threshold, lower=lower, upper=upper))
        for threshold in [.8,.5,.2,.1]:
            selected = [r for r in records[name] if r['threshold'] == threshold]
            result = fit_clock(selected)
            if not result['success']:raise AssertionError('Observation-sensitivity fit did not converge')
            fits.append(dict(convention=name, threshold=threshold, **result))
    if records['milestone_preferred'] != forecast['crossing_intervals']:
        raise AssertionError('The original frozen clock convention was not reconstructed')
    changed = [dict(original=x, alternative=y) for x,y in zip(records['milestone_preferred'], records['dense_preferred'], strict=True)
               if (x['lower'],x['upper']) != (y['lower'],y['upper'])]
    result = dict(schema='clock-observation-sensitivity-v1', status='complete', fits=fits,
        simultaneous_batch_comparisons=differences, changed_crossing_intervals=changed,
        maximum_simultaneous_row_mean_difference=max(abs(r['row_mean_difference']) for r in differences),
        frozen_clock_sha256=sha256(frozen/'results.json'),
        scope='Both conventions use the same 16 contexts and highest-horizon source priority. The frozen convention prefers the first16 rows of each batch32 milestone; the alternative prefers an available batch16 dense measurement. Earlier states lacking a dense measurement retain their batch32 observation. This is a sensitivity diagnostic of available numerical programs, not a claim that every historical state has a uniform batch16 observation. The frozen target predictions are unchanged.')
    write_json(out/'results.json', result);np.savez_compressed(out/'measurements.npz', **raw)
    write_json(out/'manifest.json', dict(status='complete', binding_sha256=sha256(out/'binding.json'),
        results_sha256=sha256(out/'results.json'), raw_sha256=sha256(out/'measurements.npz')))
    print('Changed crossing intervals:', len(changed), '; maximum row-cohort difference:', result['maximum_simultaneous_row_mean_difference'], flush=True)


if __name__ == '__main__':main()
