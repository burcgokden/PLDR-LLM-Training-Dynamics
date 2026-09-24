#!/usr/bin/env python
"""Paired fixed-cohort horizon changes and batch-shape observation sensitivity."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from analyze_scaling_collectives import collective_fields
from analyze_size_time import empirical_weights, whole_seed_statistics
from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


def paired_summary(before, after, n):
    a, ab = whole_seed_statistics(before, n)
    b, bb = whole_seed_statistics(after, n)
    count = len(before)
    means = (after-before).reshape(count, -1).mean(-1)
    mboot = empirical_weights(count)@means/count
    differences = bb-ab
    leave = []
    for i in range(count):
        x, y = np.delete(before, i, axis=0), np.delete(after, i, axis=0)
        leave.append(n*(np.var(y, axis=0, ddof=1).mean()-np.var(x, axis=0, ddof=1).mean()))
    leave = np.asarray(leave)
    return dict(before=a, after=b, susceptibility_change=b['susceptibility']-a['susceptibility'],
        susceptibility_change_empirical_percentiles=np.quantile(differences, [.025, .975]).tolist(),
        susceptibility_change_jackknife_se=float(np.sqrt((count-1)/count*np.sum((leave-leave.mean())**2))),
        mean_change=float(means.mean()), mean_change_empirical_percentiles=np.quantile(mboot, [.025, .975]).tolist(),
        per_initialization_mean_change=means.tolist(),
        paired_field_rms=float(np.sqrt(np.mean((after-before)**2))),
        finite_relative_susceptibility_change=(b['susceptibility']/a['susceptibility']-1) if a['susceptibility']>0 else None), differences


def main():
    p = argparse.ArgumentParser();p.add_argument('--root', required=True)
    p.add_argument('--study', default='critical-scaling-20260906');p.add_argument('--output', required=True)
    p.add_argument('--wait', action='store_true');a = p.parse_args();study = Path(a.root)/a.study
    controller = study/'launcher-target-collectives.json'
    followup = study/'followup-preparation.json'
    while not (json.loads(controller.read_text())['status'] == 'complete' and
               json.loads(followup.read_text())['status'] == 'complete'):
        if not a.wait:raise RuntimeError('Complete saved-state observations are required')
        time.sleep(30)
    lookup, inputs = defaultdict(dict), [controller, followup]
    for path in sorted((study/'measurements').glob('cpu-*/manifest.json')):
        m = json.loads(path.read_text());c = m['case']
        if m['status'] != 'complete':raise AssertionError('A selected observation did not complete')
        key = (c['heads'], c['multiplier'], c['shared_seed'], c['stream_seed'])
        identifier = (c['step'], c['seed'])
        if identifier in lookup[key]:raise AssertionError('Duplicated initialization and horizon')
        lookup[key][identifier] = path
        inputs.extend([path, path.parent/'measurements.npz'])
    training = []
    for path in sorted((study/'runs').glob('*/manifest.json')):
        m = json.loads(path.read_text())
        if m.get('schema') == 'native-size-time-training-v1' and m['status'] == 'complete':
            training.append(path);inputs.extend([path, path.parent/'measurements.npz'])
    out = Path(a.output);out.mkdir(parents=True, exist_ok=False);bind_run(out, inputs, vars(a))
    records, raw = [], {}
    field_names = ['row_native', 'row_reduced64', 'absolute_row_energy', 'total_energy',
                   'common_centroid', 'head_residual_energy', 'head_alignment',
                   'mean_metric_contrast', 'attention', 'operator_rms', 'prediction_entropy', 'nll', 'logit_projection']
    for (n, g, shared, stream), paths in sorted(lookup.items()):
        steps = sorted({t for t, _ in paths})
        for count in [4, 8, 16]:
            seeds = list(range(640101, 640101+count))
            balanced = [t for t in steps if all((t, seed) in paths for seed in seeds)]
            # Every adjacent available balanced horizon is specified by the
            # completed observation design; there is no sign-based selection.
            for early, late in zip(balanced, balanced[1:]):
                for precision in ['float32', 'float64']:
                    selected = [paths[t, s] for t in [early, late] for s in seeds]
                    if not all(precision in [r['precision'] for r in json.loads(path.read_text())['precision_records']] for path in selected):
                        continue
                    fields = []
                    for t in [early, late]:
                        data = defaultdict(list)
                        for seed in seeds:
                            path = paths[t, seed]
                            if sha256(path.parent/'measurements.npz') != json.loads(path.read_text())['raw_sha256']:
                                raise AssertionError('A paired observation changed')
                            with np.load(path.parent/'measurements.npz') as z:
                                for name in ['fields', 'heads', 'centroids', 'energies', 'mean_metric']:
                                    data[name].append(z[precision+'_'+name].astype(float))
                        fields.append(collective_fields({k:np.stack(v) for k,v in data.items()}, count, n))
                    row = dict(heads=n, multiplier=g, shared_seed=shared, stream_seed=stream,
                        seeds=seeds, seed_count=count, early=early, late=late, precision=precision, observables={})
                    for name in field_names:
                        row['observables'][name], bootstrap = paired_summary(fields[0][name], fields[1][name], n)
                        label = f'{precision}_h{n}_g{g}_c{shared}_b{stream}_s{count}_t{early}_{late}_{name}'
                        raw[label+'_difference_bootstrap'] = bootstrap
                    records.append(row)
                    print('Paired horizon', n, g, shared, stream, count, early, late, precision, flush=True)
    # Batch-shape differences are paired measurements, not training variation.
    # Reconstruct the same 16 contexts in dense batches of 16 and in endpoint
    # batches of 32 for every balanced four/eight/sixteen-identity condition.
    groups = defaultdict(dict)
    for path in training:
        m = json.loads(path.read_text());c = m['arguments']
        for t in m['milestones']:
            key = (c['heads'], c['multiplier'], c['shared_seed'], c['stream_seed'], t)
            candidate = (m['completed_step'], str(path))
            if c['seed'] not in groups[key] or candidate > groups[key][c['seed']]:
                groups[key][c['seed']] = candidate
    batch_comparisons = []
    for (n, g, shared, stream, t), paths in sorted(groups.items()):
        for count in [4, 8, 16]:
            seeds = list(range(640101, 640101+count))
            if not all(seed in paths for seed in seeds):continue
            left, right = defaultdict(list), defaultdict(list)
            for seed in seeds:
                path = Path(paths[seed][1])
                with np.load(path.parent/'measurements.npz') as z:
                    i = np.flatnonzero(z['steps'] == t).item()
                    for store, h, f in [(left, z['dense_heads'][i], z['dense_fields'][i]),
                                        (right, z[f'heads_{t}'][:16], z[f'fields_{t}'][:16])]:
                        h, f = h.astype(float), f.astype(float)
                        for name, values in dict(row=h[..., 2].mean(-1), attention=h[..., 0].mean(-1),
                            nll=f[..., 25], prediction_entropy=f[..., 24], logit_projection=f[..., 16:24]).items():
                            store[name].append(values)
            row = dict(heads=n, multiplier=g, shared_seed=shared, stream_seed=stream, step=t, seed_count=count, observables={})
            for name in left:
                x, y = np.stack(left[name]), np.stack(right[name])
                cx, cy, ce = [float(n*np.var(v, axis=0, ddof=1).mean()) for v in [x, y, x-y]]
                bound = 2*np.sqrt(cy*ce)+ce
                if abs(cx-cy) > bound*(1+2e-7)+1e-20:
                    raise AssertionError('Batch-shape variance inequality failed')
                row['observables'][name] = dict(batch16_susceptibility=cx, batch32_susceptibility=cy,
                    difference_susceptibility=ce, susceptibility_error_bound=float(bound),
                    relative_bound=float(bound/cy) if cy>0 else None,
                    maximum_field_difference=float(np.max(np.abs(x-y))), rms_field_difference=float(np.sqrt(np.mean((x-y)**2))))
            batch_comparisons.append(row)
    result = dict(schema='paired-scaling-paths-v1', status='complete', paired_horizons=records,
        batch_shape_comparisons=batch_comparisons,
        statistical_scope='Whole-initialization empirical resampling uses the same multiplicities at both horizons. Adjacent complete balanced horizons are retained regardless of the sign of the change. Empirical intervals have no guaranteed population coverage.',
        observation_scope='CPU paired horizon fields use 512 held-out contexts and separate 64-context calibration matrices. GPU batch-shape comparisons use exactly the same 16 held-out contexts with different evaluation batch sizes; they add no trained identities.')
    write_json(out/'results.json', result);np.savez_compressed(out/'measurements.npz', **raw)
    write_json(out/'manifest.json', dict(status='complete', binding_sha256=sha256(out/'binding.json'),
        results_sha256=sha256(out/'results.json'), raw_sha256=sha256(out/'measurements.npz'),
        paired_groups=len(records), batch_shape_groups=len(batch_comparisons)))


if __name__ == '__main__':main()
