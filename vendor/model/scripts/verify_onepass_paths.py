#!/usr/bin/env python
"""Reconstruct each single-pass drive phase from the retained native arrays."""
import argparse
import json
from pathlib import Path

import numpy as np

from model_rg.provenance import sha256, write_json


def diagnostics(values, spacing):
    x = np.asarray(values, dtype=float).reshape(len(values), -1)
    count, dimension = x.shape
    average = np.mean(x, axis=0)
    centered = x-average
    coordinate = np.arange(count, dtype=float)-(count-1)/2
    denominator = count*(count*count-1)/12
    slope = np.einsum('t,td->d', coordinate, centered)/denominator
    trend = coordinate[:, None]*slope
    residual = centered-trend
    square = lambda array: float(np.einsum('i,i->', array.reshape(-1), array.reshape(-1))/array.size)
    lags = [0]
    lag = 1
    while lag <= count//4:
        lags.append(lag)
        lag *= 2
    record = dict(samples=count, spacing=spacing, coordinates=dimension,
        mean_squared=square(average), variance=square(centered),
        endpoint_displacement_squared=square(x[-1]-x[0]),
        linear_slope_squared_per_update=square(slope)/(spacing*spacing),
        linear_trend_variance=square(trend), detrended_variance=square(residual),
        lags=[lag*spacing for lag in lags])
    for name, array in [('centered', centered), ('detrended', residual)]:
        covariance = [square(array)]
        for lag in lags[1:]:
            covariance.append(float(np.einsum('td,td->', array[:-lag], array[lag:])/((count-lag)*dimension)))
        record[name+'_covariance'] = covariance
        record[name+'_correlation'] = [v/covariance[0] if covariance[0] > 0 else None for v in covariance]
    return record


def compare(actual, expected, label):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError('Changed temporal fields: ' + label)
        for key in expected:
            compare(actual[key], expected[key], label+'.'+key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError('Changed temporal coverage: ' + label)
        for i, (left, right) in enumerate(zip(actual, expected, strict=True)):
            compare(left, right, label+'.'+str(i))
    elif expected is None or isinstance(expected, (str, int, bool)):
        if actual != expected:
            raise AssertionError('Changed temporal identity, count or undefined value: ' + label)
    else:
        np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=1e-23, err_msg=label)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--study', default='scheduled-training-feasible-20260908')
    args = p.parse_args()
    study = Path(args.root).resolve()/args.study
    folder = study/'analysis/onepass-training-paths'
    output = study/'verification/onepass-temporal-statistics.json'
    if output.exists():
        raise FileExistsError(output)
    checked = {}
    cache = {}

    def check(path, expected=None):
        path = Path(path).resolve()
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in cache:
            cache[key] = sha256(path)
        digest = cache[key]
        if expected is not None and digest != expected:
            raise AssertionError('Changed temporal evidence: ' + str(path))
        checked[str(path)] = digest

    def load(path):
        check(path)
        return json.loads(Path(path).read_text())

    meta = load(folder/'manifest.json')
    if meta['status'] != 'complete' or meta['runs'] != 54:
        raise AssertionError('All selected temporal paths are required')
    for name, key in [('binding.json', 'binding_sha256'), ('results.json', 'results_sha256'), ('measurements.npz', 'raw_sha256')]:
        check(folder/name, meta[key])
    binding = load(folder/'binding.json')
    for name, digest in binding['inputs'].items():
        check(name, digest)
    for name, digest in binding['source_files'].items():
        check(folder/'source'/name, digest)
    protocol = study/'protocols/onepass-analysis-selection.json'
    spec = load(protocol)
    selection = load(study/'protocols/onepass-training-selection.json')
    result = load(folder/'results.json')
    reported = {r['run_id']: r for r in result['runs']}
    if len(reported) != 54 or set(reported) != set(spec['run_ids']) or result['selection_sha256'] != sha256(protocol):
        raise AssertionError('A selected temporal trajectory was omitted')
    window_count = field_count = 0
    expected_arrays = set()
    with np.load(folder/'measurements.npz') as stored:
        for job in selection['jobs']:
            name = job['run_id']
            parent = study/'runs'/name
            proof = load(study/'verification/training'/(name+'.json'))
            if proof['status'] != 'complete' or proof['trajectory']['run_id'] != name:
                raise AssertionError('The complete underlying native path requires raw reconstruction')
            check(parent/'measurements.npz', proof['verified_files'][str(parent/'measurements.npz')])
            record = reported[name]
            for key in ['recipe', 'heads', 'seed', 'shared_seed', 'stream_seed']:
                if record[key] != job[key]:
                    raise AssertionError('A conditioned temporal identity changed')
            stop = job['steps']; horizon = job['schedule_horizon']; warmup = job['profile']['warmup_steps']
            if horizon == 0:
                phases = [('constant', 0, stop)]
            else:
                phases = [('warmup', 0, min(stop, warmup)), ('annealing', min(stop, warmup), min(stop, horizon)),
                          ('floor', min(stop, horizon), stop)]
                phases = [(p, a, b) for p, a, b in phases if b > a]
            selected_windows = []
            for phase, begin, end in phases:
                selected_windows.append(dict(phase=phase, kind='complete_observed_phase', begin=begin, end=end, width=end-begin))
                for width in spec['phase_local_block_widths']:
                    count = (end-begin)//width
                    for i in range(count):
                        selected_windows.append(dict(phase=phase, kind='phase_local_block', begin=begin+i*width,
                                                     end=begin+(i+1)*width, width=width))
                if phase == 'floor':
                    half = (end-begin)//2
                    if 2*half != end-begin: raise AssertionError('The two equal floor halves no longer partition the floor')
                    for i in range(2):
                        selected_windows.append(dict(phase='floor', kind='floor_half', begin=begin+i*half,
                                                     end=begin+(i+1)*half, width=half))
            for field, value in [('schedule_horizon', horizon), ('role', job['role']), ('completed_updates', stop)]:
                if record[field] != value: raise AssertionError('A native drive or stop identity changed')
            if len(record['windows']) != len(selected_windows):
                raise AssertionError('A selected phase or block was omitted')
            with np.load(parent/'measurements.npz') as raw:
                steps = raw['steps']
                heads, fields = raw['dense_heads'].astype(float), raw['dense_fields'].astype(float)
                emission = dict(common_centroid=raw['dense_centroids'], absolute_row_energy=raw['dense_energies'][..., 0],
                    total_metric_energy=raw['dense_energies'][..., 1], row=np.mean(heads[..., 2], axis=-1),
                    attention=np.mean(heads[..., 0], axis=-1), operator_rms=np.mean(heads[..., 3], axis=-1),
                    prediction_entropy=fields[..., 24], nll=fields[..., 25], logit_projection=fields[..., 16:24])
                updates, gradients = raw['shared_update_projection'], raw['shared_clipped_gradient_projection']
                moments, rates, losses = raw['shared_update_moments'], raw['applied_learning_rates'], raw['losses']
                for definition, observed in zip(selected_windows, record['windows'], strict=True):
                    begin, end = definition['begin'], definition['end']
                    grid = np.arange(((begin+63)//64)*64, end, 64)
                    indices = np.searchsorted(steps, grid)
                    np.testing.assert_array_equal(steps[indices], grid)
                    expected = dict(**definition, probe_first=int(grid[0]), probe_last=int(grid[-1]),
                        probe_samples=len(grid), observations={})
                    for field, array in [('shared_update_projection', updates), ('shared_clipped_gradient_projection', gradients)]:
                        expected['observations'][field] = diagnostics(array[begin:end], 1)
                    for field, array in emission.items():
                        expected['observations'][field] = diagnostics(array[indices], 64)
                    average = np.mean(moments[begin:end], axis=0)
                    expected['native_moments'] = dict(mean_squared_update=float(average[0]), mean_squared_weight=float(average[1]),
                        mean_update_times_weight=float(average[2]), radial_rate=float(average[2]/average[1]) if average[1] else None,
                        mean_preupdate_loss=float(np.mean(losses[begin:end])), first_applied_learning_rates=rates[begin].tolist(),
                        last_applied_learning_rates=rates[end-1].tolist(), mean_applied_learning_rates=np.mean(rates[begin:end], axis=0).tolist())
                    compare(observed, expected, name+'.'+str(begin)+'.'+str(end)+'.'+definition['kind'])
                    window_count += 1
                    field_count += len(expected['observations'])
                row = np.mean(emission['row'], axis=(1, 2))
                for threshold in spec['thresholds']:
                    selected = np.flatnonzero(row > threshold)
                    split = np.flatnonzero(np.diff(selected) > 1)+1
                    pieces = np.split(selected, split) if len(selected) else []
                    excursions = []
                    for piece in pieces:
                        first, last = int(piece[0]), int(piece[-1])
                        excursions.append(dict(first_above=int(steps[first]), last_above=int(steps[last]),
                            onset_lower=int(steps[first-1]) if first else None, onset_upper=int(steps[first]),
                            offset_lower=int(steps[last]), offset_upper=int(steps[last+1]) if last+1 < len(steps) else None,
                            mesh_duration=int(steps[last]-steps[first]), peak=float(np.max(row[piece])),
                            left_censored=first == 0, right_censored=last+1 == len(steps)))
                    compare(record['excursions'][str(threshold)], excursions, name+'.excursions.'+str(threshold))
                expected = dict(row_minimum=float(np.min(row)), row_maximum=float(np.max(row)), row_last=float(row[-1]),
                    mean_preupdate_loss=float(np.mean(losses)), mean_squared_native_update=float(np.mean(moments[:, 0])),
                    projected_weight_displacement_squared=float(np.mean(np.sum(updates, axis=0)**2)))
                compare(record['whole_path'], expected, name+'.whole_path')
                arrays = dict(steps=steps, row=row, common_centroid_mean=emission['common_centroid'].mean(1),
                              prediction_entropy_mean=emission['prediction_entropy'].mean(1), nll_mean=emission['nll'].mean(1))
                for field, array in arrays.items():
                    label = name+'_'+field
                    np.testing.assert_array_equal(stored[label], array)
                    expected_arrays.add(label)
            print('Reconstructed every scheduled temporal window', name, flush=True)
        if set(stored.files) != expected_arrays:
            raise AssertionError('Retained temporal array coverage changed')
    write_json(output, dict(status='passed', trajectories=54, temporal_windows=window_count, field_windows=field_count,
        threshold_paths=54*len(spec['thresholds']), checked_sha256=checked, verifier_sha256=sha256(__file__),
        scope='Independent native-array reconstruction of every phase-local and complete-phase covariance, every fixed-floor half, all temporal moments, exact retained path arrays and censored threshold intervals. No temporal producer is imported. The check validates finite diagnostics and does not certify stationarity, independent time samples, self-organized criticality or asymptotic exponents.'))
    print('Verified all scheduled temporal diagnostics', flush=True)


if __name__ == '__main__':
    main()
