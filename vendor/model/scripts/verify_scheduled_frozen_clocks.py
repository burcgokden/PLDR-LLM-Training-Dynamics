#!/usr/bin/env python
"""Reconstruct the frozen schedule-transfer forecasts independently of the producer."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path

import numpy as np

from model_rg.provenance import sha256, write_json
from verify_scheduled_collectives import statistics, bootstrap_roundoff_allowance


def cumulative_clock(power):
    updates = np.minimum(np.arange(262144, dtype=float), 250000)
    factors = np.empty_like(updates)
    ramp = updates <= 2000
    factors[ramp] = updates[ramp] / 2000
    angle = np.pi * (updates[~ramp] - 2000) / 248000
    factors[~ramp] = .55 + .45 * np.cos(angle)
    return np.concatenate(([0.], np.cumsum(factors ** power)))


def coordinates(folder):
    with np.load(folder / 'measurements.npz') as raw:
        heads = raw['float32_heads'].astype(float)
        fields = raw['float32_fields'].astype(float)
        centroids = raw['float32_centroids']
        energies = raw['float32_energies']
    return dict(row=np.mean(heads[..., 2], axis=-1), common_centroid=np.mean(centroids, axis=-2),
                absolute_row_energy=np.mean(energies[..., 0], axis=-1), attention=np.mean(heads[..., 0], axis=-1),
                operator_rms=np.mean(heads[..., 3], axis=-1), nll=fields[..., 25],
                prediction_entropy=fields[..., 24], logit_projection=fields[..., 16:24])


def close(actual, expected, name, atol=1e-24):
    if expected is None:
        if actual is not None:
            raise AssertionError('An undefined frozen statistic changed: ' + name)
    else:
        np.testing.assert_allclose(actual, expected, rtol=4e-10, atol=atol, err_msg=name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--study', default='scheduled-training-20260908')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    study = root / args.study
    constant = root / 'critical-scaling-20260906'
    folder = study / 'analysis/frozen-clocks'
    output = study / 'verification/frozen-clock-statistics.json'
    if output.exists():
        raise FileExistsError(output)
    checked = {}
    cache = {}

    def check(path, digest=None):
        path = Path(path).resolve()
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in cache:
            cache[key] = sha256(path)
        actual = cache[key]
        if digest is not None and digest != actual:
            raise AssertionError('Changed frozen forecast evidence: ' + str(path))
        checked[str(path)] = actual
        return actual

    def load(path):
        check(path)
        return json.loads(Path(path).read_text())

    meta = load(folder / 'manifest.json')
    if meta['status'] != 'complete':
        raise AssertionError('A complete immutable frozen forecast is required')
    for name, key in [('binding.json', 'binding_sha256'), ('results.json', 'results_sha256'), ('measurements.npz', 'raw_sha256')]:
        check(folder / name, meta[key])
    binding = load(folder / 'binding.json')
    for name, digest in binding['inputs'].items():
        check(name, digest)
    for name, digest in binding['source_files'].items():
        check(folder / 'source' / name, digest)
    result = load(folder / 'results.json')
    selection = load(study / 'protocols/regime-training-selection.json')
    jobs = [j for j in selection['jobs'] if j['recipe'] == 'controlled']
    if len(jobs) != 8 or result['controlled_target_ids'] != [j['run_id'] for j in jobs]:
        raise AssertionError('The exact eight schedule-only target identities are required')
    frozen_at = datetime.fromisoformat(result['frozen_at'])
    if result['status'] != 'frozen_before_controlled_targets':
        raise AssertionError('The prediction was not declared frozen before its targets')
    for job in jobs:
        path = study / 'runs' / job['run_id'] / 'binding.json'
        if path.exists() and frozen_at >= datetime.fromisoformat(load(path)['environment']['utc']):
            raise AssertionError('A controlled target began before the forecast was frozen')
    previous = load(constant / 'analysis/frozen-forecasts/results.json')
    previous_meta = load(constant / 'analysis/frozen-forecasts/manifest.json')
    check(constant / 'analysis/frozen-forecasts/results.json', previous_meta['results_sha256'])
    primary = [f for f in previous['clock_fits'] if f['threshold'] == .5 and f['model'] == 'free']
    if len(primary) != 1:
        raise AssertionError('The inherited primary finite clock fit is not unique')
    powers = dict(linear=1., quadratic=2., fitted=primary[0]['power'])
    close(result['fitted_path_clock_power'], powers['fitted'], 'primary power')
    times = [8192, 16384, 32768, 65536, 98304, 131072]
    if result['source_times'] != times or result['seeds'] != list(range(640101, 640105)):
        raise AssertionError('The complete declared constant reference panel changed')
    conditions = {(r['heads'], r['step'], r['model']): r for r in result['forecasts']}
    passages = {(r['heads'], r['seed'], r['threshold'], r['model']): r for r in result['first_passages']}
    if len(conditions) != 48 or len(passages) != 96:
        raise AssertionError('A declared clock alternative or threshold was omitted')
    field_conditions = 0
    maximum_bootstrap_roundoff_allowance = 0.
    maximum_bootstrap_discrepancy = 0.
    unavailable = 0
    reference_states = 0
    expected_arrays = set()
    with np.load(folder / 'measurements.npz') as stored:
        for name, power in powers.items():
            label = name + '_cumulative_clock'
            # Vectorized trigonometric construction is independent of the scalar producer.
            np.testing.assert_allclose(stored[label], cumulative_clock(power), rtol=4e-12, atol=3e-8)
            expected_arrays.add(label)
        for heads in [4, 14]:
            reference = {}
            for seed in range(640101, 640105):
                for step in times:
                    name = f'cpu-h{heads}-g1-t{step}-s{seed}' if step <= 32768 else f'cpu-extended-h{heads}-g1-s{seed}-t{step}'
                    parent = constant / 'measurements' / name
                    parent_meta = load(parent / 'manifest.json')
                    case = parent_meta['case']
                    if parent_meta['status'] != 'complete' or tuple(case[k] for k in
                            ['heads', 'seed', 'step', 'multiplier', 'shared_seed', 'stream_seed']) != (heads, seed, step, 1., 640011, 640001):
                        raise AssertionError('A constant reference identity changed')
                    check(parent / 'measurements.npz', parent_meta['raw_sha256'])
                    reference[seed, step] = coordinates(parent)
                    reference_states += 1
            target_steps = sorted({int(t) for job in jobs if job['heads'] == heads for t in job['save_steps'].split(',')})
            for step in target_steps:
                for name, power in powers.items():
                    reported = conditions[heads, step, name]
                    close(reported['power'], power, 'path clock power')
                    # The already reconstructed frozen clock preserves its exact floating-point coordinates.
                    clock = float(stored[name + '_cumulative_clock'][step])
                    close(reported['reference_clock'], clock, 'target clock')
                    if clock < times[0] or clock > times[-1]:
                        if reported['status'] != 'outside_reference_horizon' or 'observables' in reported:
                            raise AssertionError('An unavailable reference horizon was silently extrapolated')
                        unavailable += 1
                        continue
                    if reported['status'] != 'frozen_prediction':
                        raise AssertionError('An available frozen prediction was omitted')
                    low = max(t for t in times if t <= clock)
                    high = min(t for t in times if t >= clock)
                    weight = 0. if low == high else (math.log1p(clock) - math.log1p(low)) / (math.log1p(high) - math.log1p(low))
                    close(reported['interpolation'], [low, high, weight], 'field interpolation')
                    if set(reported['observables']) != set(reference[640101, low]):
                        raise AssertionError('A frozen predictive or collective field was omitted')
                    for field in reference[640101, low]:
                        expected = np.stack([(1 - weight) * reference[seed, low][field] + weight * reference[seed, high][field]
                                             for seed in range(640101, 640105)])
                        label = f'h{heads}_t{step}_{name}_{field}'
                        np.testing.assert_array_equal(stored[label], expected)
                        values, bootstrap, _ = statistics(expected, heads)
                        if set(values) != set(reported['observables'][field]):
                            raise AssertionError('A frozen moment was omitted')
                        allowance = max(1e-24, bootstrap_roundoff_allowance(expected, heads))
                        maximum_bootstrap_roundoff_allowance = max(maximum_bootstrap_roundoff_allowance, allowance)
                        maximum_bootstrap_discrepancy = max(maximum_bootstrap_discrepancy, float(np.max(np.abs(stored[label + '_bootstrap'] - bootstrap))))
                        for key, value in values.items():
                            close(reported['observables'][field][key], value, label + '.' + key, atol=allowance if key == 'empirical_percentiles' else 1e-24)
                        close(stored[label + '_bootstrap'], bootstrap, label + '.resamples', atol=allowance)
                        expected_arrays.update([label, label + '_bootstrap'])
                        field_conditions += 1
            print('Reconstructed all frozen scheduled fields for', heads, 'heads', flush=True)
        for fit in previous['clock_fits']:
            name = {'1': 'linear', '2': 'quadratic', 'free': 'fitted'}[fit['model']]
            label = f'first_passage_{fit["threshold"]}_{name}_clock'
            clock = stored[label]
            np.testing.assert_allclose(clock, cumulative_clock(fit['power']), rtol=4e-12, atol=3e-8)
            expected_arrays.add(label)
            for heads in [4, 14]:
                for seed in range(640101, 640105):
                    reported = passages[heads, seed, fit['threshold'], name]
                    offset = fit['parameters'][fit['widths'].index(heads)]
                    index = fit['seeds'].index(seed)
                    if index:
                        offset += fit['parameters'][len(fit['widths']) + index - 1]
                    median = math.exp(offset)
                    crossed = np.flatnonzero(clock >= median)
                    update = int(crossed[0]) if len(crossed) else None
                    close(reported['median_reference_clock'], median, 'passage reference median')
                    close(reported['working_log_scale'], fit['scale'], 'passage working scale')
                    close(reported['power'], fit['power'], 'passage power')
                    if reported['median_scheduled_update'] != update or reported['status'] != (
                            'frozen_prediction' if update is not None else 'right_censored_median_prediction'):
                        raise AssertionError('Inverse-clock passage or censoring changed')
        if set(stored.files) != expected_arrays:
            raise AssertionError('Frozen array coverage differs from the declared prediction family')
    write_json(output, dict(status='passed', constant_reference_states=reference_states, target_trajectories=8,
        path_conditions=48, first_passage_conditions=96, field_conditions=field_conditions,
        outside_reference_horizon=unavailable, frozen_at=result['frozen_at'],
        maximum_bootstrap_roundoff_allowance=maximum_bootstrap_roundoff_allowance,
        maximum_bootstrap_discrepancy=maximum_bootstrap_discrepancy,
        checked_sha256=checked, verifier_sha256=sha256(__file__),
        independent_statistics_sha256=sha256(Path(__file__).with_name('verify_scheduled_collectives.py')),
        scope='Independent vectorized schedule clocks, bytewise whole-field interpolation, direct seed-pair moments and all 256 resamples, and inverse-clock first-passage predictions. Every reference state and inherited working-fit input is bound. No target outcome is used to fit any prediction. The powers describe finite time transport, not thermodynamic critical exponents.'))
    print('Verified all frozen scheduled clock predictions', flush=True)


if __name__ == '__main__':
    main()
