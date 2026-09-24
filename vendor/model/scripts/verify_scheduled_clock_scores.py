#!/usr/bin/env python
"""Independently reconstruct schedule-clock forecast errors and censored scores."""
import argparse
from datetime import datetime
import json
from numerical_claims import finite_greater
from numerical_validation import load_json_strict
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad

from model_rg.provenance import sha256, write_json
from verify_scheduled_collectives import DRAW, statistics, bootstrap_roundoff_allowance
from verify_scheduled_frozen_clocks import coordinates


def normal_log_interval(lower, upper):
    """Integrate a scaled normal density instead of subtracting two normal CDFs."""
    if upper <= lower:
        raise AssertionError('A nonpositive passage interval was scored')
    if upper < 0:
        lower, upper = -upper, -lower
    if lower > 0:
        width = upper - lower
        mass, error = quad(lambda x: math.exp(-lower*x - .5*x*x), 0., width,
                           epsabs=1e-13, epsrel=1e-12, limit=200)
        log_scale = -.5*lower*lower - .5*math.log(2*math.pi)
    else:
        mass, error = quad(lambda x: math.exp(-.5*x*x), lower, upper,
                           epsabs=1e-13, epsrel=1e-12, limit=200)
        log_scale = -.5*math.log(2*math.pi)
    if mass <= 0 or finite_greater(error, max(1e-12, 1e-9*mass), 'scripts/verify_scheduled_clock_scores.py:32'):
        raise AssertionError('The independent normal interval quadrature is unresolved')
    return math.log(mass) + log_scale


def close(actual, expected, label, atol=1e-22):
    if expected is None:
        if actual is not None:
            raise AssertionError('An undefined score statistic changed: ' + label)
    else:
        np.testing.assert_allclose(actual, expected, rtol=4e-9, atol=atol, err_msg=label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--study', default='scheduled-training-20260908')
    args = parser.parse_args()
    study = Path(args.root).resolve()/args.study
    frozen = study/'analysis/frozen-clocks'
    folder = study/'analysis/clock-scores'
    output = study/'verification/clock-score-statistics.json'
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
        if expected is not None and expected != digest:
            raise AssertionError('Changed scheduled clock score evidence: ' + str(path))
        checked[str(path)] = digest
        return digest

    def load(path):
        check(path)
        return load_json_strict(Path(path).read_text())

    verification = load(study/'verification/frozen-clock-statistics.json')
    if verification['status'] != 'passed':
        raise AssertionError('Independent reconstruction of the frozen forecasts is required')
    for name in ['results.json', 'measurements.npz', 'manifest.json']:
        check(frozen/name, verification['checked_sha256'][str(frozen/name)])
    forecast = load(frozen/'results.json')
    meta = load(folder/'manifest.json')
    if meta['status'] != 'complete':
        raise AssertionError('All eight controlled targets must have complete scores')
    for name, key in [('binding.json', 'binding_sha256'), ('results.json', 'results_sha256'), ('measurements.npz', 'raw_sha256')]:
        check(folder/name, meta[key])
    binding = load(folder/'binding.json')
    for name, digest in binding['inputs'].items():
        check(name, digest)
    for name, digest in binding['source_files'].items():
        check(folder/'source'/name, digest)
    selection = load(study/'protocols/regime-training-selection.json')
    jobs = [j for j in selection['jobs'] if j['recipe'] == 'controlled']
    observations = load(study/'protocols/regime-observation-selection.json')
    cases = {(c['heads'], c['seed'], c['step']): c for c in observations['cases'] if c['recipe'] == 'controlled'}
    if len(jobs) != 8 or len(cases) != 64:
        raise AssertionError('The controlled trajectory or checkpoint selection changed')
    paths = {}
    for job in jobs:
        parent = study/'runs'/job['run_id']
        proof = load(study/'verification/training'/(job['run_id']+'.json'))
        if proof['status'] != 'complete' or proof['trajectory']['run_id'] != job['run_id']:
            raise AssertionError('A selected native target lacks independent raw reconstruction')
        for name in ['manifest.json', 'measurements.npz']:
            check(parent/name, proof['verified_files'][str(parent/name)])
        native_binding = load(parent/'binding.json')
        if datetime.fromisoformat(native_binding['environment']['utc']) <= datetime.fromisoformat(forecast['frozen_at']):
            raise AssertionError('A target preceded its frozen prediction')
        with np.load(parent/'measurements.npz') as raw:
            paths[job['heads'], job['seed']] = (raw['steps'], raw['dense_heads'][..., 2].astype(float).mean(axis=(1, 2, 3)))
    for case in cases.values():
        parent = study/'measurements'/case['name']
        proof = load(study/'verification/observations'/(case['name']+'.json'))
        if proof['status'] != 'passed' or proof['case'] != case:
            raise AssertionError('A selected CPU target lacks independent observation reconstruction')
        check(parent/'measurements.npz', proof['checked_sha256'][str(parent/'measurements.npz')])
    result = load(folder/'results.json')
    if result['status'] != 'complete' or result['trajectories'] != 8 or len(result['forecasts']) != 48 or len(result['first_passages']) != 96:
        raise AssertionError('The complete frozen prediction family was not scored')
    scored = {(r['heads'], r['step'], r['model']): r for r in result['forecasts']}
    scored_passages = {(r['heads'], r['seed'], r['threshold'], r['model']): r for r in result['first_passages']}
    field_conditions = 0
    expected_arrays = set()
    with np.load(frozen/'measurements.npz') as prediction, np.load(folder/'measurements.npz') as stored:
        for entry in forecast['forecasts']:
            heads, step, model = entry['heads'], entry['step'], entry['model']
            reported = scored[heads, step, model]
            for key in set(entry)-{'observables'}:
                if reported[key] != entry[key]:
                    raise AssertionError('A frozen target identity or availability status changed')
            if entry['status'] != 'frozen_prediction':
                if 'observables' in reported:
                    raise AssertionError('An unavailable reference horizon was given an error score')
                continue
            observations = [coordinates(study/'measurements'/cases[heads, seed, step]['name']) for seed in range(640101, 640105)]
            if set(reported['observables']) != set(observations[0]):
                raise AssertionError('A scheduled score field was omitted')
            for field in observations[0]:
                label = f'h{heads}_t{step}_{model}_{field}'
                actual = np.stack([r[field] for r in observations])
                predicted = prediction[label]
                errors = actual-predicted
                actual_stats, actual_boot, _ = statistics(actual, heads)
                predicted_stats, predicted_boot, _ = statistics(predicted, heads)
                error_stats, _, _ = statistics(errors, heads)
                means = errors.reshape(4, -1).mean(1)
                boot_means = means[DRAW].mean(1)
                boot_difference = actual_boot-predicted_boot
                allowance = max(1e-22, bootstrap_roundoff_allowance(actual, heads)+bootstrap_roundoff_allowance(predicted, heads))
                observed = reported['observables'][field]
                for prefix, stats in [('actual', actual_stats), ('predicted', predicted_stats)]:
                    if set(observed[prefix]) != set(stats):
                        raise AssertionError('A scheduled score moment was omitted')
                    for key, value in stats.items():
                        close(observed[prefix][key], value, label+'.'+prefix+'.'+key,
                              atol=allowance if key == 'empirical_percentiles' else 1e-22)
                expected = dict(mean_error=float(errors.mean()), paired_field_rmse=float(np.sqrt(np.mean(errors*errors))),
                    mean_error_empirical_percentiles=np.quantile(boot_means, [.025, .975]).tolist(),
                    susceptibility_error=actual_stats['susceptibility']-predicted_stats['susceptibility'],
                    susceptibility_error_empirical_percentiles=np.quantile(boot_difference, [.025, .975]).tolist(),
                    error_susceptibility=error_stats['susceptibility'])
                for key, value in expected.items():
                    close(observed[key], value, label+'.'+key, atol=allowance if 'susceptibility_error' in key else 1e-22)
                close(stored[label+'_paired_mean_error_bootstrap'], boot_means, label+'.mean_resamples')
                close(stored[label+'_paired_susceptibility_error_bootstrap'], boot_difference, label+'.variance_resamples', atol=allowance)
                expected_arrays.update([label+'_paired_mean_error_bootstrap', label+'_paired_susceptibility_error_bootstrap'])
                field_conditions += 1
        for entry in forecast['first_passages']:
            key = (entry['heads'], entry['seed'], entry['threshold'], entry['model'])
            reported = scored_passages[key]
            if any(reported[k] != v for k, v in entry.items()):
                raise AssertionError('A frozen passage fit was changed while scoring')
            times, values = paths[entry['heads'], entry['seed']]
            crossing = next((i for i, value in enumerate(values) if value <= entry['threshold']), None)
            if crossing is None:
                low, high = float(times[-1]), None
            else:
                low, high = (float(times[crossing-1]) if crossing else 0.), float(times[crossing])
            clock = prediction[f'first_passage_{entry["threshold"]}_{entry["model"]}_clock']
            lower_clock = float(clock[int(low)])
            upper_clock = float(clock[int(high)]) if high is not None else None
            if (reported['observed_lower'], reported['observed_upper'], reported['observed_lower_clock'], reported['observed_upper_clock']) != (low, high, lower_clock, upper_clock):
                raise AssertionError('A passage interval or censoring boundary changed')
            center = math.log(entry['median_reference_clock'])
            scale = entry['working_log_scale']
            lower = (math.log(max(lower_clock, 1e-30))-center)/scale
            upper = (math.log(upper_clock)-center)/scale if upper_clock is not None else math.inf
            expected = -normal_log_interval(lower, upper)
            close(reported['negative_log_working_probability'], expected, str(key)+'.interval_score', atol=2e-9)
        if set(stored.files) != expected_arrays:
            raise AssertionError('Scheduled score array coverage changed')
    write_json(output, dict(status='passed', trajectories=8, selected_cpu_states=64, path_conditions=48,
        first_passage_conditions=96, field_conditions=field_conditions, checked_sha256=checked,
        verifier_sha256=sha256(__file__), verifier_helpers={name: sha256(Path(__file__).with_name(name)) for name in
            ['verify_scheduled_collectives.py', 'verify_scheduled_frozen_clocks.py']},
        scope='Independent direct-seed-pair reconstruction of every frozen schedule-clock field error, all 256 paired resamples and all censored passage scores. Normal interval probabilities are independently integrated from scaled densities. Each target native path and CPU checkpoint requires its own raw reconstruction. No statistical producer is imported, and no fit is updated using a scheduled target.'))
    print('Verified all scheduled clock error scores', flush=True)


if __name__ == '__main__':
    main()
