#!/usr/bin/env python
"""Qualify independent scheduled-analysis arithmetic on completed native evidence."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from analyze_scaling_updates import covariance_diagnostics
from analyze_scheduled_paths import windows
from analyze_scheduled_predictions import empirical_mean, task_summary
from analyze_size_time import whole_seed_statistics
from freeze_scaling_forecasts import log_interval_probability
from model_rg.provenance import sha256, write_json
from verify_scheduled_clock_scores import normal_log_interval
from verify_scheduled_collectives import statistics, bootstrap_roundoff_allowance
from verify_scheduled_paths import diagnostics, compare
from verify_scheduled_predictions import tasks, equal, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--study', default='scheduled-training-20260908')
    args = p.parse_args()
    study = Path(args.root).resolve()/args.study
    repo = Path(__file__).resolve().parents[1]
    output = study/'verification/analysis-qualification.json'
    if output.exists():
        raise FileExistsError(output)
    evidence = {}

    def load(path):
        path = Path(path)
        evidence[str(path)] = sha256(path)
        return json.loads(path.read_text())

    definitions = {e['id']: e for e in load(study/'data/reasoning-cohort/cohort.json')['examples']}
    selection = load(study/'protocols/reasoning-mechanism-selection.json')
    initial = [c for c in selection['cases'] if c.get('role') != 'scheduled_endpoint']
    if len(initial) != 5:
        raise AssertionError('All five completed initial task states are required')
    task_states = 0
    for case in initial:
        proof = load(study/'verification/reasoning'/(case['name']+'.json'))
        if proof['status'] != 'passed' or proof['case'] != case:
            raise AssertionError('Qualification requires verified task observations')
        path = study/'measurements'/case['name']/'results.json'
        if sha256(path) != proof['checked_sha256'][str(path)]:
            raise AssertionError('A verified task result changed')
        modes = {m['mode']: m for m in load(path)['modes']}
        equal(tasks(modes, definitions), task_summary(modes, definitions), case['name'])
        task_states += 1
    for values in [[0., 1., 2., 3.], [0., 0., 0., 0.], [1e-11, 2e-11, 3e-11, 4e-11], [None, 1., 2., 3.]]:
        equal(summary(values), empirical_mean(values), 'four-seed empirical mean')
    native_arrays = 0
    qa = load(study/'protocols/observation-qualification-selection.json')
    for case in qa['cases']:
        check = load(study/'verification/observation-qualification-native'/(case['name']+'.json'))
        if check['status'] != 'passed' or check['risk']['native_reduction_values_replayed_bytewise'] != 640:
            raise AssertionError('The complete native reduction qualification is required')
        path = study/'runs'/case['run_id']/'measurements.npz'
        evidence[str(path)] = sha256(path)
        with np.load(path) as raw:
            for field in ['shared_update_projection', 'shared_clipped_gradient_projection', 'dense_fields',
                          'dense_heads', 'dense_centroids', 'dense_energies']:
                values = raw[field]
                spacing = 1 if field.startswith('shared') else 64
                if spacing == 64:
                    values = values[raw['steps'] % 64 == 0]
                compare(diagnostics(values, spacing), covariance_diagnostics(values, spacing), case['name']+'.'+field)
                native_arrays += 1
    for values in [np.ones((32, 3)), np.zeros((32, 3)), np.arange(32)[:, None]*np.array([1., 2., 3.])]:
        compare(diagnostics(values, 64), covariance_diagnostics(values, 64), 'constant, zero or linear trajectory')
    protocol = load(study/'protocols/scheduled-path-analysis-selection.json')
    window_counts = {}
    for recipe in protocol['warmup_updates']:
        selected = windows(protocol, recipe)
        for window in selected:
            grid = np.arange(((window['begin']+63)//64)*64, window['end'], 64)
            if len(grid) < 4 or not np.all(np.diff(grid) == 64):
                raise AssertionError('Temporal window is not supported by a regular probe grid')
        thirds = [w for w in selected if w['kind'] == 'floor_third']
        if sum(w['width'] for w in thirds) != 12144 or sum(len(np.arange(((w['begin']+63)//64)*64, w['end'], 64)) for w in thirds) != 189:
            raise AssertionError('The three disjoint floor windows do not cover the declared tail')
        window_counts[recipe] = len(selected)
    for lower, upper in [(-np.inf, -10.), (-10., -9.99), (-1., 2.), (10., 10.01), (10., np.inf), (35., 35.1)]:
        reference = float(log_interval_probability(np.array([lower]), np.array([upper]))[0])
        np.testing.assert_allclose(normal_log_interval(lower, upper), reference, rtol=1e-11, atol=1e-10)
    random = np.random.default_rng(650971)
    scalar_conditions = 0
    for heads, amplitude in itertools.product([4, 14], [0., 1e-9, 1.]):
        values = amplitude*random.normal(size=(4, 8, 3, 7))
        producer, producer_boot = whole_seed_statistics(values, heads)
        independent, independent_boot, _ = statistics(values, heads)
        allowance = max(1e-24, bootstrap_roundoff_allowance(values, heads))
        for key in producer:
            if independent[key] is None:
                if producer[key] is not None:
                    raise AssertionError('Undefined collective statistic changed')
            else:
                np.testing.assert_allclose(producer[key], independent[key], rtol=4e-10, atol=allowance)
        np.testing.assert_allclose(producer_boot, independent_boot, rtol=4e-10, atol=allowance)
        scalar_conditions += 1
    mutation = summary([1., 2., 3., 4.])
    mutation['mean'] += .01
    rejected = False
    try:
        equal(mutation, summary([1., 2., 3., 4.]), 'perturbed empirical mean')
    except AssertionError:
        rejected = True
    if not rejected:
        raise AssertionError('The summary checker accepted a changed empirical mean')
    sources = ['scripts/check_scheduled_analysis.py', 'scripts/analyze_scaling_updates.py',
        'scripts/analyze_scheduled_paths.py', 'scripts/analyze_scheduled_predictions.py', 'scripts/analyze_size_time.py',
        'scripts/freeze_scaling_forecasts.py', 'scripts/verify_scheduled_clock_scores.py',
        'scripts/verify_scheduled_collectives.py', 'scripts/verify_scheduled_paths.py',
        'scripts/verify_scheduled_predictions.py', 'scripts/verify_scheduled_frozen_clocks.py',
        'src/model_rg/scaling.py', 'src/model_rg/provenance.py']
    write_json(output, dict(status='passed', initial_task_states=task_states, candidate_mode_comparisons=5*320*4,
        native_temporal_arrays=native_arrays, deterministic_temporal_fixtures=3, empirical_mean_fixtures=4,
        scalar_collective_conditions=scalar_conditions, directly_enumerated_collective_resamples=scalar_conditions*256,
        normal_interval_fixtures=6, temporal_windows_per_recipe=window_counts, changed_summary_rejected=rejected,
        checked_sha256=evidence, qualified_sources={name: sha256(repo/name) for name in sources},
        scope='Supplemental arithmetic qualification on completed native observations and declared deterministic fixtures. This checks task and temporal reconstruction helpers, direct seed-pair empirical resampling, null handling, phase window grids and independent normal-density quadrature. It does not substitute for verification of the eventual full scheduled analyses, and it does not add scientific training runs or thermodynamic evidence.'))
    print('Qualified scheduled analysis arithmetic on completed native evidence', flush=True)


if __name__ == '__main__':
    main()
