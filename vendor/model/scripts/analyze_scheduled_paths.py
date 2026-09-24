#!/usr/bin/env python
"""Analyze every completed native scheduled path with declared phase boundaries."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_scaling_updates import covariance_diagnostics, excursion_intervals
from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


def windows(spec, recipe):
    warmup = spec['warmup_updates'][recipe]
    floor = spec['floor_start']
    horizon = spec['horizon']
    result = []
    for phase, start, stop in [('warmup', 0, warmup), ('annealing', warmup, floor), ('floor', floor, horizon)]:
        result.append(dict(phase=phase, kind='complete_phase', begin=start, end=stop, width=stop-start))
        for width in spec['phase_local_block_widths']:
            for begin in range(start, stop-width+1, width):
                result.append(dict(phase=phase, kind='phase_local_block', begin=begin, end=begin+width, width=width))
    if (horizon-floor) % 3:
        raise AssertionError('The declared equal floor thirds no longer partition the floor')
    width = (horizon-floor)//3
    for part in range(3):
        start = floor+part*width
        result.append(dict(phase='floor', kind='floor_third', begin=start, end=start+width, width=width))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--study', default='scheduled-training-20260908')
    args = p.parse_args()
    study = Path(args.root).resolve()/args.study
    repo = Path(__file__).resolve().parents[1]
    protocol = study/'protocols/scheduled-path-analysis-selection.json'
    spec = json.loads(protocol.read_text())
    for name, digest in spec['producer_sources'].items():
        if sha256(repo/name) != digest:
            raise AssertionError('The recorded scheduled temporal analysis implementation changed')
    for name, digest in spec['inputs_sha256'].items():
        if sha256(name) != digest:
            raise AssertionError('The recorded scheduled temporal analysis selection changed')
    selection = study/'protocols/regime-training-selection.json'
    jobs = json.loads(selection.read_text())['jobs']
    if len(jobs) != 40 or spec['run_ids'] != [j['run_id'] for j in jobs]:
        raise AssertionError('All 40 declared complete native trajectories are required')
    inputs = [protocol, selection, study/'protocols/regime-shared-update-observation.json']
    for job in jobs:
        parent = study/'runs'/job['run_id']
        meta = json.loads((parent/'manifest.json').read_text())
        proof_path = study/'verification/training'/(job['run_id']+'.json')
        proof = json.loads(proof_path.read_text())
        if meta['status'] != 'complete' or meta['start_step'] != 0 or meta['completed_step'] != spec['horizon']:
            raise AssertionError('A selected native path is incomplete')
        if proof['status'] != 'complete' or proof['trajectory']['run_id'] != job['run_id']:
            raise AssertionError('Every native path requires independent reconstruction')
        for name in ['manifest.json', 'measurements.npz']:
            path = parent/name
            if sha256(path) != proof['verified_files'][str(path)]:
                raise AssertionError('A reconstructed native path changed')
            inputs.append(path)
        inputs.append(proof_path)
    output = study/'analysis/training-paths'
    output.mkdir(parents=True, exist_ok=False)
    bind_run(output, inputs, vars(args))
    records = []
    retained = {}
    for job in jobs:
        parent = study/'runs'/job['run_id']
        meta = json.loads((parent/'manifest.json').read_text())
        name = job['run_id']
        with np.load(parent/'measurements.npz') as raw:
            steps = raw['steps']
            heads = raw['dense_heads'].astype(float)
            fields = raw['dense_fields'].astype(float)
            observed = dict(common_centroid=raw['dense_centroids'], absolute_row_energy=raw['dense_energies'][..., 0],
                total_metric_energy=raw['dense_energies'][..., 1], row=heads[..., 2].mean(-1),
                attention=heads[..., 0].mean(-1), operator_rms=heads[..., 3].mean(-1),
                prediction_entropy=fields[..., 24], nll=fields[..., 25], logit_projection=fields[..., 16:24])
            updates = raw['shared_update_projection']
            gradients = raw['shared_clipped_gradient_projection']
            moments = raw['shared_update_moments']
            rates = raw['applied_learning_rates']
            losses = raw['losses']
            if updates.shape != (262144, 16) or gradients.shape != updates.shape or moments.shape != (262144, 3):
                raise AssertionError('A native update or fixed projection was omitted')
            regular = steps % 64 == 0
            if not np.array_equal(steps[regular], np.arange(0, 262145, 64)):
                raise AssertionError('The regular temporal probe grid changed')
            result = dict(run_id=name, recipe=job['recipe'], heads=job['heads'], seed=job['seed'],
                shared_seed=job['shared_seed'], stream_seed=job['stream_seed'], windows=[], excursions={})
            for window in windows(spec, job['recipe']):
                begin, end = window['begin'], window['end']
                mask = regular & (steps >= begin) & (steps < end)
                expected = np.arange(((begin+63)//64)*64, end, 64)
                if not np.array_equal(steps[mask], expected) or len(expected) < 4:
                    raise AssertionError('The declared phase window changed its observation grid')
                row = dict(**window, probe_first=int(expected[0]), probe_last=int(expected[-1]),
                    probe_samples=len(expected), observations={})
                for field, array in [('shared_update_projection', updates), ('shared_clipped_gradient_projection', gradients)]:
                    row['observations'][field] = covariance_diagnostics(array[begin:end], 1)
                for field, array in observed.items():
                    row['observations'][field] = covariance_diagnostics(array[mask], 64)
                weights = moments[begin:end].mean(0)
                row['native_moments'] = dict(mean_squared_update=float(weights[0]), mean_squared_weight=float(weights[1]),
                    mean_update_times_weight=float(weights[2]), radial_rate=float(weights[2]/weights[1]) if weights[1] else None,
                    mean_preupdate_loss=float(losses[begin:end].mean()),
                    first_applied_learning_rates=rates[begin].tolist(), last_applied_learning_rates=rates[end-1].tolist(),
                    mean_applied_learning_rates=rates[begin:end].mean(0).tolist())
                result['windows'].append(row)
            row_path = observed['row'].mean(axis=(1, 2))
            for threshold in spec['thresholds']:
                result['excursions'][str(threshold)] = excursion_intervals(steps, row_path, threshold)
            result['whole_path'] = dict(row_minimum=float(row_path.min()), row_maximum=float(row_path.max()),
                row_last=float(row_path[-1]), mean_preupdate_loss=float(losses.mean()),
                mean_squared_native_update=float(moments[:, 0].mean()),
                projected_weight_displacement_squared=float(np.mean(updates.sum(0)**2)))
            retained[name+'_steps'] = steps
            retained[name+'_row'] = row_path
            retained[name+'_common_centroid_mean'] = observed['common_centroid'].mean(1)
            retained[name+'_prediction_entropy_mean'] = observed['prediction_entropy'].mean(1)
            retained[name+'_nll_mean'] = observed['nll'].mean(1)
        records.append(result)
        print('Completed phase-separated scheduled path diagnostics', name, flush=True)
    write_json(output/'results.json', dict(status='complete', schema='scheduled-temporal-laws-v1', runs=records,
        selection_sha256=sha256(protocol), phase_scope=spec['phase_rule'], probe_scope=spec['probe_rule'],
        floor_scope=spec['floor_rule'], uncertainty=spec['uncertainty'], inference=spec['inference'],
        development_scope=spec['development_scope']))
    np.savez_compressed(output/'measurements.npz', **retained)
    write_json(output/'manifest.json', dict(status='complete', runs=40, binding_sha256=sha256(output/'binding.json'),
        results_sha256=sha256(output/'results.json'), raw_sha256=sha256(output/'measurements.npz')))
    print('Completed all 40 scheduled native temporal diagnostics', flush=True)


if __name__ == '__main__':
    main()
