#!/usr/bin/env python
"""Independently reconstruct crop occupancies, target positions and native counts."""
import argparse
import json
from numerical_claims import finite_greater
from numerical_validation import load_json_strict
from pathlib import Path

import numpy as np
import torch

from model_rg.provenance import sha256, write_json


def reconstruct_old(rows, offsets, objective, documents=3072):
    crop_counts = np.bincount(rows * 449 + offsets, minlength=documents * 449).reshape(documents, 449)
    output = np.zeros((documents, 513), dtype=np.int64)
    if objective == 'external':
        output[:, 64:] = crop_counts
    else:
        prefix = np.pad(crop_counts.cumsum(axis=1), ((0, 0), (1, 0)))
        position = np.arange(1, 513)
        output[:, 1:] = prefix[:, np.minimum(position, 449)] - prefix[:, np.maximum(0, position - 64)]
    return output


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True)
    p.add_argument('--study', default='scheduled-training-feasible-20260908'); a = p.parse_args()
    study = Path(a.root).resolve() / a.study; repo = Path(__file__).resolve().parents[1]
    selection = study / 'protocols/source-reuse-selection.json'
    spec = load_json_strict(selection.read_text()); out = study / 'analysis/source-reuse'
    proof_path = study / 'verification/source-reuse.json'
    if proof_path.exists():
        raise FileExistsError(proof_path)
    checked = {str(selection): sha256(selection)}
    for path, digest in spec['inputs_sha256'].items():
        if sha256(path) != digest:
            raise AssertionError('A selected original input changed')
        checked[path] = digest
    for name, digest in spec['producer_sources'].items():
        if sha256(repo / name) != digest:
            raise AssertionError('The selected scientific implementation changed')
    meta = load_json_strict((out / 'manifest.json').read_text())
    for filename, field in [('binding.json', 'binding_sha256'), ('measurements.npz', 'raw_sha256'), ('results.json', 'results_sha256')]:
        path = out / filename
        if sha256(path) != meta[field]:
            raise AssertionError('A retained analysis artifact changed')
        checked[str(path)] = meta[field]
    checked[str(out / 'manifest.json')] = sha256(out / 'manifest.json')
    report = load_json_strict((out / 'results.json').read_text())
    groups = {g['name']: g for g in report['groups']}
    if meta['status'] != 'complete' or meta['selection_sha256'] != sha256(selection) or report['status'] != 'complete':
        raise AssertionError('An incomplete or different selected analysis cannot pass')
    expected_records = [dict(case=c['name'], group=f"{c['objective']}-t{c['step']}", checkpoint_target_histograms='exact') for c in spec['cases']]
    if report['records'] != expected_records:
        raise AssertionError('A selected checkpoint pair was omitted or duplicated')
    if len(groups) != 7 or len(report['records']) != 9 or meta['additional_training_updates'] != 0:
        raise AssertionError('The selected counting scope changed')
    torch.set_num_threads(1)
    seed = spec['stream_seed']; max_crops = 131072 * 32
    expected_rows = np.random.default_rng(seed + 1000).integers(0, 3072, size=max_crops)
    expected_offsets = np.random.default_rng(seed + 1001000).integers(0, 449, size=max_crops)
    permutation = np.arange(524288 * 8)
    np.random.default_rng(seed + 1000).shuffle(permutation)
    histograms = {}; verified_groups = set(); cache = {}; max_expectation_error = 0.
    with np.load(out / 'measurements.npz') as raw:
        for case in spec['cases']:
            step = case['step']; k = step * 32; objective = case['objective']; key = f'{objective}-t{step}'
            if case['name'] not in groups[key]['cases']:
                raise AssertionError('A selected pair is absent from its exposure group')
            for side in ['old', 'new']:
                entry = case[side]
                if entry['sampling'] not in cache:
                    with np.load(entry['sampling']) as saved:
                        cache[entry['sampling']] = (saved['rows'], saved['offsets'])
                rows, offsets = [v[:step].reshape(-1) for v in cache[entry['sampling']]]
                if side == 'old':
                    np.testing.assert_array_equal(rows, expected_rows[:k]); np.testing.assert_array_equal(offsets, expected_offsets[:k])
                else:
                    np.testing.assert_array_equal(rows, permutation[:k] // 8)
                    np.testing.assert_array_equal(offsets, (permutation[:k] % 8) * 64)
                if (key, side) not in verified_groups:
                    tokens = np.load(entry['tokens'], mmap_mode='r'); histogram = np.zeros(32000, dtype=np.int64)
                    recorded = groups[key][side]
                    if side == 'old':
                        counts = reconstruct_old(rows, offsets, objective)
                        if objective == 'all_nonpadding':
                            counts[tokens[:3072] == 0] = 0
                        np.testing.assert_array_equal(counts, raw[key + '-old-position-counts'])
                        histogram += np.bincount(tokens[:3072].reshape(-1), weights=counts.reshape(-1), minlength=32000).astype(np.int64)
                        events = int(counts.sum()); unique = int((counts > 0).sum()); maximum = int(counts.max())
                        crop_ids = rows * 449 + offsets
                        position = np.arange(1, 513)
                        incidence = np.zeros((449, 513), dtype=np.int8)
                        for offset in range(449):
                            incidence[offset, offset + 64] = 1 if objective == 'external' else 0
                            if objective == 'all_nonpadding':
                                incidence[offset, offset + 1:offset + 65] = 1
                        probabilities = incidence.sum(axis=0).astype(np.longdouble) / (3072 * 449)
                        expected = -np.expm1(np.longdouble(k) * np.log1p(-probabilities))
                        expected_unique = (expected[None, :] * (tokens[:3072] != 0)).sum() if objective == 'all_nonpadding' else expected.sum() * 3072
                        error = abs(float(expected_unique) - recorded['expected_unique_target_positions'])
                        max_expectation_error = max(max_expectation_error, error)
                        if finite_greater(error, 1e-7, 'scripts/verify_source_reuse.py:103') or recorded['available_target_positions'] != int(((probabilities[None, :] > 0) & ((tokens[:3072] != 0) if objective == 'all_nonpadding' else np.ones((3072, 513), bool))).sum()):
                            raise AssertionError('The occupancy expectation or source support changed')
                    else:
                        crop_ids = rows * 8 + offsets // 64
                        consumed = np.zeros(524288 * 8, dtype=np.uint8)
                        consumed[crop_ids] = 1
                        if int(consumed.sum()) != k:
                            raise AssertionError('The fresh stream repeated a block')
                        np.testing.assert_array_equal(consumed, raw[f't{step}-new-block-occupancy'])
                        for start in range(0, k, 32768):
                            rr, oo = rows[start:start + 32768], offsets[start:start + 32768]
                            values = tokens[rr, oo + 64] if objective == 'external' else tokens[rr[:, None], oo[:, None] + np.arange(1, 65)]
                            values = values.reshape(-1)
                            if objective == 'all_nonpadding':
                                values = values[values != 0]
                            histogram += np.bincount(values, minlength=32000)
                        events = int(histogram.sum()); unique = events; maximum = 1
                    actual = dict(target_events=events, unique_target_positions=unique,
                        repeated_target_events=events - unique, repeated_target_fraction=1 - unique / events,
                        maximum_target_exposure=maximum, unique_crops=int(np.unique(crop_ids).size),
                        crop_events=k, unique_documents=int(np.unique(rows).size))
                    for name, value in actual.items():
                        if recorded[name] != value:
                            raise AssertionError('Incorrect exact source-reuse statistic: ' + name)
                    np.testing.assert_array_equal(histogram, raw[key + '-' + side + '-target-histogram'])
                    histograms[key, side] = histogram; verified_groups.add((key, side))
                state = torch.load(entry['state'], map_location='cpu', mmap=True, weights_only=True)
                if state['step'] != step:
                    raise AssertionError('The checkpoint does not contain the counted prefix')
                np.testing.assert_array_equal(state['supervised_target_counts'].numpy(), histograms[key, side])
            print('Verified', case['name'], flush=True)
    # Boundary/overlap fixtures exercise a direct per-crop reference, with no interval algebra.
    fixture_count = 0
    for offsets in [np.array([0]), np.array([448]), np.array([0, 1, 63, 64, 448, 448])]:
        rows = np.zeros(offsets.size, dtype=np.int64)
        for objective in ['external', 'all_nonpadding']:
            direct = np.zeros((1, 513), dtype=np.int64)
            for offset in offsets:
                if objective == 'external': direct[0, offset + 64] += 1
                else: direct[0, offset + 1:offset + 65] += 1
            np.testing.assert_array_equal(direct, reconstruct_old(rows, offsets, objective, 1)); fixture_count += 1
    write_json(proof_path, dict(status='passed', schema='completed-source-reuse-verification-v1',
        verifier_sha256=sha256(__file__), checked_sha256=checked,
        source_sha256=spec['producer_sources'], matched_pairs=9, checkpoint_histograms=18,
        exposure_groups=7, exact_counting_fixtures=fixture_count,
        maximum_expectation_discrepancy=max_expectation_error, additional_training_updates=0,
        scope='Every selected recorded crop prefix, all integer source-position counts and native target histograms '
        'were independently reconstructed. Both source recipes share exposures. No training update or loss was replayed.'))
    print('Passed nine prefix pairs, eighteen native target histograms and seven exposure groups', flush=True)


if __name__ == '__main__':
    main()
