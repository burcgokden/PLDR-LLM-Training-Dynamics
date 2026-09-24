#!/usr/bin/env python
"""Check that endpoint recording preserves the archived four-branch CPU trace."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_dynamics_controls import augmented_distances, log_probabilities
from model_rg.provenance import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    study = Path(args.root) / 'criticality-dynamics-20260906'
    old = study / 'runs/smoke-collective-return-h4-g2-s640101'
    new = study / 'runs/smoke-collective-emission-h4-g2-s640101'
    inputs = {}
    for folder in [old, new]:
        meta = json.loads((folder / 'manifest.json').read_text())
        assert meta['status'] == 'complete'
        for filename, key in [('measurements.npz', 'raw_sha256'), ('binding.json', 'binding_sha256')]:
            assert sha256(folder / filename) == meta[key]
            inputs[str(folder / filename)] = meta[key]
        for branch in meta['branches']:
            assert sha256(folder / branch['checkpoint']) == branch['checkpoint_sha256']
            inputs[str(folder / branch['checkpoint'])] = branch['checkpoint_sha256']
    paired = 0
    with np.load(old / 'measurements.npz') as before, np.load(new / 'measurements.npz') as after:
        for key in before.files:
            np.testing.assert_array_equal(before[key], after[key])
            paired += 1
        tokens = np.load(study.parent / 'controlled-study-20260905/data/short/tokens.npy')
        offsets = np.load(study.parent / 'controlled-study-20260905/data/short/offsets.npy')
        cohort = after['cohort']
        targets = tokens[cohort, offsets[cohort] + 64]
        for branch in ['base', 'early_shared_weights', 'early_shared_weights_moments', 'frozen_early_shared_weights']:
            for stage, index in [('initial', 0), ('final', -1)]:
                logits = after[branch + '_' + stage + '_logits']
                assert logits.shape == (len(cohort), 32000)
                native = torch.from_numpy(logits).double().log_softmax(-1)
                independent = log_probabilities(logits)
                np.testing.assert_allclose(native.numpy(), independent, rtol=1e-12, atol=1e-13)
                np.testing.assert_allclose(-native[np.arange(len(cohort)), targets].numpy(),
                                           after[branch + '_fields'][index, :, 25], rtol=2e-6, atol=2e-6)
            if branch not in ['base', 'early_shared_weights']:
                np.testing.assert_array_equal(after[branch + '_initial_logits'], after['early_shared_weights_initial_logits'])
    tensors = 0
    for branch in ['base', 'early_shared_weights', 'early_shared_weights_moments', 'frozen_early_shared_weights']:
        filename = branch + '-training-state.pt'
        before = torch.load(old / filename, map_location='cpu', mmap=True, weights_only=True)
        after = torch.load(new / filename, map_location='cpu', mmap=True, weights_only=True)
        assert before['step'] == after['step'] == 8200
        assert before['optimizer']['param_groups'] == after['optimizer']['param_groups']
        for key in before['model']:
            torch.testing.assert_close(before['model'][key], after['model'][key], rtol=0, atol=0)
            tensors += 1
        for key, state in before['optimizer']['state'].items():
            for coordinate in ['exp_avg', 'exp_avg_sq', 'step']:
                torch.testing.assert_close(state[coordinate], after['optimizer']['state'][key][coordinate], rtol=0, atol=0)
                tensors += 1
        distances = augmented_distances(before, after)
        assert all(r['squared_distance'] == 0 for group in distances.values() for r in group.values())
        del before, after
    repo = Path(__file__).resolve().parents[1]
    write_json(args.output, dict(status='passed', unchanged_trace_arrays=paired, identical_state_tensors=tensors,
                                 branches=4, updates_per_branch=8, endpoint_logit_arrays=8, inputs=inputs,
                                 checked_sources={name: sha256(repo / name) for name in
                                                  ['scripts/check_collective_emission_smoke.py',
                                                   'scripts/measure_collective_return.py',
                                                   'scripts/analyze_dynamics_controls.py']}))
    print('Endpoint recording preserves all four CPU traces and complete states exactly.')


if __name__ == '__main__':
    main()
