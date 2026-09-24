#!/usr/bin/env python
"""Reconcile sealed early observations with their completed native trajectories."""
import argparse
import json
from pathlib import Path

import numpy as np

from model_rg.provenance import sha256, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--study', default='scheduled-training-20260908')
    a = p.parse_args()
    study = Path(a.root).resolve() / a.study
    output = study / 'verification/early-prefix-reconciliation.json'
    if output.exists():
        raise FileExistsError(output)
    checked = {}; cache = {}

    def check(path, expected=None):
        path = Path(path).resolve(); stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in cache:
            cache[key] = sha256(path)
        digest = cache[key]
        if expected is not None and digest != expected:
            raise AssertionError('Changed early-checkpoint evidence: ' + str(path))
        checked[str(path)] = digest
        return digest

    def load(path):
        check(path)
        return json.loads(Path(path).read_text())

    selection = study / 'protocols/early-prefix-selection.json'
    spec = load(selection)
    ledger = load(study / 'launcher-early-prefix.json')
    if ledger['status'] != 'observations_complete_awaiting_final_reconciliation':
        raise AssertionError('The entire early observation selection must be complete')
    check(selection, ledger['selection_sha256'])
    records = {r['name']: r for r in ledger['records']}
    if set(records) != {c['name'] for c in spec['cases']} or len(records) != 6:
        raise AssertionError('The early observation selection changed')
    all_prefix = load(study / 'protocols/scheduled-prefix-selection.json')
    results = []; byte_elements = 0
    for case in spec['cases']:
        record = records[case['name']]
        if record['status'] != 'complete':
            raise AssertionError('An early observation was not completed')
        child = study / 'early-checkpoints' / case['name']
        sealed = load(record['sealed_parent'])
        check(record['sealed_parent'], record['sealed_parent_sha256'])
        final = load(case['final_manifest'])
        if final['status'] != 'complete' or final['completed_step'] != 262144:
            raise AssertionError('The final selected native trajectory is incomplete')
        for key in ['heads', 'seed', 'recipe', 'shared_seed', 'stream_seed']:
            if sealed['arguments'][key] != case[key] or final['arguments'][key] != case[key]:
                raise AssertionError('The final trajectory differs from the sealed condition')
        saved = final['saved_states'][str(case['step'])]
        native_state = Path(case['final_manifest']).parent / saved['filename']
        if native_state != Path(case['source_checkpoint']):
            raise AssertionError('The native saved filename changed')
        digest = check(native_state, saved['sha256'])
        check(child / 'checkpoint/state.pt', digest)
        if digest != record['source_checkpoint_sha256'] or digest != sealed['source_checkpoint_sha256']:
            raise AssertionError('The early byte copy differs from the completed trajectory')
        check(case['source_binding'], sealed['source_binding_sha256'])
        early_verifier = child / 'verification/prefix' / (case['name'] + '.json')
        check(early_verifier, record['verification_sha256'])
        early_check = load(early_verifier)
        canonical_cases = [c for c in all_prefix['cases'] if c['step'] == case['step'] and
                           all(c[k] == case[k] for k in ['heads', 'seed', 'recipe', 'shared_seed', 'stream_seed'])]
        if len(canonical_cases) != 1:
            raise AssertionError('An early state has no unique final observation counterpart')
        canonical = canonical_cases[0]
        final_check = load(study / 'verification/prefix' / (canonical['name'] + '.json'))
        for verification in [early_check, final_check]:
            if verification['status'] != 'passed':
                raise AssertionError('Both native-prefix observations must pass independently')
            for name, signature in verification['checked_sha256'].items():
                check(name, signature)
        early_raw = child / 'measurements' / case['name'] / 'measurements.npz'
        final_raw = study / 'measurements' / canonical['name'] / 'measurements.npz'
        check(early_raw); check(final_raw)
        with np.load(early_raw) as x, np.load(final_raw) as y:
            if set(x.files) != set(y.files):
                raise AssertionError('The repeated observation array inventory changed')
            for key in x.files:
                left = x[key]; right = y[key]
                if left.dtype != right.dtype or left.shape != right.shape or left.tobytes() != right.tobytes():
                    raise AssertionError('The final native-prefix observation differs bytewise: ' + key)
                byte_elements += left.size
        results.append(dict(name=case['name'], final_case=canonical['name'],
            source_checkpoint_sha256=digest, completed_trajectory_manifest_sha256=check(case['final_manifest']),
            repeated_observation_byte_equal=True))
    write_json(output, dict(status='passed', states=len(results), independent_training_identities=2,
        additional_independent_training_identities=0, records=results, byte_compared_elements=byte_elements,
        checked_sha256=checked, verifier_sha256=sha256(__file__),
        scope='Every sealed checkpoint matches its original completed trajectory. Repeated final observations match all early retained arrays bytewise. These are six saved states of two existing trajectories, not six independent training runs.'))
    print('Reconciled', len(results), 'early states with completed native trajectories', flush=True)


if __name__ == '__main__':
    main()
