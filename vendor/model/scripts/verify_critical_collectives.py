#!/usr/bin/env python3
"""Reconstruct vector and predictive susceptibilities from pairwise distances."""
import argparse
from collections import defaultdict
import json
from numerical_claims import finite_greater
from numerical_validation import load_json_strict, discrepancy, array_discrepancy, finite_array, finite_scalar
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256, write_json

REPO = Path(__file__).resolve().parents[1]


def pair_variance(x, sum_last=False):
    x = finite_array(x, 'collective pair observations').astype(float)
    if len(x)<2:raise ValueError('Two complete replicas required')
    n = len(x)
    total = 0.
    for i in range(n):
        for j in range(i):
            distance = (x[i]-x[j])**2
            total += float(np.mean(distance.sum(-1) if sum_last else distance))
    return total/(n*(n-1))


def verify(root, analysis_path, output, observations_only=False):
    if output.exists():
        raise FileExistsError(output)
    analysis = load_json_strict(analysis_path.read_text())
    p = load_json_strict((root/'protocol.json').read_text())
    if analysis['status'] != 'complete' or analysis['protocol_sha256'] != sha256(root/'protocol.json'):
        raise ValueError('Missing bound complete collective analysis')
    if p['role'] != 'observation' or p['training_updates'] != 0 or analysis['training_updates'] != 0:
        raise ValueError('Unexpected training role')
    retained_checkpoints={}
    checked = {str(analysis_path): sha256(analysis_path)}
    for name, digest in analysis['checked_sha256'].items():
        if observations_only and Path(name).name=='final-state.pt':
            retained_checkpoints[name]=digest
            continue
        if sha256(name) != digest:
            raise ValueError('Changed evidence: '+name)
        checked[name] = digest
    for name, digest in analysis['source_sha256'].items():
        if sha256(REPO/name) != digest:
            raise ValueError('Changed collective reduction source')
        checked[str(REPO/name)] = digest
    parent = Path(p['study'])
    native = load_json_strict((parent/'protocol.json').read_text())
    groups = defaultdict(list)
    replay = identity = 0.
    for job in p['jobs']:
        root_job = root/'runs'/job['run_id']
        m = load_json_strict((root_job/'manifest.json').read_text())
        if m['job'] != job or m['role'] != 'observation' or m['training_updates'] != 0:
            raise ValueError('Observation job identity mismatch')
        with np.load(root_job/'collectives.npz') as raw, np.load(parent/'runs'/job['run_id']/'observations.npz') as old:
            for key in raw.files:
                if not np.isfinite(raw[key]).all():
                    raise ValueError('Nonfinite collective observation')
            np.testing.assert_array_equal(raw['head_fields'], old['heads'][-1])
            np.testing.assert_array_equal(raw['logits'], old[f'logits_{job["steps"]}'])
            replay = max(replay, float(np.max(np.abs(raw['logits']-old[f'logits_{job["steps"]}']))))
            generator = np.random.default_rng(p['projection_seed']).normal(size=(4096, 8))
            expected_basis = np.linalg.qr(generator)[0]
            np.testing.assert_array_equal(raw['operator_basis'], expected_basis)
            np.testing.assert_allclose(raw['operator_basis'].T@raw['operator_basis'], np.eye(8), rtol=0, atol=3e-15)
            energy = raw['head_fields'][..., 3]
            row = raw['head_fields'][..., 0]
            reconstructed = energy-row*np.maximum(energy, 1e-30)
            measured = np.mean(raw['common_centroids']**2, axis=-1)
            difference = float(np.max(np.abs(measured-reconstructed)/(1+np.abs(energy))))
            finite_scalar(difference, 'common energy residual')
            identity = max(identity, difference)
            if difference > 2e-13:
                raise ValueError('Common and centered energies do not reconstruct total energy')
            mean_operator_energy = np.mean(raw['mean_operators']**2, axis=(-2, -1))
            single_operator_energy = np.mean(raw['head_fields'][..., 2]**2, axis=-1)
            if np.any(mean_operator_energy > single_operator_energy+2e-13*(1+single_operator_energy)):
                raise ValueError('Mean operator violates convex squared-norm bound')
        groups[(job['environment'], job['heads'], job['control'])].append((job['seed'], root_job/'collectives.npz'))
    index = {(r['environment'], r['heads'], r['control'], r['field']): r for r in analysis['cells']}
    checked_cells = 0
    max_error = 0.
    for (e, n, g), rows in sorted(groups.items()):
        rows.sort()
        if [s for s, _ in rows] != sorted(native['design']['seeds']):
            continue
        arrays = [np.load(path) for _, path in rows]
        try:
            for field, key in [('common_vector', 'common_centroids'), ('operator_projection', 'operator_projections'),
                               ('complete_mean_operator', 'mean_operators'), ('predictive_sqrt_probability', 'logits')]:
                x = np.stack([a[key].astype(float) for a in arrays])
                record = index[(e, n, g, field)]
                if field == 'predictive_sqrt_probability':
                    exp = np.exp(x-x.max(-1, keepdims=True))
                    x = 2*np.sqrt(exp/exp.sum(-1, keepdims=True))
                    variance = pair_variance(x, sum_last=True)
                    if variance > 4+2e-13:
                        raise ValueError('Predictive susceptibility exceeds its finite simplex bound')
                elif field == 'complete_mean_operator':
                    variance = pair_variance(x)
                else:
                    variance = pair_variance(x.mean(axis=3))
                    single = pair_variance(x)
                    error = discrepancy(single,record['one_head_variance'],'one head variance',atol=2e-12,rtol=2e-12)
                    max_error = max(max_error, error)
                    if finite_greater(error, 2e-12*(1+abs(single)), 'scripts/verify_critical_collectives.py:101'):
                        raise ValueError('One-head vector variance differs')
                    np.testing.assert_allclose(record['parallel_susceptibility']+record['perpendicular_susceptibility'],
                                               record['susceptibility'], rtol=2e-12, atol=2e-12)
                chi = n*variance
                error = discrepancy(chi,record['susceptibility'],'susceptibility',atol=2e-12,rtol=2e-12)
                max_error = max(max_error, error)
                if finite_greater(error, 2e-12*(1+abs(chi)), 'scripts/verify_critical_collectives.py:108'):
                    raise ValueError('Pairwise susceptibility reconstruction differs: '+field)
                checked_cells += 1
        finally:
            for a in arrays:
                a.close()
    if checked_cells != len(index) or analysis['observed_paths'] != len(p['jobs']):
        raise ValueError('Collective observation coverage differs')
    result = dict(schema='critical-collective-verification-v1', status='passed', study=str(root),
                  training_updates=0, observed_paths=len(p['jobs']), reconstructed_cells=checked_cells,
                  maximum_pairwise_reduction_error=max_error, maximum_common_energy_identity_error=identity,
                  maximum_native_logit_replay_error=replay, checked_sha256=checked,
                  verifier_sha256=sha256(__file__),observations_only=observations_only,
                  retained_checkpoint_identities=retained_checkpoints,
                  checkpoint_scope='Retained scientific checkpoint identities are metadata in observations-only mode; all numerical observation inputs are checked.')
    write_json(output, result)
    print(json.dumps({k: result[k] for k in ['status', 'observed_paths', 'reconstructed_cells', 'training_updates']}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--observations-only',action='store_true')
    a = p.parse_args()
    verify(a.study.resolve(), a.analysis.resolve(), a.output.resolve(),a.observations_only)


if __name__ == '__main__':
    main()
