#!/usr/bin/env python3
"""Independently reconstruct all saved conditional-law observations. Historical software qualification is a separate identity. Explicit guards remain active under optimized Python."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256, write_json

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--study', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(a.root).resolve()
    study = Path(a.study).resolve()
    repo = Path(__file__).resolve().parents[1]
    checked = {}

    def check(path, expected=None):
        path = Path(path).resolve()
        if str(path) not in checked:
            checked[str(path)] = sha256(path)
        if expected is not None and checked[str(path)] != expected:
            raise AssertionError('Changed input: ' + str(path))
        return checked[str(path)]

    def load(path):
        check(path)
        return json.loads(Path(path).read_text())
    spec = load(study / 'protocol.json')
    fit = load(study / 'frozen-fit.json')
    innov = load(study / 'frozen-innovations.json')
    analysis = load(study / 'analysis/results.json')
    noise = load(study / 'analysis/innovations.json')
    if not (analysis['scientific_branches'], analysis['scientific_updates'], len(analysis['records'])) == (256, 16384, 128):
        raise ValueError('Conditional-law observation contract failed at original line 26')
    if not (len(spec['cases']) == 16 and len(noise['records']) == 32):
        raise ValueError('Conditional-law observation contract failed at original line 27')
    if not {(c['heads'], c['seed'], c['step']) for c in spec['cases']} == {(h, z, t) for h in [4, 14] for z in range(640101, 640105) for t in [8192, 32768]}:
        raise ValueError('Conditional-law observation contract failed at original line 28')
    for record in [fit, innov]:
        if not set(record['calibration_inputs']) == {c['name'] for c in spec['cases'] if c['role'] == 'calibration'}:
            raise ValueError('Conditional-law observation contract failed at original line 31')
        for name, digest in record['calibration_inputs'].items():
            check(study / 'runs' / name / 'results.json', digest)
    for n, digest in spec['producer_sources'].items():
        check(repo / n, digest)
    for n, digest in spec['inputs_sha256'].items():
        check(n, digest)
    for record, script in [(fit, 'analyze_law_closure.py'), (analysis, 'analyze_law_closure.py'), (innov, 'analyze_law_innovations.py'), (noise, 'analyze_law_innovations.py')]:
        check(repo / 'scripts' / script, record['analyzer_sha256'])
    check(study / 'frozen-fit.json', analysis['frozen_fit_sha256'])
    check(study / 'frozen-innovations.json', noise['frozen_fit_sha256'])
    for role in ['calibration', 'validation']:
        launch = load(study / ('launcher-' + role + '.json'))
        if not (launch['status'] == 'complete' and len(launch['records']) == 8):
            raise ValueError('Conditional-law observation contract failed at original line 42')
        for record in launch['records']:
            check(record['results'], record['sha256'])
    order = np.random.default_rng(641001).permutation(4194304)
    position = np.empty_like(order)
    position[order] = np.arange(len(order))
    all_q = {}
    logit_panels = 0
    for case in spec['cases']:
        folder = study / 'runs' / case['name']
        r = load(folder / 'results.json')
        meta = load(folder / 'manifest.json')
        if not (r['case'] == case and r['branches'] == 16 and (r['horizons'] == [0, 1, 4, 16, 64])):
            raise ValueError('Conditional-law observation contract failed at original line 49')
        if not (r['scientific_updates'] == 1024 and r['restored_parent_bitwise']):
            raise ValueError('Conditional-law observation contract failed at original line 50')
        check(case['parent'], case['parent_sha256'])
        check(case['parent_verification'], case['parent_verification_sha256'])
        check(folder / 'binding.json', meta['binding_sha256'])
        binding = load(folder / 'binding.json')
        for n, d in binding['inputs'].items():
            check(n, d)
        for n, d in binding['source_files'].items():
            check(folder / 'source' / n, d)
        check(folder / 'results.json', meta['results_sha256'])
        check(folder / 'sampling.npz', r['sampling_sha256'])
        with np.load(folder / 'sampling.npz') as sampling:
            blocks = sampling['block_ids']
            crops = sampling['evaluation_crops']
            if not (blocks.shape == (16, 64, 32) and np.all(position[blocks] >= 32 * case['step'])):
                raise ValueError('Conditional-law observation contract failed at original line 58')
            if not all((len(np.unique(b)) == 2048 for b in blocks)):
                raise ValueError('Conditional-law observation contract failed at original line 59')
            if not sampling['cohort'].tolist() == spec['cohort']:
                raise ValueError('Conditional-law observation contract failed at original line 60')
        rows = []
        for entry in r['records']:
            check(folder / entry['raw'], entry['sha256'])
            if not meta['branch_files'][entry['raw']] == entry['sha256']:
                raise ValueError('Conditional-law observation contract failed at original line 64')
            with np.load(folder / entry['raw']) as raw:
                if not (raw['losses'].shape == (64,) and np.all(np.isfinite(raw['gradient_norms']))):
                    raise ValueError('Conditional-law observation contract failed at original line 66')
                np.testing.assert_allclose(raw['q'][:, 0], raw['nll'].mean(1), rtol=0, atol=2e-13)
                np.testing.assert_allclose(raw['q'][:, 1], raw['entropy'].mean(1), rtol=0, atol=2e-13)
                np.testing.assert_array_equal(raw['q'][:, 10:25], raw['layers'].reshape(5, 15))
                if entry['branch'] == 0:
                    if not raw['logits'].shape == (5, 32, 32000):
                        raise ValueError('Conditional-law observation contract failed at original line 71')
                    for k, z in enumerate(raw['logits']):
                        z = z.astype(np.longdouble)
                        maximum = z.max(1, keepdims=True)
                        logp = z - maximum - np.log(np.exp(z - maximum).sum(1, keepdims=True))
                        nll = -logp[np.arange(32), crops[:, 64]]
                        np.testing.assert_allclose(nll, raw['nll'][k], rtol=0, atol=2e-12)
                        logit_panels += 1
                rows.append(raw['q'])
        all_q[case['name']] = np.stack(rows)
        if not np.array_equal(all_q[case['name']][:, 0], np.broadcast_to(all_q[case['name']][0, 0], (16, 29))):
            raise ValueError('Conditional-law observation contract failed at original line 80')
        if case['role'] == 'validation':
            if not datetime.fromisoformat(r['started_at']) > max(datetime.fromisoformat(fit['frozen_at']), datetime.fromisoformat(innov['frozen_at'])):
                raise ValueError('Conditional-law observation contract failed at original line 82')
        print('Reconstructed branch inputs and vocabulary risks', case['name'], flush=True)
    for row in analysis['records']:
        m = fit['maps'][f"h{row['heads']}-t{row['step']}-{row['variant']}"]
        q = all_q[row['case']]
        d = m['dimension']
        center = np.array(m['center'])
        scale = np.array(m['scale'])
        u = (q[:, :, :d] - center) / scale
        h = spec['horizons'].index(row['horizon'])
        matrices = [np.array(x) for x in m['matrices']]
        offsets = np.array(m['offsets'])
        bound = np.zeros(16)
        pred = u[:, 0].copy()
        for j in range(h):
            pred = pred @ matrices[j] + offsets[j]
        for j in range(h):
            cost = np.linalg.norm(u[:, j + 1] - u[:, j] @ matrices[j] - offsets[j], axis=1)
            bound += cost * np.prod(m['lipschitz'][j + 1:h])
        risk = pred[:, 0] * scale[0] + center[0]
        err = risk - q[:, h, 0]
        values = dict(mean_risk_error=err.mean(), risk_rmse=np.sqrt(np.mean(err ** 2)), telescoping_bound_mean=bound.mean(), risk_bound_mean=scale[0] * bound.mean(), risk_wasserstein_empirical=np.abs(err).mean())
        for key, value in values.items():
            np.testing.assert_allclose(value, row[key], rtol=2e-10, atol=2e-11)
        if not np.max(np.linalg.norm(pred - u[:, h], axis=1) - bound) < 1e-09:
            raise ValueError('Conditional-law observation contract failed at original line 100')
    for row in noise['records']:
        q = all_q[row['case']][:, :, 0]
        h = spec['horizons'].index(row['horizon'])
        increments = q[:, 1:h + 1] - q[:, :h]
        centered = increments - increments.mean(0)
        covariance = centered.T @ centered / 16
        np.testing.assert_allclose(covariance.sum(), row['actual_risk_variance'], rtol=2e-10, atol=1e-14)
        np.testing.assert_allclose(np.trace(covariance), row['diagonal_increment_variance'], rtol=2e-10, atol=1e-14)
        model = innov['models'][f"h{row['heads']}-t{row['step']}"]
        histories = np.array(model['centered_histories'])[:, :h, 0]
        drift = np.array(model['drift'])[:h, 0]
        centered_history = histories - histories.mean(0)
        predicted_covariance = centered_history.T @ centered_history / len(histories)
        np.testing.assert_allclose(predicted_covariance.sum(), row['history_predicted_variance'], rtol=2e-10, atol=1e-14)
        np.testing.assert_allclose(np.trace(predicted_covariance), row['white_predicted_variance'], rtol=2e-10, atol=1e-14)
        np.testing.assert_allclose(q[0, 0] + drift.sum() - q[:, h].mean(), row['mean_risk_error'], rtol=2e-10, atol=1e-13)
    output = dict(status='passed', schema='conditional-law-observation-verification-v1', scientific_branches=256, scientific_updates=16384, incoming_states=16, heldout_initializations=2, calibration_initializations=2, logit_panels=logit_panels, verified_prediction_cells=128, innovation_cells=32, verifier_sha256=sha256(__file__), checked_sha256=checked, scope='Independent saved-array reconstruction of all 256 branches, 128 prediction cells, 32 innovation cells and 80 vocabulary panels. Producer, fitting and source-history identities are checked. Historical software qualification and native replay are separate retained records; this command performs no new native replay or historical-toolchain requalification.')
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    write_json(a.output, output)
    print('Complete conditional-law observation reconstruction passed', flush=True)
if __name__ == '__main__':
    main()
