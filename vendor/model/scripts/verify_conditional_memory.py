#!/usr/bin/env python
"""Independent raw sampling, vocabulary, ridge and covariance reconstruction."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch

from model_rg.provenance import sha256, write_json


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True)
    p.add_argument('--study', required=True); p.add_argument('--output', required=True)
    args = p.parse_args(); root = Path(args.root).resolve(); study = Path(args.study).resolve()
    repo = Path(__file__).resolve().parents[1]; target = Path(args.output)
    if target.exists(): raise FileExistsError(target)
    checked = {}
    def check(path, digest=None):
        path = Path(path).resolve(); name = str(path)
        if name not in checked: checked[name] = sha256(path)
        if digest is not None: assert checked[name] == digest, name
        return checked[name]
    def load(path):
        check(path); return json.loads(Path(path).read_text())
    spec = load(study/'protocol.json'); frozen = load(study/'frozen-predictions.json')
    analysis = load(study/'analysis.json'); launcher = load(study/'launcher.json')
    assert spec['branches'] == 32 and spec['horizons'] == [0, 1, 2, 4, 8, 16, 32, 64]
    expected = {(h, s, t) for h in [4, 14] for s in [640103, 640104] for t in [8192, 32768]}
    assert {(c['heads'], c['seed'], c['step']) for c in spec['cases']} == expected
    assert len(spec['cases']) == len(analysis['records']) == len(launcher['records']) == 8
    assert analysis['status'] == launcher['status'] == 'complete'
    for name, digest in spec['inputs_sha256'].items(): check(name, digest)
    for name, digest in spec['producer_sources'].items(): check(repo/name, digest)
    check(repo/'scripts/conditional_memory_study.py', analysis['analyzer_sha256'])
    check(study/'protocol.json', analysis['protocol_sha256'])
    check(study/'frozen-predictions.json', analysis['frozen_sha256'])
    for entry in launcher['records']: check(entry['path'], entry['sha256'])
    panels = 0; max_nll = 0.; max_fit = 0.; max_cov = 0.; rows = []
    def old_risk(folder):
        result = load(folder/'results.json'); values = []
        for entry in result['records']:
            file = folder/entry['raw']; check(file, entry['sha256'])
            with np.load(file, allow_pickle=False) as raw:
                assert raw['horizons'].tolist() == [0, 1, 4, 16, 64]
                values.append(np.sum(raw['nll'].astype(np.longdouble), axis=1)/32)
        return np.array(values, dtype=np.float64)
    probe = root/'controlled-study-20260905/data/short'
    tokens = np.load(probe/'tokens.npy', mmap_mode='r'); offsets = np.load(probe/'offsets.npy')
    for case in spec['cases']:
        name = case['name']; folder = study/'runs'/name
        r = load(folder/'results.json'); meta = load(folder/'manifest.json'); binding = load(folder/'binding.json')
        assert r['status'] == 'complete' and r['branches'] == 32 and r['scientific_updates'] == 2048
        assert r['restored_parent_bitwise'] and r['case'] == case
        assert datetime.fromisoformat(r['started_at']) > datetime.fromisoformat(frozen['frozen_at'])
        check(folder/'results.json', meta['results_sha256']); check(folder/'binding.json', meta['binding_sha256'])
        for path, digest in binding['inputs'].items(): check(path, digest)
        for path, digest in binding['source_files'].items(): check(folder/'source'/path, digest)
        check(case['parent'], case['parent_sha256'])
        parent = torch.load(case['parent'], map_location='cpu', mmap=True, weights_only=True)
        order = np.random.default_rng(parent['arguments']['stream_seed']+1000).permutation(4194304)
        remaining = order[32*case['step']:]
        inverse = np.empty_like(order); inverse[order] = np.arange(len(order))
        generator = np.random.default_rng(spec['branch_seed']+case['seed']*37+case['heads']*101+case['step'])
        check(folder/'sampling.npz', meta['sampling_sha256'])
        with np.load(folder/'sampling.npz', allow_pickle=False) as z:
            sampled = z['block_ids']; cohort = z['cohort']; crops = z['evaluation_crops']
            assert sampled.shape == (32, 64, 32) and cohort.tolist() == spec['cohort']
            expected_crops = np.stack([tokens[i, offsets[i]:offsets[i]+65] for i in cohort])
            np.testing.assert_array_equal(crops, expected_crops)
            for indices in sampled:
                assert np.unique(indices).size == 2048 and np.all(inverse[indices] >= 32*case['step'])
                np.testing.assert_array_equal(indices.ravel(), generator.choice(remaining, 2048, replace=False))
        actual = []
        for entry in r['records']:
            file = folder/entry['raw']; check(file, entry['sha256'])
            assert meta['branch_files'][entry['raw']] == entry['sha256']
            with np.load(file, allow_pickle=False) as z:
                assert z['horizons'].tolist() == spec['horizons']
                nll = z['nll'].astype(np.longdouble)
                assert nll.shape == (8, 32) and np.isfinite(nll).all()
                risk = nll.sum(axis=1)/32
                np.testing.assert_allclose(risk, z['q'][:, 0], atol=5e-13, rtol=2e-13)
                if 'logits' in z:
                    logits = z['logits'].astype(np.longdouble)
                    peak = logits.max(axis=-1)
                    logsum = peak+np.log(np.exp(logits-peak[..., None]).sum(axis=-1))
                    selected = logits[:, np.arange(32), crops[:, 64]]
                    error = float(np.max(np.abs(logsum-selected-nll))); max_nll = max(max_nll, error)
                    assert error < 2e-12
                    panels += logits.shape[0]
                actual.append(risk[[0, 1, 3, 5, 7]])
        actual = np.asarray(actual, dtype=np.float64)
        original = root/'law-closure-20260910'
        design = np.concatenate([old_risk(part/'runs'/name) for part in
                                 [original, original/'state-law-calibration', original/'state-law-validation']])
        assert design.shape == (40, 5)
        fit = frozen['fits'][name]; mean = design.mean(0)
        scale = np.maximum(design[:, 1:4].std(axis=0, ddof=1), .01)
        np.testing.assert_allclose(mean, fit['mean'], atol=2e-13, rtol=0)
        np.testing.assert_allclose(scale, fit['history_scale'], atol=2e-13, rtol=0)
        assert np.ptp(actual[:, 0]) == 0
        np.testing.assert_allclose(actual[:, 0], fit['incoming_risk'], atol=2e-13, rtol=0)
        forecasts = {'state_mean': np.full(32, mean[-1])}
        for model, indices in [('last_risk', [2]), ('risk_history', [0, 1, 2])]:
            x = ((design[:, 1:4]-mean[1:4])/scale)[:, indices]
            y = design[:, -1]-mean[-1]
            augmented = np.vstack([x, 2*np.eye(len(indices))])
            beta = np.linalg.lstsq(augmented, np.r_[y, np.zeros(len(indices))], rcond=None)[0]
            error = float(np.max(np.abs(beta-np.array(fit['regressions'][model]['beta']))))
            max_fit = max(max_fit, error); assert error < 1e-11
            forecasts[model] = mean[-1]+(((actual[:, 1:4]-mean[1:4])/scale)[:, indices]@beta)
        def scatter(values):
            values = values.astype(np.longdouble)
            centered = values-values.sum(0)/len(values)
            return np.array([[sum(centered[:, i]*centered[:, j])/(len(values)-1)
                              for j in range(values.shape[1])] for i in range(values.shape[1])], dtype=float)
        d = design[:, 1:]-design[:, :-1]; covariance = scatter(d)
        np.testing.assert_allclose(covariance, fit['increment_covariance'], rtol=1e-11, atol=1e-14)
        inc = actual[:, 1:]-actual[:, :-1]; measured = scatter(inc)
        variance = float(scatter(actual[:, -1:])[0, 0]); max_cov = max(max_cov, abs(measured.sum()-variance))
        row = next(v for v in analysis['records'] if v['case'] == name)
        assert (row['heads'], row['seed'], row['step']) == (case['heads'], case['seed'], case['step'])
        np.testing.assert_allclose(measured, row['interval_covariance'], rtol=2e-11, atol=1e-14)
        for key, value in [('endpoint_variance', variance), ('joint_forecast_variance', covariance.sum()),
                           ('diagonal_forecast_variance', np.trace(covariance)),
                           ('joint_variance_error', covariance.sum()-variance),
                           ('diagonal_variance_error', np.trace(covariance)-variance)]:
            np.testing.assert_allclose(value, row[key], atol=1e-13, rtol=1e-10)
        for model, prediction in forecasts.items():
            mse = np.mean((prediction-actual[:, -1])**2)
            np.testing.assert_allclose(prediction, row['forecasts'][model], atol=1e-12, rtol=0)
            np.testing.assert_allclose(mse, row['mse'][model], atol=1e-13, rtol=1e-10)
        errors = mean[1:]-actual[:, 1:].mean(0)
        np.testing.assert_allclose(errors, row['mean_risk_errors'], atol=1e-12, rtol=0)
        np.testing.assert_allclose(mean[0]-actual[:, 1:].mean(0), row['identity_risk_errors'], atol=1e-12, rtol=0)
        assert row['target_hits'] == int(np.count_nonzero(abs(errors) <= .01))
        tube = frozen['tubes'][name]
        inside = np.all((actual[:, 1:] >= tube['lower']) & (actual[:, 1:] <= tube['upper']), axis=1)
        assert inside.tolist() == row['covered_vector'] and int(inside.sum()) == row['covered']
        rows.append(dict(case=name, branches=32, single_pass=True, full_state_bound=True))
        print('Verified', name, flush=True)
    assert panels == 64
    assert analysis['covered'] == sum(r['covered'] for r in analysis['records'])
    assert analysis['target_hits'] == sum(r['target_hits'] for r in analysis['records'])
    assert analysis['joint_covariance_better_states'] == sum(abs(r['joint_variance_error']) < abs(r['diagonal_variance_error']) for r in analysis['records'])
    for name, value in analysis['pooled_mse'].items():
        np.testing.assert_allclose(value, np.mean([r['mse'][name] for r in analysis['records']]), atol=1e-14)
    replays = []
    for heads in [4, 14]:
        path = study/f'replay-h{heads}.json'; replay = load(path)
        assert replay['status'] == 'passed' and replay['native_updates'] == 64
        assert replay['full_state_bitwise'] and replay['full_vocabulary_logits_bitwise']
        assert replay['compared_logit_coordinates'] == 8192000
        for name, digest in replay['sources'].items(): check(repo/name, digest)
        for name, digest in replay['inputs'].items(): check(name, digest)
        replays.append(dict(path=str(path), sha256=check(path)))
    write_json(target, dict(status='passed', schema='conditional-memory-verification-v1', records=rows,
               scientific_branches=256, scientific_updates=16384, native_replay_updates=128,
               original_design_paths=320, logit_panels=panels, maximum_nll_error=max_nll,
               maximum_coefficient_error=max_fit, maximum_covariance_error=max_cov,
               replays=replays, verifier_sha256=sha256(__file__), checked_sha256=checked,
               scope='All selected fresh branches, frozen forecasts and sampling reconstructed; '
                     '64 full-vocabulary panels; two complete native replays. Same eight states '
                     'and two pretraining identities, without a thermodynamic interpretation.'))
    print('Passed all 256 fresh paths and 64 vocabulary panels.', flush=True)


if __name__ == '__main__': main()
