#!/usr/bin/env python
"""Independent sampling, finite sums, eigen-score and native replay verification.

Does not import the experiment analyzer, moment estimator or score implementation.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch

from model_rg.provenance import sha256, write_json


def scatter(x):
    values = np.asarray(x, dtype=np.longdouble)
    mean = values.sum(0)/len(values)
    z = values-mean
    covariance = [[sum(z[:, j]*z[:, k])/(len(values)-1)
                   for k in range(z.shape[1])] for j in range(z.shape[1])]
    return np.array(mean, dtype=float), np.array(covariance, dtype=float)


def eigen_score(x, mean, covariance):
    values, vectors = np.linalg.eigh(covariance)
    assert np.all(values > 0)
    projected = (x-mean) @ vectors
    return np.log(values).sum()+np.sum(projected**2/values, axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True); parser.add_argument('--study', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve(); study = Path(args.study).resolve(); output = Path(args.output)
    repo = Path(__file__).resolve().parents[1]
    if output.exists(): raise FileExistsError(output)
    checked = {}
    def check(path, expected=None):
        path = Path(path).resolve(); key = str(path)
        if key not in checked: checked[key] = sha256(path)
        if expected is not None: assert checked[key] == expected, key
        return checked[key]
    def load(path):
        check(path); return json.loads(Path(path).read_text())
    spec = load(study/'protocol.json'); frozen = load(study/'frozen-predictions.json')
    result = load(study/'analysis.json'); launch = load(study/'launcher.json')
    assert result['status'] == launch['status'] == 'complete'
    assert spec['branches'] == 64 and spec['horizons'] == [0, 1, 4, 16, 64]
    assert spec['branch_seed'] == 982451653
    assert frozen['shrinkage'] == .25 and frozen['floor'] == .001
    expected = {(h, s, t) for h in [4, 14] for s in [640103, 640104] for t in [8192, 32768]}
    assert {(c['heads'], c['seed'], c['step']) for c in spec['cases']} == expected
    assert len(spec['cases']) == len(launch['records']) == 8 and len(result['records']) == 48
    assert len({(r['case'], r['target'], r['scale']) for r in result['records']}) == 48
    for name, digest in spec['producer_sources'].items(): check(repo/name, digest)
    for name, digest in spec['inputs_sha256'].items(): check(name, digest)
    for name, digest in frozen['development_inputs'].items(): check(name, digest)
    for name, digest in result['inputs_sha256'].items(): check(name, digest)
    for record in launch['records']: check(record['path'], record['sha256'])
    check(study/'protocol.json', result['protocol_sha256'])
    check(study/'protocol.json', launch['protocol_sha256'])
    check(study/'frozen-predictions.json', result['frozen_sha256'])
    check(repo/'scripts/moment_transport_study.py', result['analyzer_sha256'])
    check(study/'scores.npz', result['scores_sha256'])
    stored_scores = np.load(study/'scores.npz', allow_pickle=False)
    probe = root/'controlled-study-20260905/data/short'
    tokens = np.load(probe/'tokens.npy', mmap_mode='r'); offsets = np.load(probe/'offsets.npy')
    panels = 0; max_logits_error = 0.; max_score_error = 0.; max_identity_error = 0.
    score_differences = {}; records = []
    def read(folder):
        metadata = load(folder/'results.json'); paths = []
        for record in metadata['records']:
            raw = folder/record['raw']; check(raw, record['sha256'])
            with np.load(raw, allow_pickle=False) as z:
                indices = [z['horizons'].tolist().index(t) for t in spec['horizons']]
                paths.append(np.stack([z['nll'][indices].astype(np.longdouble).sum(1)/32,
                                       z['entropy'][indices].astype(np.longdouble).sum(1)/32], axis=1))
        return np.asarray(paths, dtype=float)
    for case in spec['cases']:
        name = case['name']; folder = study/'runs'/name
        run = load(folder/'results.json'); manifest = load(folder/'manifest.json'); binding = load(folder/'binding.json')
        assert run['branches'] == 64 and run['scientific_updates'] == 4096 and run['status'] == 'complete'
        assert run['restored_parent_bitwise'] and run['case'] == case
        assert datetime.fromisoformat(run['started_at']) > datetime.fromisoformat(frozen['frozen_at'])
        check(folder/'results.json', manifest['results_sha256'])
        check(folder/'binding.json', manifest['binding_sha256'])
        for path, digest in binding['inputs'].items(): check(path, digest)
        for path, digest in binding['source_files'].items(): check(folder/'source'/path, digest)
        for path, digest in spec['producer_sources'].items():
            assert binding['source_files'][path] == digest
        check(case['parent'], case['parent_sha256'])
        check(case['parent_verification'], case['parent_verification_sha256'])
        parent = torch.load(case['parent'], map_location='cpu', mmap=True, weights_only=True)
        assert parent['step'] == case['step'] and parent['arguments']['seed'] == case['seed']
        order = np.random.default_rng(parent['arguments']['stream_seed']+1000).permutation(4194304)
        remaining = order[32*case['step']:]
        inverse = np.empty_like(order); inverse[order] = np.arange(len(order))
        rng = np.random.default_rng(spec['branch_seed']+case['seed']*37+case['heads']*101+case['step'])
        check(folder/'sampling.npz', manifest['sampling_sha256'])
        with np.load(folder/'sampling.npz') as z:
            assert z['block_ids'].shape == (64, 64, 32) and z['cohort'].tolist() == spec['cohort']
            crops = np.stack([tokens[i, offsets[i]:offsets[i]+65] for i in z['cohort']])
            np.testing.assert_array_equal(crops, z['evaluation_crops'])
            for blocks in z['block_ids']:
                assert np.unique(blocks).size == 2048
                assert np.all(inverse[blocks] >= 32*case['step'])
                np.testing.assert_array_equal(blocks.ravel(), rng.choice(remaining, 2048, replace=False))
        assert len(run['records']) == 64 and [r['branch'] for r in run['records']] == list(range(64))
        for record in run['records']:
            raw = folder/record['raw']; check(raw, manifest['branch_files'][record['raw']])
            with np.load(raw) as z:
                assert z['horizons'].tolist() == spec['horizons']
                assert z['nll'].shape == z['entropy'].shape == (5, 32)
                assert len(z['losses']) == 64 and all(np.isfinite(z[k]).all() for k in z.files)
                np.testing.assert_allclose(z['q'][:, :2], np.c_[z['nll'].mean(1), z['entropy'].mean(1)], atol=5e-13, rtol=0)
                if 'logits' in z:
                    assert record['branch'] == 0
                    logits = z['logits'].astype(np.longdouble)
                    maximum = logits.max(-1)
                    logsum = maximum+np.log(np.exp(logits-maximum[..., None]).sum(-1))
                    logp = logits-logsum[..., None]
                    nll = -logp[:, np.arange(32), crops[:, 64]]
                    entropy = -(np.exp(logp)*logp).sum(-1)
                    error = float(max(np.abs(nll-z['nll']).max(), np.abs(entropy-z['entropy']).max()))
                    assert error < 3e-12
                    max_logits_error = max(max_logits_error, error); panels += len(logits)
        paths = read(folder)
        original = root/'law-closure-20260910'
        design = np.concatenate([read(part/'runs'/name) for part in
            [original, original/'state-law-calibration', original/'state-law-validation', root/'conditional-memory-20260911']])
        assert design.shape == (72, 5, 2) and paths.shape == (64, 5, 2)
        assert np.all(np.ptp(design[:, 0], axis=0) == 0) and np.all(np.ptp(paths[:, 0], axis=0) == 0)
        np.testing.assert_allclose(design[0, 0], frozen['fits'][name]['incoming'], atol=2e-13, rtol=0)
        np.testing.assert_allclose(paths[:, 0], np.broadcast_to(design[0, 0], (64, 2)), atol=2e-13, rtol=0)
        for target, count in [('risk', 1), ('risk_entropy', 2)]:
            x = np.concatenate([paths[:, 1:, j]-paths[:, :-1, j] for j in range(count)], axis=1)
            d = np.concatenate([design[:, 1:, j]-design[:, :-1, j] for j in range(count)], axis=1)
            mean, cov = scatter(d); scales = np.maximum(np.sqrt(cov.diagonal()), 1e-6)
            standardized = cov/np.outer(scales, scales); diag = np.diag(standardized.diagonal())
            estimates = dict(regularized=.75*standardized+.25*diag+.001*np.eye(4*count),
                             diagonal=diag+.001*np.eye(4*count))
            fit = frozen['fits'][name]['targets'][target]
            for key, value in dict(mean=mean, covariance=cov, scale=scales, **estimates).items():
                np.testing.assert_allclose(value, fit[key], atol=2e-12, rtol=2e-10)
            np.testing.assert_allclose(scatter(design[:, -1, :count])[1], fit['direct_endpoint_covariance'], atol=2e-13, rtol=2e-10)
            for scale_name, temporal in [('fine', np.eye(4)), ('paired', np.array([[1., 1., 0., 0.], [0., 0., 1., 1.]])),
                                          ('endpoint', np.ones((1, 4)))]:
                block = np.kron(np.eye(count), temporal); observed = x @ block.T
                _, empirical_cov = scatter(observed)
                raw_cov = scatter(x)[1]
                identity = np.max(np.abs(block @ raw_cov @ block.T-empirical_cov))
                if scale_name == 'endpoint':
                    identity = max(identity, np.max(np.abs(empirical_cov-scatter(paths[:, -1, :count])[1])))
                max_identity_error = max(max_identity_error, float(identity)); assert identity < 1e-12
                row = next(r for r in result['records'] if (r['case'], r['target'], r['scale']) == (name, target, scale_name))
                np.testing.assert_allclose(row['empirical_covariance'], empirical_cov, atol=2e-13, rtol=2e-10)
                reference_scales = np.sqrt(np.diag(block @ np.diag(scales**2) @ block.T))
                reference = np.outer(reference_scales, reference_scales)
                scores = {}
                for candidate, matrix in estimates.items():
                    covariance = block @ (matrix*np.outer(scales, scales)) @ block.T
                    mu = block @ mean
                    values = eigen_score(observed, mu, covariance); scores[candidate] = values
                    key = name+'__'+target+'__'+scale_name+'__'+candidate
                    max_score_error = max(max_score_error, float(np.max(np.abs(values-stored_scores[key]))))
                    np.testing.assert_allclose(values, stored_scores[key], atol=2e-9, rtol=2e-10)
                    model = row['models'][candidate]
                    np.testing.assert_allclose(covariance, model['covariance'], atol=2e-13, rtol=2e-10)
                    np.testing.assert_allclose(mu, model['mean'], atol=2e-13, rtol=2e-10)
                    relative = np.linalg.norm((covariance-empirical_cov)/reference)/np.linalg.norm(empirical_cov/reference)
                    np.testing.assert_allclose([values.mean(), relative], [model['mean_score'], model['normalized_frobenius']], atol=2e-9, rtol=2e-10)
                diff = scores['regularized']-scores['diagonal']
                np.testing.assert_allclose([diff.mean(), diff.std(ddof=1)/8], [row['score_difference'], row['score_mcse']], atol=2e-9, rtol=2e-10)
                score_differences[(name, target, scale_name)] = float(diff.mean())
        records.append(dict(case=name, scientific_branches=64, single_pass=True, full_state_bound=True))
        print('Verified', name, flush=True)
    assert panels == 40 and len(stored_scores.files) == 96
    for key, summary in result['summaries'].items():
        target, scale_name = key.split('__')
        vals = [v for (name, t, s), v in score_differences.items() if t == target and s == scale_name]
        assert len(vals) == 8 and sum(v < 0 for v in vals) == summary['favorable_states']
        np.testing.assert_allclose(np.mean(vals), summary['mean_difference'], atol=2e-9)
        for seed, value in summary['paired_identity_means'].items():
            vals = [v for (name, t, s), v in score_differences.items() if t == target and s == scale_name and f'-s{seed}-' in name]
            assert len(vals) == 4
            np.testing.assert_allclose(np.mean(vals), value, atol=2e-9)
    for heads in [4, 14]:
        replay = load(study/f'replay-h{heads}.json')
        assert replay['status'] == 'passed' and replay['native_updates'] == 64
        assert replay['full_state_bitwise'] and replay['full_vocabulary_logits_bitwise']
        assert replay['compared_logit_coordinates'] == 5120000
        for name, digest in replay['sources'].items(): check(repo/name, digest)
        for name, digest in replay['inputs'].items(): check(name, digest)
        qualifier = load(study/'qualification'/f'h{heads}-s640103-t8192'/'results.json')
        assert qualifier['status'] == 'complete' and qualifier['reset_replay_bitwise']
        assert qualifier['native_updates'] == 12 and qualifier['scientific_updates'] == 0
    assert result['scientific_branches'] == launch['scientific_branches'] == 512
    assert result['scientific_updates'] == launch['scientific_updates'] == 32768
    write_json(output, dict(status='passed', schema='conditional-moment-transport-verification-v1',
        records=records, scientific_branches=512, scientific_updates=32768, design_paths=576,
        logit_panels=panels, native_replay_updates=128, qualification_updates=24,
        maximum_logits_error=max_logits_error, maximum_score_error=max_score_error,
        maximum_blocking_identity_error=max_identity_error, verifier_sha256=sha256(__file__), checked_sha256=checked,
        scope='All fresh branch sampling, nonreuse, finite moment fits and all 48 score cells reconstructed independently; full vocabulary reductions at 40 panels and two native 64-update replays. Two fixed outer identities; no new pretraining or population covariance certificate.'))
    print('Passed all 512 paths, 48 score cells, 40 vocabulary panels and two native replays.', flush=True)


if __name__ == '__main__':
    main()
