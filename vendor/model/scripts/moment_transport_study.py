#!/usr/bin/env python
"""Freeze, execute and score compatible conditional moment forecasts on two GPUs."""
from companion_paths import child_pythonpath
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from model_rg.moment_rg import block_maps, forecast, moments, score, transport
from model_rg.provenance import sha256, write_json

HORIZONS = [0, 1, 4, 16, 64]
TARGETS = {'risk': 1, 'risk_entropy': 2}


def read_paths(study, case):
    folder = study/'runs'/case['name']
    path = folder/'results.json'
    result = json.loads(path.read_text())
    assert result['status'] == 'complete'
    values = []
    bindings = {str(path): sha256(path)}
    for record in result['records']:
        raw = folder/record['raw']
        assert sha256(raw) == record['sha256']
        with np.load(raw, allow_pickle=False) as data:
            indices = [data['horizons'].tolist().index(t) for t in HORIZONS]
            values.append(np.stack([data['nll'][indices].mean(1),
                                    data['entropy'][indices].mean(1)], axis=1))
        bindings[str(raw)] = record['sha256']
    return np.array(values), result, bindings


def increments(paths, count):
    return np.diff(paths[:, :, :count], axis=1).transpose(0, 2, 1).reshape(len(paths), -1)


def prepare(root, study, repo):
    study.mkdir(parents=True, exist_ok=False)
    original = root/'law-closure-20260910'
    memory = root/'conditional-memory-20260911'
    spec = json.loads((memory/'protocol.json').read_text())
    fits = {}; inputs = {}
    for case in spec['cases']:
        pieces = []
        for folder in [original, original/'state-law-calibration',
                       original/'state-law-validation', memory]:
            paths, _, binding = read_paths(folder, case)
            pieces.append(paths); inputs.update(binding)
        paths = np.concatenate(pieces)
        assert paths.shape == (72, 5, 2) and np.all(np.ptp(paths[:, 0], axis=0) == 0)
        fits[case['name']] = dict(incoming=paths[0, 0].tolist(),
                                 design_paths=72, targets={})
        for target, count in TARGETS.items():
            fit = forecast(increments(paths, count))
            fit['direct_endpoint_covariance'] = moments(paths[:, -1, :count])[1].tolist()
            fits[case['name']]['targets'][target] = fit
    sources = {name: sha256(repo/name) for name in spec['producer_sources']}
    sources.update({name: sha256(repo/name) for name in
                   ['scripts/moment_transport_study.py', 'src/model_rg/moment_rg.py']})
    frozen_at = datetime.now(timezone.utc).isoformat()
    write_json(study/'frozen-predictions.json', dict(status='frozen', frozen_at=frozen_at,
        fits=fits, development_inputs=inputs, shrinkage=.25, floor=.001,
        target_order=TARGETS, horizons=HORIZONS))
    spec.update(schema='conditional-moment-transport-protocol-v1', branches=64,
                branch_seed=982451653, horizons=HORIZONS, frozen_at=frozen_at,
                producer_sources=sources)
    # Bind compact design records in the frozen file, without hashing hundreds of
    # raw branch panels again inside every GPU worker.
    spec['inputs_sha256'][str(study/'frozen-predictions.json')] = sha256(study/'frozen-predictions.json')
    spec['analysis'] = dict(
        conditioner='Eight complete incoming states, two body initialization identities, fixed corpus and evaluation panel.',
        development='All 72 existing paths per state; all earlier experimental outcomes retain their original roles.',
        targets=TARGETS, coordinate_order='Four consecutive risk increments, then four entropy increments when present.',
        estimator='Unbiased design covariance, 25 percent shrinkage toward its diagonal, plus 0.001 times the design variance diagonal; standard deviation floor 1e-6.',
        control='Same mean and variance floor, all off-diagonal entries zero. This is a covariance control, not an independence hypothesis.',
        scale_maps='Four increments, two consecutive paired increments, endpoint sums. Regularize once, then push forward each covariance by B Sigma B^T.',
        primary='Paired log determinant plus Mahalanobis score difference, regularized minus diagonal, for the four risk increments. Negative favors the regularized covariance.',
        secondary='Same score for joint risk and entropy, all coarser scales, normalized Frobenius covariance discrepancies, direct endpoint sample covariance identity.',
        precision='Per-state paired sample standard error across 64 fresh branches. Equal state averages and paired identity summaries are descriptive; no eight-independent-model test.',
        selection='Retain all eight states, both targets, all scales and both candidates. No coefficients, shrinkage or coordinate selection on fresh results.',
        claims='A finite conditional moment forecast; not an autonomous successor law, forcing spectrum, unseen-state fit or thermodynamic exponent.')
    write_json(study/'protocol.json', spec)
    print('Frozen 512 branches, 32768 scientific updates, eight complete incoming states.', flush=True)


def run(study, root, repo, spec):
    if (study/'launcher.json').exists():
        raise FileExistsError(study/'launcher.json')
    env = dict(os.environ, PYTHONPATH=child_pythonpath("model"), PYTHONDONTWRITEBYTECODE='1',
               OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4')
    def lane(heads, device):
        records = []
        cases = [c for c in spec['cases'] if c['heads'] == heads]
        for case in cases:
            command = [sys.executable, str(repo/'scripts/measure_law_closure.py'),
                       '--root', str(root), '--study', str(study), '--case', case['name'], '--device', device]
            if case == cases[0]:
                with (study/(case['name']+'-qualification.log')).open('x') as log:
                    subprocess.run(command+['--qualification'], cwd=repo, env=env,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            with (study/(case['name']+'.log')).open('x') as log:
                subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            path = study/'runs'/case['name']/'results.json'
            result = json.loads(path.read_text())
            assert result['branches'] == 64 and result['scientific_updates'] == 4096
            records.append(dict(case=case['name'], path=str(path), sha256=sha256(path),
                                seconds=result['seconds'], device=device))
            print('Complete', case['name'], round(result['seconds'], 1), 'seconds', flush=True)
        return records
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(lane, 4, 'cuda:0'), pool.submit(lane, 14, 'cuda:1')]
        records = [row for future in futures for row in future.result()]
    write_json(study/'launcher.json', dict(status='complete', records=records,
        protocol_sha256=sha256(study/'protocol.json'), scientific_branches=512,
        scientific_updates=32768, qualification_updates=24))


def analyze(study, spec):
    if (study/'analysis.json').exists():
        raise FileExistsError(study/'analysis.json')
    assert json.loads((study/'launcher.json').read_text())['status'] == 'complete'
    frozen = json.loads((study/'frozen-predictions.json').read_text())
    records = []; inputs = {}; score_arrays = {}
    for case in spec['cases']:
        paths, result, bindings = read_paths(study, case); inputs.update(bindings)
        assert paths.shape == (64, 5, 2)
        assert datetime.fromisoformat(result['started_at']) > datetime.fromisoformat(spec['frozen_at'])
        fit_case = frozen['fits'][case['name']]
        assert np.all(paths[:, 0] == fit_case['incoming'])
        for target, count in TARGETS.items():
            x = increments(paths, count)
            fit = fit_case['targets'][target]
            mean = np.array(fit['mean']); scales = np.array(fit['scale'])
            models = {key: np.array(fit[key])*np.outer(scales, scales)
                      for key in ['regularized', 'diagonal']}
            observed_mean, observed_cov = moments(x)
            endpoint_map = block_maps(count)['endpoint']
            endpoint_identity = float(np.max(np.abs(endpoint_map @ observed_cov @ endpoint_map.T
                                                   - moments(paths[:, -1, :count])[1])))
            for scale_name, block in block_maps(count).items():
                observations = x @ block.T
                empirical_mean, empirical_cov = moments(observations)
                m, pushed_cov = transport(observed_mean, observed_cov, block)
                identity = max(float(np.max(np.abs(m-empirical_mean))),
                               float(np.max(np.abs(pushed_cov-empirical_cov))))
                row = dict(case=case['name'], heads=case['heads'], seed=case['seed'], step=case['step'],
                           target=target, scale=scale_name, dimension=len(block), branches=64,
                           endpoint_identity_error=endpoint_identity, blocking_identity_error=identity,
                           empirical_covariance=empirical_cov.tolist(), models={})
                all_scores = {}
                reference_scale = np.sqrt(np.maximum(np.diag(block @ np.diag(scales**2) @ block.T), 1e-30))
                reference = np.outer(reference_scale, reference_scale)
                for name, covariance in models.items():
                    mu, cov = transport(mean, covariance, block)
                    values = score(observations, mu, cov); all_scores[name] = values
                    discrepancy = (cov-empirical_cov)/reference
                    relative = np.linalg.norm(discrepancy)/np.linalg.norm(empirical_cov/reference)
                    row['models'][name] = dict(mean_score=float(values.mean()),
                        normalized_frobenius=float(relative), covariance=cov.tolist(), mean=mu.tolist())
                    score_arrays[case['name']+'__'+target+'__'+scale_name+'__'+name] = values
                differences = all_scores['regularized']-all_scores['diagonal']
                row.update(score_difference=float(differences.mean()),
                           score_mcse=float(differences.std(ddof=1)/np.sqrt(len(differences))))
                records.append(row)
    summaries = {}
    for target in TARGETS:
        for scale_name in block_maps(TARGETS[target]):
            selected = [r for r in records if r['target'] == target and r['scale'] == scale_name]
            summaries[target+'__'+scale_name] = dict(
                mean_difference=float(np.mean([r['score_difference'] for r in selected])),
                favorable_states=sum(r['score_difference'] < 0 for r in selected), states=8,
                paired_identity_means={str(seed): float(np.mean([r['score_difference'] for r in selected if r['seed'] == seed]))
                                       for seed in [640103, 640104]})
    np.savez_compressed(study/'scores.npz', **score_arrays)
    write_json(study/'analysis.json', dict(status='complete', schema='conditional-moment-transport-results-v1',
        records=records, summaries=summaries, scientific_branches=512, scientific_updates=32768,
        outer_initializations=2, protocol_sha256=sha256(study/'protocol.json'),
        frozen_sha256=sha256(study/'frozen-predictions.json'), inputs_sha256=inputs,
        scores_sha256=sha256(study/'scores.npz'), analyzer_sha256=sha256(__file__)))
    print(json.dumps(summaries, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True); parser.add_argument('--study', required=True)
    parser.add_argument('--phase', choices=['prepare', 'run', 'analyze'], required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve(); study = Path(args.study).resolve()
    repo = Path(__file__).resolve().parents[1]
    if args.phase == 'prepare':
        prepare(root, study, repo); return
    spec = json.loads((study/'protocol.json').read_text())
    for name, digest in spec['producer_sources'].items():
        assert sha256(repo/name) == digest, 'Frozen source changed: '+name
    for name, digest in spec['inputs_sha256'].items():
        assert sha256(name) == digest, 'Frozen input changed: '+name
    if args.phase == 'run':
        run(study, root, repo, spec)
    else:
        analyze(study, spec)


if __name__ == '__main__':
    main()
