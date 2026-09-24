#!/usr/bin/env python
"""Freeze, execute and analyze a finite same-state memory prediction study."""
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

from model_rg.provenance import sha256, write_json


HORIZONS = [0, 1, 2, 4, 8, 16, 32, 64]
OLD_HORIZONS = [0, 1, 4, 16, 64]
PREDICTORS = ['state_mean', 'last_risk', 'risk_history']


def read_risks(study, case):
    folder = study/'runs'/case['name']
    path = folder/'results.json'
    result = json.loads(path.read_text())
    assert result['status'] == 'complete'
    risks = []
    inputs = {str(path): sha256(path)}
    for entry in result['records']:
        raw = folder/entry['raw']
        assert sha256(raw) == entry['sha256']
        with np.load(raw, allow_pickle=False) as z:
            risks.append(z['nll'].mean(axis=1))
            assert z['horizons'].tolist() == result['horizons']
        inputs[str(raw)] = entry['sha256']
    return np.stack(risks), result, inputs


def fit_predictors(risk):
    """Ridge in physical, declared standardized risk coordinates, no selection."""
    mean = risk.mean(axis=0)
    scale = np.maximum(risk[:, 1:4].std(axis=0, ddof=1), .01)
    z = (risk[:, 1:4] - mean[1:4])/scale
    y = risk[:, 4] - mean[4]
    fits = {}
    for name, indices in [('last_risk', [2]), ('risk_history', [0, 1, 2])]:
        x = z[:, indices]
        gram = x.T @ x / len(x)
        rhs = x.T @ y / len(x)
        beta = np.linalg.solve(gram + .1*np.eye(len(indices)), rhs)
        fits[name] = dict(indices=indices, beta=beta.tolist())
    increments = np.diff(risk, axis=1)
    covariance = np.cov(increments, rowvar=False, ddof=1)
    return dict(mean=mean.tolist(), history_scale=scale.tolist(), regressions=fits,
                increment_covariance=covariance.tolist(), development_branches=len(risk),
                incoming_risk=float(risk[0, 0]))


def predictions(fit, risk):
    mean = np.array(fit['mean'])
    z = (risk[:, 1:4] - mean[1:4])/np.array(fit['history_scale'])
    pred = {'state_mean': np.full(len(risk), mean[4])}
    for name, model in fit['regressions'].items():
        pred[name] = mean[4] + z[:, model['indices']] @ np.array(model['beta'])
    return pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--study', required=True)
    p.add_argument('--phase', required=True, choices=['prepare', 'run', 'analyze'])
    a = p.parse_args()
    root = Path(a.root).resolve(); study = Path(a.study).resolve()
    repo = Path(__file__).resolve().parents[1]
    original = root/'law-closure-20260910'
    if a.phase == 'prepare':
        study.mkdir(parents=True, exist_ok=False)
        spec = json.loads((original/'protocol.json').read_text())
        cases = [c for c in spec['cases'] if c['role'] == 'validation']
        fits = {}; inputs = {str(original/'protocol.json'): sha256(original/'protocol.json')}
        for case in cases:
            paths = []
            for part in [original, original/'state-law-calibration', original/'state-law-validation']:
                risk, result, bindings = read_risks(part, case)
                assert result['horizons'] == OLD_HORIZONS
                paths.append(risk); inputs.update(bindings)
            risk = np.concatenate(paths)
            assert risk.shape == (40, 5) and np.ptp(risk[:, 0]) == 0
            fits[case['name']] = fit_predictors(risk)
        tube_path = original/'state-law-tubes.json'
        inputs[str(tube_path)] = sha256(tube_path)
        sources = {**spec['producer_sources'],
                   'scripts/conditional_memory_study.py': sha256(__file__)}
        for name in sources:
            assert sha256(repo/name) == sources[name], name
        spec.update(schema='conditional-memory-forecast-protocol-v1', cases=cases,
                    branches=32, branch_seed=975119, horizons=HORIZONS,
                    frozen_at=datetime.now(timezone.utc).isoformat(), producer_sources=sources,
                    inputs_sha256={**spec['inputs_sha256'], **inputs})
        spec['analysis'] = dict(
            conditioner='Complete incoming model, Adam, drive and remaining corpus; two outer identities.',
            development='All 40 existing branches per state, including all prior split outcomes, now design data.',
            covariance='Unbiased 4 by 4 increment covariance from 40 design paths. Joint forecast is its sum; '
                       'independent-increment forecast is its trace. No validation refit.',
            predictors=PREDICTORS, predictor_rule='State mean; ridge using risk at 16; ridge using risks at '
                       '1,4,16. Center and scale from design only, scale floor .01 nats, '
                       'ridge .1 on empirical mean-square loss in standardized coordinates.',
            assessment='All eight states and all 32 fresh branches retained. Report mean-risk errors at '
                       '1,4,16,64, endpoint covariance errors in squared nats, and each fixed endpoint '
                       'predictor MSE in squared nats. Identity remains a fixed mean-risk baseline. '
                       'No promotion to uniform .01-nat accuracy unless every mean-risk cell passes. '
                       'History predictors observe their own branch only through update16.',
            tube='Evaluate the already frozen four-horizon tubes on all new paths; no recalibration.',
            uncertainty='Report whole-state and paired-identity summaries. Branches are conditional '
                        'replicates, not additional pretraining identities or critical-scaling samples.',
            selection='No coefficient, predictor, state or horizon selection on fresh results; '
                      'no Gaussian, white-noise, stationarity or critical-exponent claim.')
        write_json(study/'frozen-predictions.json', dict(status='frozen', fits=fits,
                   frozen_at=spec['frozen_at'], sources=sources, development_inputs=inputs,
                   tubes=json.loads(tube_path.read_text())['tubes']))
        spec['inputs_sha256'][str(study/'frozen-predictions.json')] = sha256(study/'frozen-predictions.json')
        write_json(study/'protocol.json', spec)
        print('Frozen eight states, three predictors and covariance controls; 16,384 native updates.', flush=True)
        return
    spec = json.loads((study/'protocol.json').read_text())
    frozen = json.loads((study/'frozen-predictions.json').read_text())
    for name, digest in spec['producer_sources'].items():
        assert sha256(repo/name) == digest, 'Frozen source changed: '+name
    assert sha256(study/'frozen-predictions.json') == spec['inputs_sha256'][str(study/'frozen-predictions.json')]
    if a.phase == 'run':
        if (study/'launcher.json').exists():
            raise FileExistsError(study/'launcher.json')
        def lane(heads, device):
            records = []
            for case in spec['cases']:
                if case['heads'] != heads:
                    continue
                command = [sys.executable, str(repo/'scripts/measure_law_closure.py'),
                           '--root', str(root), '--study', str(study), '--case', case['name'], '--device', device]
                env = dict(os.environ, PYTHONPATH=child_pythonpath("model"), PYTHONDONTWRITEBYTECODE='1',
                           OPENBLAS_NUM_THREADS='4')
                with (study/(case['name']+'.log')).open('x') as log:
                    subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                path = study/'runs'/case['name']/'results.json'
                r = json.loads(path.read_text())
                assert r['branches'] == 32 and r['scientific_updates'] == 2048 and r['status'] == 'complete'
                records.append(dict(case=case['name'], path=str(path), sha256=sha256(path),
                                    seconds=r['seconds'], device=device))
                print('Complete', case['name'], round(r['seconds'], 1), 'seconds', flush=True)
            return records
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(lane, 4, 'cuda:0'), pool.submit(lane, 14, 'cuda:1')]
            records = [r for f in futures for r in f.result()]
        write_json(study/'launcher.json', dict(status='complete', records=records,
                   protocol_sha256=sha256(study/'protocol.json'), scientific_branches=256, scientific_updates=16384))
        return
    if (study/'analysis.json').exists():
        raise FileExistsError(study/'analysis.json')
    launcher = json.loads((study/'launcher.json').read_text())
    assert launcher['status'] == 'complete' and len(launcher['records']) == 8
    rows = []; inputs = {}
    for case in spec['cases']:
        dense, result, binding = read_risks(study, case); inputs.update(binding)
        assert datetime.fromisoformat(result['started_at']) > datetime.fromisoformat(spec['frozen_at'])
        risk = dense[:, [HORIZONS.index(h) for h in OLD_HORIZONS]]
        fit = frozen['fits'][case['name']]
        assert risk.shape == (32, 5) and np.all(risk[:, 0] == fit['incoming_risk'])
        pred = predictions(fit, risk)
        inc = np.diff(risk, axis=1)
        cov = np.cov(inc, rowvar=False, ddof=1)
        variance = float(risk[:, -1].var(ddof=1))
        predicted_cov = np.array(fit['increment_covariance'])
        joint = float(predicted_cov.sum()); white = float(np.trace(predicted_cov))
        tube = frozen['tubes'][case['name']]
        covered = np.all((risk[:, 1:] >= tube['lower']) & (risk[:, 1:] <= tube['upper']), axis=1)
        mse = {name: float(np.mean((values-risk[:, -1])**2)) for name, values in pred.items()}
        errors = (np.array(fit['mean'])[1:] - risk[:, 1:].mean(axis=0))
        identity = fit['incoming_risk'] - risk[:, 1:].mean(axis=0)
        rows.append(dict(case=case['name'], heads=case['heads'], step=case['step'], seed=case['seed'],
                         branches=32, mean_risk_errors=errors.tolist(), identity_risk_errors=identity.tolist(),
                         target_hits=int(np.count_nonzero(np.abs(errors) <= .01)),
                         endpoint_variance=variance, joint_forecast_variance=joint,
                         diagonal_forecast_variance=white, joint_variance_error=joint-variance,
                         diagonal_variance_error=white-variance, mse=mse,
                         mse_ratios={name: value/mse['state_mean'] for name, value in mse.items()},
                         interval_covariance=cov.tolist(), cross_sum=float(cov.sum()-np.trace(cov)),
                         covariance_reconstruction_error=float(cov.sum()-variance),
                         dense_mean_risks=dense.mean(axis=0).tolist(), covered=int(covered.sum()),
                         covered_vector=covered.tolist(), forecasts={k: v.tolist() for k, v in pred.items()}))
    summary = dict(status='complete', schema='conditional-memory-forecast-results-v1', records=rows,
                   validation_branches=256, scientific_updates=16384, outer_initializations=2,
                   target_hits=sum(r['target_hits'] for r in rows), target_cells=32,
                   covered=sum(r['covered'] for r in rows),
                   joint_covariance_better_states=sum(abs(r['joint_variance_error']) < abs(r['diagonal_variance_error']) for r in rows),
                   pooled_mse={name: float(np.mean([r['mse'][name] for r in rows])) for name in PREDICTORS},
                   protocol_sha256=sha256(study/'protocol.json'), frozen_sha256=sha256(study/'frozen-predictions.json'),
                   analyzer_sha256=sha256(__file__), inputs_sha256=inputs)
    write_json(study/'analysis.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ['records', 'inputs_sha256']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
