#!/usr/bin/env python
"""Freeze conditional single-pass branch identities and a held-out analysis rule."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from model_rg.provenance import sha256, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(a.root).resolve(); out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    study = root/'scheduled-training-feasible-20260908'
    selection = study/'protocols/onepass-training-selection.json'
    jobs = json.loads(selection.read_text())['jobs']
    repo = Path(__file__).resolve().parents[1]
    cases = []
    for heads in [4, 14]:
        for seed in range(640101, 640105):
            job = next(j for j in jobs if j['heads'] == heads and j['seed'] == seed
                       and j['role'].startswith('data_constant') and j['recipe'] == 'controlled')
            for step in [8192, 32768]:
                folder = study/'runs'/job['run_id']
                parent = folder/('final-training-state.pt' if step == job['steps'] else f'training-state-{step}.pt')
                proof = study/'verification/training'/(job['run_id']+'.json')
                checked = json.loads(proof.read_text())
                digest = sha256(parent)
                assert checked['status'] == 'complete' and checked['verified_files'][str(parent)] == digest
                cases.append(dict(name=f'h{heads}-s{seed}-t{step}', heads=heads, seed=seed,
                    step=step, run_id=job['run_id'], parent=str(parent), parent_sha256=digest,
                    parent_verification=str(proof), parent_verification_sha256=sha256(proof),
                    role='calibration' if seed <= 640102 else 'validation', profile=job['profile']))
    sources = ['scripts/prepare_law_closure.py', 'scripts/measure_law_closure.py',
               'src/model_rg/native.py', 'src/model_rg/training.py',
               'src/model_rg/onepass_regimes.py', 'src/model_rg/schedules.py',
               'src/model_rg/scheduled_regimes.py', 'src/model_rg/criticality.py',
               'src/model_rg/controlled.py', 'src/model_rg/provenance.py']
    inputs = [selection, root/'data/refinedweb-onepass-524288/manifest.json',
              root/'data/refinedweb-onepass-524288/tokens.npy',
              root/'controlled-study-20260905/data/short/tokens.npy',
              root/'controlled-study-20260905/data/short/offsets.npy']
    write_json(out/'protocol.json', dict(schema='conditional-law-closure-protocol-v1',
        frozen_at=datetime.now(timezone.utc).isoformat(), cases=cases,
        branches=16, horizons=[0, 1, 4, 16, 64], cohort=list(range(512, 544)),
        branch_seed=971103, batch_size=32, context_length=64,
        conditional_law='Complete incoming weights, Adam moments, phase and consumed block set fixed. '
            'Each reset branch samples 2048 distinct blocks uniformly from the remaining population. '
            'This averages unrevealed future ordering. Different branches may share blocks; no '
            'trajectory reuses a block or an incoming consumed supervised source position.',
        observation='Native float32 full batch32 with TF32 disabled; float64 reductions. '
            'External next-token risk on the same 32 disjoint evaluation documents. '
            'Full vocabulary logits are retained at every observed horizon in branch zero; '
            'per-context NLL and predictive/row/optimizer coordinates in all branches.',
        analysis=dict(calibration_seeds=[640101,640102], validation_seeds=[640103,640104],
            variants=['identity', 'translation', 'row_predictive', 'optimizer_augmented'],
            fit='Each size, incoming time and block interval uses the two calibration identities only. '
                'Translation is the mean coordinate increment. Affine candidates use ridge 1.0 '
                'on centered standardized coordinates and spectral norm at most 1.0. '
                'Center and scales are calibration-only; risk scale has floor 0.01 nats, '
                'other scales have floor 1e-4 times max(1, absolute calibration mean). '
                'Rollouts start at measured initial retained state. All candidates and outcomes retained.',
            primary='Absolute error of mean external-target NLL, separately for each validation '
                'initialization, head count, incoming time and horizon. Resolution target 0.01 nats. '
                'Mean pathwise retained-coordinate coupling costs bound the empirical-law discrepancy '
                'by telescoping; they are not uniform native Wasserstein certificates.',
            secondary='Paired rollout RMSE, covariance error with population divisor over conditional '
                'branches, per-interval residual dependence, identity baseline, and whole-identity '
                'reporting. No asymptotic confidence claim with two validation identities.',
            selection='Report all four fixed candidates; no best-model selection on validation. '
                'Any amended candidate requires a separate newly frozen validation sample.'),
        producer_sources={n:sha256(repo/n) for n in sources},
        inputs_sha256={str(x):sha256(x) for x in inputs}))
    print(out/'protocol.json', flush=True)


if __name__ == '__main__':
    main()
