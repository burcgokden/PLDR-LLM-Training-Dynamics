#!/usr/bin/env python
"""Freeze the bounded observation and paired-horizon protocols before execution."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from model_rg.provenance import sha256, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--study', default='observation-closure-20260906')
    a = p.parse_args()
    root = Path(a.root); out = root / a.study
    out.mkdir(parents=True, exist_ok=False)
    for name in ['protocols', 'logs', 'runs', 'analysis', 'scripts', 'qa']:
        (out / name).mkdir()
    dynamics = root / 'criticality-dynamics-20260906'
    jobs = []
    for seed in range(640105, 640117):
        parent = dynamics / 'runs' / f'replication-h14-g1-s{seed}' / 'final-training-state.pt'
        jobs.append(dict(run_id=f'paired-h14-g1-s{seed}', heads=14, seed=seed,
                         multiplier=1, steps=16384, normalization='variance',
                         resume=str(parent), parent_sha256=sha256(parent)))
    common = dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
                  shared_seed=640011, stream_seed=640001,
                  conditioning='Fixed shared initialization and common iid batch realization; remaining initialization is the independent unit.',
                  data_law='3072 fixed RefinedWeb prefixes, uniform document and offset 0..448, batch 32, context 64, external next token.')
    write_json(out/'protocols/paired-horizon.json', dict(**common,
        schema='paired-horizon-observation-v1', jobs=jobs,
        selection='N=14 selected because the four-seed paired empirical interval includes zero. Twelve existing 8192-update initializations are continued to the fixed 16384 horizon.',
        primary='N/(s-1) times squared centered seed head means, averaged over 512 fixed evaluation contexts and five decoders; paired change in absolute normalized-row susceptibility.',
        analysis=dict(paired_seeds=list(range(640101,640117)), resamples=20000, bootstrap_seed=660001,
                      additional_seeds=list(range(640105,640117)),
                      report=['all-16 paired change', 'additional-12 paired change', 'jackknife SE', 'empirical percentiles', 'leave-one-out range', 'each seed mean row change', 'NLL', 'context dispersion']),
        qualification='No calibrated coverage guarantee or stationary, critical, or monotone population law follows from this finite panel.',
        resource_cap='One process per RTX 4090; twelve continuations, each exactly 8192 additional updates.'))
    cells = [dict(heads=h, multiplier=g, step=8192, parent_pattern=f'pilot-h{h}-g{g}-s{{seed}}', parent_study='criticality-dynamics-20260906')
             for h in [4,8,14] for g in [2,3,4]]
    cells += [dict(heads=h, multiplier=16, step=2048,
                   parent_pattern=f'{"scan" if h==2 else "variance"}-h{h}-g16-s{{seed}}', parent_study='criticality-study-20260905') for h in [2,4,8,14]]
    write_json(out/'protocols/arithmetic.json', dict(**common,
        schema='paired-arithmetic-observation-v1', cells=cells, seeds=list(range(640101,640105)),
        rows=list(range(512,1024)), devices=['archived CUDA float32', 'CPU float32 forward, float64 statistic', 'CPU float64 forward and statistic'],
        selection='All nine rate-time endpoints with screened median fields, and all four g=16 saturation conditions; includes a repeated N4,g4 reference cell.',
        primary='Unscreened sample susceptibility in fixed row-ratio units.',
        screen_levels=[1e-15,1e-14,1e-13,1e-12], primary_screen=1e-13,
        relative_tolerance=0.01, absolute_tolerance=1e-8,
        decision='Paired agreement requires both tolerances, including the empirical discrepancy bound. Other values remain qualified observations and are excluded from quantitative scaling fits.',
        precision_scope='Stored weights are held fixed. Float64 is a reference arithmetic, not a certified exact computation; screen levels are diagnostics, not physical floors.',
        resource_cap='CPU with four threads, all 52 checkpoints and complete 512-context cohorts; no training.'))
    print(out, flush=True)


if __name__ == '__main__':
    main()
