#!/usr/bin/env python
"""Record scheduled outcome aggregation before scientific scheduled execution."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from model_rg.provenance import sha256, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-20260908');a=p.parse_args()
    study=Path(a.root).resolve()/a.study;repo=Path(__file__).resolve().parents[1]
    path=study/'protocols/scheduled-analysis-selection.json'
    if path.exists():raise FileExistsError(path)
    training=study/'protocols/regime-training-selection.json';jobs=json.loads(training.read_text())['jobs']
    if any((study/'runs'/j['run_id']).exists() for j in jobs):
        raise AssertionError('The initial scheduled analysis record must precede scientific scheduled execution')
    files=[training,*[study/'protocols'/n for n in ['regime-observation-selection.json',
        'scheduled-prefix-selection.json','reasoning-mechanism-selection.json']]]
    sources=['scripts/analyze_scheduled_collectives.py','scripts/analyze_scheduled_predictions.py',
        'scripts/analyze_scaling_collectives.py','scripts/analyze_size_time.py','src/model_rg/scaling.py',
        'scripts/score_scheduled_clocks.py','scripts/freeze_scheduled_clocks.py',
        'scripts/freeze_scaling_forecasts.py','src/model_rg/controlled.py','src/model_rg/provenance.py']
    write_json(path,dict(schema='scheduled-analysis-selection-v1',frozen_at=datetime.now(timezone.utc).isoformat(),
        native_identities=[j['run_id'] for j in jobs],cpu_states=320,precisions=['float32','float64'],
        initialization_ids=list(range(640101,640105)),contexts=512,calibration_contexts=64,
        recipe_pairs=[[near,below] for near in ['reference1','reference2'] for below in ['subcritical1','subcritical2']],
        group_definition='Recipe, head count and exact saved update, retaining every declared four-initialization group. Paired recipe contrasts use the same identities at every common saved update, without interpolating unequal warmup endpoints.',
        uncertainty='All 256 ordered empirical resamples of four whole initialization identities. Each identity remains paired across outcomes and conditions. Percentile ranges are descriptive and do not imply population coverage or multiplicity-adjusted statistical significance.',
        row_coordinates='Native normalized row field, float64-reduced row ratio, absolute centered and total row energies, fixed common-row vectors, head covariance, full mean metric and its orthogonal common/contrast sectors, fixed prediction coordinates. Distinct calibration and evaluation matrix laws remain separate.',
        predictive_laws='Both frozen training and held-out last-target and all-position objectives. Prefix loss comparisons use the same eight targets at lengths16,32,48. Two stochastic continuations of each of100 fixed64-token prompts test deductive stability. Each of80 endpoint states receives all four declared interventions on the same320 task items.',
        undefined_ratios='A zero denominator is retained as null; any undefined seed ratio prevents a four-seed mean. Seeds are never discarded to obtain a finite mean.',
        clock_transfer='The scalar clock forecasts concern only the eight schedule-only controlled trajectories. The source exponent and whole-seed constant-rate fields are frozen before those target executions. Every alternative and out-of-reference prediction is retained, without fitting scheduled outcomes.',
        criticality_scope='Two scheduled head counts and multi-parameter recipe changes do not independently identify a thermodynamic critical surface or asymptotic exponent. Input stability, task correctness and endogenous approach to criticality require separate evidence.',
        inputs_sha256={str(p):sha256(p) for p in files},producer_sources={n:sha256(repo/n) for n in sources}))
    print('Recorded scheduled analysis before all40 scientific trajectories',sha256(path),flush=True)


if __name__=='__main__':main()
