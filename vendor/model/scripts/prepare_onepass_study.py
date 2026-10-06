#!/usr/bin/env python
"""Freeze the complete finite-duration study on nonrepeated RefinedWeb blocks."""
from companion_paths import required_input
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from model_rg.onepass_regimes import recipe
from model_rg.provenance import sha256, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--study', default='scheduled-training-feasible-20260908')
    a = p.parse_args()
    root = Path(a.root).resolve()
    study = root/a.study
    repo = Path(__file__).resolve().parents[1]
    old = root/'scheduled-training-20260908'
    selection = study/'protocols/onepass-training-selection.json'
    if selection.exists() or list((study/'runs').glob('*')):
        raise AssertionError('The selection must precede every new GPU artifact')
    now = datetime.now(timezone.utc).isoformat()
    paths = [study/'verification/onepass-data.json', study/'verification/expanded-data.json',
             study/'qualification/onepass-stream/manifest.json',
             study/'qualification/onepass-stream/results.json',
             old/'development/early-training-risk/verification.json',
             old/'development/feasibility-redesign/no-repeat-data-policy.json',
             root/'critical-scaling-20260906/verification/full-selected/raw.json',
             root/'critical-scaling-20260906/verification/full-selected/statistics.json',
             root/'data/refinedweb-onepass-524288/manifest.json',
             study/'protocols/onepass-data-selection.json']
    for path in paths[:4]:
        if json.loads(path.read_text())['status'] not in ['passed','complete']:
            raise AssertionError('The nonrepeated corpus and stream must pass before selection')
    previous = json.loads((old/'protocols/regime-training-selection.json').read_text())
    references = {key:value for key,value in previous['reference_inputs'].items()
                  if str(Path(required_input('reference-schedule'))) in key or '/qualification/' in key or '2603.23539' in key}
    references.update({str(path):sha256(path) for path in paths})
    sources = dict(previous['producer_sources'])
    sources.pop('scripts/train_scheduled_regimes.py')
    sources.pop('scripts/train_scheduled_scaling.py')
    for name in ['scripts/train_onepass_regimes.py','src/model_rg/onepass_regimes.py']:
        sources[name] = sha256(repo/name)

    def job(name, heads, seed, role='compact', horizon=32768, steps=40960,
            warmup_override=-1, run_id=None):
        profile = recipe(name,heads,horizon,warmup_override)
        if role == 'compact':
            saved = [profile['warmup_steps'],8192,16384,24576,32768,40960]
        elif role == 'data_schedule':
            saved = [profile['warmup_steps'],8192,32768,65536]
        elif role == 'data_constant':
            saved = [8192,32768,65536,98304,131072]
        elif role == 'data_constant_short':
            saved = [8192,32768]
        else:
            saved = []
        return dict(run_id=run_id or f'{role}-{name}-h{heads}-s{seed}', recipe=name, heads=heads,
                    seed=seed, shared_seed=640011, stream_seed=640001,
                    steps=steps, schedule_horizon=horizon, warmup_override=warmup_override,
                    profile=profile, role=role, microbatch=32, checkpoint_decoders=False,
                    probe_every=64, save_steps=','.join(map(str,saved)))

    jobs = []
    for seed in range(640101,640105):
        for name, sizes in [('controlled',[4,14]), ('reference1',[4,8,14]),
                            ('subcritical1',[4,8,14]), ('reference2',[4]), ('subcritical2',[4])]:
            jobs.extend(job(name,n,seed) for n in sizes)
    jobs.extend(job(name,14,640101,'data_schedule',250000,65536)
                for name in ['reference1','subcritical1'])
    jobs.extend(job('controlled',n,seed,'data_constant',0,131072)
                for seed in [640101,640102] for n in [4,14])
    jobs.extend(job('controlled',n,seed,'data_constant_short',0,32768)
                for seed in range(640101,640105) for n in [4,8,14]
                if not (n in [4,14] and seed in [640101,640102]))
    if len(jobs) != 54 or sum(j['steps'] for j in jobs) != 2555904:
        raise AssertionError('The complete selected training inventory differs')

    qa = []
    for name, n, drive, device in [('controlled',4,768,0), ('reference1',4,768,0),
                                  ('reference1',8,768,0), ('reference1',14,768,1),
                                  ('controlled',4,0,0)]:
        ident = f'qualification-{name}-h{n}-d{drive}-continuous'
        j = job(name,n,650631,'qualification',drive,1024,128 if drive else -1,ident)
        j.update(device=device,stream_seed=650641)
        qa.append(j)
    continuous = 'qualification-reference1-h14-d768-continuous'
    for suffix, stop in [('prefix',512),('resumed',1024)]:
        j = job('reference1',14,650631,'qualification',768,stop,128,
                f'qualification-reference1-h14-d768-{suffix}')
        j.update(device=1,stream_seed=650641)
        if suffix == 'resumed':
            j.update(resume_from_run='qualification-reference1-h14-d768-prefix', resume_from_step=512)
        qa.append(j)
    common = dict(schema='onepass-training-selection-v1', frozen_at=now,
        producer_sources=sources, reference_inputs=references,
        training_entrypoint='scripts/train_onepass_regimes.py',
        adaptation='Five native PLDR decoders, head dimension 64, eight shared metric residual units of width 170. Variance-normalized initialization, one conditioned shared initial state and data stream. CUDA float32, no TF32, full batch 32, context 64. Every new trajectory consumes unique source target positions from 524288 distinct RefinedWeb documents. Reference recipes retain their peak rates, absolute warmup durations, Adam coefficients, epsilon, weight decay, clipping and all-nonpadding-target objective. The main cosine horizon is 32768, followed by 8192 updates at the nonzero floor. The constant controls retain the controlled last-target objective. This is a finite adaptation, not the original context-1024, approximately eight-billion-token pretraining program.',
        data=dict(corpus=str(root/'data/refinedweb-onepass-524288'), documents=524288,
                  block_targets=64, blocks=4194304, maximum_updates=131072,
                  sampling='A prefix of a single uniform permutation of all disjoint target blocks. Independent initializations share the declared data stream. No target position repeats within any run.'),
        retention='All selected outcomes, numerical failures, administrative censoring and prediction errors remain in the internal record. No completed outcome may be dropped to improve agreement.',
        classifications=previous['reference_classifications'],
        interpretation='Reference Near and Below labels identify source hyperparameters, not classifications established in this controlled family. Finite-width fluctuations, input contraction, prediction quality and augmented training criticality are tested separately. An externally imposed schedule is not evidence of endogenous critical attraction.',
        known_information='All completed constant-rate outcomes, six verified early repeated-corpus risk states and the two repeated-corpus paths through their saved update-65536 states were known. Both repeated-corpus workers were stopped at the explicit user request. Their later unexecuted horizons are not outcomes. No new GPU training outcome is known at this selection.',
        repetition_assessment=dict(purpose='Assess which conclusions from repeated-corpus training require fresh-data replication before reuse.',
            constant_comparison='Four paired initializations at head counts 4, 8 and 14 through 32768 updates; two of these initializations at head counts 4 and 14 continue through 131072 updates. Data law changes, while initial parameters, optimizer, constant rates and held-out observation cohorts match the completed constant-rate family.',
            scheduled_comparison='Reference1 and subcritical1 at head count 14, initialization 640101, through 65536 updates with the original 250000-step schedule. Compare the same saved times against retained repeated-corpus checkpoints.',
            outcomes=['frozen seen-training and held-out risks','row-contrast and common-sector means','whole-initialization covariance and finite-width susceptibility','collapse onset and persistence','predictive margins and operator stability'],
            screening='At matched time and size, flag a mean held-out NLL change exceeding 0.1 nat, a mean absolute log10 row-energy change exceeding 0.5, a susceptibility change exceeding 25 percent when its denominator exceeds 1e-10, or a reversal of late-loss or persistent-collapse behavior. These are replication screens, not criticality tests or equivalence margins. All paired differences and whole-initialization uncertainty are retained.',
            decisions='Results on pretrained fixed checkpoints and algebraic reconstruction identities keep their original scope. Any empirical training coefficient, scaling relation or regime-selection conclusion affected by the screens requires a focused single-pass replication of its supporting family, or must be restricted to the repeated-data law and withheld as a claim about single-pass pretraining. Additional replications are frozen before execution. Lack of a screen does not prove equality of the data laws.'),
        comparison_scope='Fresh versus repeated data compares a larger source corpus and a nonoverlapping single-pass stream against the existing small-corpus resampling law. Initialization, optimizer, drive and evaluation are matched. Exact isolation of corpus size from sampling order is not claimed. The data-schedule paths end deliberately at 65536, matching the available repeated-corpus checkpoint; their 250000-step schedules have not completed. Only the compact paths test a complete cosine and floor interval.')
    scientific = dict(common, jobs=jobs, scientific_trajectories=54, selected_updates=2555904,
        condition_inventory=dict(compact_paths=40, matched_schedule_data_paths=2,
                                 matched_constant_data_paths=12),
        primary_widths=[4,8,14], compact_initializations=[640101,640102,640103,640104])
    qualification = dict(common, jobs=qa, native_updates=6144,
        qualification_phases=[[j['run_id'] for j in qa[:5]], [qa[5]['run_id']], [qa[6]['run_id']]],
        comparison=dict(continuous=continuous,resumed=qa[6]['run_id']),
        role='GPU execution, resource and full-state recovery qualification; separate from scientific trajectories.')
    for name in ['runs','logs','measurements','verification','protocols/executions']:
        (study/name).mkdir(parents=True, exist_ok=True)
    write_json(selection,scientific)
    qa_path = study/'protocols/onepass-qualification-selection.json'
    write_json(qa_path,qualification)
    passive = json.loads((old/'protocols/regime-shared-update-observation.json').read_text())
    passive.update(frozen_at=now,run_ids=[j['run_id'] for j in jobs+qa],
        scheduled_selection_sha256={str(path):sha256(path) for path in [selection,qa_path]})
    write_json(study/'protocols/onepass-shared-update-observation.json',passive)

    # Every training-risk cohort samples only blocks already consumed at its time.
    population = np.random.default_rng(641001).permutation(4194304)
    tokens = np.load(root/'data/refinedweb-onepass-524288/tokens.npy',mmap_mode='r')
    cohorts = {}
    for t in sorted({int(t) for j in jobs for t in j['save_steps'].split(',')}):
        folder = study/f'data/seen-training-risk/t{t}'
        folder.mkdir(parents=True,exist_ok=False)
        positions = np.random.default_rng(651581).choice(32*t,size=1024,replace=False)
        blocks = population[positions]
        rows, offsets = blocks//8, 64*(blocks%8)
        crops = tokens[rows[:,None],offsets[:,None]+np.arange(65)]
        np.savez_compressed(folder/'cohort.npz',rows=rows,offsets=offsets,crops=crops,
                            consumed_stream_positions=positions)
        write_json(folder/'manifest.json',dict(schema='onepass-seen-risk-cohort-v1',status='complete',
            frozen_at=now,step=t,source=str(root/'data/refinedweb-onepass-524288/tokens.npy'),
            source_sha256=json.loads((root/'data/refinedweb-onepass-524288/manifest.json').read_text())['tokens_sha256'],
            stream_seed=640001,selection_seed=651581,contexts=1024,
            cohort_sha256=sha256(folder/'cohort.npz'),
            scope='Uniform without-replacement sample of blocks consumed by this checkpoint. The cohort is fixed before training and changes across horizons; it is not an unseen draw from the full future corpus.'))
        cohorts[t] = dict(training_cohort=str(folder/'cohort.npz'),
                          training_cohort_sha256=sha256(folder/'cohort.npz'),
                          training_cohort_manifest=str(folder/'manifest.json'))
    observation = json.loads((old/'protocols/regime-observation-selection.json').read_text())
    cases = []
    for j in jobs:
        for t in map(int,j['save_steps'].split(',')):
            filename = 'final-training-state.pt' if t==j['steps'] else f'training-state-{t}.pt'
            cases.append(dict(name=f'cpu-{j["run_id"]}-t{t}',run_id=j['run_id'],recipe=j['recipe'],
                role=j['role'],schedule_horizon=j['schedule_horizon'],heads=j['heads'],seed=j['seed'],step=t,
                shared_seed=j['shared_seed'],stream_seed=j['stream_seed'],
                state=str(study/'runs'/j['run_id']/filename),
                parent_manifest=str(study/'runs'/j['run_id']/'manifest.json'),
                inference_stability=(t in [32768,40960] if j['role']=='compact' else t==j['steps']),
                **cohorts[t]))
    for key in ['training_cohort','training_cohort_sha256','training_cohort_manifest']:
        observation.pop(key)
    observation.update(schema='onepass-observation-selection-v1',frozen_at=now,cases=cases,
        training_selection=str(selection),training_selection_sha256=sha256(selection),
        cpu_states=len(cases),risk_states=len(cases),
        inference_states=sum(c['inference_stability'] for c in cases),
        training_risk_scope='Each case binds its own frozen cohort of already consumed training blocks.')
    write_json(study/'protocols/onepass-observation-selection.json',observation)
    print(json.dumps(dict(trajectories=54,updates=2555904,qualification_updates=6144,
        cpu_states=len(cases),inference_states=observation['inference_states'],
        selection_sha256=sha256(selection))),flush=True)


if __name__ == '__main__':
    main()
