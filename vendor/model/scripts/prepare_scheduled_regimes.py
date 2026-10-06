#!/usr/bin/env python
"""Freeze the expanded near/subcritical comparison before scheduled GPU work."""
from companion_paths import required_input
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np

from model_rg.provenance import sha256,write_json
from model_rg.scheduled_regimes import RECIPE_NAMES,recipe


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-20260908');a=p.parse_args()
    root=Path(a.root).resolve();study=root/a.study;repo=Path(__file__).resolve().parents[1]
    if list((study/'runs').iterdir()):raise AssertionError('The expanded selection must precede all scheduled GPU artifacts')
    original=study/'protocols/training-selection.json';old=json.loads(original.read_text())
    old_qa=study/'protocols/qualification-selection.json';qa=json.loads(old_qa.read_text())
    qualification=study/'qualification/reference-regimes';qm=json.loads((qualification/'manifest.json').read_text())
    if qm['status']!='complete':raise AssertionError('The expanded reference qualification is required')
    qb=json.loads((qualification/'binding.json').read_text())
    for name in ['src/model_rg/schedules.py','src/model_rg/scheduled_regimes.py']:
        if qb['source_files'][name]!=sha256(repo/name):raise AssertionError('A qualified recipe implementation changed')
    table=Path(required_input('reference-schedule')) / 'regime-table.json';ts=json.loads(table.read_text())
    profiles={name:{str(n):recipe(name,n) for n in [4,14]} for name in RECIPE_NAMES}
    reference_inputs=dict(old['reference_inputs'])
    for path in [original,old_qa,qualification/'manifest.json',qualification/'results.json',table,Path(ts['source'])]:
        reference_inputs[str(path)]=sha256(path)
    producers=dict(old['producer_sources'])
    for name in ['scripts/train_scheduled_regimes.py','src/model_rg/scheduled_regimes.py']:
        producers[name]=sha256(repo/name)
    now=datetime.now(timezone.utc).isoformat()
    common=dict(recipes=profiles,reference_inputs=reference_inputs,producer_sources=producers,
        training_entrypoint='scripts/train_scheduled_regimes.py',frozen_at=now,
        reference_classifications={r['recipe']:r['reference_classification'] for r in ts['profiles']},
        extension_reason='Explicit comparison of the near-critical and subcritical settings in Table1, requested before any scheduled GPU execution. Labels describe the reference paper, and are not assigned to these outcomes in advance.',
        source_selections={str(original):sha256(original),str(old_qa):sha256(old_qa)})
    existing={j['run_id']:j for j in old['jobs']};jobs=[]
    for seed in range(640101,640105):
        for name in RECIPE_NAMES:
            for n in [4,14]:
                run=f'{name}-h{n}-s{seed}'
                if run in existing:j=dict(existing[run])
                else:
                    j=dict(existing[f'reference1-h{n}-s{seed}'])
                    j.update(run_id=run,recipe=name,
                        save_steps=','.join(map(str,[profiles[name][str(n)]['warmup_steps'],8192,32768,65536,131072,196608,250000,262144])))
                jobs.append(j)
    selected=dict(old,**common);selected.update(jobs=jobs,scientific_trajectories=40,selected_updates=40*262144,
        role='Forty paired scientific trajectories: one schedule-only control and four paper-specified near/subcritical recipes, two head counts, four initializations.',
        reference_order_parameter=dict(tensors=['A','A_LM','A_P','G_LM'],
            definition='Whole-model RMSE between matched terminal deductive tensors from two independent stochastic continuations, divided by the magnitude of the specified global mean. Compare terminal tensors with prompt tensors as the paper cached-reference statistic. Retain absolute RMSE and RMS-normalized error separately; report undefined mean-normalized ratios when the denominator is zero.',
            interpretation='Input/generation stability is tested separately from row concentration, seed susceptibility, a thermodynamic critical surface, and endogenous feedback toward criticality.'))
    qualified=dict(qa,**common)
    paths=[study/'protocols'/name for name in ['regime-training-selection.json','regime-qualification-selection.json',
             'regime-shared-update-observation.json','regime-observation-selection.json']]
    if any(p.exists() for p in paths):raise FileExistsError('An expanded frozen selection already exists')
    write_json(paths[0],selected);write_json(paths[1],qualified)
    passive=json.loads((study/'protocols/shared-update-observation.json').read_text())
    passive.update(frozen_at=now,run_ids=[j['run_id'] for j in jobs+qa['jobs']],
        scheduled_selection_sha256={str(path):sha256(path) for path in paths[:2]})
    write_json(paths[2],passive)
    risk=root/'critical-scaling-20260906/data/training-risk-1024/cohort.npz'
    risk_manifest=risk.parent/'manifest.json'
    cases=[]
    for j in jobs:
        for step in map(int,j['save_steps'].split(',')):
            filename='final-training-state.pt' if step==j['steps'] else f'training-state-{step}.pt'
            cases.append(dict(name=f'cpu-{j["run_id"]}-t{step}',run_id=j['run_id'],recipe=j['recipe'],
                heads=j['heads'],seed=j['seed'],step=step,shared_seed=j['shared_seed'],stream_seed=j['stream_seed'],
                state=str(study/'runs'/j['run_id']/filename),parent_manifest=str(study/'runs'/j['run_id']/'manifest.json'),
                inference_stability=step in [250000,262144]))
    data=study/'data/reference-inference';data.mkdir(parents=True,exist_ok=False)
    uniforms=np.random.default_rng(650701).random((2,100,32))
    np.savez_compressed(data/'cohort.npz',rows=np.arange(512,612),uniforms=uniforms)
    write_json(data/'manifest.json',dict(schema='scheduled-inference-cohort-v1',frozen_at=now,
        source=str(root/'controlled-study-20260905/data/short'),rows=[512,612],prompt_tokens=64,
        independent_continuations=2,maximum_new_tokens=32,uniform_seed=650701,nucleus_probability=.8,temperature=1.,
        eos_token_id=3,stopping='Stop after EOS token3 or32 newly generated tokens; recompute terminal deductive tensors including the final token.',
        emission_batch_size=1,generation_batch_size=4,
        comparison='Terminal run1/run2 and each terminal run/prompt. Prompt tensors supply the cached-reference values for the order parameter; no KV/G-cached text generation is claimed.',
        adaptation='One hundred fixed RefinedWeb prompts of64 tokens and up to32 generated tokens, in place of the paper IMDB prompts and up to256 generated tokens. Scalar sampling uniforms are fixed across recipes, sizes and training identities.',
        cohort_sha256=sha256(data/'cohort.npz')))
    observation=dict(schema='scheduled-regime-observation-selection-v1',frozen_at=now,cases=cases,
        training_selection=str(paths[0]),training_selection_sha256=sha256(paths[0]),
        cpu_states=len(cases),risk_states=len(cases),inference_states=sum(c['inference_stability'] for c in cases),
        rows=list(range(512,1024)),precisions=['float32','float64'],batch_size=32,evaluation_matrices=True,
        training_cohort=str(risk),training_cohort_sha256=sha256(risk),training_cohort_manifest=str(risk_manifest),
        risk=dict(precision='float32',batch_size=32,training_contexts=1024,heldout_contexts=512,
            objectives=['last_external_target','all_nonpadding_targets'],online_window=8192,
            raw_logits=dict(last_target_contexts=32,all_target_contexts=2),
            interpretation='Frozen-state risk is separate from the mean online loss of an evolving trajectory. All-token risks retain target masks and both batch-mean and token-weighted denominators.'),
        inference_cohort=str(data/'cohort.npz'),inference_cohort_manifest=str(data/'manifest.json'),
        inference_cohort_sha256=sha256(data/'cohort.npz'),
        source_classification='Reference labels are Table1 recipe labels; every scientific conclusion is conditioned on the executed finite training/data/precision program.')
    write_json(paths[3],observation)
    print(json.dumps(dict(trajectories=40,native_updates=40*262144,cpu_states=len(cases),risk_states=len(cases),
        inference_states=sum(c['inference_stability'] for c in cases),protocols={str(p):sha256(p) for p in paths})),flush=True)


if __name__=='__main__':main()
