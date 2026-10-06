#!/usr/bin/env python
"""Freeze paired scheduled trajectories and qualifications before their execution."""
from companion_paths import required_input
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from model_rg.provenance import sha256, write_json
from model_rg.schedules import recipe


PRODUCERS=['scripts/train_scheduled_scaling.py','src/model_rg/schedules.py',
           'src/model_rg/training.py','src/model_rg/native.py','src/model_rg/scaling.py',
           'src/model_rg/criticality.py','src/model_rg/controlled.py','src/model_rg/provenance.py',
           'src/model_rg/variance_family.py','src/model_rg/checkpointing.py','src/model_rg/update_records.py']


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-20260908');a=p.parse_args()
    root=Path(a.root).resolve();study=root/a.study;repo=Path(__file__).resolve().parents[1]
    constant=root/'critical-scaling-20260906';qualification=study/'qualification/reference-cpu-final'
    q=json.loads((qualification/'manifest.json').read_text())
    if q['status']!='complete':raise AssertionError('Reference CPU qualification is required')
    for name,key in [('results.json','results_sha256'),('measurements.npz','raw_sha256'),('binding.json','binding_sha256')]:
        if sha256(qualification/name)!=q[key]:raise AssertionError('The completed qualification changed')
    qb=json.loads((qualification/'binding.json').read_text())
    if qb['source_files']['src/model_rg/schedules.py']!=sha256(repo/'src/model_rg/schedules.py'):
        raise AssertionError('The qualified schedule implementation changed')
    for name in ['protocols','logs','runs','measurements','analysis','verification']:
        (study/name).mkdir(parents=True,exist_ok=True)
    outputs=[study/'protocols'/name for name in ['training-selection.json','qualification-selection.json','shared-update-observation.json']]
    if any(p.exists() for p in outputs):raise FileExistsError('An existing frozen scheduled selection must not be replaced')
    reference=list((Path(required_input('reference-schedule'))).iterdir())
    reference+=[qualification/'manifest.json',qualification/'results.json']
    profiles={name:{str(n):recipe(name,n) for n in [4,14]} for name in ['controlled','reference1','reference2']}
    common=dict(schema='scheduled-training-selection-v1',frozen_at=datetime.now(timezone.utc).isoformat(),
        reference_inputs={str(p):sha256(p) for p in reference if p.is_file()},recipes=profiles,
        producer_sources={name:sha256(repo/name) for name in PRODUCERS},
        adaptation='Five native PLDR decoders, head dimension64, eight shared metric residual units with width170; variance-normalized controlled initialization and a fixed shared initial state; float32 on one GPU per trajectory, no TF32, full batch32, context64. The finite training law samples3072 cached RefinedWeb documents and449 crop offsets. Reference branches reproduce the released schedule, Adam epsilon/betas/decay, common parameter rate, elementwise clipping, and all-nonpadding-target loss. They do not reproduce original context1024, BF16/FSDP, initialization law, or approximately8B-token pretraining corpus exposure.',
        schedule=dict(total_steps=250000,after_schedule='Hold the released nonzero floor through262144 updates.',
            indexing='Update k+1 applies s(k); the initial update has zero parameter rate and updates Adam state. Scheduler.step follows optimizer.step.',
            maximum_completed_horizon=262144),
        inference_observation='The same512 held-out contexts,64 matrix-calibration contexts and16 dense contexts as the constant-rate family. Evaluation uses one external next-token target after64 inputs in every recipe.',
        training_risk='A separately frozen1024-crop training-law cohort, identical to the constant-rate risk diagnostic, will be measured at all selected saved states. Both last-target and all-nonpadding-target risks will be retained, with exact valid-target denominators.',
        limitations='This paired two-size study tests schedule and recipe sensitivity. It does not identify an asymptotic head-count exponent, a native critical surface, or endogenous critical attraction from an externally imposed annealing curve. Seeds condition on one shared initialization and one full batch history.',
        retention='All selected outcomes, numerical failures, censored passages and prediction errors remain in the internal scientific record. No completed outcome may be omitted to improve agreement.')
    def job(name,n,seed,steps,run_id=None,stream=640001):
        warmup=profiles[name][str(n)]['warmup_steps']
        return dict(run_id=run_id or f'{name}-h{n}-s{seed}',heads=n,seed=seed,recipe=name,steps=steps,
            shared_seed=640011,stream_seed=stream,probe_every=64,microbatch=32,checkpoint_decoders=False,
            save_steps=','.join(str(t) for t in [warmup,8192,32768,65536,131072,196608,250000,262144] if t<=steps))
    jobs=[job(name,n,seed,262144) for seed in range(640101,640105)
          for name in ['controlled','reference1','reference2'] for n in [4,14]]
    selected={}
    for label in ['main-size-time-execution','clock-holdout','diffusive-size-map','environment-factorial','zero-horizon','extended-horizon']:
        s=json.loads((constant/'protocols'/f'{label}.json').read_text())
        for j in s['jobs']:selected[j['run_id']]=str(constant/'runs'/j['run_id']/'manifest.json')
    selected['long-h4-g1-s640101']=str(constant/'runs/long-h4-g1-s640101/manifest.json')
    if len(selected)!=96:raise AssertionError('The complete constant-rate prerequisite inventory changed')
    known={name:sha256(path) for name,path in selected.items() if Path(path).exists()}
    scientific=dict(common,jobs=jobs,role='Twenty-four paired scientific trajectories from initialization.',
        scientific_trajectories=24,selected_updates=24*262144,
        constant_prerequisites=selected,constant_manifests_known_at_selection=known,
        hypotheses=['Transfer of constant-rate row-passage clocks through the cumulative first- and second-order schedule clocks.',
            'Persistence or change of row, common-row, contrast, prediction and risk sectors across schedule phase and optimizer/loss recipes.',
            'Whether phase-conditioned temporal fluctuations remain after the externally forced annealing interval ends.'],
        planned_analysis=dict(seed_unit='Four complete paired trajectories per recipe and head count.',
            primary_passage_threshold=.1,sensitivity_thresholds=[.3,.01,.001],
            row_passage_rule='First sampled descent below threshold, with64-update mesh and explicit censoring; recurrence is recorded separately.',
            temporal_block_updates=[2048,8192],floor_interval=[250000,262144],
            empirical_resampling='All256 ordered four-seed draws; no context is treated as an independent training initialization.'),
        prior_information='The constant-rate study and its first completed long-horizon risk observations were available when this design was selected. No scheduled GPU trajectory was inspected before selection.')
    benchmark=[]
    for name in ['controlled','reference1']:
        for n,device in [(4,0),(14,1)]:
            j=job(name,n,650631,1024,f'qualification-{name}-h{n}-continuous',650641)
            j['device']=device;benchmark.append(j)
    split=job('reference1',14,650631,512,'qualification-reference1-h14-prefix',650641);split['device']=1
    resumed=job('reference1',14,650631,1024,'qualification-reference1-h14-resumed',650641)
    resumed.update(device=1,resume_from_run=split['run_id'],resume_from_step=512)
    qualification_spec=dict(common,role='Execution qualification and resource measurements, separate from scientific trajectories.',
        jobs=benchmark+[split,resumed],constant_prerequisites=selected,
        native_updates=5120,qualification_phases=[[j['run_id'] for j in benchmark],[split['run_id']],[resumed['run_id']]],
        comparison=dict(continuous='qualification-reference1-h14-continuous',resumed=resumed['run_id'],step=1024,
            rule='Parameters, Adam tensors, scheduler state, target counts and common endpoint emissions must match exactly on GPU1.'))
    passive=json.loads((constant/'protocols/shared-update-observation.json').read_text())
    passive.update(frozen_at=common['frozen_at'],run_ids=[j['run_id'] for j in jobs+benchmark+[split,resumed]],
        role='Passive scheduled shared-parameter observations, retained at every update with its applied drive phase.')
    passive.pop('training_protocol_sha256',None)
    write_json(outputs[0],scientific);write_json(outputs[1],qualification_spec)
    passive['scheduled_selection_sha256']={str(p):sha256(p) for p in outputs[:2]}
    write_json(outputs[2],passive)
    print(json.dumps(dict(scientific_trajectories=24,selected_updates=24*262144,
        maximum_horizon=262144,constant_artifacts_known=len(known),protocols={str(p):sha256(p) for p in outputs})),flush=True)


if __name__=='__main__':main()
