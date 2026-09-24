#!/usr/bin/env python
"""Select bounded native-step tests of loss drift under the remaining-data law."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

from model_rg.provenance import sha256,write_json


def bind_case(case):
    result=dict(case);parent=Path(case['parent']);proof_path=Path(case['training_verification'])
    proof=json.loads(proof_path.read_text());meta=json.loads(Path(case['parent_manifest']).read_text())
    if proof['status']!='complete' or meta['status']!='complete':raise AssertionError('A complete reconstructed parent is required')
    result['parent_sha256']=sha256(parent)
    if proof['verified_files'][str(parent)]!=result['parent_sha256']:
        raise AssertionError('The selected saved state changed')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-feasible-20260908')
    p.add_argument('--stage',choices=['qualification','science'],required=True);a=p.parse_args()
    root=Path(a.root).resolve();mainstudy=root/a.study;repo=Path(__file__).resolve().parents[1]
    study=mainstudy/'conditional-step-qualification' if a.stage=='qualification' else mainstudy
    path=study/'protocols/onepass-native-step-selection.json'
    if path.exists():raise FileExistsError(path)
    selection_path=mainstudy/'protocols/onepass-training-selection.json';selection=json.loads(selection_path.read_text())
    jobs={j['run_id']:j for j in selection['jobs']}
    planned=[('data_constant-controlled-h4-s640101',98304),('data_constant-controlled-h4-s640101',65536),
        ('data_schedule-reference1-h14-s640101',65536),('data_schedule-subcritical1-h14-s640101',65536),
        ('data_constant-controlled-h14-s640101',98304),('data_constant-controlled-h14-s640101',65536)]
    if a.stage=='qualification':planned=[planned[1],planned[2]]
    sources=['scripts/measure_onepass_native_step.py','scripts/verify_onepass_native_step.py',
        'scripts/prepare_onepass_native_steps.py','scripts/run_onepass_native_steps.py',
        'src/model_rg/onepass_regimes.py','src/model_rg/schedules.py','src/model_rg/criticality.py',
        'src/model_rg/training.py','src/model_rg/native.py','src/model_rg/controlled.py','src/model_rg/provenance.py']
    inputs=[selection_path,root/'data/refinedweb-onepass-524288/manifest.json',
        mainstudy/'verification/onepass-data.json',mainstudy/'verification/onepass-cohort-separation.json']
    if a.stage=='science':
        qualification=mainstudy/'conditional-step-qualification/launcher-onepass-native-steps.json'
        q=json.loads(qualification.read_text())
        if q['status']!='complete' or len(q['records'])!=2:raise AssertionError('Both native objective/clipping qualifications must pass')
        inputs.append(qualification)
        for row in q['records']:
            proof_path=Path(row['verification']);proof=json.loads(proof_path.read_text())
            if proof['status']!='passed' or proof['conditional_native_steps']!=2:raise AssertionError('Incomplete native-step qualification')
            if sha256(proof_path)!=row['verification_sha256']:raise AssertionError('Qualification changed')
            inputs.append(proof_path)
    cases=[]
    for run_id,step in planned:
        job=jobs[run_id];folder=mainstudy/'runs'/run_id
        case=dict(name='onepass-native-step-'+('qa-' if a.stage=='qualification' else '')+run_id+'-t'+str(step),
            run_id=run_id,heads=job['heads'],seed=job['seed'],step=step,recipe=job['recipe'],role=a.stage,
            parent=str(folder/('final-training-state.pt' if step==job['steps'] else f'training-state-{step}.pt')),
            parent_manifest=str(folder/'manifest.json'),training_verification=str(mainstudy/'verification/training'/(run_id+'.json')),
            profile=job['profile'])
        if a.stage=='qualification':case=bind_case(case)
        cases.append(case)
    write_json(path,dict(schema='onepass-native-step-selection-v1',frozen_at=datetime.now(timezone.utc).isoformat(),
        cases=cases,stage=a.stage,batch_replicas=2 if a.stage=='qualification' else 32,
        batch_seed=970031 if a.stage=='qualification' else 970071,threads=4,workers=1,
        producer_sources={name:sha256(repo/name) for name in sources},inputs_sha256={str(p):sha256(p) for p in inputs},
        conditional_law='Condition on the complete incoming model, Adam moments, schedule phase and consumed source-block set. '
        'For each independent conditional branch, draw32distinctblocks uniformly from the unconsumed population. '
        'No block consumed by the incoming path is eligible. Each branch restores the same state before one update; '
        'reusing a population across independent counterfactual branches does not create a repeated training trajectory. '
        'This averages unrevealed future data and is separate from the primary initialization law conditioned on the full stream.',
        prior_information='Source-recipe data comparisons, local input Jacobians and decoder interventions were observed. '
        'The complete first N4constant single-pass path retained nearly the same late held-out loss rise as its small-corpus counterpart. '
        'These states test conditional native loss drift and its coupled generator/body contributions, not a blind criticality forecast.',
        observation_law='All512fixedheldoutcontexts[512,1024), evaluated in nativeCPUfloat32batchesof32. '
        'Each conditional update uses the actual saved objective, clipping, Adam state and next scheduled rate. '
        'The full, generator-only and body-only weight corners share that same realized complete optimizer displacement. '
        'Every optimizer update is independently replayed; first/last emissions and every risk/KL reduction have separate checks.',
        resource_scope='One CPU worker with four threads; no additional GPU trajectory and no new training initialization. '
        'The scientific panel contains192one-stepconditionaldraws. Full native vocabulary logits and complete step digests are retained.'))
    print(path,sha256(path),flush=True)


if __name__=='__main__':main()
