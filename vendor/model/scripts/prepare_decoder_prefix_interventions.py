#!/usr/bin/env python
"""Select all single-decoder and joint operator interventions for the observed prefix defect."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from model_rg.provenance import sha256, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-feasible-20260908');a=p.parse_args()
    root=Path(a.root).resolve();study=root/a.study;repo=Path(__file__).resolve().parents[1]
    out=study/'protocols/decoder-prefix-intervention-selection.json'
    if out.exists():raise FileExistsError(out)
    base_path=study/'protocols/prefix-risk-comparison.json';base=json.loads(base_path.read_text())
    jac_path=study/'verification/onepass-row-jacobians.json';jac=json.loads(jac_path.read_text())
    if jac['status']!='passed' or jac['jacobians']!=1440:raise AssertionError('The complete input-domain derivative result is required')
    inputs={str(p):sha256(p) for p in [base_path,jac_path]};inputs.update(base['inputs_sha256'])
    definitions=json.loads((study/'protocols/reasoning-mechanism-selection.json').read_text())
    cases=[];baselines=[]
    variants=[(f'fixed_decoder_{i+1}','fixed',[i]) for i in range(5)]
    variants += [('fixed_all','fixed',list(range(5))),('row_decoder_4','row_projection',[3]),('row_all','row_projection',list(range(5)))]
    for old in base['cases']:
        if not old['name'].startswith('causal-data_schedule-') or old['step']!=65536:continue
        folder=study/'measurements/prefix-risk'/old['name'];proof_path=study/'verification/prefix-risk'/(old['name']+'.json')
        proof=json.loads(proof_path.read_text())
        if proof['status']!='passed' or proof['case']!=old:raise AssertionError('A native baseline lacks complete causal-risk verification')
        for name in ['manifest.json','results.json','measurements.npz','full-logits.npy','proper-logits.npy']:
            p=folder/name;digest=sha256(p)
            if proof['checked_sha256'][str(p)]!=digest:raise AssertionError('A verified baseline changed')
            inputs[str(p)]=digest
        inputs[str(proof_path)]=sha256(proof_path)
        tasks=[c for c in definitions['cases'] if c['run_id']==old['run_id'] and c['step']==old['step']]
        if len(tasks)!=1:raise AssertionError('A unique completed calibration law is required')
        task=tasks[0];calibration=study/'measurements'/task['name']/'measurements.npz'
        taskproof=study/'verification/reasoning'/(task['name']+'.json');verified=json.loads(taskproof.read_text())
        if verified['status']!='passed' or verified['checked_sha256'][str(calibration)]!=sha256(calibration):
            raise AssertionError('The retained native calibration cache requires independent reconstruction')
        inputs[str(calibration)]=sha256(calibration);inputs[str(taskproof)]=sha256(taskproof)
        baselines.append(dict(case=old,folder=str(folder),verification=str(proof_path)))
        for mode,operation,layers in variants:
            cases.append(dict(old,name='decoder-prefix-'+old['name'].removeprefix('causal-')+'-'+mode,
                operation=operation,layers=layers,mode=mode,calibration_raw=str(calibration),
                baseline_full_logits=str(folder/'full-logits.npy'),baseline_folder=str(folder)))
    if len(cases)!=16 or len(baselines)!=2:raise AssertionError('Both source endpoints and all eight interventions are required')
    sources=['scripts/measure_decoder_prefix_intervention.py','scripts/verify_decoder_prefix_intervention.py',
        'scripts/prepare_decoder_prefix_interventions.py','scripts/run_decoder_prefix_interventions.py',
        'src/model_rg/layer_interventions.py','src/model_rg/inference_interventions.py',
        'src/model_rg/native.py','src/model_rg/training.py','src/model_rg/controlled.py','src/model_rg/provenance.py']
    write_json(out,dict(schema='decoder-prefix-intervention-selection-v1',frozen_at=datetime.now(timezone.utc).isoformat(),
        cases=cases,baselines=baselines,rows=list(range(512,544)),prefix_lengths=list(range(1,65)),threads=4,workers=2,
        source=base['source'],inputs_sha256=inputs,producer_sources={n:sha256(repo/n) for n in sources},
        prior_information='The source endpoint losses, row geometry, task margins and all1440 prefix-domain Jacobians were observed before this intervention selection. Decoder4 of the single-pass Below endpoint retains expansive local row derivatives and a relatively large finite prefix-separation factor. All five individual decoder replacements, the joint replacement and both targeted/joint row projections are selected, retaining the Near endpoint as a matched mechanism control.',
        intervention_law='Fixed operators are the existing64-context calibration mean computed in float64 then cast to nativefloat32, exactly as in the completed task intervention. A selected decoder uses the native external-operator branch; every unselected decoder recomputes its own native operator. Row projection acts only on the declared metric tensors before the native PLGA map. KV caching is disabled and weights/optimizer state are unchanged.',
        qualification='Each native full-window output must reproduce its retained baseline bytewise. Every fixed-subset intervention must replay the native baseline when given its own context operators, and every intervention must restore the baseline bytewise. Selecting all decoders must reproduce the already qualified global helper bytewise. Full control-logit arrays and the actual calibration tensors are retained for independent comparison.',
        scope='Finite causal operator interventions on both completed single-pass source-recipe endpoints,32fixedheldcontexts and all64properprefixes. These assess changes in the measured full-to-proper predictive defect. A useful intervention does not establish a unique mechanism, population reasoning competence, a thermodynamic exponent or an endogenous critical transition. The selection is a declared follow-up to the input-domain observations.',
        resource_scope='Sixteen CPU observations, two already verified native baselines, no new training updates or independent training identities. Every planned observation and adverse intervention result is retained.'))
    print('Selected16decoder interventions with2verifiednativebaselines',sha256(out),flush=True)


if __name__=='__main__':main()
