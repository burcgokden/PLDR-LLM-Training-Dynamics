#!/usr/bin/env python
"""Bind the established prefix and task interventions to all new selected states."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from model_rg.provenance import sha256, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-feasible-20260908');a=p.parse_args()
    root=Path(a.root).resolve();study=root/a.study;old=root/'scheduled-training-20260908'
    observation=study/'protocols/onepass-observation-selection.json'
    selected=json.loads(observation.read_text())
    for f in (study/'runs').glob('*/manifest.json'):
        if not f.parent.name.startswith('qualification-'):
            raise AssertionError('Freeze the mechanism selection before scientific training outcomes')
    now=datetime.now(timezone.utc).isoformat()
    for filename,endpoint_only,label in [('scheduled-prefix-selection.json',False,'prefix'),
                                          ('reasoning-mechanism-selection.json',True,'reasoning')]:
        path=study/'protocols'/filename
        if path.exists():raise FileExistsError(path)
        origin=old/'protocols'/filename
        spec=json.loads(origin.read_text())
        cases=[]
        for c in selected['cases']:
            if endpoint_only and not c['inference_stability']:continue
            case={k:v for k,v in c.items() if not k.startswith('training_cohort')}
            case.update(name=label+'-'+c['name'].removeprefix('cpu-'),kind='trained',
                        source=str(root/'assets/PLDR-LLM-v51-SOC-110M-1'))
            cases.append(case)
        inputs={k:v for k,v in spec['inputs_sha256'].items()
                if not k.endswith('regime-observation-selection.json')}
        inputs.update({str(observation):sha256(observation),str(origin):sha256(origin)})
        spec.update(frozen_at=now,cases=cases,inputs_sha256=inputs,
            scope=f'All {len(cases)} selected single-pass training states under the fixed intervention and task-cohort laws. Every numerical outcome remains in the evidence.',
            prior_information='The initial five-state operator and task measurements and the completed prompt study are known. The full released-model ARC evaluation is separate. These cohorts and interventions are retained unchanged. Only execution-qualification models have been observed from the new data stream.',
            statistical_units='Complete matched initializations are the training units. Main compact conditions have four initializations. Data-law controls have four initializations per width at update 32768, two per width at late constant-rate times, and one per original reference-schedule condition. Question and counterfactual-pair uncertainty does not replace training replication.')
        write_json(path,spec)
        print(filename,len(cases),sha256(path),flush=True)


if __name__=='__main__':main()
