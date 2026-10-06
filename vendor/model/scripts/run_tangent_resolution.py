#!/usr/bin/env python
"""Refine unresolved local windows using a recorded outcome-dependent rule."""
from companion_paths import required_input
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from model_rg.provenance import sha256, write_json


def joint_errors(result):
    return [max([r['maximum_predictive_relative_error'],*r['augmented_relative_errors']])
            if all(x is not None for x in r['augmented_relative_errors']) else float('inf') for r in result['records']]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    args=parser.parse_args();study=Path(args.root)/'criticality-dynamics-20260906';repo=Path(__file__).resolve().parents[1]
    rule=study/'protocols/tangent-resolution-rule.json';spec=json.loads(rule.read_text())
    while True:
        ledger=study/'tangent-replication.json'
        if ledger.exists() and json.loads(ledger.read_text())['status']!='running':break
        time.sleep(15)
    original=json.loads((study/'protocols/tangent-replication.json').read_text());cases=[];inputs={}
    for case in original['cases']:
        result_path=study/'analysis'/case['name']/'results.json'
        result=json.loads(result_path.read_text());scores=joint_errors(result)
        inputs[str(result_path)]=sha256(result_path)
        if any(max(a,b)<=spec['relative_threshold'] for a,b in zip(scores,scores[1:])):continue
        cases.append(dict(**case,output_name=case['name']+'-resolution'))
    selected=dict(schema='tangent-resolution-selection-v1',rule_sha256=sha256(rule),input_results=inputs,cases=cases)
    for path in [study/'protocols/tangent-resolution-selected.json',Path(required_input('dynamics-protocols')) / 'tangent-resolution-selected.json']:
        if path.exists():raise FileExistsError(path)
        write_json(path,selected)
    records=[]
    for case in cases:
        command=[sys.executable,'scripts/measure_adam_tangent.py','--root',args.root,'--checkpoint',case['checkpoint'],
                 '--output',str(study/'analysis'/case['output_name']),'--direction',case['direction'],
                 '--amplitudes',spec['amplitudes'],'--contexts',str(case['contexts'])]
        with (study/'logs'/(case['output_name']+'.log')).open('x') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        records.append(dict(name=case['output_name'],command=command,returncode=result.returncode));print(records[-1],flush=True)
        write_json(study/'tangent-resolution.json',dict(status='running',records=records))
    write_json(study/'tangent-resolution.json',dict(status='complete' if all(r['returncode']==0 for r in records) else 'complete_with_failed_controls',records=records,
        selection_sha256=sha256(study/'protocols/tangent-resolution-selected.json')))


if __name__=='__main__':main()
