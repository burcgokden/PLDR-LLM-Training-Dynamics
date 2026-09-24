#!/usr/bin/env python
"""Resolve the coarse transition bracket after the causal controls release GPUs."""
from companion_paths import child_pythonpath
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from model_rg.provenance import write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    root=Path(args.root);study=root/'criticality-study-20260905';repo=Path(__file__).resolve().parents[1]
    ledger=study/'launcher-fine-study.json'
    if ledger.exists():raise FileExistsError(ledger)
    write_json(ledger,dict(status='waiting_for_restoration'))
    while True:
        status=json.loads((study/'launcher-restoration.json').read_text())['status']
        if status=='execution_failure':raise RuntimeError('A prerequisite causal control failed')
        if status=='complete':break
        time.sleep(15)
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4')
    commands=[
        [sys.executable,str(repo/'scripts/run_criticality.py'),'--root',str(root),'--label','fine',
         '--heads','2,4,8,14','--multipliers','1.25,1.5,2','--normalization','variance','--protocol','fine-transition.json'],
        [sys.executable,str(repo/'scripts/analyze_criticality.py'),'--root',str(root),'--pattern','fine-*',
         '--output',str(study/'analysis/fine')],
    ]
    records=[]
    for command in commands:
        result=subprocess.run(command,cwd=repo,env=env)
        records.append(dict(command=command,returncode=result.returncode))
        write_json(ledger,dict(status='running' if result.returncode==0 else 'execution_failure',records=records))
        if result.returncode:raise RuntimeError('The denser transition scan failed')
    write_json(ledger,dict(status='complete',records=records))


if __name__=='__main__':main()
