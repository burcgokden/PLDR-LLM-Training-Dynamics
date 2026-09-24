#!/usr/bin/env python
"""Check memory, then measure a withheld wider architecture on free GPUs."""
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
    ledger=study/'launcher-width-prerequisite.json'
    if ledger.exists():raise FileExistsError(ledger)
    write_json(ledger,dict(status='waiting_for_horizon'))
    while True:
        prerequisite=json.loads((study/'launcher-horizon.json').read_text())
        if prerequisite['status']=='execution_failure':raise RuntimeError('A prerequisite horizon run failed')
        if prerequisite['status']=='complete':break
        time.sleep(15)
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4')
    smoke=[sys.executable,str(repo/'scripts/train_criticality.py'),'--root',str(root),'--run-id','width-smoke-h24',
           '--heads','24','--seed','640100','--multiplier','1','--steps','64','--normalization','variance',
           '--protocol','width-holdout.json','--device','cuda:0']
    with (study/'logs/width-smoke-h24.log').open('x') as log:
        result=subprocess.run(smoke,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
    write_json(ledger,dict(status='smoke_complete' if result.returncode==0 else 'execution_failure',smoke_command=smoke,returncode=result.returncode))
    if result.returncode:raise RuntimeError('The wider-model resource control failed')
    command=[sys.executable,str(repo/'scripts/run_criticality.py'),'--root',str(root),'--label','width-holdout',
             '--heads','24','--multipliers','1','--normalization','variance','--protocol','width-holdout.json']
    result=subprocess.run(command,cwd=repo,env=env)
    write_json(ledger,dict(status='complete' if result.returncode==0 else 'execution_failure',smoke_command=smoke,training_command=command,returncode=result.returncode))
    if result.returncode:raise RuntimeError('The wider-model measurement failed')


if __name__=='__main__':main()
