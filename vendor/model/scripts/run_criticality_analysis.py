#!/usr/bin/env python
"""Analyze each completed stage and freeze size predictions before unblinding."""
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
    ledger=study/'launcher-analysis.json'
    if ledger.exists():raise FileExistsError(ledger)
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4')
    stages=[('scan',['scan'],'analyze_criticality.py',[]),
            ('variance',['variance'],'analyze_criticality.py',['--pattern','variance-*','--reuse-reference']),
            ('width-prediction',[],'predict_width_holdout.py',[]),
            ('attention-regimes',['attention-regimes'],'analyze_attention_regimes.py',[]),
            ('restoration',['restoration'],'analyze_training_restoration.py',[]),
            ('horizon',['horizon'],'analyze_horizon.py',[]),
            ('width-holdout',['width-prerequisite'],'analyze_width_holdout.py',[])]
    records=[]
    for name,dependencies,script,extra in stages:
        write_json(ledger,dict(status='waiting',stage=name,records=records))
        while True:
            states=[json.loads((study/f'launcher-{d}.json').read_text()) for d in dependencies]
            if any(s['status']=='execution_failure' for s in states):raise RuntimeError('An analysis prerequisite failed')
            if all(s['status']=='complete' for s in states):break
            time.sleep(15)
        command=[sys.executable,str(repo/'scripts'/script),'--root',str(root),'--output',str(study/'analysis'/name),*extra]
        with (study/'logs'/('analysis-'+name+'.log')).open('x') as log:
            result=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
        records.append(dict(stage=name,command=command,returncode=result.returncode))
        write_json(ledger,dict(status='running' if result.returncode==0 else 'execution_failure',records=records))
        if result.returncode:raise RuntimeError(f'Analysis failed: {name}')
        print(name,'complete',flush=True)
    write_json(ledger,dict(status='complete',records=records))


if __name__=='__main__':main()
