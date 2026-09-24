#!/usr/bin/env python
"""Run the targeted CPU precision controls alongside the GPU training jobs."""
from companion_paths import child_pythonpath
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from model_rg.provenance import write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    root=Path(args.root);study=root/'criticality-study-20260905';repo=Path(__file__).resolve().parents[1]
    ledger=study/'launcher-gain-precision.json'
    if ledger.exists():raise FileExistsError(ledger)
    parents=json.loads((study/'protocols/source-precision.json').read_text())['parents']
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4')
    commands=[(name,[sys.executable,str(repo/'scripts/measure_gain_precision.py'),'--root',str(root),'--parent',name]) for name in parents]
    commands.append(('analysis',[sys.executable,str(repo/'scripts/analyze_gain_precision.py'),'--root',str(root),'--output',str(study/'analysis/source-precision-checked')]))
    records=[]
    for name,command in commands:
        write_json(ledger,dict(status='running',stage=name,records=records))
        with (study/'logs'/('gain-precision-'+name+'.log')).open('x') as log:
            result=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
        records.append(dict(stage=name,command=command,returncode=result.returncode))
        if result.returncode:
            write_json(ledger,dict(status='execution_failure',records=records))
            raise RuntimeError('A precision control failed; retain its execution evidence')
    write_json(ledger,dict(status='complete',records=records))


if __name__=='__main__':main()
