#!/usr/bin/env python
"""Measure each completed scan checkpoint on CPU as it becomes available."""
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
    ledger=study/'launcher-attention-regimes.json'
    if ledger.exists():raise FileExistsError(ledger)
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4')
    records=[];done=set()
    while True:
        scans=[json.loads((study/f'launcher-{name}.json').read_text()) for name in ['scan','variance']]
        names=[job[0] for scan in scans for job in scan['jobs']]
        ready=[name for name in names if name not in done and (study/'runs'/name/'manifest.json').exists()]
        for name in ready:
            command=[sys.executable,str(repo/'scripts/measure_attention_regime.py'),'--root',str(root),'--parent',name]
            with (study/'logs'/('attention-'+name+'.log')).open('x') as log:
                result=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
            record=dict(parent=name,returncode=result.returncode,command=command)
            records.append(record);done.add(name)
            write_json(ledger,dict(status='running',records=records,expected=len(names)))
            if result.returncode:raise RuntimeError(f'Attention diagnostic failed for {name}')
            print(len(done),name,flush=True)
        if any(scan['status']=='execution_failure' for scan in scans):
            raise RuntimeError('A prerequisite training scan failed')
        if all(scan['status']=='complete' for scan in scans) and len(done)==len(names):break
        time.sleep(15)
    write_json(ledger,dict(status='complete',records=records,expected=len(names)))


if __name__=='__main__':main()
