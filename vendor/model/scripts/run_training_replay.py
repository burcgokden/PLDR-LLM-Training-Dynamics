#!/usr/bin/env python
"""Run paired zero-pulse precision controls when the scheduled scans release GPUs."""
from companion_paths import child_pythonpath
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time
from model_rg.provenance import write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    root=Path(args.root);study=root/'criticality-study-20260905';repo=Path(__file__).resolve().parents[1]
    ledger=study/'launcher-training-replay.json'
    if ledger.exists():raise FileExistsError(ledger)
    parents=json.loads((study/'protocols/training-replay.json').read_text())['parents']
    jobs=[('replay-'+name.split('-',1)[1],name) for name in parents]
    write_json(ledger,dict(status='waiting_for_scans',jobs=jobs))
    while True:
        states=[json.loads((study/f'launcher-{name}.json').read_text())['status'] for name in ['horizon','fine-study']]
        if 'execution_failure' in states:raise RuntimeError('A replay prerequisite failed')
        if all(state=='complete' for state in states):break
        time.sleep(15)
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4')
    pending=queue.Queue()
    for job in jobs:pending.put(job)
    def worker(device):
        records=[]
        while True:
            try:run_id,parent=pending.get_nowait()
            except queue.Empty:break
            command=[sys.executable,str(repo/'scripts/measure_training_restoration.py'),'--root',str(root),
                     '--parent',parent,'--run-id',run_id,'--device',device,'--mode','replay','--protocol','training-replay.json']
            with (study/'logs'/(run_id+'.log')).open('x') as log:
                result=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
            record=dict(run_id=run_id,parent=parent,command=command,returncode=result.returncode)
            records.append(record);write_json(study/('ledger-'+run_id+'.json'),record)
        return records
    write_json(ledger,dict(status='running',jobs=jobs))
    with ThreadPoolExecutor(max_workers=2) as pool:
        records=[r for block in pool.map(worker,['cuda:0','cuda:1']) for r in block]
    if any(r['returncode'] for r in records):
        write_json(ledger,dict(status='execution_failure',jobs=jobs,records=records))
        raise RuntimeError('A native repeatability control failed')
    command=[sys.executable,str(repo/'scripts/analyze_training_replay.py'),'--root',str(root),'--output',str(study/'analysis/training-replay')]
    with (study/'logs/analysis-training-replay.log').open('x') as log:
        result=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
    records.append(dict(stage='analysis',command=command,returncode=result.returncode))
    write_json(ledger,dict(status='complete' if not result.returncode else 'execution_failure',jobs=jobs,records=records))
    if result.returncode:raise RuntimeError('The native repeatability analysis failed')


if __name__=='__main__':main()
