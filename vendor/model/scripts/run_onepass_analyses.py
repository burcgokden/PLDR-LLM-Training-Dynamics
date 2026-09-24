#!/usr/bin/env python
"""Run each full single-pass analysis once its independently checked inputs exist."""
from companion_paths import child_pythonpath
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from model_rg.provenance import sha256, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-feasible-20260908');a=p.parse_args()
    root=Path(a.root).resolve();study=root/a.study;repo=Path(__file__).resolve().parents[1]
    protocol=study/'protocols/onepass-analysis-selection.json';spec=json.loads(protocol.read_text())
    marker=study/'launcher-onepass-analyses.json'
    if marker.exists():raise FileExistsError(marker)
    for name,digest in spec['producer_sources'].items():
        if sha256(repo/name)!=digest:raise AssertionError('The frozen analysis implementation changed')
    state=dict(status='waiting',started_at=datetime.now(timezone.utc).isoformat(),selection_sha256=sha256(protocol),records=[])
    write_json(marker,state);logs=study/'logs/onepass-analyses';logs.mkdir(parents=True,exist_ok=False)
    env={**os.environ,'PYTHONPATH':child_pythonpath("model"),'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2'}
    tasks={
        'collectives':('analyze_onepass_collectives.py','verify_onepass_collectives.py','onepass-collective-statistics.json'),
        'predictions':('analyze_onepass_predictions.py','verify_onepass_predictions.py','onepass-prediction-statistics.json'),
        'temporal':('analyze_onepass_paths.py','verify_onepass_paths.py','onepass-temporal-statistics.json')}
    def ready(name):
        if name=='temporal':
            return all((study/'verification/training'/(run_id+'.json')).exists() and
                json.loads((study/'verification/training'/(run_id+'.json')).read_text())['status']=='complete'
                for run_id in spec['run_ids'])
        path=study/'launcher-onepass-observations.json'
        return path.exists() and json.loads(path.read_text())['status']=='complete'
    def execute(name):
        started=time.time();commands=[]
        for script in tasks[name][:2]:
            command=[sys.executable,str(repo/'scripts'/script),'--root',str(root),'--study',a.study]
            path=logs/(script+'.log')
            with path.open('x') as output:
                process=subprocess.run(command,cwd=repo,env=env,stdout=output,stderr=subprocess.STDOUT)
            commands.append(dict(command=command,returncode=process.returncode,log_sha256=sha256(path)))
            if process.returncode:
                return dict(name=name,status='failed',commands=commands,seconds=time.time()-started)
        check=study/'verification'/tasks[name][2]
        if json.loads(check.read_text())['status']!='passed':raise AssertionError('Analysis verification did not pass')
        return dict(name=name,status='complete',commands=commands,verification_sha256=sha256(check),seconds=time.time()-started)
    submitted=set();running={}
    with ThreadPoolExecutor(max_workers=2) as pool:
        while len(state['records'])<len(tasks):
            for name in tasks:
                if name not in submitted and len(running)<2 and ready(name):
                    running[pool.submit(execute,name)]=name;submitted.add(name)
                    state['status']='analyzing';write_json(marker,state)
            for future in list(running):
                if not future.done():continue
                record=future.result();state['records'].append(record);del running[future]
                write_json(marker,state)
                if record['status']!='complete':
                    state['status']='failed';write_json(marker,state)
                    raise RuntimeError('Analysis failed; retained inputs and failure must be reviewed: '+record['name'])
                print('Complete single-pass analysis and independent check',record['name'],flush=True)
            if len(state['records'])<len(tasks):time.sleep(30)
    state['status']='complete';state['completed_at']=datetime.now(timezone.utc).isoformat();write_json(marker,state)
    print('All three complete single-pass analyses independently verified',flush=True)


if __name__=='__main__':main()
