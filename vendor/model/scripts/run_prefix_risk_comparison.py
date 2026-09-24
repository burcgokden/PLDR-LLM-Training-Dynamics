#!/usr/bin/env python
"""Run one qualified CPU worker over the entire frozen causal-risk panel."""
from companion_paths import child_pythonpath
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from model_rg.provenance import sha256,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-feasible-20260908');a=p.parse_args()
    root=Path(a.root).resolve();study=root/a.study;repo=Path(__file__).resolve().parents[1]
    selection=study/'protocols/prefix-risk-comparison.json';spec=json.loads(selection.read_text());signature=sha256(selection)
    marker=study/'launcher-prefix-risk-comparison.json'
    if marker.exists():raise FileExistsError(marker)
    logdir=study/'logs/prefix-risk';logdir.mkdir(parents=True,exist_ok=False)
    record=dict(status='qualifying',started_at=datetime.now(timezone.utc).isoformat(),selection_sha256=signature,records=[])
    write_json(marker,record)
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4')
    def execute(case):
        if sha256(selection)!=signature:raise AssertionError('The causal-risk panel changed')
        item=dict(name=case['name'],commands=[]);before=time.time()
        for script in ['measure_prefix_risk_comparison.py','verify_prefix_risk_comparison.py']:
            command=[sys.executable,str(repo/'scripts'/script),'--root',str(root),'--study',a.study,'--case',case['name']]
            logpath=logdir/(case['name']+'-'+script+'.log')
            with logpath.open('x') as log:r=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
            item['commands'].append(dict(command=command,returncode=r.returncode,log_sha256=sha256(logpath)))
            if r.returncode:
                item['status']='failed';record['records'].append(item);record['status']='observation_failed'
                write_json(marker,record);raise RuntimeError('Preserved causal-risk failure: '+str(logpath))
        proof=study/'verification/prefix-risk'/(case['name']+'.json')
        if json.loads(proof.read_text())['status']!='passed':raise AssertionError('Causal-risk check incomplete')
        item.update(status='complete',seconds=time.time()-before,verification_sha256=sha256(proof))
        record['records'].append(item);write_json(marker,record);print(case['name'],'complete',flush=True)
    execute(spec['qualification'])
    record['status']='observing';write_json(marker,record)
    remaining=list(spec['cases'])
    while remaining:
        progressed=False
        for case in list(remaining):
            parent=Path(case['parent_manifest'])
            if not parent.exists():continue
            if case.get('training_verification') and not Path(case['training_verification']).exists():continue
            execute(case);remaining.remove(case);progressed=True
        if remaining and not progressed:time.sleep(30)
    record['status']='complete';record['scientific_states']=20;record['qualification_states']=1
    write_json(marker,record)


if __name__=='__main__':main()
