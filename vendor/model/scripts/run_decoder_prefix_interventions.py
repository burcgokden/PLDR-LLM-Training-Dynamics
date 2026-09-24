#!/usr/bin/env python
"""Execute every selected decoder intervention and its independent causal-risk check."""
from companion_paths import child_pythonpath
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
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
    selection=study/'protocols/decoder-prefix-intervention-selection.json';spec=json.loads(selection.read_text())
    marker=study/'launcher-decoder-prefix-interventions.json'
    if marker.exists():raise FileExistsError(marker)
    for name,digest in spec['producer_sources'].items():
        if sha256(repo/name)!=digest:raise AssertionError('The selected decoder intervention implementation changed')
    state=dict(status='running',started_at=datetime.now(timezone.utc).isoformat(),selection_sha256=sha256(selection),records=[])
    write_json(marker,state);logs=study/'logs/decoder-prefix-interventions';logs.mkdir(parents=True,exist_ok=False)
    env={**os.environ,'PYTHONPATH':child_pythonpath("model"),'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4'}
    cases=sorted(spec['cases'],key=lambda c:(c['mode']!='fixed_decoder_4',c['recipe'],c['mode']))
    def execute(case):
        before=time.time();commands=[]
        for script in ['measure_decoder_prefix_intervention.py','verify_decoder_prefix_intervention.py']:
            command=[sys.executable,str(repo/'scripts'/script),'--root',str(root),'--study',a.study,'--case',case['name']]
            path=logs/(case['name']+'-'+script+'.log')
            with path.open('x') as log:process=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
            commands.append(dict(command=command,returncode=process.returncode,log_sha256=sha256(path)))
            if process.returncode:return dict(name=case['name'],status='failed',commands=commands,seconds=time.time()-before)
        proof=study/'verification/decoder-prefix-risk'/(case['name']+'.json')
        if json.loads(proof.read_text())['status']!='passed':raise AssertionError('Independent decoder-prefix verification failed')
        return dict(name=case['name'],status='complete',commands=commands,verification_sha256=sha256(proof),seconds=time.time()-before)
    with ThreadPoolExecutor(max_workers=spec['workers']) as pool:
        pending={};index=0
        while index<len(cases) or pending:
            while index<len(cases) and len(pending)<spec['workers']:
                pending[pool.submit(execute,cases[index])]=cases[index]['name'];index+=1
            for future in as_completed(pending):
                record=future.result();del pending[future];state['records'].append(record);write_json(marker,state)
                if record['status']!='complete':
                    state['status']='failed';write_json(marker,state);raise RuntimeError('Retained decoder intervention failure: '+record['name'])
                print('Complete decoder-prefix intervention and raw check',record['name'],flush=True);break
    state.update(status='complete',completed_at=datetime.now(timezone.utc).isoformat());write_json(marker,state)
    print('All16selecteddecoderinterventionscompletedandverified',flush=True)


if __name__=='__main__':main()
