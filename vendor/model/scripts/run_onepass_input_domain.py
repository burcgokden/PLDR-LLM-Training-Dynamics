#!/usr/bin/env python
"""Complete the selected CPU prefix-domain and local derivative mechanism extension."""
from companion_paths import child_pythonpath
import argparse
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
    selection=study/'protocols/onepass-input-domain-selection.json';spec=json.loads(selection.read_text())
    marker=study/'launcher-onepass-input-domain-study.json'
    if marker.exists():raise FileExistsError(marker)
    for name,digest in spec['producer_sources'].items():
        if sha256(repo/name)!=digest:raise AssertionError('The selected mechanism implementation changed')
    record=dict(status='running',started_at=datetime.now(timezone.utc).isoformat(),selection_sha256=sha256(selection),records=[])
    write_json(marker,record);logs=study/'logs/onepass-input-domain';logs.mkdir(parents=True,exist_ok=False)
    env={**os.environ,'PYTHONPATH':child_pythonpath("model"),'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4'}
    jobs=[('measure_onepass_input_domain.py',[]),('verify_onepass_input_domain.py',[]),
        ('prepare_onepass_input_domain.py',['--stage','jacobians']),('measure_onepass_row_jacobians.py',[]),
        ('verify_onepass_row_jacobians.py',[])]
    for name,extra in jobs:
        before=time.time();cmd=[sys.executable,str(repo/'scripts'/name),'--root',str(root),'--study',a.study,*extra]
        path=logs/(name+'.log')
        with path.open('x') as log:process=subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
        record['records'].append(dict(name=name,command=cmd,returncode=process.returncode,log_sha256=sha256(path),seconds=time.time()-before))
        if process.returncode:
            record['status']='failed';write_json(marker,record);raise RuntimeError('Retained mechanism failure: '+name)
        write_json(marker,record);print('Complete mechanism stage',name,flush=True)
    checks=[study/'verification'/name for name in ['onepass-input-domain.json','onepass-row-jacobians.json']]
    if any(json.loads(p.read_text())['status']!='passed' for p in checks):raise AssertionError('Independent mechanism checks are required')
    record.update(status='complete',completed_at=datetime.now(timezone.utc).isoformat(),verification_sha256={str(p):sha256(p) for p in checks})
    write_json(marker,record);print('Complete single-pass prefix-domain mechanism extension',flush=True)


if __name__=='__main__':main()
