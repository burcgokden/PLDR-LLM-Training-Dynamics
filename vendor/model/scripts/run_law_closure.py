#!/usr/bin/env python
"""One native worker per GPU; refuse unqualified or unfrozen validation."""
from companion_paths import child_pythonpath
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from model_rg.provenance import sha256, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--study',required=True)
    p.add_argument('--role',choices=['calibration','validation'],required=True);a=p.parse_args()
    study=Path(a.study).resolve();repo=Path(__file__).resolve().parents[1]
    spec=json.loads((study/'protocol.json').read_text())
    path=study/('launcher-'+a.role+'.json')
    if path.exists():raise FileExistsError(path)
    for heads in [4,14]:
        q=json.loads((study/'qualification'/f'h{heads}-s640101-t8192/results.json').read_text())
        if q['status']!='complete' or not q['reset_replay_bitwise'] or q['native_updates']!=12:
            raise AssertionError('Both native replay qualifications are required')
    if a.role=='validation':
        fit=json.loads((study/'frozen-fit.json').read_text())
        if fit['status']!='frozen' or fit['protocol_sha256']!=sha256(study/'protocol.json'):
            raise AssertionError('Freeze calibration-only maps before validation')
    records=[]
    def lane(heads,device):
        result=[]
        for c in spec['cases']:
            if c['role']!=a.role or c['heads']!=heads:continue
            command=[sys.executable,str(repo/'scripts/measure_law_closure.py'),'--root',a.root,
                '--study',str(study),'--case',c['name'],'--device',device]
            env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4')
            log=study/(c['name']+'.log')
            with log.open('x') as stream:
                proc=subprocess.run(command,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT)
            if proc.returncode:raise RuntimeError('Native branch run failed; see '+str(log))
            target=study/'runs'/c['name']/'results.json';r=json.loads(target.read_text())
            if r['status']!='complete' or r['scientific_updates']!=1024:raise AssertionError('Incomplete native state')
            result.append(dict(case=c['name'],results=str(target),sha256=sha256(target),device=device))
            print('Completed',c['name'],'seconds',round(r['seconds'],1),flush=True)
        return result
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(lane,4,'cuda:0'),pool.submit(lane,14,'cuda:1')]
        for f in futures:records.extend(f.result())
    write_json(path,dict(status='complete',role=a.role,records=records,scientific_updates=8192,
        completed_at=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha256(study/'protocol.json'),
        launcher_sha256=sha256(__file__),frozen_fit_sha256=sha256(study/'frozen-fit.json') if a.role=='validation' else None))


if __name__=='__main__':main()
