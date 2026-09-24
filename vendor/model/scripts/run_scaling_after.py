#!/usr/bin/env python
"""Execute another immutable training stage after the GPU predecessor finishes."""
from companion_paths import child_pythonpath
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from model_rg.provenance import sha256,write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--study',default='critical-scaling-20260906')
    p.add_argument('--predecessor',required=True)
    p.add_argument('--protocols',nargs='+',required=True)
    p.add_argument('--label',required=True)
    a=p.parse_args()
    study=Path(a.root)/a.study
    repo=Path(__file__).resolve().parents[1]
    path=study/('stage-'+a.label+'.json')
    if path.exists():raise FileExistsError(path)
    record=dict(status='waiting_for_predecessor_and_bound_protocols',arguments=vars(a),
                started_at=datetime.now(timezone.utc).isoformat(),controller_sha256=sha256(__file__))
    write_json(path,record)
    predecessor=study/a.predecessor
    while True:
        state=json.loads(predecessor.read_text())['status'] if predecessor.exists() else 'not_started'
        if state in ['complete','complete_with_numerical_failures'] and all((study/'protocols'/n).exists() for n in a.protocols):break
        if state not in ['not_started','running','complete','complete_with_numerical_failures']:
            raise RuntimeError('Predecessor needs an explicit resolution: '+state)
        time.sleep(30)
    signatures={n:sha256(study/'protocols'/n) for n in a.protocols}
    record.update(status='executing',protocol_sha256=signatures,predecessor_sha256=sha256(predecessor),
                  execution_started_at=datetime.now(timezone.utc).isoformat())
    write_json(path,record)
    command=[sys.executable,str(repo/'scripts/run_scaling.py'),'--root',a.root,'--study',a.study,
             '--protocols',*a.protocols,'--label',a.label]
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4')
    with (study/'logs'/('launcher-'+a.label+'.log')).open('x') as log:
        result=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
    record.update(status='complete' if result.returncode==0 else 'execution_failure',returncode=result.returncode,
                  finished_at=datetime.now(timezone.utc).isoformat(),command=command)
    write_json(path,record)
    if result.returncode:raise RuntimeError('Successor stage failed')


if __name__=='__main__':main()
