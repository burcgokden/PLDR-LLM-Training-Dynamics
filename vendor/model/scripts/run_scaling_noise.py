#!/usr/bin/env python
"""Run the frozen conditional-noise inventory with bounded CPU concurrency."""
from companion_paths import child_pythonpath
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from model_rg.provenance import sha256, source_manifest, write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--study',default='critical-scaling-20260906')
    p.add_argument('--protocol',default='conditional-noise-jvp.json')
    p.add_argument('--workers',type=int,default=2)
    a=p.parse_args()
    if a.workers not in [1,2]:
        raise ValueError('Use one or two CPU workers')
    study=Path(a.root)/a.study
    repo=Path(__file__).resolve().parents[1]
    protocol=study/'protocols'/a.protocol
    spec=json.loads(protocol.read_text())
    signature=sha256(protocol)
    ledger=study/('launcher-'+Path(a.protocol).stem+'.json')
    if ledger.exists():
        raise FileExistsError(ledger)
    for case in spec['cases']:
        if (study/'measurements'/case['name']).exists():
            raise FileExistsError(case['name'])
    record=dict(status='running',arguments=vars(a),protocol_sha256=signature,
                source_files=source_manifest(),cases=spec['cases'])
    write_json(ledger,record)
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4')
    def run(case):
        if sha256(protocol)!=signature:
            raise AssertionError('Frozen protocol changed')
        command=[sys.executable,str(repo/'scripts/measure_scaling_noise.py'),'--root',a.root,
                 '--study',a.study,'--protocol',a.protocol,'--case',case['name'],'--device','cpu','--threads','4']
        started=time.time()
        with (study/'logs'/(case['name']+'.log')).open('x') as log:
            completed=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
        result=dict(name=case['name'],command=command,returncode=completed.returncode,seconds=time.time()-started)
        manifest=study/'measurements'/case['name']/'manifest.json'
        if manifest.exists():
            result.update(status=json.loads(manifest.read_text())['status'],manifest_sha256=sha256(manifest))
        write_json(study/('ledger-'+case['name']+'.json'),result)
        print(json.dumps(result),flush=True)
        return result
    started=time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        records=list(pool.map(run,spec['cases']))
    complete=all(r['returncode']==0 and r.get('status')=='complete' for r in records)
    record.update(status='complete' if complete else 'execution_failure',records=records,seconds=time.time()-started)
    write_json(ledger,record)
    if not complete:
        raise RuntimeError('One or more declared conditional-noise measurements failed')


if __name__=='__main__':
    main()
