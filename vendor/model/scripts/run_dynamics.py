#!/usr/bin/env python
"""Run an immutable dynamics protocol with one native training worker per GPU."""
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
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--study',default='criticality-dynamics-20260906')
    parser.add_argument('--protocol',default='pilot.json')
    args=parser.parse_args()
    root=Path(args.root);study=root/args.study;repo=Path(__file__).resolve().parents[1]
    protocol_path=study/'protocols'/args.protocol
    protocol=json.loads(protocol_path.read_text())
    jobs=protocol['jobs'];label=Path(args.protocol).stem
    ledger=study/('launcher-'+label+'.json')
    if ledger.exists():raise FileExistsError(ledger)
    if len({job['run_id'] for job in jobs})!=len(jobs):raise ValueError('Repeated run identifier')
    for job in jobs:
        if (study/'runs'/job['run_id']).exists():raise FileExistsError(job['run_id'])
    sources=source_manifest();signature=sha256(protocol_path)
    write_json(ledger,dict(status='running',jobs=jobs,protocol_sha256=signature,source_files=sources))
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4')
    def worker(index):
        records=[]
        for job in jobs[index::2]:
            command=[sys.executable,str(repo/'scripts/train_dynamics.py'),'--root',str(root),
                     '--study',args.study,'--protocol',args.protocol,'--normalization',job.get('normalization','variance'),
                     '--run-id',job['run_id'],'--heads',str(job['heads']),'--seed',str(job['seed']),
                     '--multiplier',str(job['multiplier']),'--steps',str(job['steps']),'--device',f'cuda:{index}',
                     '--shared-seed',str(job.get('shared_seed',protocol.get('shared_seed',640011))),
                     '--stream-seed',str(job.get('stream_seed',protocol.get('stream_seed',640001)))]
            if job.get('resume'):command+=['--resume',job['resume']]
            started=time.time()
            with (study/'logs'/(job['run_id']+'.log')).open('x') as log:
                result=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
            record=dict(run_id=job['run_id'],command=command,returncode=result.returncode,seconds=time.time()-started)
            path=study/'runs'/job['run_id']/'manifest.json'
            if path.exists():record['scientific_status']=json.loads(path.read_text())['status']
            records.append(record)
            write_json(study/('ledger-'+job['run_id']+'.json'),record)
            print(json.dumps(record),flush=True)
            if result.returncode:break
        return records
    with ThreadPoolExecutor(max_workers=2) as pool:
        records=sum(list(pool.map(worker,range(2))),[])
    complete=len(records)==len(jobs) and all(r['returncode']==0 for r in records)
    finite=complete and all(r.get('scientific_status')=='complete' for r in records)
    status=('complete' if finite else 'complete_with_numerical_failures') if complete else 'execution_failure'
    write_json(ledger,dict(status=status,jobs=jobs,records=records,protocol_sha256=signature,source_files=sources))
    if not complete:raise RuntimeError('The declared dynamics inventory did not execute completely')


if __name__=='__main__':main()
