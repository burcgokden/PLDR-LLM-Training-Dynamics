#!/usr/bin/env python
"""Schedule a declared criticality scan with one worker per GPU."""
from companion_paths import child_pythonpath
import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from model_rg.controlled import device_name
from model_rg.provenance import source_manifest, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--label', default='scan')
    parser.add_argument('--devices', default='0,1')
    parser.add_argument('--heads', default='2,4,8,14')
    parser.add_argument('--multipliers', default='0,0.25,1,4,16')
    parser.add_argument('--seeds', default='640101,640102,640103,640104')
    parser.add_argument('--steps', type=int, default=2048)
    parser.add_argument('--normalization', choices=['fan_in','variance'], default='fan_in')
    parser.add_argument('--protocol', default='scan-fixed-environment.json')
    parser.add_argument('--stream-seed', type=int, default=640001)
    parser.add_argument('--shared-seed', type=int, default=640011)
    args = parser.parse_args()
    root = Path(args.root)
    study = root/'criticality-study-20260905'
    repo = Path(__file__).resolve().parents[1]
    devices = [device_name(d) for d in args.devices.split(',')]
    # Complete each four-seed condition before moving to the next multiplier.
    conditions = list(itertools.product(map(int,args.heads.split(',')), map(float,args.multipliers.split(',')), map(int,args.seeds.split(','))))
    jobs = [(f'{args.label}-h{h}-g{g:g}-s{s}',h,g,s) for h,g,s in conditions]
    ledger = study/f'launcher-{args.label}.json'
    if ledger.exists() or any((study/'runs'/name).exists() for name,*rest in jobs):
        raise FileExistsError('Existing launcher or run destination')
    sources = source_manifest()
    write_json(ledger,dict(status='running',arguments=vars(args),jobs=jobs,source_files=sources))
    env = dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4')
    def worker(index):
        records = []
        for name,h,g,seed in jobs[index::len(devices)]:
            command = [sys.executable,str(repo/'scripts/train_criticality.py'),'--root',str(root),'--run-id',name,
                       '--heads',str(h),'--seed',str(seed),'--multiplier',str(g),'--steps',str(args.steps),'--device',devices[index],'--stream-seed',str(args.stream_seed),'--shared-seed',str(args.shared_seed),'--normalization',args.normalization,'--protocol',args.protocol]
            started = time.time()
            print('Starting',name,devices[index],flush=True)
            with (study/'logs'/(name+'.log')).open('x') as log:
                completed = subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
            result = dict(run_id=name,command=command,returncode=completed.returncode,seconds=time.time()-started)
            manifest = study/'runs'/name/'manifest.json'
            if manifest.exists():
                result['scientific_status'] = json.loads(manifest.read_text())['status']
            records.append(result)
            write_json(study/('ledger-'+name+'.json'),result)
            print('Finished',name,result,flush=True)
            if completed.returncode:
                break
        return records
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        records = sum(list(pool.map(worker,range(len(devices)))),[])
    success = len(records)==len(jobs) and all(r['returncode']==0 for r in records)
    write_json(ledger,dict(status='complete' if success else 'execution_failure',arguments=vars(args),
                           jobs=jobs,records=records,source_files=sources))
    if not success:
        raise RuntimeError('Incomplete execution; all logs and partial outcomes are retained')


if __name__ == '__main__':
    main()
