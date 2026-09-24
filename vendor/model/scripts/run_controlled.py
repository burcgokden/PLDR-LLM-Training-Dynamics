#!/usr/bin/env python
"""One worker per GPU, complete process ledger, no replacement of run outputs."""
from companion_paths import child_pythonpath
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import time
from model_rg.controlled import device_name
from model_rg.provenance import write_json, source_manifest


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True)
    ap.add_argument('--devices',default='0,1');ap.add_argument('--steps',type=int,default=2048)
    ap.add_argument('--heads',default='2,4,8,14');ap.add_argument('--seeds',default='630101,630102,630103')
    ap.add_argument('--label',default='family');ap.add_argument('--normalization',choices=['native','fan_in'],default='native');a=ap.parse_args()
    root=Path(a.root);study=root/'controlled-study-20260905';repo=Path(__file__).resolve().parents[1]
    devices=[device_name(x) for x in a.devices.split(',')]
    jobs=[(h,s) for h in map(int,a.heads.split(',')) for s in map(int,a.seeds.split(','))]
    logs=study/'logs';logs.mkdir(parents=True,exist_ok=True)
    ledger=study/f'launcher-{a.label}.json'
    if ledger.exists():raise FileExistsError(ledger)
    ids=[f'{a.label}-h{h}-s{s}' for h,s in jobs]
    if any((study/'runs'/i).exists() for i in ids):raise FileExistsError('Run exists')
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4')
    launcher_sources=source_manifest()
    write_json(ledger,dict(status='running',arguments=vars(a),source_files=launcher_sources,jobs=ids))
    def worker(index):
        records=[]
        for h,s in jobs[index::len(devices)]:
            rid=f'{a.label}-h{h}-s{s}'
            cmd=[sys.executable,str(repo/'scripts/train_controlled.py'),'--root',str(root),
                 '--heads',str(h),'--seed',str(s),'--stream',str(s+100),'--device',devices[index],
                 '--steps',str(a.steps),'--run-id',rid,'--normalization',a.normalization]
            start=time.time();print('Starting',rid,devices[index],flush=True)
            with (logs/f'{rid}.log').open('x') as f:
                result=subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT)
            records.append(dict(run_id=rid,command=cmd,returncode=result.returncode,seconds=time.time()-start))
            write_json(study/f'ledger-{rid}.json',records[-1])
            print('Finished',rid,result.returncode,flush=True)
            if result.returncode:break
        return records
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:records=sum(list(pool.map(worker,range(len(devices)))),[])
    ok=len(records)==len(jobs) and all(x['returncode']==0 for x in records)
    write_json(ledger,dict(status='complete' if ok else 'failed',arguments=vars(a),source_files=launcher_sources,records=records))
    if not ok:raise RuntimeError('Incomplete family; see retained logs')


if __name__=='__main__':main()
