#!/usr/bin/env python
from companion_paths import child_pythonpath
from model_rg.controlled import device_name
"""Measure the full-parameter inference response of each completed training run."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from model_rg.provenance import sha256


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True)
    ap.add_argument('--devices',default='0,1');ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();root=Path(args.data_root).resolve();repo=Path(__file__).resolve().parents[1]
    devices=[device_name(x) for x in args.devices.split(',')];jobs=[]
    for h in [2,4,8,14]:
        for s in [12001,12002,12003]:
            out=root/'executed'/f'training-response-h{h}-s{s}'
            if out.exists():
                if not args.resume:raise FileExistsError(out)
                meta=json.loads((out/'manifest.json').read_text())
                if sha256(out/'response.npz')!=meta['raw_sha256']:raise RuntimeError('Invalid completed response')
            else:jobs.append((h,s))
    logs=root/'training-study/logs';logs.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4')
    def worker(index):
        for h,s in jobs[index::len(devices)]:
            cmd=[sys.executable,str(repo/'scripts/measure_training_response.py'),
                 '--run',str(root/'executed'/f'training-h{h}-s{s}'),
                 '--output',str(root/'executed'/f'training-response-h{h}-s{s}'),
                 '--device',devices[index]]
            with (logs/f'response-h{h}-s{s}.log').open('w') as stream:
                subprocess.run(cmd,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
            print('Completed response',h,s,flush=True)
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:list(pool.map(worker,range(len(devices))))


if __name__=='__main__':main()
