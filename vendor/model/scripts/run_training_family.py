#!/usr/bin/env python
from companion_paths import child_pythonpath
from model_rg.controlled import device_name
"""Reproduce the finite training family, with one native model per GPU process."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True)
    ap.add_argument('--devices',default='0,1');args=ap.parse_args()
    root=Path(args.data_root).resolve();repo=Path(__file__).resolve().parents[1]
    devices=[device_name(x) for x in args.devices.split(',')];jobs=[(h,s) for h in [2,4,8,14] for s in [12001,12002,12003]]
    destinations=[root/'executed'/f'training-h{h}-s{s}' for h,s in jobs]
    if any(p.exists() for p in destinations):raise FileExistsError('Refusing to replace an existing training run')
    logs=root/'training-study/logs';logs.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4')
    def worker(index):
        for h,s in jobs[index::len(devices)]:
            cmd=[sys.executable,str(repo/'scripts/train_width_family.py'),
                 '--source',str(root/'assets/PLDR-LLM-v51-SOC-110M-1'),
                 '--data',str(root/'data/refinedweb-4608'),
                 '--output',str(root/'executed'/f'training-h{h}-s{s}'),
                 '--heads',str(h),'--seed',str(s),'--device',devices[index]]
            print('Starting',h,s,flush=True)
            with (logs/f'h{h}-s{s}.log').open('w') as stream:
                subprocess.run(cmd,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
            print('Completed',h,s,flush=True)
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        list(pool.map(worker,range(len(devices))))


if __name__=='__main__':main()
