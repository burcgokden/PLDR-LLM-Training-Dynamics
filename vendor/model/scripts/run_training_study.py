#!/usr/bin/env python
from companion_paths import child_pythonpath
from model_rg.controlled import device_name
"""Reproduce the completed training study in an isolated output directory."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-root',required=True)
    ap.add_argument('--assets',required=True);ap.add_argument('--data',required=True)
    ap.add_argument('--devices',default='0,1');ap.add_argument('--print-commands',action='store_true')
    args=ap.parse_args();repo=Path(__file__).resolve().parents[1];root=Path(args.run_root).resolve()
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4')
    commands=[
        [sys.executable,str(repo/'scripts/run_training_family.py'),'--data-root',str(root),'--devices',args.devices],
        [sys.executable,str(repo/'scripts/run_training_response.py'),'--data-root',str(root),'--devices',args.devices]]
    devices=[device_name(x) for x in args.devices.split(',')];validations=[]
    for index,(h,s) in enumerate((h,s) for h in [2,4,8,14] for s in [12001,12002,12003]):
        validations.append([sys.executable,str(repo/'scripts/validate_training_derivative.py'),
            '--run',str(root/'executed'/f'training-h{h}-s{s}'),
            '--response',str(root/'executed'/f'training-response-h{h}-s{s}'),
            '--output',str(root/'executed'/f'training-derivative-h{h}-s{s}'),
            '--device',devices[index%len(devices)]])
    tail=[
        [sys.executable,str(repo/'scripts/analyze_training.py'),'--data-root',str(root),'--output',str(root/'analysis/training')],
        [sys.executable,str(repo/'scripts/verify_training.py'),'--data-root',str(root),
         '--output',str(root/'training-verification.json')]]
    if args.print_commands:
        for cmd in commands+validations+tail:print(shlex.join(cmd))
        return
    root.mkdir(parents=True,exist_ok=False)
    (root/'assets').symlink_to(Path(args.assets).resolve(),target_is_directory=True)
    (root/'data').mkdir();(root/'data/refinedweb-4608').symlink_to(Path(args.data).resolve(),target_is_directory=True)
    for cmd in commands:subprocess.run(cmd,cwd=repo,env=env,check=True)
    def worker(i):
        for cmd in validations[i::len(devices)]:subprocess.run(cmd,cwd=repo,env=env,check=True)
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:list(pool.map(worker,range(len(devices))))
    for cmd in tail:subprocess.run(cmd,cwd=repo,env=env,check=True)


if __name__=='__main__':main()
