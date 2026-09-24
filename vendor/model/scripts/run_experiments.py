#!/usr/bin/env python
from companion_paths import child_pythonpath
from model_rg.controlled import device_name
"""Reproduce the final design on two GPUs; refuse to replace completed runs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-root',required=True)
    p.add_argument('--assets',required=True)
    p.add_argument('--data',required=True)
    p.add_argument('--devices',default='0,1')
    p.add_argument('--print-commands',action='store_true')
    args=p.parse_args()
    repo=Path(__file__).resolve().parents[1]
    root=Path(args.run_root).resolve()
    devices=[device_name(x) for x in args.devices.split(',')]
    if len(devices)!=2 or len(set(devices))!=2:raise ValueError('Supply two distinct GPU indices')
    commands=[]
    for i,dev in enumerate(devices,1):
        common=['--model',str(Path(args.assets).resolve()/f'PLDR-LLM-v51-SOC-110M-{i}'),
                '--data',str(Path(args.data).resolve()),'--device',dev]
        stages=[('collect_features.py',f'features-soc{i}-s128',[]),
                ('measure_response.py',f'response-replication-soc{i}-s128',['--offset','768','--documents','256']),
                ('collect_segments.py',f'segments-soc{i}-s64',[]),
                ('validate_model.py',f'native-validation-soc{i}',[])]
        job=[]
        for script,name,extra in stages:
            output=root/'executed'/name
            if output.exists() and not args.print_commands:raise FileExistsError(output)
            job.append((name,[sys.executable,str(repo/'scripts'/script),*common,'--output',str(output),*extra]))
        commands.append(job)
    tail_commands = [
        [sys.executable, str(repo/'scripts/analyze.py'), '--data-root', str(root),
         '--output', str(root/'analysis/main')],
        [sys.executable, str(repo/'scripts/verify_artifacts.py'), '--data-root', str(root),
         '--data', str(Path(args.data).resolve()), '--assets', str(Path(args.assets).resolve()),
         '--output', str(root/'verification.json')],
    ]
    if args.print_commands:
        import shlex
        for job in commands:
            for _,cmd in job:print(shlex.join(cmd))
        for cmd in tail_commands:print(shlex.join(cmd))
        return
    root.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
    def worker(job):
        for name,cmd in job:
            with (root/(name+'.log')).open('w') as log:
                subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,job) for job in commands]
        for future in futures:future.result()
    for cmd in tail_commands:
        subprocess.run(cmd,cwd=repo,env=env,check=True)


if __name__=='__main__':main()
