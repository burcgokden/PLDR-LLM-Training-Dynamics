#!/usr/bin/env python
"""One completed-state replay worker per GPU."""
from companion_paths import child_pythonpath
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--study',required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];study=Path(a.study).resolve()
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4')
    def run(heads,device):
        command=[sys.executable,str(repo/'scripts/replay_outer_transfer.py'),'--root',a.root,'--study',str(study),
            '--case',f'h{heads}-c1-i1','--device',device,'--branch','31','--output',str(study/f'replay-h{heads}.json')]
        with (study/f'replay-h{heads}.log').open('x') as log:subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run,4,'cuda:0'),pool.submit(run,14,'cuda:1')]
        for future in futures:future.result()


if __name__=='__main__':main()
