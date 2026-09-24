#!/usr/bin/env python3
"""Drain an owned static scheduler, then balance only its unstarted frozen jobs.

This changes scheduling, never the protocol or worker acquisition source.
The transition receipt and this source hash are retained beside the observations.
"""
from companion_paths import child_pythonpath
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from run_matched_clock import admit,ROOT,REPO,read
from model_rg.provenance import sha256,write_json


def run(study,paused_pid):
    command=Path(f'/proc/{paused_pid}/cmdline').read_bytes().split(b'\0')
    if b'run' not in command or str(study).encode() not in command or b'scripts/run_matched_clock.py' not in command:raise ValueError('Not the owned study scheduler')
    beg=time.monotonic()
    while True:
        manifests=[read(p) for p in (study/'runs').glob('*/manifest.json')]
        if any(m['status']=='failed' for m in manifests):raise RuntimeError('Failed scientific job must be retained')
        if all(m['status']=='complete' for m in manifests):break
        if time.monotonic()-beg>1900:raise TimeoutError('Existing workers did not drain')
        time.sleep(2)
    p=admit(study,True)
    # The scheduler is stopped and all of its scientific children have finished.
    # Deliver termination before continuation so it cannot launch another job.
    os.kill(paused_pid,signal.SIGTERM);os.kill(paused_pid,signal.SIGCONT)
    pending=[i for i,j in enumerate(p['jobs']) if not (study/'runs'/j['run_id']).exists()]
    pending.sort(key=lambda i:p['jobs'][i]['heads'],reverse=True)
    receipt=dict(schema='matched-clock-balanced-schedule-v1',status='running',protocol_sha256=sha256(study/'protocol.json'),
        scheduler_sha256=sha256(__file__),drained_jobs=len(manifests),remaining_jobs=[p['jobs'][i]['run_id'] for i in pending],
        scientific_source_changes=False,additional_trajectories=0,assignments=[])
    path=study/'balanced-schedule.json';write_json(path,receipt);lock=threading.Lock()
    def queue(device):
        while True:
            with lock:
                if not pending:return
                index=pending.pop(0);job=p['jobs'][index]
                elapsed=sum(read(f).get('elapsed_seconds',0) for f in (study/'runs').glob('*/manifest.json'))
                if elapsed+3600>p['worker_hour_cap']*3600:raise RuntimeError('Aggregate time ceiling')
                receipt['assignments'].append(dict(run_id=job['run_id'],device=device));write_json(path,receipt)
            logfile=study/'logs'/(job['run_id']+'.log')
            if logfile.exists() or (study/'runs'/job['run_id']).exists():raise ValueError('Job already claimed')
            with logfile.open('x') as log:
                subprocess.run([sys.executable,'-B',str(REPO/'scripts/run_matched_clock.py'),'worker','--study',str(study),'--index',str(index),'--device',device],
                    cwd=REPO,env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1900)
            print('COMPLETE '+job['run_id'],flush=True)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(queue,f'cuda:{d}') for d in range(2)]
            for f in futures:f.result()
        receipt['status']='complete'
    except Exception as e:receipt.update(status='failed',error=repr(e));raise
    finally:
        receipt['elapsed_seconds']=time.monotonic()-beg;write_json(path,receipt)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--study',type=Path,required=True);a.add_argument('--paused-pid',type=int,required=True);args=a.parse_args()
    if not args.study.resolve().is_relative_to(ROOT):raise ValueError('Authorized study required')
    run(args.study.resolve(),args.paused_pid)
