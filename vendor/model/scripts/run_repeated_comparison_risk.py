#!/usr/bin/env python
"""Run all four retained-state risks, then reconstruct every recorded target."""
from companion_paths import child_pythonpath
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from model_rg.provenance import sha256, write_json


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True)
    p.add_argument('--study', default='scheduled-training-feasible-20260908')
    p.add_argument('--prepare', action='store_true'); a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]; root = Path(a.root).resolve(); study = root/a.study
    env = dict(os.environ, PYTHONPATH=child_pythonpath("model"), PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='4')
    selection = study/'protocols/repeated-risk-comparison.json'
    marker = study/'launcher-repeated-risk-comparison.json'
    if marker.exists(): raise FileExistsError(marker)
    logdir = study/'logs/repeated-risk'; logdir.mkdir(parents=True, exist_ok=False)
    def run(script, suffix, extra=()):
        command = [sys.executable,str(repo/'scripts'/script),'--root',str(root),'--study',a.study,*extra]
        before = time.time()
        with (logdir/(suffix+'.log')).open('x') as log:
            r = subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
        return dict(command=command,returncode=r.returncode,seconds=time.time()-before,log=str(logdir/(suffix+'.log')))
    record = dict(status='preparing' if a.prepare else 'measuring',started_at=datetime.now(timezone.utc).isoformat(),records=[])
    write_json(marker,record)
    if a.prepare:
        item = run('prepare_repeated_comparison.py','preparation'); record['preparation'] = item
        if item['returncode']:
            record['status']='preparation_failed'; write_json(marker,record); raise RuntimeError('Preserved preparation failure')
    spec = json.loads(selection.read_text()); signature = sha256(selection)
    record.update(status='measuring',selection_sha256=signature); write_json(marker,record)
    def measure(case):
        if sha256(selection) != signature: raise AssertionError('Selection changed during observation')
        return dict(name=case['name'],**run('measure_repeated_comparison_risk.py',case['name'],['--case',case['name']]))
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = [pool.submit(measure,case) for case in spec['cases']]
        for future in as_completed(pending):
            item=future.result(); record['records'].append(item); write_json(marker,record)
            print(json.dumps(item),flush=True)
    if any(item['returncode'] for item in record['records']):
        record['status']='measurement_failed'; write_json(marker,record); raise RuntimeError('Preserved risk failure')
    record['status']='verifying'; write_json(marker,record)
    item=run('verify_repeated_comparison_risk.py','verification'); record['verification']=item
    record['status']='complete' if item['returncode']==0 else 'verification_failed'
    if item['returncode']==0: record['verification_sha256']=sha256(study/'verification/repeated-risk-comparison.json')
    write_json(marker,record); print(record['status'],flush=True)
    if item['returncode']: raise RuntimeError('Preserved risk reconstruction failure')


if __name__ == '__main__': main()
