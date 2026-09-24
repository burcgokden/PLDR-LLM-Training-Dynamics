#!/usr/bin/env python
"""Measure all selected risk states as their immutable parents become available."""
from companion_paths import child_pythonpath
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from model_rg.provenance import sha256, write_json


def main():
    p = argparse.ArgumentParser();p.add_argument('--root', required=True)
    p.add_argument('--study', default='critical-scaling-20260906');p.add_argument('--workers', type=int, default=2)
    p.add_argument('--resume-completed', action='store_true')
    a = p.parse_args();study = Path(a.root)/a.study;repo = Path(__file__).resolve().parents[1]
    protocol = study/'protocols/training-risk-selection.json';spec = json.loads(protocol.read_text())
    ledger = study/'launcher-training-risk.json'
    if ledger.exists():raise FileExistsError(ledger)
    if a.workers not in [1,2]:raise ValueError('Use one or two CPU workers')
    record = dict(status='running', arguments=vars(a), protocol_sha256=sha256(protocol), records=[])
    adopted = set()
    for case in spec['cases']:
        folder = study/'measurements'/case['name']
        if not folder.exists():continue
        if not a.resume_completed:raise FileExistsError(folder)
        m = json.loads((folder/'manifest.json').read_text())
        if m['status'] != 'complete' or m['case'] != case:raise AssertionError('Cannot adopt an incomplete or changed risk state')
        b = json.loads((folder/'binding.json').read_text())
        if b['inputs'][str(protocol.resolve())] != record['protocol_sha256']:raise AssertionError('The adopted risk selection differs')
        for name,key in [('binding.json','binding_sha256'),('results.json','results_sha256'),('measurements.npz','raw_sha256')]:
            if sha256(folder/name) != m[key]:raise AssertionError('Changed completed risk artifact')
        for name,signature in b['inputs'].items():
            if sha256(name) != signature:raise AssertionError('Changed adopted risk input')
        for name,signature in b['source_files'].items():
            if sha256(folder/'source'/name) != signature:raise AssertionError('Changed adopted risk source')
        adopted.add(case['name'])
        record['records'].append(dict(name=case['name'],returncode=0,adopted_completed=True,
            manifest_sha256=sha256(folder/'manifest.json')))
    write_json(ledger, record)
    env = dict(os.environ, PYTHONPATH=child_pythonpath("model"), PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='4')
    def execute(case):
        if sha256(protocol) != record['protocol_sha256']:raise AssertionError('The risk selection changed')
        command = [sys.executable, str(repo/'scripts/measure_scaling_training_risk.py'),
                   '--root', a.root, '--study', a.study, '--case', case['name']]
        with (study/'logs'/(case['name']+'.log')).open('x') as log:
            result = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        row = dict(name=case['name'], command=command, returncode=result.returncode)
        path = study/'measurements'/case['name']/'manifest.json'
        if path.exists():row['manifest_sha256'] = sha256(path)
        return row
    pending = [c for c in spec['cases'] if c['name'] not in adopted];active = {}
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        while pending or active:
            for case in list(pending):
                if len(active) >= a.workers:break
                paths = [Path(case[k]) for k in ['parent_manifest', 'reference_observation']]
                if all(p.exists() and json.loads(p.read_text())['status'] == 'complete' for p in paths):
                    active[pool.submit(execute, case)] = case;pending.remove(case)
            if not active:
                time.sleep(30);continue
            done, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                case = active.pop(future);row = future.result();record['records'].append(row)
                write_json(ledger, record);print(json.dumps(row), flush=True)
                if row['returncode']:
                    record['status'] = 'execution_failure';write_json(ledger, record)
                    raise RuntimeError('Resolve the selected risk observation: '+case['name'])
    record['status'] = 'complete';write_json(ledger, record)


if __name__ == '__main__':main()
