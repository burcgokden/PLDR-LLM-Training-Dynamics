#!/usr/bin/env python
"""Execute every selected fresh-batch predictive-subspace validation on CPUs."""
from companion_paths import child_pythonpath
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

from model_rg.provenance import sha256, write_json


def main():
    p = argparse.ArgumentParser();p.add_argument('--root', required=True)
    p.add_argument('--study', default='critical-scaling-20260906');p.add_argument('--workers', type=int, default=2)
    a = p.parse_args();study = Path(a.root)/a.study;repo = Path(__file__).resolve().parents[1]
    protocol = study/'protocols/source-basis.json';spec = json.loads(protocol.read_text())
    qualification = study/'measurements/qa-source-basis-h4-t16384-s640101'
    meta = json.loads((qualification/'manifest.json').read_text())
    result = json.loads((qualification/'results.json').read_text())
    if meta['status'] != 'complete' or not all(r['duality_passed'] for r in result['numerical']):
        raise AssertionError('The independent-batch source qualification did not pass')
    for case in spec['cases']:
        if (study/'measurements'/case['name']).exists():raise FileExistsError(case['name'])
    ledger = study/'launcher-source-basis.json'
    if ledger.exists():raise FileExistsError(ledger)
    record = dict(status='running', arguments=vars(a), protocol_sha256=sha256(protocol),
        qualification_manifest_sha256=sha256(qualification/'manifest.json'), cases=spec['cases'], records=[])
    write_json(ledger, record)
    env = dict(os.environ, PYTHONPATH=child_pythonpath("model"), PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='4')
    def execute(case):
        if sha256(protocol) != record['protocol_sha256']:raise AssertionError('Frozen protocol changed')
        command = [sys.executable, str(repo/'scripts/measure_scaling_source_basis.py'), '--root', a.root,
            '--study', a.study, '--protocol', protocol.name, '--case', case['name'], '--threads', '4']
        with (study/'logs'/(case['name']+'.log')).open('x') as log:
            run = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        row = dict(name=case['name'], command=command, returncode=run.returncode,
                   status='complete' if run.returncode == 0 else 'execution_failure')
        path = study/'measurements'/case['name']/'manifest.json'
        if path.exists():row['manifest_sha256'] = sha256(path)
        return row
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for row in pool.map(execute, spec['cases']):
            record['records'].append(row);write_json(ledger, record);print(json.dumps(row), flush=True)
    record['status'] = 'complete' if all(r['status'] == 'complete' for r in record['records']) else 'complete_with_failures'
    write_json(ledger, record)
    if record['status'] != 'complete':raise RuntimeError('Selected source validations require resolution')


if __name__ == '__main__':main()
