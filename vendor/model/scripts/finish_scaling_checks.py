#!/usr/bin/env python
"""Run independent full raw/statistical checks after the selected analysis completes."""
from companion_paths import child_pythonpath
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from model_rg.provenance import sha256, write_json


def main():
    p = argparse.ArgumentParser();p.add_argument('--root', required=True)
    p.add_argument('--study', default='critical-scaling-20260906');a = p.parse_args()
    study = Path(a.root)/a.study;repo = Path(__file__).resolve().parents[1]
    marker = study/'raw-statistical-check-stage.json'
    if marker.exists():raise FileExistsError(marker)
    dependencies = [study/'analysis-completion-stage.json', study/'verification/native-step-bytewise.json']
    record = dict(status='waiting_for_complete_selected_evidence', arguments=vars(a),
        started_at=datetime.now(timezone.utc).isoformat(), records=[],
        scope='Full independent checks of the explicitly selected study. This controller neither publishes a manuscript nor treats authorization for future extensions as executed evidence.')
    write_json(marker, record)
    while True:
        states = [json.loads(path.read_text())['status'] if path.exists() else None for path in dependencies]
        if any(state and ('fail' in state or 'error' in state) for state in states):
            raise RuntimeError('An inherited execution or replay needs resolution')
        if states == ['complete', 'passed']:break
        time.sleep(30)
    out = study/'verification/full-selected';out.mkdir(parents=True, exist_ok=False)
    record.update(status='independent_reconstructions_running',
        predecessor_sha256={str(path):sha256(path) for path in dependencies})
    write_json(marker, record)
    env = dict(os.environ, PYTHONPATH=child_pythonpath("model"), PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='4')
    def execute(item):
        label, script = item
        destination = out/(label+'.json')
        command = [sys.executable, str(repo/'scripts'/script), '--root', a.root,
                   '--study', a.study, '--output', str(destination)]
        started = time.time()
        with (study/'logs'/('verify-complete-'+label+'.log')).open('x') as log:
            result = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        row = dict(label=label, command=command, returncode=result.returncode, seconds=time.time()-started)
        if destination.exists():
            row.update(path=str(destination), sha256=sha256(destination),
                       status=json.loads(destination.read_text())['status'])
        else:row['status'] = 'execution_failure'
        return row
    with ThreadPoolExecutor(max_workers=2) as pool:
        for row in pool.map(execute, [('raw', 'verify_scaling_raw.py'),
                                     ('statistics', 'verify_scaling_statistics.py')]):
            record['records'].append(row);write_json(marker, record);print(json.dumps(row), flush=True)
    record['status'] = ('complete' if all(r['status'] == 'passed' and r['returncode'] == 0
                                        for r in record['records']) else 'complete_with_failures')
    write_json(marker, record)
    if record['status'] != 'complete':raise RuntimeError('Complete selected checks need resolution')


if __name__ == '__main__':main()
