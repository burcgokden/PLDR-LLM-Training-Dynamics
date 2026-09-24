#!/usr/bin/env python
"""Wait for the declared final analyses, then run the full evidence gate."""
from companion_paths import child_pythonpath
from companion_paths import legacy_path
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPO = Path(legacy_path('/pldr-code/model'))
ROOT = Path(legacy_path('/pldr-data/model'))
STUDY = ROOT / 'criticality-dynamics-20260906'
ANALYSES = [
    'pilot', 'replication', 'long', 'controls', 'row-transport-summary',
    'row-projection-summary', 'optimizer-scales', 'row-adjoint-summary',
    'row-gradient-projection-summary', 'metric-collectives',
]


def main():
    record = STUDY / 'qa/final-verification-queue.json'
    output = STUDY / 'verification.json'
    if record.exists() or output.exists():
        raise FileExistsError('A final verification record already exists')
    command = [sys.executable, str(REPO / 'scripts/verify_dynamics.py'),
               '--root', str(ROOT), '--output', str(output)]
    state = dict(status='waiting_for_completed_analyses', command=command,
                 queued_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                 queue_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    record.write_text(json.dumps(state, indent=2) + '\n')
    while True:
        complete = True
        for name in ANALYSES:
            path = STUDY / 'analysis' / name / 'manifest.json'
            if not path.exists():
                complete = False
                continue
            if json.loads(path.read_text())['status'] != 'complete':
                raise AssertionError('Unresolved analysis outcome: ' + name)
        if complete:
            break
        for name in ['launcher-pilot', 'launcher-replication',
                     'launcher-long-horizon', 'collective-return']:
            path = STUDY / (name + '.json')
            if path.exists() and json.loads(path.read_text())['status'] not in ['running', 'complete']:
                raise AssertionError('Unresolved scientific execution: ' + name)
        time.sleep(30)
    state['status'] = 'verifying'
    state['started_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    record.write_text(json.dumps(state, indent=2) + '\n')
    env = dict(os.environ, PYTHONPATH=child_pythonpath("model"),
               PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='4')
    with (STUDY / 'qa/final-verification-attempt-01.log').open('x') as log:
        result = subprocess.run(command, cwd=REPO, env=env, stdout=log,
                                stderr=subprocess.STDOUT)
    state['returncode'] = result.returncode
    state['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    state['status'] = 'passed' if result.returncode == 0 else 'verification_failed'
    record.write_text(json.dumps(state, indent=2) + '\n')
    print(json.dumps(state, indent=2), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
