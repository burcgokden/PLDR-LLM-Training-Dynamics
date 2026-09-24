#!/usr/bin/env python
"""Wait for complete inputs, then check and render the selected schedule pairs."""
from companion_paths import child_pythonpath
import argparse
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
    p.add_argument('--study', default='scheduled-training-feasible-20260908'); args = p.parse_args()
    root = Path(args.root).resolve(); study = root/args.study; repo = Path(__file__).resolve().parents[1]
    protocol = study/'protocols/onepass-schedule-comparison-selection.json'
    spec = json.loads(protocol.read_text())
    scripts = ['analyze_onepass_schedule_comparison.py', 'verify_onepass_schedule_comparison.py',
               'render_onepass_schedule_comparison.py']
    execution_sources = {name:sha256(repo/'scripts'/name) for name in [*scripts, Path(__file__).name]}
    for name, digest in spec['sources'].items():
        if sha256(repo/name) != digest:
            raise AssertionError('A selected schedule-comparison implementation changed')
    marker = study/'launcher-onepass-schedule-comparison.json'
    if marker.exists():
        raise FileExistsError(marker)
    os.nice(8)
    state = dict(status='waiting', started_at=datetime.now(timezone.utc).isoformat(),
        selection_sha256=sha256(protocol), execution_sources=execution_sources, records=[])
    write_json(marker, state)
    paths = [study/'verification'/name for name in [
        'onepass-collective-statistics.json', 'onepass-prediction-statistics.json']]
    while not all(path.exists() and json.loads(path.read_text())['status']=='passed' for path in paths):
        upstream = study/'launcher-onepass-analyses.json'
        if upstream.exists() and json.loads(upstream.read_text())['status']=='failed':
            state['status']='failed'; state['reason']='An upstream full analysis failed'; write_json(marker, state)
            raise RuntimeError(state['reason'])
        time.sleep(30)
    for name, digest in execution_sources.items():
        if sha256(repo/'scripts'/name) != digest:
            state['status']='failed'; state['reason']='An execution entrypoint changed'; write_json(marker, state)
            raise AssertionError(state['reason'])
    state['status']='analyzing'; write_json(marker, state)
    logs = study/'logs/onepass-schedule-comparison'; logs.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, 'PYTHONPATH':child_pythonpath("model"), 'PYTHONDONTWRITEBYTECODE':'1',
           'OPENBLAS_NUM_THREADS':'1', 'OMP_NUM_THREADS':'1'}
    proof = study/'verification/onepass-schedule-comparison.json'
    commands = []; started = time.time()
    for script in scripts:
        command = [sys.executable, str(repo/'scripts'/script)]
        if script.startswith('render_'):
            command += ['--data-root', str(root), '--manuscript', str(repo/'manuscript')]
        else:
            command += ['--root', str(root)]
        command += ['--study', args.study]
        if script.startswith('verify_'):
            command += ['--output', str(proof)]
        log = logs/(script+'.log')
        with log.open('x') as output:
            process = subprocess.run(command, cwd=repo, env=env, stdout=output, stderr=subprocess.STDOUT)
        commands.append(dict(command=command, returncode=process.returncode, log_sha256=sha256(log)))
        if process.returncode:
            state['status']='failed'; state['records']=[dict(name='schedule-comparison', status='failed', commands=commands)]
            write_json(marker, state)
            raise RuntimeError('The paired schedule execution failed: '+script)
    if json.loads(proof.read_text())['status'] != 'passed':
        raise AssertionError('The paired comparison lacks an independent successful check')
    state.update(status='complete', completed_at=datetime.now(timezone.utc).isoformat(), records=[dict(
        name='schedule-comparison', status='complete', commands=commands, verification_sha256=sha256(proof),
        seconds=time.time()-started)])
    write_json(marker, state)
    print('Completed, checked and rendered the matched single-pass schedule comparison', flush=True)


if __name__ == '__main__':
    main()
