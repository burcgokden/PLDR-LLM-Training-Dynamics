#!/usr/bin/env python
"""Execute frozen size-time protocols with one native training worker per GPU."""
from companion_paths import child_pythonpath
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time

from model_rg.provenance import sha256, source_manifest, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--study', default='critical-scaling-20260906')
    p.add_argument('--protocols', nargs='+', required=True)
    p.add_argument('--label', required=True)
    a = p.parse_args()
    root = Path(a.root)
    study = root/a.study
    repo = Path(__file__).resolve().parents[1]
    ledger = study/('launcher-'+a.label+'.json')
    if ledger.exists():
        raise FileExistsError(ledger)
    jobs, signatures, prerequisites = [], {}, {}
    for name in a.protocols:
        path = study/'protocols'/name
        spec = json.loads(path.read_text())
        signatures[name] = sha256(path)
        prerequisites.update(spec.get('device_prerequisites', {}))
        jobs.extend(dict(protocol=name, job=j, shared_seed=spec['shared_seed'],
                         stream_seed=spec['stream_seed'], probe_every=spec['probe_every']) for j in spec['jobs'])
    if len({x['job']['run_id'] for x in jobs}) != len(jobs):
        raise ValueError('Run identities must be unique')
    for entry in jobs:
        if (study/'runs'/entry['job']['run_id']).exists():
            raise FileExistsError(entry['job']['run_id'])
    sources = source_manifest()
    write_json(ledger, dict(status='running', jobs=jobs, protocol_sha256=signatures, source_files=sources))
    tasks = queue.Queue()
    for entry in jobs:
        tasks.put(entry)
    env = dict(os.environ, PYTHONPATH=child_pythonpath("model"), PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='4')

    def worker(device):
        records = []
        prerequisite = prerequisites.get(str(device))
        if prerequisite:
            marker = Path(prerequisite['manifest'])
            while not marker.exists():
                try:
                    os.kill(prerequisite['pid'], 0)
                except ProcessLookupError:
                    raise RuntimeError('Prerequisite worker ended without a completed manifest')
                time.sleep(2)
            if json.loads(marker.read_text())['status'] != 'complete':
                raise RuntimeError('Prerequisite trajectory did not complete')
        while True:
            try:
                entry = tasks.get_nowait()
            except queue.Empty:
                break
            j = entry['job']
            if sha256(study/'protocols'/entry['protocol']) != signatures[entry['protocol']]:
                raise AssertionError('Frozen protocol changed')
            if j.get('resume') and sha256(j['resume']) != j['parent_sha256']:
                raise AssertionError('Parent checkpoint changed')
            command = [sys.executable, str(repo/'scripts/train_scaling.py'), '--root', str(root),
                       '--study', a.study, '--protocol', entry['protocol'], '--normalization', 'variance',
                       '--run-id', j['run_id'], '--heads', str(j['heads']), '--seed', str(j['seed']),
                       '--multiplier', str(j['multiplier']), '--steps', str(j['steps']), '--device', f'cuda:{device}',
                       '--shared-seed', str(j.get('shared_seed', entry['shared_seed'])),
                       '--stream-seed', str(j.get('stream_seed', entry['stream_seed'])),
                       '--probe-every', str(entry['probe_every']), '--microbatch', str(j.get('microbatch', 32)),
                       '--save-steps', j.get('save_steps', '')]
            if j.get('checkpoint_decoders'):
                command += ['--checkpoint-decoders']
            if j.get('resume'):
                command += ['--resume', j['resume']]
            start = time.time()
            with (study/'logs'/(j['run_id']+'.log')).open('x') as log:
                completed = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
            record = dict(run_id=j['run_id'], protocol=entry['protocol'], command=command,
                          returncode=completed.returncode, seconds=time.time()-start)
            manifest = study/'runs'/j['run_id']/'manifest.json'
            if manifest.exists():
                record.update(scientific_status=json.loads(manifest.read_text())['status'], manifest_sha256=sha256(manifest))
            write_json(study/('ledger-'+j['run_id']+'.json'), record)
            print(json.dumps(record), flush=True)
            records.append(record)
            tasks.task_done()
            if completed.returncode:
                break
        return records

    start = time.time()
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = sum(list(pool.map(worker, range(2))), [])
    executed = len(records) == len(jobs) and all(x['returncode'] == 0 for x in records)
    finite = executed and all(x.get('scientific_status') == 'complete' for x in records)
    status = ('complete' if finite else 'complete_with_numerical_failures') if executed else 'execution_failure'
    write_json(ledger, dict(status=status, jobs=jobs, records=records, protocol_sha256=signatures,
                           source_files=sources, seconds=time.time()-start))
    if not executed:
        raise RuntimeError('The declared size-time inventory did not execute completely')


if __name__ == '__main__':
    main()
