#!/usr/bin/env python
"""Execute the two frozen reference benchmark states with bounded CPU concurrency."""
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
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--study', default='scheduled-training-20260908')
    args = p.parse_args()
    root = Path(args.root).resolve()
    study = root/args.study
    repo = Path(__file__).resolve().parents[1]
    protocol = study/'protocols/reference-benchmark-selection.json'
    spec = json.loads(protocol.read_text())
    signature = sha256(protocol)
    marker = study/'launcher-reference-benchmarks.json'
    if marker.exists():
        raise FileExistsError(marker)
    sources = {name: sha256(repo/name) for name in ['scripts/run_reference_benchmarks.py',
        'scripts/measure_reference_benchmarks.py', 'scripts/verify_reference_benchmarks.py',
        'scripts/verify_reference_benchmark_cohort.py', 'src/model_rg/inference_interventions.py',
        'src/model_rg/native.py', 'src/model_rg/training.py', 'src/model_rg/controlled.py', 'src/model_rg/provenance.py']}
    ledger = dict(status='checking_cohort', started_at=datetime.now(timezone.utc).isoformat(),
                  selection_sha256=signature, producer_sources=sources, records=[])
    write_json(marker, ledger)
    env = dict(os.environ, PYTHONPATH=child_pythonpath("model"), PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='4')
    common = ['--root', str(root), '--study', args.study]
    check = study/'verification/reference-benchmark-cohort.json'
    if not check.exists():
        with (study/'logs/reference-benchmark-cohort-check.log').open('x') as log:
            run = subprocess.run([sys.executable, str(repo/'scripts/verify_reference_benchmark_cohort.py'), *common],
                                 cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        if run.returncode:
            ledger['status'] = 'cohort_verification_failure'
            write_json(marker, ledger)
            raise RuntimeError('Resolve the preserved benchmark cohort verification failure')
    proof = json.loads(check.read_text())
    if proof['status'] != 'passed' or proof['cohort_sha256'] != spec['cohort_sha256']:
        raise AssertionError('The complete benchmark cohort must be verified before any model score')
    ledger.update(status='measuring', cohort_verification_sha256=sha256(check))
    write_json(marker, ledger)

    def execute(case):
        for name, digest in {**sources, **spec['producer_sources']}.items():
            if sha256(repo/name) != digest:
                raise AssertionError('A frozen benchmark implementation changed: ' + name)
        if sha256(protocol) != signature:
            raise AssertionError('The benchmark selection changed')
        before = time.time()
        records = []
        for script in ['measure_reference_benchmarks.py', 'verify_reference_benchmarks.py']:
            command = [sys.executable, str(repo/'scripts'/script), *common, '--case', case['name'], '--selection', protocol.name]
            log = study/'logs'/(case['name']+'-'+Path(script).stem+'.log')
            with log.open('x') as stream:
                run = subprocess.run(command, cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT)
            records.append(dict(command=command, returncode=run.returncode, log_sha256=sha256(log)))
            if run.returncode:
                return dict(name=case['name'], status='measurement_or_verification_failure', records=records, seconds=time.time()-before)
        verification = study/'verification/reasoning'/(case['name']+'.json')
        verified = json.loads(verification.read_text())
        if verified['status'] != 'passed' or verified['case'] != case:
            raise AssertionError('Benchmark state reconstruction is incomplete')
        return dict(name=case['name'], status='complete', records=records, seconds=time.time()-before,
                    verification_sha256=sha256(verification))

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(execute, case): case['name'] for case in spec['cases']}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as error:
                result = dict(name=futures[future], status='controller_failure', error=repr(error))
            ledger['records'].append(result)
            write_json(marker, ledger)
            print(json.dumps(result), flush=True)
    ledger['status'] = 'complete' if len(ledger['records']) == 2 and all(r['status'] == 'complete' for r in ledger['records']) else 'preserved_failure'
    write_json(marker, ledger)
    if ledger['status'] != 'complete':
        raise RuntimeError('At least one selected benchmark observation requires resolution')
    print('Both full reference benchmark panels are complete and independently reconstructed', flush=True)


if __name__ == '__main__':
    main()
