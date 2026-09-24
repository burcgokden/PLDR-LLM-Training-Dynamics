#!/usr/bin/env python
"""Wait for all selected benchmark states, then analyze and verify every outcome."""
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
    p.add_argument('--study', default='scheduled-training-20260908'); args = p.parse_args()
    root = Path(args.root).resolve(); study = root/args.study; repo = Path(__file__).resolve().parents[1]
    marker = study/'launcher-reference-benchmark-analysis.json'
    if marker.exists(): raise FileExistsError(marker)
    names = ['scripts/finish_reference_benchmarks.py', 'scripts/analyze_reference_benchmarks.py',
        'scripts/verify_reference_benchmark_summary.py', 'src/model_rg/controlled.py', 'src/model_rg/provenance.py']
    sources = {name: sha256(repo/name) for name in names}
    selection = study/'protocols/reference-benchmark-selection.json'; signature = sha256(selection)
    ledger = dict(status='waiting_verified_benchmarks', started_at=datetime.now(timezone.utc).isoformat(),
        source_files=sources, selection_sha256=signature, records=[]); write_json(marker, ledger)
    controller = study/'launcher-reference-benchmarks.json'
    while True:
        data = json.loads(controller.read_text())
        if data['status'] == 'complete': break
        if data['status'] not in ['checking_cohort', 'measuring']:
            ledger['status'] = 'preserved_measurement_failure'; write_json(marker, ledger)
            raise RuntimeError('Resolve the preserved benchmark measurement failure')
        time.sleep(30)
    if len(data['records']) != 2 or any(r['status'] != 'complete' for r in data['records']): raise AssertionError('A released benchmark state is missing')
    env = dict(os.environ, PYTHONPATH=child_pythonpath("model"), PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='4')
    for script in ['analyze_reference_benchmarks.py', 'verify_reference_benchmark_summary.py']:
        for name, digest in sources.items():
            if sha256(repo/name) != digest: raise AssertionError('The selected benchmark analysis implementation changed')
        if sha256(selection) != signature: raise AssertionError('The complete benchmark selection changed')
        ledger.update(status='analyzing' if script.startswith('analyze') else 'verifying', active=script); write_json(marker, ledger)
        command = [sys.executable, str(repo/'scripts'/script), '--root', str(root), '--study', args.study]
        log = study/'logs'/('complete-'+Path(script).stem+'.log')
        with log.open('x') as stream: run = subprocess.run(command, cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT)
        ledger['records'].append(dict(command=command, returncode=run.returncode, log_sha256=sha256(log)))
        if run.returncode:
            ledger['status'] = 'preserved_analysis_failure'; write_json(marker, ledger)
            raise RuntimeError('Resolve the preserved full benchmark analysis failure')
    proof = study/'verification/reference-benchmark-summary.json'
    result = json.loads(proof.read_text())
    if result['status'] != 'passed' or result['candidate_margin_comparisons'] != 113536: raise AssertionError('Incomplete full ARC reconstruction')
    ledger.pop('active', None); ledger.update(status='complete', verification_sha256=sha256(proof)); write_json(marker, ledger)
    print('All selected full reference benchmark outcomes analyzed and independently verified', flush=True)


if __name__ == '__main__': main()
