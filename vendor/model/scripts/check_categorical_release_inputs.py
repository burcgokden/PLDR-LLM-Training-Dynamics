#!/usr/bin/env python3
"""Refresh positive raw reconstructions under the maintained finite-value guards."""
from companion_paths import child_pythonpath
from companion_paths import legacy_path
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from model_rg.provenance import sha256,write_json

REPO=Path(__file__).resolve().parents[1]
ROOT=Path(legacy_path('/pldr-data/model'))


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    jobs=[]
    for family in ['coarse','refinement']:
        root=ROOT/f'critical-onepass-{family}-20260914'
        jobs.append((family+'-scalar',['verify_critical_onepass.py','--study',str(root),'--analysis',str(root/'analysis.json'),'--source-root',str(REPO/'internal/revision27/retained-sources'/family),'--observations-only']))
        collective=ROOT/f'critical-collective-{family}-20260914'
        jobs.append((family+'-collective',['verify_critical_collectives.py','--study',str(collective),'--analysis',str(ROOT/'singlepass-uncertainty-stable-20260915'/family/'collective-analysis.json'),'--observations-only']))
    jobs.append(('categorical',['verify_categorical_visibility.py','--study',str(ROOT/'categorical-visibility-20260915')]))
    jobs.append(('metric',['verify_metric_studies.py','--metric-root',str(ROOT/'finite-metric-20260914'),
        '--matched-analysis',str(ROOT/'matched-onepass-20260914/analysis.json'),
        '--matched-refinement',str(ROOT/'matched-onepass-20260914/refinement.json'),'--source-root',str(REPO/'internal/revision27/retained-sources/metric')]))
    def execute(item):
        label,args=item;command=[sys.executable,str(REPO/'scripts'/args[0]),*args[1:],'--output',str(output/(label+'.json'))]
        started=time.monotonic()
        with (output/(label+'.log')).open('x') as log:
            result=subprocess.run(command,cwd=REPO,env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT)
        record=dict(label=label,command=command,returncode=result.returncode,seconds=time.monotonic()-started,
            verifier_sha256=sha256(REPO/'scripts'/args[0]),log_sha256=sha256(output/(label+'.log')))
        write_json(output/(label+'-execution.json'),record)
        print(label,result.returncode,flush=True)
        return record
    with ThreadPoolExecutor(max_workers=2) as pool:records=list(pool.map(execute,jobs))
    from verify_optimizer_compact import verify
    optimizer=verify(REPO/'manuscript');write_json(output/'optimizer.json',dict(status='passed',**optimizer,
        verifier_sha256=sha256(REPO/'scripts/verify_optimizer_compact.py')))
    write_json(output/'verification.json',dict(status='passed' if all(r['returncode']==0 for r in records) else 'failed',
        reconstructions=records,source_sha256=sha256(__file__),support_sha256={str(REPO/'scripts'/name):sha256(REPO/'scripts'/name) for name in ['numerical_validation.py','numerical_claims.py']},
        native_forward_calls=0,training_updates=0,scope='All numerical observations and qualification replay states are checked; scientific checkpoint digests are retained metadata in the explicit observation-only routes.'))
    if any(r['returncode'] for r in records):raise ValueError('A retained raw reconstruction failed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output.resolve())
