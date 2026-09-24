#!/usr/bin/env python3
"""Run the fixed avalanche assay on two GPU workers, one native job per GPU."""
from companion_paths import child_pythonpath, dispatch_worker, validate_worker_cli
from companion_paths import legacy_path
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from model_rg.provenance import write_json
ROOT=Path(legacy_path('/pldr-data/model/potential-avalanche-20260913'))
REPO=Path(__file__).resolve().parents[1]

def worker(device,jobs):
    records=[]
    for job in jobs:
        rid=job['run_id'];out=ROOT/'runs'/rid
        if (out/'manifest.json').exists():
            meta=json.loads((out/'manifest.json').read_text())
            if meta['status']=='complete':
                records.append(dict(run_id=rid,status='already_complete'));continue
        log=ROOT/'logs'/f'{rid}.log';log.parent.mkdir(exist_ok=True)
        env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        with log.open('w') as f:
            r=dispatch_worker([sys.executable,str(REPO/'scripts/train_potential_avalanches.py'),
                '--study',str(ROOT),'--run-id',rid,'--device',f'cuda:{device}'],cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
        records.append(dict(run_id=rid,exit_code=r.returncode))
        print(json.dumps(dict(device=device,**records[-1])),flush=True)
        write_json(ROOT/f'worker-{device}.json',records)
        if r.returncode:raise RuntimeError('Training failed; retained diagnostics: '+rid)
    return records

if __name__ == '__main__':
    validate_worker_cli(__file__, target=Path(__file__).with_name('train_potential_avalanches.py'), worker_action=False)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__ + '\nUse --validate-worker for a corpus-free child check.')
    parser.add_argument('--study', type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.study.resolve()
    spec=json.loads((ROOT/'protocols/training.json').read_text())
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,i,spec['jobs'][i::2]) for i in range(2)]
        for f in futures:f.result()
