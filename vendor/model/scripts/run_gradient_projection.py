#!/usr/bin/env python
"""Execute the fixed matched-gradient design with two bounded CPU workers."""
from companion_paths import child_pythonpath
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from model_rg.provenance import sha256, source_manifest, write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    root=Path(args.root).resolve();study=root/'criticality-dynamics-20260906';repo=Path(__file__).resolve().parents[1]
    protocol=study/'protocols/row-gradient-projection.json';spec=json.loads(protocol.read_text());ledger=study/'row-gradient-projection.json'
    if ledger.exists():raise FileExistsError(ledger)
    names=[c['name'] for c in spec['cases']]
    if len(names)!=48 or len(set(names))!=48 or spec['maximum_workers']!=2:raise AssertionError('CPU diagnostic design changed')
    for name in names:
        if (study/'analysis'/name).exists() or (study/'logs'/(name+'.log')).exists():raise FileExistsError(name)
    source=source_manifest();signature=sha256(protocol);records=[]
    def save(status):write_json(ledger,dict(status=status,protocol_sha256=signature,source_files=source,records=records))
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
    def run(case):
        command=[sys.executable,str(repo/'scripts/measure_row_gradient_projection.py'),'--root',str(root),'--case',case['name']];started=time.time()
        with (study/'logs'/(case['name']+'.log')).open('x') as log:
            result=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
        return dict(name=case['name'],command=command,returncode=result.returncode,seconds=time.time()-started)
    save('running')
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run,c) for c in spec['cases']]
        for future in as_completed(futures):
            record=future.result();records.append(record);save('running');print(json.dumps(record),flush=True)
    failed=any(r['returncode'] for r in records);save('execution_failure' if failed else 'complete')
    if failed:raise RuntimeError('Every failed gradient diagnostic requires explicit outcome handling')


if __name__=='__main__':main()
