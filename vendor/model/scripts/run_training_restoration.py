#!/usr/bin/env python
"""Run the declared restoration controls after the two training scans finish."""
from companion_paths import child_pythonpath
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from model_rg.provenance import source_manifest, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root); study = root/'criticality-study-20260905'
    repo = Path(__file__).resolve().parents[1]
    ledger = study/'launcher-restoration.json'
    if ledger.exists():
        raise FileExistsError(ledger)
    jobs = [(f'restoration-h{h}-g1-s{s}', f'{"scan" if h==2 else "variance"}-h{h}-g1-s{s}')
            for h in [2,14] for s in [640101,640102,640103,640104]]
    sources = source_manifest()
    write_json(ledger, dict(status='waiting_for_training', jobs=jobs, source_files=sources))
    # These background waits do not block the interactive research session.
    while True:
        scans = [json.loads((study/f'launcher-{name}.json').read_text()) for name in ['scan','variance']]
        if any(s['status']=='execution_failure' for s in scans):
            raise RuntimeError('A prerequisite scan failed; inspect the retained execution ledger')
        if all(s['status']=='complete' for s in scans):
            break
        time.sleep(15)
    write_json(ledger, dict(status='running', jobs=jobs, source_files=sources))
    env = dict(os.environ, PYTHONPATH=child_pythonpath("model"), PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='4')
    def worker(index):
        records=[]
        for name,parent in jobs[index::2]:
            command=[sys.executable,str(repo/'scripts/measure_training_restoration.py'),'--root',str(root),
                     '--parent',parent,'--run-id',name,'--device',f'cuda:{index}']
            started=time.time()
            with (study/'logs'/(name+'.log')).open('x') as log:
                outcome=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
            record=dict(run_id=name,parent=parent,command=command,returncode=outcome.returncode,seconds=time.time()-started)
            manifest=study/'runs'/name/'manifest.json'
            if manifest.exists():record['scientific_status']=json.loads(manifest.read_text())['status']
            records.append(record);write_json(study/('ledger-'+name+'.json'),record)
            print(record,flush=True)
            if outcome.returncode:break
        return records
    with ThreadPoolExecutor(max_workers=2) as pool:
        records=sum(list(pool.map(worker,range(2))),[])
    success=len(records)==len(jobs) and all(r['returncode']==0 for r in records)
    write_json(ledger,dict(status='complete' if success else 'execution_failure',jobs=jobs,records=records,source_files=sources))
    if not success:raise RuntimeError('Incomplete restoration experiment')


if __name__=='__main__':
    main()
