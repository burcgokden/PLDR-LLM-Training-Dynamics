#!/usr/bin/env python
"""Resume matched models to 8192 updates after the restoration controls finish."""
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
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    root=Path(args.root);study=root/'criticality-study-20260905';repo=Path(__file__).resolve().parents[1]
    ledger=study/'launcher-horizon.json'
    if ledger.exists():raise FileExistsError(ledger)
    jobs=[(f'horizon-h{h}-g1-s{s}',f'{"scan" if h==2 else "variance"}-h{h}-g1-s{s}',h,s) for h in [2,14] for s in [640101,640102,640103,640104]]
    sources=source_manifest();write_json(ledger,dict(status='waiting_for_restoration',jobs=jobs,source_files=sources))
    while True:
        prerequisite=json.loads((study/'launcher-restoration.json').read_text())
        if prerequisite['status']=='execution_failure':raise RuntimeError('A prerequisite restoration run failed')
        if prerequisite['status']=='complete':break
        time.sleep(15)
    write_json(ledger,dict(status='running',jobs=jobs,source_files=sources))
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4')
    def worker(index):
        records=[]
        for name,parent,h,seed in jobs[index::2]:
            command=[sys.executable,str(repo/'scripts/train_criticality.py'),'--root',str(root),'--run-id',name,
                     '--heads',str(h),'--seed',str(seed),'--multiplier','1','--steps','8192','--device',f'cuda:{index}',
                     '--resume',str(study/'runs'/parent/'final-training-state.pt'),'--normalization','fan_in' if h==2 else 'variance',
                     '--protocol','horizon-extension.json']
            started=time.time()
            with (study/'logs'/(name+'.log')).open('x') as log:
                result=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
            record=dict(run_id=name,parent=parent,command=command,returncode=result.returncode,seconds=time.time()-started)
            manifest=study/'runs'/name/'manifest.json'
            if manifest.exists():record['scientific_status']=json.loads(manifest.read_text())['status']
            records.append(record);write_json(study/('ledger-'+name+'.json'),record)
            print(record,flush=True)
            if result.returncode:break
        return records
    with ThreadPoolExecutor(max_workers=2) as pool:records=sum(list(pool.map(worker,range(2))),[])
    success=len(records)==len(jobs) and all(r['returncode']==0 for r in records)
    write_json(ledger,dict(status='complete' if success else 'execution_failure',jobs=jobs,records=records,source_files=sources))
    if not success:raise RuntimeError('Incomplete horizon extension')


if __name__=='__main__':main()
