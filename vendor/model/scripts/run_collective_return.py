#!/usr/bin/env python
"""Run the four recorded component-return parents on two GPUs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from model_rg.provenance import sha256, write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    study=Path(args.root)/'criticality-dynamics-20260906';protocol=study/'protocols/collective-return.json'
    spec=json.loads(protocol.read_text());ledger=study/'collective-return.json'
    if ledger.exists() or any((study/'runs'/case['name']).exists() for case in spec['cases']):raise FileExistsError('A component-return outcome already exists')
    def worker(index):
        records=[]
        for case in spec['cases'][index::2]:
            command=[sys.executable,'scripts/measure_collective_return.py','--root',args.root,'--case',case['name'],'--device',f'cuda:{index}']
            with (study/'logs'/(case['name']+'.log')).open('x') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            records.append(dict(name=case['name'],command=command,returncode=result.returncode));print(records[-1],flush=True)
        return records
    with ThreadPoolExecutor(2) as pool:records=sum(list(pool.map(worker,range(2))),[])
    write_json(ledger,dict(status='complete' if all(r['returncode']==0 for r in records) else 'execution_failure',protocol_sha256=sha256(protocol),records=records))


if __name__=='__main__':main()
