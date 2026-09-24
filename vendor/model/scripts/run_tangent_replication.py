#!/usr/bin/env python
"""Execute each recorded CPU augmented-response case and preserve every outcome."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from model_rg.provenance import sha256, write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    args=parser.parse_args();root=Path(args.root);study=root/'criticality-dynamics-20260906'
    protocol=study/'protocols/tangent-replication.json';spec=json.loads(protocol.read_text())
    ledger=study/'tangent-replication.json'
    if ledger.exists():raise FileExistsError(ledger)
    records=[]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4')
    for case in spec['cases']:
        out=study/'analysis'/case['name']
        command=[sys.executable,'scripts/measure_adam_tangent.py','--root',str(root),
                 '--checkpoint',case['checkpoint'],'--output',str(out),
                 '--direction',case['direction'],'--amplitudes',case['amplitudes'],
                 '--contexts',str(case['contexts'])]
        started=time.time()
        with (study/'logs'/(case['name']+'.log')).open('x') as log:
            result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        record=dict(name=case['name'],command=command,returncode=result.returncode,seconds=time.time()-started)
        if (out/'manifest.json').exists():record['manifest_sha256']=sha256(out/'manifest.json')
        records.append(record);print(json.dumps(record),flush=True)
        write_json(ledger,dict(status='running',protocol_sha256=sha256(protocol),records=records))
    write_json(ledger,dict(status='complete' if all(r['returncode']==0 for r in records) else 'complete_with_failed_controls',
                          protocol_sha256=sha256(protocol),records=records))


if __name__=='__main__':main()
