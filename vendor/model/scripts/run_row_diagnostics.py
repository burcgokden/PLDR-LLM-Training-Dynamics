#!/usr/bin/env python
"""Execute an immutable row diagnostic protocol sequentially on the CPU."""
from companion_paths import child_pythonpath
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from model_rg.provenance import sha256, source_manifest, write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--protocol',required=True,choices=['row-transport.json','row-transport-transfer.json','row-projection.json','row-adjoint.json','row-adjoint-source-control.json'])
    args=parser.parse_args();root=Path(args.root).resolve();study=root/'criticality-dynamics-20260906';repo=Path(__file__).resolve().parents[1]
    protocol=study/'protocols'/args.protocol;spec=json.loads(protocol.read_text());label=protocol.stem;ledger=study/(label+'.json')
    if ledger.exists():raise FileExistsError('A diagnostic ledger already exists: '+str(ledger))
    names=[case.get('output_name',case['name']) for case in spec['cases']]
    if len(names)!=len(set(names)):raise AssertionError('Repeated diagnostic identifier')
    for name in names:
        if (study/'analysis'/name).exists() or (study/'logs'/(name+'.log')).exists():raise FileExistsError(name)
    records=[];sources=source_manifest();signature=sha256(protocol)
    def save(status):write_json(ledger,dict(status=status,records=records,protocol_sha256=signature,source_files=sources))
    save('running');env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4')
    for case,name in zip(spec['cases'],names,strict=True):
        script=('measure_row_projection.py' if label=='row-projection' else ('measure_row_adjoint.py' if label.startswith('row-adjoint') else 'measure_row_transport.py'))
        command=[sys.executable,str(repo/'scripts'/script),'--root',str(root),'--case',case['name']]
        if label!='row-projection':command+=['--protocol',args.protocol]
        started=time.time()
        with (study/'logs'/(name+'.log')).open('x') as log:
            completed=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
        record=dict(name=name,command=command,returncode=completed.returncode,seconds=time.time()-started)
        records.append(record);save('execution_failure' if completed.returncode else 'running');print(json.dumps(record),flush=True)
        if completed.returncode:raise RuntimeError('A diagnostic needs explicit outcome handling: '+name)
    save('complete')


if __name__=='__main__':main()
