#!/usr/bin/env python
"""Verify immutable native scheduled paths as they finish, then combine all forty."""
from companion_paths import child_pythonpath
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from model_rg.provenance import sha256,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-20260908');a=p.parse_args()
    root=Path(a.root).resolve();study=root/a.study;repo=Path(__file__).resolve().parents[1]
    selection=study/'protocols/regime-training-selection.json';spec=json.loads(selection.read_text());selection_hash=sha256(selection)
    marker=study/'launcher-training-checks.json'
    if marker.exists():raise FileExistsError(marker)
    source_names=['scripts/watch_scheduled_training_checks.py','scripts/verify_scheduled_training_case.py',
        'scripts/verify_scheduled_raw.py','scripts/verify_scaling_raw.py','src/model_rg/provenance.py']
    source_hashes={n:sha256(repo/n) for n in source_names}
    ledger=dict(status='waiting_native_trajectories',started_at=datetime.now(timezone.utc).isoformat(),
        selection_sha256=selection_hash,source_sha256=source_hashes,records=[])
    write_json(marker,ledger);pending={j['run_id']:j for j in spec['jobs']}
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4')
    while pending:
        progressed=False
        for name,job in list(pending.items()):
            parent=study/'runs'/name/'manifest.json'
            if not parent.exists():continue
            metadata=json.loads(parent.read_text())
            if metadata['status']!='complete':
                ledger['status']='noncomplete_native_outcome';ledger['native_outcome']=dict(run_id=name,status=metadata['status'],manifest_sha256=sha256(parent));write_json(marker,ledger)
                raise RuntimeError('A selected native outcome requires scientific interpretation before a complete evidence gate')
            if sha256(selection)!=selection_hash:raise AssertionError('The selected native panel changed')
            for path,digest in source_hashes.items():
                if sha256(repo/path)!=digest:raise AssertionError('An active independent native checker changed')
            output=study/'verification/training'/(name+'.json');commands=[]
            if not output.exists():
                command=[sys.executable,str(repo/'scripts/verify_scheduled_training_case.py'),'--root',str(root),'--study',a.study,'--run-id',name]
                ledger['status']='verifying';ledger['active']=name;write_json(marker,ledger)
                with (study/'logs'/(name+'-independent-training-check.log')).open('x') as log:
                    result=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
                commands.append(dict(command=command,returncode=result.returncode))
                if result.returncode:
                    ledger['status']='verification_failure';ledger['failure']=dict(run_id=name,commands=commands);write_json(marker,ledger)
                    raise RuntimeError('Resolve the preserved native reconstruction failure')
            report=json.loads(output.read_text())
            if report['status']!='complete' or report['trajectory']['run_id']!=name or report['selection_sha256']!=selection_hash:
                raise AssertionError('An individual native reconstruction is incomplete or changed')
            ledger['records'].append(dict(run_id=name,verification_sha256=sha256(output),commands=commands))
            ledger.pop('active',None);write_json(marker,ledger);del pending[name];progressed=True
            print('Verified native scheduled path',name,flush=True)
        if pending and not progressed:
            ledger['status']='waiting_native_trajectories';write_json(marker,ledger);time.sleep(30)
    files={};records=[];individual=[]
    for job in spec['jobs']:
        path=study/'verification/training'/(job['run_id']+'.json');report=json.loads(path.read_text())
        records.append(report['trajectory']);individual.append(dict(path=str(path),sha256=sha256(path)))
        for name,digest in report['verified_files'].items():
            if name in files and files[name]!=digest:raise AssertionError('Conflicting native evidence bindings')
            files[name]=digest
        for name,digest in report['verifier_sources'].items():
            if sha256(repo/name)!=digest:raise AssertionError('A native reconstruction implementation changed')
    if len(records)!=40 or sum(r['updates'] for r in records)!=spec['selected_updates']:
        raise AssertionError('A selected scientific trajectory or update was omitted')
    output=study/'verification/training-raw.json'
    if output.exists():raise FileExistsError(output)
    write_json(output,dict(schema='scheduled-raw-reconstruction-v1',status='complete',kind='training',
        trajectories=records,artifacts=40,native_updates=sum(r['updates'] for r in records),
        selection_sha256=selection_hash,verified_files=files,verifier_sources=source_hashes,
        individual_reconstructions=individual,
        scope='Complete aggregation of forty independent per-trajectory reconstructions using the scheduled raw verifier. Every selected sampling history, saved native optimizer/scheduler state and recorded observation path is checked. Aggregation does not claim a replay of all native training updates.'))
    ledger['status']='complete';ledger['combined_verification_sha256']=sha256(output);write_json(marker,ledger)
    print('Complete independent reconstruction of all40 scheduled native trajectories',flush=True)


if __name__=='__main__':main()
