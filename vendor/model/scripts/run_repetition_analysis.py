#!/usr/bin/env python
"""Complete each paired panel and its independent check as inputs become ready."""
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
    p.add_argument('--study',default='scheduled-training-feasible-20260908');a=p.parse_args()
    root=Path(a.root).resolve();study=root/a.study;repo=Path(__file__).resolve().parents[1]
    selection=study/'protocols/repetition-analysis-selection.json';spec=json.loads(selection.read_text());signature=sha256(selection)
    marker=study/'launcher-repetition-analysis.json'
    if marker.exists():raise FileExistsError(marker)
    logdir=study/'logs/repetition-analysis';logdir.mkdir(parents=True,exist_ok=False)
    record=dict(status='waiting',selection_sha256=signature,started_at=datetime.now(timezone.utc).isoformat(),records=[])
    write_json(marker,record)
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4')
    for panel in ['reference_schedule','constant','complete']:
        pairs=[q for q in spec['pairs'] if panel=='complete' or q['family']==panel]
        required={q[side]['verification'] for q in pairs for side in ['old','new']}
        required.update(q['new']['native_verification'] for q in pairs)
        required.update(q[side]['risk_verification'] for q in pairs for side in ['old','new'] if q[side].get('risk_verification'))
        if panel=='complete':required.add(str(study/'launcher-onepass-observations.json'))
        while True:
            ready=True
            for path in required:
                pth=Path(path)
                if not pth.exists():ready=False;continue
                status=json.loads(pth.read_text())['status']
                if 'fail' in status:raise RuntimeError('Resolve the preserved paired-data prerequisite failure: '+path)
                if status not in ['complete','passed']:ready=False
            for q in pairs:
                parent=Path(q['new']['case']['parent_manifest'])
                if parent.exists() and json.loads(parent.read_text())['status']!='complete':
                    raise RuntimeError('A selected paired native trajectory did not complete: '+str(parent))
            if ready:break
            time.sleep(30)
        if sha256(selection)!=signature:raise AssertionError('The paired-data selection changed')
        row=dict(panel=panel,commands=[]);before=time.time();record['status']='analyzing_'+panel;write_json(marker,record)
        for script in ['analyze_repetition_comparison.py','verify_repetition_comparison.py']:
            command=[sys.executable,str(repo/'scripts'/script),'--root',str(root),'--study',a.study,'--panel',panel]
            logpath=logdir/(panel+'-'+script+'.log')
            with logpath.open('x') as log:r=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
            row['commands'].append(dict(command=command,returncode=r.returncode,log_sha256=sha256(logpath)))
            if r.returncode:
                row['status']='failed';record['records'].append(row);record['status']='analysis_failed';write_json(marker,record)
                raise RuntimeError('Preserved paired-data analysis failure: '+str(logpath))
        proof=study/'verification'/('repetition-'+panel+'.json')
        if json.loads(proof.read_text())['status']!='passed':raise AssertionError('Independent paired-data proof missing')
        row.update(status='complete',seconds=time.time()-before,verification_sha256=sha256(proof))
        record['records'].append(row);record['status']='waiting';write_json(marker,record);print(panel,'complete',flush=True)
    record['status']='complete';write_json(marker,record)


if __name__=='__main__':main()
