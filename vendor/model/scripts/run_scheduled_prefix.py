#!/usr/bin/env python
"""Run and independently verify the complete frozen scheduled prefix-state panel."""
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
    selection=study/'protocols/scheduled-prefix-selection.json';spec=json.loads(selection.read_text())
    selected_sha=sha256(selection);marker=study/'launcher-scheduled-prefix.json'
    if marker.exists():raise FileExistsError(marker)
    record=dict(status='running',started_at=datetime.now(timezone.utc).isoformat(),selection_sha256=selected_sha,
        controller_sha256=sha256(Path(__file__)),records=[])
    write_json(marker,record);remaining={c['name']:c for c in spec['cases']}
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4')
    while remaining:
        progress=False
        for name,case in list(remaining.items()):
            if sha256(selection)!=selected_sha:raise AssertionError('The frozen prefix selection changed')
            out=study/'measurements'/name;manifest=out/'manifest.json'
            if out.exists() and not manifest.exists():continue
            if case['kind']!='pretrained':
                parent=Path(case['parent_manifest'])
                if not parent.exists():continue
                pm=json.loads(parent.read_text())
                if pm['status']!='complete':
                    record['records'].append(dict(name=name,status='noncomplete_parent',parent_status=pm['status'],parent_manifest_sha256=sha256(parent)))
                    del remaining[name];progress=True;write_json(marker,record);continue
            for path,digest in spec['inputs_sha256'].items():
                if sha256(path)!=digest:raise AssertionError('A selected prefix input changed: '+path)
            commands=[];before=time.time()
            if not manifest.exists():
                commands.append([sys.executable,str(repo/'scripts/measure_scheduled_prefix.py'),'--root',str(root),'--study',a.study,'--case',name])
            verification=study/'verification/prefix'/(name+'.json')
            if not verification.exists():
                commands.append([sys.executable,str(repo/'scripts/verify_scheduled_prefix.py'),'--root',str(root),'--study',a.study,'--case',name])
            runs=[]
            record['active']=name;write_json(marker,record)
            for index,command in enumerate(commands):
                with (study/'logs'/(name+'-prefix-stage'+str(index)+'.log')).open('x') as log:
                    completed=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
                runs.append(dict(command=command,returncode=completed.returncode,script_sha256=sha256(command[1])))
                if completed.returncode:break
            if len(runs)!=len(commands) or any(r['returncode'] for r in runs):
                record['status']='execution_failure';record['records'].append(dict(name=name,status='execution_failure',commands=runs));write_json(marker,record)
                raise RuntimeError('Resolve the preserved prefix execution failure before continuing')
            m=json.loads(manifest.read_text());v=json.loads(verification.read_text())
            if m['status']!='complete' or v['status']!='passed' or m['case']!=case or v['case']!=case:
                raise AssertionError('An adopted prefix result is incomplete or incorrectly selected')
            row=dict(name=name,status='complete',manifest_sha256=sha256(manifest),verification_sha256=sha256(verification),commands=runs,seconds=time.time()-before)
            record['records'].append(row);record.pop('active',None);write_json(marker,record)
            del remaining[name];progress=True;print(json.dumps(row),flush=True)
        if remaining and not progress:
            record['status']='waiting_selected_states';write_json(marker,record);time.sleep(30)
        elif remaining:
            record['status']='running';write_json(marker,record)
    record['status']='complete' if all(r['status']=='complete' for r in record['records']) else 'complete_with_noncomplete_parents'
    write_json(marker,record);print(record['status'],len(record['records']),'prefix states',flush=True)


if __name__=='__main__':main()
