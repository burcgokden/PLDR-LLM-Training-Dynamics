#!/usr/bin/env python
"""Measure every selected saved state as its complete training parent becomes available."""
from companion_paths import child_pythonpath
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from model_rg.provenance import sha256, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='critical-scaling-20260906')
    p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();study=Path(a.root)/a.study;repo=Path(__file__).resolve().parents[1]
    selection=study/'protocols/target-collective-selection.json'
    marker=study/'launcher-target-collectives.json'
    if selection.exists() or marker.exists():raise FileExistsError('The observation selection is immutable')
    jobs=[];inputs={}
    for name in ['clock-holdout','diffusive-size-map','environment-factorial','zero-horizon']:
        path=study/'protocols'/f'{name}.json';spec=json.loads(path.read_text());inputs[str(path)]=sha256(path)
        for case in spec['jobs']:
            steps=([8192,16384,32768] if name in ['clock-holdout','zero-horizon'] else [case['steps']])
            jobs.append(dict(run_id=case['run_id'],steps=steps,heads=case['heads'],seed=case['seed'],
                             multiplier=case['multiplier'],protocol=name+'.json'))
    jobs.extend(dict(run_id=f'extended-h{n}-g1-s{s}',steps=[65536,98304,131072],heads=n,seed=s,
                     multiplier=1,protocol='extended-horizon.json') for s in range(640101,640105) for n in [4,14])
    if len(jobs)!=60 or sum(len(x['steps']) for x in jobs)!=108:raise AssertionError('Target observation count changed')
    write_json(selection,dict(schema='future-parent-observation-selection-v1',
        frozen_at=datetime.now(timezone.utc).isoformat(),jobs=jobs,training_protocol_sha256=inputs,
        selection='Every target trajectory: all three specified clock, zero-boundary and extended horizons; every joint-clock endpoint; all factorial endpoints at 8192.',
        measurement=dict(rows=list(range(512,1024)),precisions=['float32','float64'],evaluation_matrices=True,batch_size=32),
        conditioning='The same 512 held-out contexts and separate 64 calibration contexts. These measurements add no training initialization identities.',
        qualification='All outcomes retained. A parent with a numerical failure is recorded explicitly and not replaced by another initialization.'))
    record=dict(status='running',arguments=vars(a),selection_sha256=sha256(selection),records=[])
    write_json(marker,record)
    remaining={job['run_id']:job for job in jobs};running={}
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4')

    def execute(job,parent):
        metadata=json.loads(parent.read_text())
        if metadata['status']!='complete':
            return dict(run_id=job['run_id'],status='noncomplete_parent',parent_status=metadata['status'],
                        parent_manifest_sha256=sha256(parent))
        condition=metadata['arguments']
        for key in ['heads','seed','multiplier']:
            if condition[key]!=job[key]:raise AssertionError('Selected training condition changed')
        cases=[]
        for step in job['steps']:
            saved=metadata['saved_states'][str(step)]
            cases.append(dict(name=f"cpu-{job['run_id']}-t{step}",step=step,
                **{k:condition[k] for k in ['heads','seed','multiplier','shared_seed','stream_seed']},
                state=str(parent.parent/saved['filename']),state_sha256=saved['sha256'],parent_manifest=str(parent)))
        protocol=study/'protocols'/f"observations-{job['run_id']}.json"
        if protocol.exists():raise FileExistsError(protocol)
        write_json(protocol,dict(schema='frozen-collective-observation-protocol-v1',
            frozen_at=datetime.now(timezone.utc).isoformat(),rows=list(range(512,1024)),
            batch_size=32,precisions=['float32','float64'],evaluation_matrices=True,cases=cases,
            selection=str(selection),selection_sha256=sha256(selection)))
        command=[sys.executable,str(repo/'scripts/measure_scaling_checkpoints.py'),'--root',a.root,
                 '--study',a.study,'--protocol',protocol.name,'--device','cpu','--threads','4']
        with (study/'logs'/f"observations-{job['run_id']}.log").open('x') as log:
            result=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
        return dict(run_id=job['run_id'],status='complete' if result.returncode==0 else 'measurement_failure',
                    returncode=result.returncode,protocol_sha256=sha256(protocol),cases=len(cases))

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        while remaining or running:
            for run_id,job in list(remaining.items()):
                if len(running)>=a.workers:break
                parent=study/'runs'/run_id/'manifest.json'
                if parent.exists():
                    running[pool.submit(execute,job,parent)]=run_id;del remaining[run_id]
            for future,run_id in list(running.items()):
                if not future.done():continue
                try:result=future.result()
                except Exception as exc:result=dict(run_id=run_id,status='controller_failure',error=repr(exc))
                record['records'].append(result);write_json(marker,record)
                print(json.dumps(result),flush=True);del running[future]
            if remaining or running:time.sleep(30)
    record['status']='complete' if all(x['status']=='complete' for x in record['records']) else 'complete_with_failures'
    write_json(marker,record)


if __name__=='__main__':main()
