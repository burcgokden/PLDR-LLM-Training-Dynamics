#!/usr/bin/env python
"""Bind completed parents for matched observations and selected long horizons."""
from companion_paths import child_pythonpath
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from model_rg.provenance import sha256, write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--study',default='critical-scaling-20260906')
    a=p.parse_args()
    study=Path(a.root)/a.study
    repo=Path(__file__).resolve().parents[1]
    marker=study/'followup-preparation.json'
    if marker.exists():
        raise FileExistsError(marker)
    record=dict(status='waiting_for_main_training',arguments=vars(a),started_at=datetime.now(timezone.utc).isoformat(),
                observation_selection='All twenty four-identity 32768 endpoints, all sixteen additional 16384 endpoints, and four new N24 8192 parents.',
                long_selection='N4 and N14, all four original identities, through 131072, with 65536 and 98304 saved states.',
                rationale='Resolve the surviving common and predictive sectors beyond row concentration, including the slower nominal generator decay clock. The long panel does not treat unobserved spectator relaxation as criticality.')
    write_json(marker,record)
    predecessor=study/'launcher-main-size-time-execution.json'
    while True:
        state=json.loads(predecessor.read_text())['status']
        if state=='complete':break
        if state!='running':raise RuntimeError('Main execution needs resolution: '+state)
        time.sleep(30)
    parents={}
    for path in sorted((study/'runs').glob('*/manifest.json')):
        m=json.loads(path.read_text());c=m.get('arguments',{})
        if (m.get('schema')=='native-size-time-training-v1' and m['status']=='complete' and
            c.get('multiplier')==1 and c.get('shared_seed')==640011 and c.get('stream_seed')==640001):
            for step,saved in m['saved_states'].items():
                key=(c['heads'],c['seed'],int(step))
                if key in parents:raise AssertionError('Duplicate continued identity')
                parents[key]=(path,saved,c)
    selected=[(n,seed,32768) for n in [2,4,8,14,24] for seed in range(640101,640105)]
    selected += [(n,seed,16384) for n in [2,4,8,24] for seed in range(640105,640109)]
    selected += [(24,seed,8192) for seed in range(640105,640109)]
    cases=[]
    for n,seed,step in selected:
        path,saved,c=parents[n,seed,step]
        cases.append(dict(name=f'cpu-h{n}-g1-t{step}-s{seed}',heads=n,seed=seed,step=step,
            multiplier=1,shared_seed=640011,stream_seed=640001,parent_manifest=str(path),
            state=str(path.parent/saved['filename']),state_sha256=saved['sha256']))
    observation=study/'protocols/late-collectives.json'
    write_json(observation,dict(schema='frozen-collective-observation-protocol-v1',
        frozen_at=datetime.now(timezone.utc).isoformat(),rows=list(range(512,1024)),batch_size=32,
        precisions=['float32','float64'],evaluation_matrices=True,cases=cases,
        purpose='Complete matched size-time panels on one CPU platform, with paired arithmetic and full mean matrices on all 512 held-out contexts. Calibration matrices retain their separate 64-context law.',
        independent_unit='Training initialization; repeated arithmetic and contexts add no training identities.',
        parent_execution_sha256=sha256(predecessor)))
    jobs=[]
    for seed in range(640101,640105):
        for n in [4,14]:
            path,saved,c=parents[n,seed,32768]
            jobs.append(dict(run_id=f'extended-h{n}-g1-s{seed}',heads=n,seed=seed,multiplier=1,
                steps=131072,microbatch=32,save_steps='65536,98304',resume=str(path.parent/saved['filename']),
                parent_sha256=saved['sha256']))
    extended=study/'protocols/extended-horizon.json'
    write_json(extended,dict(schema='native-critical-scaling-protocol-v1',frozen_at=datetime.now(timezone.utc).isoformat(),
        shared_seed=640011,stream_seed=640001,probe_every=64,normalization='variance',jobs=jobs,
        rationale=record['rationale'],selection=record['long_selection'],
        effective_batch_size=32,decoders=5,head_dimension=64,
        primary='Paired row/common/predictive laws and late time response, including every declared initialization and saved horizon.',
        interpretation='A continuation adds observation time, not independent initialization. An unchanging conditional checkpoint is not the definition of averaged stationarity.',
        parent_execution_sha256=sha256(predecessor)))
    record.update(status='protocols_bound_observations_running',observation_protocol_sha256=sha256(observation),
                  extended_protocol_sha256=sha256(extended),observation_cases=len(cases),extended_jobs=len(jobs))
    write_json(marker,record)
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4')
    command=[sys.executable,str(repo/'scripts/measure_scaling_checkpoints.py'),'--root',a.root,'--study',a.study,
             '--protocol',observation.name,'--device','cpu','--threads','4']
    with (study/'logs'/'late-collectives.log').open('x') as log:
        result=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
    record.update(status='complete' if result.returncode==0 else 'observation_failure',returncode=result.returncode)
    write_json(marker,record)
    if result.returncode:raise RuntimeError('Late observations failed')


if __name__=='__main__':main()
