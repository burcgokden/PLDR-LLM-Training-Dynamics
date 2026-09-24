#!/usr/bin/env python
"""Freeze and observe all retained schedule states for matched mechanism checks."""
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
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-feasible-20260908'); a=p.parse_args()
    root=Path(a.root).resolve(); mainstudy=root/a.study; repo=Path(__file__).resolve().parents[1]
    study=mainstudy/'repeated-observations'; study.mkdir(exist_ok=False)
    for name in ['protocols','logs','measurements','verification']: (study/name).mkdir()
    source=mainstudy/'protocols/repeated-risk-comparison.json'; saved=json.loads(source.read_text())
    selection=study/'protocols/repeated-observation-selection.json'; cases=[]
    for c in saved['cases']:
        v=dict(c,name='cpu-'+c['name'],parent_manifest=c['sealed_manifest'])
        v.pop('sealed_manifest'); cases.append(v)
    names=['scripts/run_repeated_comparison_mechanisms.py','scripts/verify_repeated_collectives.py',
        'scripts/measure_scaling_checkpoints.py','scripts/verify_scaling_raw.py',
        'scripts/measure_scheduled_prefix.py','scripts/verify_scheduled_prefix.py',
        'scripts/measure_reasoning_mechanism.py','scripts/verify_reasoning_mechanism.py',
        'src/model_rg/training.py','src/model_rg/native.py','src/model_rg/scaling.py',
        'src/model_rg/criticality.py','src/model_rg/controlled.py','src/model_rg/provenance.py',
        'src/model_rg/precision.py','src/model_rg/inference_interventions.py']
    sources={name:sha256(repo/name) for name in names}
    spec=dict(schema='retained-repeated-observation-selection-v1',frozen_at=datetime.now(timezone.utc).isoformat(),
        cases=cases,producer_sources=sources,inputs_sha256={str(source):sha256(source)},
        rows=list(range(512,1024)),batch_size=32,precisions=['float32','float64'],evaluation_matrices=True,
        scope='All four retained schedule checkpoints, with fixed held-out collective, proper-prefix and task cohorts. '
        'One initialization per recipe; checkpoints along a trajectory are not independent replicates. '
        'No additional training identities or completed source trajectories.')
    write_json(selection,spec)
    for filename,label in [('scheduled-prefix-selection.json','prefix'),('reasoning-mechanism-selection.json','reasoning')]:
        origin=mainstudy/'protocols'/filename; q=json.loads(origin.read_text()); chosen=[]
        for c in cases:
            chosen.append(dict(c,name=label+'-'+c['name'].removeprefix('cpu-'),kind='trained',
                source=str(root/'assets/PLDR-LLM-v51-SOC-110M-1')))
        q.update(cases=chosen,frozen_at=spec['frozen_at'],scope=spec['scope'],source_selection_sha256=sha256(origin))
        q['inputs_sha256'].update({str(selection):sha256(selection),str(source):sha256(source)})
        write_json(study/'protocols'/filename,q)
    marker=mainstudy/'launcher-repeated-mechanisms.json'
    if marker.exists(): raise FileExistsError(marker)
    record=dict(status='observing',selection_sha256=sha256(selection),records=[]); write_json(marker,record)
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4')
    relative=str(study.relative_to(root))
    for case in cases:
        for name,digest in sources.items():
            if sha256(repo/name)!=digest: raise AssertionError('A selected mechanism producer changed')
        protocol=study/'protocols'/('observation-'+case['name']+'.json')
        write_json(protocol,dict(spec,cases=[case],selection_sha256=sha256(selection)))
        suffix=case['name'].removeprefix('cpu-')
        plans=[['measure_scaling_checkpoints.py','--protocol',protocol.name,'--device','cpu','--threads','4'],
            ['verify_repeated_collectives.py','--case',case['name']],
            ['measure_scheduled_prefix.py','--case','prefix-'+suffix],
            ['verify_scheduled_prefix.py','--case','prefix-'+suffix],
            ['measure_reasoning_mechanism.py','--case','reasoning-'+suffix],
            ['verify_reasoning_mechanism.py','--case','reasoning-'+suffix]]
        row=dict(name=case['name'],commands=[]); before=time.time()
        for index,plan in enumerate(plans):
            command=[sys.executable,str(repo/'scripts'/plan[0]),'--root',str(root),'--study',relative,*plan[1:]]
            path=study/'logs'/(case['name']+'-stage'+str(index)+'.log')
            with path.open('x') as log: r=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
            row['commands'].append(dict(command=command,returncode=r.returncode,log_sha256=sha256(path)))
            if r.returncode:
                row['status']='observation_failed'; record['records'].append(row)
                record['status']='observation_failed';write_json(marker,record)
                raise RuntimeError('Resolve the preserved retained-state observation failure: '+str(path))
        proofs=[study/'verification'/kind/(label+suffix+'.json') for kind,label in
                [('collectives','cpu-'),('prefix','prefix-'),('reasoning','reasoning-')]]
        if any(json.loads(path.read_text())['status']!='passed' for path in proofs):
            raise AssertionError('Incomplete independent retained-state observation proof')
        row.update(status='complete',seconds=time.time()-before,
                   verification_sha256={str(path):sha256(path) for path in proofs})
        record['records'].append(row);write_json(marker,record);print(case['name'],'complete',flush=True)
    record['status']='complete';write_json(marker,record)


if __name__=='__main__': main()
