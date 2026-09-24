#!/usr/bin/env python
"""Qualify and complete the joint CPU observation panel with bounded concurrency."""
from companion_paths import child_pythonpath
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np

from model_rg.provenance import sha256, write_json


SOURCES = ['scripts/run_onepass_observations.py','scripts/measure_scaling_checkpoints.py',
    'scripts/measure_onepass_risk.py','scripts/measure_scheduled_inference.py',
    'scripts/verify_onepass_observation.py','scripts/verify_onepass_training_case.py',
    'scripts/verify_onepass_raw.py','scripts/verify_scaling_raw.py',
    'scripts/measure_scheduled_prefix.py','scripts/verify_scheduled_prefix.py',
    'scripts/measure_reasoning_mechanism.py','scripts/verify_reasoning_mechanism.py',
    'src/model_rg/training.py','src/model_rg/native.py','src/model_rg/scaling.py',
    'src/model_rg/criticality.py','src/model_rg/controlled.py','src/model_rg/provenance.py',
    'src/model_rg/precision.py','src/model_rg/variance_family.py',
    'src/model_rg/inference_interventions.py']


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-feasible-20260908')
    p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if not 1<=a.workers<=4:raise ValueError('At most four CPU workers are selected')
    root=Path(a.root).resolve();study=root/a.study;repo=Path(__file__).resolve().parents[1]
    selection=study/'protocols/onepass-observation-selection.json'
    spec=json.loads(selection.read_text());selection_sha=sha256(selection)
    marker=study/'launcher-onepass-observations.json'
    implementation=study/'protocols/onepass-observation-implementation.json'
    if marker.exists() or implementation.exists():raise FileExistsError('An observer execution already exists')
    sources={name:sha256(repo/name) for name in SOURCES}
    now=datetime.now(timezone.utc).isoformat()
    write_json(implementation,dict(schema='onepass-observer-implementation-v1',frozen_at=now,
        selection_sha256=selection_sha,source_files=sources,workers=a.workers,
        scope='Full paired-precision collective observations, risk on already consumed training blocks, generation stability, proper-prefix interventions and endpoint task margins. Each native parent passes independent full-path reconstruction before scientific observations. Four execution-qualification states exercise the entire CPU pipeline first.'))
    record=dict(status='waiting_gpu_qualification',arguments=vars(a),records=[],
                implementation_sha256=sha256(implementation))
    write_json(marker,record)
    qa_study=study/'observer-qualification'
    qa_name=a.study+'/observer-qualification'
    for folder in ['protocols','logs','measurements','verification','data']:
        (qa_study/folder).mkdir(parents=True,exist_ok=False)
    qa_source=json.loads((study/'protocols/onepass-qualification-selection.json').read_text())
    ids=['qualification-controlled-h4-d768-continuous','qualification-controlled-h4-d0-continuous',
         'qualification-reference1-h8-d768-continuous','qualification-reference1-h14-d768-continuous']
    jobs={j['run_id']:j for j in qa_source['jobs']}
    corpus=root/'data/refinedweb-onepass-524288'
    tokens=np.load(corpus/'tokens.npy',mmap_mode='r')
    blocks=np.random.default_rng(651641).permutation(4194304)
    positions=np.random.default_rng(651581).choice(32768,size=1024,replace=False)
    selected=blocks[positions];rows=selected//8;offsets=64*(selected%8)
    cohort=qa_study/'data/cohort.npz'
    np.savez_compressed(cohort,rows=rows,offsets=offsets,
        crops=tokens[rows[:,None],offsets[:,None]+np.arange(65)],consumed_stream_positions=positions)
    cm=qa_study/'data/manifest.json'
    write_json(cm,dict(schema='onepass-seen-risk-cohort-v1',status='complete',frozen_at=now,
        step=1024,source=str(corpus/'tokens.npy'),
        source_sha256=json.loads((corpus/'manifest.json').read_text())['tokens_sha256'],
        stream_seed=650641,selection_seed=651581,cohort_sha256=sha256(cohort)))
    del tokens,blocks
    qa_cases=[]
    for name in ids:
        j=jobs[name]
        qa_cases.append(dict(name='cpu-'+name+'-t1024',run_id=name,recipe=j['recipe'],
            role='qualification',schedule_horizon=j['schedule_horizon'],heads=j['heads'],seed=j['seed'],step=1024,
            shared_seed=j['shared_seed'],stream_seed=j['stream_seed'],inference_stability=True,
            state=str(study/'runs'/name/'final-training-state.pt'),
            parent_manifest=str(study/'runs'/name/'manifest.json'),
            training_cohort=str(cohort),training_cohort_sha256=sha256(cohort),training_cohort_manifest=str(cm)))
    qa_spec=dict(spec,cases=qa_cases,frozen_at=now,cpu_states=4,risk_states=4,inference_states=4,
        source_selection_sha256=selection_sha,scope='Complete observer execution qualification, not scientific training identities.')
    qa_selection=qa_study/'protocols/onepass-observation-selection.json';write_json(qa_selection,qa_spec)
    for filename,label in [('scheduled-prefix-selection.json','prefix'),('reasoning-mechanism-selection.json','reasoning')]:
        original=study/'protocols'/filename;q=json.loads(original.read_text());cases=[]
        for c in qa_cases:
            v={k:x for k,x in c.items() if not k.startswith('training_cohort')}
            v.update(name=label+'-'+c['name'].removeprefix('cpu-'),kind='trained',
                     source=str(root/'assets/PLDR-LLM-v51-SOC-110M-1'))
            cases.append(v)
        q.update(cases=cases,frozen_at=now,scope='Four CPU execution-qualification states only.',
                 source_selection_sha256=sha256(original))
        q['inputs_sha256']=dict(q['inputs_sha256'],**{str(qa_selection):sha256(qa_selection)})
        write_json(qa_study/'protocols'/filename,q)
    gpu=study/'verification/gpu-qualification.json'
    while not gpu.exists():time.sleep(30)
    if json.loads(gpu.read_text())['status']!='complete':raise AssertionError('GPU qualification is incomplete')
    record['gpu_qualification_sha256']=sha256(gpu);write_json(marker,record)
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4')
    locks={name:threading.Lock() for name in {c['run_id'] for c in spec['cases']}}

    def call(command,log):
        started=time.time()
        with log.open('x') as handle:
            result=subprocess.run(command,cwd=repo,env=env,stdout=handle,stderr=subprocess.STDOUT)
        value=dict(command=command,returncode=result.returncode,seconds=time.time()-started,log_sha256=sha256(log))
        if result.returncode:raise RuntimeError(json.dumps(value))
        return value

    def execute(case,folder,selected,study_name,role):
        for name,digest in sources.items():
            if sha256(repo/name)!=digest:raise AssertionError('A frozen observer changed: '+name)
        if sha256(selection)!=selection_sha:raise AssertionError('The scientific selection changed')
        before=time.time();commands=[]
        parent=Path(case['parent_manifest']);meta=json.loads(parent.read_text())
        if meta['status']!='complete':raise AssertionError('The selected native parent is incomplete')
        if role=='scientific':
            with locks[case['run_id']]:
                proof=study/'verification/training'/(case['run_id']+'.json')
                if not proof.exists():
                    commands.append(call([sys.executable,str(repo/'scripts/verify_onepass_training_case.py'),
                        '--root',str(root),'--study',a.study,'--run-id',case['run_id']],
                        study/'logs'/(case['run_id']+'-raw-reconstruction.log')))
                if json.loads(proof.read_text())['status']!='complete':raise AssertionError('Native reconstruction failed')
        saved=meta['saved_states'][str(case['step'])]
        if Path(case['state'])!=parent.parent/saved['filename']:raise AssertionError('The selected checkpoint path changed')
        protocol=folder/'protocols'/('observation-'+case['name']+'.json')
        if protocol.exists():raise FileExistsError(protocol)
        write_json(protocol,dict(schema='frozen-onepass-state-observation-v1',frozen_at=datetime.now(timezone.utc).isoformat(),
            cases=[dict(case,state_sha256=saved['sha256'])],rows=spec['rows'],batch_size=32,
            precisions=['float32','float64'],evaluation_matrices=True,
            selection=str(selected),selection_sha256=sha256(selected),implementation_sha256=sha256(implementation)))
        common=['--root',str(root),'--study',study_name]
        plans=[['measure_scaling_checkpoints.py','--protocol',protocol.name,'--device','cpu','--threads','4'],
               ['measure_onepass_risk.py','--case',case['name']],]
        if case['inference_stability']:plans.append(['measure_scheduled_inference.py','--case',case['name'],'--selection',selected.name])
        plans.append(['verify_onepass_observation.py','--case',case['name']])
        prefix='prefix-'+case['name'].removeprefix('cpu-')
        plans.extend([['measure_scheduled_prefix.py','--case',prefix],['verify_scheduled_prefix.py','--case',prefix]])
        reasoning='reasoning-'+case['name'].removeprefix('cpu-')
        if case['inference_stability']:
            plans.extend([['measure_reasoning_mechanism.py','--case',reasoning],
                          ['verify_reasoning_mechanism.py','--case',reasoning]])
        for index,plan in enumerate(plans):
            command=[sys.executable,str(repo/'scripts'/plan[0]),*common,*plan[1:]]
            commands.append(call(command,folder/'logs'/(case['name']+'-stage'+str(index)+'.log')))
        proofs=[folder/'verification/observations'/(case['name']+'.json'),
                folder/'verification/prefix'/(prefix+'.json')]
        if case['inference_stability']:proofs.append(folder/'verification/reasoning'/(reasoning+'.json'))
        for path in proofs:
            if json.loads(path.read_text())['status']!='passed':raise AssertionError('An observation proof is incomplete')
        row=dict(name=case['name'],role=role,status='complete',seconds=time.time()-before,commands=commands,
                 verification_sha256={str(path):sha256(path) for path in proofs})
        write_json(folder/('ledger-observation-'+case['name']+'.json'),row)
        return row

    def phase(cases,folder,selected,study_name,role):
        remaining={c['name']:c for c in cases};active={};failed=False
        record['status']='observing_'+role;write_json(marker,record)
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            while remaining or active:
                for name,case in list(remaining.items()):
                    if len(active)>=a.workers or failed:break
                    if Path(case['parent_manifest']).exists():
                        active[pool.submit(execute,case,folder,selected,study_name,role)]=name
                        del remaining[name]
                for future,name in list(active.items()):
                    if not future.done():continue
                    try:row=future.result()
                    except Exception as exc:row=dict(name=name,role=role,status='execution_failure',error=repr(exc))
                    record['records'].append(row);write_json(marker,record);print(json.dumps(row),flush=True)
                    del active[future]
                    if row['status']!='complete':failed=True
                if failed and not active:
                    record.update(status='observation_failure',unexecuted=list(remaining));write_json(marker,record)
                    raise RuntimeError('Resolve the preserved observation failure before continuing')
                if remaining or active:time.sleep(30)

    phase(qa_cases,qa_study,qa_selection,qa_name,'qualification')
    proof=study/'verification/observer-qualification.json'
    write_json(proof,dict(status='passed',cases=qa_cases,records=record['records'],
        implementation_sha256=sha256(implementation),selection_sha256=sha256(qa_selection)))
    phase(spec['cases'],study,selection,a.study,'scientific')
    record['status']='complete';write_json(marker,record)
    print('complete',len(spec['cases']),'scientific states',flush=True)


if __name__=='__main__':main()
