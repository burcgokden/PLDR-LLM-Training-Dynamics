#!/usr/bin/env python
"""Complete constant-rate prerequisites, qualify scheduled execution, then train."""
from companion_paths import child_pythonpath
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from model_rg.provenance import sha256, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-20260908');p.add_argument('--protocol-prefix',default='regime-')
    p.add_argument('--training-priority',default='');a=p.parse_args()
    root=Path(a.root).resolve();study=root/a.study;repo=Path(__file__).resolve().parents[1]
    record_path=study/'launcher-scheduled.json'
    if record_path.exists():raise FileExistsError(record_path)
    protocols={name:study/'protocols'/f'{a.protocol_prefix}{name}-selection.json' for name in ['training','qualification']}
    specs={name:json.loads(path.read_text()) for name,path in protocols.items()}
    signatures={name:sha256(path) for name,path in protocols.items()}
    record=dict(status='waiting_constant_rate',arguments=vars(a),protocol_sha256=signatures,phases=[],records=[])
    write_json(record_path,record)
    prerequisite=root/'critical-scaling-20260906/launcher-boundary-and-extended.json'
    while True:
        paths=list(specs['training']['constant_prerequisites'].values())+[str(prerequisite)]
        present=[Path(path) for path in paths if Path(path).exists()]
        invalid=[str(path) for path in present if json.loads(path.read_text()).get('status') not in ['complete','running']]
        if invalid:raise RuntimeError('A constant-rate prerequisite requires resolution: '+str(invalid))
        if len(present)==len(paths) and all(json.loads(path.read_text()).get('status')=='complete' for path in present):
            break
        time.sleep(30)
    record['constant_prerequisite_sha256']={str(path):sha256(path) for path in present}
    record['status']='qualifying';write_json(record_path,record)
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4')

    def execute(kind,job,device):
        spec=specs[kind]
        if sha256(protocols[kind])!=signatures[kind]:raise AssertionError('A scheduled selection changed')
        for name,signature in spec['producer_sources'].items():
            if sha256(repo/name)!=signature:raise AssertionError('A frozen scheduled producer changed: '+name)
        j=dict(job);execution=dict(spec)
        execution['source_selection']=dict(path=str(protocols[kind]),sha256=signatures[kind])
        execution['reference_inputs']=dict(spec['reference_inputs'])
        execution['reference_inputs'][str(protocols[kind])]=signatures[kind]
        if j.get('resume_from_run'):
            parent=study/'runs'/j['resume_from_run'];meta=json.loads((parent/'manifest.json').read_text())
            if meta['status']!='complete':raise AssertionError('A qualified continuation requires a complete parent')
            saved=meta['saved_states'][str(j['resume_from_step'])]
            j['resume']=str(parent/saved['filename']);j['parent_sha256']=saved['sha256']
            if sha256(j['resume'])!=j['parent_sha256']:raise AssertionError('The bound parent checkpoint changed')
            execution['reference_inputs'][str(parent/'manifest.json')]=sha256(parent/'manifest.json')
        execution['jobs']=[j]
        filename='executions/'+j['run_id']+'.json';bound=study/'protocols'/filename
        if bound.exists():raise FileExistsError(bound)
        write_json(bound,execution)
        command=[sys.executable,str(repo/spec.get('training_entrypoint','scripts/train_scheduled_scaling.py')),'--root',str(root),
            '--study',a.study,'--protocol',filename,'--run-id',j['run_id'],'--heads',str(j['heads']),
            '--seed',str(j['seed']),'--recipe',j['recipe'],'--steps',str(j['steps']),'--device',f'cuda:{device}',
            '--stream-seed',str(j['stream_seed']),'--shared-seed',str(j['shared_seed']),
            '--probe-every',str(j['probe_every']),'--microbatch',str(j['microbatch']),'--save-steps',j['save_steps']]
        if j['checkpoint_decoders']:command+=['--checkpoint-decoders']
        if j.get('resume'):command+=['--resume',j['resume']]
        before=time.time()
        with (study/'logs'/(j['run_id']+'.log')).open('x') as log:
            completed=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
        row=dict(run_id=j['run_id'],role=kind,device=device,command=command,returncode=completed.returncode,
                 seconds=time.time()-before,execution_protocol_sha256=sha256(bound))
        path=study/'runs'/j['run_id']/'manifest.json'
        if path.exists():row.update(scientific_status=json.loads(path.read_text())['status'],manifest_sha256=sha256(path))
        write_json(study/('ledger-'+j['run_id']+'.json'),row)
        return row

    def phase(kind,ids):
        selected={j['run_id']:j for j in specs[kind]['jobs']}
        pending=[selected[name] for name in ids];active={};free={0,1};failed=False
        phase_record=dict(role=kind,run_ids=ids,status='running');record['phases'].append(phase_record)
        write_json(record_path,record)
        with ThreadPoolExecutor(max_workers=2) as pool:
            while pending or active:
                for j in list(pending):
                    available=free if 'device' not in j else free.intersection({j['device']})
                    if not available or failed:continue
                    device=min(available);free.remove(device);pending.remove(j)
                    active[pool.submit(execute,kind,j,device)]=(j,device)
                if not active:
                    if pending:raise RuntimeError('No executable selected scheduled job remains')
                    break
                done,_=wait(active,timeout=30,return_when=FIRST_COMPLETED)
                for future in done:
                    j,device=active.pop(future);free.add(device)
                    try:row=future.result()
                    except Exception as exc:
                        row=dict(run_id=j['run_id'],role=kind,device=device,returncode=-1,error=repr(exc))
                    record['records'].append(row);write_json(record_path,record);print(json.dumps(row),flush=True)
                    if row['returncode'] or (kind=='qualification' and row.get('scientific_status')!='complete'):
                        failed=True;phase_record['status']='execution_failure';record['status']='execution_failure'
                        phase_record['unexecuted']=[j['run_id'] for j in pending];pending=[];write_json(record_path,record)
        if failed:raise RuntimeError('Resolve the recorded scheduled execution failure before continuing')
        phase_record['status']='complete';write_json(record_path,record)

    for ids in specs['qualification']['qualification_phases']:phase('qualification',ids)
    command=[sys.executable,str(repo/'scripts/verify_scheduled_qualification.py'),'--root',str(root),'--study',a.study]
    with (study/'logs/qualification-verification.log').open('x') as log:
        result=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        record['status']='qualification_verification_failure';write_json(record_path,record)
        raise RuntimeError('The scheduled GPU qualification did not pass')
    record['qualification_verification_sha256']=sha256(study/'verification/gpu-qualification.json')
    record['status']='training';write_json(record_path,record)
    selected_ids=[j['run_id'] for j in specs['training']['jobs']]
    priority=[x for x in a.training_priority.split(',') if x]
    if len(set(priority))!=len(priority) or any(x not in selected_ids for x in priority):
        raise AssertionError('The scheduling priority must be a unique subset of selected trajectory identities')
    order=priority+[x for x in selected_ids if x not in priority]
    record['scientific_execution_order']=order;write_json(record_path,record)
    phase('training',order)
    scientific=[r for r in record['records'] if r['role']=='training']
    record['status']='complete' if all(r.get('scientific_status')=='complete' for r in scientific) else 'complete_with_numerical_failures'
    write_json(record_path,record);print(record['status'],len(scientific),'scheduled trajectories',flush=True)


if __name__=='__main__':main()
