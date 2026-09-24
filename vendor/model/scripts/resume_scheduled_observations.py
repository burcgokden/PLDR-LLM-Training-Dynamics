#!/usr/bin/env python
"""Resume immutable scheduled observations after qualifying native reduction replay."""
from companion_paths import child_pythonpath
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from model_rg.provenance import sha256,write_json


PRODUCERS=['scripts/resume_scheduled_observations.py','scripts/measure_scaling_checkpoints.py',
    'scripts/measure_scheduled_risk.py','scripts/measure_scheduled_inference.py',
    'scripts/verify_scheduled_observation_native.py','scripts/verify_scaling_raw.py',
    'src/model_rg/training.py','src/model_rg/native.py','src/model_rg/scaling.py',
    'src/model_rg/criticality.py','src/model_rg/controlled.py','src/model_rg/provenance.py',
    'src/model_rg/precision.py','src/model_rg/variance_family.py']


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-20260908');p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();root=Path(a.root).resolve();study=root/a.study;repo=Path(__file__).resolve().parents[1]
    if not 1<=a.workers<=2:raise ValueError('The observation budget allows one or two CPU workers')
    selection=study/'protocols/regime-observation-selection.json';spec=json.loads(selection.read_text())
    signatures={name:sha256(repo/name) for name in PRODUCERS};selection_signature=sha256(selection)
    marker=study/'launcher-scheduled-observations-recovered.json';frozen=study/'protocols/observation-recovered-implementation.json'
    original_marker=study/'launcher-scheduled-observations.json'
    original_implementation=study/'protocols/observation-implementation.json'
    original=json.loads(original_marker.read_text());original_spec=json.loads(original_implementation.read_text())
    if original['status']!='observation_failure' or len(original['records'])!=4 or any(r['role']!='qualification' for r in original['records']):
        raise AssertionError('Recovery requires the preserved four-case qualification failure before scientific observations')
    for name,digest in original_spec['producer_sources'].items():
        if sha256(repo/name)!=digest:raise AssertionError('An original frozen observation producer changed: '+name)
    if marker.exists() or frozen.exists():raise FileExistsError('An immutable observation execution already exists')
    write_json(frozen,dict(frozen_at=datetime.now(timezone.utc).isoformat(),selection_sha256=selection_signature,
        producer_sources=signatures,workers=a.workers,
        qualification='Reconstruct all four existing immutable qualification observations with exact native float32 retained-logit reduction replay and separately reported NumPy float64 discrepancies. The existing generation and matrix observations are retained and independently reconstructed again. No qualification model or observed data is rerun or replaced.',
        recovery=dict(original_controller_sha256=sha256(original_marker),original_implementation_sha256=sha256(original_implementation),
            reason='The original N=4 reference qualification exceeded the fixed float64 comparison tolerance for the all-position transposed float32 cross-entropy reduction. Every retained native reduction reproduced bytewise. The original failure and data are preserved; only the checker and continuation controller differ.'),
        completion='Every selected scientific state is measured after its native parent is immutable and complete. Numerical parent failures are retained explicitly and are never replaced by another state.'))
    record=dict(status='waiting_gpu_qualification',arguments=vars(a),implementation_sha256=sha256(frozen),records=[])
    write_json(marker,record)
    gpu=study/'verification/gpu-qualification.json'
    while not gpu.exists():time.sleep(30)
    if json.loads(gpu.read_text())['status']!='complete':raise AssertionError('Scheduled native GPU qualification is incomplete')
    record['gpu_qualification_sha256']=sha256(gpu)
    qa_selection=study/'protocols/observation-qualification-selection.json'
    qa=json.loads(qa_selection.read_text())
    if len(qa['cases'])!=4 or qa['source_selection_sha256']!=selection_signature:
        raise AssertionError('The original observation qualification selection changed')
    record.update(original_controller_sha256=sha256(original_marker),qualification_selection_sha256=sha256(qa_selection))
    write_json(marker,record)
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4')

    def requalify(case,selected):
        for name,signature in signatures.items():
            if sha256(repo/name)!=signature:raise AssertionError('A recovered observer changed: '+name)
        output=study/'verification/observation-qualification-native'/(case['name']+'.json')
        command=[sys.executable,str(repo/'scripts/verify_scheduled_observation_native.py'),'--root',str(root),
            '--study',a.study,'--case',case['name'],'--selection',selected.name,'--output',str(output)]
        log=study/'logs'/(case['name']+'-native-reduction-qualification.log');before=time.time()
        with log.open('x') as stream:run=subprocess.run(command,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT)
        value=dict(name=case['name'],role='qualification',status='complete' if run.returncode==0 else 'observation_failure',
            seconds=time.time()-before,commands=[dict(command=command,returncode=run.returncode,log_sha256=sha256(log))])
        if run.returncode==0:
            check=json.loads(output.read_text())
            if check['status']!='passed' or check['case']!=case:raise AssertionError('Recovered qualification scope changed')
            value.update(verification=str(output),verification_sha256=sha256(output),risk=check['risk'])
        return value

    def execute(case,selected,role):
        for name,signature in signatures.items():
            if sha256(repo/name)!=signature:raise AssertionError('A frozen observer changed: '+name)
        if sha256(selection)!=selection_signature:raise AssertionError('Scheduled observation selection changed')
        parent=Path(case['parent_manifest']);meta=json.loads(parent.read_text())
        if meta['status']!='complete':
            return dict(name=case['name'],role=role,status='noncomplete_parent',parent_status=meta['status'],parent_manifest_sha256=sha256(parent))
        for key in ['heads','seed','recipe','shared_seed','stream_seed']:
            if meta['arguments'][key]!=case[key]:raise AssertionError('Scheduled observation parent condition changed')
        saved=meta['saved_states'][str(case['step'])]
        if Path(case['state'])!=parent.parent/saved['filename']:raise AssertionError('Scheduled checkpoint path changed')
        bound=dict(case,state_sha256=saved['sha256']);protocol=study/'protocols'/('observation-'+case['name']+'.json')
        if protocol.exists():raise FileExistsError(protocol)
        write_json(protocol,dict(schema='frozen-scheduled-state-observation-v1',frozen_at=datetime.now(timezone.utc).isoformat(),
            cases=[bound],rows=spec['rows'],batch_size=32,precisions=spec['precisions'],evaluation_matrices=True,
            selection=str(selected),selection_sha256=sha256(selected),implementation_sha256=sha256(frozen)))
        common=['--root',str(root),'--study',a.study]
        commands=[[sys.executable,str(repo/'scripts/measure_scaling_checkpoints.py'),*common,
                   '--protocol',protocol.name,'--device','cpu','--threads','4'],
                  [sys.executable,str(repo/'scripts/measure_scheduled_risk.py'),*common,'--case',case['name'],'--selection',selected.name]]
        if case['inference_stability']:
            commands.append([sys.executable,str(repo/'scripts/measure_scheduled_inference.py'),*common,'--case',case['name'],'--selection',selected.name])
        commands.append([sys.executable,str(repo/'scripts/verify_scheduled_observation_native.py'),*common,'--case',case['name'],'--selection',selected.name])
        results=[];before=time.time()
        for index,command in enumerate(commands):
            log=study/'logs'/(case['name']+'-observation-'+str(index)+'.log')
            with log.open('x') as stream:r=subprocess.run(command,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT)
            results.append(dict(command=command,returncode=r.returncode,log_sha256=sha256(log)))
            if r.returncode:break
        status='complete' if len(results)==len(commands) and all(r['returncode']==0 for r in results) else 'observation_failure'
        value=dict(name=case['name'],role=role,status=status,seconds=time.time()-before,commands=results,protocol_sha256=sha256(protocol))
        if status=='complete':value['verification_sha256']=sha256(study/'verification/observations'/(case['name']+'.json'))
        write_json(study/('ledger-observation-'+case['name']+'.json'),value)
        return value

    def phase(cases,selected,role):
        remaining={c['name']:c for c in cases};active={};failed=False
        record['status']='observing_'+role;write_json(marker,record)
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            while remaining or active:
                for name,case in list(remaining.items()):
                    if len(active)>=a.workers or failed:break
                    if Path(case['parent_manifest']).exists():
                        future=pool.submit(requalify,case,selected) if role=='qualification' else pool.submit(execute,case,selected,role)
                        active[future]=name;del remaining[name]
                for future,name in list(active.items()):
                    if not future.done():continue
                    try:value=future.result()
                    except Exception as exc:value=dict(name=name,role=role,status='controller_failure',error=repr(exc))
                    record['records'].append(value);write_json(marker,record);print(json.dumps(value),flush=True);del active[future]
                    if value['status'] not in ['complete','noncomplete_parent'] or (role=='qualification' and value['status']!='complete'):failed=True
                if failed and not active:
                    record['status']='observation_failure';record['unexecuted']=[*remaining];write_json(marker,record)
                    raise RuntimeError('Resolve the preserved observation failure before further execution')
                if remaining or active:time.sleep(30)
    phase(qa['cases'],qa_selection,'qualification')
    phase(spec['cases'],selection,'scientific')
    scientific=[r for r in record['records'] if r['role']=='scientific']
    record['status']='complete' if all(r['status']=='complete' for r in scientific) else 'complete_with_noncomplete_parents'
    write_json(marker,record);print(record['status'],len(scientific),'selected states',flush=True)


if __name__=='__main__':main()
