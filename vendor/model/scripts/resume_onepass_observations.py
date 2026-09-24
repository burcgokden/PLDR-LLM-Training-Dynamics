#!/usr/bin/env python
"""Resume preserved observation stages under an explicitly qualified arithmetic check."""
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

from model_rg.provenance import sha256, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--study', default='scheduled-training-feasible-20260908')
    p.add_argument('--recovery', required=True)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    if not 1 <= a.workers <= 4:
        raise ValueError('One through four observation workers are supported')
    root = Path(a.root).resolve(); study = root/a.study
    repo = Path(__file__).resolve().parents[1]; recovery = Path(a.recovery).resolve()
    marker = study/'launcher-onepass-observations.json'
    selection = study/'protocols/onepass-observation-selection.json'
    implementation = study/'protocols/onepass-observation-implementation.json'
    original = recovery/'original-launcher-onepass-observations.json'
    qualification = recovery/'arithmetic-qualification.json'
    protocol = recovery/'recovery-protocol.json'
    if protocol.exists():
        raise FileExistsError('An immutable recovery execution already exists')
    state = json.loads(marker.read_text())
    if state['status'] != 'observation_failure' or sha256(marker) != sha256(original):
        raise AssertionError('Recovery requires the exact preserved stopped controller')
    spec = json.loads(selection.read_text()); selected = {c['name']: c for c in spec['cases']}
    original_impl = json.loads(implementation.read_text())
    sources = dict(original_impl['source_files'])
    sources.update({name: sha256(repo/name) for name in [
        'scripts/resume_onepass_observations.py',
        'scripts/verify_onepass_observation_arithmetic.py']})
    for name, digest in sources.items():
        if sha256(repo/name) != digest:
            raise AssertionError('A frozen observation implementation changed: '+name)
    qa = json.loads(qualification.read_text())
    if qa['status'] != 'passed' or qa['cases'] != 6 or len(qa['records']) != 6:
        raise AssertionError('All six arithmetic qualification cases must pass')
    for name, digest in qa['sources'].items():
        if sha256(repo/name) != digest:
            raise AssertionError('The arithmetic qualification predates its source')
    for row in qa['records']:
        if row['returncode'] or sha256(row['proof']) != row['proof_sha256']:
            raise AssertionError('An arithmetic qualification changed')
    failures = [r for r in state['records'] if r['status'] != 'complete']
    completed = [r for r in state['records'] if r['status'] == 'complete']
    if len(failures) != 2 or {r['name'] for r in failures} != {
        'cpu-compact-reference1-h8-s640101-t32768',
        'cpu-compact-reference1-h8-s640101-t40960'}:
        raise AssertionError('Unexpected preserved failure scope')
    for row in completed:
        for path, digest in row['verification_sha256'].items():
            if sha256(path) != digest or json.loads(Path(path).read_text())['status'] != 'passed':
                raise AssertionError('A previously completed observation changed')
    contract = dict(schema='onepass-observation-arithmetic-recovery-v1',
        at=datetime.now(timezone.utc).isoformat(), sources=sources,
        selection_sha256=sha256(selection), original_implementation_sha256=sha256(implementation),
        original_controller=str(original), original_controller_sha256=sha256(original),
        qualification=str(qualification), qualification_sha256=sha256(qualification),
        failed_attempts=failures, completed_records_retained=len(completed),
        workers=a.workers,
        verification_entrypoint='scripts/verify_onepass_observation_arithmetic.py',
        scope='Resume exactly the original 284 selected observations. Reuse complete producer artifacts without modification. Require bytewise native arithmetic replay, independent extended-precision reference checks and explicit rounding discrepancies. No training update, endpoint, estimator or recorded measurement is changed.')
    write_json(protocol, contract)
    logs = recovery/'logs'; logs.mkdir(exist_ok=False)
    state['records'] = completed
    state['preserved_failed_attempts'] = failures
    state['recovery'] = dict(protocol=str(protocol), protocol_sha256=sha256(protocol),
        original_controller=str(original), original_controller_sha256=sha256(original),
        qualification=str(qualification), qualification_sha256=sha256(qualification))
    state.pop('unexecuted', None)
    state['status'] = 'observing_scientific'
    write_json(marker, state)
    env = dict(os.environ, PYTHONPATH=child_pythonpath("model"), PYTHONDONTWRITEBYTECODE='1',
        OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4')
    os.nice(8)
    locks = {c['run_id']: threading.Lock() for c in selected.values()}

    def call(command, log):
        before = time.time()
        with log.open('x') as handle:
            result = subprocess.run(command, cwd=repo, env=env, stdout=handle, stderr=subprocess.STDOUT)
        row = dict(command=command, returncode=result.returncode,
            seconds=time.time()-before, log_sha256=sha256(log))
        if result.returncode:
            raise RuntimeError(json.dumps(row))
        return row

    def execute(case):
        for name, digest in sources.items():
            if sha256(repo/name) != digest:
                raise AssertionError('Recovery source changed: '+name)
        if sha256(selection) != contract['selection_sha256']:
            raise AssertionError('Selected observation identities changed')
        name = case['name']; before = time.time(); commands = []
        parent = Path(case['parent_manifest']); meta = json.loads(parent.read_text())
        if meta['status'] != 'complete':
            raise AssertionError('The native parent did not complete')
        with locks[case['run_id']]:
            proof = study/'verification/training'/(case['run_id']+'.json')
            if not proof.exists():
                commands.append(call([sys.executable, str(repo/'scripts/verify_onepass_training_case.py'),
                    '--root',str(root),'--study',a.study,'--run-id',case['run_id']],
                    logs/(case['run_id']+'-native.log')))
            if json.loads(proof.read_text())['status'] != 'complete':
                raise AssertionError('The retained native path check failed')
        saved = meta['saved_states'][str(case['step'])]
        if Path(case['state']) != parent.parent/saved['filename']:
            raise AssertionError('The saved state identity changed')
        case_protocol = study/'protocols'/('observation-'+name+'.json')
        if not case_protocol.exists():
            write_json(case_protocol, dict(schema='frozen-onepass-state-observation-v1',
                frozen_at=datetime.now(timezone.utc).isoformat(),
                cases=[dict(case,state_sha256=saved['sha256'])],rows=spec['rows'],batch_size=32,
                precisions=['float32','float64'],evaluation_matrices=True,
                selection=str(selection),selection_sha256=sha256(selection),
                implementation_sha256=sha256(implementation),
                recovery_protocol=str(protocol),recovery_protocol_sha256=sha256(protocol)))
        else:
            retained = json.loads(case_protocol.read_text())
            if retained['cases'] != [dict(case,state_sha256=saved['sha256'])] or retained['selection_sha256'] != sha256(selection):
                raise AssertionError('A retained observation protocol changed')
        suffix = name.removeprefix('cpu-'); prefix = 'prefix-'+suffix; reasoning = 'reasoning-'+suffix
        plans = [(['measure_scaling_checkpoints.py','--protocol',case_protocol.name,'--device','cpu','--threads','4'], name),
            (['measure_onepass_risk.py','--case',name], 'risk-'+suffix)]
        if case['inference_stability']:
            plans.append((['measure_scheduled_inference.py','--case',name,'--selection',selection.name], 'inference-'+suffix))
        plans.append((['verify_onepass_observation_arithmetic.py','--case',name,'--recovery-protocol',str(protocol)], None))
        plans.extend([(['measure_scheduled_prefix.py','--case',prefix], prefix),
            (['verify_scheduled_prefix.py','--case',prefix], None)])
        if case['inference_stability']:
            plans.extend([(['measure_reasoning_mechanism.py','--case',reasoning], reasoning),
                (['verify_reasoning_mechanism.py','--case',reasoning], None)])
        for index, (plan, artifact) in enumerate(plans):
            if artifact:
                folder = study/'measurements'/artifact; manifest = folder/'manifest.json'
                if folder.exists():
                    if not manifest.exists() or json.loads(manifest.read_text())['status'] != 'complete':
                        raise AssertionError('A partial producer artifact requires separate recovery: '+str(folder))
                    m = json.loads(manifest.read_text())
                    retained_files = {}
                    for filename,key in [('binding.json','binding_sha256'),('results.json','results_sha256'),('measurements.npz','raw_sha256')]:
                        if key in m:
                            if sha256(folder/filename) != m[key]:
                                raise AssertionError('A preserved producer artifact changed')
                            retained_files[str(folder/filename)] = m[key]
                    commands.append(dict(status='reused_complete_producer',manifest=str(manifest),
                        manifest_sha256=sha256(manifest),checked_sha256=retained_files))
                    continue
            command = [sys.executable,str(repo/'scripts'/plan[0]),'--root',str(root),'--study',a.study,*plan[1:]]
            commands.append(call(command,logs/(name+'-stage'+str(index)+'.log')))
        proofs = [study/'verification/observations'/(name+'.json'), study/'verification/prefix'/(prefix+'.json')]
        if case['inference_stability']:
            proofs.append(study/'verification/reasoning'/(reasoning+'.json'))
        if any(json.loads(q.read_text())['status'] != 'passed' for q in proofs):
            raise AssertionError('A required observation proof did not pass')
        row = dict(name=name,role='scientific',status='complete',seconds=time.time()-before,
            commands=commands,verification_sha256={str(q):sha256(q) for q in proofs},
            recovery_protocol_sha256=sha256(protocol))
        ledger = study/('ledger-observation-'+name+'.json')
        if ledger.exists():
            raise FileExistsError('A completed observation ledger already exists')
        write_json(ledger,row)
        return row

    done = {r['name'] for r in completed if r['role']=='scientific'}
    failed_names = {r['name'] for r in failures}
    ordered = sorted(selected.values(),key=lambda c:(c['name'] not in failed_names, list(selected).index(c['name'])))
    remaining = {c['name']:c for c in ordered if c['name'] not in done}
    running = {}; failed = False
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        while remaining or running:
            for name, case in list(remaining.items()):
                if failed or len(running) >= a.workers:break
                if Path(case['parent_manifest']).exists():
                    running[pool.submit(execute,case)] = name; del remaining[name]
            for future,name in list(running.items()):
                if not future.done():continue
                try:row = future.result()
                except Exception as exc:row = dict(name=name,role='scientific',status='execution_failure',error=repr(exc))
                state['records'].append(row);write_json(marker,state)
                print(name,row['status'],flush=True);del running[future]
                if row['status'] != 'complete':failed=True
            if failed and not running:
                state.update(status='observation_failure',unexecuted=list(remaining));write_json(marker,state)
                raise RuntimeError('A further observation failure is preserved and requires investigation')
            if remaining or running:time.sleep(15)
    if len(state['records']) != 288:
        raise AssertionError('The original qualification and scientific coverage is incomplete')
    state['status']='complete';state['completed_at']=datetime.now(timezone.utc).isoformat();write_json(marker,state)
    print('All 284 original scientific observations completed',flush=True)


if __name__ == '__main__':
    main()
