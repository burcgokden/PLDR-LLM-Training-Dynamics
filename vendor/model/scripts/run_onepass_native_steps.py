#!/usr/bin/env python
"""Run the selected conditional CPU probes beside the existing GPU queue."""
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
from prepare_onepass_native_steps import bind_case


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-feasible-20260908');a=p.parse_args()
    root=Path(a.root).resolve();study=root/a.study;repo=Path(__file__).resolve().parents[1]
    selection=study/'protocols/onepass-native-step-selection.json';spec=json.loads(selection.read_text())
    marker=study/'launcher-onepass-native-steps.json'
    if marker.exists():raise FileExistsError(marker)
    state=dict(status='waiting',selection_sha256=sha256(selection),started_at=datetime.now(timezone.utc).isoformat(),records=[])
    write_json(marker,state);logs=study/'logs/onepass-native-steps';logs.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
    for chosen in spec['cases']:
        while not Path(chosen['training_verification']).exists():time.sleep(20)
        for name,digest in spec['producer_sources'].items():
            if sha256(repo/name)!=digest:raise AssertionError('A selected native-step source changed')
        case=bind_case(chosen)
        execution=study/'protocols/onepass-native-step-executions'/(case['name']+'.json')
        if execution.exists():raise FileExistsError(execution)
        inputs=dict(spec['inputs_sha256']);inputs[str(selection)]=sha256(selection)
        for name in ['parent','parent_manifest','training_verification']:inputs[case[name]]=sha256(case[name])
        write_json(execution,dict(spec,cases=[case],inputs_sha256=inputs,selection_sha256=sha256(selection)))
        proof=study/'verification/onepass-native-steps'/(case['name']+'.json')
        plans=[['measure_onepass_native_step.py','--threads',str(spec['threads'])],
               ['verify_onepass_native_step.py','--output',str(proof)]]
        record=dict(name=case['name'],commands=[]);before=time.time();state['status']='observing';write_json(marker,state)
        for plan in plans:
            command=[sys.executable,str(repo/'scripts'/plan[0]),'--root',str(root),'--study',a.study,
                '--protocol',str(execution.relative_to(study/'protocols')),'--case',case['name'],*plan[1:]]
            logpath=logs/(case['name']+'-'+plan[0]+'.log')
            with logpath.open('x') as log:process=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
            record['commands'].append(dict(command=command,returncode=process.returncode,log_sha256=sha256(logpath)))
            if process.returncode:
                record['status']='failed';state['records'].append(record);state['status']='failed';write_json(marker,state)
                raise RuntimeError('Resolve the preserved conditional-step failure: '+str(logpath))
        if json.loads(proof.read_text())['status']!='passed':raise AssertionError('The complete conditional-step check did not pass')
        record.update(status='complete',seconds=time.time()-before,verification=str(proof),verification_sha256=sha256(proof))
        state['records'].append(record);state['status']='waiting';write_json(marker,state);print(case['name'],'complete',flush=True)
    state['status']='complete';write_json(marker,state)


if __name__=='__main__':main()
