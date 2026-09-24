#!/usr/bin/env python
"""Seal saved intermediate checkpoints for early native prefix measurements."""
from companion_paths import child_pythonpath
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

from model_rg.controlled import bind_run
from model_rg.provenance import sha256,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-20260908');a=p.parse_args()
    root=Path(a.root).resolve();study=root/a.study;repo=Path(__file__).resolve().parents[1]
    selection=study/'protocols/early-prefix-selection.json';spec=json.loads(selection.read_text())
    selected_sha=sha256(selection);marker=study/'launcher-early-prefix.json'
    if marker.exists():raise FileExistsError(marker)
    record=dict(status='waiting_saved_checkpoints',selection_sha256=selected_sha,records=[])
    write_json(marker,record);remaining={c['name']:c for c in spec['cases']};seen={}
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4')
    torch.set_num_threads(4)
    while remaining:
        progressed=False
        for name,case in list(remaining.items()):
            source=Path(case['source_checkpoint']);final=Path(case['final_manifest'])
            if not source.exists():
                if final.exists() and json.loads(final.read_text())['status']!='complete':
                    record['records'].append(dict(name=name,status='checkpoint_unavailable_after_parent_failure',parent_manifest_sha256=sha256(final)))
                    del remaining[name];progressed=True;write_json(marker,record)
                continue
            stat=source.stat();fingerprint=(stat.st_size,stat.st_mtime_ns)
            if seen.get(name)!=fingerprint:seen[name]=fingerprint;continue
            if sha256(selection)!=selected_sha:raise AssertionError('The early prefix selection changed')
            for path,digest in spec['inputs_sha256'].items():
                if sha256(path)!=digest:raise AssertionError('An early prefix input changed')
            for path,digest in spec['producer_sources'].items():
                if sha256(repo/path)!=digest:raise AssertionError('The early prefix producer changed')
            # Each selected source filename is written once by the frozen producer.
            # Two observations of unchanged file metadata precede a complete zip
            # parse. A verified byte copy is the immutable observation parent.
            try:saved=torch.load(source,map_location='cpu',mmap=True,weights_only=True)
            except (OSError,RuntimeError,EOFError):continue
            arguments=saved['arguments']
            if saved['step']!=case['step'] or arguments['run_id']!=case['run_id'] or arguments['steps']!=262144:
                raise AssertionError('The early checkpoint identity or horizon changed')
            for key in ['heads','seed','recipe','shared_seed','stream_seed']:
                if arguments[key]!=case[key]:raise AssertionError('The early checkpoint condition changed')
            del saved
            source_binding=Path(case['source_binding']);binding=json.loads(source_binding.read_text())
            training=json.loads(Path(spec['training_selection']).read_text())
            for path,digest in training['producer_sources'].items():
                if binding['source_files'][path]!=digest:raise AssertionError('The native early checkpoint producer differs from the frozen selection')
            child=study/'early-checkpoints'/name;snapshot=child/'checkpoint';snapshot.mkdir(parents=True,exist_ok=False)
            before=time.time();source_signature=sha256(source)
            temporary=snapshot/'state.pt.copying';shutil.copyfile(source,temporary)
            if sha256(temporary)!=source_signature or sha256(source)!=source_signature:
                raise AssertionError('The source checkpoint changed during sealing')
            state=snapshot/'state.pt';temporary.rename(state)
            bind_run(snapshot,[selection,source,source_binding,Path(spec['training_selection'])],vars(a))
            trajectory_status=json.loads(final.read_text())['status'] if final.exists() else 'running'
            parent=snapshot/'manifest.json'
            write_json(parent,dict(schema='sealed-native-checkpoint-observation-parent-v1',status='complete',
                role='Immutable observation copy of one saved state; this is not a new training trajectory.',
                captured_at=datetime.now(timezone.utc).isoformat(),arguments=arguments,completed_step=case['step'],
                saved_states={str(case['step']):dict(filename='state.pt',sha256=source_signature)},
                binding_sha256=sha256(snapshot/'binding.json'),source_checkpoint=str(source),source_checkpoint_sha256=source_signature,
                source_binding=str(source_binding),source_binding_sha256=sha256(source_binding),
                source_trajectory_manifest=str(final),source_trajectory_status_at_capture=trajectory_status,
                final_selected_horizon=262144))
            prefix_case=dict(name=name,kind='trained',source=str(root/'assets/PLDR-LLM-v51-SOC-110M-1'),
                state=str(state),parent_manifest=str(parent),step=case['step'],
                **{key:case[key] for key in ['heads','seed','recipe','shared_seed','stream_seed']})
            child_protocol=child/'protocols/scheduled-prefix-selection.json'
            base=json.loads(Path(spec['prefix_selection']).read_text())
            write_json(child_protocol,dict(schema='sealed-early-prefix-observation-v1',cases=[prefix_case],
                rows=spec['rows'],prefix_lengths=spec['prefix_lengths'],producer_sources=base['producer_sources'],
                inputs_sha256={**spec['inputs_sha256'],str(selection):selected_sha,str(parent):sha256(parent),
                    str(source):source_signature,str(source_binding):sha256(source_binding)},
                scope='Early observation of a sealed saved state in a continuing selected trajectory. Final reconciliation must match the original completed trajectory checkpoint hash. No extra independent training seed or trajectory is counted.'))
            relative=str(child.relative_to(root));commands=[]
            for script in ['measure_scheduled_prefix.py','verify_scheduled_prefix.py']:
                command=[sys.executable,str(repo/'scripts'/script),'--root',str(root),'--study',relative,'--case',name]
                with (child/(script+'.log')).open('x') as log:r=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
                commands.append(dict(command=command,returncode=r.returncode))
                if r.returncode:break
            status='complete' if len(commands)==2 and all(c['returncode']==0 for c in commands) else 'observation_failure'
            item=dict(name=name,status=status,commands=commands,seconds=time.time()-before,
                sealed_parent=str(parent),sealed_parent_sha256=sha256(parent),source_checkpoint_sha256=source_signature,
                final_reconciliation_required=True)
            if status=='complete':item['verification_sha256']=sha256(child/'verification/prefix'/(name+'.json'))
            record['records'].append(item);write_json(marker,record);print(json.dumps(item),flush=True)
            if status!='complete':raise RuntimeError('Resolve the preserved early prefix observation failure')
            del remaining[name];progressed=True
        if remaining:
            record['status']='waiting_saved_checkpoints';write_json(marker,record);time.sleep(30)
    record['status']='observations_complete_awaiting_final_reconciliation';write_json(marker,record)
    print(record['status'],len(record['records']),flush=True)


if __name__=='__main__':main()
