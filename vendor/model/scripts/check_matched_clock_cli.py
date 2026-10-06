#!/usr/bin/env python3
"""Actual matched-clock CLI admission checks with scientific boundaries intercepted."""
from companion_paths import child_pythonpath
from companion_paths import configured_path
import argparse
import copy
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import numpy as np
from model_rg.provenance import sha256,write_json

REPO=Path(__file__).resolve().parents[1]
ROOT=Path(configured_path('data:model'))
PARENT=ROOT/'matched-clock-onepass-20260916'


def probe(output,optimized):
    from unittest.mock import patch
    from run_matched_clock import SOURCES
    study=output/'prepared';script=REPO/'scripts/run_matched_clock.py'
    subprocess.run([sys.executable,'-B',str(script),'prepare','--study',str(study),'--reproduce-panel',str(PARENT)],cwd=REPO,check=True)
    original=json.loads((study/'protocol.json').read_text());qoriginal=json.loads((PARENT/'qualification/manifest.json').read_text())
    (study/'qualification').mkdir();selection=(study/'selection.npz').read_bytes()
    variants=[('valid',lambda p,q:None),('empty-sources',lambda p,q:p.update(source_sha256={})),
        ('empty-inputs',lambda p,q:p.update(input_sha256={})),('missing-contract',lambda p,q:p.pop('admission_contract')),
        ('unknown-role',lambda p,q:p['source_sha256'].update({'scripts/foreign.py':'0'*64})),
        ('wrong-source',lambda p,q:p['source_sha256'].update({SOURCES[0]:'0'*64})),
        ('wrong-input',lambda p,q:p['input_sha256'].update({next(iter(p['input_sha256'])):'0'*64})),
        ('wrong-selection',lambda p,q:p.update(selection_sha256='0'*64)),
        ('wrong-reservation',lambda p,q:p.update(reservation_sha256='0'*64)),
        ('wrong-seed',lambda p,q:p.update(shared_seed=1)),
        ('duplicate-job',lambda p,q:p['jobs'].__setitem__(1,copy.deepcopy(p['jobs'][0]))),
        ('wrong-family',lambda p,q:p['family'].update(consumed_fraction=.5)),
        ('boolean-control',lambda p,q:p['jobs'][0].update(control=False)),
        ('wrong-ages',lambda p,q:p.update(ages=[0,.5,1.])),
        ('wrong-cache-conditions',lambda p,q:p.update(cache_conditions=[])),
        ('wrong-target',lambda p,q:p['targets'].update(centered_rms=.5)),
        ('wrong-contexts',lambda p,q:p.update(contexts=32)),
        ('wrong-calls',lambda p,q:p.update(scientific_updates=1)),
        ('unqualified-replay',lambda p,q:q.update(optimizer_replay_exact=False)),
        ('unqualified-cache',lambda p,q:q.update(operator_replay_exact=False)),
        ('wrong-qualification-calls',lambda p,q:q.update(forwards=7))]
    for field in ['source_sha256','input_sha256']:
        for name in original[field]:variants.append(('missing-'+field+'-'+name,lambda p,q,f=field,n=name:p[f].pop(n)))
    def alter_selection(p,q):
        with np.load(study/'selection.npz') as a:arrays={k:a[k].copy() for k in a.files}
        arrays['probes'][0,0]=(int(arrays['probes'][0,0])+1)%32000
        np.savez_compressed(study/'selection.npz',**arrays);p['selection_sha256']=sha256(study/'selection.npz')
    variants.append(('refrozen-wrong-tokens',alter_selection))
    snapshot=study/'executed-source'/SOURCES[0];snapshot_bytes=snapshot.read_bytes()
    variants.append(('changed-source-snapshot',lambda p,q:snapshot.write_bytes(b'# altered snapshot\n')))
    def alter_order(p,q):
        with np.load(study/'selection.npz') as a:arrays={k:a[k].copy() for k in a.files}
        arrays['order'][[0,1]]=arrays['order'][[1,0]]
        np.savez_compressed(study/'selection.npz',**arrays);p['selection_sha256']=sha256(study/'selection.npz')
    variants.append(('refrozen-wrong-order',alter_order))
    rows=[]
    def boundary(*args,**kwargs):raise SystemExit(77)
    for action in ['run','worker']:
        for name,change in variants:
            (study/'selection.npz').write_bytes(selection);snapshot.write_bytes(snapshot_bytes)
            p=copy.deepcopy(original);q=copy.deepcopy(qoriginal);change(p,q)
            write_json(study/'protocol.json',p);q['protocol_sha256']=sha256(study/'protocol.json');write_json(study/'qualification/manifest.json',q)
            sys.argv=[str(script),action,'--study',str(study),'--index','0','--device','cuda:0'];code=0;error=None
            try:
                with patch('subprocess.run',boundary),patch('model_rg.training.TrainingModel',boundary),patch('torch.cuda.set_device'),patch('torch.cuda.reset_peak_memory_stats'):
                    runpy.run_path(str(script),run_name='__main__')
            except SystemExit as e:code=e.code
            except Exception as e:code=1;error=repr(e)
            created=[key for key in ['runs','logs'] if (study/key).exists()]
            ok=code==77 if name=='valid' else code not in [0,77] and not created
            rows.append(dict(action=action,variant=name,optimized=optimized,returncode=code,error=error,passed=ok,no_forbidden_output=name=='valid' or not created))
            if not ok:write_json(output/'partial.json',rows);raise RuntimeError(str(rows[-1]))
            for key in created:shutil.rmtree(study/key)
    (study/'selection.npz').write_bytes(selection);snapshot.write_bytes(snapshot_bytes);write_json(study/'protocol.json',original)
    qoriginal['protocol_sha256']=sha256(study/'protocol.json');write_json(study/'qualification/manifest.json',qoriginal)
    for label,index,device in [('negative-index','-1','cuda:0'),('excess-index','48','cuda:0'),('missing-index',None,'cuda:0'),('invalid-device','0','cuda:2')]:
        sys.argv=[str(script),'worker','--study',str(study),'--device',device]+(['--index',index] if index is not None else [])
        code=0;error=None
        try:
            with patch('model_rg.training.TrainingModel',boundary),patch('torch.cuda.set_device'),patch('torch.cuda.reset_peak_memory_stats'):runpy.run_path(str(script),run_name='__main__')
        except SystemExit as e:code=e.code
        except Exception as e:code=1;error=repr(e)
        clean=not any((study/name).exists() for name in ['runs','logs']);ok=code not in [0,77] and clean
        rows.append(dict(action='worker',variant=label,optimized=optimized,returncode=code,error=error,passed=ok,no_forbidden_output=clean))
        if not ok:raise RuntimeError(str(rows[-1]))
    write_json(output/'cases.json',rows)


def main(output):
    if output.exists() or not output.resolve().is_relative_to(ROOT):raise ValueError('Fresh authorized fixture directory required')
    output.mkdir();cases=[]
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model")+os.pathsep+str(REPO/'scripts'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    for optimized in [False,True]:
        dest=output/('optimized' if optimized else 'ordinary');dest.mkdir()
        with (dest/'execution.log').open('x') as log:subprocess.run([sys.executable,*(['-O'] if optimized else []),'-B',__file__,'--probe','--output',str(dest)],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        cases+=json.loads((dest/'cases.json').read_text())
    from run_matched_clock import SOURCES
    result=dict(status='passed',schema='matched-clock-cli-v1',case_count=len(cases),cases=cases,checker_sha256=sha256(__file__),
        tested_sources={n:sha256(REPO/n) for n in SOURCES},scientific_forwards=0,optimizer_updates=0,
        scope='Actual prepared named-panel fixtures and ordinary/optimized queue/worker CLIs; boundaries are intercepted, never scientific acquisition.')
    write_json(output/'verification.json',result);print(dict(status='passed',cases=len(cases)),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);a.add_argument('--probe',action='store_true');v=a.parse_args()
    if v.probe:probe(v.output,not __debug__)
    else:main(v.output)
