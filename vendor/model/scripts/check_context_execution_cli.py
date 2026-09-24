#!/usr/bin/env python3
"""Actual preparation and ordinary/optimized CLI regression matrix; no native work."""
from companion_paths import child_pythonpath
from companion_paths import legacy_path
import argparse
import copy
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
from model_rg.provenance import sha256,write_json

REPO=Path(__file__).resolve().parents[1]
ROOT=Path(legacy_path('/pldr-data/model'))
ROUTES=[('context_categorical_study.py','context-risk-128-20260915'),
        ('run_operator_cache_study.py','operator-cache-confirmation-20260916'),
        ('run_cache_risk_study.py','cache-risk-transfer-20260916')]


def probe(output,mode):
    from unittest.mock import patch
    import torch
    from numerical_validation import load_json_strict
    rows=[]
    for script,old in ROUTES:
        study=output/script.removesuffix('.py')
        command=[sys.executable,'-B',str(REPO/'scripts'/script),'prepare','--study',str(study),'--reproduce-panel',str(ROOT/old)]
        if script=='context_categorical_study.py':command+=['--contexts','128']
        subprocess.run(command,cwd=REPO,check=True)
        original=load_json_strict((study/'protocol.json').read_text())
        variants=[('valid',lambda p:None),('empty-source',lambda p:p.update(source_sha256={})),
                  ('empty-external',lambda p:p.update(input_sha256={})),
                  ('wrong-native',lambda p:p['input_sha256'].update({str(ROOT/'assets/PLDR-LLM-v51-SOC-110M-1/modeling_pldrllm.py'):'0'*64})),
                  ('wrong-prefix-or-context',lambda p:p.update(contexts=129)),
                  ('boolean-batch',lambda p:p.update(batch_size=True)),
                  ('duplicate-job',lambda p:p['jobs'].__setitem__(1,copy.deepcopy(p['jobs'][0]))),
                  ('wrong-checkpoint',lambda p:p['jobs'][0].update(checkpoint_sha256='0'*64)),
                  ('wrong-selection',lambda p:p.update(selection_sha256='0'*64)),
                  ('wrong-reservation',lambda p:p.update(reservation_sha256='0'*64)),
                  ('foreign-external',lambda p:p['input_sha256'].update({'/tmp/foreign':'0'*64})),
                  ('foreign-source',lambda p:p['source_sha256'].update({'scripts/foreign.py':'0'*64}))]
        for field in ['source_sha256','input_sha256']:
            for role in original[field]:
                variants.append(('missing-'+field+'-'+role,lambda p,f=field,k=role:p[f].pop(k)))
        def boundary(*args,**kwargs):raise SystemExit(77)
        for route in ['run','worker']:
            for label,change in variants:
                p=copy.deepcopy(original);change(p);write_json(study/'protocol.json',p)
                sys.argv=[str(REPO/'scripts'/script),route,'--study',str(study)]
                if route=='worker':sys.argv+=['--index','0','--device','cuda:0']
                code=0;error=None
                try:
                    with patch('subprocess.run',boundary),patch('torch.load',boundary),patch('model_rg.training.TrainingModel',boundary),patch('torch.cuda.set_device'),patch('torch.cuda.reset_peak_memory_stats'):
                        runpy.run_path(str(REPO/'scripts'/script),run_name='__main__')
                except SystemExit as e:code=e.code
                except Exception as e:code=1;error=repr(e)
                created=[name for name in ['runs','logs'] if (study/name).exists()]
                expected=label=='valid'
                ok=code==77 if expected else code not in [0,77] and not created
                rows.append(dict(script=script,route=route,variant=label,optimized=mode,returncode=code,error=error,created=created,passed=ok))
                if not ok:write_json(output/'partial.json',rows);raise RuntimeError(str(rows[-1]))
                for name in created:shutil.rmtree(study/name)
        write_json(study/'protocol.json',original)
        print(script,len(rows),'checks',flush=True)
    # Exercise the maintained risk wrapper itself, including its frozen plan.
    study=output/'context_categorical_study'
    source_names=['scripts/run_context_risk_study.py','scripts/analyze_context_risk.py']
    for name in source_names:
        dest=study/'analysis-source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/name,dest)
    protocol=load_json_strict((study/'protocol.json').read_text())
    plan=dict(schema='context-risk-plan-v1',status='frozen_before_acquisition',protocol_sha256=sha256(study/'protocol.json'),
        primary_resolution=8192,target=.25,resolutions=protocol['sizes'],replicas_per_cell=6,contexts=128,
        conditions=[[8,0],[8,1.5],[24,0],[24,1.5]],primary_test='Complete aggregate grid',descriptive_tests=['Every context'],
        uncertainty='Conditional paired seed units',adaptation_policy='No refit',new_training_updates=0,
        source_sha256={name:sha256(REPO/name) for name in source_names})
    for label,change in [('valid',lambda p:None),('empty-sources',lambda p:p.update(source_sha256={})),
                         ('missing-source',lambda p:p['source_sha256'].pop(source_names[0])),
                         ('missing-analysis',lambda p:p['source_sha256'].pop(source_names[1])),
                         ('wrong-target',lambda p:p.update(target=.3)),('wrong-primary',lambda p:p.update(primary_resolution=128)),
                         ('wrong-contexts',lambda p:p.update(contexts=16)),('wrong-status',lambda p:p.update(status='foreign'))]:
        value=copy.deepcopy(plan);change(value);write_json(study/'risk-plan.json',value)
        sys.argv=['run_context_risk_study.py','--study',str(study),'--resume'];code=0;error=None
        try:
            with patch('subprocess.run',boundary):runpy.run_path(str(REPO/'scripts/run_context_risk_study.py'),run_name='__main__')
        except SystemExit as e:code=e.code
        except Exception as e:code=1;error=repr(e)
        log=study/'execution.log';ok=(code==77) if label=='valid' else code not in [0,77] and not log.exists()
        rows.append(dict(script='run_context_risk_study.py',route='resume',variant=label,optimized=mode,returncode=code,error=error,passed=ok))
        if not ok:raise RuntimeError(str(rows[-1]))
        if log.exists():log.unlink()
    write_json(study/'risk-plan.json',plan)
    write_json(output/'cases.json',rows)


def main(output):
    if output.exists() or not output.resolve().is_relative_to(ROOT):raise ValueError('Fresh authorized fixture directory required')
    output.mkdir(parents=True)
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model")+os.pathsep+str(REPO/'scripts'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    cases=[]
    for mode in [False,True]:
        dest=output/('optimized' if mode else 'ordinary');dest.mkdir()
        with (dest/'execution.log').open('x') as log:
            subprocess.run([sys.executable,*(['-O'] if mode else []),'-B',__file__,'--probe','--output',str(dest)],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        cases+=json.loads((dest/'cases.json').read_text())
    result=dict(status='passed',schema='context-execution-cli-v1',case_count=len(cases),cases=cases,
        checker_sha256=sha256(__file__),scientific_forwards=0,optimizer_updates=0,
        tested_sources={str(p.relative_to(REPO)):sha256(p) for p in [REPO/'scripts'/n for n,_ in ROUTES]+[REPO/'scripts/context_execution_contract.py',REPO/'scripts/context_reservations.py',REPO/'scripts/cache_state_contract.py',REPO/'scripts/run_context_risk_study.py']})
    write_json(output/'verification.json',result);print(dict(status='passed',cases=len(cases)),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);a.add_argument('--probe',action='store_true');args=a.parse_args()
    if args.probe:probe(args.output,not __debug__)
    else:main(args.output)
