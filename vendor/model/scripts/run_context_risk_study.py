#!/usr/bin/env python3
"""Run the fixed context-risk experiment with one native worker per local GPU."""
from companion_paths import child_pythonpath
from companion_paths import legacy_path
import argparse
import os
from pathlib import Path
import subprocess
import sys
from model_rg.provenance import sha256, write_json
from numerical_validation import load_json_strict
from context_execution_contract import admit as shared_admit
from cache_state_contract import sources as validate_sources, same, keys

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(legacy_path('/pldr-data/model'))


def run(args):
    study=args.study.resolve()
    if study==ROOT or not study.is_relative_to(ROOT):
        raise ValueError('Use the authorized experiment root')
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    def call(script,*parameters,log=None):
        command=[sys.executable,str(REPO/'scripts'/script),*map(str,parameters)]
        subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT if log else None,check=True)
    if not args.resume:
        parameters=['prepare','--study',study,'--contexts',args.contexts]
        if args.start is not None: parameters += ['--start',args.start]
        if args.reproduce_panel is not None: parameters += ['--reproduce-panel',args.reproduce_panel]
        for excluded in args.exclude_study:parameters+=['--exclude-study',excluded]
        call('context_categorical_study.py',*parameters)
        protocol=load_json_strict((study/'protocol.json').read_text())
        sources=['scripts/run_context_risk_study.py','scripts/analyze_context_risk.py']
        plan=dict(schema='context-risk-plan-v1',status='frozen_before_acquisition',
            protocol_sha256=sha256(study/'protocol.json'),primary_resolution=8192,target=.25,
            resolutions=protocol['sizes'],replicas_per_cell=6,contexts=protocol['contexts'],
            conditions=[[8,0],[8,1.5],[24,0],[24,1.5]],
            primary_test='Report every width/control aggregate centered RMS against 0.25 at 8192 tokens. Retain every result.',
            descriptive_tests=['All context RMS values and exceedance counts at every resolution',
                'Native-variance-weighted RMS identity and finite Markov tail bound'],
            uncertainty='Contexts and controls share trained replicas; no binomial intervals or uniform prompt claim.',
            adaptation_policy='No refit, altered resolution or target, or replacement of failures.',
            new_training_updates=0,source_sha256={name:sha256(REPO/name) for name in sources})
        write_json(study/'risk-plan.json',plan)
        for name in sources:
            dest=study/'analysis-source'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes((REPO/name).read_bytes())
    protocol=shared_admit(study)
    plan=load_json_strict((study/'risk-plan.json').read_text())
    keys(plan, ['schema','status','protocol_sha256','primary_resolution','target','resolutions','replicas_per_cell','contexts','conditions','primary_test','descriptive_tests','uncertainty','adaptation_policy','new_training_updates','source_sha256'], 'risk-plan roles')
    same(plan['schema'],'context-risk-plan-v1','risk-plan schema')
    same(plan['status'],'frozen_before_acquisition','risk-plan status')
    for key,value in dict(primary_resolution=8192,target=.25,resolutions=protocol['sizes'],replicas_per_cell=6,contexts=protocol['contexts'],conditions=[[8,0],[8,1.5],[24,0],[24,1.5]],new_training_updates=0).items():same(plan[key],value,'risk plan '+key)
    if 8192 not in protocol['sizes']: raise ValueError('Missing primary resolution')
    validate_sources(plan['source_sha256'],['scripts/run_context_risk_study.py','scripts/analyze_context_risk.py'],REPO)
    validate_sources(plan['source_sha256'],['scripts/run_context_risk_study.py','scripts/analyze_context_risk.py'],study/'analysis-source')
    if plan['protocol_sha256']!=sha256(study/'protocol.json'):raise ValueError('Changed frozen protocol')
    for name,digest in plan.get('source_sha256',{}).items():
        if sha256(REPO/name)!=digest:raise ValueError('Changed frozen analysis source: '+name)
    with (study/'execution.log').open('a' if args.resume else 'x') as log:
        call('context_categorical_study.py','run','--study',study,log=log)
    call('analyze_categorical_scales.py','--study',study,'--output',study/'analysis')
    call('verify_categorical_scales.py','--analysis',study/'analysis/analysis.json','--output',study/'analysis/verification.json')
    call('analyze_context_risk.py','--study',study,'--published-analysis',study/'analysis/analysis.json',
         '--output',study/'context-disaggregation.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--start',type=int)
    p.add_argument('--contexts',type=int,default=128);p.add_argument('--resume',action='store_true')
    p.add_argument('--exclude-study',type=Path,action='append',default=[
        ROOT/'context-categorical-20260915',ROOT/'context-transfer-64-20260915'])
    p.add_argument('--reproduce-panel',type=Path)
    run(p.parse_args())
