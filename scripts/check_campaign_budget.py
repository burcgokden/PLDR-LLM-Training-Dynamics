#!/usr/bin/env python3
"""Run bounded campaign regressions and retain split-invocation evidence."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/row/experiments'))
from confirm.campaign_budget import CONTRACT, Journal

def main():
    validation=ROOT/'validation';validation.mkdir(exist_ok=True)
    test='vendor/row/experiments/tests/test_campaign_budget.py'
    command=[sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',test,
             '--tb=short','--junitxml='+str(validation/'campaign-budget.xml')]
    run=subprocess.run(command,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),
                       text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
    (validation/'campaign-budget.log').write_text(run.stdout)
    if run.returncode:raise RuntimeError(run.stdout)
    cases=ET.parse(validation/'campaign-budget.xml').findall('.//testcase')
    if any(c.find('skipped') is not None for c in cases):raise RuntimeError('Campaign tests must not skip')
    spec=importlib.util.spec_from_file_location('campaign_fixture',ROOT/test)
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    with tempfile.TemporaryDirectory(prefix='pldr-campaign-check-') as tmp:
        plan=fixture.plan_at(Path(tmp)/'feasible')
        first=fixture.run(plan,{'n0'});second=fixture.run(plan,{'n1'});resume=fixture.run(plan)
        assert first['nodes'][0]['status']==second['nodes'][0]['status']=='passed'
        assert resume['aggregate_gpu_seconds']==second['aggregate_gpu_seconds']>first['aggregate_gpu_seconds']>0
        journal=Journal(plan)
        events={str(p.relative_to(journal.root)):json.loads(p.read_text()) for p in journal.root.rglob('*.json')}
        rejected_plan=fixture.plan_at(Path(tmp)/'excess',seconds=1)
        for node in rejected_plan['nodes']:node['caps']['wall_seconds']=5
        rejected=fixture.run(rejected_plan,cont=True)
        assert rejected['campaign_decision']=='rejected' and rejected['attempt_count']==0
        retained=dict(plan=plan,split_reports=[first,second],resume=resume,events=events,
                      infeasible_plan=rejected_plan,preflight_rejection=rejected)
    sources=[test,'vendor/row/experiments/confirm/campaign_budget.py',
             'vendor/row/experiments/confirm/campaign_plan_executor.py',
             'vendor/row/scripts/execute_block_normal_plan.py','vendor/row/scripts/execute_source_resolved_plan.py',
             'vendor/row/scripts/execute_orbitwise_plan.py',str(Path(__file__).relative_to(ROOT))]
    result=dict(status='passed',campaign_contract=CONTRACT,tests_passed=len(cases),command=command,
                sources_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
                log_sha256=hashlib.sha256(run.stdout.encode()).hexdigest(),retained=retained,
                scope='Exact ledger boundary checks and tiny CPU subprocesses, including separate CLI processes, persistence, retries, invalid history and concurrency. CUDA labels allocate no GPU. Overlaps row family tests. No scientific acquisition.')
    (validation/'campaign-budget.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('retained','sources_sha256')},indent=2))
if __name__=='__main__':main()
