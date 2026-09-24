#!/usr/bin/env python3
"""Run bounded CPU executor regressions and retain the two short-child records."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/row/experiments'))
from confirm.resource_executor import ResourceCaps,run_capped,EXECUTOR_CONTRACT


def main():
    validation=ROOT/'validation';validation.mkdir(exist_ok=True)
    test='vendor/row/experiments/tests/test_resource_executor.py'
    command=[sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',test,
             '--tb=short','--junitxml='+str(validation/'resource-execution.xml')]
    run=subprocess.run(command,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),
                       text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (validation/'resource-execution.log').write_text(run.stdout)
    if run.returncode:raise RuntimeError(run.stdout)
    cases=ET.parse(validation/'resource-execution.xml').findall('.//testcase')
    if any(case.find('skipped') is not None for case in cases):raise RuntimeError('Executor regressions must not skip')
    retained=[]
    with tempfile.TemporaryDirectory(prefix='pldr-resource-check-') as tmp:
        base=Path(tmp)
        for name,wall,output in [('terminal-output',5.0,1024),('short-deadline',.05,1024**2)]:
            record=run_capped([sys.executable,'-c',"import time;from pathlib import Path;time.sleep(.15);Path('payload.bin').write_bytes(b'x'*4096)"],
                device='cpu',output_root=base/name,record_path=base/(name+'.json'),
                caps=ResourceCaps(wall,2*1024**3,0,output,.5))
            expected='reached_output_bytes' if name=='terminal-output' else 'reached_wall_seconds'
            if record['technical_valid'] or record['cap_status']!=expected:raise RuntimeError('False successful resource record')
            if name=='terminal-output' and record['final_output_bytes']!=4096:raise RuntimeError('Final bytes omitted')
            retained.append(dict(case=name,record=record))
    sources=[test,'vendor/row/experiments/confirm/resource_executor.py','vendor/row/experiments/confirm/resource_worker.py',str(Path(__file__).relative_to(ROOT))]
    report=dict(status='passed',executor_contract=EXECUTOR_CONTRACT,tests_passed=len(cases),
        command=command,log_sha256=hashlib.sha256(run.stdout.encode()).hexdigest(),short_child_cases=retained,
        sources_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
        scope='Bounded CPU execution/admission tests, including all affected campaign consumers and reducers. Overlaps the row family suite. No native acquisition, forward call, or training update.')
    (validation/'resource-execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('short_child_cases','sources_sha256')},indent=2))

if __name__=='__main__':main()
