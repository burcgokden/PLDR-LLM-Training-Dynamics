#!/usr/bin/env python
"""Check historical/executable status separation and fail-closed stage readiness."""
from companion_paths import child_pythonpath
import argparse,copy,json,os
from pathlib import Path
import subprocess,sys,tempfile
from confirmation_status import validate,render
from model_rg.qualification import QualificationError
from model_rg.provenance import sha256,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--status',default='docs/confirmation-status.json');p.add_argument('--output',required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];base=json.loads(Path(a.status).read_text());checks=[]
    validate(base);checks.append(dict(name='complete native qualifications validate',passed=True))
    if render(base)!=render(json.loads(json.dumps(base,sort_keys=True))):raise ValueError('Nondeterministic rendering')
    checks.append(dict(name='key-order independent rendering',passed=True))
    changed=copy.deepcopy(base)
    one=next(x for x in changed['stages'] if x['id']=='P1');three=next(x for x in changed['stages'] if x['id']=='P3')
    one['executable_qualification']=three['executable_qualification']
    try:validate(changed)
    except QualificationError:checks.append(dict(name='unrelated native qualification rejected',passed=True))
    else:raise ValueError('Unrelated qualification accepted')
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1')
    for optimized in [False,True]:
        for stage in ['P2','P4']:
            cmd=[sys.executable]+(['-O'] if optimized else [])+[str(repo/'scripts/confirmation_status.py'),'require-ready','--stage',stage,'--status',str(Path(a.status).resolve())]
            result=subprocess.run(cmd,capture_output=True,text=True,env=env,cwd=repo)
            if result.returncode==0 or 'no current complete native qualification' not in result.stderr:raise ValueError('Unimplemented stage was not refused')
            checks.append(dict(name=stage+' refusal',optimized=optimized,passed=True))
    write_json(a.output,dict(status='passed',checks=checks,runner_sha256=sha256(__file__),checker_sha256=sha256(repo/'scripts/confirmation_status.py'),input_status_sha256=sha256(a.status)))
    print('Passed',len(checks),'status checks')

if __name__=='__main__':main()
