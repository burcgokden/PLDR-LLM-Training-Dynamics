#!/usr/bin/env python
"""Parse every emitted outer-study command through its actual Python parser."""
from companion_paths import legacy_path
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from model_rg.provenance import sha256,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    output=Path(a.output).resolve()
    if output.exists():raise FileExistsError(output)
    repo=Path(__file__).resolve().parents[1];wrapper=repo/'scripts/workspace-wrappers/outer-transfer-study.sh'
    records=[]
    with tempfile.TemporaryDirectory(prefix='outer-wrapper-',dir='/tmp') as directory:
        temp=Path(directory);shim=temp/'python'
        shim.write_text('#!'+sys.executable+'''\nimport argparse,json,runpy,sys
from pathlib import Path
argv=sys.argv[1:];script=Path(argv[0]).resolve();sys.path.insert(0,str(script.parent));sys.argv=argv
original=argparse.ArgumentParser.parse_args
def stop(parser,*a,**kw):
    parsed=original(parser,*a,**kw)
    print(json.dumps(dict(argv=argv,parsed=vars(parsed),scientific_body_executed=False)))
    raise SystemExit(0)
argparse.ArgumentParser.parse_args=stop
runpy.run_path(str(script),run_name='__main__')
''');shim.chmod(0o755)
        data=temp/'data with spaces';study=temp/'study with spaces';render=temp/'render with spaces'
        env=dict(os.environ,PATH=str(temp)+os.pathsep+os.environ['PATH'],MODEL_RG_PYTHON='python',
                 MODEL_RG_REPO=str(repo),MODEL_RG_DATA_ROOT=str(data),PYTHONDONTWRITEBYTECODE='1')
        for phase in ['data','freeze','qualify','run','analyze','replay','verify-data','verify','render']:
            command=['sh',str(wrapper),phase,str(study),str(render)]
            result=subprocess.run(command,cwd=temp,env=env,text=True,capture_output=True,check=True)
            parsed=json.loads(result.stdout);assert parsed['parsed']['study']==str(study)
            if 'root' in parsed['parsed']:assert parsed['parsed']['root']==str(data)
            if phase=='render':assert parsed['parsed']['output']==str(render)
            records.append(dict(phase=phase,**parsed))
        assert not data.exists() and not study.exists() and not render.exists()
    installed=[repo.parent/'scripts/outer-transfer-study.sh',Path(legacy_path('/pldr-data/model/scripts/outer-transfer-study.sh'))]
    assert all(sha256(path)==sha256(wrapper) for path in installed)
    write_json(output,dict(status='passed',phases=records,wrapper_sha256=sha256(wrapper),checker_sha256=sha256(__file__),
        installed={str(path):sha256(path) for path in installed},scope='Actual child parsers with paths containing spaces and overridden data root; no scientific body executed.'))
    print('Passed all nine outer-study wrapper phases.',flush=True)


if __name__=='__main__':main()
