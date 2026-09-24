#!/usr/bin/env python
"""Exercise the real completed-training parser through the shell wrapper."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from unittest.mock import patch


def parser_contract(wrapper, extra=None):
    repo=Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='training-wrapper-',dir='/tmp') as directory:
        folder=Path(directory);shim=folder/'python';capture=folder/'argv.json'
        shim.write_text('#!'+sys.executable+'\nimport json,os,sys\nwith open(os.environ["RG_ARGV_CAPTURE"],"w") as f: json.dump(sys.argv[1:],f)\n')
        shim.chmod(0o755);report=folder/'report with spaces.json'
        env=dict(os.environ,MODEL_RG_REPO=str(repo),MODEL_RG_DATA_ROOT=str(folder/'inputs'),PATH=str(folder)+os.pathsep+os.environ['PATH'],RG_ARGV_CAPTURE=str(capture))
        args=['--output',str(report)] if extra is None else extra
        subprocess.run(['sh',str(wrapper),*args],env=env,cwd=repo,check=True,capture_output=True,text=True)
        argv=json.loads(capture.read_text());original=argparse.ArgumentParser.parse_args;parsed={}
        class Parsed(BaseException):pass
        def stop(parser,*a,**kw):
            parsed.update(vars(original(parser,*a,**kw)));raise Parsed()
        status=None
        with patch.object(sys,'argv',argv),patch.object(argparse.ArgumentParser,'parse_args',stop),contextlib.redirect_stderr(io.StringIO()):
            try:runpy.run_path(str(repo/'scripts/verify_training.py'),run_name='__main__')
            except Parsed:status=0
            except SystemExit as error:status=error.code
        assert status is not None and not report.exists()
        return dict(status=status,argv=argv,parsed=parsed,expected_output=str(report),scientific_body_executed=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--wrapper',required=True);a=p.parse_args()
    print(json.dumps(parser_contract(Path(a.wrapper).resolve()),indent=2))


if __name__=='__main__':main()
