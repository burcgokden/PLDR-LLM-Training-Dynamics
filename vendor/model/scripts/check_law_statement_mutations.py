#!/usr/bin/env python
"""Reject omitted and wrongly transported defects in the finite Lean recurrence."""
from companion_paths import configured_path
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from model_rg.provenance import sha256, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    source=(repo/'ModelRG/Scaling.lean').read_text()
    env=dict(os.environ,ELAN_HOME=configured_path('tools:elan'),
             PATH=configured_path('tools:elan/bin')+os.pathsep+os.environ['PATH'])
    records=[]
    with tempfile.TemporaryDirectory(prefix='law-bound-mutation-',dir='/tmp') as temp:
        temp=Path(temp);shutil.copytree(repo/'ModelRG',temp/'ModelRG')
        for name in ['ModelRG.lean','lakefile.toml','lake-manifest.json','lean-toolchain']:shutil.copy2(repo/name,temp/name)
        (temp/'.lake').mkdir();(temp/'.lake/packages').symlink_to(repo/'.lake/packages',target_is_directory=True)
        shutil.copytree(repo/'.lake/build',temp/'.lake/build')
        start=source.index('theorem nonstationary_error_bound')
        old=source[start:]
        changes={
            'omitted_defect':old.replace('d j * ∏ k ∈ Finset.Ico (j + 1) m, L k','(0 : ℝ) * ∏ k ∈ Finset.Ico (j + 1) m, L k'),
            'wrong_chronological_transport':old.replace('Finset.Ico (j + 1) m','Finset.Ico 0 j'),
        }
        for name,changed in changes.items():
            assert changed!=old
            (temp/'ModelRG/Scaling.lean').write_text(source[:start]+changed)
            result=subprocess.run(['lake','build','ModelRG.Scaling'],cwd=temp,env=env,
                text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            (out/(name+'.log')).write_text(result.stdout)
            if result.returncode==0:raise AssertionError('Incorrect recurrence was accepted: '+name)
            records.append(dict(name=name,rejected=True,returncode=result.returncode,
                log_sha256=sha256(out/(name+'.log'))))
    write_json(out/'verification.json',dict(status='passed',mutations=records,
        checker_sha256=sha256(__file__),module_sha256=sha256(repo/'ModelRG/Scaling.lean'),
        scope='Both incorrect chronological product/sum statements fail elaboration in isolated copies.'))
    print('Rejected both recurrence-content mutations',flush=True)


if __name__=='__main__':main()
