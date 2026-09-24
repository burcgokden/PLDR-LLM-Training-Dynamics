#!/usr/bin/env python3
"""Freeze fresh source inventories and qualify three native single-pass routes."""
from companion_paths import child_pythonpath
from companion_paths import legacy_path
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from model_rg.provenance import sha256, write_json

REPO=Path(__file__).resolve().parents[1]
ROOT=Path(legacy_path('/pldr-data/model'))


def run(output):
    output=output.resolve()
    if not output.is_relative_to(ROOT) or output==ROOT:raise ValueError('Use the authorized experiment root')
    output.mkdir(parents=True,exist_ok=False)
    design=dict(name='native-source-qualification',heads=[24],controls=[2.],seeds=[9152601,9152602],
                environments=[1],steps=256,shared_seed=9152500)
    write_json(output/'design.json',design)
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    def command(label,script,*args):
        cmd=[sys.executable,str(REPO/'scripts'/script),*map(str,args)];start=time.monotonic()
        with (output/(label+'.log')).open('x') as log:
            subprocess.run(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        write_json(output/(label+'-command.json'),dict(command=cmd,seconds=time.monotonic()-start,
                   log_sha256=sha256(output/(label+'.log'))))
        print(label,'complete',flush=True)
    command('base-prepare','run_critical_onepass.py','prepare','--study',output/'base','--design',output/'design.json')
    command('shared-prepare','run_critical_shared_paths.py','prepare','--study',output/'shared','--design',output/'design.json',
            '--selection',REPO/'internal/revision25/refinement-selection.json')
    command('matched-prepare','run_matched_onepass.py','prepare','--study',output/'matched')
    def base_queue():
        command('base-qualify','run_critical_onepass.py','qualify','--study',output/'base','--device','cuda:0')
        command('matched-qualify','run_matched_onepass.py','qualify','--study',output/'matched','--device','cuda:0')
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(base_queue),pool.submit(command,'shared-qualify','run_critical_shared_paths.py','qualify',
                 '--study',output/'shared','--device','cuda:1')]
        for f in futures:f.result()
    records={}
    for route in ['base','shared','matched']:
        p=output/route/'qualification/verification.json';record=json.loads(p.read_text())
        if record['status']!='passed' or record['scientific_updates']!=0:raise ValueError('Qualification failed')
        records[route]=dict(path=str(p),sha256=sha256(p),qualification_updates=record.get('qualification_updates',record.get('native_updates')))
    total=sum(r['qualification_updates'] for r in records.values())
    if total!=576:raise ValueError('Qualification update accounting changed')
    write_json(output/'verification.json',dict(schema='singlepass-routes-qualification-v1',status='passed',
        scientific_updates=0,qualification_updates=total,routes=records,driver_sha256=sha256(__file__)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
