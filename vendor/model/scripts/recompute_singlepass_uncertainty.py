#!/usr/bin/env python3
"""Regenerate every dependent uncertainty output without altering acquisitions."""
from companion_paths import child_pythonpath
from companion_paths import configured_path
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from model_rg.provenance import sha256,write_json
REPO=Path(__file__).resolve().parents[1]
ROOT=Path(configured_path('data:model'))


def run(output):
    if not output.resolve().is_relative_to(ROOT):raise ValueError('Use the authorized experiment root')
    output.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
    def family(label):
        study=ROOT/f'critical-onepass-{label}-20260914';collective=ROOT/f'critical-collective-{label}-20260914'
        target=output/label;target.mkdir()
        jobs=[('joint-analysis.json','analyze_critical_joint.py',['--study',study,'--analysis',study/'analysis.json'],study/'joint-analysis.json'),
              ('prediction-analysis.json','analyze_critical_predictions.py',['--study',study,'--analysis',study/'analysis.json'],study/'prediction-analysis.json'),
              ('collective-analysis.json','analyze_critical_collectives.py',['--study',collective],collective/'analysis.json'),
              ('operator-moments.json','analyze_critical_operator_moments.py',['--study',collective],collective/'operator-moments.json')]
        rows=[]
        for name,script,args,prior in jobs:
            command=[sys.executable,str(REPO/'scripts'/script),*map(str,args),'--output',str(target/name)]
            start=time.monotonic()
            with (target/(name+'.log')).open('x') as log:subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            rows.append(dict(output=str(target/name),sha256=sha256(target/name),input_comparison=str(prior),input_comparison_sha256=sha256(prior),
                             command=command,seconds=time.monotonic()-start))
            print(label,name,'complete',flush=True)
        return rows
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(family,label) for label in ['coarse','refinement']]
        records=[row for f in futures for row in f.result()]
    # Every numerical leaf is compared, retaining arrays with different shapes
    # and every status/string change in an internal comparison record.
    changes=[]
    def compare(a,b,path):
        if isinstance(a,dict) and isinstance(b,dict):
            for k in sorted(a.keys()&b.keys()):compare(a[k],b[k],path+'/'+k)
        elif isinstance(a,list) and isinstance(b,list) and len(a)==len(b):
            for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
        elif a!=b and isinstance(a,(float,int)) and isinstance(b,(float,int)):
            changes.append(dict(path=path,before=a,after=b,absolute_change=abs(a-b)))
    for row in records:
        compare(json.loads(Path(row['input_comparison']).read_text()),json.loads(Path(row['output']).read_text()),row['output'])
    write_json(output/'comparison.json',dict(status='complete',records=records,numerical_changes=changes,
        policy='Same 10000-draw seed, complete paired counts, direct differences; original analyses remain intact.',scientific_updates=0))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
