#!/usr/bin/env python
"""Reproduce a completed finite-response design at a fresh output destination.

Output relocation changes paths only. The mathematical design, seeds and full
source-block law are those of the selected producer; this is replication, not
new independent model evidence or an amplitude-selection tool.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
from pathlib import Path
import subprocess
import sys
from model_rg.provenance import sha256,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','run','worker','verify'])
    p.add_argument('--design',choices=['response','invisible'],required=True);p.add_argument('--study',required=True)
    p.add_argument('--heads',type=int);p.add_argument('--device');a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];study=Path(a.study).resolve()
    module=importlib.import_module('finite_response_study' if a.design=='response' else 'invisible_sector_study')
    if study.parents[1] != module.ROOT:
        raise ValueError("Choose DATA_ROOT/reproduction-group/design so the independent reducer resolves the same input root")
    module.OUT=study
    if a.phase=='prepare':module.prepare();return
    if a.phase=='worker':module.worker(a.heads,a.device);return
    if a.phase=='verify':
        script='analyze_finite_response.py' if a.design=='response' else 'verify_invisible_sector.py'
        # Native producer input roots remain independently recorded in the protocol.
        subprocess.run([sys.executable,str(repo/'scripts'/script),'--study',str(study)],check=True);return
    if (study/'launcher.json').exists():raise FileExistsError(study/'launcher.json')
    def run(heads,device):
        command=[sys.executable,str(Path(__file__).resolve()),'worker','--design',a.design,'--study',str(study),'--heads',str(heads),'--device',device]
        with (study/f'h{heads}.log').open('x') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True,cwd=repo)
        return dict(heads=heads,device=device,command=command)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run,4,'cuda:1'),pool.submit(run,14,'cuda:0')]
        results=[f.result() for f in futures]
    write_json(study/'launcher.json',dict(status='complete',protocol_sha256=sha256(study/'protocol.json'),records=results))

if __name__=='__main__':main()
