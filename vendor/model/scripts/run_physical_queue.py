#!/usr/bin/env python
"""Execute one frozen model-width queue on one GPU."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


from physical_design import admit, verify_completion


def run(study,heads,device):
    study=Path(study);spec=admit(study).spec
    if heads not in (4,8):raise ValueError('Unregistered queue width')
    repo=Path(__file__).resolve().parents[1]
    for case in spec['cases']:
        if case['heads']!=heads:continue
        manifest=study/case['name']/'manifest.json'
        if manifest.exists():
            verify_completion(study,spec,case['name'])
            continue
        if (study/case['name']).exists():raise ValueError('Incomplete worker directory')
        print('starting',case['name'],flush=True)
        with (study/(case['name']+'.log')).open('x') as log:
            subprocess.run([sys.executable,str(repo/'scripts/physical_study.py'),'worker',
                '--study',str(study),'--name',case['name'],'--device',device],
                cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
        verify_completion(study,spec,case['name'])
        print('finished',case['name'],flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',required=True)
    ap.add_argument('--heads',required=True,type=int);ap.add_argument('--device',required=True)
    a=ap.parse_args();run(a.study,a.heads,a.device)


if __name__=='__main__':main()
