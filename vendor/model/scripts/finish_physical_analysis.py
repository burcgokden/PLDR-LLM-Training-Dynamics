#!/usr/bin/env python
"""Analyze and verify only after every frozen assessment case is complete."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',required=True);a=ap.parse_args()
    study=Path(a.study);cases=json.loads((study/'assessment/protocol.json').read_text())['cases']
    start=time.monotonic()
    while True:
        complete=True
        for c in cases:
            try:
                data=json.loads((study/'assessment'/c['name']/'manifest.json').read_text())
                complete=complete and data['status']=='complete'
            except (FileNotFoundError,json.JSONDecodeError):complete=False
        if complete:break
        if time.monotonic()-start>3*3600:raise TimeoutError('Complete assessment required')
        time.sleep(10)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='2')
    for name in ['analyze_physical_models.py','verify_physical_study.py','render_physical_results.py']:
        subprocess.run([sys.executable,str(Path(__file__).with_name(name)),'--study',str(study)],env=env,check=True)
        print('completed',name,flush=True)


if __name__=='__main__':main()
