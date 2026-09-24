#!/usr/bin/env python
"""Start frozen prediction tests only after their predecessor and forecasts finish."""
from companion_paths import child_pythonpath
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from model_rg.provenance import sha256, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--study', default='critical-scaling-20260906')
    p.add_argument('--predecessor', default='launcher-main-size-time-execution.json')
    p.add_argument('--forecast', default='analysis/frozen-forecasts')
    p.add_argument('--label', default='prediction-tests')
    a = p.parse_args()
    study = Path(a.root)/a.study
    repo = Path(__file__).resolve().parents[1]
    path = study/('stage-'+a.label+'.json')
    if path.exists():
        raise FileExistsError(path)
    protocol_names = ['clock-holdout.json','diffusive-size-map.json','environment-factorial.json']
    signatures = {n:sha256(study/'protocols'/n) for n in protocol_names}
    record = dict(status='waiting_for_completed_prerequisites',arguments=vars(a),
                  protocol_sha256=signatures,started_at=datetime.now(timezone.utc).isoformat())
    write_json(path,record)
    while True:
        predecessor = study/a.predecessor
        forecast_manifest = study/a.forecast/'manifest.json'
        state = json.loads(predecessor.read_text())['status']
        if state not in ['running','complete']:
            raise RuntimeError('Predecessor requires an explicit execution-failure resolution: '+state)
        if state=='complete' and forecast_manifest.exists():
            break
        time.sleep(30)
    forecast = study/a.forecast/'results.json'
    fm = json.loads(forecast_manifest.read_text())
    f = json.loads(forecast.read_text())
    if fm['status']!='complete' or fm['results_sha256']!=sha256(forecast):
        raise AssertionError('Forecast binding failed')
    if f['status']!='frozen_before_target_execution':
        raise AssertionError('Forecast must be frozen')
    for n,sig in signatures.items():
        if sha256(study/'protocols'/n)!=sig:
            raise AssertionError('Protocol changed while waiting')
        if n!='environment-factorial.json' and f['protocol_sha256'][n]!=sig:
            raise AssertionError('Forecast target changed')
    record.update(status='executing',predecessor_sha256=sha256(predecessor),
                  forecast_sha256=sha256(forecast),forecast_manifest_sha256=sha256(forecast_manifest),
                  execution_started_at=datetime.now(timezone.utc).isoformat())
    write_json(path,record)
    command = [sys.executable,str(repo/'scripts/run_scaling.py'),'--root',a.root,'--study',a.study,
               '--protocols',*protocol_names,'--label',a.label]
    env = dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4')
    with (study/'logs'/('launcher-'+a.label+'.log')).open('x') as log:
        result = subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
    record.update(status='complete' if result.returncode==0 else 'execution_failure',
                  returncode=result.returncode,finished_at=datetime.now(timezone.utc).isoformat(),command=command)
    write_json(path,record)
    if result.returncode:
        raise RuntimeError('Prediction test executor failed')


if __name__=='__main__':
    main()
