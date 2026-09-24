#!/usr/bin/env python
"""Run complete snapshots only after all selected training and CPU observations finish."""
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
    p = argparse.ArgumentParser();p.add_argument('--root', required=True)
    p.add_argument('--study', default='critical-scaling-20260906');a = p.parse_args()
    study = Path(a.root)/a.study;repo = Path(__file__).resolve().parents[1]
    marker = study/'analysis-completion-stage.json'
    if marker.exists():raise FileExistsError(marker)
    record = dict(status='waiting_for_complete_observation_selection', arguments=vars(a),
                  started_at=datetime.now(timezone.utc).isoformat(), records=[])
    write_json(marker, record)
    dependencies = [study/'launcher-target-collectives.json', study/'followup-preparation.json',
                    study/'launcher-boundary-and-extended.json']
    while not all(path.exists() and json.loads(path.read_text())['status'] == 'complete' for path in dependencies):
        for path in dependencies:
            if path.exists() and any(word in json.loads(path.read_text())['status'] for word in ['failure', 'failed']):
                raise RuntimeError('Selected data need resolution: '+str(path))
        time.sleep(30)
    record.update(status='complete_snapshot_analysis_running',
                  predecessor_sha256={str(path):sha256(path) for path in dependencies})
    write_json(marker, record)
    env = dict(os.environ, PYTHONPATH=child_pythonpath("model"), PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='4')
    tasks = [('size-time', 'analyze_size_time.py', ['--include-current']),
             ('collectives', 'analyze_scaling_collectives.py', ['--precisions', 'float32', 'float64']),
             ('complete', 'analyze_scaling_study.py', ['--wait'])]
    for label, script, extra in tasks:
        command = [sys.executable, str(repo/'scripts'/script), '--root', a.root, '--study', a.study,
                   '--output', str(study/'analysis'/label), *extra]
        with (study/'logs'/('complete-'+label+'.log')).open('x') as log:
            run = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        result = dict(label=label, returncode=run.returncode, command=command)
        record['records'].append(result);write_json(marker, record)
        if run.returncode:raise RuntimeError('Analysis failed: '+label)
        result['manifest_sha256'] = sha256(study/'analysis'/label/'manifest.json')
        write_json(marker, record)
    record['status'] = 'complete';write_json(marker, record)


if __name__ == '__main__':main()
