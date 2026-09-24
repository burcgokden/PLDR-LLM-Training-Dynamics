#!/usr/bin/env python
"""Run the full independent scheduled-state reconstruction for one selected path."""
import argparse
import json
from pathlib import Path

import torch

from model_rg.provenance import sha256, write_json
from verify_scaling_raw import BoundFiles
from verify_scheduled_raw import verify_training


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-20260908');p.add_argument('--run-id',required=True)
    a=p.parse_args();root=Path(a.root).resolve();study=root/a.study;repo=Path(__file__).resolve().parents[1]
    output=study/'verification/training'/(a.run_id+'.json')
    if output.exists():raise FileExistsError(output)
    selection=study/'protocols/regime-training-selection.json';spec=json.loads(selection.read_text())
    jobs=[j for j in spec['jobs'] if j['run_id']==a.run_id]
    if len(jobs)!=1:raise AssertionError('Exactly one selected scientific native path is required')
    execution=study/'protocols/executions'/(a.run_id+'.json');bound=json.loads(execution.read_text())
    if bound['source_selection']!={'path':str(selection),'sha256':sha256(selection)}:
        raise AssertionError('The execution selection changed')
    files=BoundFiles();torch.set_num_threads(4)
    record=verify_training(root,study,(execution,bound,bound['jobs'][0]),files)
    write_json(output,dict(schema='scheduled-single-path-reconstruction-v1',status='complete',trajectory=record,
        selection_sha256=sha256(selection),verified_files={key[0]:value for key,value in files.cache.items()},
        verifier_sources={n:sha256(repo/n) for n in ['scripts/verify_scheduled_training_case.py',
            'scripts/verify_scheduled_raw.py','scripts/verify_scaling_raw.py','src/model_rg/provenance.py']},
        scope='The complete independent scheduled raw reconstruction applied to one selected scientific trajectory, including every saved state and the entire recorded sampling, target-count, applied-rate and passive-update path. No replay of all native training updates is claimed.'))
    print('Independently reconstructed',a.run_id,flush=True)


if __name__=='__main__':main()
