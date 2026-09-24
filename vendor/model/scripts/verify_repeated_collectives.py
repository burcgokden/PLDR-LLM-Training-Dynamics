#!/usr/bin/env python
"""Check a retained repeated-data state's collective observation identities."""
import argparse
import json
from pathlib import Path

from model_rg.provenance import sha256, write_json
from verify_scaling_raw import BoundFiles, verify_measurement


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True)
    p.add_argument('--study',required=True); p.add_argument('--case',required=True); a=p.parse_args()
    root=Path(a.root).resolve(); study=root/a.study
    selection=study/'protocols/repeated-observation-selection.json'
    spec=json.loads(selection.read_text()); cases=[c for c in spec['cases'] if c['name']==a.case]
    if len(cases)!=1: raise AssertionError('One selected retained state is required')
    case=cases[0]; files=BoundFiles(); files.digest(selection)
    for name,digest in spec['inputs_sha256'].items(): files.check(name,digest)
    repo=Path(__file__).resolve().parents[1]
    for name,digest in spec['producer_sources'].items(): files.check(repo/name,digest)
    path=study/'measurements'/case['name']/'manifest.json'; meta=json.loads(path.read_text())
    if meta['status']!='complete' or meta['case']!=case: raise AssertionError('Selected collective observation changed')
    parent=json.loads(Path(case['parent_manifest']).read_text())
    if (parent['source_trajectory_status_at_capture']!='administratively_censored' or
        parent['completed_step']!=case['step'] or parent['source_checkpoint_sha256']!=case['state_sha256']):
        raise AssertionError('The sealed checkpoint scope changed')
    files.check(case['state'],case['state_sha256'])
    record=verify_measurement(root,path,files)
    out=study/'verification/collectives'/(a.case+'.json')
    if out.exists(): raise FileExistsError(out)
    write_json(out,dict(status='passed',case=case,record=record,checked_sha256={key[0]:value for key,value in files.cache.items()},
        verifier_sha256=sha256(__file__),additional_independent_training_identities=0,
        scope='Independent fixed-state coordinate, cohort and energy reconstruction. Original training remains censored.'))
    print('Retained collective reconstruction passed',a.case,flush=True)


if __name__=='__main__': main()
