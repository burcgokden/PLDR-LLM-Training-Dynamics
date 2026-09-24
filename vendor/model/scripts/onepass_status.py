#!/usr/bin/env python
"""Report the actual nonrepeated-data queue and completed evidence, read only."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-feasible-20260908')
    p.add_argument('--json',action='store_true');a=p.parse_args()
    root=Path(a.root);study=root/a.study
    selection=json.loads((study/'protocols/onepass-training-selection.json').read_text())
    jobs=selection['jobs'];finished=[];active=[];unstarted=[];other=[]
    for j in jobs:
        folder=study/'runs'/j['run_id'];path=folder/'manifest.json'
        if path.exists():
            meta=json.loads(path.read_text())
            item=dict(run_id=j['run_id'],role=j['role'],status=meta['status'],updates=meta['completed_step'],
                      selected_updates=j['steps'],seconds=meta['seconds'])
            (finished if meta['status']=='complete' else other).append(item)
        elif folder.exists():
            item=dict(run_id=j['run_id'],role=j['role'],selected_updates=j['steps'],last_observed_update=0)
            log=study/'logs'/(j['run_id']+'.log')
            if log.exists():
                for line in log.read_text(errors='replace').splitlines():
                    match=re.match(re.escape(j['run_id'])+r' (\d+) nll (\S+).* seconds (\S+)',line)
                    if match:item.update(last_observed_update=int(match[1]),heldout_last_target_nll=float(match[2]),
                                         loop_seconds=float(match[3]))
            active.append(item)
        else:unstarted.append(j['run_id'])
    panels={}
    for name in ['launcher-onepass.json','launcher-onepass-observations.json',
                 'launcher-onepass-native-steps.json','launcher-prefix-risk-comparison.json',
                 'launcher-onepass-analyses.json','launcher-repetition-analysis.json',
                 'launcher-onepass-schedule-comparison.json']:
        path=study/name
        if path.exists():
            d=json.loads(path.read_text());panels[name]=dict(status=d['status'],records=len(d.get('records',[])))
    proofs={}
    for name in ['onepass-data.json','gpu-qualification.json','observer-qualification.json','source-reuse.json']:
        path=study/'verification'/name
        proofs[name]=json.loads(path.read_text())['status'] if path.exists() else 'pending'
    observation_progress=None
    observation_selection=study/'protocols/onepass-observation-selection.json'
    observation_controller=study/'launcher-onepass-observations.json'
    if observation_selection.exists() and observation_controller.exists():
        cases=json.loads(observation_selection.read_text())['cases']
        completed_runs={v['run_id'] for v in finished}
        eligible={v['name'] for v in cases if v['run_id'] in completed_runs}
        records=json.loads(observation_controller.read_text()).get('records',[])
        complete={v['name'] for v in records if v.get('role')=='scientific' and v.get('status')=='complete'}
        observation_progress=dict(selected=len(cases),complete=len(complete),
            from_complete_training_paths=len(eligible),awaiting_completion=len(eligible-complete),
            scope='Completed observation bundles include their selected risk, prefix and endpoint task checks. Eligibility here requires a completed parent trajectory, as in the observer controller.')
    old=root/'scheduled-training-20260908'
    archived=old/'development/feasibility-redesign/no-repeat-data-policy.json'
    data=dict(at=datetime.now(timezone.utc).isoformat(),study=str(study),
        selected_trajectories=len(jobs),selected_updates=sum(j['steps'] for j in jobs),
        completed_path_updates=sum(v['updates'] for v in finished),
        recorded_update_lower_bound=sum(v['updates'] for v in finished)+sum(v['last_observed_update'] for v in active),
        selected_roles=dict(Counter(j['role'] for j in jobs)),completed=finished,active_or_started=active,
        other_outcomes=other,unstarted=unstarted,controllers=panels,proofs=proofs,
        observation_progress=observation_progress,
        repeated_corpus_queue='Stopped; all unstarted repeated-corpus jobs canceled. Retained checkpoints are diagnostic evidence, not completed long runs.' if archived.exists() else 'No recorded cancellation',
        interpretation='A started directory and its latest milestone do not establish a live process or a completed trajectory. Independent proof states are reported separately. No endpoint is inferred from elapsed time.')
    for name in ['launcher-reference-benchmarks.json','launcher-reference-benchmark-analysis.json']:
        path=old/name
        if path.exists():data.setdefault('reference_evaluation',{})[name]=json.loads(path.read_text())['status']
    if a.json:print(json.dumps(data,indent=2));return
    print(data['at'])
    print(f"Single-pass training: {len(finished)}/{len(jobs)} complete; {len(active)} started; {len(unstarted)} unstarted.")
    count=data['recorded_update_lower_bound'];total=data['selected_updates']
    print(f"Recorded update lower bound: {count:,}/{total:,} ({100*count/total:.1f} percent); completed paths: {data['completed_path_updates']:,} updates.")
    if observation_progress is not None:
        o=observation_progress
        print(f"CPU observation bundles: {o['complete']}/{o['selected']} complete; {o['awaiting_completion']} from completed training paths awaiting completion.")
    for item in active:
        print(f"  {item['run_id']}: latest saved/logged observation {item['last_observed_update']}/{item['selected_updates']} updates")
    for item in other:
        print(f"  Outcome requiring review: {item['run_id']} {item['status']} at {item['updates']} updates")
    for name,value in panels.items():print(name,value['status'],f"({value['records']} records)")
    print('Checks:',', '.join(f'{name}: {status}' for name,status in proofs.items()))
    print(data['repeated_corpus_queue'])
    for name,status in data.get('reference_evaluation',{}).items():print(name,status)


if __name__=='__main__':main()
