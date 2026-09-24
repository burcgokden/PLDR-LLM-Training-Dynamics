#!/usr/bin/env python
"""Read-only status of the selected size-time study, including unbound successors."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re


def main():
    p = argparse.ArgumentParser();p.add_argument('--root', required=True)
    p.add_argument('--study', default='critical-scaling-20260906');p.add_argument('--json', action='store_true')
    a = p.parse_args();study = Path(a.root)/a.study
    jobs = {}
    protocols = ['main-size-time-execution', 'clock-holdout', 'diffusive-size-map',
                 'environment-factorial', 'zero-horizon', 'extended-horizon']
    for name in protocols:
        path = study/'protocols'/f'{name}.json'
        if not path.exists():continue
        spec = json.loads(path.read_text())
        for job in spec['jobs']:
            if job['run_id'] in jobs:raise AssertionError('Duplicate selected training artifact')
            jobs[job['run_id']] = dict(job, protocol=name, protocol_bound=True)
    original = json.loads((study/'protocols/long-horizon.json').read_text())
    for job in original['jobs']:
        if job['run_id'] == 'long-h4-g1-s640101':jobs[job['run_id']] = dict(job, protocol='long-horizon', protocol_bound=True)
    marker = study/'followup-preparation.json'
    if marker.exists() and not (study/'protocols/extended-horizon.json').exists():
        for n in [4, 14]:
            for seed in range(640101, 640105):
                name = f'extended-h{n}-g1-s{seed}'
                jobs[name] = dict(run_id=name, heads=n, seed=seed, multiplier=1, steps=131072,
                    selected_start=32768, protocol='extended-horizon', protocol_bound=False)
    records, completed_updates, progress_updates = [], 0, 0
    for name, job in jobs.items():
        folder = study/'runs'/name;meta_path = folder/'manifest.json'
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
        start = job.get('selected_start', 0)
        if job.get('resume'):
            parent = Path(job['resume'])
            match = re.fullmatch(r'training-state-(\d+)\.pt', parent.name)
            start = int(match[1]) if match else json.loads((parent.parent/'manifest.json').read_text())['completed_step']
        last = start
        if meta:
            state = meta['status'];last = meta['completed_step']
            if state == 'complete':completed_updates += last-start
        else:
            state = 'started' if folder.exists() else ('queued' if job['protocol_bound'] else 'selected_waiting_parent_binding')
            log = study/'logs'/f'{name}.log'
            if log.exists():
                for line in log.read_text().splitlines():
                    match = re.match(re.escape(name)+r' (\d+) nll ', line)
                    if match:last = max(last, int(match[1]))
        progress_updates += last-start
        records.append(dict(run_id=name, heads=job['heads'], start=start, horizon=job['steps'],
            observed_through=last, status=state, protocol_bound=job['protocol_bound'],
            selected_additional_updates=job['steps']-start))
    measurements = Counter()
    for path in (study/'measurements').glob('*/manifest.json'):
        meta = json.loads(path.read_text())
        if meta.get('status') != 'complete':continue
        name = path.parent.name
        role = ('cpu_states' if name.startswith('cpu-') else 'conditional_noise_states' if name.startswith('noise-jvp-')
                else 'spectator_controls' if name.startswith('spectator-')
                else 'expanded_source_validation_states' if name.startswith('source-basis-')
                else 'native_conditional_step_states' if name.startswith('native-step-')
                else 'frozen_training_risk_states' if name.startswith('training-risk-')
                else 'fresh_source_validation_states' if name.startswith('visible-noise-') else 'other_or_qualification')
        measurements[role] += 1
    analyses = {path.parent.name:json.loads(path.read_text()).get('status')
                for path in (study/'analysis').glob('*/manifest.json')}
    result = dict(training=records, training_status_counts=dict(Counter(r['status'] for r in records)),
        selected_training_artifacts=len(records), selected_additional_updates=sum(r['selected_additional_updates'] for r in records),
        completed_artifact_updates=completed_updates, observed_update_lower_bound=progress_updates,
        selected_maximum_horizon=max(r['horizon'] for r in records), completed_measurements=dict(measurements), analyses=analyses,
        scope='Read-only status. Milestone progress is a lower bound. Selected successors without completed parents are distinguished from immutable bound execution protocols. Authorization for a larger horizon is not an executed experiment.')
    if a.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print('Training:', result['training_status_counts'])
        print('Selected:', len(records), 'artifacts;', result['selected_additional_updates'], 'additional updates;',
              'maximum', result['selected_maximum_horizon'])
        print('Completed artifact updates:', completed_updates, '; observed progress lower bound:', progress_updates)
        for r in records:
            if r['status'] == 'started':print('Active:', r['run_id'], r['observed_through'], '/', r['horizon'])
        print('Completed measurements:', dict(measurements))
        print('Completed analyses:', ', '.join(sorted(k for k,v in analyses.items() if v == 'complete')))


if __name__ == '__main__':main()
