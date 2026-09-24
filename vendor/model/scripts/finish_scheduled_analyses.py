#!/usr/bin/env python
"""Run complete scheduled analyses as their independently checked inputs arrive."""
from companion_paths import child_pythonpath
import argparse
from concurrent.futures import ThreadPoolExecutor
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
    p.add_argument('--study', default='scheduled-training-20260908')
    args = p.parse_args()
    study = Path(args.root).resolve()/args.study
    repo = Path(__file__).resolve().parents[1]
    marker = study/'launcher-scheduled-analyses.json'
    if marker.exists():
        raise FileExistsError(marker)
    load = lambda path: json.loads(Path(path).read_text())
    selection = study/'protocols/regime-training-selection.json'
    observation = study/'protocols/regime-observation-selection.json'
    prefix = study/'protocols/scheduled-prefix-selection.json'
    reasoning = study/'protocols/reasoning-mechanism-selection.json'
    early = study/'protocols/early-prefix-selection.json'
    jobs = load(selection)['jobs']
    states = load(observation)['cases']
    prefixes = load(prefix)['cases']
    tasks = load(reasoning)['cases']
    early_cases = load(early)['cases']
    protocols = {str(path): sha256(path) for path in [selection, observation, prefix, reasoning, early,
        study/'protocols/scheduled-analysis-selection.json', study/'protocols/scheduled-path-analysis-selection.json']}
    sources = ['scripts/finish_scheduled_analyses.py', 'scripts/analyze_scheduled_study.py',
        'scripts/analyze_scheduled_collectives.py', 'scripts/analyze_scheduled_predictions.py',
        'scripts/analyze_scheduled_paths.py', 'scripts/score_scheduled_clocks.py',
        'scripts/verify_scheduled_collectives.py', 'scripts/verify_scheduled_predictions.py',
        'scripts/verify_scheduled_paths.py', 'scripts/verify_scheduled_clock_scores.py',
        'scripts/verify_scheduled_frozen_clocks.py', 'scripts/reconcile_early_scheduled_prefix.py',
        'scripts/analyze_scaling_collectives.py', 'scripts/analyze_scaling_updates.py', 'scripts/analyze_size_time.py',
        'scripts/freeze_scaling_forecasts.py', 'scripts/freeze_scheduled_clocks.py',
        'src/model_rg/scaling.py', 'src/model_rg/controlled.py', 'src/model_rg/provenance.py']
    source_hashes = {name: sha256(repo/name) for name in sources}
    ledger = dict(status='waiting_for_verified_inputs', started_at=datetime.now(timezone.utc).isoformat(),
                  arguments=vars(args), protocols_sha256=protocols, sources_sha256=source_hashes, records=[])
    write_json(marker, ledger)
    env = dict(os.environ, PYTHONPATH=child_pythonpath("model"), PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='4')
    common = ['--root', str(Path(args.root).resolve()), '--study', args.study]

    def passing(path, status='passed'):
        return path.exists() and load(path).get('status') == status

    def case_checks(cases, kind):
        return all(passing(study/'verification'/kind/(case['name']+'.json')) for case in cases)

    def native_checks(selected):
        return all(passing(study/'verification/training'/(job['run_id']+'.json'), 'complete') for job in selected)

    def early_ready():
        if not passing(study/'launcher-early-prefix.json', 'observations_complete_awaiting_final_reconciliation'):
            return False
        for case in early_cases:
            if not passing(Path(case['final_manifest']), 'complete'):
                return False
            counterparts = [c for c in prefixes if all(c[k] == case[k] for k in ['heads', 'seed', 'recipe', 'step'])]
            if len(counterparts) != 1 or not case_checks(counterparts, 'prefix'):
                return False
        return True

    controlled = [j for j in jobs if j['recipe'] == 'controlled']
    controlled_states = [c for c in states if c['recipe'] == 'controlled']
    definitions = {
        'clock-scores': dict(ready=lambda: native_checks(controlled) and case_checks(controlled_states, 'observations') and
                            passing(study/'verification/frozen-clock-statistics.json'),
            commands=[('score_scheduled_clocks.py', []), ('verify_scheduled_clock_scores.py', [])],
            output=study/'verification/clock-score-statistics.json'),
        'collectives': dict(ready=lambda: case_checks(states, 'observations'),
            commands=[('analyze_scheduled_collectives.py', ['--output', str(study/'analysis/collectives')]),
                      ('verify_scheduled_collectives.py', [])], output=study/'verification/collective-statistics.json'),
        'training-paths': dict(ready=lambda: native_checks(jobs),
            commands=[('analyze_scheduled_paths.py', []), ('verify_scheduled_paths.py', [])],
            output=study/'verification/temporal-statistics.json'),
        'predictions': dict(ready=lambda: case_checks(states, 'observations') and case_checks(prefixes, 'prefix') and case_checks(tasks, 'reasoning'),
            commands=[('analyze_scheduled_predictions.py', []), ('verify_scheduled_predictions.py', [])],
            output=study/'verification/prediction-statistics.json'),
        'early-reconciliation': dict(ready=early_ready, commands=[('reconcile_early_scheduled_prefix.py', [])],
            output=study/'verification/early-prefix-reconciliation.json'),
    }

    def execute(name, definition):
        qualification_record=load(qualification)
        if qualification_record['status']!='passed':
            raise AssertionError('Scheduled analysis arithmetic is not qualified')
        for path,digest in qualification_record['qualified_sources'].items():
            if sha256(repo/path)!=digest:
                raise AssertionError('The scheduled analysis qualification predates a source change: '+path)
        for path, digest in protocols.items():
            if sha256(path) != digest:
                raise AssertionError('An active scheduled analysis protocol changed')
        for path, digest in source_hashes.items():
            if sha256(repo/path) != digest:
                raise AssertionError('An active scheduled analysis implementation changed: ' + path)
        commands = []
        before = time.time()
        for index, (script, extra) in enumerate(definition['commands']):
            command = [sys.executable, str(repo/'scripts'/script), *common, *extra]
            log = study/'logs'/('complete-scheduled-'+name+'-'+str(index)+'.log')
            with log.open('x') as stream:
                completed = subprocess.run(command, cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT)
            commands.append(dict(command=command, returncode=completed.returncode, log_sha256=sha256(log)))
            if completed.returncode:
                return dict(name=name, status='analysis_failure', commands=commands, seconds=time.time()-before)
        if not passing(definition['output'], 'complete' if name == 'complete' else 'passed'):
            raise AssertionError('A scheduled analysis did not produce its required complete check')
        return dict(name=name, status='complete', commands=commands, seconds=time.time()-before,
                    output=str(definition['output']), output_sha256=sha256(definition['output']))

    pending = dict(definitions)
    active = {}
    failed = False
    qualification = study/'verification/analysis-qualification.json'
    with ThreadPoolExecutor(max_workers=2) as pool:
        while pending or active:
            if passing(qualification):
                for name, definition in list(pending.items()):
                    if len(active) >= 2 or failed:
                        break
                    if definition['ready']():
                        active[pool.submit(execute, name, definition)] = name
                        del pending[name]
            for future, name in list(active.items()):
                if not future.done():
                    continue
                try:
                    record = future.result()
                except Exception as error:
                    record = dict(name=name, status='controller_failure', error=repr(error))
                ledger['records'].append(record)
                failed |= record['status'] != 'complete'
                del active[future]
                print(json.dumps(record), flush=True)
            ledger.update(status='analysis_failure' if failed else ('analyzing' if active else 'waiting_for_verified_inputs'),
                          active=list(active.values()), pending=list(pending))
            write_json(marker, ledger)
            if failed and not active:
                raise RuntimeError('Resolve the preserved scheduled analysis failure before continuing')
            if pending or active:
                time.sleep(30)
    while not passing(study/'launcher-scheduled-observations-recovered.json', 'complete') or not passing(study/'verification/training-raw.json', 'complete'):
        time.sleep(30)
    definition = dict(commands=[('analyze_scheduled_study.py', [])], output=study/'analysis/complete/manifest.json')
    ledger['status'] = 'aggregating_complete_evidence'
    write_json(marker, ledger)
    record = execute('complete', definition)
    ledger['records'].append(record)
    ledger['status'] = record['status']
    write_json(marker, ledger)
    if record['status'] != 'complete':
        raise RuntimeError('The complete scheduled aggregation did not pass')
    print('All selected scheduled analyses and their complete aggregate have finished', flush=True)


if __name__ == '__main__':
    main()
