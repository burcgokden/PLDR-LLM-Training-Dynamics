#!/usr/bin/env python
"""Pair the selected single-pass constant and warmup/cosine control states."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json
from onepass_analysis import empirical_mean, group_key, label


RISK_FIELDS = ['heldout_last_nll', 'training_last_nll', 'heldout_all_token_weighted_nll',
               'training_all_token_weighted_nll', 'proper_nll_prefix_panel']
COLLECTIVE_FIELDS = ['row_native', 'absolute_row_energy', 'common_centroid']


def paired_mean(scheduled, constant):
    x, y = np.asarray(scheduled, dtype=float), np.asarray(constant, dtype=float)
    if x.shape != (4,) or y.shape != (4,) or not np.isfinite([x, y]).all():
        raise AssertionError('All four paired initialization identities are required')
    delta = x - y
    return dict(scheduled_values=x.tolist(), constant_values=y.tolist(),
                scheduled_mean=float(x.mean()), constant_mean=float(y.mean()),
                difference=empirical_mean(delta.tolist()),
                negative_differences=int(np.sum(delta < 0)), positive_differences=int(np.sum(delta > 0)),
                exact_ties=int(np.sum(delta == 0)))


def paired_susceptibility(scheduled, constant, scheduled_boot, constant_boot):
    x, y = np.asarray(scheduled_boot, dtype=float), np.asarray(constant_boot, dtype=float)
    if x.shape != (256,) or y.shape != (256,):
        raise AssertionError('The complete ordered four-identity resampling law is required')
    delta = x - y
    leave = np.asarray(scheduled['leave_one_out']) - np.asarray(constant['leave_one_out'])
    if leave.shape != (4,):
        raise AssertionError('The paired leave-one-identity law is incomplete')
    return dict(scheduled=scheduled['susceptibility'], constant=constant['susceptibility'],
        difference=scheduled['susceptibility']-constant['susceptibility'],
        empirical_percentiles=np.quantile(delta, [.025, .975]).tolist(),
        negative_resample_fraction=float(np.mean(delta < 0)), positive_resample_fraction=float(np.mean(delta > 0)),
        exact_zero_resample_fraction=float(np.mean(delta == 0)),
        leave_one_out=leave.tolist(), jackknife_se=float(np.sqrt(.75*np.sum((leave-leave.mean())**2)))), delta


def prepare(study, repo):
    target = study/'protocols/onepass-schedule-comparison-selection.json'
    if target.exists():
        raise FileExistsError(target)
    training = study/'protocols/onepass-training-selection.json'
    jobs = json.loads(training.read_text())['jobs']
    by_id = {j['run_id']: j for j in jobs}
    groups = []
    for heads in (4, 14):
        for step in (8192, 32768):
            cases = []
            for seed in range(640101, 640105):
                scheduled = f'compact-controlled-h{heads}-s{seed}'
                prefix = 'data_constant' if seed < 640103 else 'data_constant_short'
                constant = f'{prefix}-controlled-h{heads}-s{seed}'
                left, right = by_id[scheduled], by_id[constant]
                if (study/'runs'/scheduled).exists():
                    raise AssertionError('This comparison must be selected before its scheduled control runs start')
                excluded = {'total_steps', 'warmup_steps', 'floor_fraction'}
                if {k:v for k,v in left['profile'].items() if k not in excluded} != {
                        k:v for k,v in right['profile'].items() if k not in excluded}:
                    raise AssertionError('The comparison changes more than the specified drive')
                for key in ['heads', 'seed', 'stream_seed', 'shared_seed', 'microbatch']:
                    if left[key] != right[key]:
                        raise AssertionError('A schedule pair has unmatched initialization or data conditions')
                if any(step not in map(int,j['save_steps'].split(',')) for j in [left,right]):
                    raise AssertionError('An unselected checkpoint entered the paired comparison')
                cases.append(dict(seed=seed, scheduled=scheduled, constant=constant))
            groups.append(dict(heads=heads, step=step, seeds=list(range(640101,640105)),
                scheduled_key=[32768,'controlled',heads,step,640011,640001],
                constant_key=[0,'controlled',heads,step,640011,640001], pairs=cases))
    qualification = study/'qualification/onepass-schedule-comparison/verification.json'
    proof = json.loads(qualification.read_text())
    if (proof['status'], proof['fixtures'], proof['pairing_distinctions']) != ('passed',5,2):
        raise AssertionError('The paired schedule statistics need their exact-law qualification')
    for name,digest in proof['sources'].items():
        if sha256(repo/name) != digest:
            raise AssertionError('The paired-statistic qualification predates its implementation')
    files = ['scripts/analyze_onepass_schedule_comparison.py', 'scripts/verify_onepass_schedule_comparison.py',
             'scripts/onepass_analysis.py', 'src/model_rg/controlled.py', 'src/model_rg/provenance.py']
    inputs = [training, qualification, study/'protocols/onepass-analysis-selection.json',
              study/'protocols/onepass-observation-selection.json']
    known = []
    for name in sorted({p['constant'] for g in groups for p in g['pairs']}):
        path = study/'runs'/name/'manifest.json'
        known.append(dict(run_id=name, state='complete' if path.exists() else 'started_or_unstarted',
                          manifest_sha256=sha256(path) if path.exists() else None))
    write_json(target, dict(schema='onepass-paired-schedule-selection-v1',
        selected_at=datetime.now(timezone.utc).isoformat(), groups=groups, conditions=4, native_pairs=16, trajectory_pairs=8,
        precisions=['float32','float64'], risk_fields=RISK_FIELDS, collective_fields=COLLECTIVE_FIELDS,
        additional_training_updates=0, known_constant_outcomes=known,
        sources={name:sha256(repo/name) for name in files}, inputs_sha256={str(p):sha256(p) for p in inputs},
        scope='Scheduled minus constant at exactly matched single-pass source histories, head count, initialization and checkpoint update. Only the learning-rate history changes. Both comparisons stop at already selected checkpoints; no long-horizon schedule equivalence is inferred.',
        selection_scope='Constant outcomes were partly available. No compact controlled trajectory had started when this paired analysis was selected. It adds no trajectory or endpoint.',
        uncertainty='All 256 ordered whole-initialization resamples use the same four indices on both sides. Empirical percentile and sign fractions have no guaranteed population coverage, posterior or multiple-testing interpretation. Paired horizons reuse the same initialization identities.',
        prefix_scope='The proper-prefix field averages eight fixed held-out contexts at lengths 16, 32, 48. It differs from both the 512-context risk cohort and the 32-context/all-64-position causal panel.'))
    print('Selected four matched schedule conditions and sixteen native state pairs', sha256(target), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--study', default='scheduled-training-feasible-20260908')
    p.add_argument('--prepare', action='store_true')
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    study = Path(args.root).resolve()/args.study
    if args.prepare:
        prepare(study, repo)
        return
    protocol = study/'protocols/onepass-schedule-comparison-selection.json'
    spec = json.loads(protocol.read_text())
    inputs = [protocol]
    for name,digest in spec['sources'].items():
        if sha256(repo/name) != digest:
            raise AssertionError('A selected paired-schedule implementation changed')
    for name,digest in spec['inputs_sha256'].items():
        if sha256(name) != digest:
            raise AssertionError('A paired-schedule selection input changed')
        inputs.append(Path(name))
    reports = {}
    for name,filename in [('collectives','onepass-collective-statistics.json'),
                           ('predictions','onepass-prediction-statistics.json')]:
        folder = study/'analysis'/('onepass-'+name)
        proof_path = study/'verification'/filename
        proof = json.loads(proof_path.read_text())
        if proof['status'] != 'passed':
            raise AssertionError('Both complete upstream analyses must pass first')
        inputs.append(proof_path)
        for filename in ['results.json','manifest.json'] + (['measurements.npz'] if name=='collectives' else []):
            path = folder/filename
            if sha256(path) != proof['checked_sha256'][str(path)]:
                raise AssertionError('An independently checked upstream result changed')
            inputs.append(path)
        reports[name] = json.loads((folder/'results.json').read_text())
    collective = {(group_key(r),r['precision']):r for r in reports['collectives']['conditions']}
    predictions = {group_key(r):r for r in reports['predictions']['conditions']}
    if len(collective)!=160 or len(predictions)!=80:
        raise AssertionError('The complete upstream condition inventory is required')
    raw = {}
    risk_records, collective_records = [], []
    with np.load(study/'analysis/onepass-collectives/measurements.npz') as values:
        for group in spec['groups']:
            left_key,right_key = tuple(group['scheduled_key']),tuple(group['constant_key'])
            left,right = predictions[left_key],predictions[right_key]
            if left['seeds'] != group['seeds'] or right['seeds'] != group['seeds']:
                raise AssertionError('The paired initialization order changed')
            risk = {}
            for field in spec['risk_fields']:
                a,b = left['observations'][field],right['observations'][field]
                if a['seed_ids']!=group['seeds'] or b['seed_ids']!=group['seeds']:
                    raise AssertionError('A paired risk identity is absent')
                risk[field] = paired_mean(a['seed_values'],b['seed_values'])
            risk_records.append(dict(group=group, observables=risk))
            for precision in spec['precisions']:
                a,b = collective[left_key,precision],collective[right_key,precision]
                if a['seeds']!=group['seeds'] or b['seeds']!=group['seeds']:
                    raise AssertionError('A paired collective identity is absent')
                observations = {}
                for field in spec['collective_fields']:
                    x,y = a['observables'][field],b['observables'][field]
                    left_name=label(left_key)+'_'+precision+'_'+field
                    right_name=label(right_key)+'_'+precision+'_'+field
                    paired,delta=paired_susceptibility(x,y,values[left_name+'_bootstrap'],values[right_name+'_bootstrap'])
                    observations[field] = dict(mean=paired_mean(x['cohort_means'],y['cohort_means']), susceptibility=paired)
                    raw[f'N{group["heads"]}_t{group["step"]}_{precision}_{field}_paired_bootstrap']=delta
                collective_records.append(dict(group=group, precision=precision,observables=observations,
                    context_dispersion=dict(scheduled=a['mean_context_kl'],constant=b['mean_context_kl'],
                        difference=a['mean_context_kl']-b['mean_context_kl'])))
    out = study/'analysis/onepass-schedule-comparison'
    out.mkdir(parents=True, exist_ok=False)
    bind_run(out, list(dict.fromkeys(inputs)), vars(args))
    np.savez_compressed(out/'measurements.npz', **raw)
    write_json(out/'results.json',dict(status='complete', schema='onepass-paired-schedule-results-v1',
        conditions=4,native_pairs=16,trajectory_pairs=8,precision_conditions=8,risk=risk_records,collectives=collective_records,
        additional_training_updates=0,scope=spec['scope'],uncertainty=spec['uncertainty'],prefix_scope=spec['prefix_scope']))
    write_json(out/'manifest.json',dict(status='complete',binding_sha256=sha256(out/'binding.json'),
        results_sha256=sha256(out/'results.json'),raw_sha256=sha256(out/'measurements.npz')))
    print('Completed four paired schedule conditions from sixteen existing native state pairs',flush=True)


if __name__ == '__main__':
    main()
