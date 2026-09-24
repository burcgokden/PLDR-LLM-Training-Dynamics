#!/usr/bin/env python
"""Reconstruct the single-pass ensembles with independent seed-pair arithmetic."""
import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
import numpy as np
from verify_scheduled_collectives import fields
from model_rg.provenance import sha256, write_json


KEYS = ('schedule_horizon', 'recipe', 'heads', 'step', 'shared_seed', 'stream_seed')


def statistics(q, heads):
    s = len(q); x = np.asarray(q, dtype=float).reshape(s, -1); cohort = x.mean(1)
    report = dict(mean=float(x.mean()), independent_seeds=s, contexts=q.shape[1], cohort_means=cohort.tolist(),
        susceptibility=None, central_second=None, central_fourth=None, fourth_ratio=None,
        empirical_percentiles=None, mean_empirical_percentiles=None, jackknife_se=None, leave_one_out=None,
        bootstrap_samples=0, bootstrap_zero_fraction=None, cohort_susceptibility=None,
        cohort_fourth_ratio=None, cohort_binder=None, finite_seed_gaussian_binder_mean=None)
    if s == 1:
        return report, np.array([], dtype=float), cohort
    center = x-x.mean(0); m2 = float(np.mean(center*center)); m4 = float(np.mean(center**4))
    distance = np.zeros((s, s))
    for i in range(s):
        for j in range(i):
            distance[i, j] = distance[j, i] = float(np.mean((x[i]-x[j])**2))
    indices = np.asarray(list(itertools.product(range(s), repeat=s)), dtype=int)
    boot = np.zeros(len(indices))
    for i in range(s):
        for j in range(i):
            boot += heads*distance[indices[:, i], indices[:, j]]/(s*(s-1))
    chi = heads*sum(distance[i, j] for i in range(s) for j in range(i))/(s*(s-1))
    leave = np.asarray([heads*np.var(np.delete(x, i, axis=0), axis=0, ddof=1).mean()
                        for i in range(s)]) if s > 2 else None
    d = cohort-cohort.mean(); a = float(np.mean(d*d)); b = float(np.mean(d**4))
    report.update(susceptibility=float(chi), central_second=m2, central_fourth=m4,
        fourth_ratio=m4/m2**2 if m2 else None, empirical_percentiles=np.quantile(boot, [.025, .975]).tolist(),
        mean_empirical_percentiles=np.quantile(cohort[indices].mean(-1), [.025, .975]).tolist(),
        jackknife_se=float(np.sqrt((s-1)/s*np.sum((leave-leave.mean())**2))) if leave is not None else None,
        leave_one_out=leave.tolist() if leave is not None else None, bootstrap_samples=len(indices),
        bootstrap_zero_fraction=float(np.mean(boot == 0)), cohort_susceptibility=float(heads*np.var(cohort, ddof=1)),
        cohort_fourth_ratio=b/a**2 if a else None, cohort_binder=1-b/(3*a**2) if a else None,
        finite_seed_gaussian_binder_mean=2/(s+1))
    return report, boot, cohort


def allowance(q, heads):
    x = np.asarray(q, dtype=float).reshape(len(q), -1)
    center = x-x.mean(0); count = x.shape[1]+32; eps = np.finfo(float).eps
    return float(16*heads*count*eps/(1-count*eps)*np.max(np.mean(center*center, axis=1)))


def close(actual, expected, atol=1e-24):
    if expected is None:
        if actual is not None:
            raise AssertionError('An undefined or unavailable statistic changed')
    else:
        np.testing.assert_allclose(actual, expected, rtol=4e-10, atol=atol)


def arithmetic(left, right, heads):
    result = {}
    for name, x in left.items():
        y = right[name]; error = x-y
        record = dict(rms_field_difference=float(np.sqrt(np.mean(error*error))),
            max_field_difference=float(np.max(np.abs(error))), float32_susceptibility=None,
            float64_susceptibility=None, absolute_difference=None, centered_error_susceptibility=None,
            susceptibility_error_bound=None, relative_bound=None, meets_one_percent_or_1e_minus8_absolute=None)
        if len(x) > 1:
            chi = lambda a: float(heads*np.var(a, axis=0, ddof=1).mean())
            cx, cy, ce = chi(x), chi(y), chi(error); bound = float(2*np.sqrt(cy*ce)+ce)
            record.update(float32_susceptibility=cx, float64_susceptibility=cy, absolute_difference=abs(cx-cy),
                centered_error_susceptibility=ce, susceptibility_error_bound=bound,
                relative_bound=bound/cy if cy else None,
                meets_one_percent_or_1e_minus8_absolute=bool(bound <= max(.01*cy, 1e-8)))
        result[name] = record
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True)
    p.add_argument('--study', default='scheduled-training-feasible-20260908'); a = p.parse_args()
    study = Path(a.root).resolve()/a.study; folder = study/'analysis/onepass-collectives'
    output = study/'verification/onepass-collective-statistics.json'
    if output.exists(): raise FileExistsError(output)
    checked = {}; cache = {}
    def check(path, expected=None):
        path = Path(path).resolve(); st = path.stat(); key = (str(path), st.st_size, st.st_mtime_ns)
        if key not in cache: cache[key] = sha256(path)
        digest = cache[key]
        if expected is not None and digest != expected:
            raise AssertionError('Changed single-pass collective evidence: '+str(path))
        checked[str(path)] = digest; return digest
    def load(path):
        check(path); return json.loads(Path(path).read_text())
    meta = load(folder/'manifest.json')
    if meta['status'] != 'complete': raise AssertionError('The complete analysis is required')
    for name, key in [('binding.json', 'binding_sha256'), ('results.json', 'results_sha256'), ('measurements.npz', 'raw_sha256')]:
        check(folder/name, meta[key])
    binding = load(folder/'binding.json')
    for name, digest in binding['inputs'].items(): check(name, digest)
    for name, digest in binding['source_files'].items(): check(folder/'source'/name, digest)
    spec = load(study/'protocols/onepass-analysis-selection.json')
    selection = load(study/'protocols/onepass-observation-selection.json')
    groups = defaultdict(dict)
    for case in selection['cases']:
        key = tuple(case[k] for k in KEYS)
        if case['seed'] in groups[key]: raise AssertionError('Duplicated selected identity')
        groups[key][case['seed']] = case
    expected_groups = {tuple(r[k] for k in KEYS): r['seeds'] for r in spec['groups']}
    if {k: sorted(v) for k, v in groups.items()} != expected_groups:
        raise AssertionError('Selected collective coverage changed')
    result = load(folder/'results.json')
    conditions = {tuple(r[k] for k in KEYS)+(r['precision'],): r for r in result['conditions']}
    paired = {tuple(r[k] for k in KEYS): r for r in result['paired_arithmetic']}
    if (len(conditions), len(paired), len(result['conditions']), len(result['paired_arithmetic'])) != (160, 80, 160, 80):
        raise AssertionError('A selected arithmetic condition was omitted or duplicated')
    if result['selected_states'] != 284 or result['groups'] != 80 or meta['states'] != 284 or meta['groups'] != 80:
        raise AssertionError('The selected state count changed')
    comparisons = 0; resamples = 0; expected_arrays = set()
    with np.load(folder/'measurements.npz') as stored:
        for key, cases in sorted(groups.items()):
            horizon, recipe, n, step, shared, stream = key; seeds = sorted(cases); s = len(seeds); observed = {}
            for precision in spec['precisions']:
                group = conditions[key+(precision,)]
                if group['seeds'] != seeds: raise AssertionError('Selected seeds changed')
                data = {name: [] for name in ['fields', 'heads', 'centroids', 'energies', 'mean_metric', 'evaluation_mean_metric', 'context_kl']}
                for seed in seeds:
                    case = cases[seed]; parent = study/'measurements'/case['name']
                    proof = load(study/'verification/observations'/(case['name']+'.json'))
                    if proof['status'] != 'passed' or proof['case'] != case:
                        raise AssertionError('An underlying observation is not independently reconstructed')
                    check(parent/'measurements.npz', proof['checked_sha256'][str(parent/'measurements.npz')])
                    with np.load(parent/'measurements.npz') as z:
                        for field in data: data[field].append(z[precision+'_'+field].astype(float))
                data = {k: np.stack(v) for k, v in data.items()}; values = fields(data); observed[precision] = values
                if set(group['observables']) != set(values): raise AssertionError('A declared field was omitted')
                for name, q in values.items():
                    expected, boot, cohort = statistics(q, n); report = group['observables'][name]
                    if set(expected) != set(report): raise AssertionError('A moment or diagnostic was omitted')
                    for field, value in expected.items():
                        close(report[field], value, atol=max(1e-24, allowance(q, n)) if field in ['empirical_percentiles', 'susceptibility'] else 1e-24)
                    tag = f'H{horizon}_{recipe}_N{n}_t{step}_c{shared}_b{stream}_{precision}_{name}'
                    close(stored[tag+'_bootstrap'], boot, atol=max(1e-24, allowance(q, n)))
                    close(stored[tag+'_cohort'], cohort)
                    expected_arrays.update([tag+'_bootstrap', tag+'_cohort']); comparisons += 1; resamples += len(boot)
                one = float(np.var(data['centroids'], axis=0, ddof=1).mean()) if s > 1 else None
                chi = float(n*np.var(values['common_centroid'], axis=0, ddof=1).mean()) if s > 1 else None
                close(group['centroid_one_head_variance'], one)
                close(group['centroid_cross_head_covariance'], (chi-one)/(n-1) if s > 1 else None)
                close(group['centroid_effective_head_count'], n*one/chi if chi is not None and chi > 0 else None)
                close(group['mean_context_kl'], float(data['context_kl'].mean()))
                for name in ['mean_metric', 'evaluation_mean_metric']:
                    if s > 1:
                        whole = float(n*np.var(values[name+'_total'], axis=0, ddof=1).mean())
                        common = float(n*np.var(values[name+'_common'], axis=0, ddof=1).mean())
                        contrast = float(n*np.var(values[name+'_contrast'], axis=0, ddof=1).mean())
                        close(whole, common+contrast)
                        expected_fraction = common/whole if whole > 0 else None
                    else: expected_fraction = None
                    close(group[name+'_common_fraction'], expected_fraction)
                del data, values
            if paired[key]['seeds'] != seeds: raise AssertionError('Arithmetic pairs changed seed coverage')
            expectation = arithmetic(observed['float32'], observed['float64'], n)
            if set(expectation) != set(paired[key]['observables']): raise AssertionError('An arithmetic field was omitted')
            for name, row in expectation.items():
                reported = paired[key]['observables'][name]
                if set(row) != set(reported): raise AssertionError('An arithmetic bound was omitted')
                for field, value in row.items(): close(reported[field], value)
            print('Reconstructed all ensemble and arithmetic fields', key, flush=True)
        if set(stored.files) != expected_arrays: raise AssertionError('Retained ensemble arrays changed coverage')
    write_json(output, dict(status='passed', states=284, groups=80, precision_groups=160,
        collective_fields=comparisons, empirical_resamples=resamples, checked_sha256=checked,
        verifier_sources={str(Path(__file__).resolve()): sha256(__file__),
            str(Path(__file__).with_name('verify_scheduled_collectives.py').resolve()): sha256(Path(__file__).with_name('verify_scheduled_collectives.py'))},
        scope='Independent direct seed-pair reconstruction of every selected finite-ensemble statistic. Single-seed covariance is unavailable. Empirical resampling ranges do not certify population coverage or critical exponents.'))


if __name__ == '__main__': main()
