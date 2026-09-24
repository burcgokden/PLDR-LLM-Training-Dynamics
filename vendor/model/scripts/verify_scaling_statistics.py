#!/usr/bin/env python
"""Independent reconstruction of fixed-unit collective and forecast statistics.

This checker imports no statistical producer. It starts from saved observation
arrays and reconstructs central moments, whole-identity resampling, arithmetic
transfer and paired contrasts. A named snapshot cannot serve as a release gate.
"""
import argparse
from collections import defaultdict
from functools import lru_cache
import itertools
import json
from pathlib import Path

import numpy as np

from model_rg.provenance import sha256, write_json


def read(path):
    return json.loads(Path(path).read_text())


def close(x, y, rtol=3e-9, atol=1e-22):
    np.testing.assert_allclose(x, y, rtol=rtol, atol=atol)


BOOTSTRAP_COMPARISONS = []


def bootstrap_roundoff_allowance(q, n):
    """Comparison scale for algebraically equivalent float64 variance sums.

    The centered dot products have D coordinates. The additional 8s term
    accounts for assembly across s seeds, and 4s covers covariance and
    resample subtraction. This finite-operation allowance is recorded beside
    actual discrepancies; it is not a bound on model-forward real arithmetic.
    """
    x = np.asarray(q, dtype=np.float64).reshape(len(q), -1)
    center = x-x.mean(0)
    operations = x.shape[1]+8*len(q)
    epsilon = np.finfo(np.float64).eps
    gamma = operations*epsilon/(1-operations*epsilon)
    return 4*len(q)*n*gamma*float(np.max(np.mean(center*center, axis=1)))


def compare_bootstrap(left, right, allowance, label):
    close(left, right, atol=max(1e-22, allowance))
    difference = float(np.max(np.abs(np.asarray(left)-right)))
    BOOTSTRAP_COMPARISONS.append(dict(label=label, values=int(np.size(left)),
        maximum_absolute_difference=difference, absolute_roundoff_allowance=allowance))


@lru_cache(maxsize=8)
def multiplicities(s):
    indices = (np.array(list(itertools.product(range(s), repeat=s))) if s == 4
               else np.random.default_rng(650061).integers(s, size=(20000, s)))
    return np.stack([np.bincount(i, minlength=s) for i in indices]).astype(float)


def moments(q, n):
    q = np.asarray(q, dtype=np.float64)
    s = len(q)
    x = q.reshape(s, -1)
    centered = x - x.mean(0)
    gram = centered @ centered.T / x.shape[1]
    w = multiplicities(s)
    # Weighted second central moment, rather than the producer's pair-distance
    # expression. Pure single-identity resamples are exactly degenerate.
    boot = n * (w @ np.diag(gram) - np.einsum('bi,ij,bj->b', w, gram, w)/s)/(s-1)
    boot[np.max(w, axis=1) == s] = 0.
    if boot.min() < -1e-12 * max(float(np.max(boot)), 1e-300):
        raise AssertionError('Negative reconstructed variance')
    boot = np.maximum(boot, 0)
    m2, m4 = float(np.mean(centered**2)), float(np.mean(centered**4))
    cohort = x.mean(1)
    cc = cohort-cohort.mean()
    c2, c4 = np.mean(cc**2), np.mean(cc**4)
    leave = []
    for i in range(s):
        other = x[np.arange(s) != i]
        leave.append(n * np.mean((other-other.mean(0))**2)*(s-1)/(s-2))
    leave = np.array(leave)
    result = dict(mean=float(q.mean()), susceptibility=n*m2*s/(s-1),
        central_second=m2, central_fourth=m4,
        fourth_ratio=m4/m2**2 if m2 else None,
        empirical_percentiles=np.quantile(boot, [.025, .975]),
        jackknife_se=np.sqrt((s-1)/s*np.sum((leave-leave.mean())**2)),
        leave_one_out=leave, cohort_means=cohort,
        cohort_susceptibility=n*np.mean(cc**2)*s/(s-1),
        cohort_fourth_ratio=c4/c2**2 if c2 else None,
        cohort_binder=1-c4/(3*c2**2) if c2 else None,
        finite_seed_gaussian_binder_mean=2/(s+1),
        bootstrap_samples=len(w), bootstrap_zero_fraction=float(np.mean(boot == 0)),
        independent_seeds=s, contexts=q.shape[1])
    return result, boot


def check_moments(q, n, reported, label):
    actual, boot = moments(q, n)
    allowance = bootstrap_roundoff_allowance(q, n)
    for key, value in actual.items():
        if key not in reported:
            raise AssertionError('Reported moment is missing: '+key)
        if value is None:
            if reported[key] is not None:raise AssertionError('Undefined moment was assigned a value')
        elif key == 'empirical_percentiles':
            compare_bootstrap(value, reported[key], allowance, label+'_percentiles')
        else:
            close(value, reported[key])
    return actual, boot, allowance


def load_fields(paths, precision):
    arrays = defaultdict(list)
    for path in paths:
        with np.load(Path(path).parent/'measurements.npz') as z:
            for suffix in ['fields', 'heads', 'centroids', 'energies', 'mean_metric', 'evaluation_mean_metric']:
                key = precision+'_'+suffix
                if key in z.files:arrays[suffix].append(z[key].astype(np.float64))
    data = {k:np.stack(v) for k,v in arrays.items() if len(v) == len(paths)}
    f, h, c, e, m = [data[k] for k in ['fields', 'heads', 'centroids', 'energies', 'mean_metric']]
    centroid = c.mean(axis=3)
    fields = dict(row_native=h[..., 2].mean(axis=3),
        row_reduced64=(e[..., 0]/np.maximum(e[..., 1], 1e-30)).mean(axis=3),
        absolute_row_energy=e[..., 0].mean(axis=3), total_energy=e[..., 1].mean(axis=3),
        centroid_energy=np.mean(c*c, axis=(3,4)), common_centroid=centroid,
        head_common_energy=np.mean(centroid**2, axis=-1),
        head_residual_energy=np.mean((c-centroid[:,:,:,None,:])**2, axis=(3,4)))
    fields['head_alignment'] = fields['head_common_energy']/np.maximum(fields['centroid_energy'], 1e-30)
    fields.update(mean_metric_total=m, mean_metric_common=m.mean(axis=3),
        mean_metric_contrast=m-m.mean(axis=3, keepdims=True),
        attention=h[..., 0].mean(axis=3), operator_rms=h[..., 3].mean(axis=3),
        prediction_entropy=f[..., 24], nll=f[..., 25], logit_projection=f[..., 16:24])
    if 'evaluation_mean_metric' in data:
        m = data['evaluation_mean_metric']
        fields.update(evaluation_mean_metric_total=m, evaluation_mean_metric_common=m.mean(axis=3),
            evaluation_mean_metric_contrast=m-m.mean(axis=3, keepdims=True))
    return fields, c


class Files:
    def __init__(self):
        self.cache = {}

    def digest(self, path):
        path = Path(path).resolve()
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in self.cache:self.cache[key] = sha256(path)
        return self.cache[key]

    def check(self, path, expected):
        if self.digest(path) != expected:raise AssertionError('Changed bound evidence: '+str(path))

    def analysis(self, path):
        path = Path(path)
        meta = read(path/'manifest.json')
        if meta['status'] != 'complete':raise AssertionError('Incomplete selected analysis')
        for file, key in [('binding.json','binding_sha256'), ('results.json','results_sha256'),
                          ('measurements.npz','raw_sha256')]:
            if key in meta:self.check(path/file, meta[key])
        binding = read(path/'binding.json')
        for file, signature in binding['inputs'].items():self.check(file, signature)
        for file, signature in binding['source_files'].items():self.check(path/'source'/file, signature)
        return read(path/'results.json')


def condition_key(row):
    return tuple(row[k] for k in ['heads','multiplier','step','seed_count','shared_seed','stream_seed','precision'])


def verify_collectives(path, files):
    report = files.analysis(path)
    cache, checks = {}, 0
    with np.load(Path(path)/'measurements.npz') as raw:
        for row in report['conditions']:
            n, s, precision = row['heads'], row['seed_count'], row['precision']
            fields, c = load_fields(row['sources'], precision)
            if set(fields) != set(row['observables']):raise AssertionError('Observable inventory changed')
            stored = {}
            for name, q in fields.items():
                label = f'{precision}_N{n}_g{row["multiplier"]}_t{row["step"]}_s{s}_c{row["shared_seed"]}_b{row["stream_seed"]}_{name}'
                stats, boot, allowance = check_moments(q, n, row['observables'][name], label)
                compare_bootstrap(boot, raw[label+'_bootstrap'], allowance, label+'_bootstrap')
                close(q.reshape(s,-1).mean(1), raw[label+'_cohort'])
                stored[name] = dict(stats=stats, bootstrap=boot, roundoff_allowance=allowance)
                checks += 1
            one = np.var(c, axis=0, ddof=1).mean()
            common = row['observables']['common_centroid']['susceptibility']
            close(one, row['centroid_one_head_variance'])
            close((common-one)/(n-1), row['centroid_cross_head_covariance'])
            close(n*one/common, row['centroid_effective_head_count'])
            for prefix, key in [('mean_metric', 'calibration_common_fraction'),
                                ('evaluation_mean_metric', 'evaluation_common_fraction')]:
                if prefix+'_total' not in fields:continue
                t, cvar, residual = [row['observables'][prefix+'_'+k]['susceptibility'] for k in ['total','common','contrast']]
                close(t, cvar+residual)
                close(cvar/t, row[key])
            cache[condition_key(row)] = dict(record=row, moments=stored)
            print('Verified collective statistics', precision, n, row['multiplier'], row['step'], s, flush=True)
    for row in report['paired_arithmetic']:
        key = {**row, 'precision':'float32'}
        left = cache[condition_key(key)]['record']
        key['precision'] = 'float64'
        right = cache[condition_key(key)]['record']
        x, _ = load_fields(left['sources'], 'float32')
        y, _ = load_fields(right['sources'], 'float64')
        if set(row['observables']) != set(x)&set(y):raise AssertionError('Arithmetic fields omitted')
        n = row['heads']
        for name, r in row['observables'].items():
            cx, cy, ce = [n*np.var(q, axis=0, ddof=1).mean() for q in [x[name], y[name], x[name]-y[name]]]
            bound = 2*np.sqrt(cy*ce)+ce
            close([cx,cy,ce,abs(cx-cy),bound], [r[k] for k in ['float32_susceptibility','float64_susceptibility',
                'centered_error_susceptibility','absolute_difference','susceptibility_error_bound']])
            if r['meets_one_percent_or_1e_minus8_absolute'] != (bound <= max(.01*cy,1e-8)):
                raise AssertionError('Arithmetic reporting rule changed')
    return cache, dict(conditions=len(report['conditions']), moment_fields=checks, arithmetic_conditions=len(report['paired_arithmetic']))


def verify_paths(path, files, cache):
    report = files.analysis(path)
    fields = 0
    for row in report['paired_horizons']:
        entries = []
        for step in [row['early'],row['late']]:
            entries.append(cache[condition_key({**row,'step':step})])
        s = row['seed_count']
        paired_fields = [load_fields(entry['record']['sources'], row['precision'])[0] for entry in entries]
        for name, r in row['observables'].items():
            a,b = [entry['moments'][name] for entry in entries]
            sa,sb = a['stats'],b['stats']
            difference = paired_fields[1][name]-paired_fields[0][name]
            flat_difference = difference.reshape(s, -1)
            d = np.sum(flat_difference, axis=1)/flat_difference.shape[1]
            close(r['paired_field_rms'], np.linalg.norm(flat_difference.ravel())/np.sqrt(flat_difference.size))
            relative = sb['susceptibility']/sa['susceptibility']-1 if sa['susceptibility'] else None
            if relative is None:
                if r['finite_relative_susceptibility_change'] is not None:raise AssertionError('Undefined relative change was assigned a value')
            else:close(r['finite_relative_susceptibility_change'], relative)
            loo = sb['leave_one_out']-sa['leave_one_out']
            close([r['before']['susceptibility'], r['after']['susceptibility'], r['susceptibility_change'], r['mean_change']],
                  [sa['susceptibility'],sb['susceptibility'],sb['susceptibility']-sa['susceptibility'],d.mean()])
            compare_bootstrap(r['susceptibility_change_empirical_percentiles'],
                np.quantile(b['bootstrap']-a['bootstrap'],[.025,.975]),
                a['roundoff_allowance']+b['roundoff_allowance'],
                str(condition_key({**row,'step':row['late']}))+'_'+name+'_paired_percentiles')
            close(r['susceptibility_change_jackknife_se'],np.sqrt((s-1)/s*np.sum((loo-loo.mean())**2)))
            close(r['mean_change_empirical_percentiles'],np.quantile(multiplicities(s)@d/s,[.025,.975]))
            close(r['per_initialization_mean_change'],d)
            fields += 1
        del paired_fields
        print('Verified paired statistical fields', row['precision'], row['heads'], row['multiplier'], row['early'], row['late'], s, flush=True)
    return dict(paired_groups=len(report['paired_horizons']), paired_fields=fields,
        batch_shape_groups=len(report['batch_shape_comparisons']))


def verify_factorial(path, files):
    report = files.analysis(path)
    checked = 0
    with np.load(Path(path)/'measurements.npz') as raw:
        for row in report['conditions']:
            n = row['heads']
            for name, r in row['observables'].items():
                q = raw[f'h{n}_{name}_factorial']
                # Each factor is projected with either its averaging or its
                # centered operator. Their tensor products are orthogonal.
                contributions = {}
                for bits in itertools.product([0,1], repeat=3):
                    if not any(bits):continue
                    effect = q.copy()
                    for axis, centered in enumerate(bits):
                        mean = effect.mean(axis=axis, keepdims=True)
                        effect = effect-mean if centered else mean
                    label = ','.join(k for k,active in zip(['shared','batch','initialization'],bits) if active)
                    contributions[label] = float(np.mean(effect**2))
                if set(contributions) != set(r['components']):raise AssertionError('A finite environment sector was omitted')
                for k,v in contributions.items():close(v,r['components'][k])
                total = np.mean((q-q.mean(axis=(0,1,2),keepdims=True))**2)
                close(total,r['total_conditional_population_variance'])
                close(sum(contributions.values()),total)
                within = n*np.var(q,axis=2,ddof=1).mean()
                close(within,r['average_conditional_seed_susceptibility'])
                close(within,n*4/3*sum(v for k,v in contributions.items() if 'initialization' in k))
                checked += 1
    return dict(conditions=len(report['conditions']), fields=checked)


def verify_variance(path, files):
    report = files.analysis(path)
    study = Path(path).parent.parent
    frozen = files.analysis(study/'analysis/frozen-variance')
    lookup = {r['observable']:r for r in frozen['forecasts']}
    target_moments = {}
    for n in frozen['target_widths']:
        paths = [study/'measurements'/f'cpu-h{n}-g1-t16384-s{seed}'/'manifest.json'
                 for seed in frozen['target_initializations']]
        observations, _ = load_fields(paths, 'float32')
        for name in lookup:
            target_moments[n,name] = (*moments(observations[name],n), bootstrap_roundoff_allowance(observations[name],n))
    with np.load(Path(path)/'measurements.npz') as raw, np.load(study/'analysis/frozen-variance/measurements.npz') as cal:
        for r in report['forecasts']:
            n,name,model = r['heads'],r['observable'],r['model']
            i = frozen['target_widths'].index(n)
            prediction = lookup[name]['models'][model]['prediction'][i]
            close(prediction,r['prediction'])
            close(r['actual']-prediction,r['signed_error'])
            target = raw[f'N{n}_{name}_target_bootstrap']
            actual, reconstructed_bootstrap, allowance = target_moments[n,name]
            close(r['actual'],actual['susceptibility'])
            compare_bootstrap(target,reconstructed_bootstrap,allowance,f'N{n}_{name}_variance_target')
            expected = cal[name+'_'+model+'_bootstrap'][:,i]
            close(np.quantile(target[:,None]-expected[None,:],[.025,.975]),r['product_resampling_error_percentiles'])
    for row in report['aggregate']:
        rs = [r for r in report['forecasts'] if r['observable']==row['observable'] and r['model']==row['model']]
        errors = [(r['actual']-r['prediction'])/r['prediction'] for r in rs]
        close(np.sqrt(np.mean(np.square(errors))),row['relative_rmse'])
        close(np.max(np.abs(errors)),row['maximum_absolute_relative_error'])
    if len(report['forecasts']) != 144:raise AssertionError('Frozen variance inventory incomplete')
    return dict(predictions=len(report['forecasts']), aggregate_alternatives=len(report['aggregate']))


def main():
    p = argparse.ArgumentParser();p.add_argument('--root', required=True)
    p.add_argument('--study', default='critical-scaling-20260906');p.add_argument('--output', required=True)
    p.add_argument('--collective-snapshot', help='Check only this completed named analysis; never a complete release gate')
    a = p.parse_args();study = Path(a.root)/a.study;repo = Path(__file__).resolve().parents[1]
    if Path(a.output).exists():raise FileExistsError(a.output)
    sources = {k:sha256(repo/k) for k in ['scripts/verify_scaling_statistics.py','src/model_rg/provenance.py']}
    files = Files()
    path = study/'analysis'/(a.collective_snapshot or 'collectives')
    cache, collectives = verify_collectives(path,files)
    checks = dict(collectives=collectives)
    if not a.collective_snapshot:
        checks['paired_paths'] = verify_paths(study/'analysis/paired-paths',files,cache)
        checks['factorial'] = verify_factorial(study/'analysis/environment-factorial',files)
        checks['variance_predictions'] = verify_variance(study/'analysis/variance-scores',files)
    for name,signature in sources.items():files.check(repo/name,signature)
    write_json(a.output,dict(schema='native-scaling-statistical-verification-v1',
        status='partial_snapshot_passed' if a.collective_snapshot else 'passed',arguments=vars(a),
        checks=checks,bootstrap_roundoff_comparisons=BOOTSTRAP_COMPARISONS,
        verifier_sources=sources,verifier_sha256=sha256(__file__),
        hashed_file_versions=len(files.cache),
        scope='Independent reconstruction of all collective central moments and whole-identity empirical resampling, paired arithmetic and horizon summaries, finite environment sectors and frozen variance-score arithmetic. Raw model lineage, fresh-noise derivatives and full clock predictions have separate checks; no sampling-coverage or thermodynamic claim is certified.'))
    print('Statistical reconstruction passed',checks,flush=True)


if __name__=='__main__':main()
