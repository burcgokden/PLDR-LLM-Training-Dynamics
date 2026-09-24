#!/usr/bin/env python
"""Independently reconstruct exact conditional native-step source decompositions."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp

from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


FIELDS = dict(row_energy=0, row_ratio=2, operator_rms=4, attention_entropy=5, head_rms=6)


def decompose(base, generator, body, full):
    """Unit Euclidean coordinates; no source independence or locality assumption."""
    dg, db, df = generator-base, body-base, full-base
    components = np.stack([dg, db, df-dg-db])
    mean = components.mean(1)
    centered = components-mean[:, None]
    s = len(full)
    covariance = np.einsum('asf,bsf->ab', centered, centered)/(s-1)
    drift_gram = mean@mean.T
    fm = df.mean(0);fc = df-fm
    noise = float(np.sum(fc*fc)/(s-1))
    second = float(np.mean(np.sum(df*df, axis=1)))
    np.testing.assert_allclose(covariance.sum(), noise, rtol=2e-10, atol=1e-20)
    np.testing.assert_allclose(drift_gram.sum(), fm@fm, rtol=2e-10, atol=1e-20)
    np.testing.assert_allclose(second, fm@fm+(s-1)*noise/s, rtol=2e-10, atol=1e-20)
    def ratio(x, y):return float(x/y) if y > 0 else None
    missing = df-dg;mc = missing-missing.mean(0)
    return dict(samples=s, features=full.shape[1], component_order=['generator', 'body', 'mixed'],
        base_squared_norm=float(base@base),
        component_mean_gram=drift_gram.tolist(), component_covariance=covariance.tolist(),
        component_cohort_mean_increments=(mean.mean(-1)*np.sqrt(full.shape[1])).tolist(),
        component_cohort_mean_covariance=np.cov(components.mean(-1)*np.sqrt(full.shape[1]), ddof=1).tolist(),
        component_radial_mean_terms=(2*mean@base).tolist(),
        full_mean_squared_norm=float(fm@fm), full_covariance_trace=noise, full_second_moment=second,
        unbiased_squared_drift_estimate=float(fm@fm-noise/s),
        drift_to_noise_ratio=ratio(fm@fm-noise/s, noise),
        component_second_moments=(np.diag(drift_gram)+(s-1)*np.diag(covariance)/s).tolist(),
        covariance_contributions_relative_to_full=(covariance/noise).tolist() if noise > 0 else None,
        generator_only_relative_second_moment_error=ratio(np.mean(np.sum(missing*missing, axis=1)), second),
        generator_only_relative_covariance_error=ratio(np.sum(mc*mc)/(s-1), noise),
        mixed_relative_second_moment=ratio(np.mean(np.sum(components[2]**2, axis=1)), second),
        full_cohort_mean_increment=float(df.mean()*np.sqrt(full.shape[1])),
        full_cohort_mean_variance=float(np.var(df.mean(-1)*np.sqrt(full.shape[1]), ddof=1)))


def reconstruct(folder, case, spec):
    meta = json.loads((folder/'manifest.json').read_text())
    result = json.loads((folder/'results.json').read_text())
    if meta['status'] != 'complete' or result['status'] != 'complete' or result['case'] != case:
        raise AssertionError('A selected conditional state is missing or changed')
    for filename, key in [('measurements.npz', 'raw_sha256'), ('results.json', 'results_sha256'),
                          ('binding.json', 'binding_sha256'), ('first-step-digests.json', 'first_step_digests_sha256')]:
        if sha256(folder/filename) != meta[key]:raise AssertionError('Changed native step data: '+filename)
    if (result['batch_seed'], result['batch_replicas']) != (spec['batch_seed'], spec['batch_replicas']):
        raise AssertionError('The conditional batch selection changed')
    if not result['restored_parent_observations_bitwise_equal']:raise AssertionError('Parent cleanup failed')
    s = spec['batch_replicas'];branches = ['generator', 'body', 'full']
    with np.load(folder/'measurements.npz') as z:
        rows = np.random.default_rng(spec['batch_seed']+1000).integers(0, 3072, (s, 32))
        offsets = np.random.default_rng(spec['batch_seed']+1001000).integers(0, 449, (s, 32))
        np.testing.assert_array_equal(z['rows'], rows);np.testing.assert_array_equal(z['offsets'], offsets)
        np.testing.assert_array_equal(z['cohort'], np.arange(512, 528))
        for key in z.files:
            if not np.isfinite(z[key]).all():raise AssertionError('Nonfinite raw native step observation')
        logp = z['base_logits']-logsumexp(z['base_logits'], axis=-1, keepdims=True)
        probability = np.exp(logp)
        observations, distances = {}, {}
        for prefix in ['base', *branches]:
            fields, centroids = z[prefix+'_fields'], z[prefix+'_centroids']
            np.testing.assert_allclose(fields[..., 3], np.mean(centroids**2, axis=-1), rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(fields[..., 1], fields[..., 0]+fields[..., 3], rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(fields[..., 2], fields[..., 0]/np.maximum(fields[..., 1], 1e-30), rtol=1e-12, atol=1e-14)
        for name in [*FIELDS, 'common_centroid', 'hidden_rms', 'nll', 'prediction_entropy', 'predictive_fisher']:
            values = []
            for prefix in ['base', *branches]:
                if name in FIELDS:value = z[prefix+'_fields'][..., FIELDS[name]].mean(-1)
                elif name == 'common_centroid':value = z[prefix+'_centroids'].mean(-2)
                elif name == 'predictive_fisher':
                    value = z[prefix+'_logits']-z['base_logits']
                    value = (value-np.sum(value*probability, axis=-1, keepdims=True))*np.sqrt(probability)
                else:value = z[prefix+'_'+name]
                value = value.reshape(-1) if prefix == 'base' else value.reshape(s, -1)
                normalizer = np.sqrt(len(probability)) if name == 'predictive_fisher' else np.sqrt(value.shape[-1])
                values.append(value/normalizer)
            observations[name] = decompose(*values)
            if name == 'predictive_fisher':
                # A weighted vocabulary coordinate mean is not a scalar physical drift.
                observations[name].pop('full_cohort_mean_increment')
                observations[name].pop('full_cohort_mean_variance')
                observations[name].pop('component_cohort_mean_increments')
                observations[name].pop('component_cohort_mean_covariance')
                observations[name].pop('component_radial_mean_terms')
            else:
                observations[name]['base_cohort_mean'] = float(values[0].mean()*np.sqrt(len(values[0])))
        for branch in branches:
            logits = z[branch+'_logits']
            other_logp = logits-logsumexp(logits, axis=-1, keepdims=True)
            direct = np.sum(probability*(logp-other_logp), axis=-1)
            np.testing.assert_allclose(direct, z[branch+'_predictive_kl'], rtol=1e-6, atol=2e-12)
            d = logits-z['base_logits']
            d -= np.sum(d*probability, axis=-1, keepdims=True)
            curvature = np.sum(probability*d*d, axis=-1)
            mean_kl, mean_fisher = direct.mean(-1), curvature.mean(-1)
            ratios = 2*mean_kl/mean_fisher
            distances[branch] = dict(kl_by_batch=mean_kl.tolist(), fisher_by_batch=mean_fisher.tolist(),
                twice_kl_over_fisher_by_batch=ratios.tolist(),
                mean_predictive_kl=float(mean_kl.mean()), mean_fisher_distance=float(mean_fisher.mean()),
                ratio_range=[float(ratios.min()), float(ratios.max())])
        controls = dict(loss_range=[float(z['losses'].min()), float(z['losses'].max())],
            global_gradient_norm_range=[float(z['gradient_norms'].min()), float(z['gradient_norms'].max())],
            mean_displacement_squared_norms=z['displacement_squared_norms'].mean(0).tolist())
    return dict(case=case, observations=observations, predictive_distances=distances, controls=controls,
        manifest_sha256=sha256(folder/'manifest.json'))


def aggregate(states):
    result = []
    for heads in sorted({r['case']['heads'] for r in states}):
        for step in sorted({r['case']['step'] for r in states if r['case']['heads'] == heads}):
            cohort = [r for r in states if (r['case']['heads'], r['case']['step']) == (heads, step)]
            for field in cohort[0]['observations']:
                values = [r['observations'][field] for r in cohort]
                covariance = np.array([v['component_covariance'] for v in values]).mean(0)
                gram = np.array([v['component_mean_gram'] for v in values]).mean(0)
                second = np.array([v['component_second_moments'] for v in values]).mean(0)
                noise = np.mean([v['full_covariance_trace'] for v in values])
                full_second = np.mean([v['full_second_moment'] for v in values])
                row = dict(heads=heads, step=step, field=field, seeds=[r['case']['seed'] for r in cohort],
                    state_count=len(cohort), pooled_conditional_covariance=covariance.tolist(),
                    pooled_conditional_mean_gram=gram.tolist(),
                    pooled_full_second_moment=float(full_second), pooled_full_covariance_trace=float(noise),
                    pooled_component_second_moments=second.tolist(),
                    individual_generator_covariance_error=[v['generator_only_relative_covariance_error'] for v in values],
                    individual_mixed_second_fraction=[v['mixed_relative_second_moment'] for v in values],
                    pooled_mixed_second_fraction=float(second[2]/full_second),
                    pooled_generator_covariance_error=float((covariance[1:, 1:].sum())/noise),
                    pooled_generator_second_error=float((gram[1:, 1:].sum()+
                        (values[0]['samples']-1)*covariance[1:, 1:].sum()/values[0]['samples'])/full_second))
                if 'full_cohort_mean_increment' in values[0]:
                    row['individual_base_cohort_means'] = [v['base_cohort_mean'] for v in values]
                    row['individual_full_cohort_mean_increments'] = [v['full_cohort_mean_increment'] for v in values]
                    row['individual_component_cohort_mean_increments'] = [v['component_cohort_mean_increments'] for v in values]
                    row['mean_component_cohort_increments'] = np.mean([v['component_cohort_mean_increments'] for v in values],axis=0).tolist()
                result.append(row)
    return result


def main():
    p = argparse.ArgumentParser();p.add_argument('--root', required=True)
    p.add_argument('--study', default='critical-scaling-20260906')
    p.add_argument('--protocol', default='native-step.json');p.add_argument('--output', required=True)
    p.add_argument('--wait', action='store_true');a = p.parse_args()
    study = Path(a.root)/a.study;protocol = study/'protocols'/a.protocol;spec = json.loads(protocol.read_text())
    folders = [study/'measurements'/case['name'] for case in spec['cases']]
    while not all((folder/'manifest.json').exists() for folder in folders):
        if not a.wait:raise RuntimeError('Every selected native conditional state is required')
        ledger = study/'launcher-native-step.json'
        if ledger.exists() and 'fail' in json.loads(ledger.read_text())['status']:
            raise RuntimeError('The native conditional selection needs execution repair')
        time.sleep(30)
    inputs = [protocol]
    for folder in folders:
        inputs.extend(folder/name for name in ['manifest.json', 'results.json', 'measurements.npz',
                                               'binding.json', 'first-step-digests.json'])
    out = Path(a.output);out.mkdir(parents=True, exist_ok=False);bind_run(out, inputs, vars(a))
    states = [reconstruct(folder, case, spec) for folder, case in zip(folders, spec['cases'], strict=True)]
    result = dict(schema='exact-native-step-source-analysis-v1', status='complete', states=states,
        conditions=aggregate(states), conditional_draws=sum(spec['batch_replicas'] for _ in states),
        scope='Exact four-corner conditional finite-step decomposition. All signed covariance cross terms and every selected state are retained. Pooled errors weight statewise response amplitudes; they do not give a uniform statewise guarantee or estimate variation of conditional means across trained states. Batch draws restore the entire incoming model and Adam state. Finite KL/Fisher ratios are observed nonlinear diagnostics, not assumed Taylor identities.')
    write_json(out/'results.json', result)
    write_json(out/'manifest.json', dict(status='complete', binding_sha256=sha256(out/'binding.json'),
        results_sha256=sha256(out/'results.json'), selected_states=len(states)))
    print('Native conditional-step analysis complete:', len(states), 'states', flush=True)


if __name__ == '__main__':main()
