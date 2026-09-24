#!/usr/bin/env python
"""Analyze every selected scheduled collective with whole-initialization pairing."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from analyze_scaling_collectives import collective_fields, arithmetic_comparison
from analyze_size_time import whole_seed_statistics
from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--study', default='scheduled-training-20260908')
    p.add_argument('--output', required=True)
    a = p.parse_args(); study = Path(a.root).resolve() / a.study
    analysis_protocol = study/'protocols/scheduled-analysis-selection.json'
    contract = json.loads(analysis_protocol.read_text())
    for name, digest in contract['producer_sources'].items():
        if sha256(Path(__file__).resolve().parents[1]/name) != digest:
            raise AssertionError('A recorded scheduled analysis implementation changed')
    protocol = study/'protocols/regime-observation-selection.json'
    spec = json.loads(protocol.read_text())
    if len(spec['cases']) != 320:
        raise AssertionError('The complete selected 320-state panel is required')
    groups = defaultdict(dict); inputs = [protocol, analysis_protocol]
    for case in spec['cases']:
        folder = study/'measurements'/case['name']
        meta = json.loads((folder/'manifest.json').read_text())
        verification = study/'verification/observations'/(case['name']+'.json')
        check = json.loads(verification.read_text())
        if meta['status'] != 'complete' or check['status'] != 'passed' or check['case'] != case:
            raise AssertionError('Every selected observation must be complete and independently verified')
        for key, value in case.items():
            if meta['case'][key] != value:
                raise AssertionError('The selected collective condition changed')
        for name in ['manifest.json', 'measurements.npz']:
            path = folder/name; signature = sha256(path)
            if check['checked_sha256'][str(path)] != signature:
                raise AssertionError('A verified collective artifact changed')
            inputs.append(path)
        inputs.append(verification)
        group = groups[case['recipe'], case['heads'], case['step'], case['shared_seed'], case['stream_seed']]
        if case['seed'] in group:
            raise AssertionError('Duplicate initialization in a scheduled condition')
        group[case['seed']] = folder
    if len(groups) != 80:
        raise AssertionError('The complete eighty-condition collective panel is required')
    out = Path(a.output).resolve(); out.mkdir(parents=True, exist_ok=False)
    bind_run(out, inputs, vars(a)); records = []; paired = []; raw = {}
    for (recipe, n, step, shared, stream), folders in sorted(groups.items()):
        seeds = list(range(640101, 640105))
        if sorted(folders) != seeds:
            raise AssertionError('Every condition must retain the same four initialization identities')
        precision_fields = {}
        for precision in ['float32', 'float64']:
            columns = defaultdict(list)
            for seed in seeds:
                with np.load(folders[seed]/'measurements.npz') as z:
                    for field in ['fields', 'heads', 'centroids', 'energies', 'mean_metric', 'context_kl', 'evaluation_mean_metric']:
                        columns[field].append(z[precision+'_'+field].astype(float))
            data = {k:np.stack(v) for k, v in columns.items()}; del columns
            fields = collective_fields(data, 4, n); precision_fields[precision] = fields
            c = data['centroids']
            record = dict(recipe=recipe, heads=n, step=step, precision=precision, seeds=seeds,
                shared_seed=shared, stream_seed=stream, observables={})
            for field, values in fields.items():
                stats, bootstrap = whole_seed_statistics(values, n)
                record['observables'][field] = stats
                label = f'{recipe}_N{n}_t{step}_{precision}_{field}'
                raw[label+'_bootstrap'] = bootstrap
                raw[label+'_cohort'] = values.reshape(4, -1).mean(-1)
            moments = record['observables']
            for name in ['mean_metric', 'evaluation_mean_metric']:
                total = moments[name+'_total']['susceptibility']
                common = moments[name+'_common']['susceptibility']
                contrast = moments[name+'_contrast']['susceptibility']
                np.testing.assert_allclose(total, common+contrast, rtol=1e-11, atol=1e-20)
                record[name+'_common_fraction'] = common/total if total > 0 else None
            one_head = float(np.var(c, axis=0, ddof=1).mean())
            collective = moments['common_centroid']['susceptibility']
            record.update(centroid_one_head_variance=one_head,
                centroid_cross_head_covariance=(collective-one_head)/(n-1),
                centroid_effective_head_count=n*one_head/collective if collective > 0 else None,
                mean_context_kl=float(data['context_kl'].mean()))
            records.append(record)
            print(recipe, n, step, precision, 'row', moments['row_native']['mean'],
                  'common susceptibility', collective, 'NLL', moments['nll']['mean'], flush=True)
            del data, c, fields
        paired.append(dict(recipe=recipe, heads=n, step=step, seeds=seeds,
            shared_seed=shared, stream_seed=stream,
            observables=arithmetic_comparison(precision_fields['float32'], precision_fields['float64'], n)))
        del precision_fields
    write_json(out/'results.json', dict(status='complete', schema='scheduled-collective-analysis-v1',
        conditions=records, paired_arithmetic=paired, selected_states=320, groups=80,
        contexts=512, calibration_contexts=64, seed_count=4,
        interpretation='All eighty selected conditions and both arithmetic programs are retained. Recipe labels describe reference settings, not inferred phase classifications. Two head counts do not establish an asymptotic critical exponent. Common learned seed fluctuations are distinct from input variation at a fixed checkpoint.',
        uncertainty='The same four whole initialization identities are used at every selected condition. All 256 empirical four-seed resamples and delete-one diagnostics are retained, conditional on the fixed shared initialization, batch history and context law. Empirical percentile ranges have no guaranteed population coverage.',
        units='Mean-square-entry coordinates; unbiased whole-seed covariance at fixed context; intensive row fields average heads before computing seed variance.'))
    np.savez_compressed(out/'measurements.npz', **raw)
    write_json(out/'manifest.json', dict(status='complete', binding_sha256=sha256(out/'binding.json'),
        results_sha256=sha256(out/'results.json'), raw_sha256=sha256(out/'measurements.npz'), groups=80))


if __name__ == '__main__':
    main()
