#!/usr/bin/env python
"""Analyze every selected single-pass collective under its exact drive identity."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from analyze_scaling_collectives import collective_fields
from onepass_analysis import group_key, group_record, label, statistics, arithmetic
from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True)
    p.add_argument('--study', default='scheduled-training-feasible-20260908'); a = p.parse_args()
    study = Path(a.root).resolve()/a.study; repo = Path(__file__).resolve().parents[1]
    protocol = study/'protocols/onepass-analysis-selection.json'
    spec = json.loads(protocol.read_text())
    for name, digest in spec['producer_sources'].items():
        if sha256(repo/name) != digest:
            raise AssertionError('The recorded single-pass analysis implementation changed')
    for name, digest in spec['inputs_sha256'].items():
        if sha256(name) != digest:
            raise AssertionError('The single-pass analysis selection changed')
    selection = study/'protocols/onepass-observation-selection.json'
    cases = json.loads(selection.read_text())['cases']; inputs = [protocol, selection]
    groups = defaultdict(dict)
    for case in cases:
        folder = study/'measurements'/case['name']
        proof_path = study/'verification/observations'/(case['name']+'.json')
        meta = json.loads((folder/'manifest.json').read_text()); proof = json.loads(proof_path.read_text())
        if meta['status'] != 'complete' or proof['status'] != 'passed' or proof['case'] != case:
            raise AssertionError('Every selected collective requires a complete independent check')
        for key, value in case.items():
            if meta['case'][key] != value:
                raise AssertionError('A selected collective identity changed')
        for name in ['manifest.json', 'measurements.npz']:
            path = folder/name
            if sha256(path) != proof['checked_sha256'][str(path)]:
                raise AssertionError('An independently reconstructed collective changed')
            inputs.append(path)
        inputs.append(proof_path); key = group_key(case)
        if case['seed'] in groups[key]:
            raise AssertionError('Two selected states duplicate a drive and initialization identity')
        groups[key][case['seed']] = folder
    expected = {tuple(r[k] for k in spec['group_keys']): r['seeds'] for r in spec['groups']}
    if {k: sorted(v) for k, v in groups.items()} != expected:
        raise AssertionError('The selected collective group coverage changed')
    out = study/'analysis/onepass-collectives'; out.mkdir(parents=True, exist_ok=False)
    bind_run(out, inputs, vars(a)); records = []; paired = []; raw = {}
    for key, folders in sorted(groups.items()):
        seeds = sorted(folders); count = len(seeds); n = key[2]; precision_fields = {}
        for precision in spec['precisions']:
            columns = defaultdict(list)
            for seed in seeds:
                with np.load(folders[seed]/'measurements.npz') as z:
                    for field in ['fields', 'heads', 'centroids', 'energies', 'mean_metric', 'context_kl', 'evaluation_mean_metric']:
                        columns[field].append(z[precision+'_'+field].astype(float))
            data = {k: np.stack(v) for k, v in columns.items()}; del columns
            fields = collective_fields(data, count, n); precision_fields[precision] = fields
            record = dict(**group_record(key), precision=precision, seeds=seeds, observables={})
            for field, values in fields.items():
                stats, bootstrap, cohort = statistics(values, n)
                record['observables'][field] = stats
                identifier = label(key)+'_'+precision+'_'+field
                raw[identifier+'_bootstrap'] = bootstrap; raw[identifier+'_cohort'] = cohort
            moments = record['observables']
            for name in ['mean_metric', 'evaluation_mean_metric']:
                total = moments[name+'_total']['susceptibility']
                common = moments[name+'_common']['susceptibility']
                contrast = moments[name+'_contrast']['susceptibility']
                if count > 1:
                    np.testing.assert_allclose(total, common+contrast, rtol=1e-11, atol=1e-20)
                record[name+'_common_fraction'] = common/total if total is not None and total > 0 else None
            one = float(np.var(data['centroids'], axis=0, ddof=1).mean()) if count > 1 else None
            chi = moments['common_centroid']['susceptibility']
            record.update(centroid_one_head_variance=one,
                centroid_cross_head_covariance=(chi-one)/(n-1) if count > 1 else None,
                centroid_effective_head_count=n*one/chi if chi is not None and chi > 0 else None,
                mean_context_kl=float(data['context_kl'].mean()))
            records.append(record)
            print(label(key), precision, 'seeds', count, 'row', moments['row_native']['mean'],
                  'common susceptibility', chi, 'NLL', moments['nll']['mean'], flush=True)
            del data, fields
        paired.append(dict(**group_record(key), seeds=seeds,
                           observables=arithmetic(precision_fields['float32'], precision_fields['float64'], n)))
        del precision_fields
    write_json(out/'results.json', dict(status='complete', schema='onepass-collective-analysis-v1',
        conditions=records, paired_arithmetic=paired, selected_states=len(cases), groups=len(groups),
        contexts=512, calibration_contexts=64, interpretation=spec['criticality_scope'],
        uncertainty=spec['uncertainty'], single_seed_scope=spec['single_seed_scope'],
        units='Fixed native coordinates and mean-square-entry units. Heads are averaged before whole-initialization variance for intensive fields. The full evaluation and calibration matrices retain separate laws.'))
    np.savez_compressed(out/'measurements.npz', **raw)
    write_json(out/'manifest.json', dict(status='complete', binding_sha256=sha256(out/'binding.json'),
        results_sha256=sha256(out/'results.json'), raw_sha256=sha256(out/'measurements.npz'),
        states=len(cases), groups=len(groups)))


if __name__ == '__main__': main()
