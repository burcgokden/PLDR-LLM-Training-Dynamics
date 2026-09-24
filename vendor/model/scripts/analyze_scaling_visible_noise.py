#!/usr/bin/env python
"""Reconstruct all fresh-batch source predictions, retaining drift and covariance errors."""
import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
import time

import numpy as np

from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


def main():
    p = argparse.ArgumentParser();p.add_argument('--root', required=True)
    p.add_argument('--study', default='critical-scaling-20260906');p.add_argument('--output', required=True)
    p.add_argument('--wait', action='store_true');a = p.parse_args();study = Path(a.root)/a.study
    launcher = study/'launcher-visible-noise.json'
    while json.loads(launcher.read_text())['status'] != 'complete':
        status = json.loads(launcher.read_text())['status']
        if not a.wait or status != 'running':raise RuntimeError('Source validation executor: '+status)
        time.sleep(30)
    protocol = study/'protocols/visible-noise.json';spec = json.loads(protocol.read_text())
    paths = [study/'measurements'/c['name'] for c in spec['cases']]
    out = Path(a.output);out.mkdir(parents=True, exist_ok=False)
    inputs = [protocol, launcher]
    for path in paths:inputs.extend(path/n for n in ['manifest.json', 'results.json', 'predictor.json',
        'predictor.npz', 'measurements.npz', 'validation-directions.npy'])
    bind_run(out, inputs, vars(a));records, raw = [], {}
    expected_rows = np.random.default_rng(spec['validation_batch_seed']+1000).integers(0,3072,size=(spec['validation_batches'],32))
    expected_offsets = np.random.default_rng(spec['validation_batch_seed']+1001000).integers(0,449,size=expected_rows.shape)
    for path in paths:
        meta = json.loads((path/'manifest.json').read_text());result = json.loads((path/'results.json').read_text())
        if meta['status'] != 'complete':raise AssertionError('A selected validation is incomplete')
        for name, key in [('results.json','results_sha256'), ('predictor.npz','predictor_sha256'),
                          ('measurements.npz','raw_sha256'), ('validation-directions.npy','directions_sha256')]:
            if sha256(path/name) != meta[key]:raise AssertionError('A source-validation artifact changed')
        predictor = json.loads((path/'predictor.json').read_text())
        if predictor != result['predictor'] or predictor['predictor_sha256'] != meta['predictor_sha256']:
            raise AssertionError('The frozen source predictor changed during validation')
        with np.load(path/'predictor.npz') as z:
            basis, adjoints = z['basis'], z['adjoints']
            np.testing.assert_allclose(basis@basis.T, np.eye(len(basis)), rtol=1e-10, atol=1e-12)
        directions = np.load(path/'validation-directions.npy', mmap_mode='r')
        predicted_coefficients = np.asarray(directions)@adjoints.T
        with np.load(path/'measurements.npz') as z:
            np.testing.assert_array_equal(z['rows'], expected_rows);np.testing.assert_array_equal(z['offsets'], expected_offsets)
            teacher, coefficients = z['teacher'], z['source_coefficients']
            np.testing.assert_allclose(coefficients, predicted_coefficients, rtol=2e-8, atol=1e-11)
            oracle = teacher@basis.T
            error = np.linalg.norm(coefficients-oracle, axis=1)
            norm = np.linalg.norm(oracle, axis=1)
            if np.any(error > spec['duality_absolute_tolerance']+spec['duality_relative_tolerance']*norm):
                raise AssertionError('A fresh source failed adjoint duality')
            for r in result['projections']:
                k = min(r['requested_dimension'], len(basis))
                observed = coefficients[:,:k]@basis[:k]
                residual = teacher-observed
                total = float(np.mean(np.sum(teacher*teacher,axis=1)))
                remainder = float(np.mean(np.sum(residual*residual,axis=1)))
                centered = teacher-teacher.mean(0)
                centered_residual = residual-residual.mean(0)
                trace = float(np.sum(centered*centered)/(len(teacher)-1))
                residual_trace = float(np.sum(centered_residual*centered_residual)/(len(teacher)-1))
                np.testing.assert_allclose([total,remainder,trace,residual_trace], [r[x] for x in
                    ['total_response_second_moment','residual_second_moment','covariance_trace','residual_covariance_trace']],rtol=2e-10,atol=1e-16)
            for item in result['numerical']:
                k = item['batch'];y = teacher[k];curvature = float(y@y)
                for finite in item['finite_responses']:
                    amp = finite['amplitude'];secant = z[f'b{k}_a{amp}_secant'];kl = float(z[f'b{k}_a{amp}_symmetric_kl'])
                    first = float(np.linalg.norm(secant-y)/max(np.linalg.norm(y),1e-300))
                    fisher = abs(kl-curvature)/max(curvature,1e-300)
                    np.testing.assert_allclose([first,fisher],[finite['first_relative_error'],finite['fisher_relative_error']],rtol=2e-10,atol=1e-15)
                    if finite['local_diagnostic'] != (first<=.01 and fisher<=.05):raise AssertionError('Local diagnostic criterion changed')
            raw[path.name+'_coefficients'] = coefficients
        records.append(result);print(path.name, 'independent source reconstruction complete', flush=True)
    summaries = []
    for n in [4,14]:
        for t in [2048,16384]:
            selected = sorted([r for r in records if r['case']['heads']==n and r['case']['step']==t],key=lambda r:r['case']['seed'])
            if [r['case']['seed'] for r in selected] != list(range(640101,640105)):
                raise AssertionError('The matched training-initialization panel changed')
            for k in spec['dimensions']:
                values = [next(x for x in r['projections'] if x['requested_dimension']==k) for r in selected]
                summaries.append(dict(heads=n,step=t,requested_dimension=k,
                    total_response_residual_fractions=[r['total_response_residual_fraction'] for r in values],
                    covariance_residual_fractions=[r['covariance_residual_fraction'] for r in values],
                    pooled_total_response_residual_fraction=sum(r['residual_second_moment'] for r in values)/sum(r['total_response_second_moment'] for r in values),
                    pooled_covariance_residual_fraction=sum(r['residual_covariance_trace'] for r in values)/sum(r['covariance_trace'] for r in values)))
    draws = np.array(list(itertools.product(range(4),repeat=4)))
    paired = []
    for n in [4,14]:
        for k in spec['dimensions']:
            old = next(r for r in summaries if r['heads']==n and r['step']==2048 and r['requested_dimension']==k)
            new = next(r for r in summaries if r['heads']==n and r['step']==16384 and r['requested_dimension']==k)
            for field in ['total_response_residual_fractions','covariance_residual_fractions']:
                change = np.array(new[field])-np.array(old[field]);boot = change[draws].mean(1)
                paired.append(dict(heads=n,requested_dimension=k,field=field,per_initialization_change=change.tolist(),
                    mean_change=float(change.mean()),empirical_change_percentiles=np.quantile(boot,[.025,.975]).tolist()))
    locality = {str(amp):dict(total=0,passed=0) for amp in spec['amplitudes']}
    for r in records:
        for numerical in r['numerical']:
            for finite in numerical['finite_responses']:
                row = locality[str(finite['amplitude'])];row['total'] += 1;row['passed'] += finite['local_diagnostic']
    result = dict(schema='fresh-visible-noise-analysis-v1',status='complete',states=records,conditions=summaries,
        paired_state_changes=paired,locality=locality,
        scope='All16 fixed incoming states,16 fresh batches per state,five declared dimensions and three magnitudes. Conditional mean response and conditional covariance are reported separately. Four initialization identities are paired across states and widths. The finite empirical intervals condition on the fixed fresh-batch realization and have no guaranteed population coverage. No generator-only local reduction is interpreted as autonomous full-model training dynamics.')
    write_json(out/'results.json',result);np.savez_compressed(out/'measurements.npz',**raw)
    write_json(out/'manifest.json',dict(status='complete',binding_sha256=sha256(out/'binding.json'),
        results_sha256=sha256(out/'results.json'),raw_sha256=sha256(out/'measurements.npz'),states=len(records)))


if __name__=='__main__':main()
