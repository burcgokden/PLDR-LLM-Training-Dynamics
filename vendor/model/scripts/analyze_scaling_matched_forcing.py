#!/usr/bin/env python
"""Match exactly eight parameter-noise directions with their predictive emission."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


def summarize(gram):
    eigen = np.linalg.eigvalsh((gram+gram.T)/2)
    if eigen.min() < -1e-10*max(float(np.max(np.abs(eigen))), 1e-300):
        raise AssertionError('A noise covariance lost positivity')
    eigen = np.maximum(eigen, 0)[::-1]
    trace = float(eigen.sum())
    return dict(trace=trace, eigenvalues=eigen.tolist(),
        participation_rank=trace*trace/float(eigen@eigen) if trace else 0.,
        top_fraction=float(eigen[0]/trace) if trace else None)


def main():
    p = argparse.ArgumentParser();p.add_argument('--root', required=True)
    p.add_argument('--study', default='critical-scaling-20260906');p.add_argument('--output', required=True)
    a = p.parse_args();study = Path(a.root)/a.study
    protocol = study/'protocols/conditional-noise-jvp.json';spec = json.loads(protocol.read_text())
    selection = dict(cases=[c['name'] for c in spec['cases']], batch_indices=list(range(8)),
        reason='Match the eight already selected emission batches exactly when comparing parameter and predictive noise geometry. The 32-batch parameter spectrum has a different sample-rank bound.',
        all_generator_sectors=True, conditional_scope='Complete incoming native state, fresh IID batches; local generator-only emission.')
    out = Path(a.output);out.mkdir(parents=True, exist_ok=False)
    write_json(out/'selection.json', selection)
    inputs = [protocol, out/'selection.json']
    paths = [study/'measurements'/c['name'] for c in spec['cases']]
    for path in paths:inputs.extend(path/n for n in ['manifest.json', 'results.json', 'generator-velocities.npy', 'measurements.npz'])
    bind_run(out, inputs, vars(a));records, raw = [], {}
    for path in paths:
        meta = json.loads((path/'manifest.json').read_text());result = json.loads((path/'results.json').read_text())
        if meta['status'] != 'complete':raise AssertionError('Conditional state incomplete')
        for name, key in [('generator-velocities.npy', 'velocities_sha256'), ('measurements.npz', 'raw_sha256')]:
            if sha256(path/name) != meta[key]:raise AssertionError('Conditional direction or emission changed')
        parent = json.loads(Path(meta['case']['parent_manifest']).read_text())
        rate = 3e-4*parent['arguments']['multiplier']
        velocity = np.load(path/'generator-velocities.npy', mmap_mode='r')
        grams = {key:np.zeros((8,8)) for key in ['shared_metric_network', 'plga_affine', 'all_generator']}
        for name, (lo, hi) in zip(result['parameter_names'], result['parameter_slices'], strict=True):
            values = rate*np.asarray(velocity[:8, lo:hi])
            centered = (values-values.mean(0))/np.sqrt(7)
            gram = centered@centered.T
            key = 'shared_metric_network' if 'reslayerAs' in name else 'plga_affine'
            grams[key] += gram;grams['all_generator'] += gram
        with np.load(path/'measurements.npz') as z:
            base = z['base_logits'];probability = np.exp(base-logsumexp(base, axis=-1, keepdims=True))
            derivative = np.stack([z[f'b{k}_jvp_logits'] for k in range(8)])
            derivative -= np.sum(derivative*probability, axis=-1, keepdims=True)
            fisher = (derivative*np.sqrt(probability/len(base))).reshape(8,-1)
            centered = (fisher-fisher.mean(0))/np.sqrt(7)
            grams['predictive_fisher_tangent'] = centered@centered.T
        # Generalized gains of the emission on this exact sampled noise span.
        # This is a local source response, not a full parameter-Jacobian spectrum.
        g, h = grams['all_generator'], grams['predictive_fisher_tangent']
        eig, basis = np.linalg.eigh((g+g.T)/2)
        active = eig > max(float(eig.max()), 1e-300)*1e-10
        directions = basis[:, active]
        inverse_root = directions/np.sqrt(eig[active])
        transport = inverse_root.T@h@inverse_root
        gains = np.linalg.eigvalsh((transport+transport.T)/2)[::-1]
        projected = directions@directions.T@h@directions@directions.T
        support_error = float(np.linalg.norm(h-projected)/max(np.linalg.norm(h), 1e-300))
        if support_error > 1e-8 or gains.min() < -1e-9*max(float(gains.max()), 1e-300):
            raise AssertionError('Predictive tangent is inconsistent with the sampled source span')
        record = dict(case=meta['case'], samples=8, sample_rank_bound=7, generator_learning_rate=rate,
            sectors={key:summarize(value) for key, value in grams.items()},
            sampled_noise_span_dimension=int(active.sum()), local_fisher_squared_gains=np.maximum(gains,0).tolist(),
            source_span_support_relative_error=support_error)
        records.append(record)
        for key, gram in grams.items():raw[path.name+'_'+key+'_gram'] = gram
        print(path.name, 'matched source/emission geometry complete', flush=True)
    result = dict(schema='matched-native-noise-emission-geometry-v1', status='complete', states=records,
        selection=selection, parameter_units='Nominal native generator displacement, including the declared learning rate, in fixed Euclidean parameter coordinates.',
        predictive_units='The full-graph tangent in the full-vocabulary Fisher inner product, averaged over the same 16 fixed contexts for exactly the same eight batches.',
        interpretation='Participation ranks are finite sample statistics bounded by seven. A linear emission can either increase or decrease participation rank; the measured change is not a general monotonicity theorem. Generalized gains concern only the sampled noise span and the local emission, with the separately checked numerical amplitude windows. Body parameters are held fixed in these generator-only emission directions.')
    write_json(out/'results.json', result);np.savez_compressed(out/'measurements.npz', **raw)
    write_json(out/'manifest.json', dict(status='complete', binding_sha256=sha256(out/'binding.json'),
        results_sha256=sha256(out/'results.json'), raw_sha256=sha256(out/'measurements.npz'), states=len(records)))


if __name__ == '__main__':main()
