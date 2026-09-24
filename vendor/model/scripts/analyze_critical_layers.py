#!/usr/bin/env python3
"""Retain all decoder covariance modes of the complete conditional native grid."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256, write_json


def analyze(study, analysis_path, output):
    if output.exists():
        raise FileExistsError(output)
    p = json.loads((study/'protocol.json').read_text())
    a = json.loads(analysis_path.read_text())
    if a['status'] != 'complete' or a['protocol_sha256'] != sha256(study/'protocol.json'):
        raise ValueError('A complete bound scalar analysis is required')
    index = {(r['environment'], r['heads'], r['control'], r['step'], r['field']): r for r in a['cells']}
    groups = defaultdict(list)
    checked = {str(analysis_path): sha256(analysis_path), str(study/'protocol.json'): sha256(study/'protocol.json')}
    for job in p['jobs']:
        root = study/'runs'/job['run_id']
        m = json.loads((root/'manifest.json').read_text())
        if m['status'] != 'complete':
            continue
        if m['job'] != job or m['protocol_sha256'] != sha256(study/'protocol.json'):
            raise ValueError('Unbound decoder observation')
        if sha256(root/'observations.npz') != m['artifacts']['observations.npz']:
            raise ValueError('Changed native decoder fields')
        checked[str(root/'manifest.json')] = sha256(root/'manifest.json')
        checked[str(root/'observations.npz')] = m['artifacts']['observations.npz']
        groups[(job['environment'], job['heads'], job['control'])].append((job['seed'], root/'observations.npz'))
    cells = []
    max_trace_error = max_pair_error = 0.
    for (e, n, g), rows in sorted(groups.items()):
        rows.sort()
        if [s for s, _ in rows] != sorted(p['design']['seeds']):
            continue
        arrays = [np.load(path) for _, path in rows]
        try:
            data = np.stack([r['heads'] for r in arrays])
            s = len(rows)
            for k, t in enumerate(arrays[0]['steps']):
                for field, coordinate in [('row', 0), ('attention', 1), ('operator', 2), ('common_amplitude', None)]:
                    if coordinate is None:
                        energy = data[:, k, ..., 3]
                        x = np.sqrt(np.maximum(energy-data[:, k, ..., 0]*np.maximum(energy, 1e-30), 0.))
                    else:
                        x = data[:, k, ..., coordinate]
                    q = x.mean(-1)
                    centered = q-q.mean(0, keepdims=True)
                    covariance = n*np.einsum('scl,scm->lm', centered, centered)/((s-1)*64)
                    pair = np.zeros_like(covariance)
                    for i in range(s):
                        for j in range(i):
                            difference = q[i]-q[j]
                            pair += difference.T@difference
                    pair *= n/(s*(s-1)*64)
                    error = float(np.max(np.abs(pair-covariance)/(1+np.abs(covariance))))
                    max_pair_error = max(max_pair_error, error)
                    if error > 2e-12:
                        raise ValueError('Independent pairwise matrix reconstruction differs')
                    reference = covariance/n
                    diagonal = np.mean((x-x.mean(0, keepdims=True))**2, axis=(0, 1, 3))*s/(s-1)
                    np.fill_diagonal(reference, diagonal)
                    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                    if eigenvalues[0] < -2e-12*(1+eigenvalues[-1]):
                        raise ValueError('Decoder covariance lost positivity')
                    trace = float(np.trace(covariance)/5)
                    expected = index[(e, n, g, int(t), field)]['susceptibility']
                    error = abs(trace-expected)/(1+abs(expected))
                    max_trace_error = max(max_trace_error, error)
                    if error > 2e-12:
                        raise ValueError('Decoder trace differs from primary statistic')
                    if np.any(np.diag(covariance) > n*diagonal+2e-12*(1+n*diagonal)):
                        raise ValueError('Decoder covariance violates its head bound')
                    cells.append(dict(environment=e, heads=n, control=g, step=int(t), field=field,
                                      layer_means=x.mean((0, 1, 3)).tolist(),
                                      susceptibility_matrix=covariance.tolist(),
                                      symmetrized_one_head_reference=reference.tolist(),
                                      eigenvalues_ascending=eigenvalues.tolist(),
                                      eigenvectors_columns=eigenvectors.tolist(),
                                      trace_per_layer=trace,
                                      largest_eigenvalue_fraction=float(eigenvalues[-1]/np.trace(covariance)) if np.trace(covariance) > 0 else None))
        finally:
            for r in arrays:
                r.close()
    result = dict(schema='critical-decoder-covariance-v1', status='complete', study=str(study), cells=cells,
                  maximum_normalized_trace_error=max_trace_error, maximum_normalized_pair_error=max_pair_error,
                  checked_sha256=checked, analyzer_sha256=sha256(__file__),
                  reference_definition='Uniform independent head-label permutations in each decoder; off-diagonal one-head covariances equal the covariances of the decoder head means.',
                  scope='All five decoders and their conditional cross covariances. Contexts retain fixed weights; eigenvectors estimated from the finite seed sample are descriptive.')
    write_json(output, result)
    print(json.dumps({'status': 'complete', 'decoder_covariance_cells': len(cells), 'maximum_pair_error': max_pair_error}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    analyze(a.study.resolve(), a.analysis.resolve(), a.output.resolve())


if __name__ == '__main__':
    main()
