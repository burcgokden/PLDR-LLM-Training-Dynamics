#!/usr/bin/env python
"""Reconstruct native row-map derivatives from explicit affine/SwiGLU/LN formulas."""
import argparse
import json
from numerical_claims import finite_greater
from numerical_validation import load_json_strict
from pathlib import Path

import numpy as np
from safetensors import safe_open
import torch

from model_rg.provenance import sha256, write_json


def explicit_stack(x, params, epsilons):
    """Column-vector Jacobian, independent of native forward/autograd code."""
    dimension = len(x); identity = np.eye(dimension)
    projection = identity-np.ones((dimension, dimension))/dimension
    total = identity.copy(); unit_norms = []
    for unit, epsilon in enumerate(epsilons):
        residual = x.copy(); derivative = identity.copy()
        dense_ids = sorted({int(k.split('.')[2]) for k in params if k.startswith(str(unit)+'.denseAs.')})
        if dense_ids != [0,1]: raise AssertionError('The selected native row unit must contain two SwiGLU maps')
        for dense in dense_ids:
            prefix = f'{unit}.denseAs.{dense}.'
            w1, w2, w3 = [params[prefix+f'gluw{j}.weight'] for j in [1,2,3]]
            b1, b2, b3 = [params[prefix+f'gluw{j}.bias'] for j in [1,2,3]]
            first = w1@x+b1; second = w2@x+b2
            sigmoid = np.empty_like(first); positive = first >= 0
            sigmoid[positive] = 1/(1+np.exp(-first[positive]))
            exponent = np.exp(first[~positive]); sigmoid[~positive] = exponent/(1+exponent)
            silu = first*sigmoid
            slope = sigmoid+first*sigmoid*(1-sigmoid)
            local = w3@((second*slope)[:,None]*w1+silu[:,None]*w2)
            x = w3@(silu*second)+b3; derivative = local@derivative
        centered = x+residual; centered -= centered.mean()
        variance = float(np.mean(centered*centered)); radius = np.sqrt(variance+epsilon)
        gamma = params[f'{unit}.layernormA.weight']; beta = params[f'{unit}.layernormA.bias']
        normalization = (gamma/radius)[:,None]*(projection-np.outer(centered,centered)/(dimension*(variance+epsilon)))
        local = normalization@(derivative+identity)
        x = gamma*centered/radius+beta; total = local@total
        unit_norms.append(float(np.linalg.svd(local, compute_uv=False)[0]))
    return x, total, np.array(unit_norms)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True)
    p.add_argument('--study', default='scheduled-training-feasible-20260908'); a = p.parse_args()
    study = Path(a.root).resolve()/a.study; output = study/'verification/onepass-row-jacobians.json'
    if output.exists(): raise FileExistsError(output)
    checked = {}; cache = {}; torch.set_num_threads(4)

    def check(path, expected=None):
        path = Path(path).resolve(); stat = path.stat(); key = (str(path),stat.st_size,stat.st_mtime_ns)
        if key not in cache: cache[key] = sha256(path)
        digest = cache[key]
        if expected is not None and expected != digest: raise AssertionError('Changed Jacobian evidence: '+str(path))
        checked[str(path)] = digest; return digest

    def load(path):
        check(path); return load_json_strict(Path(path).read_text())

    selection = study/'protocols/onepass-row-jacobian-selection.json'; spec = load(selection)
    for path, digest in spec['inputs_sha256'].items(): check(path, digest)
    points = np.array([(c,h,r) for c in [0,3] for h in [0,6,13] for r in [0,31,63]])
    if spec['selected_jacobians'] != 1440: raise AssertionError('The full derivative selection is required')
    ledger = load(study/'launcher-onepass-row-jacobians.json')
    if ledger['status'] != 'complete' or ledger['selection_sha256'] != check(selection):
        raise AssertionError('The full native derivative execution must be complete')
    records = []; summaries = []; count = 0; maximum_relative_error = 0.; maximum_absolute_error = 0.
    for case in spec['cases']:
        folder = study/'measurements'/case['name']; meta = load(folder/'manifest.json')
        if meta['status'] != 'complete' or meta['case'] != case: raise AssertionError('A selected row state is incomplete')
        for name, key in [('binding.json','binding_sha256'),('results.json','results_sha256'),('measurements.npz','raw_sha256')]:
            check(folder/name, meta[key])
        binding = load(folder/'binding.json')
        for path, digest in binding['inputs'].items(): check(path, digest)
        for path, digest in binding['source_files'].items(): check(folder/'source'/path, digest)
        for path, digest in spec['producer_sources'].items(): check(folder/'source'/path, digest)
        for path, digest in case['input_sha256'].items(): check(path, digest)
        result = load(folder/'results.json')
        with np.load(folder/'measurements.npz') as z: raw = {k:z[k] for k in z.files}
        np.testing.assert_array_equal(raw['points'], points)
        with np.load(study/'measurements'/case['reference_transport']/'measurements.npz') as z: inputs = z[f'L{case["length"]}_stages'][:,:,0]
        if case['kind'] == 'pretrained':
            with safe_open(str(Path(case['source'])/'model.safetensors'), framework='pt', device='cpu') as f:
                parameters = {k:f.get_tensor(k) for k in f.keys() if '.reslayerAs.' in k}
        else:
            saved = torch.load(case['state'], map_location='cpu', mmap=True, weights_only=True)
            parameters = {k:v for k,v in saved['model'].items() if '.reslayerAs.' in k}
        layer_summaries = []
        for layer in range(5):
            prefix = f'decoder.dec_layers.{layer}.mha1.reslayerAs.'
            params = {k.removeprefix(prefix):v.numpy().astype(float) for k,v in parameters.items() if k.startswith(prefix)}
            if not params: raise AssertionError('Native row parameters are missing')
            for key, value in params.items():
                if raw[f'l{layer}_parameter_{key}'].dtype != np.float64: raise AssertionError('Local arithmetic precision changed')
                np.testing.assert_array_equal(raw[f'l{layer}_parameter_{key}'], value)
            epsilon = raw[f'l{layer}_epsilons']; np.testing.assert_array_equal(epsilon, np.full(8,1e-6))
            norms = []
            for index, (context, head, row) in enumerate(points):
                x = inputs[context,layer,head,row].astype(float)
                np.testing.assert_array_equal(x, raw[f'l{layer}_inputs'][index])
                emitted, derivative, local_norms = explicit_stack(x, params, epsilon)
                np.testing.assert_allclose(emitted, raw[f'l{layer}_outputs'][index], rtol=3e-10, atol=2e-12)
                observed = raw[f'l{layer}_jacobians'][index]
                error = float(np.linalg.norm(observed-derivative)); scale = float(np.linalg.norm(derivative))
                if finite_greater(error, 3e-7*scale+1e-24, 'scripts/verify_onepass_row_jacobians.py:107'): raise AssertionError('The explicit native derivative differs from autograd')
                maximum_absolute_error = max(maximum_absolute_error, error)
                if scale > 0: maximum_relative_error = max(maximum_relative_error, error/scale)
                singular = np.linalg.svd(derivative, compute_uv=False)
                np.testing.assert_allclose(raw[f'l{layer}_singular_values'][index], singular, rtol=3e-7, atol=3e-7*scale+1e-24)
                np.testing.assert_allclose(raw[f'l{layer}_unit_operator_norms'][index], local_norms, rtol=3e-9, atol=2e-12)
                if singular[0] > np.prod(local_norms)*(1+3e-9)+1e-24: raise AssertionError('Jacobian product norm bound failed')
                norms.append(singular[0]); count += 1
            norms = np.array(norms); report = result['layers'][layer]
            for key, value in [('minimum_operator_norm', norms.min()), ('maximum_operator_norm', norms.max()), ('mean_operator_norm', norms.mean())]:
                np.testing.assert_allclose(report[key], value, rtol=3e-7, atol=1e-24)
            if report['points_below_one'] != int(np.sum(norms < 1)): raise AssertionError('Local contraction classification changed')
            layer_summaries.append(dict(layer=layer, minimum_operator_norm=float(norms.min()),
                maximum_operator_norm=float(norms.max()), points_below_one=int(np.sum(norms<1)), points=18))
        summaries.append(dict(name=case['name'], layers=layer_summaries))
        records.append(dict(name=case['name'], manifest_sha256=check(folder/'manifest.json')))
        del parameters, raw, inputs
        if case['kind'] != 'pretrained': del saved
        print(case['name'], 'explicit native derivatives verified', flush=True)
    if ledger['records'] != records or count != 1440: raise AssertionError('A selected derivative outcome was omitted')
    write_json(output, dict(status='passed', states=16, jacobians=count, summaries=summaries,
        maximum_frobenius_error=maximum_absolute_error, maximum_relative_frobenius_error=maximum_relative_error,
        checked_sha256=checked, verifier_sha256=sha256(__file__),
        scope='Independent NumPy differentiation from the affine, SiLU, multiplication, residual and layer-normalization formulas. Every row parameter matches the selected checkpoint, and every differentiation input matches a verified native normalized Gram. All 1440 complete Jacobians and their singular values are reconstructed. This verifier does not replay native float32 row stages; that separate producer replay is retained. These are local derivatives, not uniform contraction constants or training-response eigenvalues.'))
    print('Passed',count,'explicit native row Jacobians',flush=True)


if __name__ == '__main__': main()
