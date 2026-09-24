#!/usr/bin/env python
"""Measure native shared-row Jacobians across the selected single-pass prefix domain."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from model_rg.controlled import bind_run
from model_rg.native import NativeModel
from model_rg.provenance import sha256, write_json
from model_rg.training import TrainingModel


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True)
    p.add_argument('--study', default='scheduled-training-feasible-20260908'); a = p.parse_args()
    study = Path(a.root).resolve()/a.study; repo = Path(__file__).resolve().parents[1]
    selection = study/'protocols/onepass-row-jacobian-selection.json'; spec = json.loads(selection.read_text())
    for name, digest in spec['producer_sources'].items():
        if sha256(repo/name) != digest:
            raise AssertionError('A selected row-Jacobian producer changed')
    for path, digest in spec['inputs_sha256'].items():
        if sha256(path) != digest:
            raise AssertionError('A selected row-Jacobian input changed')
    ledger = study/'launcher-onepass-row-jacobians.json'
    if ledger.exists(): raise FileExistsError(ledger)
    records = []; write_json(ledger, dict(status='running', selection_sha256=sha256(selection), records=records))
    torch.set_num_threads(4)
    points = np.array([(c,h,r) for c in spec['context_indices'] for h in spec['head_indices'] for r in spec['row_indices']])
    for case in spec['cases']:
        reference = study/'measurements'/case['reference_transport']
        for path, digest in case['input_sha256'].items():
            if sha256(path) != digest: raise AssertionError('A selected checkpoint changed')
        out = study/'measurements'/case['name']; out.mkdir(parents=True, exist_ok=False)
        inputs = [selection, reference/'manifest.json', reference/'measurements.npz', *map(Path, case['input_sha256'])]
        bind_run(out, inputs, vars(a)); before = time.time()
        if case['kind'] == 'pretrained': model = NativeModel(case['source'], 'cpu')
        else:
            saved = torch.load(case['state'], map_location='cpu', mmap=True, weights_only=True)
            if saved['step'] != case['step']: raise AssertionError('The selected row-Jacobian horizon changed')
            model = TrainingModel(case['source'], saved['arguments']['heads'], saved['arguments']['seed'], 'cpu')
            model.model.load_state_dict(saved['model']); del saved
        model.model.eval().requires_grad_(False)
        with np.load(reference/'measurements.npz') as z: stages = z[f'L{case["length"]}_stages']
        raw = dict(points=points); replay_elements = 0; layers = []
        for layer, decoder in enumerate(model.model.decoder.dec_layers):
            units = decoder.mha1.reslayerAs
            current = torch.tensor(stages[:, layer, 0].copy())
            with torch.no_grad():
                for index, unit in enumerate(units, 1):
                    current = unit([current])
                    want = stages[:, layer, index]
                    got = current.numpy()
                    if got.dtype != want.dtype or got.shape != want.shape or got.tobytes() != want.tobytes():
                        raise AssertionError('Native residual-stack replay differs bytewise')
                    replay_elements += got.size
            double_units = copy.deepcopy(units).double().requires_grad_(False)
            for name, value in double_units.state_dict().items():
                raw[f'l{layer}_parameter_{name}'] = value.numpy().copy()
            epsilons = np.array([unit.layernormA.eps for unit in double_units]); raw[f'l{layer}_epsilons'] = epsilons

            def forward(value):
                for unit in double_units: value = unit([value])
                return value

            vectors = []; outputs = []; jacobians = []; singular = []; chains = []
            for context, head, row in points:
                value = torch.tensor(stages[context, layer, 0, head, row].copy(), dtype=torch.float64, requires_grad=True)
                jacobian = torch.autograd.functional.jacobian(forward, value, vectorize=True)
                output = forward(value).detach()
                chain = torch.eye(64, dtype=torch.float64); current = value.detach()
                local_norms = []
                for unit in double_units:
                    local = torch.autograd.functional.jacobian(lambda x:unit([x]), current, vectorize=True)
                    local_norms.append(float(torch.linalg.matrix_norm(local, ord=2)))
                    chain = local @ chain; current = unit([current]).detach()
                np.testing.assert_allclose(jacobian.numpy(), chain.numpy(), rtol=2e-10, atol=2e-13)
                vectors.append(value.detach().numpy()); outputs.append(output.numpy()); jacobians.append(jacobian.numpy())
                singular.append(torch.linalg.svdvals(jacobian).numpy()); chains.append(local_norms)
            for name, values in [('inputs', vectors), ('outputs', outputs), ('jacobians', jacobians),
                                  ('singular_values', singular), ('unit_operator_norms', chains)]:
                raw[f'l{layer}_{name}'] = np.asarray(values)
            top = raw[f'l{layer}_singular_values'][:,0]
            layers.append(dict(layer=layer, points=len(points), minimum_operator_norm=float(top.min()),
                maximum_operator_norm=float(top.max()), mean_operator_norm=float(top.mean()),
                points_below_one=int(np.sum(top < 1)),
                maximum_product_unit_norms=float(np.prod(raw[f'l{layer}_unit_operator_norms'], axis=1).max())))
            print(case['name'], 'layer', layer, 'Jacobian norms', float(top.min()), float(top.max()), flush=True)
            del double_units
        np.savez_compressed(out/'measurements.npz', **raw)
        write_json(out/'results.json', dict(status='complete', case=case, layers=layers,
            native_stage_replay_byte_equal=True, native_stage_replay_elements=replay_elements,
            jacobians=5*len(points), arithmetic='Native float32 checkpoint weights and normalized-Gram inputs, lifted exactly to float64 for local differentiation; native float32 full-stack stage replay is checked separately.',
            scope='Local derivatives of the shared eight-unit residual row map at eighteen declared points per decoder. A spectral norm below one is local infinitesimal contraction at that point, not a uniform domain bound, a temporal training response eigenvalue or a critical exponent.'))
        write_json(out/'manifest.json', dict(status='complete', case=case, binding_sha256=sha256(out/'binding.json'),
            results_sha256=sha256(out/'results.json'), raw_sha256=sha256(out/'measurements.npz'), seconds=time.time()-before))
        records.append(dict(name=case['name'], manifest_sha256=sha256(out/'manifest.json')))
        write_json(ledger, dict(status='running', selection_sha256=sha256(selection), records=records))
        del model, raw, stages
    write_json(ledger, dict(status='complete', selection_sha256=sha256(selection), records=records))


if __name__ == '__main__': main()
