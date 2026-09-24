#!/usr/bin/env python
"""Qualify native fixed-G and row-projection interventions on frozen states."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model_rg.controlled import bind_run
from model_rg.inference_interventions import fixed_operators, operator_cache, projected_rows
from model_rg.native import NativeModel
from model_rg.provenance import sha256, write_json
from model_rg.training import TrainingModel


def same(a, b):
    x = a.detach().numpy(); y = b.detach().numpy()
    return x.shape == y.shape and x.dtype == y.dtype and x.tobytes() == y.tobytes()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True)
    p.add_argument('--study', default='scheduled-training-20260908'); a = p.parse_args()
    root = Path(a.root).resolve(); study = root/a.study
    selection = study/'protocols/prefix-mechanism-initial-selection.json'
    spec = json.loads(selection.read_text()); out = study/'qualification/inference-interventions'
    out.mkdir(parents=True, exist_ok=False)
    probe = root/'controlled-study-20260905/data/short'
    inputs = [selection, probe/'tokens.npy', probe/'offsets.npy']
    for case in spec['cases']:
        inputs.extend(Path(name) for name in case['input_sha256'])
    bind_run(out, sorted(set(inputs)), vars(a)); torch.set_num_threads(4)
    tokens = np.load(probe/'tokens.npy', mmap_mode='r'); offsets = np.load(probe/'offsets.npy')
    rows = np.array(spec['rows'][:2]); crops = tokens[rows[:, None], offsets[rows, None]+np.arange(64)]
    ids = torch.tensor(crops, dtype=torch.long); altered = ids.clone(); altered[:, 32:] = ids.flip(0)[:, 32:]
    raw = dict(inputs=crops, altered=altered.numpy()); records = []
    for case in spec['cases']:
        if case['kind'] == 'pretrained':
            model = NativeModel(case['source'], 'cpu')
        else:
            saved = torch.load(case['state'], map_location='cpu', mmap=True, weights_only=True)
            model = TrainingModel(case['source'], saved['arguments']['heads'], saved['arguments']['seed'], 'cpu')
            model.model.load_state_dict(saved['model']); del saved
        model.model.eval().requires_grad_(False)

        @torch.no_grad()
        def forward(x):
            model.eta = None; model.capture = False
            return model.model(x, use_cache=False, logits_to_keep=0, output_pldr_attentions=True)

        baseline = forward(ids); cache = operator_cache(baseline)
        with fixed_operators(model, cache):
            replay = forward(ids)
            suffix = forward(altered)
        restored = forward(ids)
        with projected_rows(model):
            projected = forward(ids)
        after_projection = forward(ids)
        comparisons = dict(native_external_G_replay=same(baseline.logits, replay.logits),
            fixed_G_suffix_prefix=same(replay.logits[:, :32], suffix.logits[:, :32]),
            restored_after_fixed_G=same(baseline.logits, restored.logits),
            restored_after_projection=same(baseline.logits, after_projection.logits))
        if not all(comparisons.values()): raise AssertionError((case['name'], comparisons))
        for layer in projected.pldr_attentions:
            if not same(layer[0], layer[0][:, :, :1].expand_as(layer[0])):
                raise AssertionError('The native row projection did not produce identical rows')
        for label, output in [('native', baseline), ('replay', replay), ('suffix', suffix),
                              ('restored', restored), ('projected', projected), ('restored2', after_projection)]:
            raw[case['name']+'_'+label+'_logits'] = output.logits.detach().numpy().copy()
        raw[case['name']+'_cache'] = cache.numpy()
        raw[case['name']+'_projected_A'] = np.stack([v[0].numpy() for v in projected.pldr_attentions])
        records.append(dict(name=case['name'],checks=comparisons,
            checked_logit_elements=baseline.logits.numel(), projected_rows_identical=True))
        print(case['name'],comparisons,flush=True)
        del model,baseline,replay,suffix,restored,projected,after_projection,cache
    np.savez_compressed(out/'measurements.npz',**raw)
    write_json(out/'results.json',dict(status='complete',records=records,
        scope='Qualification of a reversible implementation intervention on five previously selected frozen states and two contexts. Exact native replay, suffix-prefix equality with fixed operators, and restoration are byte comparisons. This is not a scientific reasoning outcome.'))
    write_json(out/'manifest.json',dict(status='complete',binding_sha256=sha256(out/'binding.json'),
        raw_sha256=sha256(out/'measurements.npz'),results_sha256=sha256(out/'results.json')))


if __name__ == '__main__': main()
