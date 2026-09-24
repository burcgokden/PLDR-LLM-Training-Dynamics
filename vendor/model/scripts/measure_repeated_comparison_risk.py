#!/usr/bin/env python
"""Fixed-state risk on the four retained repeated-data comparison checkpoints.

This does not complete a native parent, replace the selected full-cohort risk
program, or add training identities. The selection is written before scoring.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json
from model_rg.training import TrainingModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--study', default='scheduled-training-feasible-20260908')
    parser.add_argument('--case', required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve(); study = root/args.study
    repo = Path(__file__).resolve().parents[1]
    selection = study/'protocols/repeated-risk-comparison.json'
    if not args.case: raise ValueError('--case is required for a selected measurement')
    spec = json.loads(selection.read_text())
    for path, digest in spec['inputs_sha256'].items():
        if sha256(path) != digest: raise AssertionError('An internal diagnostic input changed')
    for name, digest in spec['producer_sources'].items():
        if sha256(repo/name) != digest: raise AssertionError('A frozen diagnostic source changed')
    selected = [c for c in spec['cases'] if c['name'] == args.case]
    if len(selected) != 1: raise AssertionError('One selected sealed checkpoint is required')
    case = selected[0]
    if sha256(case['state']) != case['state_sha256']: raise AssertionError('The sealed model state changed')
    out = study/'measurements/repeated-risk'/case['name']; out.mkdir(parents=True, exist_ok=False)
    bind_run(out, [selection, Path(case['state']), *map(Path, spec['inputs_sha256'])], vars(args))
    torch.set_num_threads(spec['threads']); before = time.time()
    saved = torch.load(case['state'], map_location='cpu', mmap=True, weights_only=True)
    if saved['step'] != case['step']: raise AssertionError('The observed horizon changed')
    for name in ['heads', 'seed', 'recipe', 'shared_seed', 'stream_seed']:
        if saved['arguments'][name] != case[name]: raise AssertionError('The checkpoint condition changed')
    model = TrainingModel(spec['source'], case['heads'], case['seed'], 'cpu')
    model.model.load_state_dict(saved['model']); del saved
    model.model.eval().requires_grad_(False)
    with np.load(spec['training_cohort']) as stored: training = stored['crops'][spec['training_indices']].copy()
    probe = root/'controlled-study-20260905/data/short'
    tokens = np.load(probe/'tokens.npy', mmap_mode='r'); offsets = np.load(probe/'offsets.npy')
    rows = np.asarray(spec['heldout_rows']); heldout = tokens[rows[:, None], offsets[rows, None]+np.arange(65)]
    raw = {}; sidecars = {}; results = {}
    with torch.no_grad():
        for label, crops in [('training', training), ('heldout', heldout)]:
            all_nll = []; last_nll = []; entropies = []
            for start in range(0, len(crops), spec['batch_size']):
                batch = torch.tensor(crops[start:start+spec['batch_size']], dtype=torch.long)
                output = model.model(batch[:, :64], use_cache=False, logits_to_keep=0).logits
                lp = output.log_softmax(-1)
                nll = torch.nn.functional.cross_entropy(output.transpose(1, 2), batch[:, 1:], reduction='none')
                last = -lp[:, -1].gather(1, batch[:, 64, None]).squeeze(1)
                entropy = -(lp.exp()*lp).sum(-1)
                if not all(torch.isfinite(x).all() for x in [output, nll, last, entropy]):
                    raise FloatingPointError('Nonfinite internal risk observation')
                filename = f'{label}-logits-{start:03d}.npy'
                np.save(out/filename, output.numpy())
                sidecars[filename] = sha256(out/filename)
                all_nll.append(nll.numpy()); last_nll.append(last.numpy()); entropies.append(entropy.numpy())
            mask = crops[:, 1:] != 0
            raw[label+'_crops'] = crops
            raw[label+'_all_nll'] = np.concatenate(all_nll)
            raw[label+'_last_nll'] = np.concatenate(last_nll)
            raw[label+'_entropy'] = np.concatenate(entropies)
            raw[label+'_mask'] = mask
            results[label] = dict(contexts=len(crops), valid_targets=int(mask.sum()),
                last_nll=float(raw[label+'_last_nll'].astype(float).mean()),
                all_token_nll=float(np.sum(raw[label+'_all_nll'].astype(float)*mask)/mask.sum()),
                all_token_entropy=float(np.sum(raw[label+'_entropy'].astype(float)*mask)/mask.sum()))
            print(case['name'], label, results[label], flush=True)
    write_json(out/'results.json', dict(status='complete', schema='retained-repeated-risk-v1', case=case, cohorts=results,
        heldout_minus_training={field: results['heldout'][field]-results['training'][field]
            for field in ['last_nll', 'all_token_nll', 'all_token_entropy']}, scope=spec['interpretation']))
    np.savez_compressed(out/'measurements.npz', **raw)
    write_json(out/'manifest.json', dict(status='complete', case=case, binding_sha256=sha256(out/'binding.json'),
        results_sha256=sha256(out/'results.json'), raw_sha256=sha256(out/'measurements.npz'), logit_sidecars=sidecars,
        seconds=time.time()-before, additional_independent_training_identities=0))


if __name__ == '__main__': main()
