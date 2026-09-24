#!/usr/bin/env python
"""Independently reconstruct every retained token in the internal early-risk panel."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model_rg.provenance import sha256, write_json


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True)
    parser.add_argument('--study', default='scheduled-training-20260908'); args = parser.parse_args()
    root = Path(args.root).resolve(); study = root/args.study
    folder = study/'development/early-training-risk'; output = folder/'verification.json'
    if output.exists(): raise FileExistsError(output)
    checked = {}

    def check(path, digest=None):
        path = Path(path).resolve()
        if str(path) not in checked: checked[str(path)] = sha256(path)
        if digest is not None and checked[str(path)] != digest: raise AssertionError('Changed internal risk evidence: '+str(path))

    selection = study/'protocols/early-training-risk-diagnostic.json'; check(selection)
    spec = json.loads(selection.read_text()); torch.set_num_threads(spec['threads'])
    for name, digest in spec['inputs_sha256'].items(): check(name, digest)
    repo = Path(__file__).resolve().parents[1]
    for name, digest in spec['producer_sources'].items(): check(repo/name, digest)
    with np.load(spec['training_cohort']) as stored: training = stored['crops'][spec['training_indices']]
    probe = root/'controlled-study-20260905/data/short'
    tokens = np.load(probe/'tokens.npy', mmap_mode='r'); offsets = np.load(probe/'offsets.npy')
    rows = np.asarray(spec['heldout_rows']); held = tokens[rows[:, None], offsets[rows, None]+np.arange(65)]
    records = []; elements = 0; values = 0; maximum64 = 0.
    for case in spec['cases']:
        parent = folder/case['name']; check(parent/'manifest.json')
        meta = json.loads((parent/'manifest.json').read_text())
        if meta['status'] != 'complete' or meta['case'] != case: raise AssertionError('Missing selected internal risk state')
        check(case['state'], case['state_sha256'])
        for name, key in [('binding.json', 'binding_sha256'), ('results.json', 'results_sha256'), ('measurements.npz', 'raw_sha256')]:
            check(parent/name, meta[key])
        binding = json.loads((parent/'binding.json').read_text())
        for name, digest in binding['inputs'].items(): check(name, digest)
        for name, digest in binding['source_files'].items(): check(parent/'source'/name, digest)
        report = json.loads((parent/'results.json').read_text()); observed = {}
        with np.load(parent/'measurements.npz') as raw:
            for label, crops in [('training', training), ('heldout', held)]:
                np.testing.assert_array_equal(raw[label+'_crops'], crops)
                mask = crops[:, 1:] != 0; np.testing.assert_array_equal(raw[label+'_mask'], mask)
                for start in range(0, len(crops), spec['batch_size']):
                    filename = f'{label}-logits-{start:03d}.npy'; check(parent/filename, meta['logit_sidecars'][filename])
                    logits = np.load(parent/filename)
                    if logits.shape != (32, 64, 32000) or logits.dtype != np.float32 or not np.isfinite(logits).all():
                        raise AssertionError('Incomplete vocabulary-logit record')
                    target = crops[start:start+32, 1:]
                    tensor = torch.from_numpy(logits); y = torch.tensor(target, dtype=torch.long)
                    lp = torch.log_softmax(tensor, dim=-1)
                    nll = torch.nn.functional.cross_entropy(tensor.transpose(1, 2), y, reduction='none').numpy()
                    last = -torch.gather(lp[:, -1], 1, y[:, -1:]).squeeze(1).numpy()
                    entropy = -(lp.exp()*lp).sum(-1).numpy()
                    for key, actual in [('all_nll', nll), ('last_nll', last), ('entropy', entropy)]:
                        expected = raw[label+'_'+key][start:start+32]
                        if expected.shape != actual.shape or expected.dtype != actual.dtype or expected.tobytes() != actual.tobytes():
                            raise AssertionError('A native risk reduction did not replay bytewise')
                        values += actual.size
                    # Independent real-formula reduction, reported rather than substituted.
                    z = logits.astype(float); z -= z.max(-1, keepdims=True)
                    logp = z-np.log(np.exp(z).sum(-1, keepdims=True))
                    mathematical = -np.take_along_axis(logp, target[..., None], -1).squeeze(-1)
                    maximum64 = max(maximum64, float(np.max(np.abs(mathematical-nll))))
                    elements += logits.size
                    del logits, tensor, lp, z, logp, mathematical
                observed[label] = dict(contexts=len(crops), valid_targets=int(mask.sum()),
                    last_nll=float(np.mean(raw[label+'_last_nll'].astype(float))),
                    all_token_nll=float(np.sum(raw[label+'_all_nll'].astype(float)*mask)/mask.sum()),
                    all_token_entropy=float(np.sum(raw[label+'_entropy'].astype(float)*mask)/mask.sum()))
                if report['cohorts'][label] != observed[label]: raise AssertionError('An internal risk mean changed')
        difference = {key: observed['heldout'][key]-observed['training'][key]
            for key in ['last_nll', 'all_token_nll', 'all_token_entropy']}
        if difference != report['heldout_minus_training']: raise AssertionError('An internal risk gap changed')
        records.append(dict(case=case, cohorts=observed, heldout_minus_training=difference))
        print('Independently reconstructed', case['name'], flush=True)
    if len(records) != 6 or elements != 3145728000 or values != 198144:
        raise AssertionError('The selected six-state risk panel is incomplete')
    write_json(output, dict(status='passed', states=6, records=records, retained_logit_elements=elements,
        native_values_replayed_bytewise=values, maximum_float64_all_target_reduction_difference=maximum64,
        additional_independent_training_identities=0, verifier_sha256=sha256(__file__), checked_sha256=checked,
        scope='Every native float32 per-token loss and entropy in this internal diagnostic is reconstructed from '
        'retained full vocabulary logits. Float64 all-target reduction differences are separate observations. '
        'No source trajectory is marked complete by this checkpoint-only diagnostic.'))
    print('Complete internal risk panel verified', flush=True)


if __name__ == '__main__': main()
