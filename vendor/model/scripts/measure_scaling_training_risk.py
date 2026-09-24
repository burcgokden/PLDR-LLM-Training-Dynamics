#!/usr/bin/env python
"""Observe fresh training-law risk at one immutable selected native state."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json
from model_rg.training import TrainingModel


def main():
    p = argparse.ArgumentParser();p.add_argument('--root', required=True)
    p.add_argument('--study', default='critical-scaling-20260906');p.add_argument('--case', required=True)
    a = p.parse_args();root = Path(a.root);study = root/a.study
    protocol = study/'protocols/training-risk-selection.json';spec = json.loads(protocol.read_text())
    cases = [c for c in spec['cases'] if c['name'] == a.case]
    if len(cases) != 1:raise ValueError('A uniquely selected risk state is required')
    case = cases[0];state = Path(case['state']);parent = Path(case['parent_manifest'])
    reference = Path(case['reference_observation']);meta = json.loads(parent.read_text())
    ref = json.loads(reference.read_text());cohort = Path(spec['cohort'])
    if meta['status'] != 'complete' or ref['status'] != 'complete':
        raise AssertionError('Complete immutable parents and reference observations are required')
    if sha256(cohort) != spec['cohort_sha256']:raise AssertionError('The frozen risk cohort changed')
    source = root/'assets/PLDR-LLM-v51-SOC-110M-1'
    probe = root/'controlled-study-20260905/data/short'
    out = study/'measurements'/case['name'];out.mkdir(parents=True, exist_ok=False)
    inputs = [protocol, cohort, Path(spec['cohort_manifest']), state, parent,
              parent.parent/'measurements.npz', reference, reference.parent/'measurements.npz',
              source/'modeling_pldrllm.py', source/'configuration_pldrllm.py',
              probe/'tokens.npy', probe/'offsets.npy']
    binding = bind_run(out, inputs, vars(a))
    state_hash = binding['inputs'][str(state.resolve())]
    if (state_hash != meta['saved_states'][str(case['step'])]['sha256'] or
        state_hash != ref['case']['state_sha256']):
        raise AssertionError('The checkpoint differs from the completed training and observation records')
    if binding['inputs'][str((reference.parent/'measurements.npz').resolve())] != ref['raw_sha256']:
        raise AssertionError('The reference observation changed')
    if binding['inputs'][str((parent.parent/'measurements.npz').resolve())] != meta['raw_sha256']:
        raise AssertionError('The online training observations changed')
    torch.set_num_threads(spec['threads'])
    before = time.time();saved = torch.load(state, map_location='cpu', mmap=True, weights_only=True)
    condition = saved['arguments']
    for key in ['heads', 'seed', 'multiplier', 'shared_seed', 'stream_seed']:
        if condition[key] != case[key] or ref['case'][key] != case[key]:
            raise AssertionError('The frozen risk condition changed: '+key)
    if saved['step'] != case['step'] or ref['case']['step'] != case['step']:
        raise AssertionError('The frozen risk horizon changed')
    model = TrainingModel(source, case['heads'], case['seed'], 'cpu')
    model.model.load_state_dict(saved['model']);del saved
    model.model.eval().requires_grad_(False)
    with np.load(cohort) as z:
        train_crops = z['crops'].copy()
    rows = np.arange(512, 1024);pt = np.load(probe/'tokens.npy', mmap_mode='r')
    po = np.load(probe/'offsets.npy');heldout = np.asarray(pt[rows[:, None], po[rows, None]+np.arange(65)])
    raw = {};batch_size = spec['batch_size']
    if train_crops.shape != (1024, 65) or batch_size != 32 or spec['precision'] != 'float32':
        raise AssertionError('The fixed risk observation program changed')
    with torch.no_grad():
        for label, crops in [('training', train_crops), ('heldout', heldout)]:
            losses, entropies = [], []
            for begin in range(0, len(crops), batch_size):
                batch = torch.tensor(crops[begin:begin+batch_size], dtype=torch.long)
                logits = model.forward(batch[:, :64], capture=True).logits[:, -1]
                logp = logits.log_softmax(-1)
                nll = -logp.gather(1, batch[:, 64, None]).squeeze(1)
                entropy = -(logp.exp()*logp).sum(-1)
                if not torch.isfinite(nll).all() or not torch.isfinite(entropy).all():
                    raise FloatingPointError('Nonfinite frozen risk observation')
                losses.append(nll.numpy());entropies.append(entropy.numpy())
                if begin == 0:
                    raw[label+'_first_logits'] = logits.numpy().copy()
                    raw[label+'_first_targets'] = batch[:, 64].numpy().copy()
            raw[label+'_nll'] = np.concatenate(losses)
            raw[label+'_entropy'] = np.concatenate(entropies)
    with np.load(reference.parent/'measurements.npz') as z:
        ref_fields = z['float32_fields']
    differences = np.stack([raw['heldout_nll']-ref_fields[:, 25],
                            raw['heldout_entropy']-ref_fields[:, 24]])
    maximum_error = float(np.max(np.abs(differences)))
    if maximum_error > 1e-5:raise AssertionError('The held-out reference program differs materially')
    with np.load(parent.parent/'measurements.npz') as z:
        losses = z['losses']
    if len(losses) != meta['completed_step']-meta['start_step']:
        raise AssertionError('Online loss counts differ from completed updates')
    end = case['step']-meta['start_step'];start = end-spec['online_window']
    if start < 0:raise AssertionError('The requested online window is unavailable')
    raw['preceding_online_losses'] = losses[start:end]
    results = dict(schema='native-training-risk-state-v1', status='complete', case=case,
        cohorts={label:{name:float(raw[label+'_'+name].astype(float).mean())
            for name in ['nll', 'entropy']} for label in ['training', 'heldout']},
        heldout_reference_maximum_error=maximum_error,
        heldout_reference_bitwise_equal=all(raw['heldout_'+name].dtype == ref_fields[:,column].dtype and
            raw['heldout_'+name].shape == ref_fields[:,column].shape and
            np.ascontiguousarray(raw['heldout_'+name]).tobytes() ==
            np.ascontiguousarray(ref_fields[:, column]).tobytes() for name,column in [('nll',25),('entropy',24)]),
        online_window=dict(begin=case['step']-spec['online_window'], end=case['step'],
            preupdate_mean_loss=float(raw['preceding_online_losses'].mean())),
        scope='CPU float32 full native forward, batches of32, one external next-token target after64 inputs. The training cohort is fresh and fixed across states. Online loss averages concern an evolving trajectory and are separate from frozen-state risk. No native update is performed.')
    write_json(out/'results.json', results);np.savez_compressed(out/'measurements.npz', **raw)
    write_json(out/'manifest.json', dict(schema='native-training-risk-observation-v1', status='complete',
        case=case, binding_sha256=sha256(out/'binding.json'), results_sha256=sha256(out/'results.json'),
        raw_sha256=sha256(out/'measurements.npz'), seconds=time.time()-before))
    print(case['name'], results['cohorts'], 'reference error', maximum_error, flush=True)


if __name__ == '__main__':main()
