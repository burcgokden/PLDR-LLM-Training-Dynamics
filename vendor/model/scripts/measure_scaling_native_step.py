#!/usr/bin/env python
"""Observe all four weight corners of a conditional, complete native AdamW step."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from model_rg.controlled import bind_run
from model_rg.criticality import generator_parameter, optimizer_for, predictive_kl, sample_batches
from model_rg.provenance import sha256, write_json
from model_rg.training import TrainingModel


FIELD_NAMES = ['row_energy', 'matrix_energy', 'row_ratio', 'centroid_squared_rms',
               'operator_rms', 'attention_entropy', 'head_rms']


def observe(model, crops):
    """Native float32 forward; fixed-unit observations accumulated in float64."""
    model.model.eval()
    batch = torch.as_tensor(crops, dtype=torch.long)
    with torch.no_grad():
        output = model.forward(batch[:, :64], capture=True)
        logits = output.logits[:, -1].double()
        logp = logits.log_softmax(-1)
        matrices = torch.stack([x[0].double() for x in output.pldr_attentions], 1)
        operators = torch.stack([x[5].double() for x in output.pldr_attentions], 1)
        attention = torch.stack([x[6][:, :, -1].double() for x in output.pldr_attentions], 1)
        centroid = matrices.mean(-2)
        energy = (matrices-centroid[..., None, :]).square().mean((-2, -1))
        total = matrices.square().mean((-2, -1))
        fields = torch.stack([energy, total, energy/total.clamp_min(1e-30),
            centroid.square().mean(-1), operators.square().mean((-2, -1)).sqrt(),
            -(attention*attention.clamp_min(1e-300).log()).sum(-1),
            torch.stack([model.head_outputs[k].double() for k in range(5)], 1)], -1)
        hidden = torch.stack([x[:, -1].double().square().mean(-1).sqrt()
                              for x in output.hidden_states], 1)
        result = dict(fields=fields, centroids=centroid, hidden_rms=hidden, logits=logits,
            nll=-logp.gather(1, batch[:, 64, None]).squeeze(1),
            prediction_entropy=-(logp.exp()*logp).sum(-1))
    return {name:value.numpy().copy() for name, value in result.items()}


def state_digests(model, optimizer):
    """Every named parameter and Adam tensor, with explicit scalar/group records."""
    parameters, states = {}, {}
    for name, parameter in model.model.named_parameters():
        value = parameter.detach().cpu().contiguous().numpy()
        parameters[name] = dict(shape=list(value.shape), dtype=str(value.dtype),
            sha256=hashlib.sha256(value.tobytes()).hexdigest())
        states[name] = {}
        for key, entry in optimizer.state[parameter].items():
            if isinstance(entry, torch.Tensor):
                value = entry.detach().cpu().contiguous().numpy()
                states[name][key] = dict(shape=list(value.shape), dtype=str(value.dtype),
                    sha256=hashlib.sha256(value.tobytes()).hexdigest())
            else:
                states[name][key] = entry
    groups = [{k:v for k, v in group.items() if k != 'params'} for group in optimizer.param_groups]
    return dict(parameters=parameters, optimizer_states=states, optimizer_groups=groups)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--study', default='critical-scaling-20260906')
    p.add_argument('--protocol', default='native-step.json')
    p.add_argument('--case', required=True)
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    root = Path(a.root);study = root/a.study
    protocol = study/'protocols'/a.protocol;spec = json.loads(protocol.read_text())
    case = next(c for c in spec['cases'] if c['name'] == a.case)
    out = study/'measurements'/case['name'];out.mkdir(parents=True, exist_ok=False)
    source = root/'assets/PLDR-LLM-v51-SOC-110M-1'
    data = root/'data/refinedweb-4608';probe = root/'controlled-study-20260905/data/short'
    if sha256(case['parent']) != case['parent_sha256']:
        raise AssertionError('Incoming complete training state changed')
    bind_run(out, [protocol, Path(case['parent']), Path(case['parent_manifest']),
        data/'tokens.npy', probe/'tokens.npy', probe/'offsets.npy',
        source/'modeling_pldrllm.py', source/'configuration_pldrllm.py'], vars(a))
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
    started = time.time()
    parent = torch.load(case['parent'], map_location='cpu', mmap=True, weights_only=True)
    condition = parent['arguments']
    if (condition['heads'], condition['seed'], parent['step']) != (case['heads'], case['seed'], case['step']):
        raise AssertionError('Incoming state identity changed')
    if condition['multiplier'] != 1:raise AssertionError('This selection fixes the native generator rate')
    model = TrainingModel(source, case['heads'], case['seed'], 'cpu')
    model.model.load_state_dict(parent['model'])
    optimizer = optimizer_for(model, condition['multiplier'])
    optimizer.load_state_dict(copy.deepcopy(parent['optimizer']))
    names = dict(model.model.named_parameters())
    generator_names = [name for name in names if generator_parameter(name)]
    body_names = [name for name in names if not generator_parameter(name)]
    if not generator_names or not body_names:raise AssertionError('Both weight partitions must be present')
    rows, offsets, batches = sample_batches(np.load(data/'tokens.npy'), spec['batch_seed'], spec['batch_replicas'])
    tokens, probe_offsets = np.load(probe/'tokens.npy'), np.load(probe/'offsets.npy')
    cohort = np.arange(512, 528)
    crops = tokens[cohort[:, None], probe_offsets[cohort, None]+np.arange(65)]
    base = observe(model, crops)
    raw = dict(rows=rows, offsets=offsets, cohort=cohort)
    raw.update({'base_'+key:value for key, value in base.items()})
    branch_values = {name:[] for name in ['generator', 'body', 'full']}
    losses, norms, displacement_norms = [], [], []
    for k, batch_array in enumerate(batches):
        # Fresh batches are conditional draws at exactly the same complete state.
        model.model.load_state_dict(parent['model'])
        optimizer.load_state_dict(copy.deepcopy(parent['optimizer']))
        model.model.train();optimizer.zero_grad(set_to_none=True)
        batch = torch.as_tensor(batch_array, dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(model.logits(batch[:, :64]), batch[:, 64])
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite native training loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()));norms.append(float(norm))
        squared = [sum(float((names[name].detach().double()-parent['model'][name].double()).square().sum())
                       for name in group) for group in [generator_names, body_names]]
        displacement_norms.append(squared)
        if k == 0:
            write_json(out/'first-step-digests.json', state_digests(model, optimizer))
        branch_values['full'].append(observe(model, crops))
        # The two partial corners use the displacements from this same globally
        # clipped step. They are interventions on weights, not separate optimizers.
        generator_after = {name:names[name].detach().clone() for name in generator_names}
        with torch.no_grad():
            for name in generator_names:names[name].copy_(parent['model'][name])
        branch_values['body'].append(observe(model, crops))
        with torch.no_grad():
            for name in body_names:names[name].copy_(parent['model'][name])
            for name in generator_names:names[name].copy_(generator_after[name])
        branch_values['generator'].append(observe(model, crops))
        print(case['name'], 'native conditional batch', k+1, 'of', len(batches),
              'seconds', round(time.time()-started, 1), flush=True)
    for branch, values in branch_values.items():
        for key in base:raw[branch+'_'+key] = np.stack([value[key] for value in values])
        raw[branch+'_predictive_kl'] = np.stack([
            predictive_kl(torch.from_numpy(base['logits']), torch.from_numpy(value['logits'])).numpy()
            for value in values])
    raw.update(losses=np.array(losses), gradient_norms=np.array(norms),
               displacement_squared_norms=np.array(displacement_norms))
    if any(not np.isfinite(value).all() for value in raw.values()):
        raise FloatingPointError('Nonfinite native step observation')
    # A repeat forward at the restored parent checks intervention cleanup.
    model.model.load_state_dict(parent['model'])
    restored = observe(model, crops)
    for key in base:
        np.testing.assert_array_equal(base[key], restored[key])
        if (base[key].shape != restored[key].shape or base[key].dtype != restored[key].dtype
                or base[key].tobytes() != restored[key].tobytes()):
            raise AssertionError('Restored parent emission bytes differ: '+key)
    np.savez_compressed(out/'measurements.npz', **raw)
    result = dict(schema='native-conditional-step-corners-v1', status='complete', case=case,
        batch_seed=spec['batch_seed'], batch_replicas=spec['batch_replicas'],
        branches=['generator', 'body', 'full'], field_names=FIELD_NAMES,
        generator_parameter_names=generator_names, body_parameter_names=body_names,
        generator_dimension=sum(names[name].numel() for name in generator_names),
        body_dimension=sum(names[name].numel() for name in body_names),
        restored_parent_observations_bitwise_equal=True,
        started_at=datetime.fromtimestamp(started, timezone.utc).isoformat(),
        seconds=time.time()-started,
        scope='Fresh full batch32 CPU float32 gradients and the actual complete AdamW step at a fixed incoming model and optimizer. Partial weight corners retain the realized globally clipped displacements of that same step. The 16 contexts are fixed. These draws are not sequential training updates or independent initializations; no infinitesimal approximation or scalar Markov closure is imposed.')
    write_json(out/'results.json', result)
    write_json(out/'manifest.json', dict(schema=result['schema'], status='complete', case=case,
        binding_sha256=sha256(out/'binding.json'), raw_sha256=sha256(out/'measurements.npz'),
        first_step_digests_sha256=sha256(out/'first-step-digests.json'),
        results_sha256=sha256(out/'results.json')))


if __name__ == '__main__':main()
