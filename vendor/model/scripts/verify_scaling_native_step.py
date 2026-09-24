#!/usr/bin/env python
"""Replay each first conditional native step independently and check all tensors."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from model_rg.provenance import sha256, write_json
from model_rg.training import TrainingModel


def bitwise_equal(a, b):
    """Compare shape, dtype and logical C-order bytes, including signed zero."""
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes(order='C') == b.tobytes(order='C')


def tensor_record(value):
    array = value.detach().cpu().contiguous().numpy()
    return {'dtype':str(array.dtype), 'shape':list(array.shape),
            'sha256':hashlib.sha256(array.tobytes()).hexdigest()}


def replay(root, folder, case, spec, threads):
    manifest = json.loads((folder/'manifest.json').read_text())
    binding = json.loads((folder/'binding.json').read_text())
    if manifest['status'] != 'complete' or sha256(case['parent']) != case['parent_sha256']:
        raise AssertionError('The selected incoming state is missing or changed')
    for name, field in [('measurements.npz', 'raw_sha256'), ('results.json', 'results_sha256'),
                        ('binding.json', 'binding_sha256'), ('first-step-digests.json', 'first_step_digests_sha256')]:
        if sha256(folder/name) != manifest[field]:raise AssertionError('Changed observation artifact')
    source = root/'assets/PLDR-LLM-v51-SOC-110M-1'
    for path in [root/'data/refinedweb-4608/tokens.npy', root/'controlled-study-20260905/data/short/tokens.npy',
                 root/'controlled-study-20260905/data/short/offsets.npy', source/'modeling_pldrllm.py',
                 source/'configuration_pldrllm.py']:
        if sha256(path) != binding['inputs'][str(path)]:raise AssertionError('Changed data or architecture')
    parent = torch.load(case['parent'], weights_only=True, mmap=True, map_location='cpu')
    model = TrainingModel(source, case['heads'], case['seed'], 'cpu')
    model.model.load_state_dict(parent['model'])
    named = dict(model.model.named_parameters())
    # Keep this partition and optimizer construction independent of the producer.
    generator = [name for name in named if any(x in name for x in ['reslayerAs', 'plgatt_layer', 'layernormA'])]
    body = [name for name in named if name not in generator]
    optimizer = torch.optim.AdamW([
        dict(params=[named[name] for name in generator], lr=.0003),
        dict(params=[named[name] for name in body], lr=.0006/case['heads'])],
        betas=(.9, .95), eps=1e-8, weight_decay=.01, foreach=False)
    optimizer.load_state_dict(copy.deepcopy(parent['optimizer']))
    rows = np.random.default_rng(spec['batch_seed']+1000).integers(0, 3072, (spec['batch_replicas'], 32))
    offsets = np.random.default_rng(spec['batch_seed']+1001000).integers(0, 449, (spec['batch_replicas'], 32))
    tokens = np.load(root/'data/refinedweb-4608/tokens.npy')
    batch = torch.as_tensor(tokens[rows[0, :, None], offsets[0, :, None]+np.arange(65)], dtype=torch.long)
    probe = root/'controlled-study-20260905/data/short'
    pt, po = np.load(probe/'tokens.npy'), np.load(probe/'offsets.npy')
    selected = np.arange(512, 528)
    crops = torch.as_tensor(pt[selected[:, None], po[selected, None]+np.arange(65)], dtype=torch.long)
    checked = 0
    with np.load(folder/'measurements.npz') as raw:
        np.testing.assert_array_equal(raw['rows'], rows);np.testing.assert_array_equal(raw['offsets'], offsets)
        def check_emission(prefix, index=None):
            nonlocal checked
            model.model.eval()
            with torch.no_grad():
                output = model.forward(crops[:, :64], capture=True)
                z = output.logits[:, -1].double()
                lp = z.log_softmax(-1)
                means, field_layers = [], []
                for layer, entry in enumerate(output.pldr_attentions):
                    matrix, operator = entry[0].double(), entry[5].double()
                    mu = matrix.mean(-2);means.append(mu)
                    contrast = ((matrix-mu.unsqueeze(-2))**2).mean((-2, -1))
                    total = (matrix**2).mean((-2, -1))
                    weights = entry[6][:, :, -1].double()
                    field_layers.append(torch.stack([contrast, total, contrast/total.clamp_min(1e-30),
                        (mu**2).mean(-1), (operator**2).mean((-2, -1)).sqrt(),
                        -(weights*weights.clamp_min(1e-300).log()).sum(-1),
                        model.head_outputs[layer].double()], -1))
                actual = dict(logits=z, fields=torch.stack(field_layers, 1), centroids=torch.stack(means, 1),
                    hidden_rms=torch.stack([(h[:, -1].double()**2).mean(-1).sqrt() for h in output.hidden_states], 1),
                    nll=-lp.gather(1, crops[:, 64, None]).squeeze(1),
                    prediction_entropy=-(lp.exp()*lp).sum(-1))
            for name, value in actual.items():
                expected = raw[prefix+'_'+name]
                if index is not None:expected = expected[index]
                np.testing.assert_array_equal(value.numpy(), expected)
                if not bitwise_equal(value.numpy(), expected):
                    raise AssertionError('Emission bytes differ: '+prefix+'_'+name)
                checked += value.numel()
        check_emission('base')
        model.model.train();optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model.logits(batch[:, :64]), batch[:, 64])
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1., error_if_nonfinite=True)
        np.testing.assert_equal(float(loss.detach()), float(raw['losses'][0]))
        np.testing.assert_equal(float(norm), float(raw['gradient_norms'][0]))
        optimizer.step()
        expected = json.loads((folder/'first-step-digests.json').read_text())
        for name, parameter in named.items():
            if tensor_record(parameter) != expected['parameters'][name]:raise AssertionError('Native parameter replay differs: '+name)
            for key, value in optimizer.state[parameter].items():
                actual = tensor_record(value) if isinstance(value, torch.Tensor) else value
                if actual != expected['optimizer_states'][name][key]:
                    raise AssertionError('Native optimizer replay differs: '+name+'/'+key)
            if set(optimizer.state[parameter]) != set(expected['optimizer_states'][name]):
                raise AssertionError('Optimizer slot inventory changed')
        groups = [{k:v for k, v in group.items() if k != 'params'} for group in optimizer.param_groups]
        # JSON represents beta tuples as arrays.
        if json.loads(json.dumps(groups)) != expected['optimizer_groups']:
            raise AssertionError('Optimizer groups changed')
        if set(named) != set(expected['parameters']):raise AssertionError('Parameter inventory changed')
        check_emission('full', 0)
        updated_generator = {name:named[name].detach().clone() for name in generator}
        with torch.no_grad():
            for name in generator:named[name].copy_(parent['model'][name])
        check_emission('body', 0)
        with torch.no_grad():
            for name in body:named[name].copy_(parent['model'][name])
            for name in generator:named[name].copy_(updated_generator[name])
        check_emission('generator', 0)
    return dict(case=case, parameter_tensors=len(named),
        optimizer_tensors=sum(isinstance(value, torch.Tensor) for parameter in named.values()
                              for value in optimizer.state[parameter].values()),
        emission_elements_bitwise_checked=checked, first_step_bitwise_replay_passed=True,
        parent_sha256=case['parent_sha256'], manifest_sha256=sha256(folder/'manifest.json'),
        binding_sha256=manifest['binding_sha256'], raw_sha256=manifest['raw_sha256'])


def main():
    p = argparse.ArgumentParser();p.add_argument('--root', required=True)
    p.add_argument('--study', default='critical-scaling-20260906')
    p.add_argument('--protocol', default='native-step.json');p.add_argument('--output', required=True)
    p.add_argument('--threads', type=int, default=4);p.add_argument('--wait', action='store_true')
    a = p.parse_args();root = Path(a.root);study = root/a.study
    out = Path(a.output)
    if out.exists():raise FileExistsError(out)
    protocol = study/'protocols'/a.protocol;spec = json.loads(protocol.read_text())
    while not all((study/'measurements'/c['name']/'manifest.json').exists() for c in spec['cases']):
        if not a.wait:raise RuntimeError('The complete selected conditional states are required')
        time.sleep(30)
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
    records = []
    for case in spec['cases']:
        records.append(replay(root, study/'measurements'/case['name'], case, spec, a.threads))
        print('Independent complete native step replay passed:', case['name'], flush=True)
    repo = Path(__file__).resolve().parents[1]
    write_json(out, dict(status='passed', protocol=str(protocol), protocol_sha256=sha256(protocol),
        states=records, selected_states=len(records), arguments=vars(a),
        emission_equality='Shape, dtype and logical C-order bytes, including signed-zero and NaN payload bits.',
        verifier_sources={str(path.relative_to(repo)):sha256(path) for path in
            [Path(__file__), repo/'src/model_rg/training.py', repo/'src/model_rg/native.py', repo/'src/model_rg/provenance.py']},
        scope='Independent first-batch full native CPU replay at every selected incoming state, with every named parameter and Adam tensor checked bitwise, and every saved scalar of all four first-batch emissions checked bitwise. The remaining conditional draws are raw measurements reconstructed by the separate statistical analysis, not additional bitwise replay cases.'))


if __name__ == '__main__':main()
