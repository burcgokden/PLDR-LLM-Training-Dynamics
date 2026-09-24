#!/usr/bin/env python
"""Resolve selected native gain-response tails with a full-graph float64 JVP."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from model_rg.controlled import bind_run
from model_rg.criticality import predictive_kl
from model_rg.precision import preserve_response_dtype, expand_smooth_response
from model_rg.provenance import sha256, write_json
from model_rg.training import TrainingModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--parent', required=True)
    args = parser.parse_args()
    root = Path(args.root); study = root/'criticality-study-20260905'
    protocol_path = study/'protocols/source-precision.json'
    protocol = json.loads(protocol_path.read_text())
    if args.parent not in protocol['parents']:
        raise ValueError('The parent is outside the declared precision selection')
    parent = study/'runs'/args.parent
    meta = json.loads((parent/'manifest.json').read_text())
    if meta['status'] != 'complete':
        raise RuntimeError('The parent training result must be complete')
    out = study/'source-precision'/args.parent
    out.mkdir(parents=True, exist_ok=False)
    probe = root/'controlled-study-20260905/data/short'
    source = root/'assets/PLDR-LLM-v51-SOC-110M-1'
    bind_run(out, [parent/'manifest.json', parent/'measurements.npz', parent/'final-training-state.pt',
                   probe/'manifest.json', probe/'tokens.npy', probe/'offsets.npy',
                   source/'modeling_pldrllm.py', source/'configuration_pldrllm.py', protocol_path], vars(args))
    if sha256(parent/'measurements.npz') != meta['raw_sha256']:
        raise AssertionError('Parent response hash mismatch')
    data = np.load(parent/'measurements.npz')
    selected = np.array(sorted(set([0, 16, 32, 48]) | set(np.argsort(data['source_half_curvature'])[-4:].tolist())))
    tokens = np.load(probe/'tokens.npy'); offsets = np.load(probe/'offsets.npy')
    rows = 512 + selected
    crops = tokens[rows[:, None], offsets[rows, None] + np.arange(65)]
    torch.set_num_threads(4)
    checkpoint = torch.load(parent/'final-training-state.pt', map_location='cpu', mmap=True, weights_only=True)
    condition = checkpoint['arguments']; started = time.time()
    model = TrainingModel(source, condition['heads'], condition['seed'], 'cpu')
    model.model.load_state_dict(checkpoint['model'])
    del checkpoint
    model.model.eval().requires_grad_(False)
    ids = torch.tensor(crops[:, :64], dtype=torch.long)
    raw = dict(selected_source_indices=selected, probe_rows=rows,
               amplitudes=np.array(protocol['amplitudes']),
               native_gpu_logits=data['source_baseline_logits'][selected],
               native_gpu_half_secant=data['source_half_secant'][selected],
               native_gpu_half_curvature=data['source_half_curvature'][selected])
    with torch.no_grad():
        native = model.logits(ids).double()
        raw['native_cpu_logits'] = native.numpy()
        model.model.double()
        preserve_response_dtype(model)
        dtype_reference = model.logits(ids).detach()
        expand_smooth_response(model)
        def logits(alpha):
            return model.logits(ids, eta=alpha.expand(5, condition['heads']))
        zero = torch.tensor(0., dtype=torch.float64)
        primal, derivative = torch.func.jvp(logits, (zero,), (torch.ones_like(zero),))
        ordinary = logits(zero)
        probability = primal.softmax(-1)
        centered = derivative - (probability * derivative).sum(-1, keepdim=True)
        curvature = (probability * centered.square()).sum(-1)
        positive, negative, secants, errors = [], [], [], []
        for amplitude in protocol['amplitudes']:
            plus = logits(torch.tensor(amplitude, dtype=torch.float64))
            minus = logits(torch.tensor(-amplitude, dtype=torch.float64))
            secant = (plus - minus)/(2*amplitude)
            secant -= (probability * secant).sum(-1, keepdim=True)
            discrepancy = (probability * (secant-centered).square()).sum(-1).sqrt()
            positive.append(plus.numpy()); negative.append(minus.numpy())
            secants.append(secant.numpy()); errors.append(discrepancy.numpy())
        raw.update(dtype_reference_logits=dtype_reference.numpy(), smooth64_logits=primal.numpy(),
                   smooth64_ordinary_logits=ordinary.numpy(), smooth64_jvp=derivative.numpy(),
                   smooth64_centered_jvp=centered.numpy(), smooth64_curvature=curvature.numpy(),
                   positive_logits=np.array(positive), negative_logits=np.array(negative),
                   smooth64_secants=np.array(secants), absolute_errors=np.array(errors),
                   relative_errors=np.array(errors)/np.maximum(curvature.sqrt().numpy()[None], 1e-14),
                   native_cpu_gpu_kl=predictive_kl(torch.tensor(raw['native_gpu_logits']), native).numpy(),
                   smooth_native_cpu_kl=predictive_kl(native, primal).numpy())
    for name, array in raw.items():
        if not np.isfinite(array).all():
            raise RuntimeError(f'Nonfinite precision result: {name}')
    np.savez_compressed(out/'measurements.npz', **raw)
    write_json(out/'manifest.json', dict(schema='gain-precision-v1', status='complete', parent=args.parent,
        condition=condition, selected_source_indices=selected.tolist(),
        dtype_reference_max_logit_error=float(np.max(np.abs(raw['dtype_reference_logits']-raw['smooth64_logits']))),
        jvp_primal_max_logit_error=float(np.max(np.abs(raw['smooth64_ordinary_logits']-raw['smooth64_logits']))),
        seconds=time.time()-started, binding_sha256=sha256(out/'binding.json'), raw_sha256=sha256(out/'measurements.npz'),
        interpretation=protocol['interpretive_boundary']))
    print(args.parent, 'complete', round(time.time()-started, 1), flush=True)


if __name__ == '__main__':
    main()
