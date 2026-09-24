#!/usr/bin/env python
"""Compare native initial residual kernels with the derived head-kernel recursion."""
import argparse
import gc
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from model_rg.controlled import bind_run
from model_rg.criticality import fix_shared_generator
from model_rg.provenance import sha256, write_json
from model_rg.training import TrainingModel
from model_rg.variance_family import normalize_variance_initialization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    study = root/'criticality-study-20260905'
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    source = root/'assets/PLDR-LLM-v51-SOC-110M-1'
    data = root/'controlled-study-20260905/data/short'
    initial = study/'analysis/initialization-family'
    limit = study/'analysis/initialization-limit'
    bind_run(output, [source/'modeling_pldrllm.py', source/'configuration_pldrllm.py',
        data/'tokens.npy', data/'offsets.npy', initial/'results.json', initial/'measurements.npz',
        limit/'results.json', limit/'measurements.npz', study/'protocols/initialization-limit.json'], vars(args))
    torch.set_num_threads(4)
    tokens = np.load(data/'tokens.npy')
    offsets = np.load(data/'offsets.npy')
    rows = np.arange(512, 528)
    crops = tokens[rows[:, None], offsets[rows, None] + np.arange(64)]
    batch = torch.tensor(crops, dtype=torch.long)
    old = np.load(initial/'measurements.npz')
    initial_results = json.loads((initial/'results.json').read_text())
    kernel_raw = np.load(limit/'measurements.npz')
    prediction = kernel_raw['kernels'].mean(0)
    raw = dict(cohort=rows, crops=crops, predicted_kernels=prediction)
    records = []
    started = time.time()
    for heads in [2, 4, 8, 14, 24]:
        for seed in [640301, 640302, 640303, 640304]:
            model = TrainingModel(source, heads, seed, 'cpu')
            normalize_variance_initialization(model)
            shared = fix_shared_generator(model, 640011)
            model.model.eval()
            boundary = {}
            hook = model.model.decoder.layernorm1.register_forward_hook(
                lambda module, inputs, value: boundary.update(normalized_input=value.detach().clone()))
            with torch.no_grad():
                result = model.forward(batch, capture=True)
                # The native first exported state aliases the scaled embedding before input normalization.
                states = [boundary['normalized_input'], *result.hidden_states[1:]]
                kernels = torch.stack([h.double() @ h.double().transpose(-1, -2) / h.shape[-1]
                                       for h in states], 1).numpy()
                hook.remove()
                fields = []
                for values in result.pldr_attentions:
                    metric = values[0]
                    probability = values[-1][:, :, -1]
                    entropy = -(probability * probability.clamp_min(1e-38).log()).sum(-1) / math.log(64)
                    centered = metric - metric.mean(-2, keepdim=True)
                    row = centered.square().mean((-1, -2)) / metric.square().mean((-1, -2)).clamp_min(1e-30)
                    fields.append(torch.stack([entropy, row], -1))
                fields = torch.stack(fields, 1).numpy()
            name = f'variance-h{heads}-s{seed}'
            identity_error = float(np.max(np.abs(fields - old[name+'_heads'][..., [0, 2]])))
            if identity_error > 1e-6:
                raise AssertionError('The repeated initialization differs from its original diagnostic')
            expected = next(x for x in initial_results['rows'] if x['family']=='variance' and x['heads']==heads and x['seed']==seed)
            if shared != expected['shared_generator_sha256']:
                raise AssertionError('The shared initialization changed')
            raw[name+'_kernels'] = kernels
            raw[name+'_fields'] = fields
            error = kernels - prediction
            records.append(dict(heads=heads, seed=seed, identity_error=identity_error,
                                rmse_by_layer=np.sqrt(np.mean(error**2, axis=(0, 2, 3))).tolist()))
            print(name, 'seconds', round(time.time()-started, 1), flush=True)
            del model, result, kernels, fields, states, boundary
            gc.collect()
    comparison = []
    for heads in [2, 4, 8, 14, 24]:
        selected = [x for x in records if x['heads']==heads]
        errors = np.array([x['rmse_by_layer'] for x in selected])
        comparison.append(dict(heads=heads, rmse_by_layer=np.sqrt(np.mean(errors**2, axis=0)).tolist(),
                               identity_max_error=max(x['identity_error'] for x in selected)))
    mc = kernel_raw['kernels']
    mc_se = mc.std(0, ddof=1) / np.sqrt(len(mc))
    field_mc = kernel_raw['head_fields'].mean(3)
    field_se = field_mc.std(0, ddof=1) / np.sqrt(len(field_mc))
    np.savez_compressed(output/'measurements.npz', **raw)
    results = dict(schema='native-initial-kernel-comparison-v1',cohort=rows.tolist(),records=records,
        comparison=comparison,seconds=time.time()-started,
        monte_carlo_kernel_se_rms_by_layer=np.sqrt(np.mean(mc_se**2,axis=(0,2,3))).tolist(),
        monte_carlo_field_se_rms=np.sqrt(np.mean(field_se**2,axis=(0,1))).tolist(),
        interpretation='Twenty repeated initializations from the existing sixty-model diagnostic; no new independent model seeds. Kernel error uses all within-context token pairs at six residual stages. Monte Carlo standard errors describe repeated numerical integration and do not bound integration bias or errors of the infinite-width limit.')
    write_json(output/'results.json',results)
    write_json(output/'manifest.json',dict(results_sha256=sha256(output/'results.json'),
        raw_sha256=sha256(output/'measurements.npz'),binding_sha256=sha256(output/'binding.json')))


if __name__ == '__main__':
    main()
