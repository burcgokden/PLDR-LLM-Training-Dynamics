#!/usr/bin/env python
"""Measure declared initialization variances and emissions on a fixed cohort."""
import argparse
import gc
from pathlib import Path
import time
import numpy as np
import torch
from model_rg.controlled import bind_run
from model_rg.criticality import normalize_initialization, fix_shared_generator, observations
from model_rg.variance_family import normalize_variance_initialization
from model_rg.provenance import sha256, write_json
from model_rg.training import TrainingModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--heads', default='2,4,8,14,24')
    parser.add_argument('--seeds', default='640301,640302,640303,640304')
    args = parser.parse_args()
    root = Path(args.root); out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    source = root/'assets/PLDR-LLM-v51-SOC-110M-1'
    data = root/'controlled-study-20260905/data/short'
    protocol = root/'criticality-study-20260905/protocols/initialization-variance.json'
    bind_run(out, [source/'modeling_pldrllm.py', source/'configuration_pldrllm.py',
                   data/'manifest.json', data/'tokens.npy', data/'offsets.npy', protocol], vars(args))
    torch.set_num_threads(4)
    tokens = np.load(data/'tokens.npy'); offsets = np.load(data/'offsets.npy')
    indices = np.arange(512, 528)
    crops = tokens[indices[:, None], offsets[indices, None] + np.arange(65)]
    batch = torch.tensor(crops, dtype=torch.long)
    rows = []; raw = {}; started = time.time()
    for heads in map(int, args.heads.split(',')):
        for seed in map(int, args.seeds.split(',')):
            # Reinitialization uses the identical model seed for each family.
            for family in ['native', 'fan_in', 'variance']:
                model = TrainingModel(source, heads, seed, 'cpu')
                if family == 'fan_in':
                    normalize_initialization(model)
                elif family == 'variance':
                    normalize_variance_initialization(model)
                shared = fix_shared_generator(model, 640011)
                model.model.eval()
                weights = {}
                for name, parameter in model.model.named_parameters():
                    if ('.dec_layers.0.' in name or name == 'final_layer.weight') and parameter.ndim >= 2 and 'reslayerAs' not in name:
                        values = parameter.detach().double()
                        weights[name] = dict(shape=list(values.shape), variance=float(values.var(unbiased=False)),
                                             mean_square=float(values.square().mean()), mean=float(values.mean()))
                with torch.no_grad():
                    fields, per_head, logits, _ = observations(model, batch[:, :64], batch[:, 64])
                name = f'{family}-h{heads}-s{seed}'
                raw[name+'_fields'] = fields.numpy()
                raw[name+'_heads'] = per_head.numpy()
                rows.append(dict(family=family, heads=heads, seed=seed, shared_generator_sha256=shared,
                                 weights=weights, mean_nll=float(fields[:, 25].mean()),
                                 logit_mean_square=float(logits.square().mean()),
                                 mean_head_fields=per_head.mean((0, 2)).numpy().tolist()))
                print(name, 'NLL', rows[-1]['mean_nll'], 'seconds', round(time.time()-started, 1), flush=True)
                del model, fields, per_head, logits
                gc.collect()
    np.savez_compressed(out/'measurements.npz', **raw)
    write_json(out/'results.json', dict(schema='initialization-family-v1', arguments=vars(args), rows=rows,
                                       cohort=indices.tolist(), seconds=time.time()-started))
    write_json(out/'manifest.json', dict(results_sha256=sha256(out/'results.json'),
                                        raw_sha256=sha256(out/'measurements.npz'), binding_sha256=sha256(out/'binding.json')))


if __name__ == '__main__':
    main()
