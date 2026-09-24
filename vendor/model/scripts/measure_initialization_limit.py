#!/usr/bin/env python
"""Evaluate a derived native-head initialization kernel, without training."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.special import expit
import torch
from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json
from model_rg.training import TrainingModel


def normalize_kernel(kernel, epsilon):
    scale = np.sqrt(np.diag(kernel) + epsilon)
    return kernel / scale[:, None] / scale[None, :]


def silu_covariance(kernel, order):
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    nodes = np.sqrt(2) * nodes
    weights = weights / np.sqrt(np.pi)
    scale = np.sqrt(np.maximum(np.diag(kernel), 0))
    denominator = scale[:, None] * scale[None, :]
    correlation = np.divide(kernel, denominator, out=np.zeros_like(kernel), where=denominator > 0)
    if np.max(np.abs(correlation)) > 1 + 1e-9:
        raise AssertionError('Invalid Gaussian correlation')
    correlation = np.clip(correlation, -1, 1)
    result = np.empty_like(kernel)
    for i in range(len(kernel)):
        first = scale[i] * nodes
        second = scale[:, None, None] * (correlation[i, :, None, None] * nodes[None, :, None]
            + np.sqrt(1-correlation[i, :, None, None]**2) * nodes[None, None, :])
        result[i] = np.einsum('u,juv,u,v->j', first * expit(first), second * expit(second), weights, weights)
    return .5 * (result + result.T)


@torch.no_grad()
def native_head_kernel(reference, layer, kernel, samples, chunk, generator):
    attention = reference.model.decoder.dec_layers[layer].mha1
    dimension = 64
    eigenvalues, vectors = np.linalg.eigh(.5 * (kernel + kernel.T))
    if eigenvalues.min() < -1e-9:
        raise AssertionError('A kernel is not positive semidefinite')
    factor = torch.tensor(vectors * np.sqrt(np.maximum(eigenvalues, 0)), dtype=torch.float64)
    covariance = torch.tensor(kernel, dtype=torch.float64)
    accumulated = torch.zeros_like(covariance)
    fields = []
    length = len(kernel)
    mask = torch.triu(torch.full((length, length), torch.finfo(torch.float32).min), diagonal=1)[None, None]
    for start in range(0, samples, chunk):
        count = min(chunk, samples-start)
        q = (factor @ torch.randn(count, length, dimension, generator=generator, dtype=torch.float64)).float()
        k = (factor @ torch.randn(count, length, dimension, generator=generator, dtype=torch.float64)).float()
        q = attention.rotary_embedding(q.transpose(0, 1)[None], input_pos=None).permute(0, 2, 1, 3)
        k = attention.rotary_embedding(k.transpose(0, 1)[None], input_pos=None).permute(0, 2, 1, 3)
        metric = attention.layernorm1(q.transpose(-1, -2) @ q)
        for unit in attention.reslayerAs:
            metric = unit([metric])
        power = reference.module.PlgaLayer(reference.config, dimension, count, layer, device='cpu').eval()
        limit = math.sqrt(6 / (64 * 66))
        for name in ['Wlst', 'pwlst', 'alst']:
            getattr(power, name).uniform_(-limit, limit, generator=generator)
        power.blst.zero_()
        power.balst.zero_()
        _, details = power((q, k, torch.zeros_like(q), metric, mask), use_cache=False)
        probability = details[-1][0].double()
        accumulated += (probability @ covariance @ probability.transpose(-1, -2)).sum(0)
        last = probability[:, -1]
        entropy = -(last * last.clamp_min(1e-38).log()).sum(-1) / math.log(length)
        metric = metric[0].double()
        centered = metric - metric.mean(-2, keepdim=True)
        row = centered.square().mean((-1, -2)) / metric.square().mean((-1, -2)).clamp_min(1e-30)
        fields.append(torch.stack([entropy, row], -1).numpy())
    return accumulated.numpy() / samples, np.concatenate(fields)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--samples', type=int, default=512)
    parser.add_argument('--replicas', type=int, default=4)
    parser.add_argument('--contexts', type=int, default=16)
    parser.add_argument('--chunk', type=int, default=64)
    args = parser.parse_args()
    root = Path(args.root)
    study = root / 'criticality-study-20260905'
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    source = root / 'assets/PLDR-LLM-v51-SOC-110M-1'
    data = root / 'controlled-study-20260905/data/short'
    diagnostic = study / 'analysis/initialization-family'
    bind_run(output, [source/'modeling_pldrllm.py', source/'configuration_pldrllm.py',
        data/'tokens.npy', data/'offsets.npy', diagnostic/'results.json', diagnostic/'measurements.npz',
        study/'protocols/initialization-limit.json'], vars(args))
    torch.set_num_threads(4)
    reference = TrainingModel(source, 2, 640011, 'cpu')
    reference.model.eval()
    tokens = np.load(data/'tokens.npy')
    offsets = np.load(data/'offsets.npy')
    rows = np.arange(512, 512+args.contexts)
    crops = tokens[rows[:, None], offsets[rows, None] + np.arange(64)]
    if (crops == 0).any():
        raise AssertionError('This kernel control requires nonpadding contexts')
    first_gain = 256 / 469
    last_gain = 682 / 469
    vocabulary_gain = 256 / 32128
    raw = dict(cohort=rows, crops=crops)
    kernels = np.empty((args.replicas, args.contexts, 6, 64, 64))
    head_fields = np.empty((args.replicas, args.contexts, 5, args.samples, 2))
    quadrature_errors = []
    started = time.time()
    for replica in range(args.replicas):
        generator = torch.Generator().manual_seed(640411+replica)
        for context, crop in enumerate(crops):
            kernel = (crop[:, None] == crop[None, :]).astype(float)
            kernels[replica, context, 0] = kernel
            for layer in range(5):
                attention_kernel, fields = native_head_kernel(reference, layer, kernel, args.samples, args.chunk, generator)
                head_fields[replica, context, layer] = fields
                middle = normalize_kernel(kernel + attention_kernel, reference.config.layer_norm_eps)
                activation = silu_covariance(first_gain * middle, 20)
                if replica == 0:
                    checked = silu_covariance(first_gain * middle, 28)
                    quadrature_errors.append(float(np.max(np.abs(activation-checked))))
                feedforward = last_gain * first_gain * middle * activation
                kernel = normalize_kernel(middle + feedforward, reference.config.layer_norm_eps)
                kernels[replica, context, layer+1] = kernel
            print('replica', replica, 'context', int(rows[context]), 'seconds', round(time.time()-started, 1), flush=True)
    raw.update(kernels=kernels, head_fields=head_fields, quadrature_errors=np.array(quadrature_errors))
    np.savez_compressed(output/'measurements.npz', **raw)
    prediction = head_fields.mean(3)
    observed = np.load(diagnostic/'measurements.npz')
    initial = json.loads((diagnostic/'results.json').read_text())
    comparison = []
    for heads in [2, 4, 8, 14, 24]:
        records = [r for r in initial['rows'] if r['family']=='variance' and r['heads']==heads]
        values = np.stack([observed[f"variance-h{heads}-s{r['seed']}_heads"][:args.contexts, ..., [0, 2]].mean(2) for r in records])
        residual = values - prediction.mean(0)[None]
        comparison.append(dict(heads=heads, seeds=[r['seed'] for r in records],
            empirical_means=values.mean((0, 1)).tolist(),
            rmse=np.sqrt(np.mean(residual**2, axis=(0, 1, 2))).tolist(),
            rmse_by_seed=np.sqrt(np.mean(residual**2, axis=(1, 2))).tolist(),
            logit_mean_square=float(np.mean([r['logit_mean_square'] for r in records]))))
    logit_variance=vocabulary_gain*np.diagonal(kernels[:, :, -1], axis1=-2, axis2=-1)[:, :, -1]
    results=dict(schema='native-initialization-limit-v1',arguments=vars(args),cohort=rows.tolist(),
        monte_carlo_seeds=list(range(640411,640411+args.replicas)),
        predicted_means=prediction.mean((0, 1)).tolist(),
        predicted_means_by_replica=prediction.mean(1).tolist(),
        prediction_by_context=prediction.mean(0).tolist(),
        predicted_logit_mean_square=float(logit_variance.mean()),
        quadrature_max_difference=max(quadrature_errors),comparison=comparison,seconds=time.time()-started,
        interpretation='Initialization kernel with native finite-dimensional PLGA heads and analytically integrated values. Monte Carlo replicas quantify numerical integration; empirical replicas are the independent diagnostic initializations. This is an initial-law comparison, not a trained criticality or stationary-law measurement.')
    write_json(output/'results.json',results)
    write_json(output/'manifest.json',dict(results_sha256=sha256(output/'results.json'),
        raw_sha256=sha256(output/'measurements.npz'),binding_sha256=sha256(output/'binding.json')))


if __name__ == '__main__':
    main()
