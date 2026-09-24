#!/usr/bin/env python
"""Check native observation compatibility, energy decomposition and accumulation."""
import argparse
from pathlib import Path

import numpy as np
import torch

from model_rg.controlled import bind_run
from model_rg.criticality import (observations as previous_observations, fix_shared_generator,
                                  optimizer_for, sample_batches)
from model_rg.provenance import sha256, write_json
from model_rg.scaling import observations
from model_rg.checkpointing import enable_decoder_checkpointing
from model_rg.training import TrainingModel
from model_rg.variance_family import normalize_variance_initialization


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--checkpoint-decoders', action='store_true')
    a = p.parse_args()
    root = Path(a.root)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    source = root/'assets/PLDR-LLM-v51-SOC-110M-1'
    data = root/'data/refinedweb-4608/tokens.npy'
    bind_run(out, [data, source/'modeling_pldrllm.py', source/'configuration_pldrllm.py'], vars(a))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = TrainingModel(source, a.heads, 640100, a.device)
    normalize_variance_initialization(model)
    fix_shared_generator(model, 640011)
    _, _, batches = sample_batches(np.load(data), 640001, 1)
    batch = torch.tensor(batches[0], dtype=torch.long, device=a.device)
    model.model.eval()
    with torch.no_grad():
        old = previous_observations(model, batch[:, :64], batch[:, 64], matrices=True)
        new = observations(model, batch[:, :64], batch[:, 64], matrices=True)
        for i in range(3):
            torch.testing.assert_close(old[i], new[i], rtol=0, atol=0)
        for before, after in zip(old[3], new[3], strict=True):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        extra = new[4]
        energy, total = extra['energies'].unbind(-1)
        torch.testing.assert_close(total, energy+extra['centroids'].square().mean(-1), rtol=1e-12, atol=1e-14)
        assert torch.isfinite(extra['centroids']).all() and torch.isfinite(extra['energies']).all()
    del old, new, extra
    model.model.train()
    gradients, losses = [], []
    for index, micro in enumerate([32, 32] if a.checkpoint_decoders else [32, 8]):
        if index == 1 and a.checkpoint_decoders:
            enable_decoder_checkpointing(model)
        model.model.zero_grad(set_to_none=True)
        value = 0.
        for begin in range(0, 32, micro):
            piece = batch[begin:begin+micro]
            loss = torch.nn.functional.cross_entropy(model.logits(piece[:, :64]), piece[:, 64])
            (loss*(micro/32)).backward()
            value += float(loss.detach())*(micro/32)
        gradients.append([x.grad.detach().cpu().double().clone() for x in model.model.parameters()])
        losses.append(value)
    squared = sum(float(x.square().sum()) for x in gradients[0])
    discrepancy = sum(float((x-y).square().sum()) for x,y in zip(*gradients, strict=True))
    relative = (discrepancy/squared)**.5
    np.savez_compressed(out/'measurements.npz', losses=losses, gradient_squared=squared,
                        discrepancy_squared=discrepancy)
    write_json(out/'verification.json', dict(status='passed' if relative <= 1e-4 else 'accumulation_not_accepted', heads=a.heads,
        observation_bitwise_compatible=True, energy_decomposition_relative_tolerance=1e-12,
        microbatch_gradient_relative_discrepancy=relative, effective_batch_size=32,
        microbatch_sizes=[32,32] if a.checkpoint_decoders else [32,8],
        decoder_checkpointing_comparison=a.checkpoint_decoders, binding_sha256=sha256(out/'binding.json'),
        raw_sha256=sha256(out/'measurements.npz'),
        scope='Fixed native state and batch; gradient accumulation control does not certify equivalence of long finite-precision trajectories.'))
    print('Native observation checks passed; accumulation relative gradient discrepancy:', relative, flush=True)
    if relative > 1e-4:
        raise AssertionError('Accumulation differs materially in the declared gradient norm; diagnostic retained')


if __name__ == '__main__':
    main()
