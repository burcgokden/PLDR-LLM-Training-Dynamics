"""Declared normalized family and observations for criticality experiments."""
import hashlib
import math
import numpy as np
import torch
from model_rg.controlled import stable_kl


def generator_parameter(name):
    return any(part in name for part in ('reslayerAs', 'plgatt_layer', 'layernormA'))


def normalize_initialization(model):
    width = model.config.hidden_size
    with torch.no_grad():
        for name, module in model.model.named_modules():
            if isinstance(module, torch.nn.Linear) and not generator_parameter(name):
                reference = 128 if module.in_features == width else (128 * 8) // 3
                if module.in_features in (width, (width * 8) // 3):
                    module.weight.mul_(math.sqrt(reference / module.in_features))



def fix_shared_generator(model, seed):
    """Condition the head family on its fixed-dimensional shared metric network."""
    from model_rg.training import TrainingModel
    with torch.random.fork_rng(devices=[model.device.index] if model.device.type == 'cuda' else []):
        reference = TrainingModel(model.path, 2, seed, 'cpu')
    shared = {n:p.detach() for n,p in reference.model.named_parameters() if 'reslayerAs' in n}
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, parameter in model.model.named_parameters():
            if name in shared:
                parameter.copy_(shared[name].to(parameter.device))
                digest.update(name.encode())
                digest.update(shared[name].contiguous().numpy().tobytes())
    del reference
    return digest.hexdigest()


def optimizer_for(model, multiplier):
    named = list(model.model.named_parameters())
    groups = [dict(params=[p for n, p in named if generator_parameter(n)], lr=3e-4 * multiplier),
              dict(params=[p for n, p in named if not generator_parameter(n)],
                   lr=3e-4 * 128 / model.config.hidden_size)]
    return torch.optim.AdamW(groups, betas=(.9, .95), eps=1e-8,
                             weight_decay=.01, foreach=False)


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def sample_batches(tokens, seed, steps):
    # Separate generators preserve exactly the same prefix at every horizon.
    rows = np.random.default_rng(seed + 1000).integers(0, 3072, size=(steps, 32))
    offsets = np.random.default_rng(seed + 1001000).integers(0, 449, size=(steps, 32))
    return rows, offsets, tokens[rows[:, :, None], offsets[:, :, None] + np.arange(65)]


def observations(model, ids, targets, matrices=False):
    """Keep the per-head fields as well as the established 36 common fields."""
    output = model.forward(ids, capture=True)
    logits = output.logits[:, -1]
    logp = logits.log_softmax(-1)
    probability = logp.exp()
    per_head, operators = [], []
    for layer, values in enumerate(output.pldr_attentions):
        a, _, _, _, _, g, weights = values
        attention = weights[:, :, -1]
        entropy = -(attention * attention.clamp_min(1e-38).log()).sum(-1)
        centered = a - a.mean(-2, keepdim=True)
        row = centered.square().mean((-2, -1)) / a.square().mean((-2, -1)).clamp_min(1e-30)
        per_head.append(torch.stack([entropy / math.log(ids.shape[1]), model.head_outputs[layer],
                                     row, g.square().mean((-2, -1)).sqrt()], -1))
        if matrices:
            operators.append(g.detach().double())
    heads = torch.stack(per_head, 1)
    layer_means = heads.mean(2)
    hidden = torch.stack([x[:, -1].square().mean(-1).sqrt() for x in output.hidden_states], 1)
    if not hasattr(model, '_common_logit_projection'):
        projection = np.random.default_rng(630061).normal(size=(model.config.vocab_size, 8))
        projection /= np.sqrt(model.config.vocab_size)
        projection -= projection.mean(0, keepdims=True)
        model._common_logit_projection = torch.tensor(projection, device=logits.device, dtype=logits.dtype)
    nll = -logp.gather(1, targets[:, None]).squeeze(1)
    entropy = -(probability * logp).sum(-1)
    fields = torch.cat([layer_means[:, :, 0] * math.log(ids.shape[1]), layer_means[:, :, 1], hidden,
                        logits @ model._common_logit_projection, entropy[:, None], nll[:, None],
                        layer_means[:, :, 2], layer_means[:, :, 3]], 1)
    return fields, heads, logits, operators


def predictive_kl(first, second):
    """Stable small increments and overflow-safe large logit differences."""
    first, second = first.double(), second.double()
    logp = first.log_softmax(-1)
    probability = logp.exp()
    ordinary = (probability * (logp - second.log_softmax(-1))).sum(-1)
    small = (second - first).abs().amax(-1) < .1
    if small.any():
        ordinary[small] = stable_kl(first[small], second[small])
    return ordinary


def conditional_covariances(head_fields):
    """Unbiased replica covariance, conditioning on each fixed context.

    Input is (independent training seeds, contexts, layers, heads).
    The one-head reference and the uniform-head susceptibility use the same
    seed centering. The reference is averaged over independent layerwise head
    permutations; its cross-layer entries are susceptibility / head count. The context sector is kept separately and is descriptive
    for the fixed cohort. No document is counted as an independent seed.
    """
    values = np.asarray(head_fields, dtype=np.float64)
    if values.ndim != 4 or values.shape[0] < 2:
        raise ValueError('At least two complete training realizations are required')
    seeds, contexts, layers, heads = values.shape
    centered = values - values.mean(0, keepdims=True)
    collective = centered.mean(-1)
    susceptibility = heads * np.einsum('sxl,sxm->lm', collective, collective) / ((seeds - 1) * contexts)
    # Average the one-head reference over independent head relabelings in
    # each layer. This removes arbitrary cross-layer head-index alignment.
    diagonal = susceptibility / heads
    per_layer = np.einsum('sxla,sxla->l', centered, centered) / ((seeds - 1) * contexts * heads)
    np.fill_diagonal(diagonal, per_layer)
    mean_by_context = values.mean((0, 3))
    residual = mean_by_context - mean_by_context.mean(0, keepdims=True)
    context = heads * residual.T @ residual / contexts
    return susceptibility, diagonal, context
