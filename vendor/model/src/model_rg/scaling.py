"""Native collective observations and numerical contracts for size-time scaling."""
import math

import numpy as np
import torch


def metric_coordinates(matrices):
    """Exact row-sector coordinates, accumulated in float64 in fixed entry units.

    Input axes are context, decoder, head, row and column. No empirical
    centering across independent training realizations is performed here.
    """
    a = matrices.double()
    centroids = a.mean(-2)
    energy = (a - centroids.unsqueeze(-2)).square().mean((-2, -1))
    total = a.square().mean((-2, -1))
    return centroids, torch.stack((energy, total), -1)


def observations(model, ids, targets, matrices=False):
    """Retain the existing 36 observables and add physical common-row coordinates."""
    output = model.forward(ids, capture=True)
    logits = output.logits[:, -1]
    logp = logits.log_softmax(-1)
    probability = logp.exp()
    per_head, operators, centroids, energies, mean_matrices = [], [], [], [], []
    for layer, values in enumerate(output.pldr_attentions):
        a, _, _, _, _, g, weights = values
        attention = weights[:, :, -1]
        entropy = -(attention * attention.clamp_min(1e-38).log()).sum(-1)
        centered = a - a.mean(-2, keepdim=True)
        row = centered.square().mean((-2, -1)) / a.square().mean((-2, -1)).clamp_min(1e-30)
        per_head.append(torch.stack((entropy / math.log(ids.shape[1]), model.head_outputs[layer],
                                     row, g.square().mean((-2, -1)).sqrt()), -1))
        c, e = metric_coordinates(a)
        centroids.append(c)
        energies.append(e)
        if matrices:
            operators.append(g.detach().double())
            mean_matrices.append(a.double().mean(1))
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
    fields = torch.cat((layer_means[:, :, 0] * math.log(ids.shape[1]), layer_means[:, :, 1], hidden,
                        logits @ model._common_logit_projection, entropy[:, None], nll[:, None],
                        layer_means[:, :, 2], layer_means[:, :, 3]), 1)
    extra = dict(centroids=torch.stack(centroids, 1), energies=torch.stack(energies, 1))
    if matrices:
        extra['mean_matrices'] = torch.stack(mean_matrices, 1)
    return fields, heads, logits, operators, extra


def collective_moments(values, heads):
    """Seed-conditional moments of a fixed intensive vector at fixed contexts.

    The vector axes after (seed, context) are averaged with equal fixed weights.
    Susceptibility uses the unbiased whole-seed covariance divisor. The fourth
    moment ratio uses empirical central moments with a common population divisor.
    """
    q = np.asarray(values, dtype=np.float64)
    if q.ndim < 2 or q.shape[0] < 2 or not np.isfinite(q).all():
        raise ValueError('Finite observations from at least two independent seeds are required')
    d = q - q.mean(0, keepdims=True)
    m2 = np.mean(d * d, axis=0)
    m4 = np.mean(d ** 4, axis=0)
    second = float(m2.mean())
    fourth = float(m4.mean())
    return dict(mean=float(q.mean()), susceptibility=float(heads * second * len(q) / (len(q)-1)),
                central_second=second, central_fourth=fourth,
                fourth_ratio=fourth/(second*second) if second > 0 else None,
                independent_seeds=len(q), contexts=q.shape[1])
