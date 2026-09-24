"""Finite physical observables and explicit exponent-transfer bounds.

All lattices have periodic boundaries. The q=2 Potts Hamiltonian is equivalent
to the usual Ising Hamiltonian after a factor-two temperature conversion.
"""
from __future__ import annotations
import math
import numpy as np


def critical_temperature(q):
    if q not in (2, 3):
        raise ValueError("Only the two specified continuous-transition classes")
    return 1.0 / math.log1p(math.sqrt(q))


def observables(configurations, q):
    x = np.asarray(configurations)
    if x.ndim != 3 or x.shape[-1] != x.shape[-2] or not np.isin(x, range(q)).all():
        raise ValueError("Expected valid square-lattice configurations")
    L = x.shape[-1]
    fractions = np.stack([(x == a).mean((1, 2)) for a in range(q)], axis=1)
    m2 = np.maximum(0, (q * np.square(fractions).sum(1) - 1) / (q - 1))
    bonds = ((x == np.roll(x, 1, axis=1)).sum((1, 2)) +
             (x == np.roll(x, 1, axis=2)).sum((1, 2)))
    answer = dict(m=np.sqrt(m2), m2=m2, m4=m2*m2, energy=-bonds.astype(float),
                  field=(q * fractions[:, 0] - 1) / (q - 1))
    for r in sorted(set([1, max(1, L//4), max(1, L//2)])):
        same = ((x == np.roll(x, r, axis=1)).mean((1, 2)) +
                (x == np.roll(x, r, axis=2)).mean((1, 2))) / 2
        answer[f'corr{r}'] = (q * same - 1) / (q - 1)
    return answer


def summarize(configurations, q):
    obs = observables(configurations, q)
    L = configurations.shape[-1]
    means = {k: float(v.mean()) for k, v in obs.items()}
    means['chi'] = L*L*means['m2']
    means['binder_ratio'] = means['m4'] / means['m2']**2
    return means


def exact_small(q, L, temperature):
    volume = L*L
    if q**volume > 100000:
        raise ValueError("Exact enumeration exceeds the declared finite budget")
    ids = np.arange(q**volume, dtype=np.int64)
    x = np.stack([(ids // q**j) % q for j in range(volume)], 1).reshape(-1, L, L)
    obs = observables(x, q)
    logw = -obs['energy']/temperature
    weights = np.exp(logw-logw.max());weights /= weights.sum()
    return {k: float(weights@v) for k,v in obs.items()}


def dyadic_block(configurations, q, seed=0):
    x = np.asarray(configurations)
    n,L,L2 = x.shape
    if L != L2 or L % 2:
        raise ValueError("Even square lattice required")
    cells = x.reshape(n,L//2,2,L//2,2).transpose(0,1,3,2,4).reshape(n,L//2,L//2,4)
    counts = np.stack([(cells==a).sum(-1) for a in range(q)],axis=-1)
    # Independent uniform tie breaking defines a stochastic, color-symmetric map.
    tied = counts == counts.max(-1,keepdims=True)
    rng=np.random.default_rng(seed)
    uniforms=rng.random(tied.shape)
    return np.argmax(np.where(tied,uniforms,-1),axis=-1).astype(np.uint8)


def slope_bound(relative_error, scale_ratio):
    """Two-size log-slope error under |observed/true-1| <= epsilon < 1."""
    if not 0 <= relative_error < 1 or not scale_ratio > 1:
        raise ValueError("Require epsilon in [0,1) and scale ratio > 1")
    return math.log((1+relative_error)/(1-relative_error))/math.log(scale_ratio)


def prefix_tv_bound(mean_prefix_kl, sites):
    if mean_prefix_kl < 0 or sites < 1:
        raise ValueError("Nonnegative divergence and positive site count required")
    return min(1.0, math.sqrt(sites*mean_prefix_kl/2))
