"""Finite conditional statistics for native single-pass criticality observations."""
import numpy as np


def conditional_statistics(values):
    """Values have independent-seed, fixed-context, layer, and head axes.

    The sample covariance uses S-1. Moment ratios describe the empirical,
    context-centered distribution and are not bias-corrected population cumulants.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 4 or x.shape[0] < 2 or min(x.shape[1:]) < 1 or not np.isfinite(x).all():
        raise ValueError('Expected finite seed/context/layer/head observations')
    seeds, _, _, heads = x.shape
    centered = x-x.mean(0, keepdims=True)
    q = x.mean(-1)
    dq = q-q.mean(0, keepdims=True)
    variance = np.mean(dq*dq)*seeds/(seeds-1)
    single = np.mean(centered*centered)*seeds/(seeds-1)
    chi = heads*variance
    cross = (chi-single)/(heads-1) if heads > 1 else 0.
    m2 = float(np.mean(dq**2))
    m4 = float(np.mean(dq**4))
    return dict(mean=float(x.mean()), mean_by_seed=q.mean((1, 2)).tolist(),
                susceptibility=float(chi), intensive_variance=float(variance),
                one_head_variance=float(single), cross_head_covariance=float(cross),
                enhancement=float(chi/single) if single > 0 else None,
                empirical_centered_moment_ratio=float(m4/(m2*m2)) if m2 > 0 else None,
                empirical_binder=float(1-m4/(3*m2*m2)) if m2 > 0 else None,
                pooled_centered_second_moment=m2, pooled_centered_fourth_moment=m4,
                covariance_identity_residual=float(chi-single-(heads-1)*cross),
                all_entry_minimum=float(x.min()), all_entry_maximum=float(x.max()))


def seed_uncertainty(values, draws=1000, seed=9152590):
    """Resample full training realizations, preserving all context/head dependence."""
    x = np.asarray(values, dtype=np.float64)
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    q = x.mean(-1)
    indices = rng.integers(0, n, size=(draws, n))
    boot = q[indices]
    chi = x.shape[-1]*np.var(boot, axis=1, ddof=1).mean((1, 2))
    means = boot.mean((1, 2, 3))
    omission = np.array([conditional_statistics(np.delete(x, k, axis=0))['susceptibility']
                         for k in range(n)]) if n >= 3 else np.array([])
    return dict(training_replicas=n, bootstrap_draws=draws,
                susceptibility_percentile_95=np.quantile(chi, [.025, .975]).tolist(),
                mean_percentile_95=np.quantile(means, [.025, .975]).tolist(),
                susceptibility_leave_one_seed=omission.tolist(),
                scope='Descriptive finite-seed bootstrap; no contexts, heads or times are resampled as independent training replicas.')


def peak_profile(controls, responses):
    g = np.asarray(controls, dtype=float)
    y = np.asarray(responses, dtype=float)
    if g.ndim != 1 or y.shape != g.shape or len(g) < 3 or not np.all(np.diff(g) > 0):
        raise ValueError('Expected an ordered control grid')
    if not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError('Expected finite nonnegative susceptibility')
    k = int(np.argmax(y))
    half = .5*y[k]
    left = right = None
    if y[k] > 0:
        for i in range(k-1, -1, -1):
            if y[i] <= half < y[i+1]:
                left = float(g[i]+(half-y[i])*(g[i+1]-g[i])/(y[i+1]-y[i])); break
        for i in range(k, len(g)-1):
            if y[i] > half >= y[i+1]:
                right = float(g[i]+(half-y[i])*(g[i+1]-g[i])/(y[i+1]-y[i])); break
    return dict(sampled_maximum_control=float(g[k]), sampled_maximum=float(y[k]),
                interior=k not in [0, len(g)-1], half_left=left, half_right=right,
                half_width=right-left if left is not None and right is not None else None,
                scope='Grid maximum and piecewise-linear half-height crossings; no inferred infinite-size critical point.')
