"""Finite data-mixture observations; no criticality inference is built in."""
import numpy as np


def mixture_stream(technical, narrative, uniforms, rho):
    """Couple every mixture by common Bernoulli uniforms and domain permutations."""
    technical = np.asarray(technical); narrative = np.asarray(narrative)
    uniforms = np.asarray(uniforms)
    if not 0 <= rho <= 1 or uniforms.ndim != 2 or uniforms.shape[1] != 32:
        raise ValueError('Invalid mixture or batch shape')
    choose = uniforms.reshape(-1) < rho
    nt = int(choose.sum()); nn = len(choose)-nt
    if nt > len(technical) or nn > len(narrative): raise ValueError('Exhausted domain resource')
    blocks = np.empty(len(choose), dtype=np.int64)
    blocks[choose] = technical[:nt]; blocks[~choose] = narrative[:nn]
    if len(np.unique(blocks)) != len(blocks): raise ValueError('Repeated block in a continuation')
    return blocks.reshape(uniforms.shape)


def paired_order(sse, reference_sum, reference_sumsq, entries):
    """Report the original mean normalization and a separately named RMS ratio."""
    sse, reference_sum, reference_sumsq = map(np.asarray, (sse, reference_sum, reference_sumsq))
    if entries <= 0 or np.any(sse < 0) or np.any(reference_sumsq < 0):
        raise ValueError('Invalid sufficient sums')
    rms = np.sqrt(sse/entries)
    mean = reference_sum/entries
    reference_rms = np.sqrt(reference_sumsq/entries)
    mean_ratio = np.divide(rms, np.abs(mean), out=np.zeros_like(rms, dtype=float), where=mean != 0)
    rms_ratio = np.divide(rms, reference_rms, out=np.zeros_like(rms, dtype=float), where=reference_rms != 0)
    return dict(rmse=rms, mean=mean, reference_rms=reference_rms,
                mean_ratio=mean_ratio, mean_defined=mean != 0,
                rms_ratio=rms_ratio, rms_defined=reference_rms != 0)


def quadratic_mixture(left, middle, right, rho):
    """Endpoint interpolation plus the separately fitted midpoint interaction."""
    left, middle, right = map(np.asarray, (left, middle, right))
    if left.shape != middle.shape or left.shape != right.shape: raise ValueError('Shape mismatch')
    return (1-rho)*left+rho*right+4*rho*(1-rho)*(middle-(left+right)/2)


def centered_gram(paths):
    """Exact finite empirical response Gram with feature coordinates in columns."""
    x = np.asarray(paths, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all(): raise ValueError('Finite matrix required')
    return x@x.T/x.shape[1]


def covariance_split(values):
    """Within and between group covariance for an equally weighted finite law."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 3: raise ValueError('Expected group, draw and coordinate axes')
    means=x.mean(1); within=x-means[:,None]; between=means-means.mean(0)
    cw=np.einsum('gni,gnj->ij',within,within)/(x.shape[0]*x.shape[1])
    cb=between.T@between/x.shape[0]
    return cw,cb
