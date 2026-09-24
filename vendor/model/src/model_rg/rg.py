"""Coarse-graining operators, cumulants, and model-wide response geometry."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp


def block_sum(x, b: int, exponent: float = .5):
    """Aligned blocking, without silently dropping an incomplete block."""
    x = np.asarray(x)
    if not isinstance(b, int) or b < 1 or len(x) % b:
        raise ValueError("b must be positive and divide the sample count")
    return x.reshape(len(x)//b, b, *x.shape[1:]).sum(1) / b**exponent


def connected_moments(x):
    x = np.asarray(x, dtype=np.float64)
    y = x - x.mean(0)
    v = np.mean(y*y, 0)
    if np.any(v <= 0):
        raise ValueError("Cumulant standardization requires positive variance")
    return {"mean": x.mean(0), "variance": v,
            "skewness": np.mean(y**3, 0)/v**1.5,
            "excess_kurtosis": np.mean(y**4, 0)/v**2-3}


def empirical_cgf(x, sources):
    x, sources = np.asarray(x), np.asarray(sources)
    return logsumexp(x @ sources.T, axis=0)-np.log(len(x))


def tilted_mean(x, source):
    score = np.asarray(x) @ source
    weights = np.exp(score-logsumexp(score))
    return weights @ x


@dataclass(frozen=True)
class AffineJet:
    """delta x_out = A delta x_in + B eta (same eta at all stages)."""
    A: np.ndarray
    B: np.ndarray

    def after(self, earlier: "AffineJet") -> "AffineJet":
        return AffineJet(self.A @ earlier.A, self.A @ earlier.B + self.B)

    def act(self, x, source):
        return self.A @ x + self.B @ source


def fisher_pullback(logits, jacobian):
    """J.T (diag(p)-p p.T) J; no vocabulary-square allocation.

    logits: [..., vocabulary], jacobian: [..., vocabulary, source].
    """
    z, j = np.asarray(logits, dtype=np.float64), np.asarray(jacobian, dtype=np.float64)
    p = np.exp(z-logsumexp(z, axis=-1, keepdims=True))
    mean = np.einsum("...v,...vk->...k", p, j)
    return np.einsum("...v,...vi,...vj->...ij", p, j, j)-mean[..., :, None]*mean[..., None, :]


def gaussian_schur(precision, retained):
    """Integrate Gaussian coordinates; return the exact effective precision."""
    q = np.asarray(precision, dtype=np.float64)
    keep = np.asarray(retained, dtype=int)
    if q.ndim != 2 or q.shape[0] != q.shape[1] or len(np.unique(keep)) != len(keep):
        raise ValueError("Expected square precision and distinct retained coordinates")
    np.linalg.cholesky(q)
    drop = np.setdiff1d(np.arange(len(q)), keep)
    aa, ab, bb = q[np.ix_(keep, keep)], q[np.ix_(keep, drop)], q[np.ix_(drop, drop)]
    return aa-ab @ np.linalg.solve(bb, ab.T) if len(drop) else aa.copy()


def finite_susceptibility(x, b):
    """b Cov(block average), preserving actual order and all cross terms."""
    y = block_sum(x, b, exponent=.5)
    return np.cov(y, rowvar=False, ddof=1)
