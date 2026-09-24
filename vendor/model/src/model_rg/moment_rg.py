"""Finite conditional moment maps and non-Gaussian mean/covariance scores."""
import numpy as np


def moments(samples):
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('At least two finite vector observations are required')
    mean = x.mean(axis=0)
    centered = x - mean
    return mean, centered.T @ centered / (len(x) - 1)


def transport(mean, covariance, block):
    b = np.asarray(block, dtype=np.float64)
    return b @ mean, b @ covariance @ b.T


def score(samples, mean, covariance):
    """log det Sigma + squared Mahalanobis distance, no Gaussian assumption."""
    x = np.asarray(samples, dtype=np.float64)
    chol = np.linalg.cholesky(covariance)
    z = np.linalg.solve(chol, (x - mean).T)
    return 2 * np.log(np.diag(chol)).sum() + (z * z).sum(axis=0)


def forecast(samples, shrinkage=0.25, floor=0.001):
    if not 0 <= shrinkage <= 1 or floor <= 0:
        raise ValueError('Shrinkage must lie in [0,1] and floor must be positive')
    mean, covariance = moments(samples)
    scale = np.maximum(np.sqrt(np.diag(covariance)), 1e-6)
    standardized = covariance / np.outer(scale, scale)
    diagonal = np.diag(np.diag(standardized))
    ridge = floor * np.eye(len(scale))
    return dict(mean=mean.tolist(), scale=scale.tolist(),
                covariance=covariance.tolist(),
                regularized=((1-shrinkage)*standardized+shrinkage*diagonal+ridge).tolist(),
                diagonal=(diagonal+ridge).tolist())


def block_maps(observables):
    """Four chronological increments per observable, same maps at every state."""
    pair = np.array([[1., 1., 0., 0.], [0., 0., 1., 1.]])
    return dict(fine=np.eye(4*observables),
                paired=np.kron(np.eye(observables), pair),
                endpoint=np.kron(np.eye(observables), np.ones((1, 4))))
