"""Covariance-prioritized coordinates within a fixed empirical response span."""
import numpy as np


def covariance_rotation(images, basis):
    """Order the fixed source span by calibration innovation covariance.

    All returned rows are orthonormal. The last row can contain calibration
    drift outside the centered-noise span. Retaining every row therefore
    recovers the same complete calibration response span as the total basis.
    """
    y = np.asarray(images, dtype=np.float64).reshape(len(images), -1)
    b = np.asarray(basis, dtype=np.float64)
    coefficients = y @ b.T
    centered = coefficients-coefficients.mean(axis=0)
    covariance = centered.T @ centered/(len(y)-1)
    eigenvalues, directions = np.linalg.eigh(covariance)
    scale = max(float(np.max(np.abs(eigenvalues))), 1e-300)
    if np.min(eigenvalues) < -1e-12*scale:
        raise AssertionError('The calibration covariance is not positive semidefinite')
    order = np.argsort(eigenvalues)[::-1]
    return directions[:, order].T, np.maximum(eigenvalues[order], 0)
