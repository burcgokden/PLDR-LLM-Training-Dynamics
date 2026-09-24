"""Small deterministic statistical utilities with no SciPy dependency."""

from __future__ import annotations

import math

import numpy as np


def standardized_moments(values) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(array) < 4 or not np.isfinite(array).all():
        raise ValueError("at least four finite values are required")
    mean = float(array.mean())
    centered = array - mean
    variance = float(np.mean(centered * centered))
    if variance <= 0.0:
        raise ValueError("sample variance must be positive")
    scale = math.sqrt(variance)
    normalized = centered / scale
    return {
        "count": int(len(array)),
        "mean": mean,
        "standard_deviation": scale,
        "skewness": float(np.mean(normalized**3)),
        "excess_kurtosis": float(np.mean(normalized**4) - 3.0),
    }


def normal_ks_distance(values) -> float:
    """One-sample Kolmogorov distance to the standard normal law."""

    array = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("finite nonempty values are required")
    cdf = np.fromiter(
        (0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))
         for value in array),
        dtype=np.float64,
        count=len(array),
    )
    upper = np.arange(1, len(array) + 1, dtype=np.float64) / len(array)
    lower = np.arange(0, len(array), dtype=np.float64) / len(array)
    return float(max(np.max(np.abs(upper - cdf)), np.max(np.abs(cdf - lower))))


def wasserstein_1d(left, right) -> float:
    """Equal-mass empirical one-dimensional Wasserstein-1 distance."""

    left_array = np.sort(np.asarray(left, dtype=np.float64).reshape(-1))
    right_array = np.sort(np.asarray(right, dtype=np.float64).reshape(-1))
    if len(left_array) != len(right_array) or not len(left_array):
        raise ValueError("samples must be nonempty and have equal length")
    if not np.isfinite(left_array).all() or not np.isfinite(right_array).all():
        raise ValueError("samples must be finite")
    return float(np.mean(np.abs(left_array - right_array)))


def through_origin_fit(x, y) -> dict[str, float | bool | None]:
    """Least-squares fit ``y = slope * x`` with descriptive diagnostics."""

    x_array = np.asarray(x, dtype=np.float64).reshape(-1)
    y_array = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(x_array) < 2 or x_array.shape != y_array.shape:
        raise ValueError("aligned vectors with at least two entries are required")
    denominator = float(x_array @ x_array)
    if denominator <= 0.0:
        raise ValueError("fit abscissae must not all vanish")
    slope = float((x_array @ y_array) / denominator)
    fitted = slope * x_array
    residual = y_array - fitted
    centered_total = float(np.sum((y_array - y_array.mean()) ** 2))
    residual_sum = float(residual @ residual)
    degenerate_fit = centered_total == 0.0
    r_squared = (
        1.0 - residual_sum / centered_total
        if not degenerate_fit else None
    )
    return {
        "slope": slope,
        "root_mean_square_residual": float(math.sqrt(residual_sum / len(x_array))),
        "maximum_absolute_residual": float(np.max(np.abs(residual))),
        "r_squared_about_mean": r_squared,
        "degenerate_fit": degenerate_fit,
    }


def ordinary_linear_fit(x, y) -> dict[str, float | bool | None]:
    """Least-squares fit ``y = intercept + slope * x``.

    Report a JSON-safe null coefficient of determination when the response
    has zero total variance.  This is a descriptive fit without p-values.
    """

    x_array = np.asarray(x, dtype=np.float64).reshape(-1)
    y_array = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(x_array) < 2 or x_array.shape != y_array.shape:
        raise ValueError("aligned vectors with at least two entries are required")
    if not np.isfinite(x_array).all() or not np.isfinite(y_array).all():
        raise ValueError("fit inputs must be finite")
    design = np.column_stack([np.ones(len(x_array)), x_array])
    intercept, slope = np.linalg.lstsq(design, y_array, rcond=None)[0]
    fitted = intercept + slope * x_array
    residual = y_array - fitted
    centered_total = float(np.sum((y_array - y_array.mean()) ** 2))
    residual_sum = float(residual @ residual)
    degenerate_fit = centered_total == 0.0
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "root_mean_square_residual": float(
            math.sqrt(residual_sum / len(x_array))
        ),
        "maximum_absolute_residual": float(np.max(np.abs(residual))),
        "r_squared_about_mean": (
            1.0 - residual_sum / centered_total
            if not degenerate_fit else None
        ),
        "degenerate_fit": degenerate_fit,
    }
