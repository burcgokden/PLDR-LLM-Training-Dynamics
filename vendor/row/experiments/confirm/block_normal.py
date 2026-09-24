"""Block Lyapunov metrics and chronological forced-envelope analysis."""

from __future__ import annotations

from typing import Any

import numpy as np

from confirm.block_normal_specs import SCIENTIFIC_OUTCOMES


def _vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector")
    return result


def _matrix(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if (
        result.ndim != 2
        or result.shape[0] != result.shape[1]
        or result.size == 0
        or not np.isfinite(result).all()
    ):
        raise ValueError(f"{name} must be a finite square matrix")
    return result


def _inverse_sqrt(metric: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(
        0.5 * (metric + metric.T)
    )
    if eigenvalues[0] <= 0.0:
        raise ValueError("scheduled metric is not positive definite")
    return (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T


def backward_metrics(
    operators: list[Any] | tuple[Any, ...],
    terminal_metric: Any,
) -> dict[str, Any]:
    """Construct finite-horizon metrics and their exact scheduled gains."""

    matrices = tuple(
        _matrix(value, f"operator[{index}]")
        for index, value in enumerate(operators)
    )
    terminal = _matrix(terminal_metric, "terminal_metric")
    dimension = terminal.shape[0]
    if any(value.shape != (dimension, dimension) for value in matrices):
        raise ValueError("all block operators must share one dimension")
    metrics: list[np.ndarray] = [np.empty((0, 0))] * (len(matrices) + 1)
    metrics[-1] = 0.5 * (terminal + terminal.T)
    if np.linalg.eigvalsh(metrics[-1])[0] <= 0.0:
        raise ValueError("terminal metric is not positive definite")
    for index in range(len(matrices) - 1, -1, -1):
        operator = matrices[index]
        metrics[index] = (
            np.eye(dimension) + operator.T @ metrics[index + 1] @ operator
        )
    gains = []
    residuals = []
    conditions = []
    minimum_eigenvalues = []
    maximum_eigenvalues = []
    for index, operator in enumerate(matrices):
        current = metrics[index]
        following = metrics[index + 1]
        inverse_sqrt = _inverse_sqrt(current)
        action = inverse_sqrt @ (
            operator.T @ following @ operator
        ) @ inverse_sqrt
        maximum = max(0.0, float(np.linalg.eigvalsh(action)[-1]))
        gain = float(np.nextafter(np.sqrt(maximum), np.inf))
        residual = float(np.linalg.eigvalsh(
            operator.T @ following @ operator - gain * gain * current
        )[-1])
        eigenvalues = np.linalg.eigvalsh(current)
        gains.append(gain)
        residuals.append(residual)
        conditions.append(float(eigenvalues[-1] / eigenvalues[0]))
        minimum_eigenvalues.append(float(eigenvalues[0]))
        maximum_eigenvalues.append(float(eigenvalues[-1]))
    terminal_eigenvalues = np.linalg.eigvalsh(metrics[-1])
    minimum_eigenvalues.append(float(terminal_eigenvalues[0]))
    maximum_eigenvalues.append(float(terminal_eigenvalues[-1]))
    conditions.append(
        float(terminal_eigenvalues[-1] / terminal_eigenvalues[0])
    )
    return {
        "metrics": np.stack(metrics),
        "gains": np.asarray(gains),
        "lyapunov_residuals": np.asarray(residuals),
        "condition_numbers": np.asarray(conditions),
        "minimum_metric_eigenvalues": np.asarray(minimum_eigenvalues),
        "maximum_metric_eigenvalues": np.asarray(maximum_eigenvalues),
    }


def ordered_product_convolution(
    gains: Any, forces: Any, initial: float
) -> dict[str, Any]:
    """Compute every chronological prefix and transported force term."""

    gain = _vector(gains, "gains")
    force = _vector(forces, "forces")
    if gain.shape != force.shape:
        raise ValueError("gains and forces must have equal length")
    if (
        np.any(gain < 0.0)
        or np.any(force < 0.0)
        or not np.isfinite(initial)
        or initial < 0.0
    ):
        raise ValueError("envelope inputs must be finite and nonnegative")
    count = len(gain)
    prefix = np.ones(count + 1)
    homogeneous = np.empty(count + 1)
    homogeneous[0] = initial
    contributions = np.zeros((count + 1, count))
    envelope = np.empty(count + 1)
    envelope[0] = initial
    for block in range(count):
        prefix[block + 1] = prefix[block] * gain[block]
        homogeneous[block + 1] = prefix[block + 1] * initial
        contributions[block + 1, :block] = (
            gain[block] * contributions[block, :block]
        )
        contributions[block + 1, block] = force[block]
        envelope[block + 1] = (
            homogeneous[block + 1]
            + np.sum(contributions[block + 1])
        )
    return {
        "prefix_products": prefix,
        "homogeneous_contribution": homogeneous,
        "force_contributions": contributions,
        "forcing_convolution": np.sum(contributions, axis=1),
        "envelope": envelope,
    }


def block_prediction(
    gains: Any,
    quadratic_coefficients: Any,
    forces: Any,
    radii: Any,
    initial: float,
) -> dict[str, Any]:
    """Construct the effective affine block envelope and tube residuals."""

    gain = _vector(gains, "gains")
    quadratic = _vector(quadratic_coefficients, "quadratic_coefficients")
    force = _vector(forces, "forces")
    radius = _vector(radii, "radii")
    if (
        quadratic.shape != gain.shape
        or force.shape != gain.shape
        or radius.shape != (len(gain) + 1,)
        or np.any(gain < 0.0)
        or np.any(quadratic < 0.0)
        or np.any(force < 0.0)
        or np.any(radius < 0.0)
        or not np.isfinite(initial)
        or initial < 0.0
    ):
        raise ValueError("block prediction inputs are inconsistent")
    effective = gain + quadratic * radius[:-1]
    tube_residual = (
        gain * radius[:-1]
        + quadratic * radius[:-1] ** 2
        + force
        - radius[1:]
    )
    chronological = ordered_product_convolution(
        effective, force, initial
    )
    return {
        "source_values_only": True,
        "gains": gain,
        "quadratic_coefficients": quadratic,
        "forces": force,
        "radii": radius,
        "effective_gains": effective,
        "invariant_radius_residuals": tube_residual,
        "tube_valid": bool(initial <= radius[0]
                           and np.all(tube_residual <= 0.0)),
        **chronological,
    }


def normal_to_physical(
    normal_upper: Any,
    observability: Any,
    arithmetic_radius: Any,
) -> np.ndarray:
    """Apply the source-frozen normal-to-row diameter edge."""

    normal = _vector(normal_upper, "normal_upper")
    edge = np.broadcast_to(
        np.asarray(observability, dtype=np.float64), normal.shape
    )
    radius = np.broadcast_to(
        np.asarray(arithmetic_radius, dtype=np.float64), normal.shape
    )
    if (
        not np.isfinite(edge).all()
        or not np.isfinite(radius).all()
        or np.any(normal < 0.0)
        or np.any(edge < 0.0)
        or np.any(radius < 0.0)
    ):
        raise ValueError("normal-to-physical inputs must be nonnegative")
    return np.sqrt(2.0) * (edge * normal + radius)


def classify_coverage(
    observed_lower: Any,
    observed_upper: Any,
    predicted_lower: Any,
    predicted_upper: Any,
    *,
    tolerance: float,
) -> np.ndarray:
    """Classify resolved coverage while preserving unresolved intervals."""

    lower = _vector(observed_lower, "observed_lower")
    upper = _vector(observed_upper, "observed_upper")
    bound_lower = _vector(predicted_lower, "predicted_lower")
    bound_upper = _vector(predicted_upper, "predicted_upper")
    if not (
        lower.shape == upper.shape == bound_lower.shape == bound_upper.shape
    ):
        raise ValueError("coverage interval arrays disagree")
    if (
        np.any(lower > upper)
        or np.any(bound_lower > bound_upper)
        or not np.isfinite(tolerance)
        or tolerance < 0.0
    ):
        raise ValueError("coverage intervals or tolerance are invalid")
    outcome = np.full(lower.shape, "unresolved", dtype="<U13")
    outcome[upper <= bound_lower + tolerance] = "confirmed"
    outcome[lower > bound_upper + tolerance] = "not_confirmed"
    if any(value not in SCIENTIFIC_OUTCOMES for value in outcome):
        raise AssertionError("scientific classification is not total")
    return outcome
