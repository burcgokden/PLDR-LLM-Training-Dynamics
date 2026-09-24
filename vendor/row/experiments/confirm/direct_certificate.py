"""Numerical kernels for the complete direct row-map certificate.

Every function consumes explicit finite arrays and returns the quantities
printed in the manuscript. The functions do not infer an infinite-tail
claim from sampled states. They evaluate the deterministic inequalities that
constitute that claim.
"""

from __future__ import annotations

import math

import numpy as np


def _array(value, name, *, ndim=None):
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains a nonfinite value")
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have dimension {ndim}")
    return result


def _square(value, name):
    result = _array(value, name, ndim=2)
    if result.shape[0] == 0 or result.shape[0] != result.shape[1]:
        raise ValueError(f"{name} must be nonempty and square")
    return result


def _scalar(value, name, *, lower=None, strict_lower=False):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if lower is not None:
        if strict_lower and not result > lower:
            raise ValueError(f"{name} must be greater than {lower}")
        if not strict_lower and result < lower:
            raise ValueError(f"{name} must be at least {lower}")
    return result


def _symmetric_positive(metric, name="metric"):
    metric = _square(metric, name)
    if not np.allclose(metric, metric.T, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    metric = 0.5 * (metric + metric.T)
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    if eigenvalues[0] <= 0.0:
        raise ValueError(f"{name} must be positive definite")
    return metric, eigenvalues, eigenvectors


def metric_operator_norm(matrix, metric):
    """Return the induced H-norm of a square matrix."""
    matrix = _square(matrix, "matrix")
    metric, eigenvalues, eigenvectors = _symmetric_positive(metric)
    if matrix.shape != metric.shape:
        raise ValueError("matrix and metric shapes disagree")
    root = (
        eigenvectors
        @ np.diag(np.sqrt(eigenvalues))
        @ eigenvectors.T
    )
    inverse_root = (
        eigenvectors
        @ np.diag(1.0 / np.sqrt(eigenvalues))
        @ eigenvectors.T
    )
    transformed = root @ matrix @ inverse_root
    return float(np.linalg.svd(transformed, compute_uv=False)[0])


def solve_discrete_lyapunov(nominal_matrix, q_matrix=None):
    """Construct H from H - A.T H A = Q and certify the nominal gain."""
    matrix = _square(nominal_matrix, "nominal_matrix")
    dimension = matrix.shape[0]
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(matrix))))
    if not spectral_radius < 1.0:
        raise ValueError("nominal_matrix must be strictly Schur stable")
    if q_matrix is None:
        q_matrix = np.eye(dimension)
    q_matrix, q_eigenvalues, _ = _symmetric_positive(q_matrix, "q_matrix")
    if q_matrix.shape != matrix.shape:
        raise ValueError("q_matrix and nominal_matrix shapes disagree")
    operator = (
        np.eye(dimension * dimension)
        - np.kron(matrix.T, matrix.T)
    )
    vector = np.linalg.solve(
        operator, q_matrix.reshape(-1, order="F")
    )
    metric = vector.reshape((dimension, dimension), order="F")
    metric = 0.5 * (metric + metric.T)
    metric, metric_eigenvalues, _ = _symmetric_positive(metric)
    residual = metric - matrix.T @ metric @ matrix - q_matrix
    residual_scale = max(
        float(np.linalg.norm(metric, ord=2)),
        float(np.linalg.norm(q_matrix, ord=2)),
        1e-30,
    )
    q0_formula = math.sqrt(max(
        0.0,
        1.0 - float(q_eigenvalues[0] / metric_eigenvalues[-1]),
    ))
    exact_gain = metric_operator_norm(matrix, metric)
    q0_bound = max(q0_formula, exact_gain)
    if q0_bound > 0.0:
        q0_bound = math.nextafter(q0_bound, math.inf)
    return {
        "metric": metric,
        "q_matrix": q_matrix,
        "spectral_radius": spectral_radius,
        "metric_lower": float(metric_eigenvalues[0]),
        "metric_upper": float(metric_eigenvalues[-1]),
        "q_matrix_lower": float(q_eigenvalues[0]),
        "nominal_q_bound": q0_bound,
        "nominal_exact_h_gain": exact_gain,
        "relative_residual": float(np.linalg.norm(
            residual, ord=2) / residual_scale),
    }


def robust_family_certificate(nominal_matrix, family, metric, nominal_q,
                              *, euclidean_radius=None):
    """Extend one nominal metric to a finite or interval-vertex family."""
    nominal = _square(nominal_matrix, "nominal_matrix")
    metric, metric_eigenvalues, _ = _symmetric_positive(metric)
    if metric.shape != nominal.shape:
        raise ValueError("metric and nominal_matrix shapes disagree")
    q0 = _scalar(nominal_q, "nominal_q", lower=0.0)
    if not q0 < 1.0:
        raise ValueError("nominal_q must be below one")
    matrices = [_square(value, "family_matrix") for value in family]
    if not matrices:
        raise ValueError("family must contain at least one matrix")
    if any(value.shape != nominal.shape for value in matrices):
        raise ValueError("family matrix shapes disagree")
    euclidean_differences = [
        float(np.linalg.norm(value - nominal, ord=2))
        for value in matrices
    ]
    observed_epsilon = max(euclidean_differences)
    if euclidean_radius is None:
        epsilon = observed_epsilon
    else:
        epsilon = _scalar(
            euclidean_radius, "euclidean_radius", lower=0.0)
        if observed_epsilon > epsilon + 1e-12 * max(1.0, epsilon):
            raise ValueError(
                "certified Euclidean radius does not enclose the family")
    condition_factor = math.sqrt(
        float(metric_eigenvalues[-1] / metric_eigenvalues[0])
    )
    delta = condition_factor * epsilon
    exact_gains = [
        metric_operator_norm(value, metric) for value in matrices
    ]
    robust_q = max(q0 + delta, max(exact_gains))
    if robust_q > 0.0:
        robust_q = math.nextafter(robust_q, math.inf)
    return {
        "euclidean_family_radius": epsilon,
        "observed_euclidean_family_radius": observed_epsilon,
        "certified_euclidean_family_radius": epsilon,
        "metric_condition_factor": condition_factor,
        "metric_family_radius": delta,
        "nominal_q": q0,
        "robust_q": robust_q,
        "strictly_contracting": bool(robust_q < 1.0),
        "exact_family_h_gains": exact_gains,
        "maximum_exact_h_gain": max(exact_gains),
        "observed_gain_excess": max(exact_gains) - robust_q,
        "euclidean_differences": euclidean_differences,
    }


def lifted_block_matrix(normal_operator, decay_operator, *,
                        alpha, beta1):
    """Build the complete homogeneous block in the manuscript."""
    normal = _square(normal_operator, "normal_operator")
    dimension = normal.shape[0]
    decay = np.asarray(decay_operator, dtype=float)
    if decay.ndim == 0:
        decay = float(decay) * np.eye(dimension)
    decay = _square(decay, "decay_operator")
    if decay.shape != normal.shape:
        raise ValueError("normal and decay operator shapes disagree")
    alpha = _scalar(alpha, "alpha", lower=0.0)
    beta1 = _scalar(beta1, "beta1", lower=0.0)
    if not beta1 < 1.0:
        raise ValueError("beta1 must be below one")
    identity = np.eye(dimension)
    gain = (1.0 - beta1) * normal
    return np.block([
        [
            identity - decay - alpha * gain,
            -alpha * beta1 * identity,
        ],
        [gain, beta1 * identity],
    ])


def complete_defect_vector(*, alpha, beta1, normal_residual,
                           decay_defect, row_motion_defect,
                           intervention_defect, taylor_remainder,
                           coordinate_motion_defect):
    """Assemble the two-block defect exactly as in the lifted recurrence."""
    alpha = _scalar(alpha, "alpha", lower=0.0)
    beta1 = _scalar(beta1, "beta1", lower=0.0)
    if not beta1 < 1.0:
        raise ValueError("beta1 must be below one")
    values = {
        "normal_residual": _array(normal_residual, "normal_residual", ndim=1),
        "decay_defect": _array(decay_defect, "decay_defect", ndim=1),
        "row_motion_defect": _array(
            row_motion_defect, "row_motion_defect", ndim=1),
        "intervention_defect": _array(
            intervention_defect, "intervention_defect", ndim=1),
        "taylor_remainder": _array(
            taylor_remainder, "taylor_remainder", ndim=1),
        "coordinate_motion_defect": _array(
            coordinate_motion_defect, "coordinate_motion_defect", ndim=1),
    }
    shapes = {value.shape for value in values.values()}
    if len(shapes) != 1:
        raise ValueError("all primitive defects must have the same shape")
    normal = values["normal_residual"]
    first = (
        -alpha * (1.0 - beta1) * normal
        + values["decay_defect"]
        + values["row_motion_defect"]
        + values["intervention_defect"]
        + values["taylor_remainder"]
    )
    second = (
        (1.0 - beta1) * normal
        + values["coordinate_motion_defect"]
    )
    return np.concatenate([first, second])


def normal_residual_envelope(*, displacement_norm,
                             linear_residual_bound,
                             second_derivative_bound):
    """Evaluate the primitive-coordinate Taylor residual bound."""
    displacement = _scalar(
        displacement_norm, "displacement_norm", lower=0.0)
    linear = _scalar(
        linear_residual_bound, "linear_residual_bound", lower=0.0)
    second = _scalar(
        second_derivative_bound, "second_derivative_bound", lower=0.0)
    return linear * displacement + 0.5 * second * displacement ** 2


def componentwise_self_map_budget(*, q, lifted_radius, disturbance_bound,
                                  next_lifted_radius, center_errors,
                                  lifted_lipschitz, auxiliary_lipschitz,
                                  auxiliary_radii, next_auxiliary_radii):
    """Evaluate one lifted transition and every successor-radius slack.

    A single graph edge may be expansive. Strict contraction is therefore
    imposed on complete registered windows, not on this local transition.
    """
    q = _scalar(q, "q", lower=0.0)
    lifted_radius = _scalar(
        lifted_radius, "lifted_radius", lower=0.0)
    disturbance = _scalar(
        disturbance_bound, "disturbance_bound", lower=0.0)
    next_lifted = _scalar(
        next_lifted_radius, "next_lifted_radius", lower=0.0)
    center = _array(center_errors, "center_errors", ndim=1)
    ell = _array(lifted_lipschitz, "lifted_lipschitz", ndim=1)
    matrix = _array(
        auxiliary_lipschitz, "auxiliary_lipschitz", ndim=2)
    radii = _array(auxiliary_radii, "auxiliary_radii", ndim=1)
    next_radii = _array(
        next_auxiliary_radii, "next_auxiliary_radii", ndim=1)
    dimension = len(center)
    if dimension == 0:
        raise ValueError("at least one auxiliary component is required")
    if (
        ell.shape != (dimension,)
        or matrix.shape != (dimension, dimension)
        or radii.shape != (dimension,)
        or next_radii.shape != (dimension,)
    ):
        raise ValueError("auxiliary budget dimensions disagree")
    if (
        (center < 0.0).any()
        or (ell < 0.0).any()
        or (matrix < 0.0).any()
        or (radii < 0.0).any()
        or (next_radii < 0.0).any()
    ):
        raise ValueError("radius budget data must be nonnegative")
    lifted_required = q * lifted_radius + disturbance
    auxiliary_required = (
        center + ell * lifted_radius + matrix @ radii
    )
    lifted_slack = next_lifted - lifted_required
    auxiliary_slacks = next_radii - auxiliary_required
    return {
        "lifted_required": lifted_required,
        "lifted_slack": lifted_slack,
        "auxiliary_required": auxiliary_required,
        "auxiliary_slacks": auxiliary_slacks,
        "minimum_auxiliary_slack": float(np.min(auxiliary_slacks)),
        "certified": bool(
            lifted_slack >= 0.0 and np.all(auxiliary_slacks >= 0.0)
        ),
    }


def geometric_convolution(n, q, r):
    """Return sum_{k=0}^{n-1} q^(n-1-k) r^k without cancellation."""
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    q = _scalar(q, "q", lower=0.0)
    r = _scalar(r, "r", lower=0.0)
    if not q < 1.0 or not r < 1.0:
        raise ValueError("q and r must be below one")
    return float(sum(q ** (n - 1 - k) * r ** k for k in range(n)))


def nonautonomous_schedule_bounds(*, gains, disturbance_bounds,
                                  initial_h_norm, metric_lower,
                                  cover_floor, cover_transient_constant,
                                  cover_rate, criterion):
    """Unroll y_(t+1) <= q_t y_t + w_t and project to direct bounds."""
    gains = _array(gains, "gains", ndim=1)
    disturbances = _array(
        disturbance_bounds, "disturbance_bounds", ndim=1)
    if len(gains) == 0 or gains.shape != disturbances.shape:
        raise ValueError(
            "gains and disturbance_bounds must be nonempty and aligned")
    if (gains < 0.0).any():
        raise ValueError("every schedule gain must be nonnegative")
    if (disturbances < 0.0).any():
        raise ValueError("schedule disturbances must be nonnegative")
    initial = _scalar(
        initial_h_norm, "initial_h_norm", lower=0.0)
    metric = _scalar(
        metric_lower, "metric_lower", lower=0.0, strict_lower=True)
    floor = _scalar(cover_floor, "cover_floor", lower=0.0)
    cover_constant = _scalar(
        cover_transient_constant,
        "cover_transient_constant",
        lower=0.0,
    )
    rho = _scalar(cover_rate, "cover_rate", lower=0.0)
    if not rho < 1.0:
        raise ValueError("cover_rate must be below one")
    threshold = _scalar(
        criterion, "criterion", lower=0.0, strict_lower=True)

    lifted = [initial]
    product = [1.0]
    for gain, disturbance in zip(gains, disturbances):
        product.append(product[-1] * float(gain))
        lifted.append(
            float(gain) * lifted[-1] + float(disturbance))
    direct = [
        value / math.sqrt(metric)
        + floor
        + cover_constant * rho ** index
        for index, value in enumerate(lifted)
    ]
    first_sustained = None
    for index in range(len(direct)):
        if all(value < threshold for value in direct[index:]):
            first_sustained = index
            break
    return {
        "gains": gains.tolist(),
        "disturbance_bounds": disturbances.tolist(),
        "cumulative_products": product,
        "lifted_bounds": lifted,
        "direct_bounds": direct,
        "final_product_convolution_upper": lifted[-1],
        "first_sustained_relative_entry": first_sustained,
        "criterion": threshold,
    }


def deductive_output_order_bridge(*, output_a, output_b, rows_a, rows_b,
                                  direct_row_map_upper,
                                  downstream_lipschitz=1.0,
                                  mean_floor=None):
    """Evaluate the one-way bridge from direct collapse to PLDR output RMSE.

    The output arrays use their first axis for aligned samples. The row arrays
    contain the corresponding row-map inputs. mean_floor is a certified
    positive lower bound on the absolute reference-output mean; when omitted,
    the observed absolute mean is used.
    """
    first = _array(output_a, "output_a")
    second = _array(output_b, "output_b")
    if first.shape != second.shape or first.size == 0:
        raise ValueError("output arrays must be nonempty and aligned")
    if first.ndim == 0:
        raise ValueError("output arrays must have a sample axis")
    row_first = _array(rows_a, "rows_a", ndim=2)
    row_second = _array(rows_b, "rows_b", ndim=2)
    if (
        row_first.shape != row_second.shape
        or row_first.shape[0] == 0
        or row_first.shape[0] != first.shape[0]
    ):
        raise ValueError("row arrays must be nonempty and sample-aligned")
    direct = _scalar(
        direct_row_map_upper, "direct_row_map_upper", lower=0.0)
    downstream = _scalar(
        downstream_lipschitz, "downstream_lipschitz", lower=0.0)
    rmse = float(np.sqrt(np.mean((first - second) ** 2)))
    mean_abs = abs(float(np.mean(second)))
    if not mean_abs > 0.0:
        raise ValueError("reference output mean magnitude must be positive")
    if mean_floor is None:
        floor = mean_abs
    else:
        floor = _scalar(
            mean_floor, "mean_floor", lower=0.0, strict_lower=True)
        if floor > mean_abs:
            raise ValueError(
                "mean_floor must not exceed the observed mean magnitude")
    row_spread = float(np.sqrt(np.mean(np.sum(
        (row_first - row_second) ** 2, axis=1))))
    order_parameter = rmse / mean_abs
    upper = downstream * direct * row_spread / floor
    if upper == 0.0:
        ratio = 0.0 if order_parameter == 0.0 else float(np.finfo(float).max)
    else:
        ratio = order_parameter / upper
    return {
        "source_rmse": rmse,
        "tensor_mean_abs": mean_abs,
        "mean_floor": floor,
        "row_input_spread": row_spread,
        "downstream_lipschitz": downstream,
        "direct_row_map_upper": direct,
        "source_order_parameter": order_parameter,
        "order_parameter_upper": upper,
        "order_parameter_bridge_ratio": ratio,
        "bridge_holds": bool(ratio <= 1.0),
    }


def direct_bound_at(n, *, q, r, initial_h_norm, forcing_constant,
                    metric_lower, cover_floor,
                    cover_transient_constant, cover_rate,
                    persistent_forcing=0.0):
    """Evaluate the complete direct upper bound at a nonnegative horizon."""
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    q = _scalar(q, "q", lower=0.0)
    r = _scalar(r, "r", lower=0.0)
    rho = _scalar(cover_rate, "cover_rate", lower=0.0)
    if not q < 1.0 or not r < 1.0 or not rho < 1.0:
        raise ValueError("all geometric rates must be below one")
    y0 = _scalar(initial_h_norm, "initial_h_norm", lower=0.0)
    forcing = _scalar(
        forcing_constant, "forcing_constant", lower=0.0)
    persistent = _scalar(
        persistent_forcing, "persistent_forcing", lower=0.0)
    h_lower = _scalar(
        metric_lower, "metric_lower", lower=0.0, strict_lower=True)
    floor = _scalar(cover_floor, "cover_floor", lower=0.0)
    cover_constant = _scalar(
        cover_transient_constant,
        "cover_transient_constant",
        lower=0.0,
    )
    persistent_partial = persistent * (1.0 - q ** n) / (1.0 - q)
    lifted = (
        q ** n * y0
        + persistent_partial
        + forcing * geometric_convolution(n, q, r)
    )
    transient = cover_constant * rho ** n
    complete_floor = floor + persistent / (
        (1.0 - q) * math.sqrt(h_lower))
    return {
        "lifted_h_bound": lifted,
        "persistent_lifted_partial": persistent_partial,
        "complete_persistent_floor": complete_floor,
        "cover_transient": transient,
        "direct_upper": lifted / math.sqrt(h_lower) + floor + transient,
    }


def monotone_entry_horizon(*, q, r, initial_h_norm, forcing_constant,
                           metric_lower, cover_floor,
                           cover_transient_constant, cover_rate,
                           criterion, persistent_forcing=0.0,
                           search_limit=10_000_000):
    """Compute the manuscript's monotone all-future entry horizon."""
    q = _scalar(q, "q", lower=0.0)
    r = _scalar(r, "r", lower=0.0)
    rho = _scalar(cover_rate, "cover_rate", lower=0.0)
    if not q < 1.0 or not r < 1.0 or not rho < 1.0:
        raise ValueError("all geometric rates must be below one")
    y0 = _scalar(initial_h_norm, "initial_h_norm", lower=0.0)
    forcing = _scalar(
        forcing_constant, "forcing_constant", lower=0.0)
    persistent = _scalar(
        persistent_forcing, "persistent_forcing", lower=0.0)
    h_lower = _scalar(
        metric_lower, "metric_lower", lower=0.0, strict_lower=True)
    floor = _scalar(cover_floor, "cover_floor", lower=0.0)
    cover_constant = _scalar(
        cover_transient_constant,
        "cover_transient_constant",
        lower=0.0,
    )
    criterion = _scalar(
        criterion, "criterion", lower=0.0, strict_lower=True)
    complete_floor = floor + persistent / (
        (1.0 - q) * math.sqrt(h_lower))
    if not complete_floor < criterion:
        raise ValueError(
            "complete persistent floor must be strictly below criterion")
    if (
        not isinstance(search_limit, int)
        or isinstance(search_limit, bool)
        or search_limit < 1
    ):
        raise ValueError("search_limit must be a positive integer")
    s = max(q, r)
    monotonicity_start = max(
        1, int(math.ceil(s / (1.0 - s)))
    )
    delta = criterion - complete_floor

    def envelope(n):
        equal_rate_majorant = n * s ** (n - 1)
        return (
            (
                q ** n * y0
                + forcing * equal_rate_majorant
            )
            / math.sqrt(h_lower)
            + cover_constant * rho ** n
        )

    for n in range(monotonicity_start, search_limit + 1):
        if envelope(n) < delta:
            exact = direct_bound_at(
                n,
                q=q,
                r=r,
                initial_h_norm=y0,
                forcing_constant=forcing,
                persistent_forcing=persistent,
                metric_lower=h_lower,
                cover_floor=floor,
                cover_transient_constant=cover_constant,
                cover_rate=rho,
            )
            return {
                "horizon": n,
                "monotonicity_start": monotonicity_start,
                "monotone_envelope": envelope(n),
                "strict_margin": delta - envelope(n),
                "exact_bound_at_horizon": exact["direct_upper"],
                "criterion": criterion,
                "cover_floor": floor,
                "persistent_forcing": persistent,
                "complete_persistent_floor": complete_floor,
            }
    raise RuntimeError("horizon search exceeded search_limit")
