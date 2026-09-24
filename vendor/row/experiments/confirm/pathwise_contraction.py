"""Structured primitive-family and pathwise contraction certificates."""

from __future__ import annotations

from fractions import Fraction
from itertools import product
import math

import numpy as np

from direct_certificate import (
    _array,
    _scalar,
    lifted_block_matrix,
    metric_operator_norm,
    solve_discrete_lyapunov,
)
from validated_interval import as_fraction
from validated_linear_algebra import (
    convex_family_lyapunov_certificate,
    graph_window_certificate,
)


PRIMITIVE_NAMES = ("alpha", "decay", "normal_eigenvalue")


def cross_metric_family_certificate(vertices, source_metric, target_metric,
                                    search_iterations=96):
    """Find a replayable exact gain for a source-to-target metric edge.

    The exact feasibility test is monotone in the squared gain. Bisection is
    therefore performed on rational candidates. Unlike a one-cell search,
    this constructor permits an individual cross-metric edge gain above one.
    """
    if (
        not isinstance(search_iterations, int)
        or isinstance(search_iterations, bool)
        or search_iterations < 1
    ):
        raise ValueError("search_iterations must be a positive integer")

    def attempt(candidate):
        try:
            return convex_family_lyapunov_certificate(
                vertices,
                source_metric,
                target_metric=target_metric,
                gain_squared=candidate,
            )
        except ValueError:
            return None

    lower = Fraction(0)
    upper = Fraction(1)
    certificate = attempt(upper)
    for _ in range(32):
        if certificate is not None:
            break
        lower = upper
        upper *= 2
        certificate = attempt(upper)
    if certificate is None:
        raise ValueError("cross-metric edge gain search did not terminate")

    for _ in range(search_iterations):
        midpoint = (lower + upper) / 2
        candidate = attempt(midpoint)
        if candidate is None:
            lower = midpoint
        else:
            upper = midpoint
            certificate = candidate
    return certificate


def _bounds(value):
    if not isinstance(value, dict) or set(value) != set(PRIMITIVE_NAMES):
        raise ValueError("primitive bounds have missing or unknown coordinates")
    result = {}
    for name in PRIMITIVE_NAMES:
        interval = value[name]
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError(f"{name} bounds must have two endpoints")
        lower = float(interval[0])
        upper = float(interval[1])
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise ValueError(f"{name} bounds are invalid")
        if lower < 0.0:
            raise ValueError(f"{name} bounds must be nonnegative")
        result[name] = (lower, upper)
    return result


def lifted_mode_primitives(matrix, beta1):
    """Invert the scalar lifted-mode parameterization."""

    matrix = _array(matrix, "lifted mode matrix", ndim=2)
    if matrix.shape != (2, 2):
        raise ValueError("a scalar lifted mode must be two dimensional")
    beta1 = _scalar(beta1, "beta1", lower=0.0)
    if not beta1 < 1.0 or abs(matrix[1, 1] - beta1) > 1e-12:
        raise ValueError("lifted mode does not carry the declared beta1")
    alpha = -float(matrix[0, 1]) / beta1 if beta1 > 0.0 else 0.0
    normal = float(matrix[1, 0]) / (1.0 - beta1)
    decay = 1.0 - alpha * (1.0 - beta1) * normal - float(matrix[0, 0])
    values = {
        "alpha": alpha,
        "decay": decay,
        "normal_eigenvalue": normal,
    }
    if any(value < -1e-12 or not math.isfinite(value)
           for value in values.values()):
        raise ValueError("lifted mode has invalid primitive coordinates")
    return {name: max(0.0, value) for name, value in values.items()}


def primitive_box_from_matrices(matrices, beta1):
    points = [lifted_mode_primitives(matrix, beta1) for matrix in matrices]
    if not points:
        raise ValueError("a primitive family must be nonempty")
    return {
        name: [min(point[name] for point in points),
               max(point[name] for point in points)]
        for name in PRIMITIVE_NAMES
    }


def multiaffine_corner_matrices(primitive_bounds, beta1):
    bounds = _bounds(primitive_bounds)
    beta1 = _scalar(beta1, "beta1", lower=0.0)
    if not beta1 < 1.0:
        raise ValueError("beta1 must lie below one")
    vertices = []
    coordinates = []
    endpoint_lists = [bounds[name] for name in PRIMITIVE_NAMES]
    for alpha, decay, normal in product(*endpoint_lists):
        coordinates.append({
            "alpha": alpha,
            "decay": decay,
            "normal_eigenvalue": normal,
        })
        vertices.append(lifted_block_matrix(
            np.asarray([[normal]], dtype=float),
            decay,
            alpha=alpha,
            beta1=beta1,
        ))
    return coordinates, vertices


def multiaffine_weights(primitive_bounds, point):
    bounds = _bounds(primitive_bounds)
    if not isinstance(point, dict) or set(point) != set(PRIMITIVE_NAMES):
        raise ValueError("primitive point has missing or unknown coordinates")
    axis_weights = []
    for name in PRIMITIVE_NAMES:
        lower, upper = bounds[name]
        value = float(point[name])
        tolerance = 1e-12 * max(1.0, abs(lower), abs(upper))
        if value < lower - tolerance or value > upper + tolerance:
            raise ValueError(f"{name} lies outside its primitive box")
        value = min(upper, max(lower, value))
        if upper == lower:
            axis_weights.append((1.0, 0.0))
        else:
            high = (value - lower) / (upper - lower)
            axis_weights.append((1.0 - high, high))
    return np.asarray([
        first * second * third
        for first, second, third in product(*axis_weights)
    ], dtype=float)


def structured_family_certificate(nominal_matrix, family_matrices, beta1,
                                  q_matrix=None, window_length=1):
    """Certify the multiaffine primitive box and its finite window."""

    nominal = _array(nominal_matrix, "nominal_matrix", ndim=2)
    family = [_array(matrix, "family_matrix", ndim=2)
              for matrix in family_matrices]
    if not family or any(matrix.shape != nominal.shape for matrix in family):
        raise ValueError("structured family matrices are empty or misdimensioned")
    bounds = primitive_box_from_matrices(family, beta1)
    coordinates, corners = multiaffine_corner_matrices(bounds, beta1)
    construction = solve_discrete_lyapunov(nominal, q_matrix)
    metric = construction["metric"]
    exact = convex_family_lyapunov_certificate(corners, metric)
    exact_gain = float(as_fraction(exact["gain_upper"]))
    corner_gains = [metric_operator_norm(matrix, metric) for matrix in corners]
    observed_gains = [metric_operator_norm(matrix, metric) for matrix in family]
    reconstruction_residuals = []
    enclosure_ratios = []
    for matrix in family:
        point = lifted_mode_primitives(matrix, beta1)
        weights = multiaffine_weights(bounds, point)
        reconstructed = sum(
            (weight * corner for weight, corner in zip(weights, corners)),
            np.zeros_like(matrix),
        )
        scale = max(float(np.linalg.norm(matrix, ord=2)), 1e-30)
        reconstruction_residuals.append(
            float(np.linalg.norm(matrix - reconstructed, ord=2) / scale))
        enclosure_ratios.extend(
            0.0 if bounds[name][1] == bounds[name][0]
            else abs(point[name] - sum(bounds[name]) / 2.0)
            / ((bounds[name][1] - bounds[name][0]) / 2.0)
            for name in PRIMITIVE_NAMES
        )
    if (
        not isinstance(window_length, int)
        or isinstance(window_length, bool)
        or window_length < 1
    ):
        raise ValueError("window_length must be a positive integer")
    edge_gain = as_fraction(exact["gain_upper"])
    graph = graph_window_certificate([{
        "edge_id": "tail_cell",
        "source": "tail_cell",
        "target": "tail_cell",
        "gain_upper": exact["gain_upper"],
    }], window_length)
    window_gain = float(as_fraction(graph["maximum_window_gain"]))
    observed_max = max(observed_gains)
    corner_max = max(corner_gains)
    return {
        "schema_version": "pldr-structured-path-family-v1",
        "primitive_bounds": bounds,
        "primitive_corner_coordinates": coordinates,
        "corner_matrices": [matrix.tolist() for matrix in corners],
        "construction": construction,
        "exact_vertex_certificate": exact,
        "graph_window_certificate": graph,
        "structured_q": exact_gain,
        "structured_q_squared": float(edge_gain * edge_gain),
        "corner_h_gains": corner_gains,
        "maximum_corner_h_gain": corner_max,
        "family_h_gains": observed_gains,
        "maximum_observed_h_gain": observed_max,
        "maximum_reconstruction_residual": max(reconstruction_residuals),
        "maximum_primitive_box_ratio": max(enclosure_ratios, default=0.0),
        "window_length": window_length,
        "maximum_window_gain": window_gain,
        "strictly_contracting": bool(exact_gain < 1.0),
    }


def path_product_convolution(gains, disturbances, initial):
    """Unroll a forced path while allowing expanding individual edges."""

    gains = _array(gains, "gains", ndim=1)
    disturbances = _array(disturbances, "disturbances", ndim=1)
    if gains.shape != disturbances.shape or not len(gains):
        raise ValueError("gains and disturbances must be nonempty and aligned")
    if (gains < 0.0).any() or (disturbances < 0.0).any():
        raise ValueError("path gains and disturbances must be nonnegative")
    initial = _scalar(initial, "initial", lower=0.0)
    values = [initial]
    transition_products = [1.0]
    for gain, disturbance in zip(gains, disturbances):
        values.append(float(gain) * values[-1] + float(disturbance))
        transition_products.append(transition_products[-1] * float(gain))
    return {
        "gains": gains.tolist(),
        "disturbances": disturbances.tolist(),
        "bounds": values,
        "prefix_gain_products": transition_products,
        "terminal_bound": values[-1],
    }
