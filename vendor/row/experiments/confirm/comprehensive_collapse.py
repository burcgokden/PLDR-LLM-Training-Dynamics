"""Numerical kernels for the comprehensive row-map collapse certificate.

The functions in this module are independent of checkpoint I/O.  They define
the exact finite-dimensional objects used by the live producer and by the
independent analyzer: finite-stencil normals, selected pairwise softmax
frames, the true cross-entropy Hessian split, the full three-block AdamW
successor Jacobian, scheduled path metrics, variable-radius envelopes, and
oriented end-to-end application bounds.
"""

from __future__ import annotations

import math
import platform
import sys
from typing import Iterable, Sequence

import numpy as np


def _finite_array(value, name, *, ndim=None):
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a nonfinite value")
    return array


def finite_stencil_stack(phi, rows, directions, step):
    """Evaluate ``(phi(r + h v) - phi(r)) / h`` in registry order."""

    rows = _finite_array(rows, "rows", ndim=2)
    directions = _finite_array(directions, "directions", ndim=2)
    if rows.shape != directions.shape:
        raise ValueError("rows and directions must have identical shapes")
    step = float(step)
    if not math.isfinite(step) or step <= 0:
        raise ValueError("stencil step must be finite and positive")
    outputs = []
    for row, direction in zip(rows, directions):
        norm = float(np.linalg.norm(direction))
        if not np.isclose(norm, 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError("every stencil direction must be unit length")
        base = _finite_array(phi(row), "phi(row)")
        displaced = _finite_array(
            phi(row + step * direction), "phi(row + h direction)")
        if base.shape != displaced.shape:
            raise ValueError("row-map output shape changed across a stencil")
        outputs.append(((displaced - base) / step).reshape(-1))
    if not outputs:
        raise ValueError("the finite stencil cannot be empty")
    return np.concatenate(outputs)


def stencil_direct_bound(
        maximum_stencil, second_row_derivative, stencil_step,
        direction_net_radius, row_net_radius, numerical_error=0.0):
    """Use a joint row-direction cover to bound the uniform derivative."""

    values = [
        maximum_stencil,
        second_row_derivative,
        stencil_step,
        direction_net_radius,
        row_net_radius,
        numerical_error,
    ]
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("stencil bound inputs must be finite")
    if any(float(value) < 0 for value in values):
        raise ValueError("stencil bound inputs must be nonnegative")
    if direction_net_radius >= 1:
        raise ValueError("the direction-net radius must be below one")
    return (
        float(maximum_stencil)
        + float(numerical_error)
        + 0.5 * float(second_row_derivative) * float(stencil_step)
        + float(second_row_derivative) * float(row_net_radius)
    ) / (1.0 - float(direction_net_radius))


def joint_row_direction_cover(
        registered_rows, registered_directions, query_rows, query_directions,
        *, row_scale, block_size=256):
    """Measure a deterministic joint cover by registered row-direction pairs."""

    registered_rows = _finite_array(
        registered_rows, "registered_rows", ndim=2)
    registered_directions = _finite_array(
        registered_directions, "registered_directions", ndim=2)
    query_rows = _finite_array(query_rows, "query_rows", ndim=2)
    query_directions = _finite_array(
        query_directions, "query_directions", ndim=2)
    if registered_rows.shape != registered_directions.shape:
        raise ValueError("registered rows and directions must have one shape")
    if query_rows.shape != query_directions.shape:
        raise ValueError("query rows and directions must have one shape")
    if registered_rows.shape[1:] != query_rows.shape[1:]:
        raise ValueError("registered and query dimensions disagree")
    if not len(registered_rows) or not len(query_rows):
        raise ValueError("joint cover registries cannot be empty")
    row_scale = float(row_scale)
    if not math.isfinite(row_scale) or row_scale <= 0:
        raise ValueError("joint-cover row scale must be finite and positive")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("joint-cover block size must be positive")
    for name, directions in (
            ("registered", registered_directions),
            ("query", query_directions)):
        norms = np.linalg.norm(directions, axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError(f"{name} joint-cover directions must be unit length")

    chosen = []
    row_radii = []
    direction_radii = []
    for start in range(0, len(query_rows), block_size):
        stop = min(start + block_size, len(query_rows))
        row_distance = np.linalg.norm(
            query_rows[start:stop, None, :] - registered_rows[None, :, :],
            axis=2)
        direction_distance = np.linalg.norm(
            query_directions[start:stop, None, :]
            - registered_directions[None, :, :], axis=2)
        score = row_distance / row_scale + direction_distance
        index = np.argmin(score, axis=1)
        local = np.arange(stop - start)
        chosen.append(index)
        row_radii.append(row_distance[local, index])
        direction_radii.append(direction_distance[local, index])
    chosen = np.concatenate(chosen).astype(np.int64, copy=False)
    row_radii = np.concatenate(row_radii)
    direction_radii = np.concatenate(direction_radii)
    return {
        "selected_registered_index": chosen,
        "row_radii": row_radii,
        "direction_radii": direction_radii,
        "maximum_row_radius": float(np.max(row_radii)),
        "maximum_direction_radius": float(np.max(direction_radii)),
        "row_scale": row_scale,
    }


def softmax_fisher(probability):
    probability = _finite_array(probability, "probability", ndim=1)
    if np.any(probability <= 0):
        raise ValueError("softmax probabilities must be strictly positive")
    if not np.isclose(probability.sum(), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("softmax probabilities must sum to one")
    return np.diag(probability) - np.outer(probability, probability)


def pairwise_softmax_energy(probability, vector):
    """Return the exact complete-graph weighted Laplacian energy."""

    probability = _finite_array(probability, "probability", ndim=1)
    vector = _finite_array(vector, "vector", ndim=1)
    if probability.shape != vector.shape:
        raise ValueError("probability and vector shapes disagree")
    softmax_fisher(probability)
    difference = vector[:, None] - vector[None, :]
    weights = probability[:, None] * probability[None, :]
    return float(np.sum(np.triu(weights * difference ** 2, k=1)))


def _validated_edges(edges, vocabulary_size):
    result = []
    seen = set()
    for raw_left, raw_right in edges:
        left = int(raw_left)
        right = int(raw_right)
        if left == right:
            raise ValueError("a selected softmax edge must join two vertices")
        if not (0 <= left < vocabulary_size and 0 <= right < vocabulary_size):
            raise ValueError("a selected softmax edge is outside the vocabulary")
        edge = (min(left, right), max(left, right))
        if edge in seen:
            raise ValueError("selected softmax edges must be unique per source")
        seen.add(edge)
        result.append(edge)
    if not result:
        raise ValueError("every source needs at least one selected edge")
    return result


def selected_pair_frame(
        logit_jacobians, probabilities, edge_sets, *, weights=None,
        lower_products=None):
    """Construct exact and selected-lower Fisher-normal Gram matrices.

    ``logit_jacobians[s]`` has shape ``(vocabulary, normal_dimension)``.
    ``lower_products[s][e]`` is a sealed lower bound for the probability
    product on selected edge ``e``.  If omitted, the exact product is used.
    """

    jacobians = _finite_array(
        logit_jacobians, "logit_jacobians", ndim=3)
    probabilities = _finite_array(probabilities, "probabilities", ndim=2)
    source_count, vocabulary_size, normal_dimension = jacobians.shape
    if probabilities.shape != (source_count, vocabulary_size):
        raise ValueError("probability and logit-Jacobian registries disagree")
    if len(edge_sets) != source_count:
        raise ValueError("edge-set registry has the wrong source count")
    if weights is None:
        weights = np.full(source_count, 1.0 / source_count)
    weights = _finite_array(weights, "weights", ndim=1)
    if weights.shape != (source_count,) or np.any(weights <= 0):
        raise ValueError("source weights must be positive and complete")
    if not np.isclose(weights.sum(), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("source weights must sum to one")
    if lower_products is not None and len(lower_products) != source_count:
        raise ValueError("lower-product registry has the wrong source count")

    exact = np.zeros((normal_dimension, normal_dimension), dtype=np.float64)
    lower = np.zeros_like(exact)
    realized_products = []
    for source in range(source_count):
        probability = probabilities[source]
        fisher = softmax_fisher(probability)
        jacobian = jacobians[source]
        exact += weights[source] * jacobian.T @ fisher @ jacobian
        edges = _validated_edges(edge_sets[source], vocabulary_size)
        source_products = []
        if lower_products is None:
            source_lower = None
        else:
            source_lower = _finite_array(
                lower_products[source], "lower_products", ndim=1)
            if source_lower.shape != (len(edges),):
                raise ValueError("an edge and lower-product count disagree")
        for edge_index, (left, right) in enumerate(edges):
            product = float(probability[left] * probability[right])
            source_products.append(product)
            bound = product if source_lower is None else float(source_lower[edge_index])
            if bound < 0 or bound > product * (1.0 + 1e-12):
                raise ValueError("a selected probability-product bound is invalid")
            difference = jacobian[left] - jacobian[right]
            lower += weights[source] * bound * np.outer(difference, difference)
        realized_products.append(source_products)
    return {
        "exact": (exact + exact.T) / 2,
        "lower": (lower + lower.T) / 2,
        "realized_products": realized_products,
    }


def full_cross_entropy_response(
        logit_jacobians, probabilities, targets, logit_second_jets,
        *, weights=None):
    """Return Fisher, signed second-jet, and true-loss normal Hessians."""

    jacobians = _finite_array(
        logit_jacobians, "logit_jacobians", ndim=3)
    probabilities = _finite_array(probabilities, "probabilities", ndim=2)
    targets = _finite_array(targets, "targets", ndim=2)
    second = _finite_array(
        logit_second_jets, "logit_second_jets", ndim=4)
    sources, vocabulary, normal = jacobians.shape
    if probabilities.shape != (sources, vocabulary):
        raise ValueError("probability registry shape is invalid")
    if targets.shape != probabilities.shape:
        raise ValueError("target registry shape is invalid")
    if second.shape != (sources, vocabulary, normal, normal):
        raise ValueError("second-logit-jet registry shape is invalid")
    if weights is None:
        weights = np.full(sources, 1.0 / sources)
    weights = _finite_array(weights, "weights", ndim=1)
    if weights.shape != (sources,) or np.any(weights < 0):
        raise ValueError("loss weights are invalid")
    if not np.isclose(weights.sum(), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("loss weights must sum to one")

    fisher = np.zeros((normal, normal), dtype=np.float64)
    residual = np.zeros_like(fisher)
    for source in range(sources):
        probability = probabilities[source]
        hessian = softmax_fisher(probability)
        jacobian = jacobians[source]
        fisher += weights[source] * jacobian.T @ hessian @ jacobian
        signed = probability - targets[source]
        residual += weights[source] * np.einsum(
            "v,vij->ij", signed, second[source])
    fisher = (fisher + fisher.T) / 2
    residual = (residual + residual.T) / 2
    return {"fisher": fisher, "residual": residual, "full": fisher + residual}


def relative_curvature_edge(pair_edge, residual, fisher, *, tolerance=1e-12):
    """Compute the least relative charge and remaining true-loss edge."""

    fisher = _finite_array(fisher, "fisher", ndim=2)
    residual = _finite_array(residual, "residual", ndim=2)
    if fisher.shape != residual.shape or fisher.shape[0] != fisher.shape[1]:
        raise ValueError("curvature matrices must be square and aligned")
    values, vectors = np.linalg.eigh((fisher + fisher.T) / 2)
    if values[0] <= tolerance:
        raise ValueError("relative curvature requires a positive Fisher frame")
    inverse_root = vectors @ np.diag(values ** -0.5) @ vectors.T
    relative = inverse_root @ ((residual + residual.T) / 2) @ inverse_root
    least = float(np.linalg.eigvalsh((relative + relative.T) / 2)[0])
    chi = max(0.0, -least)
    full_edge = float(np.linalg.eigvalsh(fisher + residual)[0])
    certified = (1.0 - chi) * float(pair_edge)
    return {
        "relative_least_eigenvalue": least,
        "chi": chi,
        "relative_bound_edge": certified,
        "full_edge": full_edge,
    }


def vector_bernstein_radius(variance, almost_sure_bound, batch_size,
                            dimension, failure_probability):
    values = [variance, almost_sure_bound, failure_probability]
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("Bernstein inputs must be finite")
    if variance < 0 or almost_sure_bound < 0:
        raise ValueError("Bernstein scale inputs must be nonnegative")
    if batch_size <= 0 or dimension <= 0:
        raise ValueError("Bernstein batch and dimension must be positive")
    if not 0 < failure_probability < 1:
        raise ValueError("failure probability must lie strictly between zero and one")
    logarithm = math.log(2.0 * (dimension + 1) / failure_probability)
    return (
        math.sqrt(2.0 * float(variance) * logarithm / batch_size)
        + 2.0 * float(almost_sure_bound) * logarithm / (3.0 * batch_size)
    )


def adamw_successor_jacobian(
        gradient_jacobian, clipped_gradient, first_moment, second_moment,
        *, learning_rate, beta1, beta2, epsilon, optimizer_step,
        weight_decay, clip_derivative=None, decay_mask=None):
    """Differentiate the live clipped-AdamW successor in operation order.

    The lifted state is ``(normal_position, first_moment, second_moment)``.
    Bias-correction clocks, clipping branch, learning rate, and decay mask
    are fixed discrete data of the smooth successor cell.
    """

    gradient_jacobian = _finite_array(
        gradient_jacobian, "gradient_jacobian", ndim=2)
    clipped_gradient = _finite_array(
        clipped_gradient, "clipped_gradient", ndim=1)
    first_moment = _finite_array(first_moment, "first_moment", ndim=1)
    second_moment = _finite_array(second_moment, "second_moment", ndim=1)
    dimension = clipped_gradient.size
    if gradient_jacobian.shape != (dimension, dimension):
        raise ValueError("gradient Jacobian has the wrong shape")
    if first_moment.shape != (dimension,) or second_moment.shape != (dimension,):
        raise ValueError("Adam moments have the wrong shape")
    if np.any(second_moment < 0):
        raise ValueError("Adam second moments must be nonnegative")
    if optimizer_step <= 0:
        raise ValueError("optimizer step must be positive")
    scalars = [learning_rate, beta1, beta2, epsilon, weight_decay]
    if any(not math.isfinite(float(value)) for value in scalars):
        raise ValueError("AdamW coefficients must be finite")
    if learning_rate < 0 or epsilon <= 0 or weight_decay < 0:
        raise ValueError("AdamW rate, epsilon, and decay are outside their domains")
    if not (0 <= beta1 < 1 and 0 <= beta2 < 1):
        raise ValueError("AdamW beta coefficients must lie in [0, 1)")
    if clip_derivative is None:
        clip_derivative = np.eye(dimension)
    clip_derivative = _finite_array(
        clip_derivative, "clip_derivative", ndim=2)
    if clip_derivative.shape != (dimension, dimension):
        raise ValueError("clip derivative has the wrong shape")
    if decay_mask is None:
        decay_mask = np.ones(dimension)
    decay_mask = _finite_array(decay_mask, "decay_mask", ndim=1)
    if decay_mask.shape != (dimension,):
        raise ValueError("decay mask has the wrong shape")

    clipped_jacobian = clip_derivative @ gradient_jacobian
    first_next = beta1 * first_moment + (1.0 - beta1) * clipped_gradient
    second_next = (
        beta2 * second_moment
        + (1.0 - beta2) * clipped_gradient ** 2
    )
    correction1 = 1.0 - beta1 ** optimizer_step
    correction2 = 1.0 - beta2 ** optimizer_step
    first_hat = first_next / correction1
    second_hat = second_next / correction2
    if np.any(second_hat <= 0):
        raise ValueError(
            "the differentiated Adam denominator needs a positive live second moment")
    root = np.sqrt(second_hat)
    denominator = root + epsilon
    derivative_first = np.diag(1.0 / (correction1 * denominator))
    derivative_second = np.diag(
        -first_hat / (2.0 * correction2 * root * denominator ** 2)
    )

    first_normal = (1.0 - beta1) * clipped_jacobian
    second_normal = (
        2.0 * (1.0 - beta2)
        * np.diag(clipped_gradient)
        @ clipped_jacobian
    )
    identity = np.eye(dimension)
    top_normal = (
        identity
        - learning_rate * weight_decay * np.diag(decay_mask)
        - learning_rate * (
            derivative_first @ first_normal
            + derivative_second @ second_normal
        )
    )
    top_first = -learning_rate * derivative_first * beta1
    top_second = -learning_rate * derivative_second * beta2
    zeros = np.zeros_like(identity)
    operator = np.block([
        [top_normal, top_first, top_second],
        [first_normal, beta1 * identity, zeros],
        [second_normal, zeros, beta2 * identity],
    ])
    return {
        "operator": operator,
        "first_moment_next": first_next,
        "second_moment_next": second_next,
        "first_moment_hat": first_hat,
        "second_moment_hat": second_hat,
        "update_derivative_first": derivative_first,
        "update_derivative_second": derivative_second,
    }


def metric_gain(operator, source_metric, target_metric):
    operator = _finite_array(operator, "operator", ndim=2)
    source = _finite_array(source_metric, "source_metric", ndim=2)
    target = _finite_array(target_metric, "target_metric", ndim=2)
    if operator.shape[1] != source.shape[0] or operator.shape[0] != target.shape[0]:
        raise ValueError("operator and path metrics have incompatible shapes")
    source_values, source_vectors = np.linalg.eigh((source + source.T) / 2)
    if source_values[0] <= 0:
        raise ValueError("source path metric must be positive definite")
    if np.linalg.eigvalsh((target + target.T) / 2)[0] <= 0:
        raise ValueError("target path metric must be positive definite")
    inverse_root = source_vectors @ np.diag(source_values ** -0.5) @ source_vectors.T
    comparison = inverse_root @ operator.T @ target @ operator @ inverse_root
    return math.sqrt(max(0.0, float(np.linalg.eigvalsh(
        (comparison + comparison.T) / 2)[-1])))


def scheduled_path_metrics(operators, terminal_metric=None):
    """Construct an explicit backward finite-horizon Lyapunov schedule."""

    operators = [_finite_array(value, "operator", ndim=2) for value in operators]
    if not operators:
        raise ValueError("a path metric schedule needs at least one operator")
    dimension = operators[0].shape[0]
    if any(value.shape != (dimension, dimension) for value in operators):
        raise ValueError("all scheduled operators must be square and aligned")
    if terminal_metric is None:
        terminal = np.eye(dimension)
    else:
        terminal = _finite_array(terminal_metric, "terminal_metric", ndim=2)
        if terminal.shape != (dimension, dimension):
            raise ValueError("terminal metric has the wrong shape")
        if np.linalg.eigvalsh((terminal + terminal.T) / 2)[0] <= 0:
            raise ValueError("terminal metric must be positive definite")
    metrics = [None] * (len(operators) + 1)
    metrics[-1] = (terminal + terminal.T) / 2
    identity = np.eye(dimension)
    for index in range(len(operators) - 1, -1, -1):
        value = identity + operators[index].T @ metrics[index + 1] @ operators[index]
        metrics[index] = (value + value.T) / 2
    gains = [
        metric_gain(operator, metrics[index], metrics[index + 1])
        for index, operator in enumerate(operators)
    ]
    return {"metrics": metrics, "gains": np.asarray(gains)}


def robust_path_gains(nominal_operators, live_operators, metrics):
    nominal = [_finite_array(value, "nominal_operator", ndim=2)
               for value in nominal_operators]
    live = [_finite_array(value, "live_operator", ndim=2)
            for value in live_operators]
    if len(nominal) != len(live) or len(metrics) != len(nominal) + 1:
        raise ValueError("operator and metric schedules have incompatible lengths")
    rows = []
    for index, (base, actual) in enumerate(zip(nominal, live)):
        base_gain = metric_gain(base, metrics[index], metrics[index + 1])
        perturbation_gain = metric_gain(
            actual - base, metrics[index], metrics[index + 1])
        rows.append((base_gain, perturbation_gain, base_gain + perturbation_gain))
    return np.asarray(rows)


def variable_radius_envelope(
        initial, gains, quadratic, forces, radii, *, tolerance=1e-12,
        require_invariant=True):
    """Evaluate or diagnose the variable-radius product-convolution bound."""

    gains = _finite_array(gains, "gains", ndim=1)
    quadratic = _finite_array(quadratic, "quadratic", ndim=1)
    forces = _finite_array(forces, "forces", ndim=1)
    radii = _finite_array(radii, "radii", ndim=1)
    count = gains.size
    if quadratic.shape != (count,) or forces.shape != (count,):
        raise ValueError("recurrence schedules have incompatible lengths")
    if radii.shape != (count + 1,):
        raise ValueError("the radius schedule needs one boundary per state")
    if initial < 0 or np.any(gains < 0) or np.any(quadratic < 0):
        raise ValueError("recurrence amplitudes and coefficients must be nonnegative")
    if np.any(forces < 0) or np.any(radii < 0):
        raise ValueError("forces and radii must be nonnegative")
    initial_inside = bool(initial <= radii[0] + tolerance)
    image = gains * radii[:-1] + quadratic * radii[:-1] ** 2 + forces
    forward_invariant = bool(np.all(image <= radii[1:] + tolerance))
    if require_invariant and not initial_inside:
        raise ValueError("the initial state lies outside the first radius")
    if require_invariant and not forward_invariant:
        raise ValueError("the supplied radius schedule is not forward invariant")
    effective = gains + quadratic * radii[:-1]
    envelope = np.empty(count + 1, dtype=np.float64)
    envelope[0] = initial
    for index in range(count):
        envelope[index + 1] = effective[index] * envelope[index] + forces[index]
    envelope_inside = bool(np.all(envelope <= radii + 8 * tolerance))
    if require_invariant and not envelope_inside:
        raise RuntimeError("variable-radius envelope escaped a verified tube")
    return {
        "effective_gains": effective,
        "envelope": envelope,
        "radius_images": image,
        "initial_inside": initial_inside,
        "forward_invariant": forward_invariant,
        "envelope_inside": envelope_inside,
    }


def oriented_integral_bound(centers, error_radii, interval_widths):
    """Enclose one vector-valued path integral while retaining orientation."""

    centers = _finite_array(centers, "centers", ndim=2)
    error_radii = _finite_array(error_radii, "error_radii", ndim=1)
    widths = _finite_array(interval_widths, "interval_widths", ndim=1)
    intervals = centers.shape[0]
    if error_radii.shape != (intervals,) or widths.shape != (intervals,):
        raise ValueError("oriented-integral arrays have incompatible lengths")
    if np.any(error_radii < 0) or np.any(widths <= 0):
        raise ValueError("integral errors must be nonnegative and widths positive")
    if not np.isclose(widths.sum(), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("path-subdivision widths must sum to one")
    oriented_center = np.einsum("i,ij->j", widths, centers)
    error = float(np.dot(widths, error_radii))
    return {
        "oriented_center": oriented_center,
        "error_radius": error,
        "upper_norm": float(np.linalg.norm(oriented_center) + error),
        "stagewise_triangle_bound": float(
            np.dot(widths, np.linalg.norm(centers, axis=1) + error_radii)),
    }


def environment_metadata():
    """Return diagnostic software and hardware fields for every live record."""

    result = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "host": platform.node(),
        "machine": platform.machine(),
    }
    try:
        import torch
    except ImportError:
        result.update({
            "torch": None,
            "cuda_runtime": None,
            "cuda_driver": None,
            "cuda_devices": [],
            "deterministic_algorithms": None,
        })
        return result
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append({
                "index": index,
                "name": properties.name,
                "total_memory": int(properties.total_memory),
                "capability": list(torch.cuda.get_device_capability(index)),
            })
    driver = None
    try:
        driver = int(torch._C._cuda_getDriverVersion())
    except (AttributeError, RuntimeError):
        pass
    result.update({
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_driver": driver,
        "cuda_devices": devices,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    })
    return result
