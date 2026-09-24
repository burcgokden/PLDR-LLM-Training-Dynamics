"""Exact numerical kernels for the PLDR gate-shape collapse theory.

The final LayerNorm of a shared PLDR row map writes every output as

    beta + diag(gamma) @ normalized_shape.

This module implements that identity, its exact contrast-energy dynamics,
the dense gate-sector cross-entropy split, the full three-block AdamW
successor, ordered products, and the exact PLGA secant chain.  The kernels
contain no checkpoint I/O and never accept a caller-supplied pass/fail flag.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def _finite_array(value, name, *, ndim=None):
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a nonfinite value")
    return array


def _positive_scalar(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def normalized_shape(preactivation, *, epsilon=1e-6):
    """Return the exact normalized core of LayerNorm on the final axis."""

    preactivation = _finite_array(preactivation, "preactivation")
    if preactivation.ndim < 1 or preactivation.shape[-1] < 1:
        raise ValueError("preactivation must have a nonempty final axis")
    epsilon = _positive_scalar(epsilon, "LayerNorm epsilon")
    centered = preactivation - np.mean(
        preactivation, axis=-1, keepdims=True)
    denominator = np.sqrt(
        epsilon + np.mean(centered * centered, axis=-1, keepdims=True))
    shape = centered / denominator
    dimension = preactivation.shape[-1]
    radius_squared = np.sum(shape * shape, axis=-1)
    if np.any(radius_squared >= dimension):
        tolerance = 64.0 * np.finfo(np.float64).eps * max(1, dimension)
        if np.any(radius_squared > dimension + tolerance):
            raise ArithmeticError("normalized shape escaped its exact radius")
    return shape


def gate_shape_output(beta, gamma, shape):
    """Apply the final affine LayerNorm gate to normalized row shapes."""

    beta = _finite_array(beta, "beta", ndim=1)
    gamma = _finite_array(gamma, "gamma", ndim=1)
    shape = _finite_array(shape, "shape")
    if beta.shape != gamma.shape or shape.shape[-1:] != gamma.shape:
        raise ValueError("gate, bias, and normalized-shape dimensions disagree")
    return beta + gamma * shape


def global_gate_oscillation_bound(gamma):
    """Return the architecture-owned bound ``2 sqrt(d) ||gamma||_inf``."""

    gamma = _finite_array(gamma, "gamma", ndim=1)
    if gamma.size == 0:
        raise ValueError("gamma cannot be empty")
    return 2.0 * math.sqrt(gamma.size) * float(np.max(np.abs(gamma)))


def row_contrast_geometry(gamma, left_shape, right_shape, weights=None):
    """Recompute the diagonal shape metric and exact row-contrast energy."""

    gamma = _finite_array(gamma, "gamma", ndim=1)
    left_shape = _finite_array(left_shape, "left_shape", ndim=2)
    right_shape = _finite_array(right_shape, "right_shape", ndim=2)
    if left_shape.shape != right_shape.shape:
        raise ValueError("left and right shape registries disagree")
    if left_shape.shape[1:] != gamma.shape or not len(left_shape):
        raise ValueError("pair registry has the wrong gate dimension")
    if weights is None:
        weights = np.full(len(left_shape), 1.0 / len(left_shape))
    weights = _finite_array(weights, "weights", ndim=1)
    if weights.shape != (len(left_shape),) or np.any(weights <= 0.0):
        raise ValueError("pair weights must be positive and complete")
    if not np.isclose(weights.sum(), 1.0, rtol=1e-12, atol=1e-14):
        raise ValueError("pair weights must sum to one")
    difference = left_shape - right_shape
    q = np.sum(weights[:, None] * difference * difference, axis=0)
    output_difference = difference * gamma
    pair_energy = np.sum(output_difference * output_difference, axis=1)
    energy = float(weights @ pair_energy)
    diagonal_energy = float(q @ (gamma * gamma))
    residual = abs(energy - diagonal_energy)
    scale = max(abs(energy), abs(diagonal_energy), 1.0)
    if residual > 128.0 * np.finfo(np.float64).eps * scale:
        raise ArithmeticError("contrast energy failed its diagonal identity")
    return {
        "shape_difference": difference,
        "q": q,
        "pair_energy": pair_energy,
        "energy": energy,
        "diagonal_energy": diagonal_energy,
        "identity_residual": residual,
    }


def gate_shape_energy_increment(gamma, gamma_next, q, q_next):
    """Evaluate the exact gate-work plus shape-work energy increment."""

    gamma = _finite_array(gamma, "gamma", ndim=1)
    gamma_next = _finite_array(gamma_next, "gamma_next", ndim=1)
    q = _finite_array(q, "q", ndim=1)
    q_next = _finite_array(q_next, "q_next", ndim=1)
    if not (gamma.shape == gamma_next.shape == q.shape == q_next.shape):
        raise ValueError("gate-shape increment dimensions disagree")
    if np.any(q < 0.0) or np.any(q_next < 0.0):
        raise ValueError("shape metrics must be nonnegative")
    delta_gamma = gamma_next - gamma
    gate_work = float(
        2.0 * np.sum(q * gamma * delta_gamma)
        + np.sum(q * delta_gamma * delta_gamma)
    )
    shape_work = float(np.sum((q_next - q) * gamma_next * gamma_next))
    energy = float(np.sum(q * gamma * gamma))
    energy_next = float(np.sum(q_next * gamma_next * gamma_next))
    residual = energy_next - energy - gate_work - shape_work
    return {
        "energy": energy,
        "energy_next": energy_next,
        "gate_work": gate_work,
        "shape_work": shape_work,
        "increment": energy_next - energy,
        "identity_residual": residual,
    }


def source_owned_energy_envelope(
        gamma, delta_gamma, q, q_directional_derivative,
        shape_remainder_radius):
    """Build a one-step energy upper bound without reading target energy.

    ``shape_remainder_radius`` bounds every coordinate of
    ``q_next - q - Dq[delta_zeta]`` in absolute value.  It must be fixed
    from construction evidence before this function is used on validation.
    """

    gamma = _finite_array(gamma, "gamma", ndim=1)
    delta_gamma = _finite_array(delta_gamma, "delta_gamma", ndim=1)
    q = _finite_array(q, "q", ndim=1)
    q_directional_derivative = _finite_array(
        q_directional_derivative, "q_directional_derivative", ndim=1)
    if not (
        gamma.shape == delta_gamma.shape == q.shape
        == q_directional_derivative.shape
    ):
        raise ValueError("source energy enclosure dimensions disagree")
    if np.any(q < 0.0):
        raise ValueError("source shape metric must be nonnegative")
    shape_remainder_radius = float(shape_remainder_radius)
    if not math.isfinite(shape_remainder_radius) or shape_remainder_radius < 0:
        raise ValueError("shape remainder radius must be finite and nonnegative")
    gamma_next = gamma + delta_gamma
    gate_work = float(
        2.0 * np.sum(q * gamma * delta_gamma)
        + np.sum(q * delta_gamma * delta_gamma)
    )
    shape_linear = float(
        np.sum(q_directional_derivative * gamma_next * gamma_next))
    shape_remainder_charge = float(
        shape_remainder_radius * np.sum(gamma_next * gamma_next))
    source_energy = float(np.sum(q * gamma * gamma))
    upper_energy = (
        source_energy + gate_work + shape_linear + shape_remainder_charge)
    return {
        "source_energy": source_energy,
        "gate_work": gate_work,
        "shape_linear_work": shape_linear,
        "shape_remainder_charge": shape_remainder_charge,
        "upper_energy": upper_energy,
    }


def softmax_fisher(probability):
    probability = _finite_array(probability, "probability", ndim=1)
    if np.any(probability <= 0.0):
        raise ValueError("softmax probabilities must be strictly positive")
    if not np.isclose(probability.sum(), 1.0, rtol=1e-12, atol=1e-14):
        raise ValueError("softmax probabilities must sum to one")
    return np.diag(probability) - np.outer(probability, probability)


def pairwise_softmax_energy(probability, value):
    """Return the general-vocabulary unordered-pair covariance energy."""

    probability = _finite_array(probability, "probability", ndim=1)
    value = _finite_array(value, "value", ndim=1)
    if probability.shape != value.shape:
        raise ValueError("probability and value dimensions disagree")
    softmax_fisher(probability)
    difference = value[:, None] - value[None, :]
    weight = probability[:, None] * probability[None, :]
    return float(np.sum(np.triu(weight * difference * difference, k=1)))


def gate_loss_hessian_split(
        logit_jacobians, probabilities, targets, logit_second_jets,
        weights=None):
    """Recompute Fisher, signed second jet, force, and true gate Hessian.

    Shapes are ``(sources, vocabulary, gate_dimension)`` for Jacobians and
    ``(sources, vocabulary, gate_dimension, gate_dimension)`` for second
    jets.  This dense routine is intended for small test archives and direct
    replay checks; the live producer uses HVPs to avoid vocabulary-scale
    storage.
    """

    jacobian = _finite_array(
        logit_jacobians, "logit_jacobians", ndim=3)
    probability = _finite_array(probabilities, "probabilities", ndim=2)
    targets = np.asarray(targets)
    jets = _finite_array(logit_second_jets, "logit_second_jets", ndim=4)
    source_count, vocabulary, gate_dimension = jacobian.shape
    if probability.shape != (source_count, vocabulary):
        raise ValueError("probability registry has the wrong shape")
    if jets.shape != (
        source_count, vocabulary, gate_dimension, gate_dimension
    ):
        raise ValueError("logit second-jet registry has the wrong shape")
    if targets.shape != (source_count,) or not np.issubdtype(
        targets.dtype, np.integer
    ):
        raise ValueError("targets must be one integer per source")
    if np.any(targets < 0) or np.any(targets >= vocabulary):
        raise ValueError("a target lies outside the vocabulary")
    if weights is None:
        weights = np.full(source_count, 1.0 / source_count)
    weights = _finite_array(weights, "weights", ndim=1)
    if weights.shape != (source_count,) or np.any(weights <= 0.0):
        raise ValueError("source weights must be positive")
    if not np.isclose(weights.sum(), 1.0, rtol=1e-12, atol=1e-14):
        raise ValueError("source weights must sum to one")
    fisher = np.zeros((gate_dimension, gate_dimension), dtype=np.float64)
    signed = np.zeros_like(fisher)
    force = np.zeros(gate_dimension, dtype=np.float64)
    for source in range(source_count):
        p = probability[source]
        covariance = softmax_fisher(p)
        residual = p.copy()
        residual[int(targets[source])] -= 1.0
        j = jacobian[source]
        fisher += weights[source] * j.T @ covariance @ j
        signed += weights[source] * np.tensordot(
            residual, jets[source], axes=(0, 0))
        force += weights[source] * j.T @ residual
    fisher = 0.5 * (fisher + fisher.T)
    signed = 0.5 * (signed + signed.T)
    return {
        "fisher": fisher,
        "signed_second_jet": signed,
        "true_hessian": fisher + signed,
        "force": force,
    }


def residual_scaled_spectrum(matrix, *, reconstruction_residual=0.0):
    """Return dense eigenvalue intervals with a scale-aware zero threshold."""

    matrix = _finite_array(matrix, "matrix", ndim=2)
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("spectrum requires a nonempty square matrix")
    asymmetry = float(np.linalg.norm(matrix - matrix.T, ord=2))
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    operator_norm = float(np.max(np.abs(eigenvalues)))
    reconstruction_residual = float(reconstruction_residual)
    if not math.isfinite(reconstruction_residual) or reconstruction_residual < 0:
        raise ValueError("reconstruction residual must be finite and nonnegative")
    residual = max(reconstruction_residual, asymmetry)
    threshold = max(
        residual,
        100.0 * np.finfo(np.float64).eps * max(operator_norm, 1.0),
    )
    return {
        "eigenvalues": eigenvalues,
        "lower_intervals": eigenvalues - threshold,
        "upper_intervals": eigenvalues + threshold,
        "operator_norm": operator_norm,
        "residual": residual,
        "zero_threshold": threshold,
        "positive_lower_edge": bool(eigenvalues[0] - threshold > 0.0),
    }


def adamw_gate_step(
        gamma, first_moment, second_moment, raw_gradient, *, learning_rate,
        beta1, beta2, epsilon, optimizer_step, weight_decay,
        clip_value=None):
    """Apply one standard value-clipped AdamW gate update in operation order."""

    gamma = _finite_array(gamma, "gamma", ndim=1)
    first_moment = _finite_array(first_moment, "first_moment", ndim=1)
    second_moment = _finite_array(second_moment, "second_moment", ndim=1)
    raw_gradient = _finite_array(raw_gradient, "raw_gradient", ndim=1)
    if not (
        gamma.shape == first_moment.shape == second_moment.shape
        == raw_gradient.shape
    ):
        raise ValueError("AdamW gate-state dimensions disagree")
    if np.any(second_moment < 0.0):
        raise ValueError("AdamW second moment must be nonnegative")
    learning_rate = _positive_scalar(learning_rate, "learning rate")
    epsilon = _positive_scalar(epsilon, "Adam epsilon")
    beta1 = float(beta1)
    beta2 = float(beta2)
    if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
        raise ValueError("Adam betas must lie in [0, 1)")
    optimizer_step = int(optimizer_step)
    if optimizer_step < 1:
        raise ValueError("optimizer_step is the positive post-step clock")
    weight_decay = np.broadcast_to(
        _finite_array(weight_decay, "weight_decay"), gamma.shape)
    if clip_value is None:
        clipped_gradient = raw_gradient.copy()
    else:
        clip_value = _positive_scalar(clip_value, "clip value")
        clipped_gradient = np.clip(raw_gradient, -clip_value, clip_value)
    first_next = beta1 * first_moment + (1.0 - beta1) * clipped_gradient
    second_next = (
        beta2 * second_moment
        + (1.0 - beta2) * clipped_gradient * clipped_gradient
    )
    first_hat = first_next / (1.0 - beta1 ** optimizer_step)
    second_hat = second_next / (1.0 - beta2 ** optimizer_step)
    gamma_next = (
        (1.0 - learning_rate * weight_decay) * gamma
        - learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
    )
    return {
        "gamma": gamma_next,
        "first_moment": first_next,
        "second_moment": second_next,
        "clipped_gradient": clipped_gradient,
        "first_hat": first_hat,
        "second_hat": second_hat,
    }


def adamw_gate_block(
        gradient_jacobian, gamma, first_moment, second_moment, raw_gradient,
        *, learning_rate, beta1, beta2, epsilon, optimizer_step,
        weight_decay, clip_value=None, boundary_tolerance=0.0):
    """Construct the exact ``3d by 3d`` smooth-cell AdamW derivative."""

    gradient_jacobian = _finite_array(
        gradient_jacobian, "gradient_jacobian", ndim=2)
    gamma = _finite_array(gamma, "gamma", ndim=1)
    dimension = gamma.size
    if gradient_jacobian.shape != (dimension, dimension):
        raise ValueError("gradient Jacobian has the wrong gate dimension")
    first_moment = _finite_array(first_moment, "first_moment", ndim=1)
    second_moment = _finite_array(second_moment, "second_moment", ndim=1)
    raw_gradient = _finite_array(raw_gradient, "raw_gradient", ndim=1)
    if not (
        first_moment.shape == second_moment.shape == raw_gradient.shape
        == gamma.shape
    ):
        raise ValueError("AdamW gate-state dimensions disagree")
    step = adamw_gate_step(
        gamma, first_moment, second_moment, raw_gradient,
        learning_rate=learning_rate, beta1=beta1, beta2=beta2,
        epsilon=epsilon, optimizer_step=optimizer_step,
        weight_decay=weight_decay, clip_value=clip_value)
    if clip_value is None:
        clip_derivative = np.ones(dimension, dtype=np.float64)
    else:
        clip_value = _positive_scalar(clip_value, "clip value")
        boundary_tolerance = float(boundary_tolerance)
        if not math.isfinite(boundary_tolerance) or boundary_tolerance < 0:
            raise ValueError("boundary tolerance must be finite and nonnegative")
        distance = np.abs(np.abs(raw_gradient) - clip_value)
        if np.any(distance <= boundary_tolerance):
            raise ValueError("a gate gradient lies on the clipping boundary")
        clip_derivative = (np.abs(raw_gradient) < clip_value).astype(np.float64)
    beta1 = float(beta1)
    beta2 = float(beta2)
    learning_rate = float(learning_rate)
    epsilon = float(epsilon)
    optimizer_step = int(optimizer_step)
    first_bias = 1.0 - beta1 ** optimizer_step
    second_bias = 1.0 - beta2 ** optimizer_step
    second_hat = step["second_hat"]
    if np.any(second_hat <= 0.0):
        raise ValueError(
            "the smooth AdamW block requires a positive bias-corrected "
            "second moment")
    clipped_gradient = step["clipped_gradient"]
    first_hat = step["first_hat"]
    clipped_jacobian = np.diag(clip_derivative) @ gradient_jacobian
    moment_position = (1.0 - beta1) * clipped_jacobian
    variance_position = (
        2.0 * (1.0 - beta2)
        * np.diag(clipped_gradient) @ clipped_jacobian
    )
    root = np.sqrt(second_hat)
    d_first = np.diag(1.0 / (first_bias * (root + epsilon)))
    d_second = np.diag(
        -first_hat
        / (2.0 * second_bias * root * (root + epsilon) ** 2)
    )
    decay = np.broadcast_to(
        _finite_array(weight_decay, "weight_decay"), gamma.shape)
    identity = np.eye(dimension)
    top_left = (
        identity - learning_rate * np.diag(decay)
        - learning_rate
        * (d_first @ moment_position + d_second @ variance_position)
    )
    top_middle = -learning_rate * beta1 * d_first
    top_right = -learning_rate * beta2 * d_second
    operator = np.block([
        [top_left, top_middle, top_right],
        [moment_position, beta1 * identity, np.zeros_like(identity)],
        [variance_position, np.zeros_like(identity), beta2 * identity],
    ])
    return {
        **step,
        "operator": operator,
        "clip_derivative": clip_derivative,
        "clipped_jacobian": clipped_jacobian,
        "moment_position": moment_position,
        "variance_position": variance_position,
        "d_first": d_first,
        "d_second": d_second,
    }


def finite_difference_adamw_block(
        gradient_function, gamma, first_moment, second_moment, *, step_size,
        **optimizer):
    """Independently differentiate a complete gate update by central differences."""

    gamma = _finite_array(gamma, "gamma", ndim=1)
    first_moment = _finite_array(first_moment, "first_moment", ndim=1)
    second_moment = _finite_array(second_moment, "second_moment", ndim=1)
    if not (gamma.shape == first_moment.shape == second_moment.shape):
        raise ValueError("finite-difference gate state dimensions disagree")
    step_size = _positive_scalar(step_size, "finite-difference step")
    state = np.concatenate([gamma, first_moment, second_moment])
    dimension = gamma.size

    def successor(vector):
        position = vector[:dimension]
        first = vector[dimension:2 * dimension]
        second = vector[2 * dimension:]
        gradient = _finite_array(
            gradient_function(position), "gradient_function output", ndim=1)
        update = adamw_gate_step(
            position, first, second, gradient, **optimizer)
        return np.concatenate([
            update["gamma"], update["first_moment"], update["second_moment"]])

    columns = []
    for index in range(3 * dimension):
        direction = np.zeros_like(state)
        direction[index] = step_size
        columns.append(
            (successor(state + direction) - successor(state - direction))
            / (2.0 * step_size)
        )
    return np.column_stack(columns)


def chronological_products(operators: Sequence[np.ndarray]):
    """Return all prefix products in physical update order."""

    values = [_finite_array(value, "operator", ndim=2) for value in operators]
    if not values:
        raise ValueError("an ordered block cannot be empty")
    dimension = values[0].shape[0]
    if any(value.shape != (dimension, dimension) for value in values):
        raise ValueError("ordered operators must have one square shape")
    products = []
    current = np.eye(dimension)
    for operator in values:
        current = operator @ current
        products.append(current.copy())
    return products


def ordered_capture_envelope(operators, initial_norm, forcing_norms):
    """Evaluate the exact product-convolution norm envelope at every node."""

    operators = [_finite_array(value, "operator", ndim=2) for value in operators]
    if not operators:
        raise ValueError("capture envelope requires at least one operator")
    dimension = operators[0].shape[0]
    if any(value.shape != (dimension, dimension) for value in operators):
        raise ValueError("capture operators must have one square shape")
    forcing = _finite_array(forcing_norms, "forcing_norms", ndim=1)
    if forcing.shape != (len(operators),) or np.any(forcing < 0.0):
        raise ValueError("forcing norms must be nonnegative and step-complete")
    initial_norm = float(initial_norm)
    if not math.isfinite(initial_norm) or initial_norm < 0.0:
        raise ValueError("initial norm must be finite and nonnegative")
    bounds = []
    prefix_gains = []
    for terminal in range(1, len(operators) + 1):
        prefix = np.eye(dimension)
        for index in range(terminal):
            prefix = operators[index] @ prefix
        prefix_gain = float(np.linalg.norm(prefix, ord=2))
        value = prefix_gain * initial_norm
        for source in range(terminal):
            tail = np.eye(dimension)
            for index in range(source + 1, terminal):
                tail = operators[index] @ tail
            value += float(np.linalg.norm(tail, ord=2)) * forcing[source]
        bounds.append(value)
        prefix_gains.append(prefix_gain)
    return {
        "prefix_gains": np.asarray(prefix_gains),
        "envelope": np.asarray(bounds),
        "block_gain": prefix_gains[-1],
    }


def _sigmoid(value):
    positive = value >= 0.0
    result = np.empty_like(value, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def iswiglu(value):
    value = _finite_array(value, "iSwiGLU input")
    return value * value * _sigmoid(value)


def iswiglu_derivative(value):
    value = _finite_array(value, "iSwiGLU derivative input")
    sigmoid = _sigmoid(value)
    return 2.0 * value * sigmoid + value * value * sigmoid * (1.0 - sigmoid)


def divided_difference(function, derivative, left, right):
    """Compute an exact endpoint secant with a derivative on the diagonal."""

    left = _finite_array(left, "left endpoint")
    right = _finite_array(right, "right endpoint")
    if left.shape != right.shape:
        raise ValueError("divided-difference endpoint shapes disagree")
    left_value = _finite_array(function(left), "left function value")
    right_value = _finite_array(function(right), "right function value")
    if left_value.shape != left.shape or right_value.shape != right.shape:
        raise ValueError("divided-difference function must act coordinatewise")
    difference = right - left
    result = _finite_array(derivative(left), "diagonal derivative")
    if result.shape != left.shape:
        raise ValueError("divided-difference derivative has the wrong shape")
    off_diagonal = difference != 0.0
    result = result.copy()
    result[off_diagonal] = (
        (right_value - left_value)[off_diagonal] / difference[off_diagonal])
    return result


def plga_secant_chain(
        generator_left, generator_right, weight, bias, powers, coupling,
        coupling_bias, *, positive_floor=1e-9):
    """Replay the exact generator-to-PLGA divided-difference factorization."""

    left = _finite_array(generator_left, "generator_left", ndim=2)
    right = _finite_array(generator_right, "generator_right", ndim=2)
    weight = _finite_array(weight, "weight", ndim=2)
    bias = _finite_array(bias, "bias", ndim=2)
    powers = _finite_array(powers, "powers", ndim=2)
    coupling = _finite_array(coupling, "coupling", ndim=2)
    coupling_bias = _finite_array(coupling_bias, "coupling_bias", ndim=2)
    if left.shape != right.shape or left.shape[0] != left.shape[1]:
        raise ValueError("PLGA generator endpoints must share a square shape")
    dimension = left.shape[0]
    if any(
        value.shape != (dimension, dimension)
        for value in (weight, bias, powers, coupling, coupling_bias)
    ):
        raise ValueError("PLGA parameter matrices have incompatible shapes")
    positive_floor = _positive_scalar(positive_floor, "positive floor")
    pre_left = weight @ left + bias
    pre_right = weight @ right + bias
    metric_left = iswiglu(pre_left) + positive_floor
    metric_right = iswiglu(pre_right) + positive_floor
    potential_left = np.power(metric_left, powers)
    potential_right = np.power(metric_right, powers)
    curvature_left = coupling @ potential_left + coupling_bias
    curvature_right = coupling @ potential_right + coupling_bias
    activation_secant = divided_difference(
        iswiglu, iswiglu_derivative, pre_left, pre_right)

    def coordinate_power(base):
        return np.power(base, powers)

    def coordinate_power_derivative(base):
        return powers * np.power(base, powers - 1.0)

    power_secant = divided_difference(
        coordinate_power, coordinate_power_derivative,
        metric_left, metric_right)
    predicted = coupling @ (
        power_secant * activation_secant * (weight @ (right - left)))
    realized = curvature_right - curvature_left
    residual = float(np.linalg.norm(realized - predicted))
    return {
        "preactivation_left": pre_left,
        "preactivation_right": pre_right,
        "metric_left": metric_left,
        "metric_right": metric_right,
        "potential_left": potential_left,
        "potential_right": potential_right,
        "curvature_left": curvature_left,
        "curvature_right": curvature_right,
        "activation_secant": activation_secant,
        "power_secant": power_secant,
        "predicted_difference": predicted,
        "realized_difference": realized,
        "identity_residual": residual,
        "operator_bound": (
            float(np.linalg.norm(coupling, ord=2))
            * float(np.max(np.abs(power_secant)))
            * float(np.max(np.abs(activation_secant)))
            * float(np.linalg.norm(weight, ord=2))
        ),
    }


def centered_logit_margin_preserved(reference_logits, candidate_logits):
    """Apply the sharp Euclidean ``margin / sqrt(2)`` decision criterion."""

    reference = _finite_array(reference_logits, "reference_logits", ndim=1)
    candidate = _finite_array(candidate_logits, "candidate_logits", ndim=1)
    if reference.shape != candidate.shape or reference.size < 2:
        raise ValueError("logit vectors must share a nontrivial vocabulary")
    winner = int(np.argmax(reference))
    competitors = np.delete(reference, winner)
    margin = float(reference[winner] - np.max(competitors))
    difference = float(np.linalg.norm(candidate - reference))
    qualified = math.sqrt(2.0) * difference < margin
    return {
        "reference_winner": winner,
        "candidate_winner": int(np.argmax(candidate)),
        "margin": margin,
        "difference_norm": difference,
        "qualified": bool(qualified),
        "decision_preserved": bool(np.argmax(candidate) == winner),
    }
