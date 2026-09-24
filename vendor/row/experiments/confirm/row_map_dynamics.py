"""Numerical kernels for comprehensive PLDR row-map collapse.

The functions in this module are independent of checkpoint loading.  They
reconstruct the graph energy, logarithmic gate and shape routes, exact AdamW
gate dissipation, normalized-shape flux, and dimensionless chronological
diagnostics used by the confirmation program.
"""

from __future__ import annotations

from collections import deque
import math
from typing import Sequence

import numpy as np


def _finite(value, name, *, ndim=None):
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a nonfinite value")
    return array


def _positive(value, name):
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _graph_diameter(vertex_count, edges):
    adjacency = [[] for _ in range(vertex_count)]
    for left, right in edges:
        if left == right:
            raise ValueError("graph self-edges are not registered contrasts")
        adjacency[left].append(right)
        adjacency[right].append(left)
    diameter = 0
    for source in range(vertex_count):
        distance = [-1] * vertex_count
        distance[source] = 0
        queue = deque([source])
        while queue:
            vertex = queue.popleft()
            for neighbor in adjacency[vertex]:
                if distance[neighbor] < 0:
                    distance[neighbor] = distance[vertex] + 1
                    queue.append(neighbor)
        if any(value < 0 for value in distance):
            raise ValueError("the physical-row graph must be connected")
        diameter = max(diameter, max(distance))
    return diameter


def graph_contrast(gamma, normalized_shapes, edges, weights):
    """Rebuild weighted graph energy and its connected-graph diameter bound."""

    gate = _finite(gamma, "gamma", ndim=1)
    shape = _finite(normalized_shapes, "normalized_shapes", ndim=2)
    edge_array = np.asarray(edges)
    if not np.issubdtype(edge_array.dtype, np.integer):
        raise ValueError("graph edges must use integer vertex indices")
    edge_array = np.asarray(edge_array, dtype=np.int64)
    weight = _finite(weights, "weights", ndim=1)
    if shape.shape[1:] != gate.shape:
        raise ValueError("shape and gate dimensions disagree")
    if edge_array.ndim != 2 or edge_array.shape[1:] != (2,):
        raise ValueError("edges must have shape (edge_count, 2)")
    if len(edge_array) == 0 or weight.shape != (len(edge_array),):
        raise ValueError("the graph needs one positive weight per edge")
    if np.any(weight <= 0.0):
        raise ValueError("graph weights must be positive")
    if np.any(edge_array < 0) or np.any(edge_array >= len(shape)):
        raise ValueError("a graph edge leaves the vertex registry")
    graph_diameter = _graph_diameter(len(shape), edge_array)
    difference = shape[edge_array[:, 0]] - shape[edge_array[:, 1]]
    q = np.sum(weight[:, None] * difference * difference, axis=0)
    edge_output_difference = difference * gate
    direct_energy = float(np.sum(
        weight * np.sum(edge_output_difference ** 2, axis=1)))
    diagonal_energy = float(np.sum(q * gate * gate))
    output = shape * gate
    pairwise = output[:, None, :] - output[None, :, :]
    vertex_diameter = float(np.max(np.linalg.norm(pairwise, axis=-1)))
    minimum_weight = float(np.min(weight))
    diameter_bound = math.sqrt(
        graph_diameter * diagonal_energy / minimum_weight)
    return {
        "q": q,
        "energy": diagonal_energy,
        "direct_energy": direct_energy,
        "energy_identity_residual": abs(direct_energy - diagonal_energy),
        "graph_diameter": graph_diameter,
        "minimum_weight": minimum_weight,
        "vertex_diameter": vertex_diameter,
        "diameter_bound": diameter_bound,
        "diameter_slack": diameter_bound - vertex_diameter,
    }


def covered_domain_bound(
        energy, graph_diameter, minimum_weight, cover_radius, lipschitz):
    """Evaluate the physical-domain cover bound from the graph theorem."""

    energy = float(energy)
    graph_diameter = float(graph_diameter)
    cover_radius = float(cover_radius)
    lipschitz = float(lipschitz)
    minimum_weight = _positive(minimum_weight, "minimum_weight")
    if any(not math.isfinite(value) or value < 0.0 for value in (
            energy, graph_diameter, cover_radius, lipschitz)):
        raise ValueError("cover-bound inputs must be finite and nonnegative")
    graph_term = math.sqrt(graph_diameter * energy / minimum_weight)
    return 2.0 * lipschitz * cover_radius + graph_term


def log_route_decomposition(gamma_history, q_history, *, zero_tolerance=0.0):
    """Reconstruct every positive coordinate route and record exact zeros."""

    gamma = _finite(gamma_history, "gamma_history", ndim=2)
    q = _finite(q_history, "q_history", ndim=2)
    if gamma.shape != q.shape or len(gamma) < 2:
        raise ValueError("gate and shape histories must share at least two nodes")
    if np.any(q < 0.0):
        raise ValueError("shape metrics must be nonnegative")
    zero_tolerance = float(zero_tolerance)
    if not math.isfinite(zero_tolerance) or zero_tolerance < 0.0:
        raise ValueError("zero_tolerance must be finite and nonnegative")
    gate_square = gamma * gamma
    energy = gate_square * q
    step_count, dimension = gamma.shape[0] - 1, gamma.shape[1]
    gate_log = np.full((step_count, dimension), np.nan)
    shape_log = np.full_like(gate_log, np.nan)
    total_log = np.full_like(gate_log, np.nan)
    positive = (
        (gate_square[:-1] > zero_tolerance)
        & (gate_square[1:] > zero_tolerance)
        & (q[:-1] > zero_tolerance)
        & (q[1:] > zero_tolerance)
    )
    gate_log[positive] = np.log(
        gate_square[:-1][positive] / gate_square[1:][positive])
    shape_log[positive] = np.log(q[:-1][positive] / q[1:][positive])
    total_log[positive] = np.log(
        energy[:-1][positive] / energy[1:][positive])
    residual = np.zeros_like(total_log)
    residual[positive] = (
        total_log[positive] - gate_log[positive] - shape_log[positive])
    first_zero = np.full(dimension, -1, dtype=np.int64)
    for coordinate in range(dimension):
        indices = np.flatnonzero(energy[:, coordinate] <= zero_tolerance)
        if len(indices):
            first_zero[coordinate] = int(indices[0])
    return {
        "energy": energy,
        "gate_log_decrement": gate_log,
        "shape_log_decrement": shape_log,
        "total_log_decrement": total_log,
        "positive_step_mask": positive,
        "log_identity_residual": residual,
        "first_zero_node": first_zero,
    }


def gate_energy_dissipation(
        gamma, q, adaptive_direction, learning_rate, weight_decay,
        decay_mask=None, q_next=None):
    """Evaluate the exact fixed-Q AdamW gate decrement and total increment."""

    gate = _finite(gamma, "gamma", ndim=1)
    metric = _finite(q, "q", ndim=1)
    adaptive = _finite(adaptive_direction, "adaptive_direction", ndim=1)
    if not (gate.shape == metric.shape == adaptive.shape):
        raise ValueError("gate dissipation dimensions disagree")
    if np.any(metric < 0.0):
        raise ValueError("q must be nonnegative")
    eta = _positive(learning_rate, "learning_rate")
    decay = float(weight_decay)
    if not math.isfinite(decay) or decay < 0.0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if decay_mask is None:
        mask = np.ones_like(gate)
    else:
        mask = _finite(decay_mask, "decay_mask", ndim=1)
        if mask.shape != gate.shape or np.any((mask != 0.0) & (mask != 1.0)):
            raise ValueError("decay_mask must be binary and gate-shaped")
    direction = decay * mask * gate + adaptive
    gate_next = gate - eta * direction
    energy = float(np.sum(metric * gate * gate))
    fixed_next = float(np.sum(metric * gate_next * gate_next))
    linear = float(-2.0 * eta * np.sum(metric * gate * direction))
    quadratic = float(eta * eta * np.sum(metric * direction * direction))
    result = {
        "direction": direction,
        "gamma_next": gate_next,
        "energy": energy,
        "fixed_q_energy_next": fixed_next,
        "linear_dissipation": linear,
        "quadratic_correction": quadratic,
        "fixed_q_increment": fixed_next - energy,
        "fixed_q_identity_residual": fixed_next - energy - linear - quadratic,
        "decay_multiplier": 1.0 - eta * decay * mask,
    }
    if q_next is not None:
        metric_next = _finite(q_next, "q_next", ndim=1)
        if metric_next.shape != gate.shape or np.any(metric_next < 0.0):
            raise ValueError("q_next must be nonnegative and gate-shaped")
        shape_work = float(np.sum((metric_next - metric) * gate_next ** 2))
        total_next = float(np.sum(metric_next * gate_next ** 2))
        result.update({
            "shape_work": shape_work,
            "energy_next": total_next,
            "total_identity_residual": (
                total_next - energy - linear - quadratic - shape_work),
        })
    return result


def gate_duhamel(
        gamma_initial, adaptive_history, learning_rates, weight_decay,
        decay_masks=None):
    """Replay the exact time-varying decay-plus-loss variation of constants."""

    initial = _finite(gamma_initial, "gamma_initial", ndim=1)
    adaptive = _finite(adaptive_history, "adaptive_history", ndim=2)
    rates = _finite(learning_rates, "learning_rates", ndim=1)
    if adaptive.shape[1:] != initial.shape or rates.shape != (len(adaptive),):
        raise ValueError("Duhamel histories have incompatible shapes")
    if np.any(rates <= 0.0):
        raise ValueError("learning rates must be positive")
    decay = float(weight_decay)
    if not math.isfinite(decay) or decay < 0.0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if decay_masks is None:
        masks = np.ones_like(adaptive)
    else:
        masks = _finite(decay_masks, "decay_masks", ndim=2)
        if masks.shape != adaptive.shape or np.any(
                (masks != 0.0) & (masks != 1.0)):
            raise ValueError("decay masks must be binary and history-shaped")
    multipliers = 1.0 - rates[:, None] * decay * masks
    direct = initial.copy()
    history = [direct.copy()]
    for step in range(len(adaptive)):
        direct = multipliers[step] * direct - rates[step] * adaptive[step]
        history.append(direct.copy())
    homogeneous = np.prod(multipliers, axis=0) * initial
    convolution = np.zeros_like(initial)
    for source in range(len(adaptive)):
        tail = np.prod(multipliers[source + 1:], axis=0)
        convolution += tail * rates[source] * adaptive[source]
    reconstructed = homogeneous - convolution
    return {
        "multipliers": multipliers,
        "history": np.asarray(history),
        "homogeneous": homogeneous,
        "loss_memory_convolution": convolution,
        "reconstructed": reconstructed,
        "identity_residual": reconstructed - history[-1],
    }


def decay_loss_log_split(
        gamma, adaptive_direction, learning_rate, weight_decay,
        decay_mask=None):
    """Split one realized squared-gate log decrement into decay and loss memory."""

    gate = _finite(gamma, "gamma", ndim=1)
    adaptive = _finite(adaptive_direction, "adaptive_direction", ndim=1)
    if gate.shape != adaptive.shape:
        raise ValueError("gate and adaptive direction dimensions disagree")
    eta = _positive(learning_rate, "learning_rate")
    decay = float(weight_decay)
    if not math.isfinite(decay) or decay < 0.0:
        raise ValueError("weight_decay must be finite and nonnegative")
    mask = np.ones_like(gate) if decay_mask is None else _finite(
        decay_mask, "decay_mask", ndim=1)
    if mask.shape != gate.shape or np.any((mask != 0.0) & (mask != 1.0)):
        raise ValueError("decay_mask must be binary and gate-shaped")
    multiplier = 1.0 - eta * decay * mask
    gate_next = multiplier * gate - eta * adaptive
    valid = (gate != 0.0) & (gate_next != 0.0) & (multiplier != 0.0)
    realized = np.full_like(gate, np.nan)
    decay_log = np.full_like(gate, np.nan)
    memory_log = np.full_like(gate, np.nan)
    realized[valid] = np.log(gate[valid] ** 2 / gate_next[valid] ** 2)
    decay_log[valid] = -2.0 * np.log(np.abs(multiplier[valid]))
    ratio = np.ones_like(gate)
    ratio[valid] = (
        1.0 - eta * adaptive[valid]
        / (multiplier[valid] * gate[valid]))
    nonzero_ratio = valid & (ratio != 0.0)
    memory_log[nonzero_ratio] = -2.0 * np.log(np.abs(ratio[nonzero_ratio]))
    residual = realized - decay_log - memory_log
    return {
        "gamma_next": gate_next,
        "realized_gate_log": realized,
        "decay_log": decay_log,
        "loss_memory_log": memory_log,
        "valid_mask": nonzero_ratio,
        "identity_residual": residual,
    }


def shape_flux(shape_contrast, linear_flux, remainder):
    """Reconstruct the exact normalized-shape squared-norm increment."""

    shape = _finite(shape_contrast, "shape_contrast")
    linear = _finite(linear_flux, "linear_flux")
    residual = _finite(remainder, "remainder")
    if not (shape.shape == linear.shape == residual.shape):
        raise ValueError("shape-flux arrays must have one shape")
    successor = shape + linear + residual
    source_q = np.sum(shape * shape, axis=0)
    target_q = np.sum(successor * successor, axis=0)
    first = 2.0 * np.sum(shape * linear, axis=0)
    linear_square = np.sum(linear * linear, axis=0)
    remainder_cross = 2.0 * np.sum((shape + linear) * residual, axis=0)
    remainder_square = np.sum(residual * residual, axis=0)
    prediction = first + linear_square + remainder_cross + remainder_square
    return {
        "q": source_q,
        "q_next": target_q,
        "first_order_flux": first,
        "linear_square": linear_square,
        "remainder_cross": remainder_cross,
        "remainder_square": remainder_square,
        "increment": target_q - source_q,
        "identity_residual": target_q - source_q - prediction,
    }


def layernorm_jacobian(preactivation, *, epsilon):
    """Return the exact epsilon-LayerNorm normalized-shape Jacobian per row."""

    value = _finite(preactivation, "preactivation", ndim=2)
    epsilon = _positive(epsilon, "epsilon")
    dimension = value.shape[1]
    centered = value - value.mean(axis=1, keepdims=True)
    scale = np.sqrt(epsilon + np.mean(centered * centered, axis=1))
    projector = np.eye(dimension) - np.ones((dimension, dimension)) / dimension
    result = np.empty((len(value), dimension, dimension), dtype=np.float64)
    for row in range(len(value)):
        result[row] = (
            projector / scale[row]
            - np.outer(centered[row], centered[row])
            / (dimension * scale[row] ** 3))
    return result


def dimensionless_state_scale(second_hat, *, epsilon):
    """Construct the declared diagonal scale for (gamma, m, v)."""

    second = _finite(second_hat, "second_hat", ndim=1)
    epsilon = _positive(epsilon, "epsilon")
    if np.any(second < 0.0):
        raise ValueError("second_hat must be nonnegative")
    denominator = np.sqrt(second) + epsilon
    return np.concatenate([
        np.ones_like(second),
        1.0 / denominator,
        1.0 / (second + epsilon * epsilon),
    ])


def relative_matrix_residual(left, right):
    """Return a scale-aware Frobenius residual with an exact zero policy."""

    first = _finite(left, "left", ndim=2)
    second = _finite(right, "right", ndim=2)
    if first.shape != second.shape:
        raise ValueError("matrix residual operands must have one shape")
    numerator = float(np.linalg.norm(first - second, ord="fro"))
    denominator = max(
        float(np.linalg.norm(first, ord="fro")),
        float(np.linalg.norm(second, ord="fro")),
    )
    return 0.0 if denominator == 0.0 else numerator / denominator


def scaled_chronological_diagnostics(
        operators: Sequence[np.ndarray], state_scales: Sequence[np.ndarray]):
    """Form physical and endpoint-scaled chronological products."""

    matrices = [_finite(value, "operator", ndim=2) for value in operators]
    scales = [_finite(value, "state_scale", ndim=1) for value in state_scales]
    if not matrices:
        raise ValueError("a chronological block cannot be empty")
    dimension = matrices[0].shape[0]
    if any(value.shape != (dimension, dimension) for value in matrices):
        raise ValueError("chronological operators must share one square shape")
    if len(scales) != len(matrices) + 1 or any(
            value.shape != (dimension,) for value in scales):
        raise ValueError("one positive state scale is required at every node")
    if any(np.any(value <= 0.0) for value in scales):
        raise ValueError("state scales must be positive")
    product = np.eye(dimension)
    prefix_spectral_radii = []
    prefix_scaled_norms = []
    for index, operator in enumerate(matrices):
        product = operator @ product
        eigenvalues = np.linalg.eigvals(product)
        prefix_spectral_radii.append(float(np.max(np.abs(eigenvalues))))
        scaled = scales[index + 1][:, None] * product / scales[0][None, :]
        prefix_scaled_norms.append(float(np.linalg.norm(scaled, ord=2)))
    scaled_product = scales[-1][:, None] * product / scales[0][None, :]
    endpoint_ratio = scales[-1] / scales[0]
    return {
        "product": product,
        "scaled_product": scaled_product,
        "spectral_radius": prefix_spectral_radii[-1],
        "scaled_norm": prefix_scaled_norms[-1],
        "endpoint_scaling_condition": float(
            np.max(endpoint_ratio) / np.min(endpoint_ratio)),
        "prefix_spectral_radii": np.asarray(prefix_spectral_radii),
        "prefix_scaled_norms": np.asarray(prefix_scaled_norms),
    }


def classify_material_route(
        energy_start, energy_end, gate_work, shape_work, *,
        material_factor=3.0, dominance_ratio=1.25):
    """Classify a material energy decrease from exact additive work."""

    start = _positive(energy_start, "energy_start")
    end = _positive(energy_end, "energy_end")
    material_factor = _positive(material_factor, "material_factor")
    dominance_ratio = _positive(dominance_ratio, "dominance_ratio")
    if material_factor <= 1.0 or dominance_ratio < 1.0:
        raise ValueError("material and dominance factors are outside their domains")
    gate_work = float(gate_work)
    shape_work = float(shape_work)
    if not math.isfinite(gate_work) or not math.isfinite(shape_work):
        raise ValueError("route work must be finite")
    residual = end - start - gate_work - shape_work
    material = math.log(start / end) >= math.log(material_factor)
    if not material:
        label = "unclassified"
    else:
        gate_decrease = max(-gate_work, 0.0)
        shape_decrease = max(-shape_work, 0.0)
        if gate_decrease >= dominance_ratio * shape_decrease and gate_decrease > 0:
            label = "gate-led"
        elif shape_decrease >= dominance_ratio * gate_decrease and shape_decrease > 0:
            label = "shape-led"
        else:
            label = "mixed"
    return {
        "label": label,
        "material": material,
        "log_energy_change": math.log(end / start),
        "identity_residual": residual,
    }
