"""Regression tests for the comprehensive row-map-collapse kernels."""

from __future__ import annotations

import math

import numpy as np
import pytest

from confirm.row_map_dynamics import (
    classify_material_route,
    covered_domain_bound,
    decay_loss_log_split,
    dimensionless_state_scale,
    gate_duhamel,
    gate_energy_dissipation,
    graph_contrast,
    layernorm_jacobian,
    log_route_decomposition,
    relative_matrix_residual,
    scaled_chronological_diagnostics,
    shape_flux,
)


def test_connected_graph_energy_encloses_vertex_diameter():
    rng = np.random.default_rng(3601)
    shape = rng.normal(size=(9, 5))
    gamma = rng.normal(size=5)
    edges = np.asarray([(index, index + 1) for index in range(8)])
    weights = rng.uniform(0.2, 1.3, size=8)
    result = graph_contrast(gamma, shape, edges, weights)
    assert result["energy_identity_residual"] <= 2e-13 * max(
        result["energy"], 1.0)
    assert result["vertex_diameter"] <= result["diameter_bound"] + 1e-12
    assert result["graph_diameter"] == 8
    assert covered_domain_bound(
        result["energy"], result["graph_diameter"],
        result["minimum_weight"], 0.02, 1.7,
    ) == pytest.approx(result["diameter_bound"] + 0.068)


def test_graph_kernel_rejects_disconnected_registry():
    with pytest.raises(ValueError, match="connected"):
        graph_contrast(
            np.ones(2), np.ones((4, 2)), [[0, 1], [2, 3]], [1.0, 1.0])


def test_log_routes_reconstruct_mixed_change_and_record_zero():
    gamma = np.asarray([[2.0, 1.0], [1.5, 0.0], [1.0, 0.0]])
    q = np.asarray([[4.0, 3.0], [2.0, 2.0], [1.0, 1.0]])
    result = log_route_decomposition(gamma, q)
    assert np.nanmax(np.abs(result["log_identity_residual"])) <= 2e-15
    assert result["first_zero_node"].tolist() == [-1, 1]
    assert result["positive_step_mask"][:, 0].all()
    assert not result["positive_step_mask"][:, 1].any()


def test_exact_gate_dissipation_with_shape_work():
    rng = np.random.default_rng(3602)
    gamma = rng.normal(size=7)
    q = rng.uniform(0.2, 1.0, size=7)
    q_next = rng.uniform(0.2, 1.0, size=7)
    adaptive = rng.normal(scale=0.04, size=7)
    result = gate_energy_dissipation(
        gamma, q, adaptive, 7.5e-4, 0.1, q_next=q_next)
    assert result["fixed_q_identity_residual"] == pytest.approx(0.0, abs=2e-15)
    assert result["total_identity_residual"] == pytest.approx(0.0, abs=2e-15)


def test_duhamel_reconstructs_time_varying_decay_masks():
    rng = np.random.default_rng(3603)
    initial = rng.normal(size=6)
    adaptive = rng.normal(scale=0.1, size=(5, 6))
    rates = np.linspace(2e-4, 8e-4, 5)
    masks = rng.integers(0, 2, size=(5, 6))
    result = gate_duhamel(initial, adaptive, rates, 0.1, masks)
    assert np.allclose(result["reconstructed"], result["history"][-1])
    assert np.max(np.abs(result["identity_residual"])) <= 2e-16


def test_decay_loss_log_split_is_exact_for_signed_gates():
    gamma = np.asarray([-0.7, 0.3, 1.1])
    adaptive = np.asarray([0.02, -0.01, 0.03])
    result = decay_loss_log_split(gamma, adaptive, 7.5e-4, 0.1)
    assert result["valid_mask"].all()
    assert np.max(np.abs(result["identity_residual"])) <= 2e-15


def test_shape_flux_identity_includes_finite_remainder():
    rng = np.random.default_rng(3604)
    shape = rng.normal(size=(11, 4))
    linear = rng.normal(scale=0.02, size=shape.shape)
    remainder = rng.normal(scale=1e-3, size=shape.shape)
    result = shape_flux(shape, linear, remainder)
    assert np.max(np.abs(result["identity_residual"])) <= 2e-14


def test_layernorm_jacobian_matches_directional_difference():
    rng = np.random.default_rng(3605)
    value = rng.normal(size=(3, 5))
    direction = rng.normal(size=value.shape)
    epsilon = 1e-6
    analytic = np.einsum(
        "rij,rj->ri", layernorm_jacobian(value, epsilon=epsilon), direction)

    def normalized(argument):
        centered = argument - argument.mean(axis=1, keepdims=True)
        return centered / np.sqrt(
            epsilon + np.mean(centered * centered, axis=1, keepdims=True))

    step = 1e-6
    numerical = (
        normalized(value + step * direction)
        - normalized(value - step * direction)) / (2.0 * step)
    assert np.allclose(analytic, numerical, rtol=3e-7, atol=2e-9)


def test_scaled_chronological_diagnostics_use_endpoint_coordinates():
    operators = [
        np.asarray([[1.1, 0.2], [0.0, 0.7]]),
        np.asarray([[0.6, 0.0], [0.1, 0.8]]),
    ]
    scales = [
        np.asarray([1.0, 2.0]),
        np.asarray([1.5, 1.0]),
        np.asarray([2.0, 0.5]),
    ]
    result = scaled_chronological_diagnostics(operators, scales)
    direct = operators[1] @ operators[0]
    assert np.allclose(result["product"], direct)
    assert np.allclose(
        result["scaled_product"], scales[-1][:, None] * direct / scales[0])
    assert result["spectral_radius"] == pytest.approx(
        max(abs(np.linalg.eigvals(direct))))


def test_dimensionless_scale_and_relative_zero_policy():
    scale = dimensionless_state_scale(np.asarray([0.0, 0.25]), epsilon=1e-5)
    assert scale.shape == (6,)
    assert np.all(scale > 0.0)
    zeros = np.zeros((3, 3))
    assert relative_matrix_residual(zeros, zeros) == 0.0
    assert relative_matrix_residual(np.eye(3), np.eye(3) + 1e-10) < 1e-9


def test_material_route_rule_is_protocol_owned():
    gate = classify_material_route(9.0, 2.0, -6.0, -1.0)
    shape = classify_material_route(9.0, 2.0, -1.0, -6.0)
    stationary = classify_material_route(9.0, 8.0, -0.7, -0.3)
    expansion = classify_material_route(1.0, 4.0, 3.0, 0.0)
    assert gate["label"] == "gate-led"
    assert shape["label"] == "shape-led"
    assert stationary["label"] == "unclassified"
    assert expansion["label"] == "unclassified"
    assert not expansion["material"]
    assert math.isclose(gate["identity_residual"], 0.0)
