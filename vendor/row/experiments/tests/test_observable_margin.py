"""Focused algebra regressions for the observable-margin theory."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from confirm.gate_shape import plga_secant_chain
from confirm.observable_margin import (
    block_affine_convolution,
    bridge_sector_replay,
    certified_bridge_margin,
    deterministic_knn_mst_graph,
    effective_resistance_bound,
    effective_resistance_constant,
    graph_energy,
    lifted_affine_convolution,
    native_roundoff_charge,
    pack_union_graph,
    shape_from_edge_contrasts,
    shape_metric_secant,
    shape_secant,
    three_factor_score_secant,
    unpack_union_graph,
)
from confirm.observable_margin_controls import _counterfactual_shadow_gate
from confirm.observable_margin_live import _bridge_directions


def test_single_union_graph_detects_cross_block_nonconstancy_and_replays_digest():
    # Each four-row block is internally constant. Only a connected union graph
    # can distinguish the two constants and therefore test one shared row map.
    physical = np.asarray([
        [0.0, 0.0], [0.0, 0.1], [0.1, 0.0], [0.1, 0.1],
        [10.0, 0.0], [10.0, 0.1], [10.1, 0.0], [10.1, 0.1],
    ])
    graph = deterministic_knn_mst_graph(physical, neighbors=2)
    edges = graph["edges"]
    assert any(left < 4 <= right for left, right in edges)
    assert np.all(graph["weights"] == 1.0 / len(edges))
    shape = np.vstack((np.zeros((4, 2)), np.ones((4, 2))))
    gamma = np.asarray([1.0, -0.5])
    geometry = graph_energy(gamma, shape, edges, graph["weights"])
    assert geometry["energy"] > 0.0
    resistance = effective_resistance_constant(
        8, edges, graph["weights"])["effective_resistance_max"]
    assert geometry["vertex_diameter"] <= effective_resistance_bound(
        geometry["energy"], resistance) + 1e-12
    packed = pack_union_graph(graph)
    replay = unpack_union_graph(
        packed["vertex_count"], packed["edges"], packed["weights"],
        rule=packed["rule"], neighbors=packed["neighbors"],
        expected_sha256=packed["union_graph_sha256"])
    contrasts = shape[edges[:, 0]] - shape[edges[:, 1]]
    reconstructed = shape_from_edge_contrasts(8, edges, contrasts)
    assert reconstructed["relative_cycle_residual"] <= 1e-15
    assert replay["graph_sha256"] == packed["union_graph_sha256"]
    with pytest.raises(ValueError, match="digest"):
        unpack_union_graph(
            8, edges, graph["weights"], rule=graph["rule"], neighbors=2,
            expected_sha256="0" * 64)


def test_exact_signed_shape_secant_reconstructs_energy_change():
    rng = np.random.default_rng(3701)
    shape = rng.normal(size=(9, 5))
    increment = rng.normal(scale=0.03, size=(9, 5))
    gamma = rng.normal(size=5)
    graph = deterministic_knn_mst_graph(shape, neighbors=3)
    source = graph_energy(gamma, shape, graph["edges"], graph["weights"])
    target = graph_energy(
        gamma, shape + increment, graph["edges"], graph["weights"])
    secant = shape_secant(
        gamma, shape, increment, graph["edges"], graph["weights"])
    assert target["energy"] - source["energy"] == pytest.approx(
        -secant["signed_linear_gain"] + secant["quadratic_charge"],
        rel=1e-13, abs=1e-13)
    source_edge = shape[graph["edges"][:, 0]] - shape[graph["edges"][:, 1]]
    increment_edge = (
        increment[graph["edges"][:, 0]] - increment[graph["edges"][:, 1]])
    linear_q = 2.0 * np.sum(
        graph["weights"][:, None] * source_edge * increment_edge, axis=0)
    quadratic_q = np.sum(
        graph["weights"][:, None] * increment_edge * increment_edge, axis=0)
    compact = shape_metric_secant(
        gamma, source["q"], target["q"], linear_q, quadratic_q)
    assert compact["relative_reconstruction_residual"] <= 2e-15
    assert compact["quadratic_charge"] >= 0.0


def test_float64_shadow_moments_sectors_and_native_charge_are_exact():
    gate = torch.tensor([0.4, -0.3, 0.2], dtype=torch.float32)
    raw = torch.tensor([0.7, -1.2, 0.15], dtype=torch.float32)
    clipped = torch.clamp(raw, -1.0, 1.0)
    before = {
        "exp_avg": torch.tensor([0.1, 0.02, -0.03], dtype=torch.float32),
        "exp_avg_sq": torch.tensor([0.2, 0.3, 0.1], dtype=torch.float32),
    }
    fisher = torch.tensor([0.05, -0.1, 0.02], dtype=torch.float64)
    signed = torch.tensor([0.03, 0.04, -0.01], dtype=torch.float64)
    force = torch.tensor([0.2, -0.3, 0.05], dtype=torch.float64)
    nonlinear = raw.double() - fisher - signed - force
    components = torch.stack((fisher, signed, force, nonlinear))[None, :, :]
    source = {
        "aggregate": (fisher, signed, force, nonlinear),
        "weights": torch.ones(1, dtype=torch.float64),
        "source_counts": torch.tensor([17.0], dtype=torch.float64),
        "components": components,
    }
    group = {"betas": (0.9, 0.95), "eps": 1e-5, "lr": 7.5e-4,
             "weight_decay": 0.1}
    shadow = _bridge_directions(
        gate, raw, clipped, before, 9, group, source)
    beta1, beta2 = group["betas"]
    expected_m = beta1 * before["exp_avg"].double() + (1-beta1) * clipped.double()
    expected_v = beta2 * before["exp_avg_sq"].double() + (1-beta2) * clipped.double() ** 2
    expected_u = (
        expected_m / (1-beta1 ** 9)
        / (torch.sqrt(expected_v / (1-beta2 ** 9)) + group["eps"]))
    assert torch.equal(shadow["first_moment_shadow_after"], expected_m)
    assert torch.equal(shadow["second_moment_shadow_after"], expected_v)
    assert torch.allclose(shadow["adaptive_direction"], expected_u, rtol=0, atol=1e-15)
    sectors = shadow["sector_directions"].numpy()
    adaptive = shadow["adaptive_direction"].numpy()
    assert np.allclose(sectors.sum(axis=0), adaptive, rtol=0, atol=2e-16)
    q = np.asarray([0.4, 0.2, 0.6])
    replay = bridge_sector_replay(gate.numpy(), q, adaptive, sectors)
    assert replay["relative_direction_residual"] <= 2e-16
    certified = certified_bridge_margin(
        group["lr"], replay["sector_gain_without_learning_rate"])
    force_names = ("force", "nonlinear", "clip", "preconditioner", "lag")
    expected_charge = group["lr"] * sum(abs(
        replay["sector_gain_without_learning_rate"][name]) for name in force_names)
    assert certified["remainder_absolute_charge"] == pytest.approx(expected_charge)
    exact_successor = gate.double().numpy() - group["lr"] * (
        group["weight_decay"] * gate.double().numpy() + adaptive)
    native_successor = exact_successor.astype(np.float32).astype(np.float64)
    charge = native_roundoff_charge(exact_successor, native_successor, q)
    assert charge["floating_point_charge"] >= 0.0
    assert charge["relative_reconstruction_residual"] <= 1e-8


def test_ordered_affine_convolution_keeps_zero_intermediate_energy_and_order():
    first = block_affine_convolution(0.0, [0.5, 0.25], [1.0, 0.0])
    permuted = block_affine_convolution(0.0, [0.5, 0.25], [0.0, 1.0])
    assert first["endpoint_bound"] == pytest.approx(0.25)
    assert permuted["endpoint_bound"] == pytest.approx(1.0)
    zero_then_forced = block_affine_convolution(2.0, [0.0, 0.5], [0.0, 1.0])
    assert zero_then_forced["trajectory_bound"].tolist() == [2.0, 0.0, 1.0]



def test_lifted_affine_v7_is_chronological_and_noncommuting():
    initial = np.asarray([1.0, -1.0])
    first = np.asarray([[1.0, 1.0], [0.0, 1.0]])
    second = np.asarray([[1.0, 0.0], [1.0, 1.0]])
    forcing_first = np.asarray([0.25, 0.0])
    forcing_second = np.asarray([0.0, 0.5])
    result = lifted_affine_convolution(
        initial, [first, second], [forcing_first, forcing_second])
    explicit_first = first @ initial + forcing_first
    explicit_second = second @ explicit_first + forcing_second
    assert np.array_equal(result["trajectory"][1], explicit_first)
    assert np.array_equal(result["endpoint"], explicit_second)
    assert np.array_equal(result["operator_product"], second @ first)
    assert np.array_equal(
        result["forcing_convolution"], second @ forcing_first + forcing_second)
    reversed_endpoint = first @ (second @ initial + forcing_second) + forcing_first
    assert not np.array_equal(result["endpoint"], reversed_endpoint)


def test_plga_same_node_secant_and_full_q_g_k_score_transfer_are_exact():
    rng = np.random.default_rng(3702)
    dimension = 4
    generator = rng.normal(scale=0.2, size=(dimension, dimension))
    collapsed = np.repeat(generator[0:1], dimension, axis=0)
    weight = np.eye(dimension) * 0.3
    bias = np.ones((dimension, dimension)) * 0.4
    powers = rng.uniform(-0.7, 1.4, size=(dimension, dimension))
    coupling = np.eye(dimension) * 0.2
    coupling_bias = rng.normal(scale=0.01, size=(dimension, dimension))
    replay = plga_secant_chain(
        collapsed, generator, weight, bias, powers, coupling, coupling_bias)
    assert np.min(replay["metric_left"]) > 0.0
    assert np.min(replay["metric_right"]) > 0.0
    assert replay["identity_residual"] <= 1e-13

    query_reference = rng.normal(size=(dimension, dimension))
    query_candidate = query_reference + rng.normal(
        scale=0.03, size=(dimension, dimension))
    key_reference = rng.normal(size=(dimension, dimension))
    key_candidate = key_reference + rng.normal(
        scale=0.03, size=(dimension, dimension))
    chain = three_factor_score_secant(
        query_reference, query_candidate, replay["curvature_left"],
        replay["curvature_right"], key_reference, key_candidate)
    assert chain["identity_residual"] <= 1e-13
    assert all(
        np.linalg.norm(chain[name]) > 0.0
        for name in ("query_term", "generator_term", "key_term"))
    endpoints = [
        query_reference, query_candidate, replay["curvature_left"],
        replay["curvature_right"], key_reference, key_candidate]
    for endpoint_index in (1, 3, 5):
        mutated = [value.copy() for value in endpoints]
        mutated[endpoint_index][0, 0] += 0.2
        changed = three_factor_score_secant(*mutated)
        assert not np.allclose(
            changed["realized_difference"], chain["predicted_difference"],
            rtol=1e-12, atol=1e-12)

    controlled = three_factor_score_secant(
        query_reference, query_reference, replay["curvature_left"],
        replay["curvature_right"], key_reference, key_reference)
    assert np.array_equal(controlled["query_term"], np.zeros_like(
        controlled["query_term"]))
    assert np.array_equal(controlled["key_term"], np.zeros_like(
        controlled["key_term"]))
    expected = (
        query_reference @ replay["realized_difference"] @ key_reference.T
        / math.sqrt(dimension))
    assert np.allclose(
        controlled["generator_term"], expected, rtol=1e-13, atol=1e-13)


def test_intervention_shadow_does_not_charge_deliberate_suppression():
    gate = np.asarray([1.0, -2.0, 0.5])
    adaptive = np.asarray([0.3, -0.4, 0.2])
    learning_rate = 0.1
    weight_decay = 0.2
    baseline = _counterfactual_shadow_gate(
        gate, adaptive, learning_rate, weight_decay, "baseline")
    adaptive_suppressed = _counterfactual_shadow_gate(
        gate, adaptive, learning_rate, weight_decay,
        "aggregate_adaptive_gate_displacement_suppression")
    decay_suppressed = _counterfactual_shadow_gate(
        gate, adaptive, learning_rate, weight_decay,
        "decay_source_removal")
    assert np.allclose(
        adaptive_suppressed,
        gate - learning_rate * weight_decay * gate,
        rtol=0.0, atol=0.0)
    assert np.allclose(
        decay_suppressed, gate - learning_rate * adaptive,
        rtol=0.0, atol=0.0)
    q_next = np.ones_like(gate)
    assert native_roundoff_charge(
        adaptive_suppressed, adaptive_suppressed, q_next)[
            "floating_point_charge"] == 0.0
    assert native_roundoff_charge(
        baseline, adaptive_suppressed, q_next)[
            "floating_point_charge"] > 0.0
