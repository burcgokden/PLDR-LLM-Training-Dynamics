from __future__ import annotations

import numpy as np

from confirm.comprehensive_gate_shape import (
    adamw_gate_duhamel,
    coordinate_block_recurrence,
    coordinate_oscillations,
    diameter_slack_summary,
    exact_pair_decomposition,
    finite_registry_geometry,
    iswiglu,
    iswiglu_secant,
    occupied_power_secant,
    permuted_chunk_order,
    two_arm_shape_response,
)


def test_coordinate_oscillation_sandwich_and_zero_routes() -> None:
    gate = np.array([2.0, 0.0, -0.5])
    rows = np.array([
        [0.0, -2.0, 1.0],
        [1.0, 3.0, -1.0],
        [-2.0, 7.0, 2.0],
    ])
    coordinate = coordinate_oscillations(gate, rows)
    geometry = finite_registry_geometry(gate, rows, pair_chunk_size=1)
    assert np.allclose(coordinate["carrier"], [6.0, 0.0, 1.5])
    assert geometry["sandwich_valid"]
    assert coordinate["carrier_linf"] <= geometry["diameter"]
    assert geometry["diameter"] <= coordinate["carrier_l2"]


def test_diameter_slack_allows_pair_expansion_and_maximizer_switch() -> None:
    gate = np.ones(2)
    source = np.array([[0.0, 0.0], [3.0, 0.0], [1.5, 0.0]])
    target = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 2.5]])
    result = diameter_slack_summary(
        gate, source, gate, target, pair_chunk_size=1)
    assert result["expanding_pair_count"] > 0
    assert result["maximizer_changed"]
    assert abs(result["slack_identity_relative_residual"]) < 1e-15
    assert np.isclose(
        result["target_diameter_squared"],
        result["source_diameter_squared"] - result["minimum_exact_slack"],
    )


def test_adamw_duhamel_and_mixed_recurrence_cover_zero_factors() -> None:
    gate0 = np.array([1.0, 0.0])
    factors = np.array([[0.9, 0.8], [0.7, 0.6]])
    innovations = np.array([[0.1, -0.2], [-0.05, 0.1]])
    replay = adamw_gate_duhamel(gate0, factors, innovations)
    gate1 = replay["trajectory"][-1]
    rows0 = np.array([[0.0, 1.0], [2.0, 1.0], [1.0, 1.0]])
    rows1 = np.array([[0.0, 0.0], [1.0, 3.0], [0.5, -1.0]])
    block = coordinate_block_recurrence(
        gate0,
        rows0,
        gate1,
        rows1,
        multipliers=factors,
        innovations=innovations,
    )
    assert replay["identity_relative_residual"] < 1e-15
    assert block["duhamel_endpoint_relative_residual"] < 1e-15
    assert np.all(block["carrier_covered"])
    assert block["gate_force"][1] > 0.0
    assert block["shape_force"][1] > 0.0


def test_exact_endpoint_decomposition_matches_adamw_terms() -> None:
    gate0 = np.array([1.0, -2.0])
    decayed = np.array([0.9, -1.8])
    innovation = np.array([0.2, -0.1])
    gate1 = decayed - innovation
    c0 = np.array([[1.0, 0.5], [-2.0, 1.0]])
    c1 = np.array([[0.7, 0.3], [-1.5, 1.2]])
    value = exact_pair_decomposition(
        gate0, c0, gate1, c1,
        decayed_gate=decayed, innovation=innovation)
    assert np.max(np.abs(value["decomposition_residual"])) < 1e-14
    assert np.max(np.abs(value["gate_balance_residual"])) < 1e-14


def test_two_arm_identity_charges_shape_difference() -> None:
    source = np.array([[1.0, -0.5]])
    baseline_gate = np.array([0.8, 1.2])
    removal_gate = np.array([0.9, 1.1])
    baseline_shape = np.array([[0.9, -0.4]])
    removal_shape = np.array([[1.05, -0.55]])
    result = two_arm_shape_response(
        source,
        baseline_gate,
        baseline_shape,
        removal_gate,
        removal_shape,
    )
    assert np.max(np.abs(result["identity_residual"])) < 1e-15
    assert not np.allclose(
        result["live_response"], result["fixed_shape_response"])


def test_implemented_activation_and_power_secants_are_exact() -> None:
    left = np.array([-2.0, 0.0, 1.5])
    right = np.array([-1.0, 0.5, 2.0])
    activation_secant = iswiglu_secant(left, right)
    assert np.allclose(
        activation_secant * (right - left),
        iswiglu(right) - iswiglu(left),
        rtol=1e-14,
        atol=1e-14,
    )
    positive_left = iswiglu(left) + 3.0
    positive_right = iswiglu(right) + 3.0
    power = np.array([-1.5, 0.0, 2.25])
    power_secant = occupied_power_secant(
        positive_left, positive_right, power)
    assert np.allclose(
        power_secant * (positive_right - positive_left),
        positive_right ** power - positive_left ** power,
        rtol=1e-14,
        atol=1e-14,
    )


def test_seed_keyed_chunk_orders_are_distinct_and_probe_safe() -> None:
    reserved = [3, 7, 11]
    first = permuted_chunk_order(128, 7444, reserved_chunks=reserved)
    second = permuted_chunk_order(128, 8444, reserved_chunks=reserved)
    repeat = permuted_chunk_order(128, 7444, reserved_chunks=reserved)
    assert first["sha256"] != second["sha256"]
    assert first["sha256"] == repeat["sha256"]
    assert not set(reserved) & set(first["order"].tolist())
