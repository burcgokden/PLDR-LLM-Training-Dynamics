"""Regression tests for the crossing-safe finite-increment theory."""

from __future__ import annotations

import math

import numpy as np
import pytest

from confirm.finite_increment import (
    absolute_diameter_enclosure,
    affine_increment,
    affine_layernorm_increment,
    directed_state_margin_bounds,
    exact_diameter_slack,
    finite_source_unroll,
    gate_shape_state_margin,
    iswiglu,
    layernorm_shape,
    outward_pair_energy_bounds,
    per_map_diameters,
    positive_power,
    product_convolution,
    representative_crossing_qualification,
    two_variable_power_increment,
)


def test_two_variable_power_telescope_crosses_zero_exactly():
    z0 = np.asarray([-3e-4, -2e-8, 0.0, 2e-5, 0.7])
    z1 = np.asarray([2e-4, 4e-8, -1e-8, -4e-5, 0.7])
    p0 = np.asarray([-0.10, 0.0, 0.05, 0.11, -0.03])
    p1 = np.asarray([-0.08, 0.02, 0.05, 0.09, -0.03])
    result = two_variable_power_increment(z0, z1, p0, p1)
    direct = np.asarray(result["direct_increment"])
    scale = max(np.max(np.abs(direct)), 1.0)
    assert result["strict_crossing_count"] == 3
    assert result["touching_zero_count"] == 1
    assert result["minimum_base"] >= 1.0e-9
    assert np.allclose(result["base_first_increment"], direct, rtol=2e-14, atol=2e-14 * scale)
    assert np.allclose(result["exponent_first_increment"], direct, rtol=2e-14, atol=2e-14 * scale)


def test_power_helper_rejects_nonpositive_base():
    with pytest.raises(ValueError, match="strictly positive"):
        positive_power(np.asarray([1.0, 0.0]), np.asarray([0.2, 0.3]))


def test_affine_finite_increment_has_all_three_sources():
    rng = np.random.default_rng(42002)
    x0 = rng.normal(size=(3, 5))
    x1 = x0 + rng.normal(scale=0.03, size=x0.shape)
    w0 = rng.normal(size=(3, 4, 3))
    w1 = w0 + rng.normal(scale=0.02, size=w0.shape)
    b0 = rng.normal(size=(3, 4, 5))
    b1 = b0 + rng.normal(scale=0.01, size=b0.shape)
    result = affine_increment(x0, x1, w0, w1, b0, b1)
    assert np.allclose(
        result["reconstructed_increment"], result["direct_increment"],
        rtol=2e-14, atol=2e-14,
    )


def test_affine_layernorm_increment_uses_stable_equal_radius_chord():
    rng = np.random.default_rng(42003)
    x0 = rng.normal(size=11)
    x1 = -x0 + 2.0 * np.mean(x0)
    gamma0 = rng.normal(size=11)
    gamma1 = gamma0 + rng.normal(scale=0.05, size=11)
    beta0 = rng.normal(size=11)
    beta1 = beta0 + rng.normal(scale=0.05, size=11)
    result = affine_layernorm_increment(
        x0, x1, gamma0, gamma1, beta0, beta1
    )
    direct = (
        gamma1 * layernorm_shape(x1) + beta1
        - gamma0 * layernorm_shape(x0) - beta0
    )
    assert np.allclose(result["reconstructed_increment"], direct, rtol=2e-14, atol=2e-14)


def test_eight_unit_source_unroll_matches_recursive_execution():
    rng = np.random.default_rng(42004)
    initial = rng.normal(size=9)
    maps = [rng.normal(scale=0.15, size=(9, 9)) for _ in range(8)]
    sources = [rng.normal(scale=0.05, size=9) for _ in range(8)]
    result = finite_source_unroll(initial, maps, sources)
    assert len(result["states"]) == 9
    assert len(result["transported_sources"]) == 8
    assert np.allclose(result["unrolled"], result["recursive"], rtol=2e-14, atol=2e-14)


def test_per_map_diameter_never_uses_cross_map_pair():
    rows = np.asarray([
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        [[100.0, 100.0], [101.0, 100.0], [100.0, 101.0]],
    ])
    result = per_map_diameters(rows)
    assert result["pairs_per_map"] == 3
    assert result["cross_map_pairs_formed"] is False
    assert np.allclose(result["diameter_squared"], [2.0, 2.0])
    assert result["aggregate_diameter_squared"] == pytest.approx(2.0)


def test_exact_diameter_slack_handles_a_changing_maximizer():
    source = np.asarray([[0.0], [2.0], [3.0]])
    endpoint = np.asarray([[0.0], [2.7], [2.8]])
    result = exact_diameter_slack(source, endpoint)
    assert result["source_diameter_squared"] == pytest.approx(9.0)
    assert result["endpoint_diameter_squared"] == pytest.approx(7.84)
    assert result["identity_residual"] <= 2e-15
    assert result["strict_contraction"]


def test_gate_shape_margin_is_exact_and_covers_endpoint():
    rng = np.random.default_rng(42005)
    shape0 = rng.normal(size=(6, 7))
    shape1 = shape0 + rng.normal(scale=0.01, size=shape0.shape)
    gamma0 = rng.normal(size=7)
    decay = 0.9999
    innovation = rng.normal(scale=0.002, size=7)
    gamma1 = decay * gamma0 - innovation
    result = gate_shape_state_margin(
        shape0,
        shape1,
        gamma0,
        gamma1,
        decay_multiplier=decay,
    )
    scale = max(result["source_diameter_squared"], 1.0)
    assert result["gate_balance_max_abs_residual"] <= 1e-13 * scale
    assert result["shape_bound_minimum_slack"] >= -1e-13 * scale
    assert result["state_bound_covers_endpoint"]


def test_directed_pair_energy_bounds_enclose_float64_results():
    rows = np.asarray([
        [0.0, 0.0, 0.0],
        [0.1, -0.2, 0.3],
        [0.1, -0.2, 0.3],
        [1.0e120, -1.0e-120, 0.25],
    ])
    pairs, energies = exact_diameter_slack(rows, rows)["pairs"], (
        exact_diameter_slack(rows, rows)["source_energy"]
    )
    bounds = outward_pair_energy_bounds(rows)
    assert np.array_equal(bounds["pairs"], pairs)
    assert np.all(bounds["lower"] <= energies)
    assert np.all(energies <= bounds["upper"])
    assert np.all(bounds["lower"] >= 0.0)


def test_directed_state_margin_encloses_rows_and_certifies_decay():
    rng = np.random.default_rng(42006)
    shape = rng.normal(size=(8, 9))
    gamma0 = np.ones(9)
    gamma1 = np.full(9, 0.98)
    rows0 = shape * gamma0
    rows1 = shape * gamma1
    result = directed_state_margin_bounds(
        shape,
        shape,
        gamma0,
        gamma1,
        source_rows=rows0,
        endpoint_rows=rows1,
    )
    source = exact_diameter_slack(rows0, rows1)
    assert np.all(result["source_energy_lower"] <= source["source_energy"])
    assert np.all(source["source_energy"] <= result["source_energy_upper"])
    assert np.all(result["endpoint_energy_lower"] <= source["endpoint_energy"])
    assert np.all(source["endpoint_energy"] <= result["endpoint_energy_upper"])
    assert result["state_bound_covers_endpoint"]
    assert result["strict_contraction_certified"]


def test_absolute_diameter_fallback_survives_zero_denominator():
    result = absolute_diameter_enclosure(
        [-1e-8, 0.0], [2e-8, 3e-8], source_lower=0.0, source_upper=1e-8
    )
    assert result["absolute_lower"] == 0.0
    assert result["absolute_upper"] == pytest.approx(3e-8)
    assert result["ratio_denominator_valid"] is False
    assert result["lower_ratio"] is None and result["upper_ratio"] is None


def test_product_convolution_preserves_chronological_order():
    result = product_convolution(2.0, [0.5, 1.2, 0.25], [0.1, 0.2, 0.0])
    direct = 0.25 * (1.2 * (0.5 * 2.0 + 0.1) + 0.2)
    assert result["final"] == pytest.approx(direct)
    assert result["envelope"] == pytest.approx([2.0, 1.1, 1.52, 0.38])


def test_representative_crossing_qualification_passes():
    report = representative_crossing_qualification()
    assert report["passed"]
    assert report["plga"]["strict_crossing_count"] == 3
    assert math.isfinite(report["plga"]["maximum_abs_z_secant"])


def test_iswiglu_is_nonnegative_and_zero_at_the_crossing():
    value = iswiglu(np.asarray([-2.0, -1e-8, 0.0, 1e-8, 2.0]))
    assert np.all(value >= 0.0)
    assert value[2] == 0.0
