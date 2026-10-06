"""Regression tests for the source-resolved physical-energy theory."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from confirm.source_resolved_energy import (  # noqa: E402
    admissible_quadratic_radius,
    apply_frozen_plga_cap,
    coordinate_gate_shape_cocycle,
    gate_shape_energy,
    inactive_adamw_position_multiplier,
    layernorm_hessian_bound,
    metric_nonvacuity,
    ordered_energy_envelope,
    partition_adamw_coordinates,
    permutation_symmetric_attribution,
    plga_derivative_bounds,
    select_plga_attenuation_cap,
    source_native_energy_envelope,
    source_work_gram,
    symmetric_gate_shape_transport,
)


def test_gate_shape_factorization_and_symmetric_ledger_are_exact():
    rng = np.random.default_rng(45001)
    shape0 = rng.normal(size=(5, 7, 4))
    shape1 = shape0 + rng.normal(scale=0.2, size=shape0.shape)
    gate0 = rng.normal(size=4)
    gate1 = gate0 + rng.normal(scale=0.1, size=4)
    energy = gate_shape_energy(shape0, gate0)
    transport = symmetric_gate_shape_transport(
        shape0, shape1, gate0, gate1
    )
    assert energy["maximum_abs_factorization_residual"] < 2.0e-14
    assert transport["maximum_abs_transport_residual"] < 2.0e-15
    assert transport["maximum_abs_ledger_residual"] < 5.0e-14


def test_coordinate_cocycle_distinguishes_gate_shape_and_reopening():
    shape0 = np.asarray([[[1.0, 1.0], [-1.0, -1.0]]])
    shape1 = np.asarray([[[2.0, 0.5], [-2.0, -0.5]]])
    gate0 = np.asarray([2.0, 0.0])
    gate1 = np.asarray([1.0, 3.0])
    result = coordinate_gate_shape_cocycle(
        shape0, shape1, gate0, gate1
    )
    assert result["decomposition_defined"][0, 0]
    assert result["maximum_abs_log_residual"] < 2.0e-15
    assert result["gate_log_increment"][0, 0] == pytest.approx(
        2.0 * np.log(0.5)
    )
    assert result["shape_log_increment"][0, 0] == pytest.approx(
        2.0 * np.log(2.0)
    )
    assert result["reopened_at_endpoint"][0, 1]


def test_shapley_efficiency_is_permutation_symmetric():
    vectors = {
        "upstream": np.asarray([1.0, -2.0, 0.5]),
        "weights": np.asarray([-0.25, 1.0, 2.0]),
        "normalization": np.asarray([0.5, 0.25, -1.0]),
    }

    def evaluate(subset):
        result = np.asarray([2.0, 3.0, 4.0])
        for name in subset:
            result = result + vectors[name]
        if {"upstream", "weights"}.issubset(subset):
            result = result + np.asarray([0.2, -0.1, 0.3])
        return result

    first = permutation_symmetric_attribution(tuple(vectors), evaluate)
    second = permutation_symmetric_attribution(
        tuple(reversed(tuple(vectors))), evaluate
    )
    assert first["maximum_abs_efficiency_residual"] < 2.0e-15
    for name in vectors:
        assert np.allclose(
            first["contributions"][name], second["contributions"][name]
        )


def test_source_work_gram_reconstructs_gain_and_native_envelope():
    rng = np.random.default_rng(45002)
    rows = rng.normal(size=(3, 6, 4))
    contributions = {
        "upstream": rng.normal(scale=0.02, size=rows.shape),
        "weights": rng.normal(scale=0.02, size=rows.shape),
        "normalization": rng.normal(scale=0.02, size=rows.shape),
    }
    ledger = source_work_gram(rows, contributions)
    assert ledger["maximum_abs_gain_residual"] < 3.0e-15
    predicted = gate_shape_energy(rows + sum(contributions.values()), np.ones(4))[
        "centered_rows"
    ]
    envelope = source_native_energy_envelope(predicted, [0.0, 0.1, 0.2])
    assert envelope["native_energy_upper"][0] == pytest.approx(
        envelope["predicted_energy"][0]
    )
    assert np.all(
        envelope["native_energy_upper"] >= envelope["predicted_energy"]
    )


def test_ordered_energy_envelope_keeps_expanding_steps_and_charges():
    gains = np.asarray([[1.2, 0.8], [0.5, 1.1], [0.7, 0.6]])
    charges = np.asarray([[0.1, 0.2], [0.01, 0.02], [0.0, 0.03]])
    result = ordered_energy_envelope(gains, charges, [2.0, 3.0])
    manual = gains[1] * (gains[0] * np.asarray([2.0, 3.0]) + charges[0]) + charges[1]
    assert np.allclose(result["energy_envelope"][2], manual)
    assert result["prefix_products"][3, 0] == pytest.approx(1.2 * 0.5 * 0.7)


def test_adamw_strata_are_exhaustive_and_unresolved_fail_smooth_gate():
    moments = np.asarray([1.0, 0.0, 0.0, 2.0])
    result = partition_adamw_coordinates(
        moments, [False, True, False, False]
    )
    assert result["active_count"] == 2
    assert result["structurally_inactive_count"] == 1
    assert result["unresolved_count"] == 1
    assert not result["smooth_full_state_eligible"]
    multiplier = inactive_adamw_position_multiplier(
        [0, 1], learning_rate=0.1, weight_decay=0.2
    )
    assert np.allclose(multiplier, [1.0, 0.98])
    with pytest.raises(ValueError, match="negative"):
        partition_adamw_coordinates([-1.0], [False])


def test_nonvacuity_tube_and_analytic_bounds_are_fail_closed():
    accepted = metric_nonvacuity(1.0e4, [2.0], [3.0], [4.0])
    rejected = metric_nonvacuity(1.0e4 + 1.0, [2.0], [3.0], [4.0])
    assert accepted["eligible"][0]
    assert not rejected["eligible"][0]
    assert layernorm_hessian_bound(64, 1.0e-6) == pytest.approx(750000.0)
    plga = plga_derivative_bounds(-0.002, 0.002, -0.2, 0.2, epsilon=1.0e-9)
    assert plga["base_lower"] == 1.0e-9
    assert plga["z_second_derivative_abs_upper"] > 0.0
    radius = admissible_quadratic_radius(0.8, 0.1, 0.01)
    assert radius["eligible"]
    impossible = admissible_quadratic_radius(1.1, 0.1, 0.01)
    assert not impossible["eligible"]


def test_plga_cap_is_selected_on_construction_and_frozen_on_heldout():
    decision = select_plga_attenuation_cap([0.03, 0.08, 0.09])
    assert decision["selected_cap"] == 0.10
    outcome = apply_frozen_plga_cap([0.09, 0.11], decision["selected_cap"])
    assert outcome.tolist() == [True, False]
    rejected = select_plga_attenuation_cap([0.75])
    assert not rejected["attenuation_admitted"]
    assert rejected["selected_cap"] is None
