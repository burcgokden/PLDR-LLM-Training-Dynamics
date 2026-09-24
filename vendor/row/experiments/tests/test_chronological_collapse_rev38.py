"""Regression tests for the chronological row-map collapse kernels."""

from __future__ import annotations

import numpy as np
import pytest

from confirm.chronological_collapse import (
    adamw_chronological_direction,
    chronological_pair_certificate,
    diameter_upper_from_pair_envelope,
    executed_pair_gate_balance,
    iterate_pair_envelope,
    occupied_power_secant,
    pair_contrasts,
    pair_energies,
    provenance_tolerance,
    rectangular_score_transfer,
    source_removal_response,
    verify_shape_enclosure,
)


def test_chronological_unroll_matches_constant_gradient_adam_state():
    beta1 = 0.9
    beta2 = 0.95
    step = 100
    history_length = 4
    gradient = np.asarray([0.3, -0.7, 1.2, -0.05])
    history = np.repeat(gradient[None, :], history_length, axis=0)
    first_tail = (1.0 - beta1 ** (step - history_length)) * gradient
    second_before = (1.0 - beta2 ** (step - 1)) * gradient * gradient
    result = adamw_chronological_direction(
        history,
        first_tail,
        second_before,
        beta1=beta1,
        beta2=beta2,
        epsilon=1e-5,
        optimizer_step_after=step,
    )
    assert result["first_moment_hat_after"] == pytest.approx(gradient)
    assert result["second_moment_hat_after"] == pytest.approx(
        gradient * gradient)
    assert result["direction"] == pytest.approx(
        gradient / (np.abs(gradient) + 1e-5))
    assert result["reconstruction_relative_residual"] <= 3e-16


def test_executed_pair_balance_is_exact_with_decay_mask():
    rng = np.random.default_rng(3801)
    rows = rng.normal(size=(7, 5))
    contrasts, _pairs = pair_contrasts(rows)
    gate = rng.normal(size=5)
    direction = rng.normal(size=5)
    balance = executed_pair_gate_balance(
        gate,
        contrasts,
        direction,
        learning_rate=7.5e-4,
        weight_decay=0.1,
        decay_mask=np.asarray([1.0, 1.0, 0.0, 1.0, 0.0]),
    )
    assert np.max(balance["identity_relative_residual"]) <= 2e-15
    direct = pair_energies(balance["gate_next"], contrasts)
    assert direct == pytest.approx(balance["energy_after_fixed_shape"])


def test_directional_shape_enclosure_covers_bounded_remainder():
    rng = np.random.default_rng(3802)
    rows = rng.normal(size=(6, 4))
    contrast, _pairs = pair_contrasts(rows)
    gate = rng.normal(size=4)
    direction = 0.2 * gate
    jvp_increment = -0.03 * contrast
    radii = np.full(len(contrast), 2e-3)
    certificate = chronological_pair_certificate(
        gate,
        contrast,
        direction,
        jvp_increment,
        radii,
        np.zeros(len(contrast)),
        learning_rate=0.01,
        weight_decay=0.1,
    )
    remainder = rng.normal(size=contrast.shape)
    remainder *= (
        0.9 * radii / np.maximum(
            np.linalg.norm(remainder, axis=1), np.finfo(float).tiny)
    )[:, None]
    coverage = verify_shape_enclosure(
        contrast,
        jvp_increment + remainder,
        certificate,
        certificate["gate_next"],
    )
    assert np.all(coverage["covered"])
    assert np.min(coverage["coverage_slack"]) >= -2e-14


def test_positive_certificate_gives_contracting_all_pairs_envelope():
    rng = np.random.default_rng(3803)
    rows = rng.normal(size=(8, 6))
    contrast, pairs = pair_contrasts(rows)
    gate = rng.uniform(0.4, 1.2, size=6)
    direction = 0.25 * gate
    certificate = chronological_pair_certificate(
        gate,
        contrast,
        direction,
        -0.02 * contrast,
        np.zeros(len(pairs)),
        np.zeros(len(pairs)),
        learning_rate=0.02,
        weight_decay=0.1,
    )
    assert np.all(certificate["positive_margin"])
    assert np.max(certificate["affine_factor"]) < 1.0
    factors = np.repeat(certificate["affine_factor"][None, :], 5, axis=0)
    forcing = np.zeros_like(factors)
    envelope = iterate_pair_envelope(
        certificate["energy_before"], factors, forcing)
    diameter = diameter_upper_from_pair_envelope(envelope)
    assert np.all(np.diff(diameter) < 0.0)
    assert diameter[-1] < diameter[0]


def test_pair_envelope_retains_chronological_order():
    initial = np.asarray([2.0, 3.0])
    factors = np.asarray([[0.5, 0.8], [0.2, 0.4]])
    forcing = np.asarray([[1.0, 0.5], [0.1, 0.2]])
    envelope = iterate_pair_envelope(initial, factors, forcing)
    assert envelope[1] == pytest.approx([2.0, 2.9])
    assert envelope[2] == pytest.approx([0.5, 1.36])
    reversed_envelope = iterate_pair_envelope(
        initial, factors[::-1], forcing[::-1])
    assert not np.allclose(envelope[-1], reversed_envelope[-1])


def test_rectangular_score_transfer_is_exact():
    rng = np.random.default_rng(3804)
    q0 = rng.normal(size=(11, 4))
    q1 = rng.normal(size=(11, 4))
    g0 = rng.normal(size=(4, 4))
    g1 = rng.normal(size=(4, 4))
    k0 = rng.normal(size=(7, 4))
    k1 = rng.normal(size=(7, 4))
    result = rectangular_score_transfer(q0, q1, g0, g1, k0, k1)
    assert result["reference_score"].shape == (11, 7)
    assert result["identity_relative_residual"] <= 2e-15


@pytest.mark.parametrize("power", [-2.5, -1.0, 0.0, 0.5, 1.0, 3.2])
def test_occupied_power_secant_supports_every_real_sign(power):
    left = np.asarray([0.2, 1.0, 3.0])
    right = np.asarray([0.4, 1.0, 2.0])
    exponent = np.full(3, power)
    result = occupied_power_secant(left, right, exponent)
    assert np.max(result["identity_relative_residual"]) <= 2e-14
    assert np.all(np.isfinite(result["derivative_bound"]))


def test_source_removal_response_supplies_sign_and_magnitude():
    rng = np.random.default_rng(3805)
    contrast, _pairs = pair_contrasts(rng.normal(size=(5, 3)))
    baseline = rng.normal(size=3)
    source = rng.normal(size=3)
    eta = 0.02
    prediction = source_removal_response(
        baseline, contrast, source, learning_rate=eta)
    baseline_energy = pair_energies(baseline, contrast)
    removed_energy = pair_energies(baseline + eta * source, contrast)
    realized = removed_energy - baseline_energy
    assert prediction["predicted_energy_response"] == pytest.approx(realized)
    assert prediction["predicted_magnitude"] == pytest.approx(np.abs(realized))


def test_validation_rejects_bad_pairs_and_bad_provenance():
    rows = np.zeros((3, 2))
    with pytest.raises(ValueError, match="pair"):
        pair_contrasts(rows, np.asarray([[0, 1], [0, 1]]))
    assert provenance_tolerance("float32", 64, 10.0) > (
        provenance_tolerance("float64", 64, 10.0))
    with pytest.raises(ValueError, match="precision"):
        provenance_tolerance("int64", 2, 1.0)

