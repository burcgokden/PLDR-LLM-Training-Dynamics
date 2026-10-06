"""Regression tests for the gate-shape collapse and capture kernels."""

from __future__ import annotations

import math

import numpy as np
import pytest

from confirm.gate_shape import (
    adamw_gate_block,
    centered_logit_margin_preserved,
    chronological_products,
    finite_difference_adamw_block,
    gate_loss_hessian_split,
    gate_shape_energy_increment,
    gate_shape_output,
    global_gate_oscillation_bound,
    normalized_shape,
    ordered_capture_envelope,
    pairwise_softmax_energy,
    plga_secant_chain,
    residual_scaled_spectrum,
    row_contrast_geometry,
    softmax_fisher,
    source_owned_energy_envelope,
)


def test_layernorm_gate_factorization_and_global_bound():
    rng = np.random.default_rng(3501)
    preactivation = rng.normal(size=(19, 7))
    gamma = rng.normal(size=7)
    beta = rng.normal(size=7)
    shape = normalized_shape(preactivation, epsilon=1e-6)
    output = gate_shape_output(beta, gamma, shape)
    centered = preactivation - preactivation.mean(axis=-1, keepdims=True)
    direct = beta + gamma * centered / np.sqrt(
        1e-6 + np.mean(centered * centered, axis=-1, keepdims=True))
    assert np.allclose(output, direct, rtol=1e-13, atol=1e-13)
    assert np.all(np.sum(shape * shape, axis=1) < 7.0)
    diameter = max(
        np.linalg.norm(left - right)
        for left in output for right in output
    )
    assert diameter <= global_gate_oscillation_bound(gamma) + 1e-12
    assert np.allclose(gate_shape_output(beta, 0.0 * gamma, shape), beta)


def test_gate_scaling_scales_every_row_difference():
    rng = np.random.default_rng(3502)
    shape = normalized_shape(rng.normal(size=(11, 5)))
    gamma = rng.normal(size=5)
    beta = rng.normal(size=5)
    scale = -1.7
    output = gate_shape_output(beta, gamma, shape)
    scaled = gate_shape_output(beta, scale * gamma, shape)
    assert np.allclose(
        scaled[1:] - scaled[:-1],
        scale * (output[1:] - output[:-1]),
        rtol=1e-13,
        atol=1e-13,
    )


def test_contrast_energy_and_gate_shape_increment_are_exact():
    rng = np.random.default_rng(3503)
    left = normalized_shape(rng.normal(size=(13, 6)))
    right = normalized_shape(rng.normal(size=(13, 6)))
    weights = rng.uniform(0.1, 1.0, size=13)
    weights /= weights.sum()
    gamma = rng.normal(size=6)
    gamma_next = gamma + 0.03 * rng.normal(size=6)
    source = row_contrast_geometry(gamma, left, right, weights)
    left_next = normalized_shape(rng.normal(size=(13, 6)))
    right_next = normalized_shape(rng.normal(size=(13, 6)))
    target = row_contrast_geometry(
        gamma_next, left_next, right_next, weights)
    increment = gate_shape_energy_increment(
        gamma, gamma_next, source["q"], target["q"])
    assert increment["identity_residual"] == pytest.approx(0.0, abs=2e-14)
    assert increment["energy"] == pytest.approx(source["energy"], abs=2e-14)
    assert increment["energy_next"] == pytest.approx(
        target["energy"], abs=2e-14)


def test_source_owned_energy_envelope_does_not_need_target_energy():
    rng = np.random.default_rng(3504)
    gamma = rng.normal(size=8)
    delta = 0.02 * rng.normal(size=8)
    q = rng.uniform(0.1, 1.0, size=8)
    q_jvp = 0.03 * rng.normal(size=8)
    remainder_radius = 0.004
    remainder = rng.uniform(
        -remainder_radius, remainder_radius, size=8)
    q_next = q + q_jvp + remainder
    envelope = source_owned_energy_envelope(
        gamma, delta, q, q_jvp, remainder_radius)
    realized = float(np.sum(q_next * (gamma + delta) ** 2))
    assert realized <= envelope["upper_energy"] + 1e-14


def test_general_vocabulary_pairwise_softmax_identity():
    rng = np.random.default_rng(3505)
    logits = rng.normal(size=17)
    probability = np.exp(logits - np.max(logits))
    probability /= probability.sum()
    value = rng.normal(size=17)
    covariance_energy = float(value @ softmax_fisher(probability) @ value)
    assert pairwise_softmax_energy(probability, value) == pytest.approx(
        covariance_energy, rel=2e-13, abs=2e-13)


def test_dense_gate_loss_split_reconstructs_true_hessian():
    rng = np.random.default_rng(3506)
    sources, vocabulary, dimension = 4, 7, 3
    logits = rng.normal(size=(sources, vocabulary))
    probability = np.exp(logits - logits.max(axis=1, keepdims=True))
    probability /= probability.sum(axis=1, keepdims=True)
    jacobian = rng.normal(size=(sources, vocabulary, dimension))
    raw_jets = rng.normal(
        size=(sources, vocabulary, dimension, dimension))
    jets = 0.5 * (raw_jets + raw_jets.swapaxes(-1, -2))
    targets = np.asarray([0, 3, 2, 6], dtype=np.int64)
    split = gate_loss_hessian_split(
        jacobian, probability, targets, jets)
    assert np.allclose(
        split["true_hessian"],
        split["fisher"] + split["signed_second_jet"],
    )
    assert np.linalg.eigvalsh(split["fisher"])[0] >= -1e-12
    spectrum = residual_scaled_spectrum(
        split["true_hessian"], reconstruction_residual=2e-12)
    assert spectrum["zero_threshold"] >= 2e-12
    assert spectrum["lower_intervals"].shape == (dimension,)


def test_full_adamw_gate_block_matches_independent_finite_difference():
    rng = np.random.default_rng(3507)
    dimension = 4
    matrix = rng.normal(scale=0.15, size=(dimension, dimension))
    hessian = 0.5 * (matrix + matrix.T)
    affine = np.asarray([0.12, -0.08, 0.04, -0.02])
    gamma = rng.normal(scale=0.2, size=dimension)
    first = rng.normal(scale=0.03, size=dimension)
    second = rng.uniform(0.2, 0.5, size=dimension)

    def gradient(position):
        return affine + hessian @ position

    raw_gradient = gradient(gamma)
    optimizer = {
        "learning_rate": 7e-4,
        "beta1": 0.9,
        "beta2": 0.95,
        "epsilon": 1e-5,
        "optimizer_step": 23,
        "weight_decay": 0.1,
        "clip_value": 1.0,
    }
    analytic = adamw_gate_block(
        hessian, gamma, first, second, raw_gradient, **optimizer)
    numerical = finite_difference_adamw_block(
        gradient, gamma, first, second, step_size=2e-6, **optimizer)
    assert analytic["operator"].shape == (3 * dimension, 3 * dimension)
    assert np.allclose(
        analytic["operator"], numerical, rtol=2e-6, atol=2e-8)


def test_adamw_gate_block_rejects_clipping_boundary():
    with pytest.raises(ValueError, match="clipping boundary"):
        adamw_gate_block(
            np.eye(2), np.zeros(2), np.zeros(2), np.ones(2),
            np.asarray([1.0, 0.2]), learning_rate=1e-3,
            beta1=0.9, beta2=0.95, epsilon=1e-5,
            optimizer_step=2, weight_decay=0.1, clip_value=1.0,
            boundary_tolerance=1e-12)


def test_chronological_product_allows_transient_expansion_and_block_capture():
    operators = [
        np.diag([1.25, 0.8]),
        np.asarray([[0.42, 0.06], [0.0, 0.5]]),
        np.diag([0.7, 0.6]),
    ]
    products = chronological_products(operators)
    assert np.linalg.norm(operators[0], ord=2) > 1.0
    assert np.linalg.norm(products[-1], ord=2) < 1.0
    envelope = ordered_capture_envelope(
        operators, initial_norm=0.3, forcing_norms=[0.01, 0.02, 0.01])
    assert envelope["block_gain"] == pytest.approx(
        np.linalg.norm(products[-1], ord=2))
    assert np.all(envelope["envelope"] >= 0.0)


def test_plga_divided_difference_chain_is_exact_with_diagonal_cells():
    rng = np.random.default_rng(3508)
    dimension = 4
    left = rng.normal(size=(dimension, dimension))
    right = left + rng.normal(scale=0.04, size=left.shape)
    right[0, 0] = left[0, 0]
    weight = rng.normal(size=left.shape)
    bias = rng.normal(size=left.shape)
    powers = rng.uniform(-0.4, 1.8, size=left.shape)
    coupling = rng.normal(size=left.shape)
    coupling_bias = rng.normal(size=left.shape)
    result = plga_secant_chain(
        left, right, weight, bias, powers, coupling, coupling_bias)
    scale = max(np.linalg.norm(result["realized_difference"]), 1.0)
    assert result["identity_residual"] <= 2e-12 * scale
    assert np.linalg.norm(result["realized_difference"]) <= (
        result["operator_bound"] * np.linalg.norm(right - left) + 1e-12)


def test_sharp_centered_logit_margin_condition():
    reference = np.asarray([4.0, 1.0, 0.0])
    candidate = reference + np.asarray([-0.1, 0.1, 0.0])
    result = centered_logit_margin_preserved(reference, candidate)
    assert result["qualified"]
    assert result["decision_preserved"]
    assert math.sqrt(2.0) * result["difference_norm"] < result["margin"]
