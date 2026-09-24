from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


CONFIRM = Path(__file__).resolve().parents[1] / "confirm"
sys.path.insert(0, str(CONFIRM))

from qualification_transition_kernel_rg import qualify  # noqa: E402
from transition_kernel_rg import (  # noqa: E402
    analyze_trajectory_ensemble,
    block_affine_gaussian,
    block_fluctuation_samples,
    compose_affine_gaussian,
    empirical_lumpability_diagnostics,
    gaussian_block_semigroup,
    gaussian_fixed_point_w2_upper,
    gaussianity_diagnostics,
    ou_rg_flow,
    principal_ou_embedding,
    relative_kernel_residual,
    stationary_gaussian,
    whiten_stationary_kernel,
)


def _kernel():
    matrix = np.array([[0.75, 0.0], [0.0, 0.5]])
    bias = np.array([0.125, -0.25])
    stationary_covariance = np.diag([2.0, 0.5])
    covariance = stationary_covariance - (
        matrix @ stationary_covariance @ matrix.T)
    return matrix, bias, covariance


def _paths(seed=9, replicas=160, steps=64, hidden_shift=0.0):
    generator = np.random.default_rng(seed)
    matrix = np.diag([0.75, 0.5])
    covariance = np.eye(2) - matrix @ matrix.T
    factor = np.linalg.cholesky(covariance)
    labels = np.broadcast_to(
        (np.arange(replicas) % 2)[:, None], (replicas, steps + 1)).copy()
    paths = np.empty((replicas, steps + 1, 2))
    paths[:, 0] = generator.normal(size=(replicas, 2))
    for step in range(steps):
        drift = np.zeros((replicas, 2))
        drift[:, 0] = hidden_shift * (2 * labels[:, step] - 1)
        paths[:, step + 1] = (
            paths[:, step] @ matrix.T + drift
            + generator.normal(size=(replicas, 2)) @ factor.T
        )
    return paths, labels


def test_affine_gaussian_block_has_exact_semigroup():
    matrix, bias, covariance = _kernel()
    result = gaussian_block_semigroup(matrix, bias, covariance, 2, 4)
    assert result["relative_residual"] < 1e-14
    direct = block_affine_gaussian(matrix, bias, covariance, 8)
    assert relative_kernel_residual(direct, result["direct"]) < 1e-14


def test_two_kernel_composition_includes_transported_covariance():
    matrix, bias, covariance = _kernel()
    fine = {"matrix": matrix, "bias": bias, "covariance": covariance}
    result = compose_affine_gaussian(fine, fine)
    np.testing.assert_allclose(result["matrix"], matrix @ matrix)
    np.testing.assert_allclose(result["bias"], matrix @ bias + bias)
    np.testing.assert_allclose(
        result["covariance"], matrix @ covariance @ matrix.T + covariance)


def test_stationary_whitening_and_continuous_flow():
    matrix, bias, covariance = _kernel()
    stationary = stationary_gaussian(matrix, bias, covariance)
    np.testing.assert_allclose(stationary["mean"], [0.5, -0.5])
    np.testing.assert_allclose(stationary["covariance"], np.diag([2.0, 0.5]))
    whitened = whiten_stationary_kernel(matrix, bias, covariance)
    assert whitened["stationary_covariance_identity_residual"] < 1e-14
    embedding = principal_ou_embedding(whitened["matrix_whitened"])
    flow = ou_rg_flow(embedding["generator"], (1, 2, 4, 8))
    norms = [level["memory_operator_norm"] for level in flow]
    assert all(later < earlier for earlier, later in zip(norms, norms[1:]))
    assert gaussian_fixed_point_w2_upper(flow[-1]["matrix"], 2.0) < 0.25


def test_continuous_flow_rejects_nonreal_principal_embedding():
    with pytest.raises(ValueError, match="not real"):
        principal_ou_embedding(np.diag([-0.5, 0.5]))


def test_hidden_strata_are_visible_to_projection_closure_test():
    clean_paths, labels = _paths(hidden_shift=0.0)
    shifted_paths, _ = _paths(hidden_shift=0.4)
    clean = empirical_lumpability_diagnostics(
        clean_paths[:, :-1].reshape(-1, 2),
        clean_paths[:, 1:].reshape(-1, 2),
        labels[:, :-1].reshape(-1),
    )
    shifted = empirical_lumpability_diagnostics(
        shifted_paths[:, :-1].reshape(-1, 2),
        shifted_paths[:, 1:].reshape(-1, 2),
        labels[:, :-1].reshape(-1),
    )
    assert shifted["maximum_prediction_defect"] > (
        clean["maximum_prediction_defect"] + 0.1)


def test_block_fluctuations_gaussianize_nongaussian_independent_noise():
    generator = np.random.default_rng(101)
    paths = generator.laplace(size=(512, 128, 1))
    fine = gaussianity_diagnostics(
        block_fluctuation_samples(paths, 1)["samples"], seed=11)
    coarse = gaussianity_diagnostics(
        block_fluctuation_samples(paths, 32)["samples"], seed=12)
    assert coarse["maximum_absolute_excess_kurtosis"] < (
        fine["maximum_absolute_excess_kurtosis"])


def test_full_trajectory_analyzer_and_q0_qualification():
    paths, labels = _paths(seed=22)
    result = analyze_trajectory_ensemble(
        paths, labels, (1, 2, 4, 8, 16))
    assert result["continuous_flow"]["decision"] == "EMBEDDABLE"
    assert result["stationarity"]["standardized_mean_defect"] < 0.2
    assert result["innovation_dependence"][
        "maximum_cross_covariance_frobenius"] < 0.2
    assert [row["block_size"] for row in result["levels"]] == [1, 2, 4, 8, 16]
    assert qualify()["decision"] == "QUALIFIED"
