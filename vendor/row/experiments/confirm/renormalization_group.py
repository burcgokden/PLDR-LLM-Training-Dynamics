"""Renormalization kernels for the PLDR deductive sector.

The module keeps three logically distinct operations separate:

* exact blocking of the affine optimizer cocycle;
* token-to-query-density coarse graining and its exact covariance defect;
* conditional composition of registered full-layer closure defects.

None of the routines infer a nontrivial critical fixed point from a small
finite-sample residual. They evaluate the identities and upper bounds used
by the manuscript's RG construction.
"""

from __future__ import annotations

import math

import numpy as np


def _array(value, name, *, ndim=None):
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains a nonfinite value")
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have dimension {ndim}")
    return result


def _nonnegative(value, name):
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _block_size(value, length):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("block_size must be a positive integer")
    if length == 0 or length % value != 0:
        raise ValueError(
            "the sequence length must be positive and divisible by block_size")
    return value


def _positive_integer(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def block_average(values, block_size):
    """Average consecutive, aligned blocks along the first array axis."""
    values = _array(values, "values")
    if values.ndim < 1:
        raise ValueError("values must have a sequence axis")
    block_size = _block_size(block_size, values.shape[0])
    blocked_shape = (
        values.shape[0] // block_size,
        block_size,
        *values.shape[1:],
    )
    return values.reshape(blocked_shape).mean(axis=1)


def block_average_semigroup_residual(values, first_block_size,
                                     second_block_size):
    """Check C_c C_b = C_(bc) for aligned arithmetic block averages."""
    values = _array(values, "values")
    if values.ndim < 1:
        raise ValueError("values must have a sequence axis")
    first = _positive_integer(first_block_size, "first_block_size")
    second = _positive_integer(second_block_size, "second_block_size")
    direct = block_average(values, first * second)
    repeated = block_average(block_average(values, first), second)
    scale = max(
        float(np.linalg.norm(direct)),
        float(np.linalg.norm(repeated)),
        1e-30,
    )
    return float(np.linalg.norm(direct - repeated) / scale)


def query_density_coarse_graining(rotated_queries, block_size):
    """Return the exact fine = coarse + within-block covariance identity.

    Both density matrices are normalized by their respective sequence
    lengths. The coarse query at one block is the arithmetic mean of the
    rotated fine queries in that block.
    """
    queries = _array(rotated_queries, "rotated_queries", ndim=2)
    block_size = _block_size(block_size, queries.shape[0])
    sequence_length, width = queries.shape
    block_count = sequence_length // block_size
    blocks = queries.reshape(block_count, block_size, width)
    means = blocks.mean(axis=1)
    centered = blocks - means[:, None, :]
    fine = queries.T @ queries / sequence_length
    coarse = means.T @ means / block_count
    flat_centered = centered.reshape(sequence_length, width)
    within = flat_centered.T @ flat_centered / sequence_length
    identity = fine - coarse - within
    identity_scale = max(
        float(np.linalg.norm(fine, ord="fro")),
        float(np.linalg.norm(coarse, ord="fro")),
        float(np.linalg.norm(within, ord="fro")),
        1e-30,
    )
    within_symmetric = 0.5 * (within + within.T)
    eigenvalues = np.linalg.eigvalsh(within_symmetric)
    mean_squared_deviation = float(np.mean(np.sum(centered ** 2, axis=2)))
    return {
        "sequence_length": sequence_length,
        "block_size": block_size,
        "block_count": block_count,
        "coarse_queries": means,
        "fine_normalized_density": fine,
        "coarse_normalized_density": coarse,
        "within_block_covariance": within_symmetric,
        "density_identity_residual": float(
            np.linalg.norm(identity, ord="fro") / identity_scale),
        "within_covariance_min_eigenvalue": float(eigenvalues[0]),
        "within_covariance_trace": float(np.trace(within_symmetric)),
        "mean_squared_within_block_deviation": mean_squared_deviation,
        "trace_identity_residual": abs(
            float(np.trace(within_symmetric)) - mean_squared_deviation),
    }


def layernorm_rows(values, epsilon, gamma=None, beta=None):
    """Apply last-axis LayerNorm with explicit affine parameters."""
    values = _array(values, "values")
    if values.ndim < 1 or values.shape[-1] == 0:
        raise ValueError("values must have a nonempty feature axis")
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    width = values.shape[-1]
    gamma = (
        np.ones(width, dtype=float)
        if gamma is None else _array(gamma, "gamma", ndim=1)
    )
    beta = (
        np.zeros(width, dtype=float)
        if beta is None else _array(beta, "beta", ndim=1)
    )
    if gamma.shape != (width,) or beta.shape != (width,):
        raise ValueError("gamma and beta must match the feature width")
    centered = values - values.mean(axis=-1, keepdims=True)
    variance = np.mean(centered ** 2, axis=-1, keepdims=True)
    return gamma * centered / np.sqrt(variance + epsilon) + beta


def metric_layernorm_scale_identity(coarse_density, block_size, epsilon,
                                    gamma=None, beta=None):
    """Check LN_epsilon(b D) = LN_(epsilon/b^2)(D) row by row."""
    density = _array(coarse_density, "coarse_density", ndim=2)
    if density.shape[0] == 0 or density.shape[0] != density.shape[1]:
        raise ValueError("coarse_density must be nonempty and square")
    block_size = _positive_integer(block_size, "block_size")
    left = layernorm_rows(
        block_size * density, epsilon, gamma=gamma, beta=beta)
    right = layernorm_rows(
        density, epsilon / block_size ** 2, gamma=gamma, beta=beta)
    scale = max(
        float(np.linalg.norm(left, ord="fro")),
        float(np.linalg.norm(right, ord="fro")),
        1e-30,
    )
    return {
        "fine_scaled_layernorm": left,
        "renormalized_coarse_layernorm": right,
        "renormalized_epsilon": float(epsilon / block_size ** 2),
        "relative_residual": float(
            np.linalg.norm(left - right, ord="fro") / scale),
    }


def block_affine_cocycle(matrices, forcings, block_size):
    """Block Y_(t+1) = A_t Y_t + W_t without approximation."""
    matrices = _array(matrices, "matrices", ndim=3)
    forcings = _array(forcings, "forcings", ndim=2)
    steps, dimension, second_dimension = matrices.shape
    block_size = _block_size(block_size, steps)
    if (
        dimension == 0
        or second_dimension != dimension
        or forcings.shape != (steps, dimension)
    ):
        raise ValueError("affine cocycle dimensions disagree")
    blocked_matrices = []
    blocked_forcings = []
    for start in range(0, steps, block_size):
        matrix = np.eye(dimension)
        forcing = np.zeros(dimension)
        for offset in range(block_size):
            current_matrix = matrices[start + offset]
            current_forcing = forcings[start + offset]
            forcing = current_matrix @ forcing + current_forcing
            matrix = current_matrix @ matrix
        blocked_matrices.append(matrix)
        blocked_forcings.append(forcing)
    return {
        "block_size": block_size,
        "matrices": np.stack(blocked_matrices),
        "forcings": np.stack(blocked_forcings),
    }


def apply_affine_cocycle(matrices, forcings, initial_state):
    """Apply an affine cocycle and retain every state including the initial."""
    matrices = _array(matrices, "matrices", ndim=3)
    forcings = _array(forcings, "forcings", ndim=2)
    initial = _array(initial_state, "initial_state", ndim=1)
    steps, dimension, second_dimension = matrices.shape
    if (
        dimension == 0
        or second_dimension != dimension
        or forcings.shape != (steps, dimension)
        or initial.shape != (dimension,)
    ):
        raise ValueError("affine cocycle dimensions disagree")
    states = [initial]
    for matrix, forcing in zip(matrices, forcings):
        states.append(matrix @ states[-1] + forcing)
    return np.stack(states)


def block_softmax_aggregation(logits, block_size):
    """Aggregate causal-prefix softmax mass by exact block log-sum-exp.

    ``logits`` contains the allowed keys for one causal query row. Every
    aligned block is complete except that the last allowed block may be a
    nonempty prefix of a fine key block.
    """
    logits = _array(logits, "logits", ndim=1)
    if len(logits) == 0:
        raise ValueError("logits must be nonempty")
    block_size = _positive_integer(block_size, "block_size")
    blocks = [
        logits[start:start + block_size]
        for start in range(0, len(logits), block_size)
    ]
    shifted = logits - float(np.max(logits))
    fine = np.exp(shifted)
    fine /= float(np.sum(fine))
    fine_block_mass = np.array([
        fine[start:start + block_size].sum()
        for start in range(0, len(fine), block_size)
    ])
    maxima = np.array([float(np.max(block)) for block in blocks])
    logsumexp = np.array([
        maximum + math.log(float(np.exp(block - maximum).sum()))
        for block, maximum in zip(blocks, maxima)
    ])
    coarse_shifted = logsumexp - float(np.max(logsumexp))
    exact_coarse = np.exp(coarse_shifted)
    exact_coarse /= float(np.sum(exact_coarse))
    mean_logits = np.array([float(np.mean(block)) for block in blocks])
    mean_shifted = mean_logits - float(np.max(mean_logits))
    mean_coarse = np.exp(mean_shifted)
    mean_coarse /= float(np.sum(mean_coarse))
    within_oscillation = np.array([
        float(np.max(np.abs(block - mean_logit)))
        for block, mean_logit in zip(blocks, mean_logits)
    ])
    return {
        "fine_block_mass": fine_block_mass,
        "block_logsumexp_logits": logsumexp,
        "exact_coarse_softmax": exact_coarse,
        "mean_logit_coarse_softmax": mean_coarse,
        "aggregation_residual": float(np.max(np.abs(
            fine_block_mass - exact_coarse))),
        "mean_logit_approximation_error": float(np.max(np.abs(
            fine_block_mass - mean_coarse))),
        "maximum_within_block_logit_oscillation":
            float(np.max(within_oscillation)),
    }


def deductive_rg_closure_bound(*, rotated_queries, block_size,
                               layernorm_lipschitz,
                               direct_row_map_upper,
                               downstream_lipschitz,
                               observed_deductive_difference=None):
    """Project the exact density defect through the direct row-map bound."""
    density = query_density_coarse_graining(
        rotated_queries, block_size)
    layernorm = _nonnegative(
        layernorm_lipschitz, "layernorm_lipschitz")
    direct = _nonnegative(
        direct_row_map_upper, "direct_row_map_upper")
    downstream = _nonnegative(
        downstream_lipschitz, "downstream_lipschitz")
    density_perturbation = (
        density["block_count"]
        * float(np.linalg.norm(
            density["within_block_covariance"], ord="fro"))
    )
    row_map_upper = layernorm * direct * density_perturbation
    deductive_upper = downstream * row_map_upper
    result = {
        **density,
        "layernorm_input_perturbation_upper":
            layernorm * density_perturbation,
        "row_map_closure_upper": row_map_upper,
        "deductive_closure_upper": deductive_upper,
    }
    if observed_deductive_difference is not None:
        observed = _nonnegative(
            observed_deductive_difference,
            "observed_deductive_difference",
        )
        if deductive_upper == 0.0:
            ratio = 0.0 if observed == 0.0 else float(np.finfo(float).max)
        else:
            ratio = observed / deductive_upper
        result.update({
            "observed_deductive_difference": observed,
            "deductive_closure_ratio": ratio,
            "deductive_closure_holds": bool(ratio <= 1.0),
        })
    return result


def telescoping_closure_bound(local_defects, downstream_lipschitz):
    """Compose local coarse-graining defects through a staged layer map."""
    defects = _array(local_defects, "local_defects", ndim=1)
    lipschitz = _array(
        downstream_lipschitz, "downstream_lipschitz", ndim=1)
    if len(defects) == 0 or defects.shape != lipschitz.shape:
        raise ValueError(
            "local_defects and downstream_lipschitz must be nonempty "
            "and aligned")
    if (defects < 0.0).any() or (lipschitz < 0.0).any():
        raise ValueError("closure data must be nonnegative")
    contributions = []
    for index, defect in enumerate(defects):
        multiplier = float(np.prod(lipschitz[index + 1:]))
        contributions.append(float(defect) * multiplier)
    return {
        "local_defects": defects.tolist(),
        "stage_lipschitz": lipschitz.tolist(),
        "transported_contributions": contributions,
        "closure_upper": float(sum(contributions)),
    }
