"""Transition-kernel renormalization for PLDR normal dynamics.

The routines in this module act on stochastic transition laws, rather than
on upper comparison certificates.  Exact affine Gaussian blocking,
stationary whitening, an explicitly qualified continuous embedding, empirical
projection-closure diagnostics, and block-fluctuation diagnostics are kept as
separate operations.
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


def _positive_integer(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _kernel(matrix, bias, covariance):
    matrix = _array(matrix, "matrix", ndim=2)
    bias = _array(bias, "bias", ndim=1)
    covariance = _array(covariance, "covariance", ndim=2)
    dimension = matrix.shape[0]
    if (
        dimension == 0
        or matrix.shape != (dimension, dimension)
        or bias.shape != (dimension,)
        or covariance.shape != (dimension, dimension)
    ):
        raise ValueError("affine Gaussian kernel dimensions disagree")
    scale = max(float(np.linalg.norm(covariance, ord=2)), 1.0)
    symmetry_error = float(np.linalg.norm(
        covariance - covariance.T, ord=2))
    if symmetry_error > 1e-10 * scale:
        raise ValueError("covariance must be symmetric")
    covariance = 0.5 * (covariance + covariance.T)
    if float(np.linalg.eigvalsh(covariance)[0]) < -1e-10 * scale:
        raise ValueError("covariance must be positive semidefinite")
    return matrix, bias, covariance


def compose_affine_gaussian(later, earlier):
    """Compose two independent-noise affine Gaussian transition kernels."""

    later_matrix, later_bias, later_covariance = _kernel(
        later["matrix"], later["bias"], later["covariance"])
    earlier_matrix, earlier_bias, earlier_covariance = _kernel(
        earlier["matrix"], earlier["bias"], earlier["covariance"])
    if later_matrix.shape != earlier_matrix.shape:
        raise ValueError("kernel dimensions disagree")
    return {
        "matrix": later_matrix @ earlier_matrix,
        "bias": later_matrix @ earlier_bias + later_bias,
        "covariance": (
            later_matrix @ earlier_covariance @ later_matrix.T
            + later_covariance
        ),
    }


def block_affine_gaussian(matrix, bias, covariance, block_size):
    """Integrate out ``block_size - 1`` intermediate affine states."""

    matrix, bias, covariance = _kernel(matrix, bias, covariance)
    block_size = _positive_integer(block_size, "block_size")
    dimension = len(bias)
    blocked = {
        "matrix": np.eye(dimension),
        "bias": np.zeros(dimension),
        "covariance": np.zeros((dimension, dimension)),
    }
    fine = {
        "matrix": matrix,
        "bias": bias,
        "covariance": covariance,
    }
    for _ in range(block_size):
        blocked = compose_affine_gaussian(fine, blocked)
    return blocked


def relative_kernel_residual(left, right):
    """Maximum componentwise relative norm between two kernel triples."""

    residuals = []
    for name in ("matrix", "bias", "covariance"):
        first = _array(left[name], f"left.{name}")
        second = _array(right[name], f"right.{name}")
        if first.shape != second.shape:
            raise ValueError("kernel residual shapes disagree")
        scale = max(
            float(np.linalg.norm(first)),
            float(np.linalg.norm(second)),
            1e-30,
        )
        residuals.append(float(np.linalg.norm(first - second) / scale))
    return max(residuals)


def gaussian_block_semigroup(matrix, bias, covariance,
                             first_block_size, second_block_size):
    """Compare direct ``bc`` blocking with ``b`` blocking followed by ``c``."""

    first = _positive_integer(first_block_size, "first_block_size")
    second = _positive_integer(second_block_size, "second_block_size")
    direct = block_affine_gaussian(
        matrix, bias, covariance, first * second)
    once = block_affine_gaussian(matrix, bias, covariance, first)
    repeated = block_affine_gaussian(
        once["matrix"], once["bias"], once["covariance"], second)
    return {
        "direct": direct,
        "repeated": repeated,
        "relative_residual": relative_kernel_residual(direct, repeated),
    }


def stationary_gaussian(matrix, bias, covariance):
    """Solve the stationary mean and covariance of a stable affine kernel."""

    matrix, bias, covariance = _kernel(matrix, bias, covariance)
    dimension = len(bias)
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(matrix))))
    if spectral_radius >= 1.0:
        raise ValueError("stationary Gaussian requires spectral radius below one")
    mean = np.linalg.solve(np.eye(dimension) - matrix, bias)
    operator = (
        np.eye(dimension * dimension) - np.kron(matrix, matrix)
    )
    vector = np.linalg.solve(
        operator, covariance.reshape(-1, order="F"))
    stationary_covariance = vector.reshape(
        (dimension, dimension), order="F")
    stationary_covariance = 0.5 * (
        stationary_covariance + stationary_covariance.T)
    eigenvalues = np.linalg.eigvalsh(stationary_covariance)
    if float(eigenvalues[0]) <= 0.0:
        raise ValueError("stationary covariance is not positive definite")
    mean_residual = float(np.linalg.norm(
        matrix @ mean + bias - mean))
    covariance_residual = float(np.linalg.norm(
        matrix @ stationary_covariance @ matrix.T
        + covariance - stationary_covariance))
    return {
        "mean": mean,
        "covariance": stationary_covariance,
        "spectral_radius": spectral_radius,
        "mean_residual": mean_residual,
        "covariance_residual": covariance_residual,
    }


def _symmetric_powers(matrix):
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if float(eigenvalues[0]) <= 0.0:
        raise ValueError("matrix must be positive definite")
    square_root = (
        eigenvectors * np.sqrt(eigenvalues)[None, :]
    ) @ eigenvectors.T
    inverse_square_root = (
        eigenvectors * (1.0 / np.sqrt(eigenvalues))[None, :]
    ) @ eigenvectors.T
    return square_root, inverse_square_root


def whiten_stationary_kernel(matrix, bias, covariance):
    """Center and whiten a stable affine Gaussian transition kernel."""

    matrix, bias, covariance = _kernel(matrix, bias, covariance)
    stationary = stationary_gaussian(matrix, bias, covariance)
    color, whitener = _symmetric_powers(stationary["covariance"])
    whitened_matrix = whitener @ matrix @ color
    whitened_covariance = whitener @ covariance @ whitener.T
    identity = np.eye(len(bias))
    covariance_identity_residual = float(np.linalg.norm(
        whitened_covariance
        - (identity - whitened_matrix @ whitened_matrix.T)))
    return {
        **stationary,
        "color": color,
        "whitener": whitener,
        "matrix_whitened": whitened_matrix,
        "covariance_whitened": whitened_covariance,
        "stationary_covariance_identity_residual":
            covariance_identity_residual,
    }


def _matrix_function(matrix, scalar_function):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    if np.linalg.cond(eigenvectors) > 1e12:
        raise ValueError("matrix eigenbasis is too ill-conditioned")
    transformed = (
        eigenvectors
        @ np.diag(scalar_function(eigenvalues))
        @ np.linalg.inv(eigenvectors)
    )
    imaginary = float(np.max(np.abs(np.imag(transformed))))
    scale = max(float(np.max(np.abs(transformed))), 1.0)
    if imaginary > 1e-9 * scale:
        raise ValueError("principal matrix function is not real")
    return np.real(transformed)


def principal_ou_embedding(whitened_matrix, step_size=1.0):
    """Test a principal real Ornstein-Uhlenbeck embedding.

    A continuous flow is returned only when the principal logarithm is real,
    reconstructs the discrete map, has a positive stability edge, and its
    symmetric part defines a nonnegative diffusion in stationary-whitened
    coordinates.
    """

    matrix = _array(whitened_matrix, "whitened_matrix", ndim=2)
    if matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("whitened_matrix must be nonempty and square")
    step_size = float(step_size)
    if not math.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("step_size must be finite and positive")
    eigenvalues = np.linalg.eigvals(matrix).astype(complex)
    if np.min(np.abs(eigenvalues)) <= 1e-12:
        raise ValueError("singular maps do not have a finite logarithm")
    generator = -_matrix_function(
        matrix.astype(complex), np.log) / step_size
    reconstructed = _matrix_function(
        -step_size * generator.astype(complex), np.exp)
    reconstruction_residual = float(np.linalg.norm(
        reconstructed - matrix) / max(float(np.linalg.norm(matrix)), 1e-30))
    generator_eigenvalues = np.linalg.eigvals(generator)
    stability_edge = float(np.min(np.real(generator_eigenvalues)))
    diffusion = 0.5 * (generator + generator.T)
    diffusion_edge = float(np.linalg.eigvalsh(diffusion)[0])
    if reconstruction_residual > 1e-8:
        raise ValueError("principal generator does not reconstruct the map")
    if stability_edge <= 0.0:
        raise ValueError("embedded generator is not strictly stable")
    if diffusion_edge < -1e-9:
        raise ValueError("embedded stationary diffusion is not nonnegative")
    return {
        "generator": generator,
        "diffusion": diffusion,
        "reconstruction_residual": reconstruction_residual,
        "stability_edge": stability_edge,
        "diffusion_edge": diffusion_edge,
    }


def ou_rg_flow(generator, scales, step_size=1.0):
    """Evaluate the stationary-whitened OU RG flow at positive scales."""

    generator = _array(generator, "generator", ndim=2)
    if generator.shape[0] == 0 or generator.shape[0] != generator.shape[1]:
        raise ValueError("generator must be nonempty and square")
    step_size = float(step_size)
    if not math.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("step_size must be finite and positive")
    values = []
    identity = np.eye(generator.shape[0])
    for scale_value in scales:
        scale = float(scale_value)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("RG scales must be finite and positive")
        matrix = _matrix_function(
            (-scale * step_size * generator).astype(complex), np.exp)
        covariance = identity - matrix @ matrix.T
        covariance = 0.5 * (covariance + covariance.T)
        if float(np.linalg.eigvalsh(covariance)[0]) < -1e-9:
            raise ValueError("flow covariance is not positive semidefinite")
        beta_matrix = -scale * step_size * generator @ matrix
        beta_covariance = -(
            beta_matrix @ matrix.T + matrix @ beta_matrix.T)
        values.append({
            "scale": scale,
            "log_scale": math.log(scale),
            "matrix": matrix,
            "covariance": covariance,
            "beta_matrix": beta_matrix,
            "beta_covariance": beta_covariance,
            "memory_operator_norm": float(np.linalg.norm(matrix, ord=2)),
        })
    return values


def gaussian_fixed_point_w2_upper(whitened_matrix, radius):
    """Uniform W2 upper bound to the independent standard-Gaussian kernel."""

    matrix = _array(whitened_matrix, "whitened_matrix", ndim=2)
    if matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("whitened_matrix must be nonempty and square")
    radius = float(radius)
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("radius must be finite and nonnegative")
    gain = float(np.linalg.norm(matrix, ord=2))
    if gain >= 1.0:
        raise ValueError("fixed-point bound requires operator norm below one")
    denominator = 1.0 + math.sqrt(max(0.0, 1.0 - gain * gain))
    covariance_term = math.sqrt(matrix.shape[0]) * gain * gain / denominator
    return math.hypot(gain * radius, covariance_term)


def estimate_affine_kernel(source, target):
    """Least-squares affine transition and innovation covariance estimate."""

    source = _array(source, "source", ndim=2)
    target = _array(target, "target", ndim=2)
    if source.shape != target.shape or source.shape[0] <= source.shape[1] + 1:
        raise ValueError("source and target samples are insufficient or misaligned")
    design = np.concatenate(
        [source, np.ones((source.shape[0], 1))], axis=1)
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank < source.shape[1] + 1:
        raise ValueError("affine transition design is rank deficient")
    matrix = coefficients[:-1].T
    bias = coefficients[-1]
    residuals = target - (source @ matrix.T + bias)
    covariance = residuals.T @ residuals / len(residuals)
    covariance = 0.5 * (covariance + covariance.T)
    return {
        "matrix": matrix,
        "bias": bias,
        "covariance": covariance,
        "residuals": residuals,
        "sample_count": len(source),
        "root_mean_squared_residual": float(np.sqrt(np.mean(residuals ** 2))),
    }


def _whiten_samples(samples):
    samples = _array(samples, "samples", ndim=2)
    centered = samples - samples.mean(axis=0)
    covariance = centered.T @ centered / len(centered)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(float(eigenvalues[-1]) * 1e-12, 1e-15)
    if float(eigenvalues[-1]) <= 0.0:
        raise ValueError("samples have zero covariance")
    whitener = (
        eigenvectors * (1.0 / np.sqrt(np.maximum(eigenvalues, floor)))[None, :]
    ) @ eigenvectors.T
    return centered @ whitener.T, covariance, whitener


def sliced_wasserstein_distance(first, second, *, directions=64, seed=0,
                                maximum_samples=4096):
    """Deterministic Monte Carlo estimate of sliced one-Wasserstein distance."""

    first = _array(first, "first", ndim=2)
    second = _array(second, "second", ndim=2)
    if first.shape[1] != second.shape[1] or min(len(first), len(second)) < 2:
        raise ValueError("sliced Wasserstein samples are insufficient")
    directions = _positive_integer(directions, "directions")
    maximum_samples = _positive_integer(maximum_samples, "maximum_samples")
    generator = np.random.default_rng(int(seed))
    if len(first) > maximum_samples:
        first = first[generator.choice(
            len(first), size=maximum_samples, replace=False)]
    if len(second) > maximum_samples:
        second = second[generator.choice(
            len(second), size=maximum_samples, replace=False)]
    probes = generator.normal(size=(directions, first.shape[1]))
    probes /= np.linalg.norm(probes, axis=1, keepdims=True)
    quantiles = np.linspace(0.0, 1.0, max(len(first), len(second)))
    distances = []
    for probe in probes:
        left = np.quantile(first @ probe, quantiles)
        right = np.quantile(second @ probe, quantiles)
        distances.append(float(np.mean(np.abs(left - right))))
    return float(np.mean(distances))


def empirical_lumpability_diagnostics(source, target, hidden_labels,
                                      *, minimum_group_size=32):
    """Test whether projected transition residuals depend on hidden strata."""

    source = _array(source, "source", ndim=2)
    target = _array(target, "target", ndim=2)
    labels = np.asarray(hidden_labels)
    if source.shape != target.shape or labels.shape != (len(source),):
        raise ValueError("lumpability samples and labels are misaligned")
    minimum_group_size = _positive_integer(
        minimum_group_size, "minimum_group_size")
    global_fit = estimate_affine_kernel(source, target)
    whitened_residuals, _, whitener = _whiten_samples(global_fit["residuals"])
    innovation_scale = max(
        math.sqrt(float(np.trace(global_fit["covariance"]))), 1e-15)
    groups = []
    for label in sorted(set(labels.tolist()), key=str):
        mask = labels == label
        count = int(np.sum(mask))
        if count < max(minimum_group_size, source.shape[1] + 2):
            continue
        group_fit = estimate_affine_kernel(source[mask], target[mask])
        global_prediction = (
            source[mask] @ global_fit["matrix"].T + global_fit["bias"])
        group_prediction = (
            source[mask] @ group_fit["matrix"].T + group_fit["bias"])
        prediction_defect = float(np.sqrt(np.mean(
            (global_prediction - group_prediction) ** 2)) / innovation_scale)
        residual_defect = sliced_wasserstein_distance(
            global_fit["residuals"][mask] @ whitener.T,
            whitened_residuals,
            directions=64,
            seed=1701 + len(groups),
        )
        groups.append({
            "label": str(label),
            "sample_count": count,
            "prediction_defect": prediction_defect,
            "innovation_sliced_wasserstein": residual_defect,
        })
    if len(groups) < 2:
        raise ValueError("at least two sufficiently populated hidden strata are required")
    return {
        "groups": groups,
        "maximum_prediction_defect": max(
            row["prediction_defect"] for row in groups),
        "maximum_innovation_sliced_wasserstein": max(
            row["innovation_sliced_wasserstein"] for row in groups),
    }


def innovation_dependence_diagnostics(residual_paths, *, maximum_lag=8):
    """Measure linear and conditional dependence of fitted innovations."""

    residual_paths = _array(residual_paths, "residual_paths", ndim=3)
    maximum_lag = _positive_integer(maximum_lag, "maximum_lag")
    if residual_paths.shape[0] < 4 or residual_paths.shape[1] <= maximum_lag:
        raise ValueError("innovation paths are too short for dependence tests")
    flattened = residual_paths.reshape(-1, residual_paths.shape[2])
    residual_mean = np.mean(flattened, axis=0)
    whitened, _, whitener = _whiten_samples(flattened)
    whitened = whitened.reshape(residual_paths.shape)
    correlations = []
    for lag in range(1, maximum_lag + 1):
        earlier = whitened[:, :-lag].reshape(-1, whitened.shape[2])
        later = whitened[:, lag:].reshape(-1, whitened.shape[2])
        cross = earlier.T @ later / len(earlier)
        correlations.append({
            "lag": lag,
            "cross_covariance_frobenius": float(
                np.linalg.norm(cross, ord="fro")),
        })
    previous = whitened[:, :-1].reshape(-1, whitened.shape[2])
    current_raw = residual_paths[:, 1:].reshape(
        -1, residual_paths.shape[2])
    previous_energy = np.sum(previous ** 2, axis=1)
    median = float(np.median(previous_energy))
    low = (
        current_raw[previous_energy <= median] - residual_mean
    ) @ whitener.T
    high = (
        current_raw[previous_energy > median] - residual_mean
    ) @ whitener.T
    conditional_distance = sliced_wasserstein_distance(
        low, high, directions=64, seed=2701)
    return {
        "maximum_lag": maximum_lag,
        "lag_cross_covariances": correlations,
        "maximum_cross_covariance_frobenius": max(
            row["cross_covariance_frobenius"] for row in correlations),
        "previous_energy_conditional_sliced_wasserstein":
            conditional_distance,
    }


def block_fluctuation_samples(paths, block_size):
    """Center, sum, and covariance-whiten nonoverlapping path blocks."""

    paths = _array(paths, "paths", ndim=3)
    block_size = _positive_integer(block_size, "block_size")
    replicas, steps, dimension = paths.shape
    block_count = steps // block_size
    if replicas < 2 or block_count < 2 or dimension == 0:
        raise ValueError("not enough path data for block fluctuations")
    truncated = paths[:, :block_count * block_size]
    center = truncated.mean(axis=(0, 1))
    sums = (
        (truncated - center)
        .reshape(replicas, block_count, block_size, dimension)
        .sum(axis=2) / math.sqrt(block_size)
    ).reshape(-1, dimension)
    whitened, covariance, whitener = _whiten_samples(sums)
    return {
        "samples": whitened,
        "unwhitened_covariance": covariance,
        "whitener": whitener,
        "block_count": len(whitened),
    }


def gaussianity_diagnostics(samples, *, seed=0):
    """Standardized cumulants and sliced distance to a Gaussian reference."""

    samples = _array(samples, "samples", ndim=2)
    if len(samples) < 32:
        raise ValueError("at least 32 samples are required")
    mean = samples.mean(axis=0)
    centered = samples - mean
    covariance = centered.T @ centered / len(centered)
    variance = np.maximum(np.diag(covariance), 1e-15)
    skew = np.mean(centered ** 3, axis=0) / variance ** 1.5
    excess = np.mean(centered ** 4, axis=0) / variance ** 2 - 3.0
    reference = np.random.default_rng(int(seed)).normal(size=samples.shape)
    return {
        "sample_count": len(samples),
        "mean_norm": float(np.linalg.norm(mean)),
        "covariance_identity_residual": float(np.linalg.norm(
            covariance - np.eye(samples.shape[1]), ord="fro")),
        "maximum_absolute_skewness": float(np.max(np.abs(skew))),
        "maximum_absolute_excess_kurtosis": float(np.max(np.abs(excess))),
        "sliced_wasserstein_to_gaussian": sliced_wasserstein_distance(
            samples, reference, directions=96, seed=seed + 1),
    }


def stationarity_diagnostics(paths):
    """Compare first- and second-half normal moments on the measured window."""

    paths = _array(paths, "paths", ndim=3)
    if paths.shape[0] < 4 or paths.shape[1] < 8:
        raise ValueError("stationarity diagnostics require longer paths")
    midpoint = paths.shape[1] // 2
    first = paths[:, :midpoint].reshape(-1, paths.shape[2])
    second = paths[:, midpoint:].reshape(-1, paths.shape[2])
    first_mean = first.mean(axis=0)
    second_mean = second.mean(axis=0)
    first_centered = first - first_mean
    second_centered = second - second_mean
    first_covariance = first_centered.T @ first_centered / len(first)
    second_covariance = second_centered.T @ second_centered / len(second)
    pooled = 0.5 * (first_covariance + second_covariance)
    mean_scale = max(math.sqrt(float(np.trace(pooled))), 1e-15)
    covariance_scale = max(float(np.linalg.norm(pooled, ord="fro")), 1e-15)
    return {
        "first_sample_count": len(first),
        "second_sample_count": len(second),
        "standardized_mean_defect": float(
            np.linalg.norm(first_mean - second_mean) / mean_scale),
        "relative_covariance_defect": float(np.linalg.norm(
            first_covariance - second_covariance, ord="fro")
            / covariance_scale),
    }


def analyze_trajectory_ensemble(paths, hidden_labels, block_sizes):
    """Derive kernel, closure, flow, and fluctuation diagnostics from paths."""

    paths = _array(paths, "paths", ndim=3)
    if paths.shape[0] < 4 or paths.shape[1] < 3 or paths.shape[2] == 0:
        raise ValueError("trajectory ensemble is too small")
    labels = np.asarray(hidden_labels)
    if labels.shape != paths.shape[:2]:
        raise ValueError("hidden_labels must match the path and time axes")
    sizes = tuple(_positive_integer(value, "block_size") for value in block_sizes)
    if tuple(sorted(set(sizes))) != sizes or sizes[0] != 1:
        raise ValueError("block_sizes must be sorted, unique, and start at one")
    if sizes[-1] > paths.shape[1] // 2:
        raise ValueError("largest block leaves too few fluctuation blocks")

    one_source = paths[:, :-1].reshape(-1, paths.shape[2])
    one_target = paths[:, 1:].reshape(-1, paths.shape[2])
    one_labels = labels[:, :-1].reshape(-1)
    one = estimate_affine_kernel(one_source, one_target)
    closure = empirical_lumpability_diagnostics(
        one_source, one_target, one_labels)
    stationarity = stationarity_diagnostics(paths)
    innovation_dependence = innovation_dependence_diagnostics(
        one["residuals"].reshape(
            paths.shape[0], paths.shape[1] - 1, paths.shape[2]))

    levels = []
    for index, block_size in enumerate(sizes):
        direct_source = paths[:, :-block_size].reshape(-1, paths.shape[2])
        direct_target = paths[:, block_size:].reshape(-1, paths.shape[2])
        direct = estimate_affine_kernel(direct_source, direct_target)
        predicted = block_affine_gaussian(
            one["matrix"], one["bias"], one["covariance"], block_size)
        fluctuations = block_fluctuation_samples(paths, block_size)
        gaussianity = gaussianity_diagnostics(
            fluctuations["samples"], seed=3100 + index)
        levels.append({
            "block_size": block_size,
            "direct_matrix": direct["matrix"],
            "direct_bias": direct["bias"],
            "direct_covariance": direct["covariance"],
            "predicted_matrix": predicted["matrix"],
            "predicted_bias": predicted["bias"],
            "predicted_covariance": predicted["covariance"],
            "kernel_semigroup_residual": relative_kernel_residual(
                direct, predicted),
            "direct_memory_spectral_radius": float(np.max(np.abs(
                np.linalg.eigvals(direct["matrix"])))),
            "fluctuation": gaussianity,
        })

    flow = {"decision": "NOT_EMBEDDABLE"}
    try:
        whitened = whiten_stationary_kernel(
            one["matrix"], one["bias"], one["covariance"])
        embedding = principal_ou_embedding(whitened["matrix_whitened"])
        evaluated = ou_rg_flow(embedding["generator"], sizes)
        flow = {
            "decision": "EMBEDDABLE",
            "stationary_mean_residual": whitened["mean_residual"],
            "stationary_covariance_residual": whitened["covariance_residual"],
            "whitening_identity_residual": whitened[
                "stationary_covariance_identity_residual"],
            "embedding_reconstruction_residual": embedding[
                "reconstruction_residual"],
            "generator_stability_edge": embedding["stability_edge"],
            "diffusion_edge": embedding["diffusion_edge"],
            "memory_operator_norms": [
                row["memory_operator_norm"] for row in evaluated],
        }
    except ValueError as error:
        flow = {"decision": "NOT_EMBEDDABLE", "reason": str(error)}

    return {
        "replicas": paths.shape[0],
        "steps_per_replica": paths.shape[1] - 1,
        "normal_dimension": paths.shape[2],
        "one_step": {
            "matrix": one["matrix"],
            "bias": one["bias"],
            "covariance": one["covariance"],
            "root_mean_squared_residual": one[
                "root_mean_squared_residual"],
        },
        "projection_closure": closure,
        "stationarity": stationarity,
        "innovation_dependence": innovation_dependence,
        "levels": levels,
        "continuous_flow": flow,
    }
