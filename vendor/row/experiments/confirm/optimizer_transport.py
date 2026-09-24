"""Operation-ordered AdamW transport and lifted-mode certificates.

The functions in this module mirror the training loop order:

1. value-clip the averaged gradient,
2. update the first and second moments,
3. form bias-corrected moments,
4. apply decoupled weight decay and the adaptive loss step,
5. apply any separately recorded intervention.

The lifted scalar-mode matrix uses the updated first moment.  This ordering
is part of the certificate and is tested against torch.optim.AdamW.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


def _array(value, name):
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains a nonfinite value")
    return result


def _scalar(value, name, *, lower=None, upper=None):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if lower is not None and value < lower:
        raise ValueError(f"{name} is below its lower bound")
    if upper is not None and value > upper:
        raise ValueError(f"{name} is above its upper bound")
    return value


@dataclass(frozen=True)
class AdamWConstants:
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-5
    weight_decay: float = 0.1
    clip_value: float | None = None

    def validate(self):
        _scalar(self.beta1, "beta1", lower=0.0, upper=1.0)
        _scalar(self.beta2, "beta2", lower=0.0, upper=1.0)
        if not self.beta1 < 1.0 or not self.beta2 < 1.0:
            raise ValueError("AdamW betas must be strictly below one")
        _scalar(self.epsilon, "epsilon", lower=0.0)
        if not self.epsilon > 0.0:
            raise ValueError("epsilon must be positive")
        _scalar(self.weight_decay, "weight_decay", lower=0.0)
        if self.clip_value is not None:
            _scalar(self.clip_value, "clip_value", lower=0.0)
            if not self.clip_value > 0.0:
                raise ValueError("clip_value must be positive")


def value_clip(gradient, limit):
    gradient = _array(gradient, "gradient")
    if limit is None:
        return gradient.copy()
    limit = _scalar(limit, "clip_value", lower=0.0)
    if not limit > 0.0:
        raise ValueError("clip_value must be positive")
    return np.clip(gradient, -limit, limit)


def ordered_adamw_step(theta, gradient, first_moment, second_moment, *,
                       step: int, learning_rate: float,
                       constants: AdamWConstants = AdamWConstants(),
                       intervention=None):
    """Replay one standard AdamW step with exact repository ordering."""
    constants.validate()
    if not isinstance(step, int) or isinstance(step, bool) or step < 1:
        raise ValueError("step must be a positive integer")
    eta = _scalar(learning_rate, "learning_rate", lower=0.0)
    if not eta > 0.0:
        raise ValueError("learning_rate must be positive")
    theta = _array(theta, "theta")
    gradient = _array(gradient, "gradient")
    first = _array(first_moment, "first_moment")
    second = _array(second_moment, "second_moment")
    if not theta.shape == gradient.shape == first.shape == second.shape:
        raise ValueError("AdamW arrays must have identical shapes")
    if (second < 0.0).any():
        raise ValueError("second_moment must be nonnegative")

    clipped = value_clip(gradient, constants.clip_value)
    first_next = (
        constants.beta1 * first
        + (1.0 - constants.beta1) * clipped
    )
    second_next = (
        constants.beta2 * second
        + (1.0 - constants.beta2) * np.square(clipped)
    )
    first_hat = first_next / (1.0 - constants.beta1 ** step)
    second_hat = second_next / (1.0 - constants.beta2 ** step)
    denominator = np.sqrt(second_hat) + constants.epsilon
    pre_intervention = (
        (1.0 - eta * constants.weight_decay) * theta
        - eta * first_hat / denominator
    )
    if intervention is None:
        theta_next = pre_intervention.copy()
        intervention_delta = np.zeros_like(theta)
    else:
        theta_next = _array(
            intervention(pre_intervention.copy()), "intervened_theta")
        if theta_next.shape != theta.shape:
            raise ValueError("intervention changed the parameter shape")
        intervention_delta = theta_next - pre_intervention

    actual_delta = theta_next - theta
    loss_delta = -eta * first_hat / denominator
    decay_delta = -eta * constants.weight_decay * theta
    reconstructed_delta = loss_delta + decay_delta + intervention_delta
    residual = actual_delta - reconstructed_delta
    scale = max(
        float(np.linalg.norm(actual_delta)),
        float(np.linalg.norm(reconstructed_delta)),
        1e-30,
    )
    return {
        "theta": theta_next,
        "first_moment": first_next,
        "second_moment": second_next,
        "clipped_gradient": clipped,
        "first_hat": first_hat,
        "second_hat": second_hat,
        "denominator": denominator,
        "loss_delta": loss_delta,
        "decay_delta": decay_delta,
        "intervention_delta": intervention_delta,
        "ledger_relative_residual": float(np.linalg.norm(residual) / scale),
        "step": step,
        "learning_rate_applied": eta,
        "clock": "global_optimizer_step",
        "order": [
            "value_clip",
            "first_moment",
            "second_moment",
            "bias_correction",
            "decoupled_decay_and_loss_update",
            "post_step_intervention",
        ],
    }


def observed_step_ledger(theta_before, theta_after, first_after, second_after,
                         *, step: int, learning_rate: float,
                         constants: AdamWConstants):
    """Check an observed post-step state against the exact AdamW identity."""
    constants.validate()
    theta_before = _array(theta_before, "theta_before")
    theta_after = _array(theta_after, "theta_after")
    first_after = _array(first_after, "first_after")
    second_after = _array(second_after, "second_after")
    if not (
        theta_before.shape == theta_after.shape
        == first_after.shape == second_after.shape
    ):
        raise ValueError("observed ledger arrays must have identical shapes")
    if not isinstance(step, int) or isinstance(step, bool) or step < 1:
        raise ValueError("step must be a positive integer")
    eta = _scalar(learning_rate, "learning_rate", lower=0.0)
    first_hat = first_after / (1.0 - constants.beta1 ** step)
    second_hat = second_after / (1.0 - constants.beta2 ** step)
    predicted = (
        (1.0 - eta * constants.weight_decay) * theta_before
        - eta * first_hat / (np.sqrt(second_hat) + constants.epsilon)
    )
    residual = theta_after - predicted
    scale = max(
        float(np.linalg.norm(theta_after - theta_before)),
        float(np.linalg.norm(predicted - theta_before)),
        1e-30,
    )
    return {
        "absolute_residual": float(np.linalg.norm(residual)),
        "relative_residual": float(np.linalg.norm(residual) / scale),
        "step": step,
        "learning_rate_applied": eta,
        "clock": "global_optimizer_step",
    }


def bias_corrected_mode_coefficient(learning_rate: float, beta1: float,
                                    step: int, preconditioner: float):
    if not isinstance(step, int) or isinstance(step, bool) or step < 1:
        raise ValueError("step must be a positive integer")
    eta = _scalar(learning_rate, "learning_rate", lower=0.0)
    beta1 = _scalar(beta1, "beta1", lower=0.0, upper=1.0)
    if not beta1 < 1.0:
        raise ValueError("beta1 must be below one")
    preconditioner = _scalar(
        preconditioner, "preconditioner", lower=0.0)
    if not preconditioner > 0.0:
        raise ValueError("preconditioner must be positive")
    return eta / ((1.0 - beta1 ** step) * preconditioner)


def lifted_mode_matrix(*, curvature: float, learning_rate: float,
                       beta1: float, step: int, preconditioner: float,
                       weight_decay: float = 0.0):
    """Return the exact homogeneous map on (mode amplitude, first moment).

    The scalar gradient is curvature times the pre-step amplitude and the
    adaptive denominator is frozen at the supplied positive value.
    """
    curvature = _scalar(curvature, "curvature")
    beta1 = _scalar(beta1, "beta1", lower=0.0, upper=1.0)
    if not beta1 < 1.0:
        raise ValueError("beta1 must be below one")
    eta = _scalar(learning_rate, "learning_rate", lower=0.0)
    decay = _scalar(weight_decay, "weight_decay", lower=0.0)
    alpha = bias_corrected_mode_coefficient(
        eta, beta1, step, preconditioner)
    gain = (1.0 - beta1) * curvature
    matrix = np.array([
        [1.0 - eta * decay - alpha * gain, -alpha * beta1],
        [gain, beta1],
    ])
    return matrix


def jury_margins(matrix):
    """Strict second-order Jury margins and spectral data."""
    matrix = _array(matrix, "matrix")
    if matrix.shape != (2, 2):
        raise ValueError("Jury certificate requires a 2 by 2 matrix")
    trace = float(np.trace(matrix))
    determinant = float(np.linalg.det(matrix))
    margins = {
        "one_minus_determinant": 1.0 - determinant,
        "one_minus_trace_plus_determinant": 1.0 - trace + determinant,
        "one_plus_trace_plus_determinant": 1.0 + trace + determinant,
    }
    eigenvalues = np.linalg.eigvals(matrix)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return {
        "trace": trace,
        "determinant": determinant,
        "margins": margins,
        "strict_schur": all(value > 0.0 for value in margins.values()),
        "spectral_radius": float(np.max(np.abs(eigenvalues))),
        "eigenvalues": [
            {"real": float(value.real), "imag": float(value.imag)}
            for value in eigenvalues
        ],
        "singular_lower": float(singular_values[-1]),
        "singular_upper": float(singular_values[0]),
    }


def lyapunov_band(matrices: Iterable, metric, *, strict_margin=0.0):
    """Certify a common quadratic contraction for a matrix family."""
    matrices = [_array(value, "matrix") for value in matrices]
    if not matrices:
        raise ValueError("at least one matrix is required")
    if any(value.shape != (2, 2) for value in matrices):
        raise ValueError("all lifted matrices must be 2 by 2")
    metric = _array(metric, "metric")
    if metric.shape != (2, 2) or not np.allclose(metric, metric.T):
        raise ValueError("metric must be symmetric 2 by 2")
    metric_eigenvalues, eigenvectors = np.linalg.eigh(metric)
    if metric_eigenvalues[0] <= 0.0:
        raise ValueError("metric must be positive definite")
    inverse_root = (
        eigenvectors
        @ np.diag(1.0 / np.sqrt(metric_eigenvalues))
        @ eigenvectors.T
    )
    ratios = []
    lower_ratios = []
    for matrix in matrices:
        transported = inverse_root @ matrix.T @ metric @ matrix @ inverse_root
        values = np.linalg.eigvalsh(
            0.5 * (transported + transported.T))
        lower_ratios.append(float(values[0]))
        ratios.append(float(values[-1]))
    q2 = max(ratios)
    margin = _scalar(strict_margin, "strict_margin", lower=0.0)
    return {
        "metric_lower": float(metric_eigenvalues[0]),
        "metric_upper": float(metric_eigenvalues[-1]),
        "squared_gain_lower": min(lower_ratios),
        "squared_gain_upper": q2,
        "contraction_factor": math.sqrt(max(q2, 0.0)),
        "strict_margin": margin,
        "certified": bool(q2 <= (1.0 - margin) ** 2),
        "per_matrix_squared_upper": ratios,
    }


def stability_edge(*, learning_rate: float, beta1: float, step: int,
                   preconditioner: float, weight_decay: float = 0.0):
    """Positive-curvature edge from the third strict Jury inequality."""
    beta1 = _scalar(beta1, "beta1", lower=0.0, upper=1.0)
    if not beta1 < 1.0:
        raise ValueError("beta1 must be below one")
    eta = _scalar(learning_rate, "learning_rate", lower=0.0)
    decay = _scalar(weight_decay, "weight_decay", lower=0.0)
    alpha = bias_corrected_mode_coefficient(
        eta, beta1, step, preconditioner)
    numerator = 2.0 * (1.0 + beta1) - eta * decay * (1.0 + beta1)
    denominator = alpha * (1.0 - beta1)
    if denominator <= 0.0:
        raise ValueError("stability edge denominator must be positive")
    return numerator / denominator


def exact_source(*, coupling: float, activation: float,
                 gradient: float, jacobian: float):
    values = [
        _scalar(coupling, "coupling"),
        _scalar(activation, "activation"),
        _scalar(gradient, "gradient"),
        _scalar(jacobian, "jacobian"),
    ]
    return values[0] * (values[1] * values[2] * values[3]) ** 2


def exact_source_increment(before, after):
    required = {"coupling", "activation", "gradient", "jacobian"}
    if set(before) != required or set(after) != required:
        raise ValueError("source states must contain exactly four factors")
    return exact_source(**after) - exact_source(**before)


def threshold_source_slack(threshold_before, threshold_after,
                           source_before, source_after):
    threshold_increment = (
        _scalar(threshold_after, "threshold_after")
        - _scalar(threshold_before, "threshold_before")
    )
    source_increment = (
        _scalar(source_after, "source_after")
        - _scalar(source_before, "source_before")
    )
    return {
        "threshold_increment": threshold_increment,
        "source_increment": source_increment,
        "clearance_increment": threshold_increment - source_increment,
    }

