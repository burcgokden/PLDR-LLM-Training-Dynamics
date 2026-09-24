"""Numerical kernels for layer-resolved row-map cocycle evidence."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _finite(value: Any, name: str, *, ndim: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains a nonfinite value")
    return result


def center_rows(value: Any) -> np.ndarray:
    """Project the penultimate matrix axis onto the row quotient."""

    rows = _finite(value, "rows")
    if rows.ndim < 2 or rows.shape[-2] < 2:
        raise ValueError("row centering needs a nontrivial row axis")
    return rows - rows.mean(axis=-2, keepdims=True)


def layer_observables(
    normalized_shape: Any,
    physical_rows: Any,
    layer_gates: Any,
) -> dict[str, np.ndarray | float]:
    """Compute exact float64 gate-shape observables for all registered maps."""

    shape = _finite(normalized_shape, "normalized_shape", ndim=5)
    physical = _finite(physical_rows, "physical_rows", ndim=5)
    gates = _finite(layer_gates, "layer_gates", ndim=2)
    if shape.shape != physical.shape:
        raise ValueError("shape and physical row registries disagree")
    if gates.shape != (shape.shape[1], shape.shape[-1]):
        raise ValueError("one gate vector is required per layer")
    centered_shape = center_rows(shape)
    centered_physical = center_rows(physical)
    broadcast_gate = gates[None, :, None, None, :]
    factorized = centered_shape * broadcast_gate
    energy = np.sum(centered_physical**2, axis=(-2, -1))
    shape_energy = np.sum(centered_shape**2, axis=(-2, -1))
    effective_gate = np.divide(
        energy,
        shape_energy,
        out=np.zeros_like(energy),
        where=shape_energy > 0.0,
    )
    coordinate_energy = np.sum(centered_physical**2, axis=-2)
    coordinate_shape = np.sum(centered_shape**2, axis=-2)
    residual = centered_physical - factorized
    scale = max(float(np.linalg.norm(centered_physical)), 1.0)
    return {
        "energy": energy,
        "shape_energy": shape_energy,
        "effective_squared_gate": effective_gate,
        "coordinate_energy": coordinate_energy,
        "coordinate_shape_energy": coordinate_shape,
        "factorization_relative_residual": float(
            np.linalg.norm(residual) / scale
        ),
        "maximum_gate": np.max(np.abs(gates), axis=1),
    }


def arithmetic_resolution(
    reference: Any,
    comparison: Any,
    *,
    safety_factor: float,
    denominator_floor: float,
) -> dict[str, np.ndarray]:
    """Return absolute floors and censored relative arithmetic discrepancies."""

    high = _finite(reference, "reference")
    low = _finite(comparison, "comparison")
    if high.shape != low.shape:
        raise ValueError("arithmetic paths must have equal shape")
    if (
        not math.isfinite(safety_factor)
        or safety_factor < 1.0
        or not math.isfinite(denominator_floor)
        or denominator_floor <= 0.0
    ):
        raise ValueError("resolution policy is invalid")
    discrepancy = np.abs(high - low)
    absolute_floor = safety_factor * discrepancy
    resolved = np.abs(high) > absolute_floor
    relative = np.divide(
        discrepancy,
        np.maximum(np.abs(high), denominator_floor),
    )
    return {
        "absolute_discrepancy": discrepancy,
        "absolute_floor": absolute_floor,
        "relative_discrepancy": relative,
        "relative_resolved": resolved,
    }


def scaled_operator(operator: Any, source_scale: Any, target_scale: Any) -> np.ndarray:
    """Express a lifted operator in the declared dimensionless coordinates."""

    matrix = _finite(operator, "operator", ndim=2)
    source = _finite(source_scale, "source_scale", ndim=1)
    target = _finite(target_scale, "target_scale", ndim=1)
    if matrix.shape != (len(target), len(source)) or np.any(source <= 0.0) or np.any(
        target <= 0.0
    ):
        raise ValueError("operator and positive state scales disagree")
    return target[:, None] * matrix / source[None, :]


def operator_diagnostics(
    operator: Any,
    source_scale: Any,
    target_scale: Any,
    forcing: Any,
) -> dict[str, float]:
    """Compute nonnormal gain, spectral radius, conditioning, and force size."""

    matrix = _finite(operator, "operator", ndim=2)
    scaled = scaled_operator(matrix, source_scale, target_scale)
    force = _finite(forcing, "forcing", ndim=1)
    target = _finite(target_scale, "target_scale", ndim=1)
    if force.shape != target.shape:
        raise ValueError("forcing and target scale disagree")
    singular = np.linalg.svd(scaled, compute_uv=False)
    eigenvalues = np.linalg.eigvals(scaled)
    return {
        "scaled_operator_norm": float(singular[0]),
        "scaled_operator_min_singular": float(singular[-1]),
        "scaled_spectral_radius": float(np.max(np.abs(eigenvalues))),
        "scaled_nonnormality_ratio": float(
            singular[0] / max(np.max(np.abs(eigenvalues)), np.finfo(float).tiny)
        ),
        "endpoint_scaling_condition": float(
            np.max(target / source_scale) / np.min(target / source_scale)
        ),
        "scaled_invariance_defect": float(np.linalg.norm(target * force)),
    }


def central_secant_prediction(
    native: Any,
    plus: Any,
    minus: Any,
    validation_fraction: float,
) -> np.ndarray:
    """Predict a positive validation amplitude from a sign-paired secant."""

    origin = _finite(native, "native")
    upper = _finite(plus, "plus")
    lower = _finite(minus, "minus")
    if origin.shape != upper.shape or origin.shape != lower.shape:
        raise ValueError("secant endpoints must share one shape")
    if (
        not math.isfinite(validation_fraction)
        or validation_fraction <= 0.0
        or validation_fraction >= 1.0
    ):
        raise ValueError("validation fraction must lie strictly between zero and one")
    return origin + 0.5 * validation_fraction * (upper - lower)


def prediction_residual(predicted: Any, observed: Any, native: Any) -> dict[str, float]:
    """Measure absolute and scale-aware prospective prediction error."""

    prediction = center_rows(_finite(predicted, "predicted"))
    actual = center_rows(_finite(observed, "observed"))
    origin = center_rows(_finite(native, "native"))
    if prediction.shape != actual.shape or prediction.shape != origin.shape:
        raise ValueError("prediction operands must share one shape")
    residual = actual - prediction
    response = actual - origin
    predicted_response = prediction - origin
    denominator = max(
        float(np.linalg.norm(response)),
        float(np.linalg.norm(predicted_response)),
        np.finfo(float).tiny,
    )
    return {
        "absolute_residual": float(np.linalg.norm(residual)),
        "relative_residual": float(np.linalg.norm(residual) / denominator),
        "observed_response": float(np.linalg.norm(response)),
        "predicted_response": float(np.linalg.norm(predicted_response)),
    }


def placebo_identifiability(
    placebo_effect: Any,
    treatment_effects: Any,
    *,
    maximum_fraction: float,
    effect_floor: float,
) -> dict[str, Any]:
    """Apply a fail-capable placebo gate without treating maps as replicates."""

    placebo = _finite(placebo_effect, "placebo_effect")
    treatments = _finite(treatment_effects, "treatment_effects")
    if treatments.ndim < 1 or treatments.shape[1:] != placebo.shape:
        raise ValueError("treatment and placebo layer units disagree")
    if (
        not 0.0 < maximum_fraction < 1.0
        or not math.isfinite(effect_floor)
        or effect_floor < 0.0
    ):
        raise ValueError("placebo decision policy is invalid")
    placebo_upper = np.max(np.abs(placebo), axis=tuple(range(1, placebo.ndim)))
    treatment_scale = np.max(
        np.abs(treatments), axis=tuple(range(2, treatments.ndim))
    )
    eligible = treatment_scale > effect_floor
    smallest = np.min(
        np.where(eligible, treatment_scale, np.inf), axis=0
    )
    identified = np.isfinite(smallest) & (placebo_upper <= maximum_fraction * smallest)
    return {
        "placebo_upper": placebo_upper,
        "smallest_eligible_treatment": smallest,
        "identified": identified,
        "all_layers_identified": bool(np.all(identified)),
    }


def row_preservation_residuals(
    weight: Any,
    bias: Any,
    exponent: Any,
    coupling: Any,
    coupling_bias: Any,
) -> dict[str, float]:
    """Measure the five sufficient PLGA row-preservation conditions."""

    w = _finite(weight, "weight", ndim=2)
    b = _finite(bias, "bias", ndim=2)
    p = _finite(exponent, "exponent", ndim=2)
    a = _finite(coupling, "coupling", ndim=2)
    ba = _finite(coupling_bias, "coupling_bias", ndim=2)
    if len({value.shape for value in (w, b, p, a, ba)}) != 1 or w.shape[0] != w.shape[1]:
        raise ValueError("PLGA parameter fields must be equally sized square matrices")
    ones = np.ones(w.shape[0])

    def quotient_norm(value: np.ndarray) -> float:
        return float(np.linalg.norm(center_rows(value)))

    return {
        "weight_ones_quotient": quotient_norm((w @ ones)[:, None]),
        "bias_row_quotient": quotient_norm(b),
        "exponent_row_quotient": quotient_norm(p),
        "coupling_ones_quotient": quotient_norm((a @ ones)[:, None]),
        "coupling_bias_row_quotient": quotient_norm(ba),
    }


def plga_defect_bound(
    input_quotient: Any,
    quotient_response: Any,
    row_constant_defect: Any,
    derivative_bound: Any,
) -> dict[str, np.ndarray]:
    """Evaluate the quotient-defect upper bound and signed slack."""

    source = _finite(input_quotient, "input_quotient")
    response = _finite(quotient_response, "quotient_response")
    defect = _finite(row_constant_defect, "row_constant_defect")
    derivative = _finite(derivative_bound, "derivative_bound")
    source, response, defect, derivative = np.broadcast_arrays(
        source, response, defect, derivative
    )
    if np.any(source < 0.0) or np.any(response < 0.0) or np.any(defect < 0.0) or np.any(
        derivative < 0.0
    ):
        raise ValueError("quotient-defect quantities must be nonnegative")
    upper = derivative * source + defect
    return {
        "upper": upper,
        "signed_slack": upper - response,
        "covered": response <= upper,
    }
