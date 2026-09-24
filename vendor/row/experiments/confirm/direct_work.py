"""Model-independent direct row-energy and gate-shape ledgers."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from analysis.sign_semantics import summarize_sign_counts


def _finite(value: Any, name: str, *, ndim: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have dimension {ndim}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def center_rows(value: Any) -> np.ndarray:
    """Apply the orthogonal row-centering projector on the penultimate axis."""

    result = _finite(value, "row map")
    if result.ndim < 2 or result.shape[-2] < 2:
        raise ValueError("a row map needs at least two rows")
    return result - np.mean(result, axis=-2, keepdims=True)


def physical_energy(centered: Any) -> np.ndarray:
    value = _finite(centered, "centered row map")
    if value.ndim < 2:
        raise ValueError("centered row maps need row and coordinate axes")
    return np.sum(value * value, axis=(-2, -1))


def work_charge(source: Any, endpoint: Any) -> dict[str, np.ndarray]:
    """Return the exact finite-step physical energy ledger."""

    source_value = _finite(source, "source")
    endpoint_value = _finite(endpoint, "endpoint")
    if source_value.shape != endpoint_value.shape or source_value.ndim < 2:
        raise ValueError("source and endpoint row maps must agree")
    increment = endpoint_value - source_value
    source_energy = physical_energy(source_value)
    endpoint_energy = physical_energy(endpoint_value)
    work = 2.0 * np.sum(source_value * increment, axis=(-2, -1))
    charge = np.sum(increment * increment, axis=(-2, -1))
    change = endpoint_energy - source_energy
    return {
        "increment": increment,
        "source_energy": source_energy,
        "endpoint_energy": endpoint_energy,
        "energy_change": change,
        "signed_work": work,
        "finite_step_charge": charge,
        "identity_residual": change - work - charge,
        "reopening_charge": np.maximum(change, 0.0),
    }


def gate_shape_secants(
    shape_source: Any,
    shape_endpoint: Any,
    gate_source: Any,
    gate_endpoint: Any,
) -> dict[str, np.ndarray]:
    """Return the exact algebraic shape, gate, and interaction secants."""

    y0 = center_rows(shape_source)
    y1 = center_rows(shape_endpoint)
    gamma0 = _finite(gate_source, "source gate", ndim=1)
    gamma1 = _finite(gate_endpoint, "endpoint gate", ndim=1)
    if y0.shape != y1.shape or y0.shape[-1] != len(gamma0):
        raise ValueError("shape and gate dimensions disagree")
    if gamma0.shape != gamma1.shape:
        raise ValueError("source and endpoint gates disagree")
    dy = y1 - y0
    dgamma = gamma1 - gamma0
    shape = dy * gamma0
    gate = y0 * dgamma
    interaction = dy * dgamma
    source = y0 * gamma0
    endpoint = y1 * gamma1
    increment = endpoint - source
    reconstruction = shape + gate + interaction
    components = np.stack((shape, gate, interaction), axis=-3)
    flat = components.reshape(components.shape[:-3] + (3, -1))
    gram = np.einsum("...ia,...ja->...ij", flat, flat)
    source_flat = source.reshape(source.shape[:-2] + (-1,))
    source_work = 2.0 * np.einsum("...a,...ia->...i", source_flat, flat)
    return {
        "source_factorized": source,
        "endpoint_factorized": endpoint,
        "increment_factorized": increment,
        "shape_secant": shape,
        "gate_secant": gate,
        "interaction_secant": interaction,
        "secant_reconstruction_residual": increment - reconstruction,
        "component_gram": gram,
        "component_source_work": source_work,
    }


def precision_resolved_secants(
    native_source: Any,
    native_endpoint: Any,
    shape_source: Any,
    shape_endpoint: Any,
    gate_source: Any,
    gate_endpoint: Any,
) -> dict[str, np.ndarray]:
    """Resolve a native increment into shape, gate, interaction, and defect.

    The gate-shape secants describe the promoted real-arithmetic product of
    the captured shape and gate. A finite-precision implementation need not
    return that product bit for bit. The endpoint native-minus-factorized
    defects therefore contribute a fourth, exact secant source.
    """

    source = _finite(native_source, "native source")
    endpoint = _finite(native_endpoint, "native endpoint")
    if source.shape != endpoint.shape or source.ndim < 2:
        raise ValueError("native source and endpoint row maps must agree")
    factorized = gate_shape_secants(
        shape_source,
        shape_endpoint,
        gate_source,
        gate_endpoint,
    )
    if source.shape != factorized["source_factorized"].shape:
        raise ValueError("native and factorized row maps must agree")
    source_defect = source - factorized["source_factorized"]
    endpoint_defect = endpoint - factorized["endpoint_factorized"]
    defect = endpoint_defect - source_defect
    native_increment = endpoint - source
    shape = factorized["shape_secant"]
    gate = factorized["gate_secant"]
    interaction = factorized["interaction_secant"]
    reconstruction = shape + gate + interaction + defect
    components = np.stack((shape, gate, interaction, defect), axis=-3)
    flat = components.reshape(components.shape[:-3] + (4, -1))
    gram = np.einsum("...ia,...ja->...ij", flat, flat)
    source_flat = source.reshape(source.shape[:-2] + (-1,))
    source_work = 2.0 * np.einsum("...a,...ia->...i", source_flat, flat)
    return {
        **factorized,
        "native_source": source,
        "native_endpoint": endpoint,
        "native_increment": native_increment,
        "source_implementation_defect": source_defect,
        "endpoint_implementation_defect": endpoint_defect,
        "implementation_defect_secant": defect,
        "native_reconstruction_residual": native_increment - reconstruction,
        "precision_component_gram": gram,
        "precision_component_source_work": source_work,
    }


def relative_residual(residual: Any, *references: Any, floor: float = 1e-300) -> float:
    local = _finite(residual, "residual")
    scale = max(
        *(float(np.linalg.norm(_finite(value, "reference").reshape(-1)))
          for value in references),
        float(floor),
    )
    return float(np.linalg.norm(local.reshape(-1))) / scale


def intervention_contrast(
    natural_endpoint_energy: Any,
    control_endpoint_energy: Any,
) -> np.ndarray:
    natural = _finite(natural_endpoint_energy, "natural endpoint energy")
    control = _finite(control_endpoint_energy, "control endpoint energy")
    if natural.shape != control.shape:
        raise ValueError("paired endpoint energies disagree")
    return control - natural


def sign_summary(
    contrast: Any,
    evaluable: Any,
    effect_floor: float,
) -> dict[str, int | float | str | None]:
    values = _finite(contrast, "intervention contrast")
    mask = np.asarray(evaluable, dtype=bool)
    if values.shape != mask.shape or not math.isfinite(effect_floor) or effect_floor < 0:
        raise ValueError("contrast, evaluability, or effect floor is invalid")
    selected = values[mask]
    positive = int(np.count_nonzero(selected > effect_floor))
    negative = int(np.count_nonzero(selected < -effect_floor))
    neutral = int(len(selected) - positive - negative)
    return summarize_sign_counts(
        planned=int(values.size), evaluable=int(len(selected)),
        positive=positive, negative=negative, neutral=neutral,
    )
