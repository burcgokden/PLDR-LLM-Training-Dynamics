"""Exact finite-dimensional radial--tangential energy identities.

The routines in this module are numerical realizations of algebraic
identities.  They do not assume or infer contraction.
"""

from __future__ import annotations

from typing import Any

import numpy as np


COMPONENT_NAMES = ("shape", "gate", "interaction", "implementation_defect")


def _finite_matrix(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite nonempty vector batch")
    return array


def radial_decomposition(
    source: Any, increment: Any, *, energy_floor: float = 0.0
) -> dict[str, np.ndarray]:
    """Decompose ``increment`` into a source-radial part and its orthogonal part."""

    z = _finite_matrix(source, "source")
    d = _finite_matrix(increment, "increment")
    if z.shape != d.shape:
        raise ValueError("source and increment batches disagree")
    if not np.isfinite(energy_floor) or energy_floor < 0.0:
        raise ValueError("energy_floor must be finite and nonnegative")
    energy = np.einsum("mi,mi->m", z, z)
    endpoint = z + d
    endpoint_energy = np.einsum("mi,mi->m", endpoint, endpoint)
    eligible = energy > energy_floor
    alpha = np.full(energy.shape, np.nan)
    alpha[eligible] = -np.einsum("mi,mi->m", z, d)[eligible] / energy[eligible]
    tangent = np.full_like(d, np.nan)
    tangent[eligible] = d[eligible] + alpha[eligible, None] * z[eligible]
    tangent_energy = np.full(energy.shape, np.nan)
    tangent_energy[eligible] = np.einsum(
        "mi,mi->m", tangent[eligible], tangent[eligible]
    )
    tau_squared = np.full(energy.shape, np.nan)
    tau_squared[eligible] = tangent_energy[eligible] / energy[eligible]
    gain_squared = (1.0 - alpha) ** 2 + tau_squared
    direct_gain = np.full(energy.shape, np.nan)
    direct_gain[eligible] = endpoint_energy[eligible] / energy[eligible]
    orthogonality_residual = np.full(energy.shape, np.nan)
    orthogonality_residual[eligible] = np.einsum(
        "mi,mi->m", z[eligible], tangent[eligible]
    )
    gain_residual = direct_gain - gain_squared
    return {
        "energy": energy,
        "endpoint_energy": endpoint_energy,
        "eligible": eligible,
        "alpha": alpha,
        "tangent": tangent,
        "tangent_energy": tangent_energy,
        "tau_squared": tau_squared,
        "gain_squared": gain_squared,
        "direct_gain": direct_gain,
        "orthogonality_residual": orthogonality_residual,
        "gain_residual": gain_residual,
    }


def closing_masks(result: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Classify eligible states by the exact one-step closing criterion."""

    eligible = np.asarray(result["eligible"], dtype=bool)
    alpha = np.asarray(result["alpha"], dtype=np.float64)
    tau_squared = np.asarray(result["tau_squared"], dtype=np.float64)
    gain_squared = np.asarray(result["gain_squared"], dtype=np.float64)
    contract = eligible & (gain_squared < 1.0)
    preserve = eligible & (gain_squared == 1.0)
    reopen = eligible & (gain_squared > 1.0)
    radial_interval = eligible & (alpha > 0.0) & (alpha < 2.0)
    margin = alpha * (2.0 - alpha)
    return {
        "contract": contract,
        "preserve": preserve,
        "reopen": reopen,
        "radial_interval": radial_interval,
        "radial_failure": reopen & ~radial_interval,
        "tangent_excess": reopen & radial_interval & (tau_squared > margin),
    }


def source_resolved_decomposition(
    source: Any, components: Any, *, energy_floor: float = 0.0
) -> dict[str, np.ndarray]:
    """Return per-source radial coefficients, tangents, and tangent Gram matrices."""

    z = _finite_matrix(source, "source")
    values = np.asarray(components, dtype=np.float64)
    if (
        values.ndim != 3
        or values.shape[0] != z.shape[0]
        or values.shape[2] != z.shape[1]
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("components must have shape (maps, sources, coordinates)")
    energy = np.einsum("mi,mi->m", z, z)
    eligible = energy > energy_floor
    alpha = np.full(values.shape[:2], np.nan)
    alpha[eligible] = -np.einsum(
        "mi,mai->ma", z[eligible], values[eligible]
    ) / energy[eligible, None]
    tangents = np.full_like(values, np.nan)
    tangents[eligible] = (
        values[eligible] + alpha[eligible, :, None] * z[eligible, None, :]
    )
    gram = np.full((len(z), values.shape[1], values.shape[1]), np.nan)
    gram[eligible] = np.einsum(
        "mai,mbi->mab", tangents[eligible], tangents[eligible]
    ) / energy[eligible, None, None]
    return {
        "eligible": eligible,
        "energy": energy,
        "alpha": alpha,
        "tangents": tangents,
        "tangent_gram_normalized": gram,
        "total_alpha": np.sum(alpha, axis=1),
        "total_tangent": np.sum(tangents, axis=1),
    }


def paired_decomposition(
    source: Any,
    natural_increment: Any,
    control_increment: Any,
    *,
    energy_floor: float = 0.0,
) -> dict[str, np.ndarray]:
    """Compute the exact four-term natural-versus-control energy contrast."""

    z = _finite_matrix(source, "source")
    natural = _finite_matrix(natural_increment, "natural_increment")
    control = _finite_matrix(control_increment, "control_increment")
    if z.shape != natural.shape or z.shape != control.shape:
        raise ValueError("paired increment batches disagree")
    n = radial_decomposition(z, natural, energy_floor=energy_floor)
    h = control - natural
    intervention = radial_decomposition(z, h, energy_floor=energy_floor)
    eligible = n["eligible"] & intervention["eligible"]
    energy = n["energy"]
    alpha = n["alpha"]
    beta = intervention["alpha"]
    radial_linear = -2.0 * (1.0 - alpha) * beta
    radial_charge = beta * beta
    tangent_interaction = np.full(energy.shape, np.nan)
    tangent_interaction[eligible] = 2.0 * np.einsum(
        "mi,mi->m", n["tangent"][eligible], intervention["tangent"][eligible]
    ) / energy[eligible]
    tangent_charge = intervention["tau_squared"]
    normalized_terms = np.stack(
        (radial_linear, radial_charge, tangent_interaction, tangent_charge), axis=1
    )
    natural_endpoint = z + natural
    control_endpoint = z + control
    native_contrast = (
        np.einsum("mi,mi->m", control_endpoint, control_endpoint)
        - np.einsum("mi,mi->m", natural_endpoint, natural_endpoint)
    )
    normalized_contrast = np.full(energy.shape, np.nan)
    normalized_contrast[eligible] = native_contrast[eligible] / energy[eligible]
    return {
        "eligible": eligible,
        "energy": energy,
        "alpha_natural": alpha,
        "beta": beta,
        "natural_tangent": n["tangent"],
        "intervention_tangent": intervention["tangent"],
        "normalized_terms": normalized_terms,
        "radial_normalized": radial_linear + radial_charge,
        "tangent_normalized": tangent_interaction + tangent_charge,
        "native_contrast": native_contrast,
        "normalized_contrast": normalized_contrast,
        "identity_residual": normalized_contrast - np.sum(normalized_terms, axis=1),
    }


def face_affine_coefficients(energies: Any) -> dict[str, np.ndarray]:
    """Represent every nonnegative scalar-energy step as ``E1 = q E0 + rho``."""

    values = np.asarray(energies, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("energies must be a finite nonnegative sequence")
    source = values[:-1]
    endpoint = values[1:]
    positive = source > 0.0
    gain = np.zeros_like(source)
    reopening = np.zeros_like(source)
    gain[positive] = endpoint[positive] / source[positive]
    reopening[~positive] = endpoint[~positive]
    return {
        "gain_squared": gain,
        "face_reopening": reopening,
        "step_residual": endpoint - gain * source - reopening,
    }


def face_affine_endpoint(initial: float, gain: Any, reopening: Any) -> float:
    """Evaluate the finite Duhamel formula for the affine face cocycle."""

    q = np.asarray(gain, dtype=np.float64)
    rho = np.asarray(reopening, dtype=np.float64)
    if q.ndim != 1 or q.shape != rho.shape or np.any(q < 0.0) or np.any(rho < 0.0):
        raise ValueError("gain and reopening arrays must be aligned and nonnegative")
    value = float(initial)
    for local_gain, local_reopening in zip(q, rho, strict=True):
        value = float(local_gain) * value + float(local_reopening)
    return value
