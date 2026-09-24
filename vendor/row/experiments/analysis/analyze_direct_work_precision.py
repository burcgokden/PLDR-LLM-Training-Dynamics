#!/usr/bin/env python3
"""Digest-bound precision and eligibility audit of direct-work records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from analysis.sign_semantics import summarize_sign_counts


RECORD_SCHEMA = "pldr-direct-work-record-v1"
CAMPAIGN_ID = "pldr-direct-work-source-intervention-v1"
ANALYSIS_SCHEMA = "pldr-direct-work-precision-analysis-v1"
INVENTORY_SCHEMA = "pldr-direct-work-precision-inventory-v1"
SOURCE_STEPS = (4_096, 8_192, 16_384)
LAYERS = (0, 1, 2)
TRAJECTORIES = ("A", "B", "C")
ARMS = ("natural", "control")
MAPS_PER_UNIT = 96
MAP_SHAPE = (24, 4, 64, 64)
EFFECT_FLOOR = 1.0e-12
OLD_RELATIVE_TOLERANCE = 2.0e-5
OLD_GATE_RELATIVE_TOLERANCE = 5.0e-5
OLD_ABSOLUTE_TOLERANCE = 2.0e-10
ROUNDING_MULTIPLIER = 128.0


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def scalar(record: Any, name: str, kind: type) -> Any:
    if name not in record:
        raise ValueError(f"direct-work record omits {name}")
    value = np.asarray(record[name])
    if value.shape != ():
        raise ValueError(f"direct-work scalar is malformed: {name}")
    return kind(value.item())


def array(record: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if name not in record:
        raise ValueError(f"direct-work record omits {name}")
    value = np.asarray(record[name], dtype=np.float64)
    if value.shape != shape or not np.all(np.isfinite(value)):
        raise ValueError(f"direct-work array is malformed: {name}")
    return value


def relative_per_map(residual: np.ndarray, *references: np.ndarray) -> np.ndarray:
    local = np.linalg.norm(residual.reshape(MAPS_PER_UNIT, -1), axis=1)
    scales = [
        np.linalg.norm(value.reshape(MAPS_PER_UNIT, -1), axis=1)
        for value in references
    ]
    scale = np.maximum.reduce(
        [*scales, np.full(MAPS_PER_UNIT, np.finfo(np.float64).tiny)]
    )
    return local / scale


def magnitude_work_bound(source: np.ndarray, endpoint: np.ndarray) -> np.ndarray:
    """Conservative float64 work-charge bound with no unit scale floor."""

    z = source.reshape(MAPS_PER_UNIT, -1)
    d = (endpoint - source).reshape(MAPS_PER_UNIT, -1)
    y = endpoint.reshape(MAPS_PER_UNIT, -1)
    magnitude = (
        np.sum(np.abs(y * y), axis=1)
        + np.sum(np.abs(z * z), axis=1)
        + 2.0 * np.sum(np.abs(z * d), axis=1)
        + np.sum(np.abs(d * d), axis=1)
    )
    return (
        ROUNDING_MULTIPLIER * np.finfo(np.float64).eps * magnitude
        + np.finfo(np.float64).tiny
    )


def independent_work_residual(
    source: np.ndarray, endpoint: np.ndarray
) -> np.ndarray:
    """Accumulate the identity in extended precision when available."""

    dtype = np.longdouble
    z = source.reshape(MAPS_PER_UNIT, -1).astype(dtype)
    y = endpoint.reshape(MAPS_PER_UNIT, -1).astype(dtype)
    d = y - z
    change = np.sum(y * y, axis=1, dtype=dtype) - np.sum(
        z * z, axis=1, dtype=dtype
    )
    work = 2 * np.sum(z * d, axis=1, dtype=dtype)
    charge = np.sum(d * d, axis=1, dtype=dtype)
    return np.asarray(change - work - charge, dtype=np.longdouble)


def technical_invariants(record: Any) -> bool:
    return bool(
        scalar(record, "paired_source_bitwise", bool)
        and scalar(record, "incoming_adjoint_bitwise", bool)
        and scalar(record, "outside_gradients_bitwise", bool)
        and scalar(record, "natural_parameter_replay_max_abs", float) == 0.0
        and scalar(record, "control_parameter_replay_max_abs", float) == 0.0
        and scalar(record, "natural_observation_replay_max_abs", float) == 0.0
        and scalar(record, "control_observation_replay_max_abs", float) == 0.0
        and scalar(record, "control_outgoing_centered_adjoint_norm", float) == 0.0
    )


def arm_analysis(record: Any, arm: str, source: np.ndarray) -> dict[str, Any]:
    endpoint = array(record, f"{arm}_endpoint_z", MAP_SHAPE)
    increment = array(record, f"{arm}_increment", MAP_SHAPE)
    shape = array(record, f"{arm}_shape_secant", MAP_SHAPE)
    gate = array(record, f"{arm}_gate_secant", MAP_SHAPE)
    interaction = array(record, f"{arm}_interaction_secant", MAP_SHAPE)
    source_defect = array(
        record, f"{arm}_source_factorization_residual", MAP_SHAPE
    )
    endpoint_defect = array(
        record, f"{arm}_endpoint_factorization_residual", MAP_SHAPE
    )
    stored_gate_residual = array(
        record, f"{arm}_gate_shape_residual", MAP_SHAPE
    )
    defect_secant = endpoint_defect - source_defect
    three_source = shape + gate + interaction
    factorized_source = source - source_defect
    factorized_endpoint = endpoint - endpoint_defect
    defect_match = stored_gate_residual - defect_secant
    native_closure = increment - three_source - defect_secant
    factorized_closure = factorized_endpoint - factorized_source - three_source

    relative = relative_per_map(stored_gate_residual, increment, three_source)
    absolute = np.max(
        np.abs(stored_gate_residual).reshape(MAPS_PER_UNIT, -1), axis=1
    )
    gate_pass = (
        (relative <= OLD_GATE_RELATIVE_TOLERANCE)
        | (absolute <= OLD_ABSOLUTE_TOLERANCE)
    )

    change = array(record, f"{arm}_energy_change", (24, 4)).reshape(-1)
    work = array(record, f"{arm}_signed_work", (24, 4)).reshape(-1)
    charge = array(record, f"{arm}_finite_step_charge", (24, 4)).reshape(-1)
    stored_work_residual = array(
        record, f"{arm}_work_charge_residual", (24, 4)
    ).reshape(-1)
    old_scale = np.maximum.reduce(
        (np.abs(change), np.abs(work) + charge, np.ones(MAPS_PER_UNIT))
    )
    old_work_tolerance = (
        OLD_ABSOLUTE_TOLERANCE + OLD_RELATIVE_TOLERANCE * old_scale
    )
    work_bound = magnitude_work_bound(source, endpoint)
    old_work_pass = np.abs(stored_work_residual) <= old_work_tolerance
    work_pass = np.abs(stored_work_residual) <= work_bound
    independent = independent_work_residual(source, endpoint)

    components = np.stack((shape, gate, interaction, defect_secant), axis=2)
    flat = components.reshape(MAPS_PER_UNIT, 4, -1)
    gram = np.einsum("mia,mja->mij", flat, flat)
    source_flat = source.reshape(MAPS_PER_UNIT, -1)
    component_work = 2.0 * np.einsum("ma,mia->mi", source_flat, flat)
    four_source_energy_residual = (
        change - np.sum(component_work, axis=1) - np.sum(gram, axis=(1, 2))
    )
    increment_norm = np.linalg.norm(
        increment.reshape(MAPS_PER_UNIT, -1), axis=1
    )
    defect_norm = np.linalg.norm(
        defect_secant.reshape(MAPS_PER_UNIT, -1), axis=1
    )
    defect_ratio = np.divide(
        defect_norm,
        increment_norm,
        out=np.zeros_like(defect_norm),
        where=increment_norm > 0.0,
    )
    return {
        "endpoint": endpoint,
        "increment": increment,
        "gate_pass": gate_pass,
        "gate_relative": relative,
        "gate_absolute": absolute,
        "old_work_pass": old_work_pass,
        "work_pass": work_pass,
        "old_work_tolerance": old_work_tolerance,
        "work_bound": work_bound,
        "stored_work_residual": stored_work_residual,
        "independent_work_residual": independent,
        "defect_secant": defect_secant,
        "defect_ratio": defect_ratio,
        "defect_match": defect_match,
        "native_closure": native_closure,
        "factorized_closure": factorized_closure,
        "four_source_energy_residual": four_source_energy_residual,
        "defect_max_coordinate": np.max(
            np.abs(defect_secant).reshape(MAPS_PER_UNIT, -1), axis=1
        ),
    }


def sign_summary(contrast: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = contrast[mask]
    positive = int(np.count_nonzero(selected > EFFECT_FLOOR))
    negative = int(np.count_nonzero(selected < -EFFECT_FLOOR))
    neutral = int(len(selected) - positive - negative)
    return summarize_sign_counts(
        planned=int(contrast.size), evaluable=int(len(selected)),
        positive=positive, negative=negative, neutral=neutral,
    )


def load_unit(row: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(row["path"])).resolve(strict=True)
    if sha256_path(path) != row.get("sha256"):
        raise ValueError(f"bound direct-work record changed: {path}")
    with np.load(path, allow_pickle=False) as record:
        if (
            scalar(record, "schema_version", str) != RECORD_SCHEMA
            or scalar(record, "campaign_id", str) != CAMPAIGN_ID
            or scalar(record, "trajectory", str) != row.get("trajectory")
            or scalar(record, "source_step", int) != row.get("step")
            or scalar(record, "layer", int) != row.get("layer")
        ):
            raise ValueError(f"direct-work record identity changed: {path}")
        source = array(record, "source_z", MAP_SHAPE)
        source_energy = array(record, "source_energy", (24, 4)).reshape(-1)
        contrast = array(record, "intervention_contrast", (24, 4)).reshape(-1)
        arms = {
            arm: arm_analysis(record, arm, source)
            for arm in ARMS
        }
        technical = technical_invariants(record)
    old_work = np.logical_and.reduce(
        [arms[arm]["old_work_pass"] for arm in ARMS]
    )
    new_work = np.logical_and.reduce([arms[arm]["work_pass"] for arm in ARMS])
    gate = np.logical_and.reduce([arms[arm]["gate_pass"] for arm in ARMS])
    effect = (source_energy > EFFECT_FLOOR) | (np.abs(contrast) > EFFECT_FLOOR)
    technical_mask = np.full(MAPS_PER_UNIT, technical, dtype=bool)
    old_evaluable = technical_mask & old_work & gate & effect
    primary_evaluable = technical_mask & new_work & effect
    return {
        "path": str(path),
        "sha256": row["sha256"],
        "trajectory": row["trajectory"],
        "step": row["step"],
        "layer": row["layer"],
        "technical": technical_mask,
        "effect": effect,
        "old_work": old_work,
        "new_work": new_work,
        "gate": gate,
        "old_evaluable": old_evaluable,
        "primary_evaluable": primary_evaluable,
        "contrast": contrast,
        "source_energy": source_energy,
        "arms": arms,
        "old_summary": sign_summary(contrast, old_evaluable),
        "primary_summary": sign_summary(contrast, primary_evaluable),
    }


def qualified(summary: dict[str, Any]) -> bool:
    return bool(
        summary["evaluable"] >= 64
        and summary["majority_fraction"] >= 2.0 / 3.0
    )


def sensitivity(units: list[dict[str, Any]]) -> dict[str, Any]:
    construction = [unit for unit in units if unit["trajectory"] == "A"]
    heldout = [unit for unit in units if unit["trajectory"] != "A"]
    layers = []
    for layer in LAYERS:
        local = sorted(
            [unit for unit in construction if unit["layer"] == layer],
            key=lambda unit: unit["step"],
        )
        counts = {
            sign: sum(
                qualified(unit["primary_summary"])
                and unit["primary_summary"]["majority_sign"] == sign
                for unit in local
            )
            for sign in ("positive", "negative")
        }
        resolved = [sign for sign, count in counts.items() if count >= 2]
        if len(resolved) > 1:
            raise ArithmeticError("sensitivity construction resolves both signs")
        locked_sign = resolved[0] if resolved else None
        paths = []
        for trajectory in ("B", "C"):
            path_units = sorted(
                [
                    unit for unit in heldout
                    if unit["trajectory"] == trajectory and unit["layer"] == layer
                ],
                key=lambda unit: unit["step"],
            )
            support = sum(
                qualified(unit["primary_summary"])
                and unit["primary_summary"]["majority_sign"] == locked_sign
                for unit in path_units
            ) if locked_sign is not None else 0
            paths.append({
                "trajectory": trajectory,
                "supporting_updates": int(support),
                "planned_updates": 3,
                "units": [
                    {"step": unit["step"], **unit["primary_summary"]}
                    for unit in path_units
                ],
            })
        layers.append({
            "layer": layer,
            "status": "resolved" if locked_sign is not None else "unresolved",
            "locked_sign": locked_sign,
            "construction_units": [
                {"step": unit["step"], **unit["primary_summary"]}
                for unit in local
            ],
            "heldout_paths": paths,
        })
    return {
        "label": "post-outcome design sensitivity, not confirmatory reclassification",
        "eligibility": (
            "paired native invariants, magnitude-aware work-charge check, "
            "and frozen effect floor; factorization magnitude is diagnostic"
        ),
        "layers": layers,
    }


def maximum_abs(values: list[np.ndarray]) -> float:
    return float(max(np.max(np.abs(value)) for value in values))


def analyze(inventory_path: Path) -> dict[str, Any]:
    inventory = load_json(inventory_path)
    if inventory.get("schema_version") != INVENTORY_SCHEMA:
        raise ValueError("unknown precision-audit inventory")
    rows = inventory.get("science_records")
    if not isinstance(rows, list) or len(rows) != 27:
        raise ValueError("precision audit requires 27 science records")
    units = [load_unit(row) for row in rows]
    identities = {
        (unit["trajectory"], unit["step"], unit["layer"]) for unit in units
    }
    expected = {
        (trajectory, step, layer)
        for trajectory in TRAJECTORIES
        for step in SOURCE_STEPS
        for layer in LAYERS
    }
    if identities != expected:
        raise ValueError("precision-audit science grid is incomplete")

    locked = inventory.get("locked_analysis")
    if not isinstance(locked, dict):
        raise ValueError("locked analysis binding is missing")
    locked_path = Path(str(locked.get("path"))).resolve(strict=True)
    if sha256_path(locked_path) != locked.get("sha256"):
        raise ValueError("locked analysis changed")
    locked_summary = load_json(locked_path)
    if locked_summary.get("campaign_outcome") != "insufficient":
        raise ValueError("locked campaign outcome changed")

    heldout = [unit for unit in units if unit["trajectory"] != "A"]
    old_evaluable = int(sum(np.count_nonzero(unit["old_evaluable"]) for unit in heldout))
    primary_evaluable = int(
        sum(np.count_nonzero(unit["primary_evaluable"]) for unit in heldout)
    )
    gate_only_masks = [
        unit["technical"]
        & unit["old_work"]
        & unit["effect"]
        & ~unit["gate"]
        for unit in heldout
    ]
    work_fail_masks = [
        unit["technical"] & unit["effect"] & ~unit["old_work"]
        for unit in heldout
    ]
    effect_fail_masks = [unit["technical"] & ~unit["effect"] for unit in heldout]
    technical_fail_masks = [~unit["technical"] for unit in heldout]
    gate_only_count = int(sum(np.count_nonzero(mask) for mask in gate_only_masks))
    failing_absolute = []
    for unit, mask in zip(heldout, gate_only_masks, strict=True):
        combined = np.maximum.reduce(
            [unit["arms"][arm]["gate_absolute"] for arm in ARMS]
        )
        failing_absolute.extend(float(value) for value in combined[mask])

    arm_rows = [
        unit["arms"][arm] for unit in units for arm in ARMS
    ]
    ratios = np.concatenate([row["defect_ratio"] for row in arm_rows])
    summary = {
        "schema_version": ANALYSIS_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "analysis_role": "post-outcome precision and eligibility audit",
        "inventory_sha256": sha256_path(inventory_path),
        "locked_campaign_outcome": locked_summary["campaign_outcome"],
        "record_count": len(units),
        "arm_map_instances": len(units) * len(ARMS) * MAPS_PER_UNIT,
        "heldout": {
            "planned_maps": len(heldout) * MAPS_PER_UNIT,
            "locked_evaluable_maps": old_evaluable,
            "corrected_primary_evaluable_maps": primary_evaluable,
            "excluded_maps": len(heldout) * MAPS_PER_UNIT - old_evaluable,
            "exclusion_causes": {
                "gate_shape_only": gate_only_count,
                "work_charge": int(sum(np.count_nonzero(mask) for mask in work_fail_masks)),
                "effect_floor_only": int(
                    sum(np.count_nonzero(mask) for mask in effect_fail_masks)
                ),
                "technical_invariant": int(
                    sum(np.count_nonzero(mask) for mask in technical_fail_masks)
                ),
            },
            "failing_gate_shape_max_coordinate_abs_min": float(
                min(failing_absolute)
            ),
            "failing_gate_shape_max_coordinate_abs_max": float(
                max(failing_absolute)
            ),
            "registered_absolute_escape": OLD_ABSOLUTE_TOLERANCE,
        },
        "precision_resolved": {
            "stored_residual_equals_defect_increment_max_abs": maximum_abs(
                [row["defect_match"] for row in arm_rows]
            ),
            "native_four_source_closure_max_abs": maximum_abs(
                [row["native_closure"] for row in arm_rows]
            ),
            "factorized_three_source_closure_max_abs": maximum_abs(
                [row["factorized_closure"] for row in arm_rows]
            ),
            "four_source_energy_ledger_max_abs": maximum_abs(
                [row["four_source_energy_residual"] for row in arm_rows]
            ),
            "defect_increment_max_coordinate_abs": float(
                max(np.max(row["defect_max_coordinate"]) for row in arm_rows)
            ),
            "defect_to_native_increment_ratio": {
                "median": float(np.median(ratios)),
                "p95": float(np.quantile(ratios, 0.95)),
                "maximum": float(np.max(ratios)),
                "above_5e_minus_5": int(np.count_nonzero(ratios > 5.0e-5)),
            },
        },
        "work_charge": {
            "stored_max_abs": maximum_abs(
                [row["stored_work_residual"] for row in arm_rows]
            ),
            "independent_extended_precision_max_abs": float(
                max(np.max(np.abs(row["independent_work_residual"])) for row in arm_rows)
            ),
            "magnitude_bound_min": float(
                min(np.min(row["work_bound"]) for row in arm_rows)
            ),
            "magnitude_bound_max": float(
                max(np.max(row["work_bound"]) for row in arm_rows)
            ),
            "magnitude_bound_failures": int(sum(
                np.count_nonzero(~row["work_pass"]) for row in arm_rows
            )),
            "old_tolerance_min": float(
                min(np.min(row["old_work_tolerance"]) for row in arm_rows)
            ),
            "old_tolerance_max": float(
                max(np.max(row["old_work_tolerance"]) for row in arm_rows)
            ),
        },
        "gate_shape": {
            "maximum_relative_residual": float(
                max(np.max(row["gate_relative"]) for row in arm_rows)
            ),
            "maximum_coordinate_absolute_residual": float(
                max(np.max(row["gate_absolute"]) for row in arm_rows)
            ),
        },
        "posthoc_sensitivity": sensitivity(units),
        "record_sha256": {
            Path(unit["path"]).name: unit["sha256"] for unit in units
        },
    }
    if gate_only_count != 616:
        raise ArithmeticError(
            f"expected 616 gate-shape-only exclusions, found {gate_only_count}"
        )
    if summary["work_charge"]["magnitude_bound_failures"] != 0:
        raise ArithmeticError("magnitude-aware work-charge audit failed")
    return summary


def format_float(value: float) -> str:
    if value == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    coefficient = value / (10.0**exponent)
    return f"{coefficient:.3f}\\times10^{{{exponent}}}"








