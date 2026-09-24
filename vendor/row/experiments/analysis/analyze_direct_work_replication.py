#!/usr/bin/env python3
"""Analyze the locked temporal-context direct-work replication."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS))

from confirm.resource_executor import resource_records, record_is_admissible

from analysis.analyze_direct_work_precision import (  # noqa: E402
    array,
    canonical_bytes,
    magnitude_work_bound,
    scalar,
    sha256_path,
    write_bytes,
)
from analysis.sign_semantics import summarize_sign_counts  # noqa: E402
from confirm.direct_work_replication_specs import (  # noqa: E402
    ANALYSIS_SCHEMA,
    CAMPAIGN_ID,
    DECISION_POLICY,
    LAYERS,
    LOCK_SCHEMA,
    MAPS_PER_UNIT,
    NUMERICAL_POLICY,
    RECORD_SCHEMA,
    RELEASE_ID,
    RESOURCE_BUDGET,
    RESOURCE_SCHEMA,
    SOURCE_STEPS,
)


MAP_SHAPE = (24, 4, 64, 64)
ARMS = ("natural", "control")
COMPONENTS = ("shape", "gate", "interaction", "implementation_defect")


def write_json(path: Path, value: dict[str, Any]) -> None:
    write_bytes(path, canonical_bytes(value))


def coordinate_bound(*terms: np.ndarray) -> np.ndarray:
    magnitude = np.add.reduce([np.abs(term) for term in terms])
    return (
        float(NUMERICAL_POLICY["reconstruction_rounding_multiplier"])
        * np.finfo(np.float64).eps
        * magnitude
        + np.finfo(np.float64).tiny
    )


def coordinate_pass(residual: np.ndarray, *terms: np.ndarray) -> np.ndarray:
    local = np.abs(residual) <= coordinate_bound(*terms)
    return np.all(local.reshape(MAPS_PER_UNIT, -1), axis=1)


def arm_metrics(record: Any, arm: str, source: np.ndarray) -> dict[str, Any]:
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
    stored_defect = array(record, f"{arm}_gate_shape_residual", MAP_SHAPE)
    defect = endpoint_defect - source_defect
    three = shape + gate + interaction
    factorized_source = source - source_defect
    factorized_endpoint = endpoint - endpoint_defect
    factorized_residual = factorized_endpoint - factorized_source - three
    defect_residual = stored_defect - defect
    native_residual = increment - three - defect

    stored_work_residual = array(
        record, f"{arm}_work_charge_residual", (24, 4)
    ).reshape(-1)
    work_bound = magnitude_work_bound(source, endpoint)
    work_pass = np.abs(stored_work_residual) <= work_bound
    factorized_pass = coordinate_pass(
        factorized_residual,
        factorized_endpoint,
        factorized_source,
        shape,
        gate,
        interaction,
    )
    primitive_coordinates = (
        endpoint,
        source,
        factorized_endpoint,
        factorized_source,
        increment,
        shape,
        gate,
        interaction,
        endpoint_defect,
        source_defect,
        stored_defect,
        defect,
    )
    defect_bound = coordinate_bound(*primitive_coordinates)
    native_bound = coordinate_bound(*primitive_coordinates)
    defect_pass = np.all(
        (np.abs(defect_residual) <= defect_bound).reshape(MAPS_PER_UNIT, -1),
        axis=1,
    )
    native_pass = np.all(
        (np.abs(native_residual) <= native_bound).reshape(MAPS_PER_UNIT, -1),
        axis=1,
    )

    component_arrays = np.stack((shape, gate, interaction, defect), axis=2)
    flat = component_arrays.reshape(MAPS_PER_UNIT, 4, -1)
    source_flat = source.reshape(MAPS_PER_UNIT, -1)
    component_work = 2.0 * np.einsum("ma,mia->mi", source_flat, flat)
    gram = np.einsum("mia,mja->mij", flat, flat)
    change = array(record, f"{arm}_energy_change", (24, 4)).reshape(-1)
    energy_residual = (
        change - np.sum(component_work, axis=1) - np.sum(gram, axis=(1, 2))
    )
    endpoint_flat = endpoint.reshape(MAPS_PER_UNIT, -1)
    energy_magnitude = (
        np.sum(np.abs(endpoint_flat * endpoint_flat), axis=1)
        + np.sum(np.abs(source_flat * source_flat), axis=1)
        + np.abs(change)
        + np.sum(np.abs(component_work), axis=1)
        + np.sum(np.abs(gram), axis=(1, 2))
    )
    epsilon = np.finfo(np.float64).eps
    operation_count = source_flat.shape[1] + 32
    gamma = operation_count * epsilon / (1.0 - operation_count * epsilon)
    energy_bound = (
        float(NUMERICAL_POLICY["energy_summation_rounding_multiplier"])
        * gamma
        * energy_magnitude
        + np.finfo(np.float64).tiny
    )
    energy_pass = np.abs(energy_residual) <= energy_bound
    extended = np.longdouble
    extended_source = source_flat.astype(extended)
    extended_components = flat.astype(extended)
    extended_work = 2 * np.einsum(
        "ma,mia->mi", extended_source, extended_components
    )
    extended_gram = np.einsum(
        "mia,mja->mij", extended_components, extended_components
    )
    extended_energy_residual = change.astype(extended)
    extended_energy_residual -= np.sum(extended_work, axis=1)
    extended_energy_residual -= np.sum(extended_gram, axis=(1, 2))
    increment_norm = np.linalg.norm(
        increment.reshape(MAPS_PER_UNIT, -1), axis=1
    )
    component_norm = np.linalg.norm(flat, axis=2)
    component_ratio = np.divide(
        component_norm,
        increment_norm[:, None],
        out=np.zeros_like(component_norm),
        where=increment_norm[:, None] > 0.0,
    )
    factorization_relative = np.divide(
        np.linalg.norm(stored_defect.reshape(MAPS_PER_UNIT, -1), axis=1),
        np.maximum(
            increment_norm,
            np.linalg.norm(three.reshape(MAPS_PER_UNIT, -1), axis=1),
        ),
        out=np.zeros(MAPS_PER_UNIT),
        where=np.maximum(
            increment_norm,
            np.linalg.norm(three.reshape(MAPS_PER_UNIT, -1), axis=1),
        ) > 0.0,
    )
    return {
        "endpoint": endpoint,
        "increment": increment,
        "technical_mask": (
            work_pass
            & factorized_pass
            & defect_pass
            & native_pass
            & energy_pass
        ),
        "work_pass": work_pass,
        "factorized_pass": factorized_pass,
        "defect_pass": defect_pass,
        "native_pass": native_pass,
        "energy_pass": energy_pass,
        "work_residual": stored_work_residual,
        "work_bound": work_bound,
        "factorized_residual": factorized_residual,
        "defect_residual": defect_residual,
        "defect_bound": defect_bound,
        "native_residual": native_residual,
        "native_bound": native_bound,
        "energy_residual": energy_residual,
        "energy_bound": energy_bound,
        "extended_energy_residual": extended_energy_residual,
        "component_work": component_work,
        "gram": gram,
        "component_ratio": component_ratio,
        "factorization_relative": factorization_relative,
        "energy_change": change,
    }


def scalar_invariants(record: Any) -> bool:
    return bool(
        scalar(record, "paired_source_bitwise", bool)
        and scalar(record, "incoming_adjoint_bitwise", bool)
        and scalar(record, "outside_gradients_bitwise", bool)
        and scalar(record, "natural_parameter_replay_max_abs", float) == 0.0
        and scalar(record, "control_parameter_replay_max_abs", float) == 0.0
        and scalar(record, "natural_observation_replay_max_abs", float) == 0.0
        and scalar(record, "control_observation_replay_max_abs", float) == 0.0
        and scalar(record, "control_outgoing_centered_adjoint_norm", float) == 0.0
        and not scalar(record, "measurement_registry_matches_checkpoint", bool)
    )


def summarize_sign(
    contrast: np.ndarray, mask: np.ndarray, effect_floor: float
) -> dict[str, Any]:
    selected = contrast[mask]
    positive = int(np.count_nonzero(selected > effect_floor))
    negative = int(np.count_nonzero(selected < -effect_floor))
    neutral = int(len(selected) - positive - negative)
    return summarize_sign_counts(
        planned=int(contrast.size), evaluable=int(len(selected)),
        positive=positive, negative=negative, neutral=neutral,
    )


def load_unit(path: Path, effect_floor: float) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as record:
        if (
            scalar(record, "schema_version", str) != RECORD_SCHEMA
            or scalar(record, "campaign_id", str) != CAMPAIGN_ID
            or scalar(record, "release_id", str) != RELEASE_ID
        ):
            raise ValueError(f"unknown replication record identity: {path}")
        source = array(record, "source_z", MAP_SHAPE)
        source_energy = array(record, "source_energy", (24, 4)).reshape(-1)
        contrast = array(record, "intervention_contrast", (24, 4)).reshape(-1)
        arms = {arm: arm_metrics(record, arm, source) for arm in ARMS}
        scalar_pass = scalar_invariants(record)
        technical = np.logical_and.reduce(
            [arms[arm]["technical_mask"] for arm in ARMS]
        )
        if not scalar_pass:
            technical[:] = False
        effect = (source_energy > effect_floor) | (
            np.abs(contrast) > effect_floor
        )
        evaluable = technical & effect
        return {
            "path": str(path.resolve()),
            "sha256": sha256_path(path),
            "trajectory": scalar(record, "trajectory", str),
            "step": scalar(record, "source_step", int),
            "layer": scalar(record, "layer", int),
            "role": scalar(record, "role", str),
            "device": scalar(record, "device", str),
            "checkpoint_sha256": scalar(record, "checkpoint_sha256", str),
            "registry_sha256": scalar(record, "registry_sha256", str),
            "update_batch_mode": scalar(record, "update_batch_mode", str),
            "update_batch_sha256": scalar(record, "update_batch_sha256", str),
            "source": source,
            "source_energy": source_energy,
            "contrast": contrast,
            "technical_mask": technical,
            "effect_mask": effect,
            "evaluable_mask": evaluable,
            "summary": summarize_sign(contrast, evaluable, effect_floor),
            "arms": arms,
            "natural_change": arms["natural"]["energy_change"],
            "control_change": arms["control"]["energy_change"],
            "producer_peak_gpu_reserved_bytes": scalar(
                record, "peak_gpu_reserved_bytes", int
            ),
            "qualification_duplicate_max_abs": (
                scalar(record, "qualification_duplicate_max_abs", float)
                if "qualification_duplicate_max_abs" in record else None
            ),
        }


def unit_path(root: Path, trajectory: str, step: int, layer: int) -> Path:
    return root / "records" / f"{trajectory}-S{step}-L{layer}.npz"


def qualification(root: Path) -> dict[str, Any]:
    paths = [
        root / "records/qualification-cuda0.npz",
        root / "records/qualification-cuda1.npz",
    ]
    records = []
    for path in paths:
        with np.load(path, allow_pickle=False) as record:
            if (
                scalar(record, "schema_version", str) != RECORD_SCHEMA
                or scalar(record, "campaign_id", str) != CAMPAIGN_ID
            ):
                raise ValueError("qualification record identity changed")
            records.append({
                "path": path,
                "sha256": sha256_path(path),
                "device": scalar(record, "device", str),
                "source_z": array(record, "source_z", MAP_SHAPE),
                "source_energy": array(
                    record, "source_energy", (24, 4)
                ),
                "natural_endpoint_energy": array(
                    record, "natural_endpoint_energy", (24, 4)
                ),
                "control_endpoint_energy": array(
                    record, "control_endpoint_energy", (24, 4)
                ),
                "natural_endpoint_z": array(
                    record, "natural_endpoint_z", MAP_SHAPE
                ),
                "control_endpoint_z": array(
                    record, "control_endpoint_z", MAP_SHAPE
                ),
                "duplicate": scalar(
                    record, "qualification_duplicate_max_abs", float
                ),
                "registry_sha256": scalar(record, "registry_sha256", str),
                "checkpoint_sha256": scalar(
                    record, "checkpoint_sha256", str
                ),
            })
    if {row["device"] for row in records} != {"cuda:0", "cuda:1"}:
        raise ValueError("dual-device qualification did not use both devices")
    for name in (
        "registry_sha256",
        "checkpoint_sha256",
    ):
        if len({row[name] for row in records}) != 1:
            raise ValueError(f"qualification differs at {name}")
    cross_energy = max(
        float(np.max(np.abs(records[0][name] - records[1][name])))
        for name in (
            "source_energy",
            "natural_endpoint_energy",
            "control_endpoint_energy",
        )
    )
    cross_observer = max(
        float(np.max(np.abs(records[0][name] - records[1][name])))
        for name in (
            "source_z",
            "natural_endpoint_z",
            "control_endpoint_z",
        )
    )
    within = max(float(row["duplicate"]) for row in records)
    effect_floor = max(1.0e-12, 20.0 * within, 20.0 * cross_energy)
    return {
        "effect_floor": effect_floor,
        "within_device_duplicate_energy_max_abs": within,
        "cross_device_energy_max_abs": cross_energy,
        "cross_device_observer_max_abs": cross_observer,
        "cross_device_observer_bitwise": all(
            np.array_equal(records[0][name], records[1][name])
            for name in (
                "source_z",
                "natural_endpoint_z",
                "control_endpoint_z",
            )
        ),
        "records": [
            {
                "path": str(row["path"].resolve()),
                "sha256": row["sha256"],
                "device": row["device"],
            }
            for row in records
        ],
    }


def qualified(summary: dict[str, Any]) -> bool:
    return bool(
        summary["evaluable"] >= 64
        and summary["majority_fraction"] >= 2.0 / 3.0
    )


def construction_lock(root: Path) -> dict[str, Any]:
    q0 = qualification(root)
    effect_floor = float(q0["effect_floor"])
    units = [
        load_unit(unit_path(root, "A", step, layer), effect_floor)
        for step in SOURCE_STEPS for layer in LAYERS
    ]
    construction_technical = technical_summary(units)
    layer_decisions = []
    for layer in LAYERS:
        local = [unit for unit in units if unit["layer"] == layer]
        counts = {
            sign: sum(
                qualified(unit["summary"])
                and unit["summary"]["majority_sign"] == sign
                for unit in local
            )
            for sign in ("positive", "negative")
        }
        resolved = [sign for sign, count in counts.items() if count >= 2]
        if len(resolved) > 1:
            raise ArithmeticError("construction resolved both signs")
        layer_decisions.append({
            "layer": layer,
            "status": "resolved" if resolved else "unresolved",
            "locked_sign": resolved[0] if resolved else None,
            "positive_qualifying_updates": int(counts["positive"]),
            "negative_qualifying_updates": int(counts["negative"]),
            "units": [
                {
                    "step": unit["step"],
                    "update_batch_mode": unit["update_batch_mode"],
                    **unit["summary"],
                    "record_sha256": unit["sha256"],
                }
                for unit in local
            ],
        })
    payload = {
        "schema_version": LOCK_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "release_id": RELEASE_ID,
        "analysis_source_sha256": sha256_path(Path(__file__)),
        "qualification": q0,
        "effect_floor": effect_floor,
        "construction_technical": construction_technical,
        "decision_policy": DECISION_POLICY,
        "layer_decisions": layer_decisions,
    }
    payload["content_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    write_json(root / "records/construction-lock.json", payload)
    print(json.dumps({
        "effect_floor": effect_floor,
        "construction_technical": construction_technical,
        "layer_decisions": layer_decisions,
    }, sort_keys=True))
    return payload


def load_lock(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    digest = value.pop("content_sha256", None)
    expected = hashlib.sha256(canonical_bytes(value)).hexdigest()
    value["content_sha256"] = digest
    technical = value.get("construction_technical")
    failure_counts = (
        technical.get("failure_counts") if isinstance(technical, dict) else None
    )
    expected_failures = {
        "work_charge": 0,
        "factorized_three_source": 0,
        "implementation_defect_increment": 0,
        "native_four_source": 0,
        "four_source_energy": 0,
    }
    if (
        value.get("schema_version") != LOCK_SCHEMA
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("decision_policy") != DECISION_POLICY
        or failure_counts != expected_failures
        or digest != expected
    ):
        raise ValueError("replication construction lock changed")
    return value


def trajectory_layer_summary(
    units: list[dict[str, Any]], locked_sign: str | None
) -> dict[str, Any]:
    selected = np.concatenate([
        unit["contrast"][unit["evaluable_mask"]] for unit in units
    ])
    supporting = 0
    if locked_sign is not None:
        supporting = sum(
            qualified(unit["summary"])
            and unit["summary"]["majority_sign"] == locked_sign
            for unit in units
        )
    return {
        "trajectory": units[0]["trajectory"],
        "layer": units[0]["layer"],
        "planned": len(units) * MAPS_PER_UNIT,
        "evaluable": int(sum(unit["summary"]["evaluable"] for unit in units)),
        "positive": int(sum(unit["summary"]["positive"] for unit in units)),
        "negative": int(sum(unit["summary"]["negative"] for unit in units)),
        "neutral": int(sum(unit["summary"]["neutral"] for unit in units)),
        "locked_sign": locked_sign,
        "supporting_updates": int(supporting),
        "support": bool(locked_sign is not None and supporting >= 2),
        "median_contrast": float(np.median(selected)) if len(selected) else 0.0,
        "natural_reopening": int(sum(
            np.count_nonzero(
                unit["natural_change"][unit["evaluable_mask"]] > 0.0
            ) for unit in units
        )),
        "control_reopening": int(sum(
            np.count_nonzero(
                unit["control_change"][unit["evaluable_mask"]] > 0.0
            ) for unit in units
        )),
        "units": [
            {"step": unit["step"], **unit["summary"]} for unit in units
        ],
    }


def component_summary(units: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [unit["arms"][arm] for unit in units for arm in ARMS]
    ratios = np.concatenate([row["component_ratio"] for row in rows], axis=0)
    works = np.concatenate([row["component_work"] for row in rows], axis=0)
    factorization = np.concatenate(
        [row["factorization_relative"] for row in rows]
    )
    return {
        "component_order": list(COMPONENTS),
        "component_norm_ratio_median": [
            float(value) for value in np.median(ratios, axis=0)
        ],
        "component_norm_ratio_p95": [
            float(value) for value in np.quantile(ratios, 0.95, axis=0)
        ],
        "component_source_work_median": [
            float(value) for value in np.median(works, axis=0)
        ],
        "implementation_defect_ratio_maximum": float(
            np.max(ratios[:, COMPONENTS.index("implementation_defect")])
        ),
        "factorization_relative_median": float(np.median(factorization)),
        "factorization_relative_p95": float(
            np.quantile(factorization, 0.95)
        ),
        "factorization_relative_maximum": float(np.max(factorization)),
    }


def technical_summary(units: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [unit["arms"][arm] for unit in units for arm in ARMS]
    checks = {
        "work_charge": "work_pass",
        "factorized_three_source": "factorized_pass",
        "implementation_defect_increment": "defect_pass",
        "native_four_source": "native_pass",
        "four_source_energy": "energy_pass",
    }
    failures = {
        name: int(sum(np.count_nonzero(~row[key]) for row in rows))
        for name, key in checks.items()
    }
    maxima = {
        "work_charge_max_abs": float(max(
            np.max(np.abs(row["work_residual"])) for row in rows
        )),
        "factorized_secant_max_abs": float(max(
            np.max(np.abs(row["factorized_residual"])) for row in rows
        )),
        "defect_increment_max_abs": float(max(
            np.max(np.abs(row["defect_residual"])) for row in rows
        )),
        "native_four_source_max_abs": float(max(
            np.max(np.abs(row["native_residual"])) for row in rows
        )),
        "four_source_energy_max_abs": float(max(
            np.max(np.abs(row["energy_residual"])) for row in rows
        )),
        "four_source_energy_extended_max_abs": float(max(
            np.max(np.abs(row["extended_energy_residual"])) for row in rows
        )),
    }
    if any(failures.values()):
        raise ArithmeticError(f"replication technical checks failed: {failures}")
    return {"failure_counts": failures, **maxima}


def resource_summary(root: Path, units: list[dict[str, Any]]) -> dict[str, Any]:
    records = []
    for path in resource_records(root):
        value = json.loads(path.read_text(encoding="utf-8"))
        if "executor_contract" in value and not record_is_admissible(value):
            raise ValueError(f"resource admission failed: {path}")
        if value.get("schema_version") != RESOURCE_SCHEMA:
            raise ValueError(f"unknown replication resource record: {path}")
        records.append(value)
    gpu = [row for row in records if str(row["device"]).startswith("cuda:")]
    return {
        "gpu_attempt_count": len(gpu),
        "device_attempt_counts": {
            device: sum(row["device"] == device for row in gpu)
            for device in ("cuda:0", "cuda:1")
        },
        "reserved_device_hours": float(sum(
            row["accelerator_reservation_wall_seconds"] for row in gpu
        )) / 3600.0,
        "producer_peak_gpu_reserved_bytes": max(
            unit["producer_peak_gpu_reserved_bytes"] for unit in units
        ),
        "monitor_status_counts": {
            status: sum(row.get("gpu_monitor_status") == status for row in gpu)
            for status in ("measured", "unavailable", "inconsistent")
        },
        "working_reserved_device_hours": float(
            RESOURCE_BUDGET["working_aggregate_reserved_device_hours"]
        ),
        "hard_reserved_device_hours": float(
            RESOURCE_BUDGET["hard_aggregate_reserved_device_hours"]
        ),
    }


def format_float(value: float) -> str:
    if value == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    coefficient = value / 10.0**exponent
    return f"{coefficient:.3f}\\times10^{{{exponent}}}"




def final_payload(root: Path) -> dict[str, Any]:
    lock = load_lock(root / "records/construction-lock.json")
    effect_floor = float(lock["effect_floor"])
    construction = [
        load_unit(unit_path(root, "A", step, layer), effect_floor)
        for step in SOURCE_STEPS for layer in LAYERS
    ]
    heldout = [
        load_unit(unit_path(root, trajectory, step, layer), effect_floor)
        for trajectory in ("B", "C")
        for step in SOURCE_STEPS for layer in LAYERS
    ]
    decision = {int(row["layer"]): row for row in lock["layer_decisions"]}
    trajectory_layers = []
    for trajectory in ("B", "C"):
        for layer in LAYERS:
            units = [
                unit for unit in heldout
                if unit["trajectory"] == trajectory and unit["layer"] == layer
            ]
            trajectory_layers.append(
                trajectory_layer_summary(
                    units, decision[layer]["locked_sign"]
                )
            )
    layer_statuses = {}
    for layer in LAYERS:
        rows = [row for row in trajectory_layers if row["layer"] == layer]
        if decision[layer]["status"] != "resolved":
            status = "unresolved"
        elif all(row["support"] for row in rows):
            status = "supported"
        elif all(row["evaluable"] >= 192 for row in rows):
            status = "refuted"
        else:
            status = "insufficient"
        layer_statuses[str(layer)] = status
        for row in rows:
            row["status"] = status
    resolved = [
        layer_statuses[str(layer)] for layer in LAYERS
        if decision[layer]["status"] == "resolved"
    ]
    if not resolved:
        outcome = "insufficient"
    elif any(status == "refuted" for status in resolved):
        outcome = "refuted"
    elif all(status == "supported" for status in resolved):
        outcome = "supported"
    else:
        outcome = "insufficient"
    all_units = construction + heldout
    summary = {
        "schema_version": ANALYSIS_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "release_id": RELEASE_ID,
        "construction_lock_sha256": lock["content_sha256"],
        "qualification": lock["qualification"],
        "effect_floor": effect_floor,
        "construction_layer_decisions": lock["layer_decisions"],
        "trajectory_layer_summaries": trajectory_layers,
        "layer_statuses": layer_statuses,
        "campaign_outcome": outcome,
        "heldout_planned_maps": len(heldout) * MAPS_PER_UNIT,
        "heldout_evaluable_maps": int(sum(
            unit["summary"]["evaluable"] for unit in heldout
        )),
        "heldout_positive_maps": int(sum(
            unit["summary"]["positive"] for unit in heldout
        )),
        "heldout_negative_maps": int(sum(
            unit["summary"]["negative"] for unit in heldout
        )),
        "heldout_neutral_maps": int(sum(
            unit["summary"]["neutral"] for unit in heldout
        )),
        "heldout_natural_reopening": int(sum(
            np.count_nonzero(
                unit["natural_change"][unit["evaluable_mask"]] > 0.0
            ) for unit in heldout
        )),
        "heldout_control_reopening": int(sum(
            np.count_nonzero(
                unit["control_change"][unit["evaluable_mask"]] > 0.0
            ) for unit in heldout
        )),
        "technical": technical_summary(all_units),
        "four_source": component_summary(all_units),
        "resources": resource_summary(root, all_units),
        "record_sha256": {
            Path(unit["path"]).name: unit["sha256"] for unit in all_units
        },
        "terminal_interpretation": (
            "step 65536 is a registered reserve-batch counterfactual "
            "continuation, not a historical training successor"
        ),
    }
    return summary






