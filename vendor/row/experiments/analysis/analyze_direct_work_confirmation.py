#!/usr/bin/env python3
"""Analyze the locked direct-work source intervention."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
ROOT = EXPERIMENTS.parent
sys.path.insert(0, str(EXPERIMENTS))

from confirm.resource_executor import resource_records, record_is_admissible

from confirm.direct_work import sign_summary  # noqa: E402
from confirm.resource_clocks import normalize_resource_clocks  # noqa: E402
from confirm.direct_work_specs import (  # noqa: E402
    ANALYSIS_SCHEMA,
    CAMPAIGN_ID,
    DECISION_POLICY,
    LAYERS,
    LOCK_SCHEMA,
    NUMERICAL_POLICY,
    RECORD_SCHEMA,
    RELEASE_ID,
    RESOURCE_BUDGET,
    RESOURCE_SCHEMA,
    SOURCE_STEPS,
)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    write_bytes(path, canonical_bytes(value))


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar(record: Any, name: str, kind: type) -> Any:
    if name not in record:
        raise ValueError(f"direct-work record omits {name}")
    value = np.asarray(record[name])
    if value.shape != ():
        raise ValueError(f"direct-work scalar is malformed: {name}")
    return kind(value.item())


def _array(record: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if name not in record:
        raise ValueError(f"direct-work record omits {name}")
    value = np.asarray(record[name], dtype=np.float64)
    if value.shape != shape or not np.all(np.isfinite(value)):
        raise ValueError(f"direct-work array is malformed: {name}")
    return value


def _relative_per_map(residual: np.ndarray, *references: np.ndarray) -> np.ndarray:
    local = np.linalg.norm(residual.reshape(96, -1), axis=1)
    scales = [np.linalg.norm(value.reshape(96, -1), axis=1) for value in references]
    scale = np.maximum.reduce([*scales, np.full(96, 1.0e-300)])
    return local / scale


def _identity_mask(record: Any) -> tuple[np.ndarray, dict[str, float]]:
    relative_tolerance = float(NUMERICAL_POLICY["identity_relative_tolerance"])
    gate_shape_relative_tolerance = float(
        NUMERICAL_POLICY["gate_shape_relative_tolerance"]
    )
    absolute_tolerance = float(NUMERICAL_POLICY["identity_absolute_tolerance"])
    masks = []
    diagnostics: dict[str, float] = {}
    for arm in ("natural", "control"):
        change = _array(record, f"{arm}_energy_change", (24, 4)).reshape(-1)
        work = _array(record, f"{arm}_signed_work", (24, 4)).reshape(-1)
        charge = _array(record, f"{arm}_finite_step_charge", (24, 4)).reshape(-1)
        residual = _array(record, f"{arm}_work_charge_residual", (24, 4)).reshape(-1)
        scale = np.maximum.reduce((np.abs(change), np.abs(work) + charge, np.ones(96)))
        masks.append(np.abs(residual) <= absolute_tolerance + relative_tolerance * scale)
        diagnostics[f"{arm}_work_charge_max_abs"] = float(np.max(np.abs(residual)))
        increment = _array(record, f"{arm}_increment", (24, 4, 64, 64))
        shape = _array(record, f"{arm}_shape_secant", (24, 4, 64, 64))
        gate = _array(record, f"{arm}_gate_secant", (24, 4, 64, 64))
        interaction = _array(
            record, f"{arm}_interaction_secant", (24, 4, 64, 64)
        )
        gate_shape_residual = _array(
            record, f"{arm}_gate_shape_residual", (24, 4, 64, 64)
        )
        relative = _relative_per_map(
            gate_shape_residual, increment, shape + gate + interaction
        )
        absolute = np.max(np.abs(gate_shape_residual).reshape(96, -1), axis=1)
        masks.append(
            (relative <= gate_shape_relative_tolerance) | (absolute <= absolute_tolerance)
        )
        diagnostics[f"{arm}_gate_shape_max_relative"] = float(np.max(relative))
        diagnostics[f"{arm}_gate_shape_max_abs"] = float(np.max(absolute))
    technical = bool(
        _scalar(record, "paired_source_bitwise", bool)
        and _scalar(record, "incoming_adjoint_bitwise", bool)
        and _scalar(record, "outside_gradients_bitwise", bool)
        and _scalar(record, "natural_parameter_replay_max_abs", float) == 0.0
        and _scalar(record, "control_parameter_replay_max_abs", float) == 0.0
        and _scalar(record, "natural_observation_replay_max_abs", float) == 0.0
        and _scalar(record, "control_observation_replay_max_abs", float) == 0.0
        and _scalar(record, "control_outgoing_centered_adjoint_norm", float) == 0.0
    )
    result = np.logical_and.reduce(masks)
    if not technical:
        result[:] = False
    diagnostics["technical_pair_invariants"] = float(technical)
    return result, diagnostics


def load_unit(
    path: Path,
    effect_floor: float,
    expected_release_id: str = RELEASE_ID,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as record:
        if (
            _scalar(record, "schema_version", str) != RECORD_SCHEMA
            or _scalar(record, "campaign_id", str) != CAMPAIGN_ID
            or _scalar(record, "release_id", str) != expected_release_id
        ):
            raise ValueError(f"unknown direct-work record identity: {path}")
        environment = json.loads(_scalar(record, "environment_json", str))
        required_environment = {
            "nvidia_query_command",
            "nvidia_query_raw_output",
            "nvidia_driver_version",
            "gpu_name",
            "gpu_uuid",
            "compute_capability",
            "cuda_runtime",
            "cudnn_version",
            "torch_version",
            "cublas_workspace_config",
            "deterministic_algorithms",
            "allow_tf32_matmul",
            "allow_tf32_cudnn",
            "float32_matmul_precision",
        }
        if set(environment) != required_environment or any(
            environment[name] is None or environment[name] == ""
            for name in required_environment
        ):
            raise ValueError(f"direct-work environment seal is incomplete: {path}")
        identity, diagnostics = _identity_mask(record)
        source_energy = _array(record, "source_energy", (24, 4)).reshape(-1)
        contrast = _array(record, "intervention_contrast", (24, 4)).reshape(-1)
        evaluable = identity & (
            (source_energy > effect_floor) | (np.abs(contrast) > effect_floor)
        )
        summary = sign_summary(contrast, evaluable, effect_floor)
        natural_change = _array(
            record, "natural_energy_change", (24, 4)
        ).reshape(-1)
        control_change = _array(
            record, "control_energy_change", (24, 4)
        ).reshape(-1)
        natural_work = _array(
            record, "natural_component_source_work", (24, 4, 3)
        ).reshape(96, 3)
        natural_gram = _array(
            record, "natural_component_gram", (24, 4, 3, 3)
        ).reshape(96, 3, 3)
        return {
            "path": str(path.resolve()),
            "sha256": sha256_path(path),
            "role": _scalar(record, "role", str),
            "trajectory": _scalar(record, "trajectory", str),
            "seed": _scalar(record, "seed", int),
            "step": _scalar(record, "source_step", int),
            "layer": _scalar(record, "layer", int),
            "device": _scalar(record, "device", str),
            "evaluable_mask": evaluable,
            "contrast": contrast,
            "natural_change": natural_change,
            "control_change": control_change,
            "summary": summary,
            "diagnostics": diagnostics,
            "environment": environment,
            "producer_process_elapsed_seconds": _scalar(
                record, "producer_process_elapsed_seconds", float
            ),
            "peak_gpu_reserved_bytes": _scalar(
                record, "peak_gpu_reserved_bytes", int
            ),
            "natural_component_source_work": natural_work,
            "natural_component_gram": natural_gram,
        }


def _unit_path(campaign_root: Path, trajectory: str, step: int, layer: int) -> Path:
    return campaign_root / "records" / f"{trajectory}-S{step}-L{layer}.npz"


def _qualified_unit(unit: dict[str, Any]) -> bool:
    return bool(
        unit["summary"]["evaluable"] >= 64
        and unit["summary"]["majority_fraction"] >= 2.0 / 3.0
    )


def construction_lock(campaign_root: Path) -> dict[str, Any]:
    qualification_path = (
        campaign_root / "records/qualification-A-S1024-L0.npz"
    )
    with np.load(qualification_path, allow_pickle=False) as qualification:
        duplicate = _scalar(qualification, "qualification_duplicate_max_abs", float)
    effect_floor = max(1.0e-12, 20.0 * duplicate)
    qualification = load_unit(qualification_path, effect_floor)
    units = [
        load_unit(_unit_path(campaign_root, "A", step, layer), effect_floor)
        for step in SOURCE_STEPS
        for layer in LAYERS
    ]
    layer_decisions = []
    for layer in LAYERS:
        local = [unit for unit in units if unit["layer"] == layer]
        counts = {
            sign: sum(
                _qualified_unit(unit)
                and unit["summary"]["majority_sign"] == sign
                for unit in local
            )
            for sign in ("positive", "negative")
        }
        resolved = [
            sign for sign, count in counts.items()
            if count >= int(DECISION_POLICY["source_updates_required"])
        ]
        if len(resolved) > 1:
            raise ArithmeticError("construction resolved both causal signs")
        layer_decisions.append({
            "layer": layer,
            "status": "resolved" if resolved else "unresolved",
            "locked_sign": resolved[0] if resolved else None,
            "positive_qualifying_updates": counts["positive"],
            "negative_qualifying_updates": counts["negative"],
            "units": [
                {
                    "step": unit["step"],
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
        "effect_floor": effect_floor,
        "qualification_duplicate_max_abs": duplicate,
        "qualification_record_sha256": qualification["sha256"],
        "decision_policy": DECISION_POLICY,
        "layer_decisions": layer_decisions,
    }
    payload["content_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    output = campaign_root / "records/construction-lock.json"
    write_json(output, payload)
    print(json.dumps({
        "output": str(output.resolve()),
        "effect_floor": effect_floor,
        "layer_decisions": layer_decisions,
    }, sort_keys=True))
    return payload


def _load_lock(
    path: Path,
    expected_release_id: str = RELEASE_ID,
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    digest = value.pop("content_sha256", None)
    expected = hashlib.sha256(canonical_bytes(value)).hexdigest()
    value["content_sha256"] = digest
    if (
        value.get("schema_version") != LOCK_SCHEMA
        or digest != expected
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("release_id") != expected_release_id
        or value.get("decision_policy") != DECISION_POLICY
    ):
        raise ValueError("direct-work construction lock does not replay")
    return value


def _trajectory_layer_summary(
    units: list[dict[str, Any]], locked_sign: str | None
) -> dict[str, Any]:
    evaluable = int(sum(unit["summary"]["evaluable"] for unit in units))
    positive = int(sum(unit["summary"]["positive"] for unit in units))
    negative = int(sum(unit["summary"]["negative"] for unit in units))
    neutral = int(sum(unit["summary"]["neutral"] for unit in units))
    supporting_updates = 0
    if locked_sign is not None:
        supporting_updates = sum(
            _qualified_unit(unit)
            and unit["summary"]["majority_sign"] == locked_sign
            for unit in units
        )
    return {
        "trajectory": units[0]["trajectory"],
        "layer": units[0]["layer"],
        "planned": len(units) * 96,
        "evaluable": evaluable,
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "locked_sign": locked_sign,
        "supporting_updates": int(supporting_updates),
        "support": bool(
            locked_sign is not None
            and supporting_updates >= int(DECISION_POLICY["source_updates_required"])
        ),
        "median_contrast": float(np.median(np.concatenate([
            unit["contrast"][unit["evaluable_mask"]] for unit in units
        ]))),
        "natural_reopening": int(sum(
            np.count_nonzero(unit["natural_change"][unit["evaluable_mask"]] > 0.0)
            for unit in units
        )),
        "control_reopening": int(sum(
            np.count_nonzero(unit["control_change"][unit["evaluable_mask"]] > 0.0)
            for unit in units
        )),
    }


def _resource_summary(campaign_root: Path, units: list[dict[str, Any]]) -> dict[str, Any]:
    records = []
    clocks = []
    for path in resource_records(campaign_root):
        value = json.loads(path.read_text(encoding="utf-8"))
        if "executor_contract" in value and not record_is_admissible(value):
            raise ValueError(f"resource admission failed: {path}")
        if value.get("schema_version") != RESOURCE_SCHEMA:
            raise ValueError(f"unknown direct-work resource record: {path}")
        records.append(value)
        clocks.append(normalize_resource_clocks(value))
    gpu = [value for value in records if str(value["device"]).startswith("cuda:")]
    devices = {
        device: sum(value["device"] == device for value in gpu)
        for device in ("cuda:0", "cuda:1")
    }
    runtime_root = campaign_root / "runtime"
    attempt_directories = (
        len([path for path in runtime_root.iterdir() if path.is_dir()])
        if runtime_root.is_dir() else 0
    )
    return {
        "accelerator_reservation_wall_seconds": float(sum(
            clock.accelerator_reservation_wall_seconds for clock in clocks
        )),
        "reserved_device_hours": float(sum(
            clock.accelerator_reservation_wall_seconds for clock in clocks
        )) / 3600.0,
        "producer_process_elapsed_seconds": float(sum(
            unit["producer_process_elapsed_seconds"] for unit in units
        )),
        "gpu_attempt_record_count": len(gpu),
        "all_attempt_directory_count": attempt_directories,
        "device_attempt_counts": devices,
        "working_reserved_device_hours": float(
            RESOURCE_BUDGET["working_aggregate_reserved_device_hours"]
        ),
        "hard_reserved_device_hours": float(
            RESOURCE_BUDGET["hard_aggregate_reserved_device_hours"]
        ),
        "peak_gpu_reserved_bytes": max(
            unit["peak_gpu_reserved_bytes"] for unit in units
        ),
    }


def _format_float(value: float) -> str:
    if value == 0.0:
        return "0"
    magnitude = abs(value)
    if magnitude < 0.001 or magnitude >= 10_000:
        exponent = int(np.floor(np.log10(magnitude)))
        coefficient = value / (10.0**exponent)
        return f"{coefficient:.3f}\\times10^{{{exponent}}}"
    return f"{value:.6f}".rstrip("0").rstrip(".")








