#!/usr/bin/env python3
"""Analyze the observable complete-state cocycle confirmation campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from confirm.observable_cocycle import (  # noqa: E402
    aggregate_norm,
    calibration_cell,
    canonical_json,
    digest_object,
    directional_gain,
    relative_response_residual,
    select_layer_horizon,
)
from confirm.observable_cocycle_specs import (  # noqa: E402
    ANALYSIS_SCHEMA,
    CALIBRATION_SCHEMA,
    CAMPAIGN_ID,
    DECISION_POLICY,
    DIRECTION_CLASSES,
    HORIZON_GRID,
    LOCK_SCHEMA,
    NUMERICAL_POLICY,
    PREDICTION_SCHEMA,
    QUALIFICATION_SCHEMA,
    RESOURCE_BUDGET,
    SOURCE_STEPS,
    STRATUM_POLICY,
    STRATUM_SCHEMA,
    TRAJECTORIES,
    VALIDATION_SCHEMA,
    amplitudes,
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name]) for name in archive.files}


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _scalar(record: dict[str, np.ndarray], name: str, kind: type) -> Any:
    if name not in record or np.asarray(record[name]).shape != ():
        raise ValueError(f"record omits scalar {name}")
    return kind(np.asarray(record[name]).item())


def _trajectory(label: str) -> dict[str, Any]:
    matches = [row for row in TRAJECTORIES if row["label"] == label]
    if len(matches) != 1:
        raise ValueError(f"unknown trajectory label: {label}")
    return matches[0]


def _validate_common(
    record: dict[str, np.ndarray],
    *,
    schema: str,
    item: dict[str, Any],
    binding: dict[str, Any],
) -> None:
    label = str(item["trajectory"])
    step = int(item["step"])
    layer = int(item["layer"])
    trajectory = _trajectory(label)
    expected = {
        "schema_version": schema,
        "campaign_id": CAMPAIGN_ID,
        "trajectory": str(trajectory["name"]),
        "seed": int(trajectory["seed"]),
        "source_step": step,
        "layer": layer,
        "checkpoint_sha256": binding["checkpoints"][label][str(step)]["sha256"],
    }
    for name, value in expected.items():
        kind = int if isinstance(value, int) else str
        if _scalar(record, name, kind) != value:
            raise ValueError(f"{label}/{layer}/{step} disagrees at {name}")


def _numerical_health(record: dict[str, np.ndarray]) -> dict[str, Any]:
    replay = max(
        _scalar(record, name, float)
        for name in (
            "maximum_parameter_replay_relative_residual",
            "maximum_first_moment_replay_relative_residual",
            "maximum_second_moment_replay_relative_residual",
        )
    )
    clipping_distance = _scalar(
        record, "minimum_clipping_boundary_distance", float
    )
    reserved = _scalar(record, "resource_peak_gpu_reserved_bytes", int)
    replay_pass = (
        math.isfinite(replay)
        and replay
        <= float(NUMERICAL_POLICY["adamw_replay_relative_tolerance"])
    )
    clipping_pass = (
        math.isfinite(clipping_distance)
        and clipping_distance
        > float(NUMERICAL_POLICY["clipping_boundary_tolerance"])
    )
    memory_pass = (
        reserved <= int(RESOURCE_BUDGET["process_reserved_memory_cap_bytes"])
    )
    return {
        "maximum_adamw_replay_relative_residual": replay,
        "minimum_clipping_boundary_distance": clipping_distance,
        "peak_gpu_reserved_bytes": reserved,
        "replay_pass": replay_pass,
        "clipping_cell_resolved": clipping_pass,
        "memory_pass": memory_pass,
        "valid": replay_pass and clipping_pass and memory_pass,
    }


def calibration_rows(
    record: dict[str, np.ndarray],
    *,
    item: dict[str, Any],
    binding: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the frozen construction gates to one calibration record."""

    _validate_common(
        record, schema=CALIBRATION_SCHEMA, item=item, binding=binding
    )
    horizons = tuple(int(value) for value in record["horizons"].tolist())
    directions = tuple(str(value) for value in record["direction_names"].tolist())
    if horizons != HORIZON_GRID or directions != DIRECTION_CLASSES:
        raise ValueError("calibration grid disagrees with the design")
    expected_amplitudes = np.asarray(
        [amplitudes(name) for name in DIRECTION_CLASSES], dtype=np.float64
    )
    if not np.array_equal(record["amplitudes"], expected_amplitudes):
        raise ValueError("calibration amplitudes drifted")
    expected_shape = (
        len(DIRECTION_CLASSES),
        len(HORIZON_GRID),
        2,
        24,
        4,
        64,
        64,
    )
    if tuple(record["predicted_responses"].shape) != expected_shape:
        raise ValueError("calibration prediction tensor has the wrong shape")
    if record["observed_centered_responses"].shape != expected_shape:
        raise ValueError("calibration observation tensor has the wrong shape")
    health = _numerical_health(record)
    floor = float(DECISION_POLICY["absolute_effect_floor"])
    rows = []
    for direction_index, direction in enumerate(DIRECTION_CLASSES):
        for horizon_index, horizon in enumerate(HORIZON_GRID):
            residuals = [
                relative_response_residual(
                    record["predicted_responses"][
                        direction_index, horizon_index, amplitude_index
                    ],
                    record["observed_centered_responses"][
                        direction_index, horizon_index, amplitude_index
                    ],
                    effect_floor=floor,
                )
                for amplitude_index in range(2)
            ]
            placebo_norm = aggregate_norm(record["placebo_responses"][horizon_index])
            cell = calibration_cell(
                larger_relative_residual=residuals[0]["relative_residual"],
                smaller_relative_residual=residuals[1]["relative_residual"],
                smaller_response_norm=residuals[1]["observed_norm"],
                placebo_norm=placebo_norm,
            )
            cell["qualified"] = bool(cell["qualified"] and health["valid"])
            gain = directional_gain(
                record["source_actions"][direction_index],
                record["endpoint_actions"][direction_index, horizon_index],
                effect_floor=floor,
            )
            rows.append(
                {
                    "trajectory": str(item["trajectory"]),
                    "source_step": int(item["step"]),
                    "layer": int(item["layer"]),
                    "direction": direction,
                    "horizon": int(horizon),
                    "larger_amplitude": float(expected_amplitudes[direction_index, 0]),
                    "smaller_amplitude": float(expected_amplitudes[direction_index, 1]),
                    "larger_relative_residual": residuals[0]["relative_residual"],
                    "smaller_relative_residual": residuals[1]["relative_residual"],
                    "smaller_predicted_norm": residuals[1]["predicted_norm"],
                    "smaller_observed_norm": residuals[1]["observed_norm"],
                    "placebo_norm": placebo_norm,
                    **cell,
                    **gain,
                    "numerical_valid": bool(health["valid"]),
                }
            )
    return rows, health


def _construction(bundle: Path) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, str],
    dict[str, Any],
]:
    inventory = _load_json(bundle / "protocol" / "record-inventory.json")
    binding = _load_json(bundle / "protocol" / "input_binding.json")
    rows: list[dict[str, Any]] = []
    health_rows: list[dict[str, Any]] = []
    digests = {}
    for item in inventory["calibration"]:
        path = bundle / item["path"]
        local_rows, health = calibration_rows(
            _load_npz(path), item=item, binding=binding
        )
        rows.extend(local_rows)
        health_rows.append(
            {
                "trajectory": item["trajectory"],
                "source_step": item["step"],
                "layer": item["layer"],
                **health,
            }
        )
        digests[item["path"]] = sha256_path(path)
    expected = 3 * len(SOURCE_STEPS) * len(DIRECTION_CLASSES) * len(HORIZON_GRID)
    if len(rows) != expected:
        raise ValueError("construction calibration population is incomplete")
    return rows, health_rows, digests, inventory


def build_lock(bundle: Path) -> dict[str, Any]:
    """Freeze layer horizons from construction records only."""

    rows, health_rows, digests, _inventory = _construction(bundle)
    layers = {}
    for layer in range(3):
        selection = select_layer_horizon(
            row for row in rows if int(row["layer"]) == layer
        )
        selection["qualified_cell_count"] = sum(
            bool(row["qualified"]) for row in rows if int(row["layer"]) == layer
        )
        selection["total_cell_count"] = sum(
            1 for row in rows if int(row["layer"]) == layer
        )
        layers[str(layer)] = selection
    unsigned = {
        "schema_version": LOCK_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "selection_uses_roles": ["construction"],
        "calibration_files": digests,
        "layers": layers,
        "numerical_records_valid": all(row["valid"] for row in health_rows),
        "sampled_directions_do_not_estimate_operator_norm": True,
        "finite_horizons_do_not_establish_asymptotic_collapse": True,
    }
    result = dict(unsigned)
    result["lock_sha256"] = digest_object(unsigned)
    return result


def write_lock(bundle: Path, output: Path) -> None:
    _write_atomic(output, canonical_json(build_lock(bundle)))


def _load_lock(bundle: Path, inventory: dict[str, Any]) -> dict[str, Any]:
    path = bundle / inventory["lock_output"]
    lock = _load_json(path)
    if lock.get("schema_version") != LOCK_SCHEMA:
        raise ValueError("construction lock has the wrong schema")
    unsigned = dict(lock)
    digest = unsigned.pop("lock_sha256", None)
    if digest != digest_object(unsigned):
        raise ValueError("construction lock digest does not replay")
    if lock != build_lock(bundle):
        raise ValueError("construction lock drifted from its inputs")
    return lock


def _qualification(
    bundle: Path, inventory: dict[str, Any], binding: dict[str, Any]
) -> dict[str, Any]:
    item = inventory["qualification"]
    record = _load_npz(bundle / item["path"])
    _validate_common(
        record, schema=QUALIFICATION_SCHEMA, item=item, binding=binding
    )
    return {
        "qualified": _scalar(record, "qualified", bool),
        "relative_response_residual": _scalar(
            record, "relative_response_residual", float
        ),
        "maximum_adamw_replay_relative_residual": _scalar(
            record, "maximum_adamw_replay_relative_residual", float
        ),
        "elapsed_seconds": _scalar(record, "resource_elapsed_seconds", float),
        "peak_gpu_reserved_bytes": _scalar(
            record, "resource_peak_gpu_reserved_bytes", int
        ),
    }

def _heldout(
    bundle: Path,
    inventory: dict[str, Any],
    binding: dict[str, Any],
    lock: dict[str, Any],
) -> list[dict[str, Any]]:
    predictions = {
        (str(item["trajectory"]), int(item["layer"]), int(item["step"])): item
        for item in inventory["prediction"]
    }
    rows = []
    floor = float(DECISION_POLICY["absolute_effect_floor"])
    residual_limit = float(
        DECISION_POLICY["maximum_centered_response_relative_residual"]
    )
    for item in inventory["validation"]:
        key = (
            str(item["trajectory"]),
            int(item["layer"]),
            int(item["step"]),
        )
        prediction_item = predictions[key]
        prediction_path = bundle / prediction_item["path"]
        validation_path = bundle / item["path"]
        prediction = _load_npz(prediction_path)
        validation = _load_npz(validation_path)
        _validate_common(
            prediction,
            schema=PREDICTION_SCHEMA,
            item=prediction_item,
            binding=binding,
        )
        _validate_common(
            validation,
            schema=VALIDATION_SCHEMA,
            item=item,
            binding=binding,
        )
        if _scalar(validation, "prediction_sha256", str) != sha256_path(
            prediction_path
        ):
            raise ValueError("validation does not bind the prediction artifact")
        if _scalar(prediction, "lock_sha256", str) != lock["lock_sha256"]:
            raise ValueError("held-out prediction does not bind the lock")
        if _scalar(validation, "lock_sha256", str) != lock["lock_sha256"]:
            raise ValueError("held-out validation does not bind the lock")
        if not np.array_equal(
            prediction["training_chunk_indices"],
            validation["training_chunk_indices"],
        ):
            raise ValueError("prediction and validation use different minibatches")
        if _scalar(
            prediction, "direction_algorithm_sha256", str
        ) != _scalar(validation, "direction_algorithm_sha256", str):
            raise ValueError("prediction and validation directions differ")
        names = tuple(str(value) for value in prediction["direction_names"].tolist())
        if names != DIRECTION_CLASSES:
            raise ValueError("held-out direction registry drifted")
        horizon = _scalar(prediction, "horizon", int)
        if horizon != int(lock["layers"][str(item["layer"])]["selected_horizon"]):
            raise ValueError("held-out horizon disagrees with construction lock")
        expected_amplitudes = np.asarray(
            [amplitudes(name)[1] for name in DIRECTION_CLASSES], dtype=np.float64
        )
        if not np.array_equal(prediction["amplitudes"], expected_amplitudes):
            raise ValueError("held-out amplitudes drifted")
        if not np.array_equal(validation["amplitudes"], expected_amplitudes):
            raise ValueError("validation amplitudes drifted")
        if not np.array_equal(
            prediction["direction_names"], validation["direction_names"]
        ):
            raise ValueError("held-out direction order changed")
        predicted = prediction["predicted_centered_responses"]
        observed = validation["observed_centered_responses"]
        expected_shape = (len(DIRECTION_CLASSES), 24, 4, 64, 64)
        if predicted.shape != expected_shape or observed.shape != expected_shape:
            raise ValueError("held-out response tensor has the wrong shape")
        health = _numerical_health(prediction)
        validation_memory_pass = (
            _scalar(validation, "resource_peak_gpu_reserved_bytes", int)
            <= int(RESOURCE_BUDGET["process_reserved_memory_cap_bytes"])
        )
        for direction_index, direction in enumerate(DIRECTION_CLASSES):
            residual = relative_response_residual(
                predicted[direction_index],
                observed[direction_index],
                effect_floor=floor,
            )
            gain = directional_gain(
                prediction["source_actions"][direction_index],
                prediction["endpoint_actions"][direction_index],
                effect_floor=floor,
            )
            response_pass = bool(
                residual["effect_resolved"]
                and residual["relative_residual"] <= residual_limit
                and health["valid"]
                and validation_memory_pass
            )
            rows.append(
                {
                    "trajectory": item["trajectory"],
                    "source_step": item["step"],
                    "layer": item["layer"],
                    "direction": direction,
                    "horizon": horizon,
                    "amplitude": float(expected_amplitudes[direction_index]),
                    **residual,
                    **gain,
                    "numerical_valid": bool(
                        health["valid"] and validation_memory_pass
                    ),
                    "response_pass": response_pass,
                }
            )
    expected = 2 * len(SOURCE_STEPS) * 3 * len(DIRECTION_CLASSES)
    if len(rows) != expected:
        raise ValueError("held-out validation population is incomplete")
    return rows


def _strata(
    bundle: Path,
    inventory: dict[str, Any],
    binding: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    tolerance = float(STRATUM_POLICY["observed_force_zero_tolerance"])
    for item in inventory["stratum"]:
        record = _load_npz(bundle / item["path"])
        _validate_common(
            record, schema=STRATUM_SCHEMA, item=item, binding=binding
        )
        source = _scalar(record, "source_row_quotient_norm", float)
        force = _scalar(record, "observed_force_norm", float)
        gate_norm = aggregate_norm(record["gate_after"])
        moment_norm = aggregate_norm(record["gate_first_moment_after"])
        variance_norm = aggregate_norm(record["gate_second_moment_after"])
        source_on_face = source <= tolerance
        face_preserved = source_on_face and force <= tolerance
        rows.append(
            {
                "trajectory": item["trajectory"],
                "layer": item["layer"],
                "source_step": item["step"],
                "source_row_quotient_norm": source,
                "observed_force_norm": force,
                "gate_after_norm": gate_norm,
                "gate_first_moment_after_norm": moment_norm,
                "gate_second_moment_after_norm": variance_norm,
                "source_on_face": source_on_face,
                "face_preserved": face_preserved,
                "forced_off_face": source_on_face and not face_preserved,
            }
        )
    if len(rows) != 9:
        raise ValueError("gate-zero stratum population is incomplete")
    return rows


def _resource(bundle: Path) -> dict[str, Any]:
    seconds = 0.0
    maximum_reserved = 0
    records = 0
    attempts = 0
    device_attempts: dict[str, int] = {}
    plan = _load_json(bundle / "protocol" / "launch_plan.json")
    nodes = {str(node["id"]): node for node in plan["nodes"]}
    runtime = bundle / "runtime"
    if runtime.is_dir():
        for path in runtime.glob("*/attempt-*/attempt.json"):
            node = nodes[path.parent.parent.name]
            role = str(node.get("role", node["stage"]))
            device = str(node["device"])
            if role == "analysis" or not device.startswith("cuda:"):
                continue
            attempt = _load_json(path)
            if int(attempt["exit_code"]) != 0:
                seconds += float(attempt["elapsed_seconds"])
            attempts += 1
            device_attempts[device] = device_attempts.get(device, 0) + 1
        for path in runtime.glob("*/completed.json"):
            node = nodes[path.parent.name]
            role = str(node.get("role", node["stage"]))
            if role == "analysis":
                continue
            completion = _load_json(path)
            resource = completion.get("resource", {})
            seconds += float(resource.get("reserved_device_seconds", 0.0))
            maximum_reserved = max(
                maximum_reserved,
                int(resource.get("peak_gpu_reserved_bytes", 0)),
            )
            records += 1
    return {
        "completed_runtime_records_before_analysis": records,
        "attempt_records": attempts,
        "device_attempts": device_attempts,
        "aggregate_reserved_device_seconds": seconds,
        "aggregate_reserved_device_hours": seconds / 3600.0,
        "maximum_process_reserved_bytes": maximum_reserved,
        "working_budget_pass": (
            seconds
            <= 3600.0
            * float(RESOURCE_BUDGET["working_aggregate_reserved_device_hours"])
        ),
        "hard_budget_pass": (
            seconds
            <= 3600.0
            * float(RESOURCE_BUDGET["hard_aggregate_reserved_device_hours"])
            and maximum_reserved
            <= int(RESOURCE_BUDGET["process_reserved_memory_cap_bytes"])
        ),
    }


def _direction_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for direction in DIRECTION_CLASSES:
        local = [row for row in rows if row["direction"] == direction]
        residuals = np.asarray(
            [row["relative_residual"] for row in local], dtype=np.float64
        )
        gains = np.asarray(
            [row["visible_directional_gain"] for row in local], dtype=np.float64
        )
        result.append(
            {
                "direction": direction,
                "response_pass_count": sum(row["response_pass"] for row in local),
                "total_count": len(local),
                "median_relative_residual": float(np.median(residuals)),
                "maximum_relative_residual": float(np.max(residuals)),
                "median_visible_directional_gain": float(np.median(gains)),
                "minimum_visible_directional_gain": float(np.min(gains)),
                "maximum_visible_directional_gain": float(np.max(gains)),
                "attenuating_count": sum(
                    row["directionally_attenuating"] for row in local
                ),
            }
        )
    return result


def _number(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) >= 1.0e4 or abs(value) < 1.0e-3:
        coefficient, exponent = f"{value:.3e}".split("e")
        return rf"{coefficient}\mathord{{\times}}10^{{{int(exponent)}}}"
    return f"{value:.4f}"






def _evidence_paths(bundle: Path, inventory: dict[str, Any]) -> list[Path]:
    relative = [inventory["qualification"]["path"], inventory["lock_output"]]
    for category in ("calibration", "prediction", "validation", "stratum"):
        relative.extend(item["path"] for item in inventory[category])
    relative.extend(
        [
            "reports/final-analysis.json",
            "reports/confirmation_results.tex",
            "reports/confirmation_macros.tex",
        ]
    )
    return [bundle / path for path in relative]


def final_analysis(bundle: Path) -> dict[str, Any]:
    rows, health_rows, _digests, inventory = _construction(bundle)
    binding = _load_json(bundle / "protocol" / "input_binding.json")
    lock = _load_lock(bundle, inventory)
    qualification = _qualification(bundle, inventory, binding)
    heldout_rows = _heldout(bundle, inventory, binding, lock)
    stratum_rows = _strata(bundle, inventory, binding)
    direction_summary = _direction_summary(heldout_rows)
    resource = _resource(bundle)
    return {
        "schema_version": ANALYSIS_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "state_dimension": 61_868_742,
        "qualification": qualification,
        "construction": {
            "lock_sha256": lock["lock_sha256"],
            "layers": lock["layers"],
            "qualified_cell_count": sum(row["qualified"] for row in rows),
            "total_cell_count": len(rows),
            "numerical_record_count": len(health_rows),
            "numerical_valid_count": sum(row["valid"] for row in health_rows),
            "rows": rows,
        },
        "heldout": {
            "response_pass_count": sum(row["response_pass"] for row in heldout_rows),
            "total_count": len(heldout_rows),
            "attenuating_count": sum(
                row["directionally_attenuating"] for row in heldout_rows
            ),
            "by_direction": direction_summary,
            "rows": heldout_rows,
        },
        "stratum": {
            "preserved_count": sum(row["face_preserved"] for row in stratum_rows),
            "forced_off_face_count": sum(
                row["forced_off_face"] for row in stratum_rows
            ),
            "total_count": len(stratum_rows),
            "rows": stratum_rows,
        },
        "resource": resource,
        "interpretation": {
            "matrix_free_complete_state_dimension_checked": True,
            "sampled_directions_only": True,
            "operator_norm_inference_authorized": False,
            "finite_horizon_asymptotic_inference_authorized": False,
            "gate_zero_face_requires_invariance_test": True,
            "adverse_outcomes_retained": True,
        },
    }






