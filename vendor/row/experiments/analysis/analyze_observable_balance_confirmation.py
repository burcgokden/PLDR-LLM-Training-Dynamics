#!/usr/bin/env python3
"""Analyze the sealed finite-increment observable-balance campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import sys
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
ROOT = EXPERIMENTS.parent
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(ROOT))

from confirm.observable_balance import (  # noqa: E402
    balance_metrics,
    campaign_classification,
    select_layer_horizon,
)
from confirm.observable_balance_specs import (  # noqa: E402
    ANALYSIS_SCHEMA,
    ARCHIVE_PRIOR_ATTEMPTS,
    CAMPAIGN_ID,
    CONSTRUCTION_SCHEMA,
    DIRECTION_CLASSES,
    HORIZON_GRID,
    LOCK_SCHEMA,
    PREDICTION_SCHEMA,
    QUALIFICATION_SCHEMA,
    RESOURCE_BUDGET,
    SOURCE_STEPS,
    STRATUM_SCHEMA,
    TRAJECTORIES,
    VALIDATION_SCHEMA,
)
from confirm.observable_cocycle import canonical_json, digest_object  # noqa: E402


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(value: str) -> Path:
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(
        part in {"", ".", ".."} for part in candidate.parts
    ):
        raise ValueError(f"unsafe bundle path: {value}")
    return Path(*candidate.parts)


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def _write_atomic(path: str | Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(destination)


def _scalar(record: dict[str, np.ndarray], name: str, kind: type) -> Any:
    if name not in record or np.asarray(record[name]).shape != ():
        raise ValueError(f"record scalar is missing or malformed: {name}")
    return kind(np.asarray(record[name]).item())


def _terminal(record: dict[str, np.ndarray]) -> bool:
    return _scalar(record, "terminal_status", str) != "complete"


def _validate_common(record: dict[str, np.ndarray], schema: str) -> None:
    if _scalar(record, "schema_version", str) != schema:
        raise ValueError(f"record has the wrong schema: {schema}")
    if _scalar(record, "campaign_id", str) != CAMPAIGN_ID:
        raise ValueError("record campaign identity drifted")


def _trajectory_label(name: str) -> str:
    matches = [str(row["label"]) for row in TRAJECTORIES if row["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"unknown trajectory name: {name}")
    return matches[0]


def _inventory(bundle: Path) -> dict[str, Any]:
    value = _load_json(bundle / "protocol" / "record-inventory.json")
    if value.get("campaign_id") != CAMPAIGN_ID:
        raise ValueError("record inventory campaign identity drifted")
    return value


def construction_rows(bundle: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    terminal_records = 0
    inventory = _inventory(bundle)
    for relative in inventory["construction"]:
        record = _load_npz(bundle / _safe_relative(relative))
        _validate_common(record, CONSTRUCTION_SCHEMA)
        layer = _scalar(record, "layer", int)
        source = _scalar(record, "source_step", int)
        if _terminal(record):
            terminal_records += 1
            for direction in DIRECTION_CLASSES:
                for horizon in HORIZON_GRID:
                    for amplitude_index in (0, 1):
                        rows.append(
                            {
                                "layer": layer,
                                "source_step": source,
                                "direction": direction,
                                "horizon": horizon,
                                "amplitude_index": amplitude_index,
                                "evaluable": False,
                                "reconstruction_relative_residual": math.inf,
                                "terminal": True,
                            }
                        )
            continue
        horizons = tuple(int(value) for value in record["horizons"].tolist())
        directions = tuple(str(value) for value in record["direction_names"].tolist())
        if horizons != HORIZON_GRID or directions != DIRECTION_CLASSES:
            raise ValueError("construction grids drifted")
        for direction_index, direction in enumerate(directions):
            for horizon_index, horizon in enumerate(horizons):
                for amplitude_index in (0, 1):
                    index = (direction_index, horizon_index, amplitude_index)
                    terms = np.asarray(record["balance_terms"][index])
                    observed = np.asarray(record["observed_responses"][index])
                    recomputed = balance_metrics(
                        terms[0], terms[1], terms[2], observed
                    )
                    stored = float(
                        record["reconstruction_relative_residuals"][index]
                    )
                    if not math.isclose(
                        stored,
                        float(recomputed["reconstruction_relative_residual"]),
                        rel_tol=1.0e-10,
                        abs_tol=1.0e-13,
                    ):
                        raise ValueError("construction balance metric did not replay")
                    if not np.allclose(
                        record["cross_grams"][index],
                        recomputed["cross_gram"],
                        rtol=2.0e-6,
                        atol=1.0e-12,
                    ):
                        raise ValueError("construction cross-Gram matrix did not replay")
                    rows.append(
                        {
                            "layer": layer,
                            "source_step": source,
                            "direction": direction,
                            "horizon": horizon,
                            "amplitude_index": amplitude_index,
                            "amplitude": float(record["amplitudes"][direction_index, amplitude_index]),
                            "evaluable": bool(record["evaluable"][index]),
                            "reconstruction_relative_residual": stored,
                            "term_norms": [float(value) for value in record["term_norms"][index]],
                            "response_norm": float(record["response_norms"][index]),
                            "cancellation_index": float(record["cancellation_indices"][index]),
                            "response_to_homogeneous_ratio": float(
                                record["response_to_homogeneous_ratios"][index]
                            ),
                            "cancellation_dominant": bool(
                                record["cancellation_dominant"][index]
                            ),
                            "dominant_correction": str(
                                record["dominant_correction"][index]
                            ),
                            "route_changed": bool(record["clipping_route_changed"][index]),
                            "terminal": False,
                        }
                    )
    expected = 3 * len(SOURCE_STEPS) * len(DIRECTION_CLASSES) * len(HORIZON_GRID) * 2
    if len(rows) != expected:
        raise ValueError("construction population is incomplete")
    return rows, terminal_records


def build_lock(bundle: Path) -> dict[str, Any]:
    rows, terminal_records = construction_rows(bundle)
    layers = {}
    for layer in range(3):
        selected = select_layer_horizon(
            row for row in rows if int(row["layer"]) == layer
        )
        selected["construction_terminal_records"] = sum(
            bool(row["terminal"])
            for row in rows
            if int(row["layer"]) == layer
        ) // (len(DIRECTION_CLASSES) * len(HORIZON_GRID) * 2)
        layers[str(layer)] = selected
    unsigned = {
        "schema_version": LOCK_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "construction_record_sha256": {
            relative: sha256_path(bundle / _safe_relative(relative))
            for relative in _inventory(bundle)["construction"]
        },
        "construction_terminal_records": terminal_records,
        "layers": layers,
    }
    return {**unsigned, "lock_sha256": digest_object(unsigned)}


def write_lock(bundle: Path, output: Path) -> None:
    _write_atomic(output, canonical_json(build_lock(bundle)) + b"\n")


def _load_lock(bundle: Path) -> dict[str, Any]:
    lock = _load_json(bundle / _inventory(bundle)["lock"])
    if lock != build_lock(bundle):
        raise ValueError("construction lock does not replay from construction records")
    return lock


def heldout_rows(bundle: Path) -> tuple[list[dict[str, Any]], int]:
    inventory = _inventory(bundle)
    predictions = {Path(path).name.replace("prediction-", ""): path for path in inventory["prediction"]}
    rows: list[dict[str, Any]] = []
    terminal_records = 0
    for relative in inventory["validation"]:
        validation = _load_npz(bundle / _safe_relative(relative))
        _validate_common(validation, VALIDATION_SCHEMA)
        key = Path(relative).name.replace("validation-", "")
        prediction = _load_npz(bundle / _safe_relative(predictions[key]))
        _validate_common(prediction, PREDICTION_SCHEMA)
        if _scalar(prediction, "stage", str) != "prediction":
            raise ValueError("held-out prediction has the wrong semantic role")
        if _scalar(validation, "stage", str) != "validation":
            raise ValueError("held-out validation has the wrong semantic role")
        if _scalar(validation, "prediction_sha256", str) != sha256_path(
            bundle / _safe_relative(predictions[key])
        ):
            raise ValueError("validation is not bound to its prediction")
        trajectory = _scalar(validation, "trajectory", str)
        label = _trajectory_label(trajectory)
        layer = _scalar(validation, "layer", int)
        source = _scalar(validation, "source_step", int)
        horizon = _scalar(validation, "horizon", int)
        if _terminal(validation):
            terminal_records += 1
            for direction in DIRECTION_CLASSES:
                rows.append(
                    {
                        "trajectory": trajectory,
                        "trajectory_label": label,
                        "layer": layer,
                        "source_step": source,
                        "horizon": horizon,
                        "direction": direction,
                        "evaluable": False,
                        "cancellation_dominant": False,
                        "terminal": True,
                    }
                )
            continue
        directions = tuple(str(value) for value in validation["direction_names"].tolist())
        if directions != DIRECTION_CLASSES:
            raise ValueError("held-out direction order drifted")
        for index, direction in enumerate(directions):
            terms = np.asarray(validation["balance_terms"][index])
            observed = np.asarray(validation["observed_responses"][index])
            recomputed = balance_metrics(terms[0], terms[1], terms[2], observed)
            stored_residual = float(
                validation["reconstruction_relative_residuals"][index]
            )
            if not math.isclose(
                stored_residual,
                float(recomputed["reconstruction_relative_residual"]),
                rel_tol=1.0e-10,
                abs_tol=1.0e-13,
            ):
                raise ValueError("held-out balance metric did not replay")
            if bool(validation["cancellation_dominant"][index]) != bool(
                recomputed["cancellation_dominant"]
            ):
                raise ValueError("held-out cancellation decision did not replay")
            rows.append(
                {
                    "trajectory": trajectory,
                    "trajectory_label": label,
                    "layer": layer,
                    "source_step": source,
                    "horizon": horizon,
                    "direction": direction,
                    "amplitude": float(validation["amplitudes"][index]),
                    "term_norms": [float(value) for value in validation["term_norms"][index]],
                    "response_norm": float(validation["response_norms"][index]),
                    "reconstruction_relative_residual": stored_residual,
                    "cancellation_index": float(validation["cancellation_indices"][index]),
                    "response_to_homogeneous_ratio": float(
                        validation["response_to_homogeneous_ratios"][index]
                    ),
                    "homogeneous_correction_inner_product": float(
                        validation["homogeneous_correction_inner_products"][index]
                    ),
                    "evaluable": bool(validation["evaluable"][index]),
                    "cancellation_dominant": bool(
                        validation["cancellation_dominant"][index]
                    ),
                    "dominant_correction": str(
                        validation["dominant_correction"][index]
                    ),
                    "route_changed": bool(validation["clipping_route_changed"][index]),
                    "terminal": False,
                }
            )
    if len(rows) != 54:
        raise ValueError("held-out population is incomplete")
    return rows, terminal_records


def stratum_rows(bundle: Path) -> tuple[list[dict[str, Any]], int]:
    rows = []
    terminal_records = 0
    for relative in _inventory(bundle)["stratum_control"]:
        record = _load_npz(bundle / _safe_relative(relative))
        _validate_common(record, STRATUM_SCHEMA)
        if _terminal(record):
            terminal_records += 1
            rows.append(
                {
                    "trajectory_label": _trajectory_label(
                        _scalar(record, "trajectory", str)
                    ),
                    "layer": _scalar(record, "layer", int),
                    "source_step": _scalar(record, "source_step", int),
                    "theory_code_pass": False,
                    "terminal": True,
                }
            )
            continue
        rows.append(
            {
                "trajectory_label": _trajectory_label(
                    _scalar(record, "trajectory", str)
                ),
                "layer": _scalar(record, "layer", int),
                "source_step": _scalar(record, "source_step", int),
                "theory_code_pass": _scalar(record, "theory_code_pass", bool),
                "natural_gate_bitwise": _scalar(
                    record, "natural_gate_bitwise", bool
                ),
                "controlled_face_bitwise": _scalar(
                    record, "controlled_face_bitwise", bool
                ),
                "other_gradients_bitwise": _scalar(
                    record, "other_gradients_bitwise", bool
                ),
                "natural_force_norm": _scalar(record, "natural_force_norm", float),
                "controlled_force_norm": _scalar(
                    record, "controlled_force_norm", float
                ),
                "nonzero_clipped_gate_gradients": int(
                    np.count_nonzero(record["natural_clipped_gate_gradient"])
                ),
                "terminal": False,
            }
        )
    if len(rows) != 9:
        raise ValueError("gate-control population is incomplete")
    return rows, terminal_records


def resource_summary(bundle: Path) -> dict[str, Any]:
    plan = _load_json(bundle / "protocol" / "launch-plan.json")
    nodes = {str(row["id"]): row for row in plan["nodes"]}
    attempt_seconds = 0.0
    attempt_records = 0
    device_attempts: dict[str, int] = {}
    for path in sorted((bundle / "runtime").glob("*/attempt-*/attempt.json")):
        node_id = path.parent.parent.name
        node = nodes[node_id]
        if node["role"] == "analysis":
            continue
        attempt = _load_json(path)
        device = str(node["device"])
        if device.startswith("cuda:"):
            attempt_seconds += float(attempt["elapsed_seconds"])
            attempt_records += 1
            device_attempts[device] = device_attempts.get(device, 0) + 1
    maximum_reserved = 0
    completed = 0
    for path in sorted((bundle / "runtime").glob("*/completed.json")):
        node_id = path.parent.name
        node = nodes[node_id]
        if node["role"] == "analysis":
            continue
        record = _load_json(path)
        completed += 1
        maximum_reserved = max(
            maximum_reserved,
            int(record.get("resource", {}).get("peak_gpu_reserved_bytes", 0)),
        )
    current_hours = attempt_seconds / 3600.0
    prior_hours = sum(
        float(row["hours"]) for row in ARCHIVE_PRIOR_ATTEMPTS["components"]
    )
    return {
        "definition": ARCHIVE_PRIOR_ATTEMPTS["definition"],
        "attempt_records": attempt_records,
        "completed_nonanalysis_nodes": completed,
        "device_attempts": device_attempts,
        "aggregate_reserved_device_seconds": attempt_seconds,
        "aggregate_reserved_device_hours": current_hours,
        "maximum_process_reserved_bytes": maximum_reserved,
        "working_limit_hours": float(
            RESOURCE_BUDGET["working_aggregate_reserved_device_hours"]
        ),
        "hard_limit_hours": float(
            RESOURCE_BUDGET["hard_aggregate_reserved_device_hours"]
        ),
        "prior_archive_hours": prior_hours,
        "archive_total_hours": prior_hours + current_hours,
        "within_limits": bool(
            current_hours
            <= float(RESOURCE_BUDGET["hard_aggregate_reserved_device_hours"])
            and maximum_reserved
            <= int(RESOURCE_BUDGET["process_reserved_memory_cap_bytes"])
        ),
    }


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _construction_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    horizon_rows = []
    for horizon in HORIZON_GRID:
        local = [
            row
            for row in rows
            if row["horizon"] == horizon and row["amplitude_index"] == 1
        ]
        finite = [row for row in local if not row["terminal"]]
        horizon_rows.append(
            {
                "horizon": horizon,
                "evaluable": sum(bool(row["evaluable"]) for row in local),
                "total": len(local),
                "median_homogeneous_norm": _median(
                    [row["term_norms"][0] for row in finite]
                ),
                "median_state_correction_norm": _median(
                    [row["term_norms"][1] for row in finite]
                ),
                "median_observation_correction_norm": _median(
                    [row["term_norms"][2] for row in finite]
                ),
                "median_response_norm": _median(
                    [row["response_norm"] for row in finite]
                ),
                "median_cancellation_index": _median(
                    [row["cancellation_index"] for row in finite]
                ),
                "route_changes": sum(bool(row["route_changed"]) for row in finite),
            }
        )
    return {"horizons": horizon_rows}


def final_analysis(bundle: Path) -> dict[str, Any]:
    lock = _load_lock(bundle)
    construction, construction_terminals = construction_rows(bundle)
    heldout, heldout_terminals = heldout_rows(bundle)
    strata, stratum_terminals = stratum_rows(bundle)
    classification = campaign_classification(heldout)
    direction_summary = []
    for direction in DIRECTION_CLASSES:
        local = [row for row in heldout if row["direction"] == direction]
        finite = [row for row in local if not row["terminal"]]
        direction_summary.append(
            {
                "direction": direction,
                "total": len(local),
                "evaluable": sum(bool(row["evaluable"]) for row in local),
                "cancellation_dominant": sum(
                    bool(row["cancellation_dominant"]) for row in local
                ),
                "median_cancellation_index": _median(
                    [row["cancellation_index"] for row in finite]
                ),
                "median_response_to_homogeneous_ratio": _median(
                    [row["response_to_homogeneous_ratio"] for row in finite]
                ),
            }
        )
    result = {
        "schema_version": ANALYSIS_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "construction": {
            **_construction_summary(construction),
            "lock": lock,
            "terminal_records": construction_terminals,
        },
        "heldout": {
            "rows": heldout,
            "direction_summary": direction_summary,
            "classification": classification,
            "terminal_records": heldout_terminals,
            "total": len(heldout),
            "evaluable": sum(bool(row["evaluable"]) for row in heldout),
            "cancellation_dominant": sum(
                bool(row["cancellation_dominant"]) for row in heldout
            ),
            "route_changed": sum(bool(row.get("route_changed")) for row in heldout),
            "state_correction_dominant": sum(
                row.get("dominant_correction") == "state" for row in heldout
            ),
            "observation_correction_dominant": sum(
                row.get("dominant_correction") == "observation" for row in heldout
            ),
        },
        "stratum_control": {
            "rows": strata,
            "terminal_records": stratum_terminals,
            "theory_code_passes": sum(
                bool(row["theory_code_pass"]) for row in strata
            ),
            "natural_reopenings": sum(
                float(row.get("natural_force_norm", 0.0)) > 0.0 for row in strata
            ),
            "controlled_preservations": sum(
                bool(row.get("controlled_face_bitwise")) for row in strata
            ),
        },
        "resource": resource_summary(bundle),
    }
    if not result["resource"]["within_limits"]:
        raise ValueError("observable-balance resource gate failed")
    return result


def _number(value: float) -> str:
    if value == 0.0:
        return "0"
    magnitude = abs(value)
    if magnitude < 1.0e-3 or magnitude >= 1.0e4:
        exponent = int(math.floor(math.log10(magnitude)))
        mantissa = value / (10.0**exponent)
        return rf"${mantissa:.3f}\mathord{{\times}}10^{{{exponent}}}$"
    return f"${value:.4f}$"




def _cell_code(row: dict[str, Any]) -> str:
    if not row["evaluable"]:
        return "X"
    return "C" if row["cancellation_dominant"] else "N"




def _evidence_paths(bundle: Path) -> list[Path]:
    inventory = _inventory(bundle)
    relative = [
        inventory["qualification"],
        *inventory["construction"],
        inventory["lock"],
        *inventory["prediction"],
        *inventory["validation"],
        *inventory["stratum_control"],
        "reports/final-analysis.json",
        "reports/observable_balance_macros.tex",
        "reports/observable_balance_results.tex",
    ]
    return [bundle / _safe_relative(path) for path in relative]






