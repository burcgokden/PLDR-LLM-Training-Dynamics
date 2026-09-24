#!/usr/bin/env python3
"""Reanalyze a completed row-energy run with observer-aware rules.

The analysis leaves the historical registered decision unchanged. It applies
the declared observer floor, reports all-layer increment events, reconstructs
the holdout required by the design-variance rule, and treats represented
zeros as numerical events unless a separate checkpoint audit certifies the
underlying real-arithmetic branch.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import platform
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from row_rgmap.analysis import verify_record_seal  # noqa: E402
from row_rgmap.provenance_v3 import git_blob_descriptor  # noqa: E402
from scripts.verify_registered_artifacts import (  # noqa: E402
    load_latest_lineage,
    verify_analysis_sources,
    verify_resolved_inputs,
)


SOURCE_PATHS = (
    "scripts/analyze_observer_regime.py",
    "scripts/verify_registered_artifacts.py",
    "src/row_rgmap/analysis.py",
    "src/row_rgmap/provenance_v3.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: dict[str, Any]) -> bytes:
    unsigned = dict(value)
    unsigned.pop("record_sha256", None)
    return json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def content_record(value: dict[str, Any]) -> dict[str, Any]:
    record = dict(value)
    record["record_sha256"] = hashlib.sha256(canonical_bytes(record)).hexdigest()
    return record


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def committed_identity() -> tuple[str, list[dict[str, Any]]]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    sources = []
    for repository_path in SOURCE_PATHS:
        digest, size = git_blob_descriptor(ROOT, commit, repository_path)
        worktree_path = ROOT / repository_path
        if (
            not worktree_path.is_file()
            or file_sha256(worktree_path) != digest
            or worktree_path.stat().st_size != size
        ):
            raise RuntimeError(
                f"observer source differs from commit {commit}: {repository_path}"
            )
        sources.append(
            {
                "repository_path": repository_path,
                "code_commit": commit,
                "sha256": digest,
                "size_bytes": size,
            }
        )
    return commit, sources


def scientific_payload(record: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "observer_floor",
        "absolute_route_discrepancy_bound_at_audited_checkpoints",
        "route_bound_coverage",
        "represented_arithmetic",
        "cell_observer_bands_and_power",
        "power_summary",
        "regular_drift_heldout_prediction_stage_count",
        "interpretation",
    )
    payload = {key: record[key] for key in keys}
    represented = dict(payload["represented_arithmetic"])
    excursions = dict(represented["face_excursions"])
    excursions.pop("status", None)
    represented["face_excursions"] = excursions
    payload["represented_arithmetic"] = represented
    power = dict(payload["power_summary"])
    power.pop("status", None)
    payload["power_summary"] = power
    return payload


def load_combined(run_root: Path, result: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    descriptor = result["combined_energy_artifact"]
    path = (run_root / descriptor["path"]).resolve()
    path.relative_to(run_root)
    if file_sha256(path) != descriptor["sha256"]:
        raise ValueError("combined energy digest mismatch")
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"energies", "steps", "trajectory_ids", "map_ids"}:
            raise ValueError("combined energy archive has an unknown schema")
        energies = np.asarray(payload["energies"], dtype=np.float64)
        steps = np.asarray(payload["steps"])
        trajectory_ids = [str(value) for value in payload["trajectory_ids"]]
        map_ids = [str(value) for value in payload["map_ids"]]
    if not np.isfinite(energies).all() or np.any(energies < 0.0):
        raise ValueError("energies must be finite and nonnegative")
    if not np.array_equal(steps, np.arange(energies.shape[1])):
        raise ValueError("energy steps are not consecutive")
    return energies, steps, trajectory_ids, map_ids


def required_holdout_updates(cell: dict[str, Any], protocol: dict[str, Any]) -> int:
    """Reproduce the registered fixed-variance ``n**-1/2`` projection.

    The registered result holds its Student critical value and simultaneous
    design-variance upper bound fixed after the 64 design batches. Therefore
    the projected MDE scales exactly as the inverse square root of holdout
    updates. The nearest whole-update estimate matches that declared analysis;
    it is not a new sequential-power calculation.
    """

    projected_mde = float(cell["projected_holdout_mde"])
    target = float(protocol["analysis"]["target_mde_per_update"])
    executed = int(protocol["segments"]["holdout_updates"])
    return int(round(executed * (projected_mde / target) ** 2))


def layer_from_map_id(map_id: str) -> int:
    match = re.fullmatch(r"c[0-9]+[.]L([0-9]+)[.]H[0-9]+", map_id)
    if match is None:
        raise ValueError(f"unrecognized map identifier: {map_id}")
    return int(match.group(1))


def excursion_summary(energies: np.ndarray, floor: float) -> dict[str, Any]:
    peaks: list[float] = []
    lengths: list[int] = []
    restarts: list[float] = []
    for trajectory in energies:
        for map_series in trajectory.T:
            face = map_series == 0.0
            zero_steps = np.flatnonzero(face)
            restart_steps = np.flatnonzero(face[:-1] & ~face[1:]) + 1
            for restart_step in restart_steps:
                next_zero_index = int(np.searchsorted(zero_steps, restart_step))
                stop = (
                    int(zero_steps[next_zero_index])
                    if next_zero_index < len(zero_steps)
                    else len(map_series)
                )
                segment = map_series[restart_step:stop]
                if not len(segment):
                    raise AssertionError("a positive restart produced an empty excursion")
                restarts.append(float(segment[0]))
                peaks.append(float(segment.max()))
                lengths.append(int(len(segment)))
    peak_array = np.asarray(peaks, dtype=np.float64)
    length_array = np.asarray(lengths, dtype=np.int64)
    restart_array = np.asarray(restarts, dtype=np.float64)
    if not peaks:
        return {
            "status": "NOT_APPLICABLE_NO_EXCURSIONS",
            "excursion_count": 0,
            "all_peaks_below_observer_floor": None,
            "peak_maximum": None,
            "restart_minimum": None,
            "restart_median": None,
            "length_minimum": None,
            "length_median": None,
            "length_p95": None,
            "length_maximum": None,
            "single_state_fraction": None,
        }
    return {
        "status": "OBSERVED_EXCURSIONS",
        "excursion_count": int(len(peaks)),
        "all_peaks_below_observer_floor": bool(np.all(peak_array < floor)),
        "peak_maximum": float(peak_array.max()),
        "restart_minimum": float(restart_array.min()),
        "restart_median": float(np.median(restart_array)),
        "length_minimum": int(length_array.min()),
        "length_median": float(np.median(length_array)),
        "length_p95": float(np.quantile(length_array, 0.95)),
        "length_maximum": int(length_array.max()),
        "single_state_fraction": float(np.mean(length_array == 1)),
    }


def analyze(
    run_root: Path,
    result_path: Path,
    *,
    code_commit: str,
    analysis_sources: list[dict[str, Any]],
    predecessor_path: Path | None = None,
) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    verify_record_seal(result)
    protocol_path = run_root / "protocol-resolved.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    energies, steps, trajectory_ids, map_ids = load_combined(run_root, result)
    floor = float(protocol["observer_effect_floor"])
    layers = np.asarray([layer_from_map_id(map_id) for map_id in map_ids])
    decision_cells = {
        (row["trajectory_id"], int(row["layer"])): row
        for row in result["registered_decision"]["cells"]
    }
    metadata_by_id = {}
    absolute_route_bound = 0.0
    gate_checks = 0
    gpu_seconds = 0.0
    for descriptor in result["producer_artifacts"]:
        path = (run_root / descriptor["metadata"]["path"]).resolve()
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata_by_id[descriptor["trajectory_id"]] = metadata
        capture = metadata["capture"]
        absolute_route_bound = max(
            absolute_route_bound,
            float(capture["gate_shape_maximum_quotient_energy_absolute_residual"]),
        )
        gate_checks += int(capture["hook_identity_checks"]) * len(map_ids)
        gpu_seconds += float(metadata["wall_seconds"])

    segment = protocol["segments"]
    holdout_start = int(segment["burn_in_updates"]) + int(segment["design_updates"])
    cell_rows = []
    for trajectory_index, trajectory_id in enumerate(trajectory_ids):
        for layer in sorted(set(layers.tolist())):
            mask = layers == layer
            values = energies[trajectory_index][:, mask]
            holdout = values[holdout_start:]
            key = (trajectory_id, layer)
            decision = decision_cells[key]
            row = {
                "trajectory_id": trajectory_id,
                "layer": layer,
                "state_count": int(values.size),
                "represented_zero_state_count": int(np.count_nonzero(values == 0.0)),
                "positive_below_floor_count": int(np.count_nonzero((values > 0.0) & (values < floor))),
                "fraction_below_route_bound": float(np.mean(values < absolute_route_bound)),
                "holdout_fraction_below_route_bound": float(np.mean(holdout < absolute_route_bound)),
                "holdout_minimum_energy": float(holdout.min()),
                "registered_flow_classification": decision["flow_classification"],
            }
            if decision.get("positive_excursion_complete"):
                row["required_holdout_updates"] = required_holdout_updates(decision, protocol)
                row["required_holdout_multiple"] = (
                    row["required_holdout_updates"]
                    / int(segment["holdout_updates"])
                )
            else:
                row["required_holdout_updates"] = None
                row["required_holdout_multiple"] = None
            cell_rows.append(row)

    event_threshold = float(
        protocol["analysis"]["excursion_events"]["absolute_log_gain_threshold"]
    )
    event_rows = []
    for layer in sorted(set(layers.tolist())):
        layer_values = energies[:, :, layers == layer]
        valid = (layer_values[:, :-1, :] > 0.0) & (layer_values[:, 1:, :] > 0.0)
        logs = np.zeros_like(layer_values[:, :-1, :])
        logs[valid] = np.log(layer_values[:, 1:, :][valid]) - np.log(
            layer_values[:, :-1, :][valid]
        )
        events = valid & (np.abs(logs) >= event_threshold)
        event_rows.append(
            {
                "layer": layer,
                "event_count": int(np.count_nonzero(events)),
                "maximum_absolute_log_gain": float(np.abs(logs[events]).max())
                if np.any(events)
                else 0.0,
            }
        )

    excursion = excursion_summary(energies, floor)
    complete_power = [
        row for row in cell_rows if row["required_holdout_updates"] is not None
    ]
    regular_reached = 0
    for row in decision_cells.values():
        prediction = row.get("regular_drift_prediction", {})
        if prediction.get("eligible") and prediction.get("design_fit", {}).get(
            "r_squared_about_mean", -math.inf
        ) >= float(protocol["analysis"]["regular_drift_prediction"]["minimum_design_r_squared"]):
            gamma = prediction.get("gamma")
            rule = protocol["analysis"]["regular_drift_prediction"]
            if gamma is not None and float(rule["minimum_gamma"]) <= gamma <= float(rule["maximum_gamma"]):
                regular_reached += 1

    if complete_power:
        power_summary = {
            "status": "ESTIMATED_FROM_COMPLETE_POSITIVE_CELLS",
            "executed_holdout_updates": int(segment["holdout_updates"]),
            "minimum_required_holdout_updates": min(
                row["required_holdout_updates"] for row in complete_power
            ),
            "maximum_required_holdout_updates": max(
                row["required_holdout_updates"] for row in complete_power
            ),
            "minimum_required_multiple": min(
                row["required_holdout_multiple"] for row in complete_power
            ),
            "maximum_required_multiple": max(
                row["required_holdout_multiple"] for row in complete_power
            ),
        }
    else:
        power_summary = {
            "status": "NOT_APPLICABLE_NO_COMPLETE_POSITIVE_CELLS",
            "executed_holdout_updates": int(segment["holdout_updates"]),
            "minimum_required_holdout_updates": None,
            "maximum_required_holdout_updates": None,
            "minimum_required_multiple": None,
            "maximum_required_multiple": None,
        }

    payload: dict[str, Any] = {
            "schema_version": "pldr-row-rg-observer-regime-analysis-v5",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "code_commit": code_commit,
            "analysis_sources": analysis_sources,
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
            "source_run_id": result["run_id"],
            "source_result_record_sha256": result["record_sha256"],
            "inputs": {
                "result": {
                    "path": str(result_path.relative_to(run_root)),
                    "sha256": file_sha256(result_path),
                    "size_bytes": result_path.stat().st_size,
                    "record_sha256": result["record_sha256"],
                },
                "protocol": {
                    "path": str(protocol_path.relative_to(run_root)),
                    "sha256": file_sha256(protocol_path),
                    "size_bytes": protocol_path.stat().st_size,
                },
                "combined_energy": result["combined_energy_artifact"],
            },
            "observer_floor": floor,
            "absolute_route_discrepancy_bound_at_audited_checkpoints": absolute_route_bound,
            "route_bound_coverage": {
                "checkpoint_map_evaluations": gate_checks,
                "recorded_map_states": int(energies.size),
                "coverage_fraction": gate_checks / int(energies.size),
            },
            "represented_arithmetic": {
                "state_count": int(energies.size),
                "represented_zero_state_count": int(np.count_nonzero(energies == 0.0)),
                "positive_below_observer_floor_count": int(
                    np.count_nonzero((energies > 0.0) & (energies < floor))
                ),
                "face_excursions": excursion,
                "all_layer_large_increment_counts": event_rows,
            },
            "cell_observer_bands_and_power": cell_rows,
            "power_summary": power_summary,
            "regular_drift_heldout_prediction_stage_count": regular_reached,
            "completed_run_gpu_hours": gpu_seconds / 3600.0,
            "interpretation": {
                "represented_zero_is_real_face_certificate": False,
                "fixed_floor_establishes_zero_limit": False,
                "route_bound_is_uniform_between_checkpoints": False,
                "historical_registered_decision_was_changed": False,
            },
        }
    if predecessor_path is not None:
        predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
        verify_record_seal(predecessor)
        payload["predecessor"] = {
            "record_sha256": predecessor["record_sha256"],
            "file_sha256": file_sha256(predecessor_path),
            "size_bytes": predecessor_path.stat().st_size,
            "scientific_leaves_equal": scientific_payload(predecessor)
            == scientific_payload(payload),
        }
        if not payload["predecessor"]["scientific_leaves_equal"]:
            raise ValueError("observer scientific leaves differ from predecessor")
    record = content_record(payload)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--result")
    parser.add_argument("--predecessor")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    run_root = Path(arguments.run_root).resolve()
    root_record, root_manifest, _analysis_manifest, _events, latest_result = (
        load_latest_lineage(run_root)
    )
    result_path = Path(arguments.result).resolve() if arguments.result else latest_result
    if result_path != latest_result:
        raise ValueError("observer analysis must consume the latest complete lineage")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    verify_resolved_inputs(run_root, root_record, root_manifest, result)
    verify_analysis_sources(result)
    commit, sources = committed_identity()
    record = analyze(
        run_root,
        result_path,
        code_commit=commit,
        analysis_sources=sources,
        predecessor_path=(
            Path(arguments.predecessor).resolve() if arguments.predecessor else None
        ),
    )
    write_new_json(Path(arguments.output), record)
    print(json.dumps({
        "status": "complete",
        "record_sha256": record["record_sha256"],
        "represented_zero_state_count": record["represented_arithmetic"]["represented_zero_state_count"],
        "all_peaks_below_observer_floor": record["represented_arithmetic"]["face_excursions"]["all_peaks_below_observer_floor"],
        "required_holdout_range": [
            record["power_summary"]["minimum_required_holdout_updates"],
            record["power_summary"]["maximum_required_holdout_updates"],
        ],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
