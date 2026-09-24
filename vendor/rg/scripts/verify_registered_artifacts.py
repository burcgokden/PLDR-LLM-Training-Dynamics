#!/usr/bin/env python3
"""Independently reconstruct registered counts from a completed v3 run."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from row_rgmap.analysis import (  # noqa: E402
    canonical_json_bytes,
    file_sha256,
    verify_record_seal,
)
from row_rgmap.provenance_v3 import (  # noqa: E402
    git_blob_descriptor,
    repository_relative_source,
)


def verify_record(record: dict[str, Any]) -> None:
    verify_record_seal(record)


def resolve_artifact(
    run_root: Path, descriptor: dict[str, Any], label: str
) -> Path:
    path = (run_root / str(descriptor.get("path", ""))).resolve()
    try:
        path.relative_to(run_root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the run root") from error
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    if (
        descriptor.get("size_bytes") is not None
        and path.stat().st_size != descriptor.get("size_bytes")
    ):
        raise ValueError(f"{label} size mismatch")
    if file_sha256(path) != descriptor.get("sha256"):
        raise ValueError(f"{label} digest mismatch")
    return path


def event_rows(path: Path) -> list[dict[str, Any]]:
    """Replay an append-only event chain with the shared canonical encoding."""

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    previous = None
    for index, row in enumerate(rows):
        observed = row.get("event_sha256")
        unsigned = dict(row)
        unsigned.pop("event_sha256", None)
        expected = hashlib.sha256(
            json.dumps(
                unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()
        if row.get("sequence") != index or row.get("previous_event_sha256") != previous:
            raise ValueError("event log sequence or parent digest is invalid")
        if observed != expected:
            raise ValueError("event log integrity digest does not replay")
        previous = observed
    return rows


def _accepted_root_manifest_schema(value: Any) -> bool:
    return value in {
        "pldr-row-rg-root-manifest-v3",
        "pldr-row-rg-root-manifest-v3.1",
    }


def latest_root_manifest_path(run_root: Path) -> Path:
    """Return the highest contiguous, complete, digest-valid root manifest."""

    candidates: dict[int, Path] = {}
    for path in (run_root / "manifests").glob("manifest-*.json"):
        match = re.fullmatch(r"manifest-([0-9]{4})[.]json", path.name)
        if match:
            candidates[int(match.group(1))] = path
    if not candidates:
        raise ValueError("no root manifest was found")
    expected = set(range(1, max(candidates) + 1))
    if set(candidates) != expected:
        missing = sorted(expected - set(candidates))
        raise ValueError(f"root manifest sequence is not contiguous: {missing}")
    for index in sorted(candidates):
        path = candidates[index]
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            verify_record(manifest)
            if not _accepted_root_manifest_schema(manifest.get("schema_version")):
                raise ValueError(f"unexpected root manifest schema in {path.name}")
            if manifest.get("manifest_id") != path.stem:
                raise ValueError(f"root manifest identifier mismatch in {path.name}")
            analysis = manifest.get("analysis_manifest", {})
            analysis_path = resolve_artifact(
                run_root, analysis, "analysis manifest"
            )
            analysis_record = json.loads(analysis_path.read_text(encoding="utf-8"))
            verify_record(analysis_record)
            if analysis_record.get("record_sha256") != analysis.get("record_sha256"):
                raise ValueError(f"analysis record linkage mismatch in {path.name}")
            result_path = resolve_artifact(
                run_root, analysis_record.get("result", {}), "analysis result"
            )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            verify_record(result)
            if result.get("record_sha256") != analysis_record["result"].get(
                "record_sha256"
            ):
                raise ValueError(f"result linkage mismatch in {path.name}")
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid root manifest candidate {path.name}: {error}") from error
    return candidates[max(candidates)]


def root_manifest_chain(
    run_root: Path, selected_path: Path
) -> list[dict[str, Any]]:
    """Replay every sealed parent link through the selected root manifest."""

    match = re.fullmatch(r"manifest-([0-9]{4})[.]json", selected_path.name)
    if match is None or selected_path.parent.resolve() != (
        run_root / "manifests"
    ).resolve():
        raise ValueError("selected root manifest path is invalid")
    selected_index = int(match.group(1))
    chain: list[dict[str, Any]] = []
    previous_digest = None
    for index in range(1, selected_index + 1):
        path = selected_path.parent / f"manifest-{index:04d}.json"
        if not path.is_file():
            raise ValueError(f"root manifest chain is missing {path.name}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        verify_record(manifest)
        schema = manifest.get("schema_version")
        if not _accepted_root_manifest_schema(schema):
            raise ValueError(f"unexpected root manifest schema in {path.name}")
        if manifest.get("manifest_id") != path.stem:
            raise ValueError(f"root manifest identifier mismatch in {path.name}")
        if (
            schema == "pldr-row-rg-root-manifest-v3.1"
            and "previous_manifest_record_sha256" not in manifest
        ):
            raise ValueError(f"root manifest parent field missing in {path.name}")
        if manifest.get("previous_manifest_record_sha256") != previous_digest:
            raise ValueError(f"root manifest parent digest mismatch in {path.name}")
        chain.append(manifest)
        previous_digest = manifest["record_sha256"]
    return chain


def load_latest_lineage(
    run_root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    Path,
]:
    root_record = json.loads((run_root / "run-root.json").read_text(encoding="utf-8"))
    verify_record(root_record)
    manifest_path = latest_root_manifest_path(run_root)
    manifest = root_manifest_chain(run_root, manifest_path)[-1]
    analysis_path = resolve_artifact(
        run_root, manifest["analysis_manifest"], "analysis manifest"
    )
    analysis_manifest = json.loads(analysis_path.read_text(encoding="utf-8"))
    result_path = resolve_artifact(
        run_root, analysis_manifest["result"], "analysis result"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    events_path = resolve_artifact(run_root, manifest["event_log"], "event log")
    events = event_rows(events_path)
    if manifest["event_log"].get("event_count") != len(events):
        raise ValueError("root manifest event count mismatch")
    if (
        root_record.get("record_sha256") != manifest.get("run_root_record_sha256")
        or root_record.get("record_sha256") != analysis_manifest.get("run_root_record_sha256")
        or root_record.get("record_sha256") != result.get("run_root_record_sha256")
    ):
        raise ValueError("run-root digest linkage mismatch")
    initialized = [row for row in events if row.get("event") == "RUN_INITIALIZED"]
    if (
        len(initialized) != 1
        or initialized[0].get("payload", {}).get("root_record_sha256")
        != root_record.get("record_sha256")
    ):
        raise ValueError("RUN_INITIALIZED does not bind the run root")
    completed = [
        row
        for row in events
        if row.get("event") == "ANALYSIS_COMPLETED"
        and row.get("payload", {}).get("analysis_id") == result.get("analysis_id")
    ]
    if (
        len(completed) != 1
        or completed[0]["payload"].get("result_record_sha256")
        != result.get("record_sha256")
        or completed[0]["payload"].get("result_file_sha256")
        != file_sha256(result_path)
    ):
        raise ValueError("ANALYSIS_COMPLETED does not bind the selected result")
    return root_record, manifest, analysis_manifest, events, result_path


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def verify_resolved_inputs(
    run_root: Path,
    root_record: dict[str, Any],
    root_manifest: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, str]:
    """Verify the three frozen input copies against their recorded identities."""

    config_path = run_root / "config-resolved.json"
    protocol_path = run_root / "protocol-resolved.json"
    qualification_path = run_root / "qualification-resolved.json"
    for label, path in (
        ("configuration", config_path),
        ("protocol", protocol_path),
        ("qualification", qualification_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"resolved {label} copy is missing: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    verify_record(qualification)
    if canonical_digest(config) != root_record.get("config", {}).get("canonical_sha256"):
        raise ValueError("resolved configuration canonical digest mismatch")
    if canonical_digest(protocol) != root_record.get("protocol", {}).get("canonical_sha256"):
        raise ValueError("resolved protocol canonical digest mismatch")
    if canonical_digest(protocol) != result.get("protocol_sha256"):
        raise ValueError("analysis protocol identity mismatch")
    if qualification.get("record_sha256") != result.get(
        "qualification_record_sha256"
    ):
        raise ValueError("resolved qualification record identity mismatch")
    registered = root_manifest.get("resolved_inputs")
    if registered is not None:
        expected_paths = {
            "configuration": config_path,
            "protocol": protocol_path,
            "qualification": qualification_path,
        }
        if set(registered) != set(expected_paths):
            raise ValueError("root manifest resolved-input registry is incomplete")
        for label, expected_path in expected_paths.items():
            observed_path = resolve_artifact(
                run_root, registered[label], f"resolved {label}"
            )
            if observed_path != expected_path.resolve():
                raise ValueError(f"resolved {label} path mismatch")
        if (
            registered["configuration"].get("canonical_sha256")
            != canonical_digest(config)
            or registered["protocol"].get("canonical_sha256")
            != canonical_digest(protocol)
            or registered["qualification"].get("record_sha256")
            != qualification["record_sha256"]
        ):
            raise ValueError("root manifest resolved-input identity mismatch")
    return {
        "config_file_sha256": file_sha256(config_path),
        "protocol_file_sha256": file_sha256(protocol_path),
        "qualification_file_sha256": file_sha256(qualification_path),
    }


def verify_analysis_sources(result: dict[str, Any]) -> int:
    """Check every analyzer source as a Git blob at its recorded commit."""

    commit = result.get("analyzer_code_commit", result.get("code_commit"))
    rows = result.get("analysis_sources")
    if not isinstance(commit, str) or not isinstance(rows, list) or not rows:
        raise ValueError("analysis source provenance is incomplete")
    for row in rows:
        raw = row.get("repository_path", row.get("path", ""))
        if not isinstance(raw, str):
            raise ValueError("analysis source path is invalid")
        repository_path = row.get("repository_path")
        if not isinstance(repository_path, str) or not repository_path:
            repository_path = repository_relative_source(raw, REPOSITORY_ROOT)
        if not repository_path:
            raise ValueError("analysis source is not repository-relative")
        digest, size = git_blob_descriptor(REPOSITORY_ROOT, commit, repository_path)
        if row.get("sha256") != digest or row.get("size_bytes") != size:
            raise ValueError(f"analysis source mismatch: {repository_path}")
    return len(rows)


def verify_analysis_source_registry(
    result: dict[str, Any], analysis_manifest: dict[str, Any]
) -> int:
    """Verify both stored source registries and require them to be identical."""

    result_rows = result.get("analysis_sources")
    manifest_rows = analysis_manifest.get("analysis_sources")
    if manifest_rows != result_rows:
        raise ValueError("analysis manifest source registry differs from the result")
    manifest_view = {
        "analyzer_code_commit": analysis_manifest.get("analyzer_code_commit"),
        "analysis_sources": manifest_rows,
    }
    manifest_count = verify_analysis_sources(manifest_view)
    result_count = verify_analysis_sources(result)
    if manifest_count != result_count:
        raise ValueError("analysis source registry counts differ")
    return result_count


def close(left: float, right: float, tolerance: float = 2e-15) -> bool:
    return abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))


def verify(run_root: Path) -> dict[str, Any]:
    root_record, root_manifest, analysis_manifest, events, result_path = (
        load_latest_lineage(run_root)
    )
    record = json.loads(result_path.read_text(encoding="utf-8"))
    verify_record(record)
    if record.get("schema_version") not in {
        "pldr-row-rg-confirmation-result-v4",
        "pldr-row-rg-confirmation-result-v5",
    }:
        raise ValueError("unexpected result schema")
    resolved_inputs = verify_resolved_inputs(
        run_root, root_record, root_manifest, record
    )
    analysis_source_count = verify_analysis_source_registry(
        record, analysis_manifest
    )

    combined_path = resolve_artifact(
        run_root, record["combined_energy_artifact"], "combined energy artifact"
    )
    with np.load(combined_path, allow_pickle=False) as source:
        if set(source.files) != {"energies", "steps", "trajectory_ids", "map_ids"}:
            raise ValueError("combined NPZ key contract mismatch")
        energies = np.asarray(source["energies"])
        steps = np.asarray(source["steps"])
        trajectory_ids = np.asarray(source["trajectory_ids"]).astype(str)
        map_ids = np.asarray(source["map_ids"]).astype(str)

    exact = record["exact_rg_analysis"]
    decision = record["registered_decision"]
    segments = decision["segments"]
    expected_updates = sum(int(value) for value in segments.values())
    expected_shape = (
        int(exact["shape"]["trajectories"]),
        expected_updates + 1,
        int(exact["shape"]["maps"]),
    )
    if energies.shape != expected_shape:
        raise ValueError(f"energy shape mismatch: {energies.shape} != {expected_shape}")
    if not np.array_equal(steps, np.arange(expected_updates + 1)):
        raise ValueError("step grid is not consecutive")
    if len(set(trajectory_ids)) != len(trajectory_ids):
        raise ValueError("trajectory identifiers are not unique")
    if len(set(map_ids)) != len(map_ids):
        raise ValueError("map identifiers are not unique")
    if not np.isfinite(energies).all() or np.any(energies < 0.0):
        raise ValueError("energies are not finite and nonnegative")

    producer_by_id = {
        str(row["trajectory_id"]): row for row in record["producer_artifacts"]
    }
    for trajectory_index, trajectory_id in enumerate(trajectory_ids):
        descriptor = producer_by_id[trajectory_id]
        energy_path = resolve_artifact(
            run_root, descriptor["energies"], f"{trajectory_id} energies"
        )
        resolve_artifact(run_root, descriptor["metrics"], f"{trajectory_id} metrics")
        metadata_path = resolve_artifact(
            run_root, descriptor["metadata"], f"{trajectory_id} metadata"
        )
        for checkpoint in descriptor.get("checkpoints", []):
            resolve_artifact(
                run_root,
                checkpoint,
                f"{trajectory_id} checkpoint {checkpoint.get('step')}",
            )
        metadata_digest = file_sha256(metadata_path)
        producer_events = [
            row
            for row in events
            if row.get("event") == "PRODUCER_COMPLETED"
            and row.get("payload", {}).get("trajectory_id") == trajectory_id
        ]
        matching_completion_events = [
            row for row in producer_events
            if row["payload"].get("metadata_sha256") == metadata_digest
            and row["payload"].get("energy_sha256")
            == descriptor["energies"]["sha256"]
            and row["payload"].get("metrics_sha256")
            == descriptor["metrics"]["sha256"]
        ]
        if not matching_completion_events:
            raise ValueError(f"{trajectory_id} completion event linkage mismatch")
        with np.load(energy_path, allow_pickle=False) as source:
            if not np.array_equal(source["energies"][0], energies[trajectory_index]):
                raise ValueError(f"{trajectory_id} energies differ from combined NPZ")
            if not np.array_equal(source["steps"], steps):
                raise ValueError(f"{trajectory_id} step grid differs")
            if not np.array_equal(source["map_ids"].astype(str), map_ids):
                raise ValueError(f"{trajectory_id} map registry differs")

    face = energies == 0.0
    face_states = int(np.count_nonzero(face))
    closures = int(
        np.count_nonzero((energies[:, :-1, :] > 0.0) & face[:, 1:, :])
    )
    restarts = int(
        np.count_nonzero(face[:, :-1, :] & (energies[:, 1:, :] > 0.0))
    )
    if face_states != int(exact["fine_exact_face_state_count"]):
        raise ValueError("exact-face state count mismatch")
    if closures != int(exact["fine_face_closure_count"]):
        raise ValueError("face-closure count mismatch")
    if restarts != int(exact["fine_face_restart_count"]):
        raise ValueError("face-restart count mismatch")

    block_face_counts: dict[int, int] = {}
    update_count = energies.shape[1] - 1
    for level in exact["levels"]:
        block_size = int(level["block_size"])
        block_count = update_count // block_size
        touched = np.zeros(
            (energies.shape[0], block_count, energies.shape[2]), dtype=bool
        )
        for offset in range(block_size + 1):
            touched |= face[
                :, offset : block_count * block_size + offset : block_size, :
            ]
        count = int(np.count_nonzero(touched))
        if count != int(level["blocks_touching_exact_face"]):
            raise ValueError(f"face-block count mismatch at b={block_size}")
        block_face_counts[block_size] = count

    cell_by_key = {
        (str(row["trajectory_id"]), int(row["layer"])): row
        for row in decision["cells"]
    }
    trajectory_index = {
        trajectory_id: index for index, trajectory_id in enumerate(trajectory_ids)
    }
    burn = int(segments["burn_in_updates"])
    design_end = burn + int(segments["design_updates"])
    complete_cells = 0
    for key, cell in cell_by_key.items():
        trajectory_id, layer = key
        layer_mask = np.asarray([f".L{layer}." in name for name in map_ids])
        cell_energy = energies[trajectory_index[trajectory_id]][:, layer_mask]
        has_face = bool(np.any(cell_energy == 0.0))
        if has_face:
            if cell["flow_classification"] != "FACE_INTERRUPTED":
                raise ValueError(f"{key} contains a face but is not interrupted")
            continue
        if not cell.get("positive_excursion_complete"):
            raise ValueError(f"{key} is positive but not marked complete")
        complete_cells += 1
        log_gain = np.log(cell_energy[1:] / cell_energy[:-1])
        design_mean = float(log_gain[burn:design_end].mean())
        holdout_mean = float(log_gain[design_end:expected_updates].mean())
        if not close(design_mean, float(cell["design_mean_log_gain_per_update"])):
            raise ValueError(f"{key} design mean mismatch")
        if not close(holdout_mean, float(cell["holdout_mean_log_gain_per_update"])):
            raise ValueError(f"{key} holdout mean mismatch")
        map_means = log_gain[design_end:expected_updates].mean(axis=0)
        signs = {
            "negative": int(np.count_nonzero(map_means < 0.0)),
            "zero": int(np.count_nonzero(map_means == 0.0)),
            "positive": int(np.count_nonzero(map_means > 0.0)),
        }
        if signs != cell["holdout_map_sign_counts"]:
            raise ValueError(f"{key} holdout map-sign count mismatch")

    for field, count_field in (
        ("flow_classification", "flow_classification_counts"),
        ("phase_classification", "phase_classification_counts"),
    ):
        reconstructed = Counter(row[field] for row in decision["cells"])
        declared = decision[count_field]
        if sum(declared.values()) != len(decision["cells"]) or any(
            reconstructed[name] != value for name, value in declared.items()
        ):
            raise ValueError(f"{field} aggregate mismatch")

    if complete_cells != sum(
        bool(row.get("positive_excursion_complete")) for row in decision["cells"]
    ):
        raise ValueError("complete-cell count mismatch")
    if float(energies.min()) != float(decision["minimum_energy"]):
        raise ValueError("minimum energy mismatch")

    return {
        "selected_analysis_id": record["analysis_id"],
        "selected_root_manifest_id": root_manifest["manifest_id"],
        "event_chain_replayed": True,
        "event_count": len(events),
        "run_root_record_sha256": root_record["record_sha256"],
        "analysis_manifest_record_sha256": analysis_manifest["record_sha256"],
        "result_record_sha256": record["record_sha256"],
        "energy_shape": list(energies.shape),
        "producer_arrays_equal_combined": True,
        "finite_nonnegative": True,
        "represented_zero_states": face_states,
        "represented_zero_closures": closures,
        "represented_zero_restarts": restarts,
        "block_counts_touching_represented_zero": {
            str(k): v for k, v in block_face_counts.items()
        },
        "complete_cell_means_and_signs_checked": complete_cells,
        "classification_aggregates_checked": True,
        "resolved_inputs_verified": resolved_inputs,
        "analysis_source_count_verified": analysis_source_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    arguments = parser.parse_args()
    summary = verify(Path(arguments.run_root).resolve())
    print(json.dumps(summary, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
