#!/usr/bin/env python3
"""Independently analyze the staged predictive-state closure experiment."""

from __future__ import annotations
from companion_paths import configured_path

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from row_rgmap.analysis import file_sha256, seal_record, verify_record_seal  # noqa: E402
from row_rgmap.provenance_v3 import git_blob_descriptor  # noqa: E402
from row_rgmap.staged_closure import (  # noqa: E402
    candidate_signatures,
    canonical_transition_state_sha256,
)
from row_rgmap.staged_validation import (  # noqa: E402
    RESULT_SCHEMA,
    derive_scientific_payload,
    validate_staged_result,
)
from scripts.analyze_energy_closure_probe_v3 import (  # noqa: E402
    load_energy,
    verify_output_checkpoint_inventory,
)
from scripts.run_energy_closure_probe_v3 import (  # noqa: E402
    descriptor,
    differing_paths,
    is_sign_gauge_parameter,
    resolve_data_root_locator,
    sign_gauge_layer,
    vector_defect,
    write_new_json,
)
from scripts.run_staged_predictive_closure import (  # noqa: E402
    BRANCH_IDS,
    CANDIDATE_IDS,
    CHECKPOINT_SCHEMA,
    CONFIG_SCHEMA,
    MANIFEST_SCHEMA,
    PRODUCER_SCHEMA,
    validate_config,
)


DEFECT_KEYS = (
    "map_count",
    "bitwise_different_map_count",
    "maximum_absolute_energy_defect",
    "maximum_symmetric_relative_energy_defect",
)


def resolve_descriptor(root: Path, item: Any) -> Path:
    if not isinstance(item, dict):
        raise ValueError("artifact descriptor is not an object")
    relative = item.get("path")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ValueError("artifact descriptor path is unsafe")
    resolved_root = root.resolve()
    path = (resolved_root / relative).resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("artifact descriptor escapes its root") from error
    if (
        not path.is_file()
        or path.stat().st_size != item.get("size_bytes")
        or file_sha256(path) != item.get("sha256")
    ):
        raise ValueError(f"artifact descriptor fails: {relative}")
    return path


def resolve_anchored_descriptor(data_root: Path, item: Any) -> Path:
    if not isinstance(item, dict) or item.get("anchor") != "data_root":
        raise ValueError("external descriptor is not data-root anchored")
    return resolve_descriptor(
        data_root, {key: value for key, value in item.items() if key != "anchor"}
    )


def verify_sources(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    commit = manifest.get("code_commit")
    sources = manifest.get("analysis_sources")
    if not isinstance(commit, str) or not isinstance(sources, list) or not sources:
        raise ValueError("staged source registry is incomplete")
    for source in sources:
        repository_path = source.get("repository_path")
        if not isinstance(repository_path, str):
            raise ValueError("staged source path is invalid")
        digest, size = git_blob_descriptor(ROOT, commit, repository_path)
        if (
            source.get("code_commit") != commit
            or source.get("sha256") != digest
            or source.get("size_bytes") != size
        ):
            raise ValueError(f"staged source mismatch: {repository_path}")
    return sources


def load_measurements(path: Path) -> dict[str, np.ndarray]:
    expected = {
        "trajectory_ids",
        "branch_ids",
        "map_ids",
        "restored_recorder_energy",
        "rewritten_branch_remeasured_energy",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise ValueError("staged source-measurement archive contract changed")
        result = {key: np.asarray(archive[key]) for key in expected}
    for key in ("trajectory_ids", "branch_ids", "map_ids"):
        result[key] = result[key].astype(str)
    for key in ("restored_recorder_energy", "rewritten_branch_remeasured_energy"):
        result[key] = np.asarray(result[key], dtype=np.float64)
        if not np.all(np.isfinite(result[key])) or np.any(result[key] < 0):
            raise ValueError("staged source energies are invalid")
    return result


def output_checkpoint(
    run_root: Path, row: dict[str, Any], step: int
) -> Path:
    matches = [item for item in row["output_checkpoints"] if item["step"] == step]
    if len(matches) != 1:
        raise ValueError(f"checkpoint step {step} is not unique")
    return resolve_descriptor(run_root, matches[0])


def source_checkpoint(
    source_root: Path,
    manifest: dict[str, Any],
    trajectory_id: str,
    step: int,
) -> Path:
    matches = [
        item
        for item in manifest["source_run"]["checkpoints"]
        if item["trajectory_id"] == trajectory_id and item["step"] == step
    ]
    if len(matches) != 1:
        raise ValueError("source checkpoint descriptor is not unique")
    item = matches[0]
    return resolve_descriptor(
        source_root,
        {
            "path": item["path"],
            "sha256": item["sha256"],
            "size_bytes": item["size_bytes"],
        },
    )


def source_energy_archive(source_root: Path, trajectory_id: str) -> Path:
    metadata_path = source_root / "trajectories" / trajectory_id / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    relative = Path(metadata["artifacts"]["energies"]["path"])
    path = (metadata_path.parent / relative).resolve()
    expected = metadata["artifacts"]["energies"]
    if (
        not path.is_file()
        or file_sha256(path) != expected["sha256"]
        or path.stat().st_size != expected["size_bytes"]
    ):
        raise ValueError("source energy archive descriptor fails")
    return path


def phase_at_step(phase: dict[str, Any], step: int) -> dict[str, Any]:
    return dict(phase) | {"transition_state_step": step}


def analyze(
    run_root: Path,
    data_root: Path,
    *,
    manifest: dict[str, Any] | None = None,
    refinedweb_root: Path = Path(configured_path('assets:refinedweb')),
) -> dict[str, Any]:
    run_root = run_root.resolve()
    data_root = data_root.resolve()
    try:
        run_root.relative_to(data_root)
    except ValueError as error:
        raise ValueError("staged run root must be inside the data root") from error
    manifest_path = run_root / "run-manifest.json"
    if manifest is None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_record_seal(manifest)
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("staged run-manifest schema differs")
    sources = verify_sources(manifest)
    config_path = resolve_descriptor(run_root, manifest["inputs"]["config"])
    resolve_descriptor(run_root, manifest["inputs"]["protocol"])
    measurement_path = resolve_descriptor(
        run_root, manifest["inputs"]["source_remeasurement"]
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    if (
        config["block_sizes"] != manifest["block_sizes"]
        or config["splits"] != manifest["splits"]
        or config["candidate_states"] != manifest["candidate_states"]
        or config["branch_ids"] != manifest["branch_ids"]
    ):
        raise ValueError("frozen config and run manifest differ")
    source_root = resolve_data_root_locator(data_root, config["source_run"])
    source_config_path = source_root / "config-resolved.json"
    source_protocol_path = source_root / "protocol-resolved.json"
    if (
        file_sha256(source_config_path) != manifest["source_run"]["config_sha256"]
        or file_sha256(source_protocol_path)
        != manifest["source_run"]["protocol_sha256"]
    ):
        raise ValueError("source run identity changed")
    predecessor_manifest_path = resolve_anchored_descriptor(
        data_root, manifest["predecessor"]["run_manifest"]
    )
    predecessor_result_path = resolve_anchored_descriptor(
        data_root, manifest["predecessor"]["result"]
    )
    for path, recorded in (
        (predecessor_manifest_path, manifest["predecessor"]["run_manifest"]),
        (predecessor_result_path, manifest["predecessor"]["result"]),
    ):
        record = json.loads(path.read_text(encoding="utf-8"))
        verify_record_seal(record)
        if record["record_sha256"] != recorded["record_sha256"]:
            raise ValueError("predecessor canonical digest changed")

    measurements = load_measurements(measurement_path)
    trajectory_ids = [row["trajectory_id"] for row in config["trajectories"]]
    if (
        measurements["trajectory_ids"].tolist() != trajectory_ids
        or measurements["branch_ids"].tolist() != list(BRANCH_IDS)
        or measurements["restored_recorder_energy"].shape
        != (len(trajectory_ids), len(measurements["map_ids"]))
        or measurements["rewritten_branch_remeasured_energy"].shape
        != (len(trajectory_ids), len(BRANCH_IDS), len(measurements["map_ids"]))
    ):
        raise ValueError("staged source-measurement array shapes differ")

    import torch

    source_record_by_id = {
        row["trajectory_id"]: row
        for row in manifest["source_remeasurement"]["trajectories"]
    }
    expected_steps = {
        0,
        int(manifest["middle_step"]),
        int(manifest["final_step"]),
    }
    source_step = int(manifest["source_step"])
    baseline_replay_rows = []
    gauge_rows = []
    semigroup_rows = []
    candidate_rows: dict[str, list[dict[str, Any]]] = {
        candidate: [] for candidate in CANDIDATE_IDS
    }
    all_checkpoint_count = 0
    all_checkpoint_bytes = 0
    all_rewrites_exact = True

    for trajectory_index, trajectory_id in enumerate(trajectory_ids):
        source_record = source_record_by_id.get(trajectory_id)
        if source_record is None:
            raise ValueError("source-remeasurement trajectory is missing")
        phase = source_record["phase"]
        source_path = source_checkpoint(
            source_root, manifest, trajectory_id, source_step
        )
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        original_signatures = candidate_signatures(
            source,
            measurements["restored_recorder_energy"][trajectory_index],
            source_record["original_route"]["row_map_sha256"],
            phase=phase,
            is_gauge_parameter=is_sign_gauge_parameter,
            gauge_layer=sign_gauge_layer,
        )
        if original_signatures != source_record["original_candidate_signatures"]:
            raise ValueError("original candidate signatures do not replay")
        branches = {row["branch"]: row for row in manifest["branches"][trajectory_id]}
        if set(branches) != set(BRANCH_IDS):
            raise ValueError("staged branch registry is incomplete")
        source_branch_records = {
            row["branch"]: row for row in source_record["branches"]
        }
        loaded: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        checkpoints: dict[str, dict[str, Any]] = {}
        for branch_index, branch in enumerate(BRANCH_IDS):
            row = branches[branch]
            metadata_path = resolve_descriptor(run_root, row["metadata"])
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata.get("schema_version") != PRODUCER_SCHEMA
                or metadata.get("code_commit") != manifest["code_commit"]
                or metadata.get("trajectory_id") != trajectory_id
            ):
                raise ValueError("branch producer metadata differs")
            count, size = verify_output_checkpoint_inventory(
                run_root,
                row,
                metadata,
                trajectory_id,
                branch,
                expected_steps,
            )
            all_checkpoint_count += count
            all_checkpoint_bytes += size
            energy_path = resolve_descriptor(run_root, row["energies"])
            resolve_descriptor(run_root, row["metrics"])
            resolve_descriptor(run_root, row["producer_log"])
            resume_path = resolve_descriptor(run_root, row["resume_checkpoint"])
            checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
            if (
                checkpoint.get("schema_version") != CHECKPOINT_SCHEMA
                or checkpoint.get("step") != source_step
                or checkpoint.get("code_commit") != manifest["code_commit"]
            ):
                raise ValueError("branch resume checkpoint identity differs")
            measured = measurements["rewritten_branch_remeasured_energy"][
                trajectory_index, branch_index
            ]
            route = source_branch_records[branch]["route"]
            signatures = candidate_signatures(
                checkpoint,
                measured,
                route["row_map_sha256"],
                phase=phase,
                is_gauge_parameter=is_sign_gauge_parameter,
                gauge_layer=sign_gauge_layer,
            )
            transition = canonical_transition_state_sha256(
                checkpoint,
                phase=phase,
                is_gauge_parameter=is_sign_gauge_parameter,
                gauge_layer=sign_gauge_layer,
            )
            if (
                signatures != row["source_candidate_signatures"]
                or signatures != source_branch_records[branch]["candidate_signatures"]
                or transition != row["source_canonical_transition_state_sha256"]
                or transition
                != source_branch_records[branch][
                    "canonical_transition_state_sha256"
                ]
            ):
                raise ValueError("branch candidate signature does not replay")
            actual_paths = sorted(differing_paths(source, checkpoint))
            rewrite_exact = bool(
                actual_paths == row["checkpoint_rewrite"]["differing_leaf_paths"]
                == row["checkpoint_rewrite"]["expected_leaf_paths"]
            )
            all_rewrites_exact &= rewrite_exact
            if not rewrite_exact:
                raise ValueError("branch checkpoint rewrite does not replay")
            loaded[branch] = load_energy(energy_path)
            checkpoints[branch] = checkpoint

        baseline_energy, baseline_steps, baseline_maps = loaded["baseline-direct"]
        original_energy, original_steps, original_maps = load_energy(
            source_energy_archive(source_root, trajectory_id)
        )
        replay_equal = bool(
            np.array_equal(baseline_steps, original_steps)
            and np.array_equal(baseline_maps, original_maps)
            and np.array_equal(baseline_energy, original_energy)
        )
        baseline_replay_rows.append(
            {
                "trajectory_id": trajectory_id,
                "split": source_record["split"],
                "all_steps_bitwise_equal": replay_equal,
                "state_count": int(len(baseline_steps)),
                "map_count": int(len(baseline_maps)),
            }
        )
        if not replay_equal:
            raise ValueError("baseline continuation does not replay source run")
        source_indices = np.flatnonzero(baseline_steps == source_step)
        if len(source_indices) != 1:
            raise ValueError("source step is absent from continuation")
        source_index = int(source_indices[0])
        for branch in BRANCH_IDS:
            energy, steps, maps = loaded[branch]
            if (
                not np.array_equal(steps, baseline_steps)
                or not np.array_equal(maps, baseline_maps)
                or not np.array_equal(energy[source_index], measurements[
                    "restored_recorder_energy"
                ][trajectory_index])
            ):
                raise ValueError("branch archive source or registry changed")

        challenge_by_candidate = {
            row["candidate_id"]: row["challenge_branch"]
            for row in config["candidate_states"]
        }
        for candidate in CANDIDATE_IDS:
            challenge = challenge_by_candidate[candidate]
            source_equal = bool(
                branches["baseline-direct"]["source_candidate_signatures"][candidate]
                == branches[challenge]["source_candidate_signatures"][candidate]
            )
            scales = []
            for block_size in manifest["block_sizes"]:
                target = source_step + int(block_size)
                matches = np.flatnonzero(baseline_steps == target)
                if len(matches) != 1:
                    raise ValueError(f"target step {target} is absent")
                index = int(matches[0])
                defect = vector_defect(
                    baseline_energy[index], loaded[challenge][0][index]
                )
                scales.append(
                    {
                        "block_size": int(block_size),
                        **defect,
                        "conditional_equality_prediction_rejected": bool(
                            source_equal and defect["bitwise_different_map_count"] > 0
                        ),
                    }
                )
            candidate_rows[candidate].append(
                {
                    "trajectory_id": trajectory_id,
                    "split": source_record["split"],
                    "challenge_branch": challenge,
                    "source_candidate_bitwise_equal": source_equal,
                    "scales": scales,
                }
            )

        gauge_source_equal = bool(
            branches["baseline-direct"]["source_canonical_transition_state_sha256"]
            == branches["gauge-consistent"][
                "source_canonical_transition_state_sha256"
            ]
        )
        gauge_scales = []
        for block_size in manifest["block_sizes"]:
            target = source_step + int(block_size)
            index = int(np.flatnonzero(baseline_steps == target)[0])
            gauge_scales.append(
                {
                    "block_size": int(block_size),
                    **vector_defect(
                        baseline_energy[index],
                        loaded["gauge-consistent"][0][index],
                    ),
                }
            )
        baseline_final = torch.load(
            output_checkpoint(run_root, branches["baseline-direct"], manifest["final_step"]),
            map_location="cpu",
            weights_only=False,
        )
        gauge_final = torch.load(
            output_checkpoint(run_root, branches["gauge-consistent"], manifest["final_step"]),
            map_location="cpu",
            weights_only=False,
        )
        final_phase = phase_at_step(phase, int(manifest["final_step"]))
        final_quotient_equal = bool(
            canonical_transition_state_sha256(
                baseline_final,
                phase=final_phase,
                is_gauge_parameter=is_sign_gauge_parameter,
                gauge_layer=sign_gauge_layer,
            )
            == canonical_transition_state_sha256(
                gauge_final,
                phase=final_phase,
                is_gauge_parameter=is_sign_gauge_parameter,
                gauge_layer=sign_gauge_layer,
            )
        )
        gauge_rows.append(
            {
                "trajectory_id": trajectory_id,
                "split": source_record["split"],
                "source_canonical_transition_state_equal": gauge_source_equal,
                "successor_scales": gauge_scales,
                "final_canonical_transition_state_equal": final_quotient_equal,
            }
        )

        semigroup_list = manifest["semigroup_branches"][trajectory_id]
        if len(semigroup_list) != 1:
            raise ValueError("semigroup branch is not unique")
        semigroup = semigroup_list[0]
        semigroup_metadata_path = resolve_descriptor(run_root, semigroup["metadata"])
        semigroup_metadata = json.loads(
            semigroup_metadata_path.read_text(encoding="utf-8")
        )
        count, size = verify_output_checkpoint_inventory(
            run_root,
            semigroup,
            semigroup_metadata,
            trajectory_id,
            "baseline-iterated",
            expected_steps,
        )
        all_checkpoint_count += count
        all_checkpoint_bytes += size
        iterated_energy, iterated_steps, iterated_maps = load_energy(
            resolve_descriptor(run_root, semigroup["energies"])
        )
        resolve_descriptor(run_root, semigroup["metrics"])
        resolve_descriptor(run_root, semigroup["producer_log"])
        resolve_descriptor(run_root, semigroup["resume_checkpoint"])
        middle = int(manifest["middle_step"])
        tail = baseline_steps >= middle
        tail_equal = bool(
            np.array_equal(iterated_steps, baseline_steps)
            and np.array_equal(iterated_maps, baseline_maps)
            and np.array_equal(iterated_energy[tail], baseline_energy[tail])
        )
        iterated_final = torch.load(
            output_checkpoint(run_root, semigroup, manifest["final_step"]),
            map_location="cpu",
            weights_only=False,
        )
        final_state_equal = bool(
            canonical_transition_state_sha256(
                baseline_final,
                phase=final_phase,
                is_gauge_parameter=is_sign_gauge_parameter,
                gauge_layer=sign_gauge_layer,
            )
            == canonical_transition_state_sha256(
                iterated_final,
                phase=final_phase,
                is_gauge_parameter=is_sign_gauge_parameter,
                gauge_layer=sign_gauge_layer,
            )
        )
        semigroup_rows.append(
            {
                "trajectory_id": trajectory_id,
                "split": source_record["split"],
                "restart_step": middle,
                "tail_energy_arrays_bitwise_equal": tail_equal,
                "final_canonical_transition_state_equal": final_state_equal,
            }
        )

    expected_checkpoint_count = (
        (len(BRANCH_IDS) + 1) * len(trajectory_ids) * len(expected_steps)
    )
    checkpoint_inventory_valid = bool(
        all_checkpoint_count == expected_checkpoint_count
        and all_checkpoint_count
        == manifest["checkpoint_inventory"]["descriptor_count"]
        and all_checkpoint_bytes
        == manifest["checkpoint_inventory"]["total_size_bytes"]
        and manifest["checkpoint_inventory"]["all_emitted_files_bound"] is True
    )
    baseline_passed = all(row["all_steps_bitwise_equal"] for row in baseline_replay_rows)
    gauge_passed = all(
        row["source_canonical_transition_state_equal"]
        and row["final_canonical_transition_state_equal"]
        and all(
            scale["bitwise_different_map_count"] == 0
            for scale in row["successor_scales"]
        )
        for row in gauge_rows
    )
    semigroup_passed = all(
        row["tail_energy_arrays_bitwise_equal"]
        and row["final_canonical_transition_state_equal"]
        for row in semigroup_rows
    )

    structural_checks = {
        "manifest_seal": True,
        "source_blobs_at_execution_commit": True,
        "frozen_config_matches_manifest": True,
        "source_and_predecessor_descriptors": True,
        "checkpoint_rewrites_exact": all_rewrites_exact,
        "checkpoint_inventory_exact": checkpoint_inventory_valid,
        "baseline_replay": baseline_passed,
        "gauge_quotient_control": gauge_passed,
        "direct_vs_iterated_semigroup": semigroup_passed,
        "gpu_cap": manifest["gpu_time_accounting"]["cap_passed"] is True,
    }
    structural_valid = all(structural_checks.values())
    if not structural_valid:
        raise ValueError("staged closure structural validation failed")
    scientific_payload = derive_scientific_payload(
        config,
        candidate_rows,
        baseline_replay_rows,
        gauge_rows,
        semigroup_rows,
    )
    payload = {
        "schema_version": RESULT_SCHEMA,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": manifest["code_commit"],
        "analysis_sources": sources,
        "input_run_manifest": {
            "sha256": file_sha256(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
            "record_sha256": manifest["record_sha256"],
        },
        "stage_id": manifest["stage_id"],
        "gpu_time_accounting": manifest["gpu_time_accounting"],
        "gpu_seconds": manifest["gpu_seconds"],
        "gpu_hours": manifest["gpu_hours"],
        "validity": {
            "structural_valid": structural_valid,
            "structural_checks": structural_checks,
        },
        "scientific_payload": scientific_payload,
    }
    result = seal_record(payload)
    validate_staged_result(
        result,
        manifest,
        config,
        run_root=run_root,
        data_root=data_root,
        repository_root=ROOT,
        refinedweb_root=refinedweb_root,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--refinedweb-root", default=configured_path('assets:refinedweb'))
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    output = Path(arguments.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    result = analyze(
        Path(arguments.run_root),
        Path(arguments.data_root),
        refinedweb_root=Path(arguments.refinedweb_root),
    )
    write_new_json(output, result)
    print(
        json.dumps(
            {
                "status": result["scientific_payload"]["promotion"]["stage_status"],
                "record_sha256": result["record_sha256"],
                "promoted_candidate_ids": result["scientific_payload"]["promotion"][
                    "promoted_candidate_ids"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
