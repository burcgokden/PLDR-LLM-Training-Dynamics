#!/usr/bin/env python3
"""Execute the gated Stage-1 predictive-state closure experiment."""

from __future__ import annotations
from companion_paths import configured_path

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from row_rgmap.analysis import (  # noqa: E402
    canonical_json_bytes,
    file_sha256,
    seal_record,
    verify_record_seal,
)
from row_rgmap.provenance_v3 import git_blob_descriptor  # noqa: E402
from row_rgmap.staged_closure import (  # noqa: E402
    candidate_signatures,
    canonical_transition_state_sha256,
)
from row_rgmap.staged_validation import (  # noqa: E402
    CHECKPOINT_SCHEMA,
    CONFIG_SCHEMA,
    MANIFEST_SCHEMA,
    PRODUCER_SCHEMA,
    STAGED_SOURCE_PATHS,
    TOKEN_SELECTION_RULE,
    validate_config_shape,
)
from scripts.run_energy_closure_probe_v3 import (  # noqa: E402
    _values_equal,
    anchored_descriptor,
    apply_branch_intervention,
    checkpoint_descriptor,
    descriptor,
    differing_paths,
    emitted_checkpoint_descriptors,
    format_command,
    independent_probe_tokens,
    is_sign_gauge_parameter,
    observe_checkpoint_energy,
    observer_context,
    option,
    resolve_data_root_locator,
    restored_energy_vector,
    sign_gauge_layer,
    write_new_json,
)


BRANCH_IDS = (
    "baseline-direct",
    "gauge-consistent",
    "gauge-params-only",
    "probe-null-embedding-shift",
)
CANDIDATE_IDS = (
    "energy-vector",
    "energy-raw-adamw-phase",
    "row-map-gauge-adamw-phase",
)
SOURCE_PATHS = STAGED_SOURCE_PATHS


def committed_identity() -> tuple[str, list[dict[str, Any]]]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("staged closure requires a clean committed repository")
    sources = []
    for repository_path in SOURCE_PATHS:
        digest, size = git_blob_descriptor(ROOT, commit, repository_path)
        worktree = ROOT / repository_path
        if (
            not worktree.is_file()
            or file_sha256(worktree) != digest
            or worktree.stat().st_size != size
        ):
            raise RuntimeError(f"staged source differs from HEAD: {repository_path}")
        sources.append(
            {
                "repository_path": repository_path,
                "code_commit": commit,
                "sha256": digest,
                "size_bytes": size,
            }
        )
    return commit, sources


def validate_config(config: dict[str, Any]) -> None:
    validate_config_shape(config)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unknown staged-closure configuration schema")
    if config.get("block_sizes") != [1, 2, 4, 8, 16]:
        raise ValueError("Stage 1 requires block sizes [1,2,4,8,16]")
    if config.get("source_step") != 16 or config.get("semigroup_split") != 8:
        raise ValueError("Stage 1 requires source 16 and semigroup split 8")
    if config.get("branch_ids") != list(BRANCH_IDS):
        raise ValueError("staged branch registry differs from the frozen program")
    candidate_rows = config.get("candidate_states")
    if (
        not isinstance(candidate_rows, list)
        or [row.get("candidate_id") for row in candidate_rows]
        != list(CANDIDATE_IDS)
        or [row.get("challenge_branch") for row in candidate_rows]
        != [
            "gauge-params-only",
            "gauge-params-only",
            "probe-null-embedding-shift",
        ]
    ):
        raise ValueError("staged candidate registry differs from the frozen program")
    trajectories = config.get("trajectories")
    if (
        not isinstance(trajectories, list)
        or len(trajectories) != 4
        or [row.get("device") for row in trajectories] != [0, 1, 0, 1]
        or len({row.get("trajectory_id") for row in trajectories}) != 4
    ):
        raise ValueError("Stage 1 requires four unique trajectories in two GPU waves")
    registered = {row["trajectory_id"] for row in trajectories}
    splits = config.get("splits")
    if (
        not isinstance(splits, dict)
        or set(splits) != {"design", "holdout"}
        or len(splits["design"]) != 2
        or len(splits["holdout"]) != 2
        or set(splits["design"]).intersection(splits["holdout"])
        or set(splits["design"]).union(splits["holdout"]) != registered
    ):
        raise ValueError("design and holdout split is not a partition")
    shift = config.get("embedding_shift")
    if shift != {
        "model_key": "decoder.embedding.weight",
        "coordinate": 0,
        "offset": 0.125,
        "token_selection_rule": TOKEN_SELECTION_RULE,
    }:
        raise ValueError("source-null embedding intervention is not frozen")
    cap = config.get("gpu_hour_cap")
    if not isinstance(cap, (int, float)) or cap <= 0 or cap > 0.5:
        raise ValueError("Stage-1 GPU-hour cap must be in (0,0.5]")


def phase_record(
    *,
    source_step: int,
    trajectory: dict[str, Any],
    source_command: list[str],
    next_batch_digest: str,
    dataset_digest: str,
) -> dict[str, Any]:
    width = int(option(source_command, "--context-length"))
    batch_size = int(option(source_command, "--batch-size"))
    return {
        "source_step": source_step,
        "trajectory_id": trajectory["trajectory_id"],
        "seed": trajectory["seed"],
        "train_document_offset": trajectory["train_document_offset"],
        "tokens_per_update": batch_size * (width + 1),
        "next_batch_byte_token_sha256": next_batch_digest,
        "dataset_sha256": dataset_digest,
    }


def select_source_null_token(
    dataset: Any,
    context: dict[str, Any],
    source_command: list[str],
    trajectory: dict[str, Any],
    source_step: int,
) -> dict[str, Any]:
    width = int(option(source_command, "--context-length"))
    batch_size = int(option(source_command, "--batch-size"))
    contexts = int(option(source_command, "--contexts"))
    tokens_per_update = batch_size * (width + 1)
    values, stream = independent_probe_tokens(
        dataset,
        trajectory["train_document_offset"],
        (source_step + 1) * tokens_per_update,
    )
    next_batch = np.asarray(
        values[source_step * tokens_per_update : (source_step + 1) * tokens_per_update],
        dtype=np.int64,
    ).reshape(batch_size, width + 1)
    next_inputs = next_batch[:, :-1].reshape(-1)
    probe = np.asarray(context["probe_values"], dtype=np.int64).reshape(
        contexts, width + 1
    )
    probe_inputs = probe[:, :-1].reshape(-1)
    choices = sorted(set(next_inputs.tolist()) - set(probe_inputs.tolist()))
    if not choices:
        raise ValueError("immediate next batch has no source-null embedding token")
    selected = int(choices[0])
    return {
        "selected_token_id": selected,
        "next_input_occurrences": int(np.sum(next_inputs == selected)),
        "absent_from_probe_inputs": bool(np.all(probe_inputs != selected)),
        "next_batch_byte_token_sha256": hashlib.sha256(
            bytes(int(value) - 1 for value in next_batch.reshape(-1))
        ).hexdigest(),
        "stream_prefix": stream,
        "eligible_token_count": len(choices),
    }


def apply_source_null_embedding_shift(
    checkpoint: dict[str, Any],
    selection: dict[str, Any],
    specification: dict[str, Any],
) -> dict[str, Any]:
    import torch

    key = specification["model_key"]
    token = int(selection["selected_token_id"])
    coordinate = int(specification["coordinate"])
    offset = float(specification["offset"])
    tensor = checkpoint["model"].get(key)
    if (
        tensor is None
        or tensor.ndim != 2
        or token < 0
        or token >= tensor.shape[0]
        or coordinate < 0
        or coordinate >= tensor.shape[1]
    ):
        raise ValueError("selected embedding coordinate is invalid")
    before = tensor[token, coordinate].detach().clone()
    tensor[token, coordinate].add_(offset)
    after = tensor[token, coordinate].detach().clone()
    if torch.equal(before, after):
        raise ValueError("embedding intervention was absorbed by representation")
    return {
        "operation": "source-null-embedding-shift",
        "role": "gauge-canonical-optimizer-candidate-closure-test",
        "model_key": key,
        "token_id": token,
        "coordinate": coordinate,
        "offset": offset,
        "before_hex": before.cpu().numpy().tobytes().hex(),
        "after_hex": after.cpu().numpy().tobytes().hex(),
        "selection": selection,
    }


def intervention(
    checkpoint: dict[str, Any],
    branch: str,
    selection: dict[str, Any],
    shift_specification: dict[str, Any],
) -> dict[str, Any]:
    if branch == "baseline-direct":
        return {"operation": "none", "role": "baseline-replay"}
    if branch == "gauge-consistent":
        result = apply_branch_intervention(checkpoint, "sign-gauge-consistent")
        result["role"] = "complete-state-quotient-null-control"
        return result
    if branch == "gauge-params-only":
        result = apply_branch_intervention(checkpoint, "sign-gauge-params-only")
        result["role"] = "raw-optimizer-candidate-closure-test"
        return result
    if branch == "probe-null-embedding-shift":
        return apply_source_null_embedding_shift(
            checkpoint, selection, shift_specification
        )
    raise ValueError(f"unknown staged branch: {branch}")


def expected_difference_paths(
    base: dict[str, Any], branch: str, details: dict[str, Any]
) -> set[str]:
    expected = {"code_commit", "protocol_sha256"}
    if branch in {"gauge-consistent", "gauge-params-only"}:
        selected_names = set(details["model_keys"])
        expected.update(f"model.{name}" for name in selected_names)
        if branch == "gauge-consistent":
            names = list(base["model"])
            parameter_ids = [
                value
                for group in base["optimizer"]["param_groups"]
                for value in group["params"]
            ]
            expected.update(
                f"optimizer.state.{parameter_id}.exp_avg"
                for name, parameter_id in zip(names, parameter_ids)
                if name in selected_names
            )
    elif branch == "probe-null-embedding-shift":
        expected.add(f"model.{details['model_key']}")
    return expected


def run_rows(rows: list[dict[str, Any]], output_root: Path) -> list[dict[str, Any]]:
    completed = []
    for row in rows:
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(row["device"])
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        with row["paths"]["log"].open("xb") as stream:
            subprocess.run(
                row["command"],
                cwd=ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        metadata = json.loads(row["paths"]["metadata"].read_text(encoding="utf-8"))
        if (
            metadata.get("schema_version") != PRODUCER_SCHEMA
            or metadata.get("code_commit") != row["code_commit"]
            or metadata.get("trajectory_id") != row["trajectory_id"]
        ):
            raise ValueError("staged producer metadata identity mismatch")
        checkpoints = emitted_checkpoint_descriptors(
            metadata,
            row["paths"]["root"],
            output_root,
            row["trajectory_id"],
            row["branch"],
            set(row["checkpoint_steps"]),
        )
        completed.append(
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "paths",
                    "command",
                    "checkpoint_steps",
                    "code_commit",
                }
            }
            | {
                "resume_checkpoint": descriptor(row["paths"]["resume"], output_root),
                "energies": descriptor(row["paths"]["energies"], output_root),
                "metrics": descriptor(row["paths"]["metrics"], output_root),
                "metadata": descriptor(row["paths"]["metadata"], output_root),
                "producer_log": descriptor(row["paths"]["log"], output_root),
                "output_checkpoints": checkpoints,
                "wall_seconds": float(metadata["wall_seconds"]),
            }
        )
    return completed


def branch_paths(root: Path) -> dict[str, Path]:
    return {
        "root": root,
        "resume": root / "resume-checkpoint.pt",
        "energies": root / "energies.npz",
        "metrics": root / "metrics.npz",
        "metadata": root / "metadata.json",
        "checkpoints": root / "checkpoints",
        "log": root / "producer.log",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--refinedweb-root", default=configured_path('assets:refinedweb'))
    arguments = parser.parse_args()
    config_path = Path(arguments.config).resolve()
    data_root = Path(arguments.data_root).resolve()
    output_root = Path(arguments.output_root).resolve()
    try:
        output_root.relative_to(data_root)
    except ValueError as error:
        raise ValueError("staged output root must be inside the data root") from error
    if output_root.exists():
        raise FileExistsError(f"staged output root already exists: {output_root}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    commit, sources = committed_identity()
    source_root = resolve_data_root_locator(data_root, config["source_run"])
    predecessor_root = resolve_data_root_locator(
        data_root, config["predecessor_closure"]
    )
    source_config_path = source_root / "config-resolved.json"
    source_protocol_path = source_root / "protocol-resolved.json"
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    source_rows = {row["trajectory_id"]: row for row in source_config["trajectories"]}
    if set(source_rows) != {row["trajectory_id"] for row in config["trajectories"]}:
        raise ValueError("staged trajectories differ from the source registry")

    source_step = int(config["source_step"])
    final_step = source_step + max(config["block_sizes"])
    middle_step = source_step + int(config["semigroup_split"])
    checkpoint_steps = [0, middle_step, final_step]
    output_root.mkdir(parents=True)
    inputs_root = output_root / "inputs"
    inputs_root.mkdir()
    config_copy = inputs_root / "config.json"
    write_new_json(config_copy, config)
    protocol = copy.deepcopy(source_protocol)
    original_protocol = {
        "protocol_id": protocol["protocol_id"],
        "expected_updates": protocol["expected_updates"],
        "segments": protocol["segments"],
        "checkpoint_steps": protocol["plga_analysis"]["checkpoint_steps"],
    }
    protocol["protocol_id"] = config["stage_id"]
    protocol["expected_updates"] = final_step
    protocol["segments"] = {
        "burn_in_updates": source_step,
        "design_updates": 0,
        "holdout_updates": max(config["block_sizes"]),
    }
    protocol["plga_analysis"]["checkpoint_steps"] = checkpoint_steps
    protocol["staged_predictive_closure"] = {
        "source_step": source_step,
        "block_sizes": config["block_sizes"],
        "semigroup_split": config["semigroup_split"],
        "candidate_states": config["candidate_states"],
        "splits": config["splits"],
        "branch_ids": config["branch_ids"],
    }
    protocol_path = inputs_root / "protocol.json"
    write_new_json(protocol_path, protocol)
    protocol_digest = hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()

    import torch
    from datasets import Dataset

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    context = observer_context(source_config, Path(arguments.refinedweb_root).resolve())
    dataset_path = Path(option(source_config["producer_command"], "--dataset-shard"))
    dataset = Dataset.from_file(str(dataset_path))

    prepared_by_device: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    source_descriptors = []
    source_records = []
    restored_rows = []
    branch_rows_for_npz = []
    common_map_ids = None
    source_gpu_seconds = 0.0

    for requested in config["trajectories"]:
        trajectory_id = requested["trajectory_id"]
        source_row = source_rows[trajectory_id]
        if source_row["device"] != requested["device"]:
            raise ValueError("producing-device assignment changed")
        trajectory = {
            "trajectory_id": trajectory_id,
            "device": int(requested["device"]),
            "seed": int(source_row["seed"]),
            "train_document_offset": int(source_row["train_document_offset"]),
        }
        source_checkpoint_path, source_descriptor = checkpoint_descriptor(
            source_root, trajectory_id, source_step
        )
        source_zero_path, source_zero_descriptor = checkpoint_descriptor(
            source_root, trajectory_id, 0
        )
        source_descriptors.extend([source_zero_descriptor, source_descriptor])
        base = torch.load(source_checkpoint_path, map_location="cpu", weights_only=False)
        if base.get("schema_version") != CHECKPOINT_SCHEMA:
            raise ValueError("source checkpoint schema differs")
        restored, map_ids = restored_energy_vector(base, source_step)
        if common_map_ids is None:
            common_map_ids = map_ids
        elif not np.array_equal(common_map_ids, map_ids):
            raise ValueError("source map registries differ")
        original_energy, measured_map_ids, original_route = observe_checkpoint_energy(
            base, f"cuda:{trajectory['device']}", context
        )
        if not np.array_equal(map_ids, measured_map_ids):
            raise ValueError("source observation changed the map registry")
        source_gpu_seconds += float(original_route["wall_seconds"])
        selection = select_source_null_token(
            dataset,
            context,
            source_config["producer_command"],
            trajectory,
            source_step,
        )
        phase = phase_record(
            source_step=source_step,
            trajectory=trajectory,
            source_command=source_config["producer_command"],
            next_batch_digest=selection["next_batch_byte_token_sha256"],
            dataset_digest=context["dataset"]["sha256"],
        )
        base_transition = canonical_transition_state_sha256(
            base,
            phase=phase,
            is_gauge_parameter=is_sign_gauge_parameter,
            gauge_layer=sign_gauge_layer,
        )
        restored_rows.append(restored)
        observed_branches = []
        trajectory_sources = []

        for branch in BRANCH_IDS:
            root = output_root / "trajectories" / trajectory_id / branch
            paths = branch_paths(root)
            paths["checkpoints"].mkdir(parents=True)
            shutil.copy2(
                source_zero_path, paths["checkpoints"] / "checkpoint-step000000.pt"
            )
            checkpoint = copy.deepcopy(base)
            checkpoint["code_commit"] = commit
            checkpoint["protocol_sha256"] = protocol_digest
            details = intervention(
                checkpoint, branch, selection, config["embedding_shift"]
            )
            actual_differences = set(differing_paths(base, checkpoint))
            expected_differences = expected_difference_paths(base, branch, details)
            if actual_differences != expected_differences:
                raise RuntimeError(
                    f"unexpected checkpoint rewrite for {trajectory_id}/{branch}: "
                    f"{sorted(actual_differences ^ expected_differences)}"
                )
            branch_energy, branch_map_ids, route = observe_checkpoint_energy(
                checkpoint, f"cuda:{trajectory['device']}", context
            )
            if not np.array_equal(map_ids, branch_map_ids):
                raise ValueError("branch source observation changed map registry")
            source_gpu_seconds += float(route["wall_seconds"])
            signatures = candidate_signatures(
                checkpoint,
                branch_energy,
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
            torch.save(checkpoint, paths["resume"])
            command = format_command(
                source_config["producer_command"],
                trajectory,
                paths,
                protocol_path,
                commit,
                final_step,
            )
            prepared_by_device[trajectory["device"]].append(
                {
                    "trajectory_id": trajectory_id,
                    "branch": branch,
                    "device": trajectory["device"],
                    "split": "design"
                    if trajectory_id in config["splits"]["design"]
                    else "holdout",
                    "branch_specification": details,
                    "source_candidate_signatures": signatures,
                    "source_canonical_transition_state_sha256": transition,
                    "checkpoint_rewrite": {
                        "differing_leaf_paths": sorted(actual_differences),
                        "expected_leaf_paths": sorted(expected_differences),
                    },
                    "paths": paths,
                    "command": command,
                    "checkpoint_steps": checkpoint_steps,
                    "code_commit": commit,
                }
            )
            observed_branches.append(branch_energy)
            trajectory_sources.append(
                {
                    "branch": branch,
                    "route": route,
                    "candidate_signatures": signatures,
                    "canonical_transition_state_sha256": transition,
                }
            )
        branch_rows_for_npz.append(observed_branches)
        source_records.append(
            {
                "trajectory_id": trajectory_id,
                "split": "design"
                if trajectory_id in config["splits"]["design"]
                else "holdout",
                "phase": phase,
                "embedding_selection": selection,
                "original_route": original_route,
                "original_candidate_signatures": candidate_signatures(
                    base,
                    original_energy,
                    original_route["row_map_sha256"],
                    phase=phase,
                    is_gauge_parameter=is_sign_gauge_parameter,
                    gauge_layer=sign_gauge_layer,
                ),
                "original_canonical_transition_state_sha256": base_transition,
                "branches": trajectory_sources,
            }
        )

    assert common_map_ids is not None
    measurement_path = output_root / "source-remeasurement.npz"
    np.savez_compressed(
        measurement_path,
        trajectory_ids=np.asarray(
            [row["trajectory_id"] for row in config["trajectories"]]
        ),
        branch_ids=np.asarray(BRANCH_IDS),
        map_ids=common_map_ids,
        restored_recorder_energy=np.asarray(restored_rows, dtype=np.float64),
        rewritten_branch_remeasured_energy=np.asarray(
            branch_rows_for_npz, dtype=np.float64
        ),
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            device: executor.submit(run_rows, rows, output_root)
            for device, rows in prepared_by_device.items()
        }
        completed_rows = [
            row
            for device in sorted(futures)
            for row in futures[device].result()
        ]
    completed = {
        trajectory_id: [
            row for row in completed_rows if row["trajectory_id"] == trajectory_id
        ]
        for trajectory_id in [row["trajectory_id"] for row in config["trajectories"]]
    }

    semigroup_prepared: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for requested in config["trajectories"]:
        trajectory_id = requested["trajectory_id"]
        device = int(requested["device"])
        source_row = source_rows[trajectory_id]
        trajectory = {
            "trajectory_id": trajectory_id,
            "device": device,
            "seed": int(source_row["seed"]),
            "train_document_offset": int(source_row["train_document_offset"]),
        }
        baseline = next(
            row for row in completed[trajectory_id] if row["branch"] == "baseline-direct"
        )
        by_step = {row["step"]: row for row in baseline["output_checkpoints"]}
        root = output_root / "trajectories" / trajectory_id / "baseline-iterated"
        paths = branch_paths(root)
        paths["checkpoints"].mkdir(parents=True)
        for step in (0, middle_step):
            source = output_root / by_step[step]["path"]
            shutil.copy2(
                source, paths["checkpoints"] / f"checkpoint-step{step:06d}.pt"
            )
        shutil.copy2(
            output_root / by_step[middle_step]["path"], paths["resume"]
        )
        command = format_command(
            source_config["producer_command"],
            trajectory,
            paths,
            protocol_path,
            commit,
            final_step,
        )
        semigroup_prepared[device].append(
            {
                "trajectory_id": trajectory_id,
                "branch": "baseline-iterated",
                "device": device,
                "split": "design"
                if trajectory_id in config["splits"]["design"]
                else "holdout",
                "branch_specification": {
                    "operation": "restart-at-intermediate-checkpoint",
                    "source_branch": "baseline-direct",
                    "restart_step": middle_step,
                    "role": "direct-versus-iterated-semigroup-control",
                },
                "paths": paths,
                "command": command,
                "checkpoint_steps": checkpoint_steps,
                "code_commit": commit,
            }
        )
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            device: executor.submit(run_rows, rows, output_root)
            for device, rows in semigroup_prepared.items()
        }
        semigroup_rows = [
            row
            for device in sorted(futures)
            for row in futures[device].result()
        ]

    predecessor_manifest_path = predecessor_root / "run-manifest.json"
    predecessor_result_path = predecessor_root / "result.json"
    predecessor_manifest = json.loads(
        predecessor_manifest_path.read_text(encoding="utf-8")
    )
    predecessor_result = json.loads(predecessor_result_path.read_text(encoding="utf-8"))
    verify_record_seal(predecessor_manifest)
    verify_record_seal(predecessor_result)
    producer_seconds = sum(row["wall_seconds"] for row in completed_rows + semigroup_rows)
    gpu_seconds = source_gpu_seconds + producer_seconds
    cap_seconds = float(config["gpu_hour_cap"]) * 3600.0
    if gpu_seconds > cap_seconds:
        raise RuntimeError(
            f"Stage-1 GPU cap exceeded: {gpu_seconds:.3f} > {cap_seconds:.3f} seconds"
        )
    all_checkpoints = [
        item
        for row in completed_rows + semigroup_rows
        for item in row["output_checkpoints"]
    ]
    manifest = seal_record(
        {
            "schema_version": MANIFEST_SCHEMA,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "code_commit": commit,
            "analysis_sources": sources,
            "stage_id": config["stage_id"],
            "source_run": {
                "locator": config["source_run"],
                "run_id": source_config["run_id"],
                "config_sha256": file_sha256(source_config_path),
                "protocol_sha256": file_sha256(source_protocol_path),
                "checkpoints": source_descriptors,
            },
            "predecessor": {
                "locator": config["predecessor_closure"],
                "run_manifest": anchored_descriptor(
                    predecessor_manifest_path, data_root
                )
                | {"record_sha256": predecessor_manifest["record_sha256"]},
                "result": anchored_descriptor(predecessor_result_path, data_root)
                | {"record_sha256": predecessor_result["record_sha256"]},
            },
            "inputs": {
                "config": descriptor(config_copy, output_root),
                "protocol": descriptor(protocol_path, output_root),
                "source_remeasurement": descriptor(measurement_path, output_root),
            },
            "protocol_rewrite": {
                "source": original_protocol,
                "successor": {
                    "protocol_id": protocol["protocol_id"],
                    "expected_updates": protocol["expected_updates"],
                    "segments": protocol["segments"],
                    "checkpoint_steps": protocol["plga_analysis"]["checkpoint_steps"],
                    "staged_predictive_closure": protocol[
                        "staged_predictive_closure"
                    ],
                },
                "successor_canonical_sha256": protocol_digest,
            },
            "source_remeasurement": {
                "observer": (
                    "remeasured float32 row maps and CPU-float64 row energies on "
                    "each producing GPU"
                ),
                "reference_sources": context["reference_sources"],
                "dataset": context["dataset"],
                "probe": context["probe_metadata"],
                "trajectories": source_records,
                "cuda_wall_seconds": source_gpu_seconds,
            },
            "source_step": source_step,
            "middle_step": middle_step,
            "final_step": final_step,
            "block_sizes": config["block_sizes"],
            "splits": config["splits"],
            "candidate_states": config["candidate_states"],
            "branch_ids": list(BRANCH_IDS),
            "branches": completed,
            "semigroup_branches": {
                trajectory_id: [
                    row
                    for row in semigroup_rows
                    if row["trajectory_id"] == trajectory_id
                ]
                for trajectory_id in completed
            },
            "checkpoint_inventory": {
                "descriptor_count": len(all_checkpoints),
                "total_size_bytes": sum(row["size_bytes"] for row in all_checkpoints),
                "expected_steps_per_branch": checkpoint_steps,
                "all_emitted_files_bound": len(all_checkpoints)
                == (len(BRANCH_IDS) + 1) * len(completed) * len(checkpoint_steps),
            },
            "gpu_time_accounting": {
                "source_remeasurement_cuda_wall_seconds": source_gpu_seconds,
                "continuation_producer_reported_wall_seconds": producer_seconds,
                "cap_gpu_hours": float(config["gpu_hour_cap"]),
                "cap_passed": gpu_seconds <= cap_seconds,
            },
            "gpu_seconds": gpu_seconds,
            "gpu_hours": gpu_seconds / 3600.0,
        }
    )
    write_new_json(output_root / "run-manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "complete",
                "record_sha256": manifest["record_sha256"],
                "gpu_hours": manifest["gpu_hours"],
                "trajectory_count": len(completed),
                "continuation_count": len(completed_rows) + len(semigroup_rows),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
