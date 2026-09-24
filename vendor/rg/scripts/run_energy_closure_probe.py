#!/usr/bin/env python3
"""Run matched PLDR continuations that differ only in AdamW first moments."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from row_rgmap.analysis import (  # noqa: E402
    canonical_json_bytes,
    file_sha256,
    seal_record,
)
from row_rgmap.provenance_v3 import git_blob_descriptor  # noqa: E402


CONFIG_SCHEMA = "pldr-row-rg-energy-closure-probe-config-v1"
MANIFEST_SCHEMA = "pldr-row-rg-energy-closure-probe-run-v1"
PRODUCER_SCHEMA = "pldr-row-rg-energy-producer-metadata-v3"
CHECKPOINT_SCHEMA = "pldr-training-checkpoint-v3"
SOURCE_PATHS = (
    "scripts/run_energy_closure_probe.py",
    "scripts/pldr_energy_producer_v3.py",
    "src/row_rgmap/analysis.py",
    "src/row_rgmap/provenance_v3.py",
    "src/row_rgmap/recorder.py",
)


def write_new_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def descriptor(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def committed_identity(config_path: Path) -> tuple[str, list[dict[str, Any]]]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    if status:
        raise RuntimeError("closure probe requires a clean committed repository")
    try:
        config_source = config_path.relative_to(ROOT).as_posix()
    except ValueError as error:
        raise ValueError("closure configuration must be committed in the repository") from error
    sources = []
    for repository_path in (*SOURCE_PATHS, config_source):
        digest, size = git_blob_descriptor(ROOT, commit, repository_path)
        worktree = ROOT / repository_path
        if (
            not worktree.is_file()
            or file_sha256(worktree) != digest
            or worktree.stat().st_size != size
        ):
            raise RuntimeError(f"closure source differs from HEAD: {repository_path}")
        sources.append({
            "repository_path": repository_path,
            "code_commit": commit,
            "sha256": digest,
            "size_bytes": size,
        })
    return commit, sources


def option(command: list[str], name: str) -> str:
    positions = [index for index, value in enumerate(command) if value == name]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError(f"producer command lacks unique option {name}")
    return command[positions[0] + 1]


def replace_option(command: list[str], name: str, value: str) -> None:
    index = command.index(name)
    command[index + 1] = value


def tensor_mapping_sha256(values: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(values.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        fields = (
            name.encode("utf-8"),
            str(array.dtype).encode("ascii"),
            json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"),
            array.tobytes(order="C"),
        )
        for field in fields:
            digest.update(len(field).to_bytes(8, "big"))
            digest.update(field)
    return digest.hexdigest()


def first_moment_summary(checkpoint: dict[str, Any]) -> dict[str, Any]:
    element_count = 0
    square_sum = 0.0
    nonzero_count = 0
    tensor_count = 0
    for state in checkpoint["optimizer"]["state"].values():
        moment = state.get("exp_avg")
        if moment is None:
            continue
        tensor_count += 1
        values = moment.detach().double().cpu()
        element_count += values.numel()
        nonzero_count += int((values != 0).sum())
        square_sum += float((values * values).sum())
    return {
        "tensor_count": tensor_count,
        "element_count": element_count,
        "nonzero_count": nonzero_count,
        "l2_norm": math.sqrt(square_sum),
    }


def zero_first_moments(checkpoint: dict[str, Any]) -> None:
    for state in checkpoint["optimizer"]["state"].values():
        if "exp_avg" in state:
            state["exp_avg"].zero_()


def checkpoint_descriptor(
    source_root: Path, trajectory_id: str, step: int
) -> tuple[Path, dict[str, Any]]:
    metadata_path = source_root / "trajectories" / trajectory_id / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    matches = [row for row in metadata["checkpoints"] if int(row["step"]) == step]
    if len(matches) != 1:
        raise ValueError(f"checkpoint step {step} is not unique for {trajectory_id}")
    row = matches[0]
    path = (metadata_path.parent / row["path"]).resolve()
    if (
        not path.is_file()
        or file_sha256(path) != row["sha256"]
        or path.stat().st_size != row["size_bytes"]
    ):
        raise ValueError(f"checkpoint descriptor fails for {trajectory_id} step {step}")
    return path, {
        "trajectory_id": trajectory_id,
        "step": step,
        "path": path.relative_to(source_root).as_posix(),
        "sha256": row["sha256"],
        "size_bytes": row["size_bytes"],
        "metadata_path": metadata_path.relative_to(source_root).as_posix(),
        "metadata_sha256": file_sha256(metadata_path),
        "metadata_size_bytes": metadata_path.stat().st_size,
    }


def format_command(
    template: list[str], trajectory: dict[str, Any], paths: dict[str, Path],
    protocol_path: Path, commit: str, final_step: int,
) -> list[str]:
    substitutions = {
        "device": str(trajectory["device"]),
        "trajectory_id": trajectory["trajectory_id"],
        "seed": str(trajectory["seed"]),
        "train_document_offset": str(trajectory["train_document_offset"]),
        "output": str(paths["energies"]),
        "metrics": str(paths["metrics"]),
        "metadata": str(paths["metadata"]),
        "checkpoint_dir": str(paths["checkpoints"]),
        "protocol": str(protocol_path),
        "code_commit": commit,
    }
    command = [value.format(**substitutions) for value in template]
    command[0] = sys.executable
    command[1] = "scripts/pldr_energy_producer_v3.py"
    replace_option(command, "--steps", str(final_step))
    command.extend(["--resume-from", str(paths["resume"])])
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    arguments = parser.parse_args()
    config_path = Path(arguments.config).resolve()
    output_root = Path(arguments.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"closure output root already exists: {output_root}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unknown closure-probe configuration schema")
    block_sizes = config.get("block_sizes")
    if block_sizes != [1, 2, 4, 8] or config.get("baseline_replicates") != 2:
        raise ValueError("closure probe requires scales [1,2,4,8] and two baselines")
    if config.get("intervention") != "zero_adamw_first_moment":
        raise ValueError("unknown closure intervention")
    trajectories = config.get("trajectories")
    if (
        not isinstance(trajectories, list)
        or len(trajectories) != 2
        or sorted(row.get("device") for row in trajectories) != [0, 1]
        or len({row.get("trajectory_id") for row in trajectories}) != 2
        or len({row.get("source_step") for row in trajectories}) != 1
    ):
        raise ValueError("closure probe requires two unique trajectories on devices 0 and 1")
    commit, sources = committed_identity(config_path)
    source_root = Path(config["source_run_root"]).resolve()
    source_config_path = source_root / "config-resolved.json"
    source_protocol_path = source_root / "protocol-resolved.json"
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    source_rows = {row["trajectory_id"]: row for row in source_config["trajectories"]}
    source_step = int(trajectories[0]["source_step"])
    final_step = source_step + max(block_sizes)

    output_root.mkdir(parents=True)
    inputs_root = output_root / "inputs"
    inputs_root.mkdir()
    config_copy = inputs_root / "config.json"
    write_new_json(config_copy, config)
    protocol = copy.deepcopy(source_protocol)
    protocol["protocol_id"] = config["probe_id"]
    protocol["expected_updates"] = final_step
    protocol["segments"] = {
        "burn_in_updates": source_step,
        "design_updates": 0,
        "holdout_updates": max(block_sizes),
    }
    protocol["plga_analysis"]["checkpoint_steps"] = [0, final_step]
    protocol["closure_probe"] = {
        "source_step": source_step,
        "block_sizes": block_sizes,
        "intervention": config["intervention"],
    }
    protocol_path = inputs_root / "protocol.json"
    write_new_json(protocol_path, protocol)
    protocol_digest = hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()

    import torch

    prepared: dict[str, list[dict[str, Any]]] = {}
    source_descriptors = []
    for requested in trajectories:
        trajectory_id = requested["trajectory_id"]
        if trajectory_id not in source_rows:
            raise ValueError(f"unknown source trajectory: {trajectory_id}")
        source_trajectory = source_rows[trajectory_id]
        trajectory = {
            "trajectory_id": trajectory_id,
            "device": int(requested["device"]),
            "seed": int(source_trajectory["seed"]),
            "train_document_offset": int(source_trajectory["train_document_offset"]),
        }
        source_checkpoint, source_descriptor = checkpoint_descriptor(
            source_root, trajectory_id, source_step
        )
        source_zero, source_zero_descriptor = checkpoint_descriptor(
            source_root, trajectory_id, 0
        )
        source_descriptors.extend([source_zero_descriptor, source_descriptor])
        base_checkpoint = torch.load(
            source_checkpoint, map_location="cpu", weights_only=False
        )
        if base_checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
            raise ValueError("unknown source checkpoint schema")
        model_digest = tensor_mapping_sha256(base_checkpoint["model"])
        moments_before = first_moment_summary(base_checkpoint)
        if moments_before["nonzero_count"] == 0:
            raise ValueError("source AdamW first moments are identically zero")
        branch_rows = []
        for branch_name in ("baseline-a", "baseline-b", "zero-first-moment"):
            branch_root = output_root / "trajectories" / trajectory_id / branch_name
            checkpoint_dir = branch_root / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            shutil.copy2(
                source_zero,
                checkpoint_dir / "checkpoint-step000000.pt",
            )
            checkpoint = copy.deepcopy(base_checkpoint)
            checkpoint["code_commit"] = commit
            checkpoint["protocol_sha256"] = protocol_digest
            if branch_name == "zero-first-moment":
                zero_first_moments(checkpoint)
            moments_after = first_moment_summary(checkpoint)
            if branch_name.startswith("baseline") and moments_after != moments_before:
                raise RuntimeError("baseline checkpoint changed optimizer moments")
            if branch_name == "zero-first-moment" and moments_after["nonzero_count"] != 0:
                raise RuntimeError("AdamW first-moment intervention is incomplete")
            if tensor_mapping_sha256(checkpoint["model"]) != model_digest:
                raise RuntimeError("closure intervention changed model parameters")
            paths = {
                "root": branch_root,
                "resume": branch_root / "resume-checkpoint.pt",
                "energies": branch_root / "energies.npz",
                "metrics": branch_root / "metrics.npz",
                "metadata": branch_root / "metadata.json",
                "checkpoints": checkpoint_dir,
                "log": branch_root / "producer.log",
            }
            torch.save(checkpoint, paths["resume"])
            command = format_command(
                source_config["producer_command"], trajectory, paths,
                protocol_path, commit, final_step,
            )
            branch_rows.append({
                "branch": branch_name,
                "trajectory": trajectory,
                "paths": paths,
                "command": command,
                "source_zero_checkpoint": source_zero_descriptor,
                "model_state_sha256": model_digest,
                "first_moment_before": moments_before,
                "first_moment_after": moments_after,
            })
        prepared[trajectory_id] = branch_rows

    def run_trajectory(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        completed_rows = []
        for row in rows:
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(row["trajectory"]["device"])
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            with row["paths"]["log"].open("xb") as stream:
                subprocess.run(
                    row["command"], cwd=ROOT, env=environment,
                    stdout=stream, stderr=subprocess.STDOUT, check=True,
                )
            metadata = json.loads(row["paths"]["metadata"].read_text(encoding="utf-8"))
            if (
                metadata.get("schema_version") != PRODUCER_SCHEMA
                or metadata.get("code_commit") != commit
                or metadata.get("trajectory_id") != row["trajectory"]["trajectory_id"]
            ):
                raise ValueError("closure producer metadata identity mismatch")
            completed_rows.append({
                "branch": row["branch"],
                "device": row["trajectory"]["device"],
                "model_state_sha256": row["model_state_sha256"],
                "first_moment_before": row["first_moment_before"],
                "first_moment_after": row["first_moment_after"],
                "resume_checkpoint": descriptor(row["paths"]["resume"], output_root),
                "energies": descriptor(row["paths"]["energies"], output_root),
                "metrics": descriptor(row["paths"]["metrics"], output_root),
                "metadata": descriptor(row["paths"]["metadata"], output_root),
                "producer_log": descriptor(row["paths"]["log"], output_root),
                "wall_seconds": float(metadata["wall_seconds"]),
            })
        return completed_rows

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            trajectory_id: executor.submit(run_trajectory, rows)
            for trajectory_id, rows in prepared.items()
        }
        completed = {
            trajectory_id: future.result()
            for trajectory_id, future in futures.items()
        }
    gpu_seconds = sum(
        row["wall_seconds"] for rows in completed.values() for row in rows
    )
    manifest = seal_record({
        "schema_version": MANIFEST_SCHEMA,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": commit,
        "analysis_sources": sources,
        "probe_id": config["probe_id"],
        "source_run": {
            "run_id": source_config["run_id"],
            "config_sha256": file_sha256(source_config_path),
            "protocol_sha256": file_sha256(source_protocol_path),
            "checkpoints": source_descriptors,
        },
        "inputs": {
            "config": descriptor(config_copy, output_root),
            "protocol": descriptor(protocol_path, output_root),
        },
        "source_step": source_step,
        "final_step": final_step,
        "block_sizes": block_sizes,
        "intervention": config["intervention"],
        "branches": completed,
        "gpu_seconds": gpu_seconds,
        "gpu_hours": gpu_seconds / 3600.0,
    })
    manifest_path = output_root / "run-manifest.json"
    write_new_json(manifest_path, manifest)
    print(json.dumps({
        "status": "complete",
        "record_sha256": manifest["record_sha256"],
        "gpu_hours": manifest["gpu_hours"],
        "trajectory_count": len(completed),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
