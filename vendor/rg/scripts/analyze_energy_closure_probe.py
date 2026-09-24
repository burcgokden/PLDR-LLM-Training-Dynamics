#!/usr/bin/env python3
"""Verify matched continuation artifacts and measure scale-indexed closure defects."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from row_rgmap.analysis import file_sha256, seal_record, verify_record_seal  # noqa: E402
from row_rgmap.provenance_v3 import git_blob_descriptor  # noqa: E402


RUN_SCHEMA = "pldr-row-rg-energy-closure-probe-run-v1"
RESULT_SCHEMA = "pldr-row-rg-energy-closure-analysis-v1"
SOURCE_PATHS = (
    "scripts/analyze_energy_closure_probe.py",
    "src/row_rgmap/analysis.py",
    "src/row_rgmap/provenance_v3.py",
)


def committed_identity() -> tuple[str, list[dict[str, Any]]]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    if status:
        raise RuntimeError("closure analysis requires a clean committed repository")
    sources = []
    for repository_path in SOURCE_PATHS:
        digest, size = git_blob_descriptor(ROOT, commit, repository_path)
        sources.append({
            "repository_path": repository_path,
            "code_commit": commit,
            "sha256": digest,
            "size_bytes": size,
        })
    return commit, sources


def verify_sources(record: dict[str, Any]) -> None:
    commit = record.get("code_commit")
    sources = record.get("analysis_sources")
    if not isinstance(commit, str) or not isinstance(sources, list) or not sources:
        raise ValueError("closure run source registry is incomplete")
    for source in sources:
        path = source.get("repository_path")
        if not isinstance(path, str):
            raise ValueError("closure run source is not repository-relative")
        digest, size = git_blob_descriptor(ROOT, commit, path)
        if (
            source.get("code_commit") != commit
            or source.get("sha256") != digest
            or source.get("size_bytes") != size
        ):
            raise ValueError(f"closure run source mismatch: {path}")


def resolve_descriptor(root: Path, descriptor: dict[str, Any]) -> Path:
    relative = descriptor.get("path")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("closure artifact path is not relative")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("closure artifact path escapes its root") from error
    if (
        not path.is_file()
        or path.stat().st_size != descriptor.get("size_bytes")
        or file_sha256(path) != descriptor.get("sha256")
    ):
        raise ValueError(f"closure artifact descriptor fails: {relative}")
    return path


def _digest_field(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def object_fingerprint(value: Any, *, omit_first_moment: bool = False) -> str:
    """Deterministically fingerprint the nested state saved by torch."""

    import torch

    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            _digest_field(digest, b"tensor")
            _digest_field(digest, str(array.dtype).encode("ascii"))
            _digest_field(
                digest,
                json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"),
            )
            _digest_field(digest, array.tobytes(order="C"))
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            _digest_field(digest, b"ndarray")
            _digest_field(digest, str(array.dtype).encode("ascii"))
            _digest_field(
                digest,
                json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"),
            )
            _digest_field(digest, array.tobytes(order="C"))
        elif isinstance(item, dict):
            _digest_field(digest, b"dict")
            keys = sorted(item, key=lambda key: (type(key).__name__, repr(key)))
            for key in keys:
                visit(key)
                if omit_first_moment and key == "exp_avg":
                    _digest_field(digest, b"omitted-first-moment")
                else:
                    visit(item[key])
        elif isinstance(item, (list, tuple)):
            _digest_field(digest, type(item).__name__.encode("ascii"))
            for member in item:
                visit(member)
        elif isinstance(item, bytes):
            _digest_field(digest, b"bytes")
            _digest_field(digest, item)
        elif item is None or isinstance(item, (bool, int, float, str)):
            _digest_field(digest, type(item).__name__.encode("ascii"))
            _digest_field(
                digest,
                json.dumps(item, allow_nan=False, separators=(",", ":")).encode("utf-8"),
            )
        else:
            raise TypeError(f"unsupported checkpoint value: {type(item).__name__}")

    visit(value)
    return digest.hexdigest()


def tensor_mapping_sha256(values: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(values.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        for field in (
            name.encode("utf-8"),
            str(array.dtype).encode("ascii"),
            json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"),
            array.tobytes(order="C"),
        ):
            _digest_field(digest, field)
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


def verify_source_run(manifest: dict[str, Any], run_root: Path) -> None:
    """Replay the external smoke-checkpoint descriptors copied into the manifest."""

    config_path = resolve_descriptor(run_root, manifest["inputs"]["config"])
    resolve_descriptor(run_root, manifest["inputs"]["protocol"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_root = Path(config["source_run_root"]).resolve()
    source_config = source_root / "config-resolved.json"
    source_protocol = source_root / "protocol-resolved.json"
    source = manifest["source_run"]
    if (
        not source_root.is_dir()
        or file_sha256(source_config) != source["config_sha256"]
        or file_sha256(source_protocol) != source["protocol_sha256"]
        or json.loads(source_config.read_text(encoding="utf-8")).get("run_id")
        != source["run_id"]
    ):
        raise ValueError("closure source run identity changed")
    for descriptor in source["checkpoints"]:
        checkpoint = (source_root / descriptor["path"]).resolve()
        metadata = (source_root / descriptor["metadata_path"]).resolve()
        for path in (checkpoint, metadata):
            try:
                path.relative_to(source_root)
            except ValueError as error:
                raise ValueError("closure source descriptor escapes its run root") from error
        if (
            not checkpoint.is_file()
            or checkpoint.stat().st_size != descriptor["size_bytes"]
            or file_sha256(checkpoint) != descriptor["sha256"]
            or not metadata.is_file()
            or metadata.stat().st_size != descriptor["metadata_size_bytes"]
            or file_sha256(metadata) != descriptor["metadata_sha256"]
        ):
            raise ValueError("closure source checkpoint descriptor changed")


def load_energy(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"energies", "steps", "trajectory_ids", "map_ids"}:
            raise ValueError("closure energy archive has an unexpected contract")
        energies = np.asarray(archive["energies"], dtype=np.float64)
        steps = np.asarray(archive["steps"], dtype=np.int64)
        map_ids = np.asarray(archive["map_ids"]).astype(str)
    if (
        energies.ndim != 3
        or energies.shape[0] != 1
        or energies.shape[1] != len(steps)
        or energies.shape[2] != len(map_ids)
        or not np.all(np.isfinite(energies))
        or np.any(energies < 0.0)
        or not np.array_equal(steps, np.arange(steps[-1] + 1))
    ):
        raise ValueError("closure energy archive is malformed")
    return energies[0], steps, map_ids


def closure_defect(reference: np.ndarray, intervention: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float64)
    intervention = np.asarray(intervention, dtype=np.float64)
    if reference.shape != intervention.shape or reference.ndim != 1:
        raise ValueError("closure vectors must have equal one-dimensional shape")
    absolute = np.abs(reference - intervention)
    denominator = np.abs(reference) + np.abs(intervention)
    symmetric = np.zeros_like(absolute)
    nonzero = denominator > 0.0
    symmetric[nonzero] = 2.0 * absolute[nonzero] / denominator[nonzero]
    bitwise = reference.view(np.uint64) != intervention.view(np.uint64)
    return {
        "map_count": int(reference.size),
        "bitwise_different_map_count": int(np.sum(bitwise)),
        "maximum_absolute_energy_defect": float(absolute.max(initial=0.0)),
        "maximum_symmetric_relative_energy_defect": float(
            symmetric.max(initial=0.0)
        ),
        "l2_energy_defect": float(np.linalg.norm(absolute)),
    }


def analyze(run_root: Path, *, commit: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    manifest_path = run_root / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_record_seal(manifest)
    if manifest.get("schema_version") != RUN_SCHEMA:
        raise ValueError("unknown closure run schema")
    verify_sources(manifest)
    verify_source_run(manifest, run_root)
    import torch

    block_sizes = manifest["block_sizes"]
    trajectories = []
    aggregate = {
        block_size: {
            "block_size": block_size,
            "trajectory_count": 0,
            "bitwise_different_map_count": 0,
            "maximum_absolute_energy_defect": 0.0,
            "maximum_symmetric_relative_energy_defect": 0.0,
            "closure_rejected": False,
        }
        for block_size in block_sizes
    }
    all_baselines_reproduce = True
    all_initial_energies_equal = True
    all_interventions_valid = True
    for trajectory_id, branch_rows in sorted(manifest["branches"].items()):
        branches = {row["branch"]: row for row in branch_rows}
        if set(branches) != {"baseline-a", "baseline-b", "zero-first-moment"}:
            raise ValueError("closure branch registry is incomplete")
        loaded = {}
        checkpoints = {}
        checkpoint_fingerprints = {}
        checkpoint_fingerprints_without_moment = {}
        recomputed_moments = {}
        recomputed_model_digests = {}
        for branch, row in branches.items():
            path = resolve_descriptor(run_root, row["energies"])
            resolve_descriptor(run_root, row["metadata"])
            resolve_descriptor(run_root, row["producer_log"])
            resolve_descriptor(run_root, row["metrics"])
            resume_path = resolve_descriptor(run_root, row["resume_checkpoint"])
            loaded[branch] = load_energy(path)
            checkpoints[branch] = torch.load(
                resume_path, map_location="cpu", weights_only=False
            )
            checkpoint_fingerprints[branch] = object_fingerprint(
                checkpoints[branch]
            )
            checkpoint_fingerprints_without_moment[branch] = object_fingerprint(
                checkpoints[branch], omit_first_moment=True
            )
            recomputed_moments[branch] = first_moment_summary(checkpoints[branch])
            recomputed_model_digests[branch] = tensor_mapping_sha256(
                checkpoints[branch]["model"]
            )
            if (
                recomputed_moments[branch] != row["first_moment_after"]
                or recomputed_model_digests[branch] != row["model_state_sha256"]
            ):
                raise ValueError("closure checkpoint summary does not replay")
        baseline, steps, map_ids = loaded["baseline-a"]
        baseline_b, steps_b, map_ids_b = loaded["baseline-b"]
        intervention, steps_i, map_ids_i = loaded["zero-first-moment"]
        baseline_reproduces = (
            np.array_equal(steps, steps_b)
            and np.array_equal(map_ids, map_ids_b)
            and np.array_equal(baseline, baseline_b)
        )
        aligned = (
            np.array_equal(steps, steps_i)
            and np.array_equal(map_ids, map_ids_i)
        )
        if not aligned:
            raise ValueError("closure branches have different step or map registries")
        source_matches = np.flatnonzero(steps == int(manifest["source_step"]))
        if len(source_matches) != 1:
            raise ValueError("closure source step is absent or duplicated")
        source_index = int(source_matches[0])
        initial_equal = np.array_equal(
            baseline[source_index], intervention[source_index]
        )
        model_digests = set(recomputed_model_digests.values())
        baseline_checkpoints_equal = (
            checkpoint_fingerprints["baseline-a"]
            == checkpoint_fingerprints["baseline-b"]
        )
        checkpoints_equal_except_moment = (
            len(set(checkpoint_fingerprints_without_moment.values())) == 1
        )
        source_moment = recomputed_moments["baseline-a"]
        stored_before_equal = all(
            row["first_moment_before"] == source_moment for row in branch_rows
        )
        intervention_valid = (
            len(model_digests) == 1
            and baseline_checkpoints_equal
            and checkpoints_equal_except_moment
            and stored_before_equal
            and source_moment["nonzero_count"] > 0
            and recomputed_moments["baseline-b"] == source_moment
            and recomputed_moments["zero-first-moment"]["nonzero_count"] == 0
            and recomputed_moments["zero-first-moment"]["l2_norm"] == 0.0
        )
        scale_rows = []
        for block_size in block_sizes:
            target = int(manifest["source_step"]) + int(block_size)
            matches = np.flatnonzero(steps == target)
            if len(matches) != 1:
                raise ValueError(f"closure target step is missing: {target}")
            defect = closure_defect(
                baseline[int(matches[0])], intervention[int(matches[0])]
            )
            defect.update({
                "block_size": int(block_size),
                "closure_rejected": defect["bitwise_different_map_count"] > 0,
            })
            scale_rows.append(defect)
            combined = aggregate[block_size]
            combined["trajectory_count"] += 1
            combined["bitwise_different_map_count"] += defect[
                "bitwise_different_map_count"
            ]
            combined["maximum_absolute_energy_defect"] = max(
                combined["maximum_absolute_energy_defect"],
                defect["maximum_absolute_energy_defect"],
            )
            combined["maximum_symmetric_relative_energy_defect"] = max(
                combined["maximum_symmetric_relative_energy_defect"],
                defect["maximum_symmetric_relative_energy_defect"],
            )
            combined["closure_rejected"] = (
                combined["closure_rejected"] or defect["closure_rejected"]
            )
        trajectories.append({
            "trajectory_id": trajectory_id,
            "map_count": int(len(map_ids)),
            "baseline_bitwise_reproducible": baseline_reproduces,
            "baseline_resume_checkpoints_equal": baseline_checkpoints_equal,
            "initial_energy_vector_bitwise_equal": initial_equal,
            "moment_only_intervention_validated": intervention_valid,
            "resume_checkpoints_equal_except_first_moment": (
                checkpoints_equal_except_moment
            ),
            "source_model_state_sha256": next(iter(model_digests)),
            "scales": scale_rows,
        })
        all_baselines_reproduce &= baseline_reproduces
        all_initial_energies_equal &= initial_equal
        all_interventions_valid &= intervention_valid
    scale_summary = [aggregate[value] for value in block_sizes]
    inference_valid = (
        all_baselines_reproduce
        and all_initial_energies_equal
        and all_interventions_valid
    )
    rejected_scales = [
        row["block_size"] for row in scale_summary if row["closure_rejected"]
    ] if inference_valid else []
    payload = {
        "schema_version": RESULT_SCHEMA,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": commit,
        "analysis_sources": sources,
        "input_run_manifest": {
            "path": "run-manifest.json",
            "sha256": file_sha256(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
            "record_sha256": manifest["record_sha256"],
        },
        "probe_id": manifest["probe_id"],
        "source_run": manifest["source_run"],
        "source_step": manifest["source_step"],
        "block_sizes": block_sizes,
        "intervention": manifest["intervention"],
        "trajectories": trajectories,
        "scale_summary": scale_summary,
        "gpu_hours": float(manifest["gpu_hours"]),
        "checks": {
            "baseline_bitwise_reproducibility": all_baselines_reproduce,
            "initial_energy_vector_bitwise_equality": all_initial_energies_equal,
            "moment_only_intervention": all_interventions_valid,
            "inference_valid": inference_valid,
        },
        "conclusion": {
            "rejected_scales": rejected_scales,
            "energy_vector_deterministic_closure_rejected_on_intervention_domain": (
                len(rejected_scales) > 0
            ),
            "global_reachable_training_manifold_closure_decided": False,
            "complete_state_kernel_remains_required": True,
        },
    }
    return seal_record(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    commit, sources = committed_identity()
    result = analyze(Path(arguments.run_root).resolve(), commit=commit, sources=sources)
    output = Path(arguments.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "status": "complete",
        "record_sha256": result["record_sha256"],
        "rejected_scales": result["conclusion"]["rejected_scales"],
        "gpu_hours": result["gpu_hours"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
