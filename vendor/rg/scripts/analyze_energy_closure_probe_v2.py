#!/usr/bin/env python3
"""Verify source-remeasured PLDR fiber-coordinate closure experiments."""

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

from row_rgmap.analysis import (  # noqa: E402
    file_sha256,
    seal_record,
    verify_record_seal,
)
from row_rgmap.provenance_v3 import git_blob_descriptor  # noqa: E402


RUN_SCHEMA = "pldr-row-rg-energy-closure-probe-run-v2"
RESULT_SCHEMA = "pldr-row-rg-energy-closure-analysis-v2"
PREDECESSOR_RUN_SCHEMA = "pldr-row-rg-energy-closure-probe-run-v1"
PREDECESSOR_RESULT_SCHEMA = "pldr-row-rg-energy-closure-analysis-v1"
PRODUCER_SCHEMA = "pldr-row-rg-energy-producer-metadata-v3"
CHECKPOINT_SCHEMA = "pldr-training-checkpoint-v3"
SOURCE_PATHS = (
    "scripts/analyze_energy_closure_probe_v2.py",
    "src/row_rgmap/analysis.py",
    "src/row_rgmap/provenance_v3.py",
)
ROUTE_IDS = ("producing-gpu", "other-gpu", "cpu")
BRANCH_SPECS = {
    "baseline-a": {"operation": "none", "registered_role": "repeatability-control"},
    "baseline-b": {"operation": "none", "registered_role": "repeatability-control"},
    "first-moment-0.999": {
        "operation": "scale-first-moment",
        "factor": 0.999,
        "registered_role": "same-fiber-closure-test",
    },
    "first-moment-0.9": {
        "operation": "scale-first-moment",
        "factor": 0.9,
        "registered_role": "same-fiber-closure-test",
    },
    "zero-first-moment": {
        "operation": "scale-first-moment",
        "factor": 0.0,
        "registered_role": "same-fiber-closure-test",
    },
    "zero-second-moment": {
        "operation": "zero-second-moment",
        "registered_role": "same-fiber-closure-test",
    },
    "first-moment-one-ulp": {
        "operation": "one-ulp-first-moment",
        "registered_role": "represented-resolution-control",
    },
    "layernorm-bias-shift": {
        "operation": "final-layernorm-bias-shift",
        "offset": 0.01,
        "registered_role": "model-coordinate-sensitivity-control",
    },
}
DEFECT_KEYS = (
    "map_count",
    "bitwise_different_map_count",
    "maximum_absolute_energy_defect",
    "maximum_symmetric_relative_energy_defect",
)


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
        raise RuntimeError("closure analysis requires a clean committed repository")
    sources = []
    for repository_path in SOURCE_PATHS:
        digest, size = git_blob_descriptor(ROOT, commit, repository_path)
        sources.append(
            {
                "repository_path": repository_path,
                "code_commit": commit,
                "sha256": digest,
                "size_bytes": size,
            }
        )
    return commit, sources


def verify_sources(record: dict[str, Any]) -> None:
    commit = record.get("code_commit")
    sources = record.get("analysis_sources")
    if not isinstance(commit, str) or not isinstance(sources, list) or not sources:
        raise ValueError("closure run source registry is incomplete")
    for source in sources:
        repository_path = source.get("repository_path")
        if not isinstance(repository_path, str):
            raise ValueError("closure run source path is not repository-relative")
        digest, size = git_blob_descriptor(ROOT, commit, repository_path)
        if (
            source.get("code_commit") != commit
            or source.get("sha256") != digest
            or source.get("size_bytes") != size
        ):
            raise ValueError(f"closure run source mismatch: {repository_path}")


def resolve_data_root_locator(
    data_root: Path, locator: Any, *, kind: str = "directory"
) -> Path:
    root = data_root.resolve()
    if (
        not isinstance(locator, dict)
        or set(locator) != {"anchor", "path"}
        or locator.get("anchor") != "data_root"
        or not isinstance(locator.get("path"), str)
        or not locator["path"]
    ):
        raise ValueError("data-root locator must contain only anchor=data_root and path")
    relative = Path(locator["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("data-root locator must be relative and cannot traverse parents")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("data-root locator escapes the data root") from error
    if kind == "directory" and not resolved.is_dir():
        raise ValueError(f"data-root directory does not resolve: {relative}")
    if kind == "file" and not resolved.is_file():
        raise ValueError(f"data-root file does not resolve: {relative}")
    return resolved


def resolve_descriptor(root: Path, descriptor: Any) -> Path:
    if not isinstance(descriptor, dict):
        raise ValueError("closure artifact descriptor is not an object")
    relative = descriptor.get("path")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ValueError("closure artifact path is not a safe relative path")
    resolved_root = root.resolve()
    path = (resolved_root / relative).resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("closure artifact path escapes its root") from error
    if (
        not path.is_file()
        or path.stat().st_size != descriptor.get("size_bytes")
        or file_sha256(path) != descriptor.get("sha256")
    ):
        raise ValueError(f"closure artifact descriptor fails: {relative}")
    return path


def resolve_anchored_descriptor(data_root: Path, descriptor: Any) -> Path:
    if not isinstance(descriptor, dict) or descriptor.get("anchor") != "data_root":
        raise ValueError("external descriptor is not anchored to data_root")
    local = {key: value for key, value in descriptor.items() if key != "anchor"}
    return resolve_descriptor(data_root, local)


def _values_equal(left: Any, right: Any) -> bool:
    import torch

    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left, right)
        )
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and np.array_equal(left, right)
        )
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _values_equal(a, b) for a, b in zip(left, right)
        )
    return bool(left == right)


def differing_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    if type(left) is not type(right):
        return [prefix or "<root>"]
    if isinstance(left, dict):
        paths = []
        keys = sorted(
            set(left).union(right),
            key=lambda value: (type(value).__name__, repr(value)),
        )
        for key in keys:
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(differing_paths(left[key], right[key], child))
        return paths
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return [prefix or "<root>"]
        paths = []
        for index, (a, b) in enumerate(zip(left, right)):
            child = f"{prefix}.{index}" if prefix else str(index)
            paths.extend(differing_paths(a, b, child))
        return paths
    return [] if _values_equal(left, right) else [prefix or "<root>"]


def _digest_field(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


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


def moment_summary(checkpoint: dict[str, Any], key: str) -> dict[str, Any]:
    element_count = 0
    square_sum = 0.0
    nonzero_count = 0
    tensor_count = 0
    for state in checkpoint["optimizer"]["state"].values():
        moment = state.get(key)
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


def vector_defect(reference: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    reference = np.ascontiguousarray(reference, dtype=np.float64)
    observed = np.ascontiguousarray(observed, dtype=np.float64)
    if reference.shape != observed.shape or reference.ndim != 1:
        raise ValueError("closure vectors must have equal one-dimensional shape")
    absolute = np.abs(reference - observed)
    denominator = np.abs(reference) + np.abs(observed)
    symmetric = np.zeros_like(absolute)
    nonzero = denominator > 0
    symmetric[nonzero] = 2.0 * absolute[nonzero] / denominator[nonzero]
    return {
        "map_count": int(reference.size),
        "bitwise_different_map_count": int(
            np.sum(reference.view(np.uint64) != observed.view(np.uint64))
        ),
        "maximum_absolute_energy_defect": float(absolute.max(initial=0.0)),
        "maximum_symmetric_relative_energy_defect": float(
            symmetric.max(initial=0.0)
        ),
        "l2_energy_defect": float(np.linalg.norm(absolute)),
    }


def assert_recorded_defect(actual: dict[str, Any], recorded: dict[str, Any]) -> None:
    if any(actual[key] != recorded.get(key) for key in DEFECT_KEYS):
        raise ValueError("recorded source-remeasurement defect does not replay")


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
        or np.any(energies < 0)
        or len(steps) == 0
        or not np.array_equal(steps, np.arange(steps[-1] + 1))
    ):
        raise ValueError("closure energy archive is malformed")
    return energies[0], steps, map_ids


def load_measurements(path: Path) -> dict[str, np.ndarray]:
    expected = {
        "trajectory_ids",
        "route_ids",
        "branch_ids",
        "map_ids",
        "restored_recorder_energy",
        "original_checkpoint_remeasured_energy",
        "rewritten_branch_remeasured_energy",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise ValueError("source-remeasurement archive has an unexpected contract")
        result = {name: np.asarray(archive[name]) for name in expected}
    result["trajectory_ids"] = result["trajectory_ids"].astype(str)
    result["route_ids"] = result["route_ids"].astype(str)
    result["branch_ids"] = result["branch_ids"].astype(str)
    result["map_ids"] = result["map_ids"].astype(str)
    for name in (
        "restored_recorder_energy",
        "original_checkpoint_remeasured_energy",
        "rewritten_branch_remeasured_energy",
    ):
        result[name] = np.asarray(result[name], dtype=np.float64)
        if not np.all(np.isfinite(result[name])) or np.any(result[name] < 0):
            raise ValueError("source-remeasurement energies are invalid")
    return result


def verify_checkpoint_intervention(
    baseline: dict[str, Any], checkpoint: dict[str, Any], branch: str
) -> list[str]:
    import torch

    specification = BRANCH_SPECS[branch]
    operation = specification["operation"]
    paths = differing_paths(baseline, checkpoint)
    if operation == "none":
        if paths:
            raise ValueError(f"baseline checkpoint changed: {branch}")
        return paths

    if operation in {"scale-first-moment", "one-ulp-first-moment"}:
        allowed = lambda path: (
            path.startswith("optimizer.state.") and path.endswith(".exp_avg")
        )
    elif operation == "zero-second-moment":
        allowed = lambda path: (
            path.startswith("optimizer.state.") and path.endswith(".exp_avg_sq")
        )
    elif operation == "final-layernorm-bias-shift":
        allowed = lambda path: (
            path.startswith("model.decoder.dec_layers.")
            and path.endswith(".mha1.reslayerAs.1.layernormA.bias")
        )
    else:
        raise ValueError(f"unknown branch operation: {operation}")
    if not paths or any(not allowed(path) for path in paths):
        raise ValueError(f"checkpoint has undeclared intervention changes: {branch}")

    if operation == "scale-first-moment":
        factor = float(specification["factor"])
        count = 0
        for key, base_state in baseline["optimizer"]["state"].items():
            if "exp_avg" not in base_state:
                continue
            expected = base_state["exp_avg"] * factor
            if not torch.equal(checkpoint["optimizer"]["state"][key]["exp_avg"], expected):
                raise ValueError(f"first-moment dose does not replay: {branch}")
            count += 1
        if count == 0:
            raise ValueError("first-moment intervention has no tensors")
    elif operation == "zero-second-moment":
        moments = [
            state["exp_avg_sq"]
            for state in checkpoint["optimizer"]["state"].values()
            if "exp_avg_sq" in state
        ]
        if not moments or any(torch.count_nonzero(value) for value in moments):
            raise ValueError("second-moment zeroing does not replay")
    elif operation == "one-ulp-first-moment":
        changed = []
        for key, base_state in baseline["optimizer"]["state"].items():
            if "exp_avg" not in base_state:
                continue
            before = base_state["exp_avg"]
            after = checkpoint["optimizer"]["state"][key]["exp_avg"]
            locations = torch.nonzero(before != after, as_tuple=False)
            for location in locations:
                index = tuple(int(value) for value in location)
                changed.append((before[index], after[index]))
        if len(changed) != 1:
            raise ValueError("one-ulp intervention did not change exactly one element")
        before, after = changed[0]
        expected = torch.nextafter(before, torch.full_like(before, float("inf")))
        if not torch.equal(after, expected):
            raise ValueError("one-ulp intervention is not the registered nextafter")
    elif operation == "final-layernorm-bias-shift":
        expected_keys = sorted(
            name
            for name in baseline["model"]
            if name.endswith(".mha1.reslayerAs.1.layernormA.bias")
        )
        if len(expected_keys) != 3 or len(paths) != 3:
            raise ValueError("LayerNorm-bias intervention has the wrong support")
        offset = float(specification["offset"])
        for name in expected_keys:
            expected = baseline["model"][name] + offset
            if not torch.equal(checkpoint["model"][name], expected):
                raise ValueError("LayerNorm-bias shift does not replay")
    return paths


def verify_external_inputs(
    manifest: dict[str, Any], run_root: Path, data_root: Path
) -> tuple[dict[str, Any], Path, dict[str, Any], dict[str, Any], Path]:
    config_path = resolve_descriptor(run_root, manifest["inputs"]["config"])
    resolve_descriptor(run_root, manifest["inputs"]["protocol"])
    measurement_path = resolve_descriptor(
        run_root, manifest["inputs"]["source_remeasurement"]
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("source_run") != manifest["source_run"]["locator"]
        or config.get("predecessor_closure") != manifest["predecessor"]["locator"]
    ):
        raise ValueError("closure manifest locators differ from the frozen config")

    source_root = resolve_data_root_locator(data_root, config["source_run"])
    source_config = source_root / "config-resolved.json"
    source_protocol = source_root / "protocol-resolved.json"
    if (
        file_sha256(source_config) != manifest["source_run"]["config_sha256"]
        or file_sha256(source_protocol) != manifest["source_run"]["protocol_sha256"]
        or json.loads(source_config.read_text(encoding="utf-8")).get("run_id")
        != manifest["source_run"]["run_id"]
    ):
        raise ValueError("closure source-run identity changed")
    for descriptor in manifest["source_run"]["checkpoints"]:
        checkpoint = resolve_descriptor(
            source_root,
            {
                "path": descriptor["path"],
                "sha256": descriptor["sha256"],
                "size_bytes": descriptor["size_bytes"],
            },
        )
        metadata = resolve_descriptor(
            source_root,
            {
                "path": descriptor["metadata_path"],
                "sha256": descriptor["metadata_sha256"],
                "size_bytes": descriptor["metadata_size_bytes"],
            },
        )
        if not checkpoint.is_file() or not metadata.is_file():
            raise ValueError("closure source descriptor does not resolve")

    predecessor_root = resolve_data_root_locator(
        data_root, config["predecessor_closure"]
    )
    predecessor_manifest_path = resolve_anchored_descriptor(
        data_root, manifest["predecessor"]["run_manifest"]
    )
    predecessor_result_path = resolve_anchored_descriptor(
        data_root, manifest["predecessor"]["result"]
    )
    if (
        predecessor_manifest_path.parent != predecessor_root
        or predecessor_result_path.parent != predecessor_root
    ):
        raise ValueError("predecessor descriptors differ from the registered root")
    predecessor_manifest = json.loads(
        predecessor_manifest_path.read_text(encoding="utf-8")
    )
    predecessor_result = json.loads(
        predecessor_result_path.read_text(encoding="utf-8")
    )
    verify_record_seal(predecessor_manifest)
    verify_record_seal(predecessor_result)
    if (
        predecessor_manifest.get("schema_version") != PREDECESSOR_RUN_SCHEMA
        or predecessor_result.get("schema_version") != PREDECESSOR_RESULT_SCHEMA
    ):
        raise ValueError("predecessor closure schema mismatch")
    if (
        predecessor_manifest["record_sha256"]
        != manifest["predecessor"]["run_manifest"].get("record_sha256")
        or predecessor_result["record_sha256"]
        != manifest["predecessor"]["result"].get("record_sha256")
    ):
        raise ValueError("predecessor canonical record digest does not replay")
    return (
        config,
        source_root,
        predecessor_manifest,
        predecessor_result,
        measurement_path,
    )


def predecessor_scientific_rows(
    predecessor_result: dict[str, Any],
) -> dict[tuple[str, int], dict[str, Any]]:
    rows = {}
    for trajectory in predecessor_result["trajectories"]:
        for scale in trajectory["scales"]:
            rows[(trajectory["trajectory_id"], int(scale["block_size"]))] = scale
    return rows


def analyze(
    run_root: Path,
    data_root: Path,
    *,
    commit: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest_path = run_root / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_record_seal(manifest)
    if manifest.get("schema_version") != RUN_SCHEMA:
        raise ValueError("unknown closure run schema")
    verify_sources(manifest)
    (
        config,
        source_root,
        predecessor_manifest,
        predecessor_result,
        measurement_path,
    ) = verify_external_inputs(manifest, run_root, data_root)
    measurements = load_measurements(measurement_path)

    trajectory_ids = [row["trajectory_id"] for row in config["trajectories"]]
    branch_ids = list(BRANCH_SPECS)
    map_count = len(measurements["map_ids"])
    expected_shapes = {
        "restored_recorder_energy": (len(trajectory_ids), map_count),
        "original_checkpoint_remeasured_energy": (
            len(trajectory_ids),
            len(ROUTE_IDS),
            map_count,
        ),
        "rewritten_branch_remeasured_energy": (
            len(trajectory_ids),
            len(branch_ids),
            map_count,
        ),
    }
    if (
        measurements["trajectory_ids"].tolist() != trajectory_ids
        or measurements["route_ids"].tolist() != list(ROUTE_IDS)
        or measurements["branch_ids"].tolist() != branch_ids
        or any(measurements[name].shape != shape for name, shape in expected_shapes.items())
        or manifest.get("branch_ids") != branch_ids
        or set(manifest.get("branches", {})) != set(trajectory_ids)
    ):
        raise ValueError("source-remeasurement registry or shape is malformed")

    recorded_source = {
        row["trajectory_id"]: row
        for row in manifest["source_remeasurement"]["trajectories"]
    }
    route_rows = []
    branch_source_rows: dict[tuple[str, str], dict[str, Any]] = {}
    producing_exact = True
    other_gpu_exact = True
    for trajectory_index, trajectory_id in enumerate(trajectory_ids):
        restored = measurements["restored_recorder_energy"][trajectory_index]
        recorded = recorded_source[trajectory_id]
        route_records = {
            row["route_id"]: row for row in recorded["original_checkpoint_routes"]
        }
        for route_index, route_id in enumerate(ROUTE_IDS):
            defect = vector_defect(
                restored,
                measurements["original_checkpoint_remeasured_energy"][
                    trajectory_index, route_index
                ],
            )
            assert_recorded_defect(defect, route_records[route_id])
            exact = defect["bitwise_different_map_count"] == 0
            route_rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "route_id": route_id,
                    **defect,
                    "bitwise_equal": exact,
                }
            )
            if route_id == "producing-gpu":
                producing_exact &= exact
            elif route_id == "other-gpu":
                other_gpu_exact &= exact

        branch_records = {
            row["branch"]: row
            for row in recorded["rewritten_branch_producing_routes"]
        }
        for branch_index, branch in enumerate(branch_ids):
            defect = vector_defect(
                restored,
                measurements["rewritten_branch_remeasured_energy"][
                    trajectory_index, branch_index
                ],
            )
            assert_recorded_defect(defect, branch_records[branch])
            branch_source_rows[(trajectory_id, branch)] = {
                "trajectory_id": trajectory_id,
                "branch": branch,
                **defect,
                "bitwise_equal": defect["bitwise_different_map_count"] == 0,
            }

    import torch

    predecessor_rows = predecessor_scientific_rows(predecessor_result)
    trajectory_results = []
    all_baselines_reproduce = True
    all_rewrites_valid = True
    retained_rows_equal = True
    aggregate: dict[str, dict[int, dict[str, Any]]] = {
        branch: {
            int(block_size): {
                "branch": branch,
                "block_size": int(block_size),
                "trajectory_count": 0,
                "bitwise_different_map_count": 0,
                "maximum_absolute_energy_defect": 0.0,
                "maximum_symmetric_relative_energy_defect": 0.0,
                "closure_rejected": False,
            }
            for block_size in manifest["block_sizes"]
        }
        for branch in branch_ids
    }

    source_descriptors = {
        (row["trajectory_id"], int(row["step"])): row
        for row in manifest["source_run"]["checkpoints"]
    }
    for trajectory_index, trajectory_id in enumerate(trajectory_ids):
        branch_rows = {
            row["branch"]: row for row in manifest["branches"][trajectory_id]
        }
        if set(branch_rows) != set(branch_ids):
            raise ValueError("closure branch registry is incomplete")
        loaded = {}
        checkpoints = {}
        for branch in branch_ids:
            row = branch_rows[branch]
            energy_path = resolve_descriptor(run_root, row["energies"])
            resolve_descriptor(run_root, row["metrics"])
            metadata_path = resolve_descriptor(run_root, row["metadata"])
            resolve_descriptor(run_root, row["producer_log"])
            checkpoint_path = resolve_descriptor(run_root, row["resume_checkpoint"])
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata.get("schema_version") != PRODUCER_SCHEMA
                or metadata.get("code_commit") != manifest["code_commit"]
                or metadata.get("trajectory_id") != trajectory_id
            ):
                raise ValueError("closure producer metadata does not replay")
            loaded[branch] = load_energy(energy_path)
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            if (
                checkpoint.get("schema_version") != CHECKPOINT_SCHEMA
                or checkpoint.get("step") != manifest["source_step"]
                or checkpoint.get("code_commit") != manifest["code_commit"]
            ):
                raise ValueError("rewritten resume-checkpoint identity is invalid")
            first = moment_summary(checkpoint, "exp_avg")
            second = moment_summary(checkpoint, "exp_avg_sq")
            model_digest = tensor_mapping_sha256(checkpoint["model"])
            if (
                first != row["first_moment_after"]
                or second != row["second_moment_after"]
                or model_digest != row["model_state_sha256"]
                or row["checkpoint_rewrite"]["unexpected_leaf_paths"] != []
                or row["checkpoint_rewrite"]["assignment_performed"]
                != ["code_commit", "protocol_sha256"]
            ):
                raise ValueError("rewritten checkpoint summary does not replay")
            checkpoints[branch] = checkpoint

        baseline = checkpoints["baseline-a"]
        actual_paths = {}
        for branch in branch_ids:
            actual_paths[branch] = verify_checkpoint_intervention(
                baseline, checkpoints[branch], branch
            )
        all_rewrites_valid &= True

        source_descriptor = source_descriptors[
            (trajectory_id, int(manifest["source_step"]))
        ]
        source_checkpoint_path = resolve_descriptor(
            source_root,
            {
                "path": source_descriptor["path"],
                "sha256": source_descriptor["sha256"],
                "size_bytes": source_descriptor["size_bytes"],
            },
        )
        source_checkpoint = torch.load(
            source_checkpoint_path, map_location="cpu", weights_only=False
        )
        source_model_digest = tensor_mapping_sha256(source_checkpoint["model"])
        if any(
            row["source_model_state_sha256"] != source_model_digest
            for row in branch_rows.values()
        ):
            raise ValueError("source model-state digest does not replay")

        baseline_energy, steps, map_ids = loaded["baseline-a"]
        baseline_b, steps_b, maps_b = loaded["baseline-b"]
        baseline_reproduces = (
            np.array_equal(steps, steps_b)
            and np.array_equal(map_ids, maps_b)
            and np.array_equal(baseline_energy, baseline_b)
        )
        all_baselines_reproduce &= baseline_reproduces
        source_matches = np.flatnonzero(steps == int(manifest["source_step"]))
        if len(source_matches) != 1:
            raise ValueError("closure source step is absent or duplicated")
        source_index = int(source_matches[0])
        restored = measurements["restored_recorder_energy"][trajectory_index]

        branch_results = []
        for branch in branch_ids:
            energy, branch_steps, branch_maps = loaded[branch]
            if (
                not np.array_equal(steps, branch_steps)
                or not np.array_equal(map_ids, branch_maps)
                or not np.array_equal(energy[source_index], restored)
            ):
                raise ValueError("closure branch registry or restored source row changed")
            source_gate = branch_source_rows[(trajectory_id, branch)]
            registered_role = BRANCH_SPECS[branch]["registered_role"]
            closure_eligible = (
                registered_role == "same-fiber-closure-test"
                or (
                    branch == "layernorm-bias-shift"
                    and source_gate["bitwise_equal"]
                )
            )
            effective_role = (
                "same-fiber-closure-test"
                if closure_eligible
                else registered_role
            )
            scales = []
            for block_size in manifest["block_sizes"]:
                target = int(manifest["source_step"]) + int(block_size)
                matches = np.flatnonzero(steps == target)
                if len(matches) != 1:
                    raise ValueError(f"closure target step is missing: {target}")
                defect = vector_defect(
                    baseline_energy[int(matches[0])],
                    energy[int(matches[0])],
                )
                rejected = bool(
                    closure_eligible
                    and source_gate["bitwise_equal"]
                    and defect["bitwise_different_map_count"] > 0
                )
                scale = {
                    "block_size": int(block_size),
                    **defect,
                    "closure_rejected": rejected,
                }
                scales.append(scale)
                combined = aggregate[branch][int(block_size)]
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
                combined["closure_rejected"] |= rejected
                if branch == "zero-first-moment":
                    prior = predecessor_rows[(trajectory_id, int(block_size))]
                    for key in (
                        "bitwise_different_map_count",
                        "maximum_absolute_energy_defect",
                        "maximum_symmetric_relative_energy_defect",
                        "l2_energy_defect",
                    ):
                        retained_rows_equal &= scale[key] == prior[key]
            branch_results.append(
                {
                    "branch": branch,
                    "registered_role": registered_role,
                    "effective_role": effective_role,
                    "source_gate": source_gate,
                    "checkpoint_differing_leaf_paths": actual_paths[branch],
                    "scales": scales,
                }
            )
        trajectory_results.append(
            {
                "trajectory_id": trajectory_id,
                "map_count": int(len(map_ids)),
                "baseline_bitwise_reproducible": baseline_reproduces,
                "branches": branch_results,
            }
        )

    aggregate_rows = {
        branch: [
            aggregate[branch][int(block_size)]
            for block_size in manifest["block_sizes"]
        ]
        for branch in branch_ids
    }
    expected_retained_pairs = {
        (trajectory_id, branch)
        for trajectory_id in trajectory_ids
        for branch in ("baseline-a", "baseline-b", "zero-first-moment")
    }
    comparison_rows = manifest["predecessor"].get("comparisons", [])
    observed_retained_pairs = {
        (row.get("trajectory_id"), row.get("branch"))
        for row in comparison_rows
    }
    predecessor_retained = (
        manifest["predecessor"].get("retained_scientific_leaves_bitwise_equal")
        is True
        and observed_retained_pairs == expected_retained_pairs
        and len(comparison_rows) == len(expected_retained_pairs)
        and all(
            row.get("energy_arrays_bitwise_equal") is True
            for row in comparison_rows
        )
        and retained_rows_equal
        and predecessor_manifest["record_sha256"]
        == manifest["predecessor"]["run_manifest"]["record_sha256"]
        and predecessor_result["record_sha256"]
        == manifest["predecessor"]["result"]["record_sha256"]
    )
    primary_source_exact = all(
        branch_source_rows[(trajectory_id, "zero-first-moment")]["bitwise_equal"]
        for trajectory_id in trajectory_ids
    )
    inference_valid = all(
        (
            producing_exact,
            other_gpu_exact,
            all_baselines_reproduce,
            all_rewrites_valid,
            predecessor_retained,
            primary_source_exact,
        )
    )
    primary_summary = aggregate_rows["zero-first-moment"]
    rejected_scales = (
        [row["block_size"] for row in primary_summary if row["closure_rejected"]]
        if inference_valid
        else []
    )
    demonstrated_coordinates = []
    for branch, label in (
        ("zero-first-moment", "AdamW first moments"),
        ("zero-second-moment", "AdamW second moments"),
        ("layernorm-bias-shift", "model coordinates"),
    ):
        if any(row["closure_rejected"] for row in aggregate_rows[branch]):
            demonstrated_coordinates.append(label)

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
        "source_step": manifest["source_step"],
        "block_sizes": manifest["block_sizes"],
        "trajectory_ids": trajectory_ids,
        "branch_ids": branch_ids,
        "source_remeasurement": {
            "decision_rule": (
                "bitwise equality of every stored float64 energy after an "
                "independent forward observation"
            ),
            "representation": (
                "row maps are float32; quotient energies are reduced and stored "
                "as float64"
            ),
            "original_checkpoint_routes": route_rows,
            "rewritten_branch_routes": [
                branch_source_rows[(trajectory_id, branch)]
                for trajectory_id in trajectory_ids
                for branch in branch_ids
            ],
        },
        "trajectories": trajectory_results,
        "branch_scale_summary": aggregate_rows,
        "predecessor_retention": {
            "energy_arrays_bitwise_equal": manifest["predecessor"][
                "retained_scientific_leaves_bitwise_equal"
            ],
            "derived_scientific_rows_equal": retained_rows_equal,
            "all_retained": predecessor_retained,
        },
        "gpu_time_accounting": manifest["gpu_time_accounting"],
        "gpu_seconds": float(manifest["gpu_seconds"]),
        "gpu_hours": float(manifest["gpu_hours"]),
        "checks": {
            "producing_gpu_source_remeasurement_bitwise": producing_exact,
            "other_gpu_source_remeasurement_bitwise": other_gpu_exact,
            "baseline_bitwise_reproducibility": all_baselines_reproduce,
            "checkpoint_rewrites_validated": all_rewrites_valid,
            "predecessor_scientific_leaves_retained": predecessor_retained,
            "primary_source_fiber_bitwise": primary_source_exact,
            "inference_valid": inference_valid,
        },
        "resolution_control": {
            "branch": "first-moment-one-ulp",
            "source_gate_bitwise": all(
                branch_source_rows[(trajectory_id, "first-moment-one-ulp")][
                    "bitwise_equal"
                ]
                for trajectory_id in trajectory_ids
            ),
            "successor_changed_map_counts": [
                row["bitwise_different_map_count"]
                for row in aggregate_rows["first-moment-one-ulp"]
            ],
            "interpretation": (
                "A represented one-ulp first-moment perturbation is tested at "
                "the float32-state and float64-energy observation resolution."
            ),
        },
        "conclusion": {
            "rejected_scales": rejected_scales,
            "energy_vector_deterministic_closure_rejected_on_intervention_domain": (
                inference_valid and bool(rejected_scales)
            ),
            "demonstrated_additional_fiber_coordinates": demonstrated_coordinates,
            "unique_or_minimal_markov_augmentation_decided": False,
            "optimizer_specific_scaling_law_established": False,
            "finite_horizon_sensitivity_only": True,
            "global_reachable_training_manifold_closure_decided": False,
            "complete_state_kernel_remains_required": True,
        },
    }
    return seal_record(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    commit, sources = committed_identity()
    result = analyze(
        Path(arguments.run_root).resolve(),
        Path(arguments.data_root).resolve(),
        commit=commit,
        sources=sources,
    )
    output = Path(arguments.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "record_sha256": result["record_sha256"],
                "rejected_scales": result["conclusion"]["rejected_scales"],
                "gpu_hours": result["gpu_hours"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
