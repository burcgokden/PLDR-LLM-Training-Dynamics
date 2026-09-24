#!/usr/bin/env python3
"""Verify source-remeasured PLDR fiber-coordinate closure experiments."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from row_rgmap.analysis import (  # noqa: E402
    canonical_json_bytes,
    file_sha256,
    seal_record,
    verify_record_seal,
)
from row_rgmap.closure_contract import (  # noqa: E402
    BRANCH_SPECS,
    decision_status,
    validate_serialized_branch_specification,
)
from row_rgmap.provenance_v3 import git_blob_descriptor  # noqa: E402


RUN_SCHEMA = "pldr-row-rg-energy-closure-probe-run-v3"
RESULT_SCHEMA = "pldr-row-rg-energy-closure-analysis-v4"
PREDECESSOR_RUN_SCHEMA = "pldr-row-rg-energy-closure-probe-run-v2"
PREDECESSOR_RESULT_SCHEMA = "pldr-row-rg-energy-closure-analysis-v2"
PRODUCER_SCHEMA = "pldr-row-rg-energy-producer-metadata-v3"
CHECKPOINT_SCHEMA = "pldr-training-checkpoint-v3"
SOURCE_PATHS = (
    "scripts/analyze_energy_closure_probe_v3.py",
    "src/row_rgmap/analysis.py",
    "src/row_rgmap/closure_contract.py",
    "src/row_rgmap/provenance_v3.py",
)
ROUTE_IDS = ("producing-gpu", "other-gpu", "cpu")
RETAINED_BRANCHES = (
    "baseline-a",
    "baseline-b",
    "first-moment-0.999",
    "first-moment-0.9",
    "zero-first-moment",
    "zero-second-moment",
    "first-moment-one-ulp",
    "layernorm-bias-shift",
)
GAUGE_BRANCHES = tuple(
    branch
    for branch, specification in BRANCH_SPECS.items()
    if specification["operation"] == "sign-gauge"
)
GAUGE_COMPARISON_SPECS = {
    "full-moment-only-vs-parameter-only": (
        "sign-gauge-moments-only",
        "sign-gauge-params-only",
    ),
    "layer0-consistent-vs-baseline": (
        "sign-gauge-layer0-consistent",
        "baseline-a",
    ),
    "layer0-parameter-only-vs-baseline": (
        "sign-gauge-layer0-params-only",
        "baseline-a",
    ),
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


def optimizer_states_by_model_name(
    checkpoint: dict[str, Any],
) -> list[tuple[str, Any, dict[str, Any]]]:
    names = list(checkpoint["model"])
    parameter_ids = [
        parameter_id
        for group in checkpoint["optimizer"]["param_groups"]
        for parameter_id in group["params"]
    ]
    states = checkpoint["optimizer"]["state"]
    if (
        len(parameter_ids) != len(names)
        or len(set(parameter_ids)) != len(names)
        or set(states) != set(parameter_ids)
    ):
        raise ValueError("optimizer/model parameter registry is not bijective")
    rows = []
    for name, parameter_id in zip(names, parameter_ids):
        state = states[parameter_id]
        if (
            tuple(state["exp_avg"].shape) != tuple(checkpoint["model"][name].shape)
            or tuple(state["exp_avg_sq"].shape)
            != tuple(checkpoint["model"][name].shape)
        ):
            raise ValueError("optimizer moments do not match the model parameter")
        rows.append((name, parameter_id, state))
    return rows


def is_sign_gauge_parameter(name: str) -> bool:
    return bool(
        name.endswith(".mha1.layernorm1.weight")
        or name.endswith(".mha1.layernorm1.bias")
        or (
            ".mha1.reslayerAs." in name
            and ".denseAs." in name
            and (
                name.endswith("gluw1.weight")
                or name.endswith("gluw2.weight")
                or name.endswith("gluw3.weight")
                or name.endswith("gluw3.bias")
            )
        )
        or name.endswith(".mha1.reslayerAs.0.layernormA.bias")
        or name.endswith(".mha1.reslayerAs.1.layernormA.weight")
    )


_GAUGE_LAYER = re.compile(r"(?:^|\.)decoder\.dec_layers\.(\d+)\.mha1\.")


def sign_gauge_layer(name: str) -> int:
    match = _GAUGE_LAYER.search(name)
    if match is None:
        raise ValueError(f"sign-gauge tensor has no layer coordinate: {name}")
    return int(match.group(1))


def measured_sign_gauge_support(
    checkpoint: dict[str, Any], layers: list[int] | None = None
) -> dict[str, Any]:
    """Derive the sign-action support from the checkpoint, including layers."""

    selected_layers = None if layers is None else set(layers)
    names = sorted(
        name
        for name in checkpoint["model"]
        if is_sign_gauge_parameter(name)
        and (selected_layers is None or sign_gauge_layer(name) in selected_layers)
    )
    if not names:
        raise ValueError("sign-gauge support is empty")
    layers: dict[str, dict[str, int]] = {}
    for name in names:
        layer = str(sign_gauge_layer(name))
        row = layers.setdefault(layer, {"tensor_count": 0, "element_count": 0})
        row["tensor_count"] += 1
        row["element_count"] += int(checkpoint["model"][name].numel())
    return {
        "tensor_count": len(names),
        "element_count": sum(int(checkpoint["model"][name].numel()) for name in names),
        "layers": {key: layers[key] for key in sorted(layers, key=int)},
        "model_keys_sha256": hashlib.sha256(canonical_json_bytes(names)).hexdigest(),
    }


def serialized_gauge_support_matches(
    declared: dict[str, Any],
    measured: dict[str, Any],
    expected_names: list[str],
) -> bool:
    """Map serialized intervention names to independently measured support names."""

    return bool(
        declared.get("changed_tensor_count") == measured.get("tensor_count")
        and declared.get("changed_element_count") == measured.get("element_count")
        and declared.get("model_keys_sha256") == measured.get("model_keys_sha256")
        and declared.get("model_keys") == expected_names
    )


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
    elif operation == "scale-second-moment":
        allowed = lambda path: (
            path.startswith("optimizer.state.") and path.endswith(".exp_avg_sq")
        )
    elif operation == "final-layernorm-bias-shift":
        allowed = lambda path: (
            path.startswith("model.decoder.dec_layers.")
            and path.endswith(".mha1.reslayerAs.1.layernormA.bias")
        )
    elif operation == "sign-gauge":
        allowed = lambda path: (
            path.startswith("model.")
            and specification["transform_parameters"]
            and is_sign_gauge_parameter(path.removeprefix("model."))
            and sign_gauge_layer(path.removeprefix("model."))
            in specification["layers"]
        ) or bool(
            specification["co_transform_first_moment"]
            and path.startswith("optimizer.state.")
            and path.endswith(".exp_avg")
        )
    else:
        raise ValueError(f"unknown branch operation: {operation}")
    if not paths or any(not allowed(path) for path in paths):
        raise ValueError(f"checkpoint has undeclared intervention changes: {branch}")

    if operation in {"scale-first-moment", "scale-second-moment"}:
        field = "exp_avg" if operation == "scale-first-moment" else "exp_avg_sq"
        factor = float(specification["factor"])
        count = 0
        for key, base_state in baseline["optimizer"]["state"].items():
            if field not in base_state:
                continue
            expected = base_state[field] * factor
            if not torch.equal(checkpoint["optimizer"]["state"][key][field], expected):
                raise ValueError(f"optimizer-moment dose does not replay: {branch}")
            count += 1
        if count == 0:
            raise ValueError("optimizer-moment intervention has no tensors")
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
        if torch.equal(before, expected):
            expected = torch.nextafter(before, torch.full_like(before, float("-inf")))
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
    elif operation == "sign-gauge":
        baseline_rows = {
            name: (parameter_id, state)
            for name, parameter_id, state in optimizer_states_by_model_name(baseline)
        }
        checkpoint_rows = {
            name: (parameter_id, state)
            for name, parameter_id, state in optimizer_states_by_model_name(checkpoint)
        }
        selected = {
            name
            for name in baseline["model"]
            if is_sign_gauge_parameter(name)
            and sign_gauge_layer(name) in specification["layers"]
        }
        if not selected:
            raise ValueError("sign-gauge action has empty measured support")
        transform_parameters = bool(specification["transform_parameters"])
        co_transform = bool(specification["co_transform_first_moment"])
        for name, baseline_value in baseline["model"].items():
            expected = (
                -baseline_value
                if transform_parameters and name in selected
                else baseline_value
            )
            if not torch.equal(checkpoint["model"][name], expected):
                raise ValueError(f"sign-gauge model action does not replay: {name}")
            base_parameter_id, base_state = baseline_rows[name]
            current_parameter_id, current_state = checkpoint_rows[name]
            if base_parameter_id != current_parameter_id:
                raise ValueError("sign-gauge optimizer binding changed")
            expected_first = (
                -base_state["exp_avg"]
                if co_transform and name in selected
                else base_state["exp_avg"]
            )
            if (
                not torch.equal(current_state["exp_avg"], expected_first)
                or not torch.equal(current_state["exp_avg_sq"], base_state["exp_avg_sq"])
            ):
                raise ValueError(f"sign-gauge optimizer action does not replay: {name}")
    return sorted(paths)


def verify_output_checkpoint_inventory(
    run_root: Path,
    row: dict[str, Any],
    metadata: dict[str, Any],
    trajectory_id: str,
    branch: str,
    expected_steps: set[int],
) -> tuple[int, int]:
    branch_root = (run_root / "trajectories" / trajectory_id / branch).resolve()
    checkpoint_root = (branch_root / "checkpoints").resolve()
    descriptors = row.get("output_checkpoints")
    metadata_rows = metadata.get("checkpoints")
    if not isinstance(descriptors, list) or not isinstance(metadata_rows, list):
        raise ValueError("checkpoint inventories are missing")
    if any(
        not isinstance(item, dict)
        or set(item)
        != {"trajectory_id", "branch", "step", "path", "sha256", "size_bytes"}
        for item in descriptors
    ):
        raise ValueError("run-manifest checkpoint descriptor is malformed")
    if any(
        not isinstance(item, dict)
        or set(item) != {"step", "path", "sha256", "size_bytes"}
        for item in metadata_rows
    ):
        raise ValueError("producer checkpoint descriptor is malformed")
    resolved_manifest = {}
    for item in descriptors:
        if (
            item["trajectory_id"] != trajectory_id
            or item["branch"] != branch
            or type(item["step"]) is not int
            or item["step"] in resolved_manifest
        ):
            raise ValueError("run-manifest checkpoint identity is ambiguous")
        path = resolve_descriptor(run_root, item)
        try:
            path.relative_to(checkpoint_root)
        except ValueError as error:
            raise ValueError("run-manifest checkpoint escapes its branch") from error
        resolved_manifest[item["step"]] = (path, item)
    resolved_metadata = {}
    for item in metadata_rows:
        relative = Path(item["path"])
        if (
            type(item["step"]) is not int
            or item["step"] in resolved_metadata
            or relative.is_absolute()
            or ".." in relative.parts
        ):
            raise ValueError("producer checkpoint identity is ambiguous")
        path = (branch_root / relative).resolve()
        try:
            path.relative_to(checkpoint_root)
        except ValueError as error:
            raise ValueError("producer checkpoint escapes its branch") from error
        if (
            not path.is_file()
            or path.stat().st_size != item["size_bytes"]
            or file_sha256(path) != item["sha256"]
        ):
            raise ValueError("producer checkpoint descriptor fails")
        resolved_metadata[item["step"]] = (path, item)
    actual = {path.resolve() for path in checkpoint_root.glob("*") if path.is_file()}
    if (
        set(resolved_manifest) != expected_steps
        or set(resolved_metadata) != expected_steps
        or actual != {value[0] for value in resolved_manifest.values()}
        or actual != {value[0] for value in resolved_metadata.values()}
    ):
        raise ValueError("checkpoint file set is not exactly bound")
    for step in expected_steps:
        manifest_path, manifest_item = resolved_manifest[step]
        metadata_path, metadata_item = resolved_metadata[step]
        if (
            manifest_path != metadata_path
            or manifest_item["sha256"] != metadata_item["sha256"]
            or manifest_item["size_bytes"] != metadata_item["size_bytes"]
        ):
            raise ValueError("checkpoint registries disagree")
    return len(actual), sum(path.stat().st_size for path in actual)


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
) -> dict[tuple[str, str, int], dict[str, Any]]:
    rows = {}
    for trajectory in predecessor_result["trajectories"]:
        for branch in trajectory["branches"]:
            for scale in branch["scales"]:
                rows[
                    (
                        trajectory["trajectory_id"],
                        branch["branch"],
                        int(scale["block_size"]),
                    )
                ] = scale
    return rows


def analyze(
    run_root: Path,
    data_root: Path,
    *,
    commit: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    run_root = run_root.resolve()
    data_root = data_root.resolve()
    try:
        run_root.relative_to(data_root)
    except ValueError as error:
        raise ValueError("closure run root must be inside the data root") from error
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
    if config.get("branches") != list(BRANCH_SPECS):
        raise ValueError("closure config branch registry is not frozen v3")
    dataset = manifest.get("source_remeasurement", {}).get("dataset")
    if (
        not isinstance(dataset, dict)
        or dataset.get("root_role") != "refinedweb_read_only_root"
        or not isinstance(dataset.get("relative_path"), str)
        or Path(dataset["relative_path"]).is_absolute()
        or ".." in Path(dataset["relative_path"]).parts
    ):
        raise ValueError("source dataset locator is not portable")

    trajectory_ids = [row["trajectory_id"] for row in config["trajectories"]]
    branch_ids = list(BRANCH_SPECS)
    source_config_record = json.loads(
        (source_root / "config-resolved.json").read_text(encoding="utf-8")
    )
    source_devices = {
        row["trajectory_id"]: row["device"]
        for row in source_config_record["trajectories"]
    }
    if any(
        row["trajectory_id"] not in source_devices
        or row["device"] != source_devices[row["trajectory_id"]]
        for row in config["trajectories"]
    ):
        raise ValueError("configured producing device differs from the source run")
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
        if set(route_records) != set(ROUTE_IDS) or len(route_records) != len(ROUTE_IDS):
            raise ValueError("original source route registry is incomplete")
        expected_route_devices = {
            "producing-gpu": f"cuda:{source_devices[trajectory_id]}",
            "other-gpu": f"cuda:{1 - source_devices[trajectory_id]}",
            "cpu": "cpu",
        }
        for route_id, route_record in route_records.items():
            timing = route_record.get("timing")
            if (
                route_record.get("device") != expected_route_devices[route_id]
                or not isinstance(timing, dict)
                or any(
                    not isinstance(value, (int, float)) or value < 0
                    for key, value in timing.items()
                    if key.endswith("_seconds")
                )
                or timing.get("total_route_elapsed_seconds")
                != route_record.get("wall_seconds")
            ):
                raise ValueError("source route device or timing record is invalid")
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
                    "device": route_records[route_id]["device"],
                    "device_type": route_records[route_id]["device_type"],
                    "timing": route_records[route_id]["timing"],
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
        if set(branch_records) != set(branch_ids) or len(branch_records) != len(branch_ids):
            raise ValueError("rewritten source route registry is incomplete")
        if any(
            row.get("device") != f"cuda:{source_devices[trajectory_id]}"
            or row.get("device_type") != "cuda"
            or not isinstance(row.get("timing"), dict)
            for row in branch_records.values()
        ):
            raise ValueError("rewritten sources were not measured on the producing GPU")
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
                "device": branch_records[branch]["device"],
                "device_type": branch_records[branch]["device_type"],
                "timing": branch_records[branch]["timing"],
            }

    import torch

    predecessor_rows = predecessor_scientific_rows(predecessor_result)
    trajectory_results = []
    all_baselines_reproduce = True
    all_rewrites_valid = True
    retained_rows_equal = True
    checkpoint_descriptor_count = 0
    checkpoint_total_size_bytes = 0
    raw_first_moments_preserved = True
    raw_second_moments_preserved = True
    gauge_support_rows: list[dict[str, Any]] = []
    full_consistent_successors_exact = True
    layer0_consistent_successors_exact = True
    full_parameter_only_successors_different = True
    layer0_parameter_only_successors_different = True
    gauge_comparison_aggregate: dict[str, dict[int, dict[str, Any]]] = {
        comparison_id: {
            int(block_size): {
                "comparison_id": comparison_id,
                "left_branch": branches[0],
                "right_branch": branches[1],
                "block_size": int(block_size),
                "trajectory_count": 0,
                "map_count": 0,
                "bitwise_different_map_count": 0,
                "maximum_absolute_energy_defect": 0.0,
                "maximum_symmetric_relative_energy_defect": 0.0,
            }
            for block_size in manifest["block_sizes"]
        }
        for comparison_id, branches in GAUGE_COMPARISON_SPECS.items()
    }
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
            validate_serialized_branch_specification(
                branch, row.get("branch_specification")
            )
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
                or row.get("device") != source_devices[trajectory_id]
            ):
                raise ValueError("closure producer metadata does not replay")
            count, size = verify_output_checkpoint_inventory(
                run_root,
                row,
                metadata,
                trajectory_id,
                branch,
                {0, int(manifest["final_step"])},
            )
            checkpoint_descriptor_count += count
            checkpoint_total_size_bytes += size
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
        gauge_support = measured_sign_gauge_support(baseline, [0, 1, 2])
        gauge_support_rows.append(
            {"trajectory_id": trajectory_id, **gauge_support}
        )
        for gauge_branch in GAUGE_BRANCHES:
            declared = branch_rows[gauge_branch]["branch_specification"]
            branch_support = measured_sign_gauge_support(
                baseline, BRANCH_SPECS[gauge_branch]["layers"]
            )
            expected_names = sorted(
                name
                for name in baseline["model"]
                if is_sign_gauge_parameter(name)
                and sign_gauge_layer(name) in BRANCH_SPECS[gauge_branch]["layers"]
            )
            if not serialized_gauge_support_matches(
                declared, branch_support, expected_names
            ):
                raise ValueError(
                    f"serialized gauge support does not replay: {gauge_branch}"
                )
        actual_paths = {}
        for branch in branch_ids:
            actual_paths[branch] = verify_checkpoint_intervention(
                baseline, checkpoints[branch], branch
            )
        parameter_only = checkpoints["sign-gauge-params-only"]
        if set(baseline["optimizer"]["state"]) != set(
            parameter_only["optimizer"]["state"]
        ):
            raise ValueError("parameter-only gauge changed the optimizer registry")
        raw_first_moments_preserved &= all(
            torch.equal(
                state["exp_avg"],
                parameter_only["optimizer"]["state"][key]["exp_avg"],
            )
            for key, state in baseline["optimizer"]["state"].items()
        )
        raw_second_moments_preserved &= all(
            torch.equal(
                state["exp_avg_sq"],
                parameter_only["optimizer"]["state"][key]["exp_avg_sq"],
            )
            for key, state in baseline["optimizer"]["state"].items()
        )

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
        source_first = moment_summary(source_checkpoint, "exp_avg")
        source_second = moment_summary(source_checkpoint, "exp_avg_sq")
        successor_protocol_digest = manifest["protocol_rewrite"][
            "successor_canonical_sha256"
        ]
        for branch in branch_ids:
            row = branch_rows[branch]
            checkpoint = checkpoints[branch]
            rewrite = row["checkpoint_rewrite"]
            assigned = rewrite.get("assigned_values")
            full_paths = sorted(differing_paths(source_checkpoint, checkpoint))
            rewrite_valid = bool(
                row["source_model_state_sha256"] == source_model_digest
                and row["first_moment_before"] == source_first
                and row["second_moment_before"] == source_second
                and rewrite.get("unexpected_leaf_paths") == []
                and rewrite.get("assignment_performed")
                == ["code_commit", "protocol_sha256"]
                and rewrite.get("differing_leaf_paths") == full_paths
                and isinstance(assigned, dict)
                and assigned.get("code_commit")
                == {
                    "source": source_checkpoint.get("code_commit"),
                    "successor": manifest["code_commit"],
                }
                and assigned.get("protocol_sha256")
                == {
                    "source": source_checkpoint.get("protocol_sha256"),
                    "successor": successor_protocol_digest,
                }
                and checkpoint.get("code_commit") == manifest["code_commit"]
                and checkpoint.get("protocol_sha256") == successor_protocol_digest
            )
            all_rewrites_valid &= rewrite_valid
            if not rewrite_valid:
                raise ValueError(f"checkpoint rewrite record does not replay: {branch}")

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
            registered_role = BRANCH_SPECS[branch]["role"]
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
                if branch == "sign-gauge-consistent":
                    full_consistent_successors_exact &= (
                        defect["bitwise_different_map_count"] == 0
                    )
                elif branch == "sign-gauge-layer0-consistent":
                    layer0_consistent_successors_exact &= (
                        defect["bitwise_different_map_count"] == 0
                    )
                elif branch == "sign-gauge-params-only":
                    full_parameter_only_successors_different &= (
                        defect["bitwise_different_map_count"] > 0
                    )
                elif branch == "sign-gauge-layer0-params-only":
                    layer0_parameter_only_successors_different &= (
                        defect["bitwise_different_map_count"] > 0
                    )
                if branch in RETAINED_BRANCHES:
                    prior = predecessor_rows[
                        (trajectory_id, branch, int(block_size))
                    ]
                    retained_rows_equal &= scale == prior
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
        trajectory_gauge_comparisons = []
        for comparison_id, (left_branch, right_branch) in (
            GAUGE_COMPARISON_SPECS.items()
        ):
            comparison_scales = []
            for block_size in manifest["block_sizes"]:
                target = int(manifest["source_step"]) + int(block_size)
                matches = np.flatnonzero(steps == target)
                if len(matches) != 1:
                    raise ValueError(f"gauge comparison target is missing: {target}")
                target_index = int(matches[0])
                defect = vector_defect(
                    loaded[right_branch][0][target_index],
                    loaded[left_branch][0][target_index],
                )
                scale = {"block_size": int(block_size), **defect}
                comparison_scales.append(scale)
                combined = gauge_comparison_aggregate[comparison_id][
                    int(block_size)
                ]
                combined["trajectory_count"] += 1
                combined["map_count"] += defect["map_count"]
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
            trajectory_gauge_comparisons.append(
                {
                    "comparison_id": comparison_id,
                    "left_branch": left_branch,
                    "right_branch": right_branch,
                    "scales": comparison_scales,
                }
            )
        trajectory_results.append(
            {
                "trajectory_id": trajectory_id,
                "map_count": int(len(map_ids)),
                "baseline_bitwise_reproducible": baseline_reproduces,
                "branches": branch_results,
                "gauge_comparisons": trajectory_gauge_comparisons,
            }
        )

    aggregate_rows = {
        branch: [
            aggregate[branch][int(block_size)]
            for block_size in manifest["block_sizes"]
        ]
        for branch in branch_ids
    }
    gauge_comparison_rows = {
        comparison_id: [
            rows[int(block_size)] for block_size in manifest["block_sizes"]
        ]
        for comparison_id, rows in gauge_comparison_aggregate.items()
    }
    expected_retained_pairs = {
        (trajectory_id, branch)
        for trajectory_id in trajectory_ids
        for branch in RETAINED_BRANCHES
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
    inventory_record = manifest.get("checkpoint_inventory")
    expected_checkpoint_count = len(trajectory_ids) * len(branch_ids) * 2
    checkpoint_inventory_valid = bool(
        isinstance(inventory_record, dict)
        and checkpoint_descriptor_count == expected_checkpoint_count
        and inventory_record.get("descriptor_count") == checkpoint_descriptor_count
        and inventory_record.get("total_size_bytes") == checkpoint_total_size_bytes
        and inventory_record.get("expected_steps_per_branch")
        == [0, int(manifest["final_step"])]
        and inventory_record.get("all_emitted_files_bound")
        == (checkpoint_descriptor_count == expected_checkpoint_count)
    )
    if not checkpoint_inventory_valid:
        raise ValueError("global checkpoint inventory summary does not replay")
    primary_source_exact = all(
        branch_source_rows[(trajectory_id, "zero-first-moment")]["bitwise_equal"]
        for trajectory_id in trajectory_ids
    )
    gauge_sources_exact = all(
        branch_source_rows[(trajectory_id, branch)]["bitwise_equal"]
        for trajectory_id in trajectory_ids
        for branch in GAUGE_BRANCHES
    )
    full_consistent_sources_exact = all(
        branch_source_rows[(trajectory_id, "sign-gauge-consistent")][
            "bitwise_equal"
        ]
        for trajectory_id in trajectory_ids
    )
    full_parameter_only_sources_exact = all(
        branch_source_rows[(trajectory_id, "sign-gauge-params-only")][
            "bitwise_equal"
        ]
        for trajectory_id in trajectory_ids
    )
    gauge_null_control_passed = bool(
        full_consistent_sources_exact
        and full_consistent_successors_exact
        and all_rewrites_valid
    )
    raw_augmented_source_equal = bool(
        full_parameter_only_sources_exact
        and raw_first_moments_preserved
        and raw_second_moments_preserved
    )
    if not gauge_support_rows:
        raise ValueError("gauge support registry is empty")
    canonical_gauge_support = {
        key: value
        for key, value in gauge_support_rows[0].items()
        if key != "trajectory_id"
    }
    if any(
        {key: value for key, value in row.items() if key != "trajectory_id"}
        != canonical_gauge_support
        for row in gauge_support_rows[1:]
    ):
        raise ValueError("gauge support differs across trajectories")

    gauge_decomposition_checks = {
        "full_moment_only_matches_parameter_only": all(
            row["bitwise_different_map_count"] == 0
            for row in gauge_comparison_rows[
                "full-moment-only-vs-parameter-only"
            ]
        ),
        "layer0_consistent_matches_baseline": all(
            row["bitwise_different_map_count"] == 0
            for row in gauge_comparison_rows[
                "layer0-consistent-vs-baseline"
            ]
        ),
        "layer0_parameter_only_differs_everywhere": all(
            row["bitwise_different_map_count"] == row["map_count"]
            for row in gauge_comparison_rows[
                "layer0-parameter-only-vs-baseline"
            ]
        ),
    }

    structural_checks = {
        "producing_gpu_source_remeasurement_bitwise": producing_exact,
        "other_gpu_source_remeasurement_bitwise": other_gpu_exact,
        "baseline_bitwise_reproducibility": all_baselines_reproduce,
        "checkpoint_rewrites_validated": all_rewrites_valid,
        "all_emitted_checkpoints_bound": checkpoint_inventory_valid,
        "predecessor_scientific_leaves_retained": predecessor_retained,
        "branch_specifications_match_registry": True,
        "gauge_support_derived_from_checkpoints": True,
    }
    structural_valid = all(structural_checks.values())
    energy_vector_eligible = bool(structural_valid and primary_source_exact)
    raw_moment_eligible = bool(
        structural_valid
        and raw_augmented_source_equal
        and gauge_null_control_passed
    )

    primary_summary = aggregate_rows["zero-first-moment"]
    rejected_scales = (
        [row["block_size"] for row in primary_summary if row["closure_rejected"]]
        if energy_vector_eligible
        else []
    )
    energy_vector_status = decision_status(
        eligible=energy_vector_eligible, rejected=bool(rejected_scales)
    )
    raw_successor_difference_observed = any(
        row["bitwise_different_map_count"] > 0
        for row in aggregate_rows["sign-gauge-params-only"]
    )
    raw_moment_status = decision_status(
        eligible=raw_moment_eligible,
        rejected=raw_successor_difference_observed,
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
            "anchor": "data_root",
            "path": manifest_path.relative_to(data_root).as_posix(),
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
            "measurement_scope": {
                "source_originals": "producing GPU, other GPU, and CPU",
                "rewritten_branch_copies": "producing GPU only",
            },
            "dataset_locator": dataset,
            "original_checkpoint_routes": route_rows,
            "rewritten_branch_routes": [
                branch_source_rows[(trajectory_id, branch)]
                for trajectory_id in trajectory_ids
                for branch in branch_ids
            ],
        },
        "trajectories": trajectory_results,
        "branch_scale_summary": aggregate_rows,
        "checkpoint_inventory": {
            "descriptor_count": checkpoint_descriptor_count,
            "total_size_bytes": checkpoint_total_size_bytes,
            "expected_descriptor_count": expected_checkpoint_count,
            "all_emitted_files_bound": checkpoint_inventory_valid,
        },
        "gauge_decomposition": {
            "group": "(Z/2Z)^3",
            "registered_branches": [
                "baseline-a",
                "sign-gauge-consistent",
                "sign-gauge-params-only",
                "sign-gauge-moments-only",
                "sign-gauge-layer0-consistent",
                "sign-gauge-layer0-params-only",
            ],
            "comparisons": gauge_comparison_rows,
            "checks": gauge_decomposition_checks,
        },
        "gauge_control": {
            "support_tensor_count": canonical_gauge_support["tensor_count"],
            "support_element_count": canonical_gauge_support["element_count"],
            "support_by_layer": canonical_gauge_support["layers"],
            "support_model_keys_sha256": canonical_gauge_support[
                "model_keys_sha256"
            ],
            "trajectory_support": gauge_support_rows,
            "source_energy_bitwise_equal": gauge_sources_exact,
            "full_consistent_source_energy_bitwise_equal": (
                full_consistent_sources_exact
            ),
            "full_parameter_only_source_energy_bitwise_equal": (
                full_parameter_only_sources_exact
            ),
            "consistent_action_successors_bitwise_equal": (
                full_consistent_successors_exact
                and layer0_consistent_successors_exact
            ),
            "full_consistent_action_successors_bitwise_equal": (
                full_consistent_successors_exact
            ),
            "layer0_consistent_action_successors_bitwise_equal": (
                layer0_consistent_successors_exact
            ),
            "consistent_action_null_control_passed": gauge_null_control_passed,
            "parameter_only_raw_first_moments_equal": raw_first_moments_preserved,
            "parameter_only_raw_second_moments_equal": raw_second_moments_preserved,
            "parameter_only_successors_different_at_every_tested_scale": (
                full_parameter_only_successors_different
                and layer0_parameter_only_successors_different
            ),
            "full_parameter_only_successors_different_at_every_tested_scale": (
                full_parameter_only_successors_different
            ),
            "layer0_parameter_only_successors_different_at_every_tested_scale": (
                layer0_parameter_only_successors_different
            ),
            "raw_moment_augmentation_closure_status": raw_moment_status,
        },
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
        "validity": {
            "structural_valid": structural_valid,
            "structural_checks": structural_checks,
        },
        "eligibility": {
            "energy_vector_deterministic_closure": energy_vector_eligible,
            "raw_moment_augmentation_closure": raw_moment_eligible,
            "primary_source_fiber_bitwise": primary_source_exact,
            "raw_moment_source_state_equal": raw_augmented_source_equal,
            "gauge_orbit_null_control": gauge_null_control_passed,
        },
        "checks": structural_checks,
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
            "energy_vector_deterministic_closure_status": energy_vector_status,
            "raw_adamw_moment_augmentation_closure_status": raw_moment_status,
            "demonstrated_additional_fiber_coordinates": demonstrated_coordinates,
            "unique_or_minimal_markov_augmentation_status": "not decided",
            "optimizer_specific_scaling_law_status": "not established",
            "scope": "finite-horizon intervention experiment",
            "global_reachable_training_manifold_closure_status": "not decided",
            "complete_state_kernel_required_for_exact_general_theory": (
                energy_vector_status == "REJECTED"
            ),
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
