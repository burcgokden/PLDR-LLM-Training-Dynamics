#!/usr/bin/env python3
"""Run portable, source-remeasured PLDR closure and sensitivity continuations."""

from __future__ import annotations
from companion_paths import configured_path

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
import time
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
from row_rgmap.closure_contract import (  # noqa: E402
    BRANCH_SPECS,
    branch_specification,
)
from row_rgmap.provenance_v3 import git_blob_descriptor  # noqa: E402


CONFIG_SCHEMA = "pldr-row-rg-energy-closure-probe-config-v3"
MANIFEST_SCHEMA = "pldr-row-rg-energy-closure-probe-run-v3"
PRODUCER_SCHEMA = "pldr-row-rg-energy-producer-metadata-v3"
CHECKPOINT_SCHEMA = "pldr-training-checkpoint-v3"
SOURCE_PATHS = (
    "scripts/run_energy_closure_probe_v3.py",
    "scripts/pldr_energy_producer_v3.py",
    "src/row_rgmap/analysis.py",
    "src/row_rgmap/closure_contract.py",
    "src/row_rgmap/provenance_v3.py",
    "src/row_rgmap/recorder.py",
)
ROUTE_IDS = ("producing-gpu", "other-gpu", "cpu")


def write_new_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def descriptor(path: Path, root: Path) -> dict[str, Any]:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("artifact descriptor escapes its root") from error
    return {
        "path": relative.as_posix(),
        "sha256": file_sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def anchored_descriptor(path: Path, data_root: Path) -> dict[str, Any]:
    return {"anchor": "data_root"} | descriptor(path, data_root)


def resolve_data_root_locator(
    data_root: Path, locator: Any, *, kind: str = "directory"
) -> Path:
    """Resolve a nonescaping data-root-relative file or directory locator."""

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


def committed_identity(config_path: Path) -> tuple[str, list[dict[str, Any]]]:
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
        sources.append(
            {
                "repository_path": repository_path,
                "code_commit": commit,
                "sha256": digest,
                "size_bytes": size,
            }
        )
    return commit, sources


def option(command: list[str], name: str) -> str:
    positions = [index for index, value in enumerate(command) if value == name]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError(f"producer command lacks unique option {name}")
    return command[positions[0] + 1]


def replace_option(command: list[str], name: str, value: str) -> None:
    positions = [index for index, item in enumerate(command) if item == name]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError(f"producer command lacks unique option {name}")
    command[positions[0] + 1] = value


def independent_probe_tokens(
    dataset: Any, start: int, count: int
) -> tuple[list[int], dict[str, Any]]:
    """Extract the fixed probe without calling the training producer's stream."""

    raw = bytearray()
    index = start
    while len(raw) < count:
        if index >= len(dataset):
            raise ValueError("dataset shard ended before the source probe")
        content = dataset[index]["content"]
        if not isinstance(content, str):
            raise ValueError(f"dataset row {index} has no string content")
        raw.extend(content.encode("utf-8", errors="strict"))
        raw.append(10)
        index += 1
    selected = bytes(raw[:count])
    return [value + 1 for value in selected], {
        "first_document": start,
        "last_document_exclusive": index,
        "document_count": index - start,
        "token_count": count,
        "byte_token_sha256": hashlib.sha256(selected).hexdigest(),
        "streaming": True,
        "maximum_buffer_policy": "one current UTF-8 document",
    }


def independent_row_centered_energy(value: Any) -> float:
    """Reimplement the registered CPU-float64 quotient-energy reduction."""

    matrix = value.detach().double().cpu().numpy()
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("source row map is not a finite matrix")
    if np.all(matrix == matrix[0:1, :]):
        return 0.0
    centered = matrix - np.mean(matrix, axis=0, keepdims=True, dtype=np.float64)
    energy = float(np.sum(centered * centered, dtype=np.float64))
    if not np.isfinite(energy) or energy < 0.0:
        raise ValueError("source row-map energy is invalid")
    return energy


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
    """Bind optimizer state identifiers to model parameter names by param-group order."""

    names = list(checkpoint["model"])
    parameter_ids = [
        parameter_id
        for group in checkpoint["optimizer"]["param_groups"]
        for parameter_id in group["params"]
    ]
    if len(parameter_ids) != len(names) or len(set(parameter_ids)) != len(names):
        raise ValueError("optimizer/model parameter registry is not bijective")
    states = checkpoint["optimizer"]["state"]
    if set(states) != set(parameter_ids):
        raise ValueError("optimizer state registry differs from parameter groups")
    rows = []
    for name, parameter_id in zip(names, parameter_ids):
        state = states[parameter_id]
        if tuple(state["exp_avg"].shape) != tuple(checkpoint["model"][name].shape):
            raise ValueError("optimizer first moment does not match its model parameter")
        if tuple(state["exp_avg_sq"].shape) != tuple(checkpoint["model"][name].shape):
            raise ValueError("optimizer second moment does not match its model parameter")
        rows.append((name, parameter_id, state))
    return rows


def is_sign_gauge_parameter(name: str) -> bool:
    """The closed sign set for the implemented row program."""

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


def sign_gauge_layer(name: str) -> int:
    marker = "decoder.dec_layers."
    if marker not in name:
        raise ValueError(f"sign-gauge parameter has no layer coordinate: {name}")
    suffix = name.split(marker, 1)[1]
    try:
        return int(suffix.split(".", 1)[0])
    except (IndexError, ValueError) as error:
        raise ValueError(
            f"sign-gauge parameter has malformed layer coordinate: {name}"
        ) from error


def _values_equal(left: Any, right: Any) -> bool:
    import torch

    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(
            left, right
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
    """Return one path per changed nested leaf, treating each tensor as one leaf."""

    if type(left) is not type(right):
        return [prefix or "<root>"]
    if isinstance(left, dict):
        paths = []
        keys = sorted(set(left).union(right), key=lambda value: (type(value).__name__, repr(value)))
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


def _allowed_checkpoint_difference(branch: str, path: str) -> bool:
    if path in {"protocol_sha256", "code_commit"}:
        return True
    operation = BRANCH_SPECS[branch]["operation"]
    if operation in {"scale-first-moment", "one-ulp-first-moment"}:
        return path.startswith("optimizer.state.") and path.endswith(".exp_avg")
    if operation == "scale-second-moment":
        return path.startswith("optimizer.state.") and path.endswith(".exp_avg_sq")
    if operation == "final-layernorm-bias-shift":
        return (
            path.startswith("model.decoder.dec_layers.")
            and path.endswith(".mha1.reslayerAs.1.layernormA.bias")
        )
    if operation == "sign-gauge":
        specification = BRANCH_SPECS[branch]
        if path.startswith("model."):
            name = path.removeprefix("model.")
            return bool(
                specification["transform_parameters"]
                and is_sign_gauge_parameter(name)
                and sign_gauge_layer(name) in specification["layers"]
            )
        return bool(
            specification["co_transform_first_moment"]
            and path.startswith("optimizer.state.")
            and path.endswith(".exp_avg")
        )
    return False


def apply_branch_intervention(
    checkpoint: dict[str, Any], branch: str
) -> dict[str, Any]:
    import torch

    specification = BRANCH_SPECS[branch]
    operation = specification["operation"]
    details = branch_specification(branch)
    if operation == "none":
        return details
    if operation == "scale-first-moment":
        factor = float(specification["factor"])
        count = 0
        for state in checkpoint["optimizer"]["state"].values():
            if "exp_avg" in state:
                state["exp_avg"].mul_(factor)
                count += 1
        details.update({"factor": factor, "changed_tensor_count": count})
    elif operation == "scale-second-moment":
        factor = float(specification["factor"])
        count = 0
        for state in checkpoint["optimizer"]["state"].values():
            if "exp_avg_sq" in state:
                state["exp_avg_sq"].mul_(factor)
                count += 1
        details.update({"factor": factor, "changed_tensor_count": count})
    elif operation == "one-ulp-first-moment":
        selected = None
        for state_key in sorted(
            checkpoint["optimizer"]["state"],
            key=lambda value: (type(value).__name__, repr(value)),
        ):
            moment = checkpoint["optimizer"]["state"][state_key].get("exp_avg")
            if moment is None or moment.numel() == 0:
                continue
            flat = moment.reshape(-1)
            candidates = torch.nonzero(flat != 0, as_tuple=False).reshape(-1)
            index = int(candidates[0]) if candidates.numel() else 0
            before = flat[index].detach().clone()
            direction = torch.full_like(before, float("inf"))
            after = torch.nextafter(before, direction)
            if torch.equal(before, after):
                direction = torch.full_like(before, float("-inf"))
                after = torch.nextafter(before, direction)
            flat[index] = after
            selected = {
                "state_key": repr(state_key),
                "flat_index": index,
                "before_hex": before.detach().cpu().numpy().tobytes().hex(),
                "after_hex": after.detach().cpu().numpy().tobytes().hex(),
            }
            break
        if selected is None:
            raise ValueError("one-ulp intervention found no AdamW first moment")
        details.update({"changed_tensor_count": 1, "selected_element": selected})
    elif operation == "final-layernorm-bias-shift":
        offset = float(specification["offset"])
        keys = sorted(
            name
            for name in checkpoint["model"]
            if name.endswith(".mha1.reslayerAs.1.layernormA.bias")
        )
        if len(keys) != 3:
            raise ValueError("final LayerNorm bias control requires exactly three tensors")
        for name in keys:
            checkpoint["model"][name].add_(offset)
        details.update(
            {
                "offset": offset,
                "changed_tensor_count": len(keys),
                "model_keys": keys,
            }
        )
    elif operation == "sign-gauge":
        rows = optimizer_states_by_model_name(checkpoint)
        layers = set(specification["layers"])
        selected = sorted(
            (
                (name, state)
                for name, _parameter_id, state in rows
                if is_sign_gauge_parameter(name)
                and sign_gauge_layer(name) in layers
            ),
            key=lambda row: row[0],
        )
        element_count = sum(
            checkpoint["model"][name].numel() for name, _ in selected
        )
        if not selected or element_count < 1:
            raise ValueError("sign-gauge action has empty measured support")
        transform_parameters = bool(specification["transform_parameters"])
        co_transform = bool(specification["co_transform_first_moment"])
        for name, state in selected:
            if transform_parameters:
                checkpoint["model"][name].neg_()
            if co_transform:
                state["exp_avg"].neg_()
        names = [name for name, _ in selected]
        details.update(
            {
                "changed_tensor_count": len(selected),
                "changed_element_count": int(element_count),
                "model_keys_sha256": hashlib.sha256(
                    canonical_json_bytes(names)
                ).hexdigest(),
                "model_keys": names,
            }
        )
    else:
        raise ValueError(f"unknown closure branch operation: {operation}")
    if details.get("changed_tensor_count", 0) < 1:
        raise ValueError(f"closure intervention changed no tensors: {branch}")
    return details


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
    try:
        path.relative_to(source_root)
    except ValueError as error:
        raise ValueError("source checkpoint descriptor escapes its run root") from error
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
    template: list[str],
    trajectory: dict[str, Any],
    paths: dict[str, Path],
    protocol_path: Path,
    commit: str,
    final_step: int,
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


def restored_energy_vector(
    checkpoint: dict[str, Any], source_step: int
) -> tuple[np.ndarray, np.ndarray]:
    recorder = checkpoint.get("recorder", {})
    steps = np.asarray(recorder.get("steps"))
    energies = np.asarray(recorder.get("energies"))
    map_ids = np.asarray(recorder.get("map_ids", [])).astype(str)
    matches = np.flatnonzero(steps == source_step)
    if (
        len(matches) != 1
        or energies.dtype != np.dtype(np.float64)
        or energies.shape != (len(steps), len(map_ids))
    ):
        raise ValueError("source checkpoint recorder state is malformed")
    return np.asarray(energies[int(matches[0])], dtype=np.float64), map_ids


def observer_context(
    source_config: dict[str, Any], refinedweb_root: Path
) -> dict[str, Any]:
    from datasets import Dataset

    command = source_config["producer_command"]
    reference_root = Path(option(command, "--reference-experiments")).resolve()
    dataset_shard = Path(option(command, "--dataset-shard")).resolve()
    refinedweb_root = refinedweb_root.resolve()
    try:
        dataset_relative = dataset_shard.relative_to(refinedweb_root)
    except ValueError as error:
        raise ValueError("source dataset shard is outside --refinedweb-root") from error
    for path in (
        reference_root / "pldr_model_v510.py",
        reference_root / "power_law_attention_layer_v510.py",
        dataset_shard,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    sys.path.insert(0, str(reference_root))
    from pldr_model_v510 import PLDR_Model

    specification = {
        "layers": int(option(command, "--layers")),
        "heads": int(option(command, "--heads")),
        "dk": int(option(command, "--dk")),
        "contexts": int(option(command, "--contexts")),
        "context_length": int(option(command, "--context-length")),
        "max_sequence_length": int(option(command, "--max-sequence-length")),
    }
    probe_offset = int(option(command, "--probe-document-offset"))
    probe_count = specification["contexts"] * (specification["context_length"] + 1)
    dataset = Dataset.from_file(str(dataset_shard))
    probe_values, probe_metadata = independent_probe_tokens(
        dataset, probe_offset, probe_count
    )
    return {
        "model_class": PLDR_Model,
        "specification": specification,
        "probe_values": probe_values,
        "probe_metadata": probe_metadata,
        "reference_sources": [
            {
                "path": name,
                "sha256": file_sha256(reference_root / name),
                "size_bytes": (reference_root / name).stat().st_size,
            }
            for name in ("pldr_model_v510.py", "power_law_attention_layer_v510.py")
        ],
        "dataset": {
            "root_role": "refinedweb_read_only_root",
            "relative_path": dataset_relative.as_posix(),
            "sha256": file_sha256(dataset_shard),
            "size_bytes": dataset_shard.stat().st_size,
        },
    }


def observe_checkpoint_energy(
    checkpoint: dict[str, Any],
    device_label: str,
    context: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Rebuild the model and time the represented observation route."""

    import torch

    device = torch.device(device_label)
    if device.type == "cuda" and (
        not torch.cuda.is_available()
        or device.index is None
        or device.index >= torch.cuda.device_count()
    ):
        raise RuntimeError(f"source-remeasurement device is unavailable: {device_label}")

    synchronization_seconds = 0.0

    def synchronize() -> None:
        nonlocal synchronization_seconds
        if device.type == "cuda":
            started_sync = time.perf_counter()
            torch.cuda.synchronize(device)
            synchronization_seconds += time.perf_counter() - started_sync

    synchronize()
    route_started = time.perf_counter()
    specification = context["specification"]
    torch.manual_seed(0)
    d_model = specification["heads"] * specification["dk"]

    construction_started = time.perf_counter()
    model = context["model_class"](
        num_layers=specification["layers"],
        d_model=d_model,
        num_heads=specification["heads"],
        dff=int(2.7 * d_model),
        input_vocab_size=257,
        A_dff=2 * specification["dk"],
        num_reslayerA=2,
        num_denseA=2,
        max_seq_len=specification["max_sequence_length"],
        device=device,
    )
    synchronize()
    construction_seconds = time.perf_counter() - construction_started

    transfer_started = time.perf_counter()
    model = model.to(device=device, dtype=torch.float32)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    synchronize()
    transfer_seconds = time.perf_counter() - transfer_started

    observation_started = time.perf_counter()
    width = specification["context_length"]
    probe = torch.tensor(
        context["probe_values"], dtype=torch.long, device=device
    ).reshape(specification["contexts"], width + 1)
    mask = torch.triu(
        torch.ones(width, width, device=device, dtype=torch.float32), diagonal=1
    )[None, None]
    captured: dict[int, Any] = {}

    def make_hook(layer_index: int):
        def hook(_module: Any, _inputs: Any, result: Any) -> None:
            captured[layer_index] = result.detach()

        return hook

    handles = [
        model.decoder.dec_layers[index]
        .mha1.reslayerAs[-1]
        .layernormA.register_forward_hook(make_hook(index))
        for index in range(specification["layers"])
    ]
    try:
        with torch.no_grad():
            _, _, _, cache = model([probe[:, :-1], mask])
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(range(specification["layers"])):
        raise RuntimeError("source-remeasurement hooks did not fire exactly once")
    for layer in range(specification["layers"]):
        if not torch.equal(captured[layer], cache[layer][2]):
            raise RuntimeError("source-remeasurement hook differs from cached row map")
    map_ids = np.asarray(
        [
            f"c{context_index}.L{layer}.H{head}"
            for context_index in range(specification["contexts"])
            for layer in range(specification["layers"])
            for head in range(specification["heads"])
        ]
    )
    rows = {
        f"c{context_index}.L{layer}.H{head}": cache[layer][2][context_index, head]
        for context_index in range(specification["contexts"])
        for layer in range(specification["layers"])
        for head in range(specification["heads"])
    }
    energies = np.asarray(
        [independent_row_centered_energy(rows[name]) for name in map_ids],
        dtype=np.float64,
    )
    row_map_sha256 = tensor_mapping_sha256(rows)
    row_map_element_count = sum(int(value.numel()) for value in rows.values())
    synchronize()
    observation_seconds = time.perf_counter() - observation_started
    elapsed = time.perf_counter() - route_started
    route = {
        "device": device_label,
        "device_type": device.type,
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "host CPU"
        ),
        "wall_seconds": elapsed,
        "timing": {
            "model_construction_seconds": construction_seconds,
            "state_load_and_device_transfer_seconds": transfer_seconds,
            "forward_and_energy_seconds": observation_seconds,
            "cuda_synchronization_seconds": synchronization_seconds,
            "total_route_elapsed_seconds": elapsed,
            "component_intervals_include_their_terminal_synchronization": True,
        },
        "native_model_dtype": "float32",
        "energy_route": "float32 row map converted to CPU float64, row centered and summed in float64",
        "hook_identity_check": True,
        "row_map_sha256": row_map_sha256,
        "row_map_tensor_count": len(rows),
        "row_map_element_count": row_map_element_count,
    }
    del model, probe, mask, cache, captured, rows
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return energies, map_ids, route


def vector_defect(reference: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
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
    }


def load_energy_archive(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return (
            np.asarray(archive["energies"], dtype=np.float64),
            np.asarray(archive["steps"], dtype=np.int64),
            np.asarray(archive["map_ids"]).astype(str),
        )


def emitted_checkpoint_descriptors(
    metadata: dict[str, Any],
    branch_root: Path,
    output_root: Path,
    trajectory_id: str,
    branch: str,
    expected_steps: set[int],
) -> list[dict[str, Any]]:
    """Validate and bind the exact checkpoint file set emitted by one branch."""

    rows = metadata.get("checkpoints")
    if not isinstance(rows, list):
        raise ValueError("producer checkpoint registry is missing")
    by_step = {}
    declared_paths = set()
    descriptors = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "path", "sha256", "size_bytes", "step"
        }:
            raise ValueError("producer checkpoint descriptor is malformed")
        step = row["step"]
        relative = Path(row["path"])
        if (
            not isinstance(step, int)
            or step in by_step
            or relative.is_absolute()
            or ".." in relative.parts
        ):
            raise ValueError("producer checkpoint registry is ambiguous")
        path = (branch_root / relative).resolve()
        try:
            path.relative_to((branch_root / "checkpoints").resolve())
        except ValueError as error:
            raise ValueError("producer checkpoint escapes its branch directory") from error
        if (
            not path.is_file()
            or path.stat().st_size != row["size_bytes"]
            or file_sha256(path) != row["sha256"]
        ):
            raise ValueError("producer checkpoint descriptor does not replay")
        declared_paths.add(path)
        by_step[step] = row
        descriptors.append(
            {
                "trajectory_id": trajectory_id,
                "branch": branch,
                "step": step,
                **descriptor(path, output_root),
            }
        )
    actual_paths = {
        path.resolve()
        for path in (branch_root / "checkpoints").glob("*")
        if path.is_file()
    }
    if set(by_step) != expected_steps or actual_paths != declared_paths:
        raise ValueError("producer checkpoint file set differs from the frozen registry")
    return sorted(descriptors, key=lambda row: row["step"])


def compare_retained_scientific_leaves(
    predecessor_root: Path,
    output_root: Path,
    completed: dict[str, list[dict[str, Any]]],
) -> tuple[bool, list[dict[str, Any]]]:
    manifest_path = predecessor_root / "run-manifest.json"
    result_path = predecessor_root / "result.json"
    predecessor_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    predecessor_result = json.loads(result_path.read_text(encoding="utf-8"))
    verify_record_seal(predecessor_manifest)
    verify_record_seal(predecessor_result)
    rows = []
    all_equal = True
    retained_branches = {
        "baseline-a",
        "baseline-b",
        "first-moment-0.999",
        "first-moment-0.9",
        "zero-first-moment",
        "zero-second-moment",
        "first-moment-one-ulp",
        "layernorm-bias-shift",
    }
    for trajectory_id, branch_rows in sorted(completed.items()):
        current = {row["branch"]: row for row in branch_rows}
        prior = {
            row["branch"]: row
            for row in predecessor_manifest["branches"][trajectory_id]
        }
        for branch in sorted(retained_branches):
            current_path = output_root / current[branch]["energies"]["path"]
            prior_path = predecessor_root / prior[branch]["energies"]["path"]
            current_energy, current_steps, current_maps = load_energy_archive(current_path)
            prior_energy, prior_steps, prior_maps = load_energy_archive(prior_path)
            equal = (
                np.array_equal(current_steps, prior_steps)
                and np.array_equal(current_maps, prior_maps)
                and np.array_equal(current_energy, prior_energy)
            )
            rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "branch": branch,
                    "energy_arrays_bitwise_equal": bool(equal),
                }
            )
            all_equal &= bool(equal)
    return all_equal, rows


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
        raise ValueError("closure output root must be inside the data root") from error
    if output_root.exists():
        raise FileExistsError(f"closure output root already exists: {output_root}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unknown closure-probe configuration schema")
    block_sizes = config.get("block_sizes")
    if block_sizes != [1, 2, 4, 8] or config.get("baseline_replicates") != 2:
        raise ValueError("closure probe requires scales [1,2,4,8] and two baselines")
    if config.get("branches") != list(BRANCH_SPECS):
        raise ValueError("closure branch registry differs from the frozen v3 program")
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
    source_root = resolve_data_root_locator(data_root, config.get("source_run"))
    predecessor_root = resolve_data_root_locator(
        data_root, config.get("predecessor_closure")
    )
    source_config_path = source_root / "config-resolved.json"
    source_protocol_path = source_root / "protocol-resolved.json"
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    source_rows = {
        row["trajectory_id"]: row for row in source_config["trajectories"]
    }
    source_step = int(trajectories[0]["source_step"])
    if len(source_rows) != len(source_config["trajectories"]):
        raise ValueError("source-run trajectory registry contains duplicates")
    producing_devices = {}
    for requested in trajectories:
        trajectory_id = requested["trajectory_id"]
        if trajectory_id not in source_rows:
            raise ValueError(f"unknown source trajectory: {trajectory_id}")
        source_device = source_rows[trajectory_id].get("device")
        requested_device = requested.get("device")
        if (
            not isinstance(source_device, int)
            or source_device not in source_config.get("devices", [])
            or requested_device != source_device
        ):
            raise ValueError(
                f"configured producing device differs from source run: {trajectory_id}"
            )
        producing_devices[trajectory_id] = source_device
    final_step = source_step + max(block_sizes)

    output_root.mkdir(parents=True)
    inputs_root = output_root / "inputs"
    inputs_root.mkdir()
    config_copy = inputs_root / "config.json"
    write_new_json(config_copy, config)
    protocol = copy.deepcopy(source_protocol)
    original_protocol_fields = {
        "protocol_id": source_protocol["protocol_id"],
        "expected_updates": source_protocol["expected_updates"],
        "segments": source_protocol["segments"],
        "checkpoint_steps": source_protocol["plga_analysis"]["checkpoint_steps"],
    }
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
        "branches": config["branches"],
    }
    protocol_path = inputs_root / "protocol.json"
    write_new_json(protocol_path, protocol)
    protocol_digest = hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()
    protocol_rewrite = {
        "source": original_protocol_fields,
        "successor": {
            "protocol_id": protocol["protocol_id"],
            "expected_updates": protocol["expected_updates"],
            "segments": protocol["segments"],
            "checkpoint_steps": protocol["plga_analysis"]["checkpoint_steps"],
            "closure_probe": protocol["closure_probe"],
        },
        "source_file_sha256": file_sha256(source_protocol_path),
        "successor_canonical_sha256": protocol_digest,
    }

    import torch

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    context = observer_context(
        source_config, Path(arguments.refinedweb_root).resolve()
    )

    branch_ids = list(BRANCH_SPECS)
    restored_rows = []
    original_measurement_rows = []
    branch_measurement_rows = []
    measurement_records = []
    prepared: dict[str, list[dict[str, Any]]] = {}
    source_descriptors = []
    common_map_ids: np.ndarray | None = None
    source_measurement_gpu_seconds = 0.0

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
        restored, recorder_map_ids = restored_energy_vector(
            base_checkpoint, source_step
        )
        if common_map_ids is None:
            common_map_ids = recorder_map_ids
        elif not np.array_equal(common_map_ids, recorder_map_ids):
            raise ValueError("source checkpoints have different row-map registries")
        restored_rows.append(restored)

        metadata_path = source_root / source_descriptor["metadata_path"]
        source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            context["probe_metadata"]["byte_token_sha256"]
            != source_metadata["data"]["probe"]["byte_token_sha256"]
        ):
            raise ValueError("reconstructed source probe differs from producer metadata")

        physical_device = producing_devices[trajectory_id]
        route_devices = (
            f"cuda:{physical_device}",
            f"cuda:{1 - physical_device}",
            "cpu",
        )
        trajectory_original = []
        route_records = []
        for route_id, device_label in zip(ROUTE_IDS, route_devices):
            measured, measured_map_ids, route = observe_checkpoint_energy(
                base_checkpoint, device_label, context
            )
            if not np.array_equal(recorder_map_ids, measured_map_ids):
                raise ValueError("source remeasurement changed the map registry")
            trajectory_original.append(measured)
            comparison = vector_defect(restored, measured)
            route_records.append(
                {"route_id": route_id, **route, **comparison}
            )
            if route["device_type"] == "cuda":
                source_measurement_gpu_seconds += float(route["wall_seconds"])
        original_measurement_rows.append(trajectory_original)

        original_model_digest = tensor_mapping_sha256(base_checkpoint["model"])
        first_before = moment_summary(base_checkpoint, "exp_avg")
        second_before = moment_summary(base_checkpoint, "exp_avg_sq")
        if first_before["nonzero_count"] == 0:
            raise ValueError("source AdamW first moments are identically zero")
        if second_before["nonzero_count"] == 0:
            raise ValueError("source AdamW second moments are identically zero")

        branch_rows = []
        trajectory_branch_measurements = []
        branch_source_records = []
        for branch_name in branch_ids:
            branch_root = (
                output_root / "trajectories" / trajectory_id / branch_name
            )
            checkpoint_dir = branch_root / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            shutil.copy2(
                source_zero,
                checkpoint_dir / "checkpoint-step000000.pt",
            )
            checkpoint = copy.deepcopy(base_checkpoint)
            checkpoint["code_commit"] = commit
            checkpoint["protocol_sha256"] = protocol_digest
            intervention = apply_branch_intervention(checkpoint, branch_name)
            differences = differing_paths(base_checkpoint, checkpoint)
            unexpected = [
                path
                for path in differences
                if not _allowed_checkpoint_difference(branch_name, path)
            ]
            if unexpected:
                raise RuntimeError(
                    f"checkpoint rewrite has undeclared changes for {branch_name}: {unexpected}"
                )
            first_after = moment_summary(checkpoint, "exp_avg")
            second_after = moment_summary(checkpoint, "exp_avg_sq")
            model_digest = tensor_mapping_sha256(checkpoint["model"])
            measured, measured_map_ids, route = observe_checkpoint_energy(
                checkpoint, f"cuda:{physical_device}", context
            )
            if not np.array_equal(recorder_map_ids, measured_map_ids):
                raise ValueError("branch source remeasurement changed the map registry")
            trajectory_branch_measurements.append(measured)
            comparison = vector_defect(restored, measured)
            branch_source_records.append(
                {
                    "branch": branch_name,
                    **route,
                    **comparison,
                }
            )
            source_measurement_gpu_seconds += float(route["wall_seconds"])

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
                source_config["producer_command"],
                trajectory,
                paths,
                protocol_path,
                commit,
                final_step,
            )
            branch_rows.append(
                {
                    "branch": branch_name,
                    "branch_specification": intervention,
                    "trajectory": trajectory,
                    "paths": paths,
                    "command": command,
                    "source_zero_checkpoint": source_zero_descriptor,
                    "model_state_sha256": model_digest,
                    "source_model_state_sha256": original_model_digest,
                    "first_moment_before": first_before,
                    "first_moment_after": first_after,
                    "second_moment_before": second_before,
                    "second_moment_after": second_after,
                    "checkpoint_rewrite": {
                        "assignment_performed": [
                            "code_commit",
                            "protocol_sha256",
                        ],
                        "assigned_values": {
                            "code_commit": {
                                "source": base_checkpoint.get("code_commit"),
                                "successor": commit,
                            },
                            "protocol_sha256": {
                                "source": base_checkpoint.get("protocol_sha256"),
                                "successor": protocol_digest,
                            },
                        },
                        "differing_leaf_paths": differences,
                        "unexpected_leaf_paths": unexpected,
                    },
                }
            )
        prepared[trajectory_id] = branch_rows
        branch_measurement_rows.append(trajectory_branch_measurements)
        measurement_records.append(
            {
                "trajectory_id": trajectory_id,
                "source_checkpoint_sha256": source_descriptor["sha256"],
                "restored_recorder_source": {
                    "step": source_step,
                    "map_count": int(len(recorder_map_ids)),
                },
                "original_checkpoint_routes": route_records,
                "rewritten_branch_producing_routes": branch_source_records,
            }
        )

    assert common_map_ids is not None
    measurement_path = output_root / "source-remeasurement.npz"
    np.savez_compressed(
        measurement_path,
        trajectory_ids=np.asarray(
            [row["trajectory_id"] for row in trajectories]
        ),
        route_ids=np.asarray(ROUTE_IDS),
        branch_ids=np.asarray(branch_ids),
        map_ids=common_map_ids,
        restored_recorder_energy=np.asarray(restored_rows, dtype=np.float64),
        original_checkpoint_remeasured_energy=np.asarray(
            original_measurement_rows, dtype=np.float64
        ),
        rewritten_branch_remeasured_energy=np.asarray(
            branch_measurement_rows, dtype=np.float64
        ),
    )

    def run_trajectory(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        completed_rows = []
        for row in rows:
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(row["trajectory"]["device"])
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
            metadata = json.loads(
                row["paths"]["metadata"].read_text(encoding="utf-8")
            )
            if (
                metadata.get("schema_version") != PRODUCER_SCHEMA
                or metadata.get("code_commit") != commit
                or metadata.get("trajectory_id")
                != row["trajectory"]["trajectory_id"]
            ):
                raise ValueError("closure producer metadata identity mismatch")
            output_checkpoints = emitted_checkpoint_descriptors(
                metadata,
                row["paths"]["root"],
                output_root,
                row["trajectory"]["trajectory_id"],
                row["branch"],
                {0, final_step},
            )
            completed_rows.append(
                {
                    "branch": row["branch"],
                    "branch_specification": row["branch_specification"],
                    "device": row["trajectory"]["device"],
                    "model_state_sha256": row["model_state_sha256"],
                    "source_model_state_sha256": row[
                        "source_model_state_sha256"
                    ],
                    "first_moment_before": row["first_moment_before"],
                    "first_moment_after": row["first_moment_after"],
                    "second_moment_before": row["second_moment_before"],
                    "second_moment_after": row["second_moment_after"],
                    "checkpoint_rewrite": row["checkpoint_rewrite"],
                    "resume_checkpoint": descriptor(
                        row["paths"]["resume"], output_root
                    ),
                    "energies": descriptor(row["paths"]["energies"], output_root),
                    "metrics": descriptor(row["paths"]["metrics"], output_root),
                    "metadata": descriptor(row["paths"]["metadata"], output_root),
                    "producer_log": descriptor(row["paths"]["log"], output_root),
                    "output_checkpoints": output_checkpoints,
                    "wall_seconds": float(metadata["wall_seconds"]),
                }
            )
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

    retained_equal, retained_rows = compare_retained_scientific_leaves(
        predecessor_root, output_root, completed
    )
    predecessor_manifest_path = predecessor_root / "run-manifest.json"
    predecessor_result_path = predecessor_root / "result.json"
    predecessor_manifest_record = json.loads(
        predecessor_manifest_path.read_text(encoding="utf-8")
    )
    predecessor_result_record = json.loads(
        predecessor_result_path.read_text(encoding="utf-8")
    )
    producer_gpu_seconds = sum(
        row["wall_seconds"] for rows in completed.values() for row in rows
    )
    gpu_seconds = producer_gpu_seconds + source_measurement_gpu_seconds
    checkpoint_inventory = [
        checkpoint
        for rows in completed.values()
        for row in rows
        for checkpoint in row["output_checkpoints"]
    ]
    manifest = seal_record(
        {
            "schema_version": MANIFEST_SCHEMA,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "code_commit": commit,
            "analysis_sources": sources,
            "probe_id": config["probe_id"],
            "source_run": {
                "locator": config["source_run"],
                "run_id": source_config["run_id"],
                "config_sha256": file_sha256(source_config_path),
                "protocol_sha256": file_sha256(source_protocol_path),
                "checkpoints": source_descriptors,
            },
            "predecessor": {
                "locator": config["predecessor_closure"],
                "run_manifest": (
                    anchored_descriptor(predecessor_manifest_path, data_root)
                    | {
                        "record_sha256": predecessor_manifest_record["record_sha256"]
                    }
                ),
                "result": (
                    anchored_descriptor(predecessor_result_path, data_root)
                    | {"record_sha256": predecessor_result_record["record_sha256"]}
                ),
                "retained_scientific_leaves_bitwise_equal": retained_equal,
                "comparisons": retained_rows,
            },
            "inputs": {
                "config": descriptor(config_copy, output_root),
                "protocol": descriptor(protocol_path, output_root),
                "source_remeasurement": descriptor(
                    measurement_path, output_root
                ),
            },
            "protocol_rewrite": protocol_rewrite,
            "source_remeasurement": {
                "routes": list(ROUTE_IDS),
                "observer": (
                    "independently reconstructed probe and model; hooked float32 "
                    "row maps followed by an independently implemented CPU "
                    "float64 row-centering and summation contract"
                ),
                "reference_sources": context["reference_sources"],
                "dataset": context["dataset"],
                "probe": context["probe_metadata"],
                "trajectories": measurement_records,
                "cuda_wall_seconds": source_measurement_gpu_seconds,
            },
            "source_step": source_step,
            "final_step": final_step,
            "block_sizes": block_sizes,
            "branch_ids": branch_ids,
            "branches": completed,
            "checkpoint_inventory": {
                "descriptor_count": len(checkpoint_inventory),
                "total_size_bytes": sum(
                    row["size_bytes"] for row in checkpoint_inventory
                ),
                "expected_steps_per_branch": [0, final_step],
                "all_emitted_files_bound": (
                    len(checkpoint_inventory)
                    == len(trajectories) * len(branch_ids) * 2
                ),
            },
            "gpu_time_accounting": {
                "source_remeasurement_cuda_wall_seconds": (
                    source_measurement_gpu_seconds
                ),
                "continuation_producer_reported_wall_seconds": (
                    producer_gpu_seconds
                ),
                "method": (
                    "sum of CUDA route wall seconds plus producer-reported "
                    "per-branch wall seconds"
                ),
            },
            "gpu_seconds": gpu_seconds,
            "gpu_hours": gpu_seconds / 3600.0,
        }
    )
    manifest_path = output_root / "run-manifest.json"
    write_new_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "complete",
                "record_sha256": manifest["record_sha256"],
                "gpu_hours": manifest["gpu_hours"],
                "trajectory_count": len(completed),
                "branch_count": len(branch_ids),
                "retained_scientific_leaves_bitwise_equal": retained_equal,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
