"""Canonical artifacts and complete-checkpoint validation for confirmation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch


CHECKPOINT_SCHEMA_VERSION = "pldr-complete-training-checkpoint-v3"
MODEL_ONLY_CHECKPOINT_SCHEMA_VERSION = "pldr-model-only-training-checkpoint-v1"
REQUEST_SCHEMA_VERSION = "pldr-composite-observability-stage-request-v3"
STAGE_RECORD_SCHEMA_VERSION = "pldr-composite-observability-stage-record-v3"

COMPLETE_CHECKPOINT_FIELDS = (
    "schema_version",
    "step",
    "model",
    "opt",
    "scheduler",
    "rng_states",
    "data_state",
    "optimizer_config",
    "model_config",
    "operation_order",
    "intervention_state",
    "code_manifest",
    "branch_state",
    "measurement_registry",
    "state_manifest",
    "config",
    "data_offset",
    "data_offset_end",
    "segment_global_offset",
    "segment_local_step",
    "schedule_total_steps",
)

MODEL_ONLY_CHECKPOINT_FIELDS = (
    "schema_version",
    "step",
    "model",
    "data_state",
    "optimizer_config",
    "model_config",
    "operation_order",
    "intervention_state",
    "code_manifest",
    "measurement_registry",
    "state_manifest",
    "config",
    "data_offset",
    "data_offset_end",
    "segment_global_offset",
    "segment_local_step",
    "schedule_total_steps",
    "run_identity",
)

TRAIN_UPDATE_ORDER = (
    "restore_rng",
    "load_registered_batch",
    "forward",
    "loss",
    "backward",
    "value_clip",
    "first_moment",
    "second_moment",
    "bias_correction",
    "decoupled_weight_decay_and_parameter_update",
    "post_step_intervention",
    "scheduler_step",
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def digest_object(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_bytes(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode(
            "utf-8") + b"\n")
    os.replace(temporary, destination)


def load_json_object(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _tensor_bytes(value: torch.Tensor | np.ndarray) -> bytes:
    if torch.is_tensor(value):
        array = value.detach().cpu().contiguous().numpy()
    else:
        array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise ValueError("object arrays are not scientific tensors")
    header = canonical_json_bytes({
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    })
    return header + b"\0" + array.tobytes(order="C")


def tensor_mapping_digest(mapping: dict[str, Any]) -> str:
    if not isinstance(mapping, dict):
        raise ValueError("tensor state must be a mapping")
    digest = hashlib.sha256()
    for name in sorted(mapping):
        value = mapping[name]
        if not torch.is_tensor(value) and not isinstance(value, np.ndarray):
            raise ValueError(f"state entry {name} is not a tensor")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_tensor_bytes(value))
        digest.update(b"\0")
    return digest.hexdigest()


def _update_nested_digest(digest: Any, value: Any) -> None:
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        digest.update(b"T")
        digest.update(_tensor_bytes(value))
    elif isinstance(value, dict):
        digest.update(b"D")
        for key in sorted(value, key=lambda item: repr(item)):
            _update_nested_digest(digest, key)
            _update_nested_digest(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b"L")
        for item in value:
            _update_nested_digest(digest, item)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        digest.update(b"J")
        digest.update(canonical_json_bytes(value))
    else:
        raise ValueError(
            f"unsupported checkpoint state type: {type(value).__name__}")


def nested_state_digest(value: Any) -> str:
    digest = hashlib.sha256()
    _update_nested_digest(digest, value)
    return digest.hexdigest()


def capture_rng_states() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": [
            value.cpu() for value in (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available()
                else []
            )
        ],
    }


def restore_rng_states(states: dict) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if not isinstance(states, dict) or set(states) != required:
        raise ValueError("checkpoint RNG registry is incomplete")
    numpy_state = states["numpy"]
    numpy_required = {
        "bit_generator", "keys", "position", "has_gauss",
        "cached_gaussian",
    }
    if not isinstance(numpy_state, dict) or set(numpy_state) != numpy_required:
        raise ValueError("checkpoint NumPy RNG state is incomplete")
    random.setstate(states["python"])
    np.random.set_state((
        numpy_state["bit_generator"],
        numpy_state["keys"].cpu().numpy().astype(np.uint32, copy=False),
        numpy_state["position"],
        numpy_state["has_gauss"],
        numpy_state["cached_gaussian"],
    ))
    torch.set_rng_state(states["torch_cpu"].cpu())
    if states["torch_cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA RNG states cannot be restored without CUDA")
        torch.cuda.set_rng_state_all(
            [value.cpu() for value in states["torch_cuda"]])


def rng_manifest(states: dict) -> dict:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if not isinstance(states, dict) or set(states) != required:
        raise ValueError("checkpoint RNG registry is incomplete")
    return {
        "python_sha256": nested_state_digest(states["python"]),
        "numpy_sha256": nested_state_digest(states["numpy"]),
        "torch_cpu_sha256": nested_state_digest(states["torch_cpu"]),
        "torch_cuda_sha256": [
            nested_state_digest(value) for value in states["torch_cuda"]
        ],
    }

def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_measurement_registry(
        value: dict, *, require_nonempty: bool = False,
) -> bool:
    """Validate the checkpoint-owned construction/validation/application map."""

    required = {
        "schema_version", "context_length", "construction", "validation",
        "dataset_sha256", "tokenizer_sha256",
    }
    legacy_required = required | {"application_pairs"}
    comprehensive = {
        "finite_stencil", "selected_edge_rule", "challenge_seed",
        "registry_sha256",
    }
    gate_shape = {
        "anchors", "pair_rule", "registry_sha256",
    }
    observable = {
        "anchors", "context_head_block_rule", "block_row_rule",
        "graph_transfer", "plga_binding_rule", "intervention_units",
        "intervention_unit_rule", "registry_sha256",
    }
    chronological = {
        "anchors", "context_head_block_rule", "block_row_rule",
        "history_length", "intervention_units", "intervention_unit_rule",
        "registry_sha256",
    }
    contrast_energy = {
        "anchors", "context_head_block_rule", "block_row_rule",
        "history_length", "pair_rule", "layers", "registry_sha256",
    }
    finite_increment = {
        "campaign_id", "total_chunks", "trainer_probe_reserve_chunks",
        "trainer_training_limit", "heads", "layers", "rows_per_map",
        "pairs_per_map", "pair_rule", "cross_map_pairs_forbidden",
        "registry_sha256",
    }

    if not isinstance(value, dict):
        raise ValueError("measurement registry must be an object")
    schema = value.get("schema_version")
    if schema == "pldr-measurement-registry-v3":
        expected = legacy_required
    elif schema == "pldr-comprehensive-measurement-registry-v4":
        expected = legacy_required | comprehensive
    elif schema == "pldr-gate-shape-registry-v1":
        expected = legacy_required | gate_shape
    elif schema == "pldr-observable-probe-registry-v1":
        expected = required | observable
    elif schema == "pldr-chronological-probe-registry-v1":
        expected = required | chronological
    elif schema == "pldr-causal-probe-registry-v1":
        expected = required | chronological
    elif schema == "pldr-contrast-energy-registry-v1":
        expected = required | contrast_energy
    elif schema in {
        "pldr-finite-increment-registry-v1",
        "pldr-source-restoring-registry-v1",
    }:
        expected = required | finite_increment
    else:
        raise ValueError("unknown measurement-registry schema")
    if set(value) != expected:
        raise ValueError("measurement registry has missing or unknown fields")
    if schema in {
        "pldr-comprehensive-measurement-registry-v4",
        "pldr-gate-shape-registry-v1",
        "pldr-observable-probe-registry-v1",
        "pldr-chronological-probe-registry-v1",
        "pldr-causal-probe-registry-v1",
        "pldr-contrast-energy-registry-v1",
        "pldr-finite-increment-registry-v1",
        "pldr-source-restoring-registry-v1",
    }:
        unsigned = dict(value)
        recorded = unsigned.pop("registry_sha256")
        if not _is_sha256(recorded) or digest_object(unsigned) != recorded:
            raise ValueError("measurement registry digest does not replay")
    if schema == "pldr-gate-shape-registry-v1":
        anchors = value["anchors"]
        if (
            not isinstance(anchors, list) or not anchors
            or any(not isinstance(step, int) or isinstance(step, bool) or step < 0
                   for step in anchors)
            or anchors != sorted(set(anchors))
            or value["pair_rule"] != "adjacent-flattened-row-pairs-v1"
        ):
            raise ValueError("gate-shape registry geometry is invalid")
    if schema == "pldr-contrast-energy-registry-v1":
        if (
            value["anchors"] != list(range(1000, 9001, 500))
            or value["context_head_block_rule"]
            != "head-equals-context-index-modulo-four-v1"
            or value["block_row_rule"]
            != "complete-ordered-generator-rows-0-through-63-v1"
            or value["history_length"] != 48
            or value["pair_rule"]
            != "all-unordered-pairs-lexicographic-v1"
            or value["layers"] != [0, 1, 2]
            or len(value["construction"]) != 8
            or len(value["validation"]) != 16
        ):
            raise ValueError("contrast-energy registry geometry is invalid")
    if schema == "pldr-finite-increment-registry-v1":
        if (
            value["campaign_id"]
            != "pldr-finite-increment-collapse-confirmation-v1"
            or value["heads"] != [0, 1, 2, 3]
            or value["layers"] != [0, 1, 2]
            or value["rows_per_map"] != 64
            or value["pairs_per_map"] != 2016
            or value["pair_rule"]
            != "within-context-layer-head-all-unordered-pairs-v1"
            or value["cross_map_pairs_forbidden"] is not True
            or len(value["construction"]) != 8
            or len(value["validation"]) != 16
            or not isinstance(value["total_chunks"], int)
            or value["total_chunks"] <= 0
            or not isinstance(value["trainer_probe_reserve_chunks"], int)
            or value["trainer_probe_reserve_chunks"] <= 0
            or value["trainer_training_limit"]
            != value["total_chunks"] - value["trainer_probe_reserve_chunks"]
        ):
            raise ValueError("finite-increment registry geometry is invalid")
    if schema == "pldr-observable-probe-registry-v1":
        if (
            value["anchors"] != sorted(set(value["anchors"]))
            or not value["anchors"]
            or any(not isinstance(step, int) or isinstance(step, bool) or step < 0
                   for step in value["anchors"])
            or value["context_head_block_rule"]
            != "head-equals-context-index-modulo-four-v1"
            or value["block_row_rule"]
            != "complete-ordered-generator-rows-0-through-63-v1"
            or value["graph_transfer"]
            != "single-union-ordinal-block-transfer-v1"
            or value["plga_binding_rule"]
            != "same-node-context-block-anchor-row-zero-b-locks-layer-node-v1"
            or not isinstance(value["intervention_unit_rule"], str)
            or not value["intervention_unit_rule"]
        ):
            raise ValueError("observable registry geometry is invalid")
        units = value["intervention_units"]
        if not isinstance(units, list) or len(units) != 24:
            raise ValueError("observable intervention registry needs 24 units")
        unit_ids = set()
        substreams = set()
        seed_counts = {5555: 0, 6666: 0}
        intervals = {5555: [], 6666: []}
        for unit in units:
            if not isinstance(unit, dict) or set(unit) != {
                "id", "seed", "ordinal", "future_update_start",
                "update_count", "rng_substream",
            }:
                raise ValueError("observable intervention unit is malformed")
            identifier = unit["id"]
            seed = unit["seed"]
            ordinal = unit["ordinal"]
            start = unit["future_update_start"]
            count = unit["update_count"]
            substream = unit["rng_substream"]
            if (
                not isinstance(identifier, str) or not identifier
                or identifier in unit_ids or seed not in seed_counts
                or not isinstance(ordinal, int) or isinstance(ordinal, bool)
                or ordinal != seed_counts[seed]
                or not isinstance(start, int) or isinstance(start, bool) or start < 0
                or not isinstance(count, int) or isinstance(count, bool) or count < 1
                or not isinstance(substream, int) or isinstance(substream, bool)
                or substream < 0 or substream in substreams
            ):
                raise ValueError("observable intervention unit identity is invalid")
            stop = start + count
            if any(not (stop <= left or right <= start)
                   for left, right in intervals[seed]):
                raise ValueError("observable intervention intervals overlap")
            unit_ids.add(identifier)
            substreams.add(substream)
            seed_counts[seed] += 1
            intervals[seed].append((start, stop))
        if seed_counts != {5555: 12, 6666: 12}:
            raise ValueError("observable intervention units are not split 12/12")

    if schema == "pldr-causal-probe-registry-v1":
        expected_anchors = list(range(1000, 9001, 500))
        if (
            value["anchors"] != expected_anchors
            or value["context_head_block_rule"]
            != "head-equals-context-index-modulo-four-v1"
            or value["block_row_rule"]
            != "complete-ordered-generator-rows-0-through-63-v1"
            or value["history_length"] != 48
            or len(value["construction"]) != 8
            or len(value["validation"]) != 16
            or value["intervention_unit_rule"]
            != "four-anchors-by-four-horizons-per-heldout-seed-v1"
        ):
            raise ValueError("causal registry geometry is invalid")
        units = value["intervention_units"]
        required_unit = {
            "id", "seed", "ordinal", "future_update_start",
            "update_count", "rng_substream",
        }
        expected_units = {
            (seed, ordinal)
            for seed in (9444, 10444, 11444, 12444)
            for ordinal in range(16)
        }
        if (
            not isinstance(units, list)
            or len(units) != 64
            or any(
                not isinstance(unit, dict) or set(unit) != required_unit
                for unit in units
            )
            or {(unit["seed"], unit["ordinal"]) for unit in units}
            != expected_units
            or any(
                unit["id"]
                != "cri-s{}-{:02d}".format(
                    unit["seed"], unit["ordinal"])
                or unit["future_update_start"] != 0
                or unit["update_count"] != unit["ordinal"] % 4 + 1
                or not isinstance(unit["rng_substream"], int)
                for unit in units
            )
        ):
            raise ValueError("causal intervention units are malformed")

    if schema == "pldr-chronological-probe-registry-v1":
        if (
            value["anchors"] != [2000, 4000, 8000, 16000, 24000]
            or value["context_head_block_rule"]
            != "head-equals-context-index-modulo-four-v1"
            or value["block_row_rule"]
            != "complete-ordered-generator-rows-0-through-63-v1"
            or value["history_length"] != 48
            or len(value["construction"]) != 8
            or len(value["validation"]) != 16
            or value["intervention_unit_rule"]
            != "twelve-fixed-four-update-sites-per-heldout-seed-v1"
        ):
            raise ValueError("chronological registry geometry is invalid")
        units = value["intervention_units"]
        required_unit = {
            "id", "seed", "ordinal", "future_update_start",
            "update_count", "rng_substream",
        }
        expected_units = {
            (seed, ordinal)
            for seed in (9444, 10444, 11444, 12444)
            for ordinal in range(12)
        }
        if (
            not isinstance(units, list)
            or len(units) != 48
            or any(
                not isinstance(unit, dict) or set(unit) != required_unit
                for unit in units
            )
        ):
            raise ValueError("chronological intervention units are malformed")
        for unit in units:
            seed = unit["seed"]
            ordinal = unit["ordinal"]
            start = unit["future_update_start"]
            count = unit["update_count"]
            substream = unit["rng_substream"]
            if (
                not isinstance(unit["id"], str)
                or unit["id"] != f"cci-s{seed}-{ordinal:02d}"
                or not isinstance(seed, int) or isinstance(seed, bool)
                or seed not in {9444, 10444, 11444, 12444}
                or not isinstance(ordinal, int) or isinstance(ordinal, bool)
                or not 0 <= ordinal < 12
                or not isinstance(start, int) or isinstance(start, bool)
                or start != ordinal * 4
                or not isinstance(count, int) or isinstance(count, bool)
                or count != 4
                or not isinstance(substream, int) or isinstance(substream, bool)
                or substream < 0
            ):
                raise ValueError("chronological intervention unit is invalid")
        if (
            {(unit["seed"], unit["ordinal"]) for unit in units}
            != expected_units
            or len({unit["id"] for unit in units}) != 48
            or len({unit["rng_substream"] for unit in units}) != 48
        ):
            raise ValueError("chronological intervention units overlap")
    if schema == "pldr-comprehensive-measurement-registry-v4":
        stencil = value["finite_stencil"]
        if (not isinstance(stencil, dict)
                or set(stencil) != {
                    "step", "direction_algorithm", "direction_seed",
                    "one_direction_per_physical_row",
                    "joint_cover_algorithm", "cover_directions_per_row",
                    "cover_seed"}
                or not isinstance(stencil["step"], (int, float))
                or stencil["step"] <= 0
                or stencil["direction_algorithm"] != "numpy-pcg64-normalized-v1"
                or not isinstance(stencil["direction_seed"], int)
                or stencil["one_direction_per_physical_row"] is not True
                or stencil["joint_cover_algorithm"]
                != "scaled-row-plus-direction-nearest-v1"
                or not isinstance(stencil["cover_directions_per_row"], int)
                or stencil["cover_directions_per_row"] <= 0
                or not isinstance(stencil["cover_seed"], int)):
            raise ValueError("comprehensive finite-stencil registry is invalid")
        edge_rule = value["selected_edge_rule"]
        if (not isinstance(edge_rule, dict)
                or set(edge_rule) != {"kind", "competitor_ids"}
                or edge_rule["kind"] != "target-plus-fixed-competitors"
                or not isinstance(edge_rule["competitor_ids"], list)
                or not edge_rule["competitor_ids"]
                or len(edge_rule["competitor_ids"])
                != len(set(edge_rule["competitor_ids"]))
                or any(not isinstance(index, int) or index < 0
                       for index in edge_rule["competitor_ids"])
                or not isinstance(value["challenge_seed"], int)):
            raise ValueError("comprehensive selected-edge registry is invalid")
    context_length = value["context_length"]
    if (
        not isinstance(context_length, int)
        or isinstance(context_length, bool)
        or context_length < 2
    ):
        raise ValueError("measurement registry context length is invalid")
    if not _is_sha256(value["dataset_sha256"]):
        raise ValueError("measurement registry dataset digest is invalid")
    tokenizer = value["tokenizer_sha256"]
    if tokenizer is not None and not _is_sha256(tokenizer):
        raise ValueError("measurement registry tokenizer digest is invalid")

    identifiers = set()
    chunks = set()
    for role in ("construction", "validation"):
        rows = value[role]
        if not isinstance(rows, list):
            raise ValueError(f"measurement registry {role} rows are malformed")
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"id", "chunk_index"}:
                raise ValueError(f"measurement registry {role} row is malformed")
            identifier = row["id"]
            chunk = row["chunk_index"]
            if (
                not isinstance(identifier, str) or not identifier
                or identifier in identifiers
                or not isinstance(chunk, int) or isinstance(chunk, bool)
                or chunk < 0 or chunk in chunks
            ):
                raise ValueError(
                    "measurement registry identifiers or chunks overlap")
            identifiers.add(identifier)
            chunks.add(chunk)

    if "application_pairs" in value:
        pairs = value["application_pairs"]
        if not isinstance(pairs, list):
            raise ValueError("measurement registry application pairs are malformed")
        for row in pairs:
            if not isinstance(row, dict) or set(row) != {
                "id", "left_chunk", "right_chunk",
            }:
                raise ValueError("measurement registry application row is malformed")
            identifier = row["id"]
            left = row["left_chunk"]
            right = row["right_chunk"]
            if (
                not isinstance(identifier, str) or not identifier
                or identifier in identifiers
                or any(
                    not isinstance(chunk, int) or isinstance(chunk, bool) or chunk < 0
                    for chunk in (left, right)
                )
                or left == right or left in chunks or right in chunks
            ):
                raise ValueError("measurement registry application pair overlaps")
            identifiers.add(identifier)
            chunks.update((left, right))

    if require_nonempty and (
        not value["construction"] or not value["validation"]
        or (
            "application_pairs" in value and not value["application_pairs"]
        )
        or (
            schema == "pldr-observable-probe-registry-v1"
            and not value["intervention_units"]
        )
        or (
            schema == "pldr-chronological-probe-registry-v1"
            and not value["intervention_units"]
        )
    ):
        raise ValueError("confirmation measurement registry is empty")
    return True


def empty_measurement_registry(
        *, dataset_sha256: str, tokenizer_sha256: str | None,
        context_length: int,
) -> dict:
    value = {
        "schema_version": "pldr-measurement-registry-v3",
        "context_length": int(context_length),
        "construction": [],
        "validation": [],
        "application_pairs": [],
        "dataset_sha256": dataset_sha256,
        "tokenizer_sha256": tokenizer_sha256,
    }
    validate_measurement_registry(value)
    return value



def build_state_manifest(payload: dict) -> dict:
    required = {
        "model", "opt", "scheduler", "rng_states", "data_state",
        "optimizer_config", "model_config", "operation_order", "branch_state",
        "measurement_registry", "intervention_state", "code_manifest",
    }
    if not required.issubset(payload):
        missing = sorted(required - set(payload))
        raise ValueError(
            "cannot build complete-state manifest; missing " + ", ".join(missing))
    value = {
        "model_sha256": tensor_mapping_digest(payload["model"]),
        "optimizer_sha256": nested_state_digest(payload["opt"]),
        "scheduler_sha256": nested_state_digest(payload["scheduler"]),
        "rng": rng_manifest(payload["rng_states"]),
        "data_state_sha256": digest_object(payload["data_state"]),
        "optimizer_config_sha256": digest_object(payload["optimizer_config"]),
        "model_config_sha256": digest_object(payload["model_config"]),
        "operation_order_sha256": digest_object(payload["operation_order"]),
        "branch_state_sha256": nested_state_digest(payload["branch_state"]),
        "measurement_registry_sha256": digest_object(
            payload["measurement_registry"]
        ),
        "intervention_state_sha256": digest_object(
            payload["intervention_state"]),
        "code_manifest_sha256": digest_object(payload["code_manifest"]),
    }
    if "run_identity" in payload:
        value["run_identity_sha256"] = digest_object(payload["run_identity"])
    value["complete_state_sha256"] = digest_object(value)
    return value


def build_model_only_state_manifest(payload: dict) -> dict:
    required = {
        "model", "data_state", "optimizer_config", "model_config",
        "operation_order", "measurement_registry", "intervention_state",
        "code_manifest", "run_identity",
    }
    if not required.issubset(payload):
        missing = sorted(required - set(payload))
        raise ValueError(
            "cannot build model-only manifest; missing " + ", ".join(missing))
    value = {
        "model_sha256": tensor_mapping_digest(payload["model"]),
        "data_state_sha256": digest_object(payload["data_state"]),
        "optimizer_config_sha256": digest_object(payload["optimizer_config"]),
        "model_config_sha256": digest_object(payload["model_config"]),
        "operation_order_sha256": digest_object(payload["operation_order"]),
        "measurement_registry_sha256": digest_object(
            payload["measurement_registry"]),
        "intervention_state_sha256": digest_object(
            payload["intervention_state"]),
        "code_manifest_sha256": digest_object(payload["code_manifest"]),
        "run_identity_sha256": digest_object(payload["run_identity"]),
    }
    value["model_state_sha256"] = digest_object(value)
    return value


def validate_model_only_checkpoint(payload: dict) -> bool:
    if not isinstance(payload, dict) or set(payload) != set(
        MODEL_ONLY_CHECKPOINT_FIELDS
    ):
        raise ValueError("model-only checkpoint has missing or unknown fields")
    if payload["schema_version"] != MODEL_ONLY_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unknown model-only checkpoint schema")
    if (
        not isinstance(payload["step"], int)
        or isinstance(payload["step"], bool)
        or payload["step"] < 0
        or not isinstance(payload["model"], dict)
        or not payload["model"]
        or tuple(payload["operation_order"]) != TRAIN_UPDATE_ORDER
        or payload["data_state"].get("cursor_end")
        != payload["data_offset_end"]
    ):
        raise ValueError("model-only checkpoint state is malformed")
    validate_measurement_registry(payload["measurement_registry"])
    if payload["state_manifest"] != build_model_only_state_manifest(payload):
        raise ValueError("model-only checkpoint manifest does not replay")
    return True


def validate_complete_checkpoint(payload: dict) -> bool:
    if not isinstance(payload, dict):
        raise ValueError("complete checkpoint must be a mapping")
    expected_fields = set(COMPLETE_CHECKPOINT_FIELDS)
    actual_fields = set(payload)
    allowed_fields = (expected_fields, expected_fields | {"run_identity"})
    if actual_fields not in allowed_fields:
        nearest = min(allowed_fields, key=lambda fields: len(fields ^ actual_fields))
        raise ValueError(
            "complete checkpoint has missing or unknown fields: "
            + ", ".join(sorted(nearest ^ actual_fields)))
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unknown complete checkpoint schema")
    step = payload["step"]
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("checkpoint step must be a nonnegative integer")
    if not isinstance(payload["model"], dict) or not payload["model"]:
        raise ValueError("checkpoint model state is empty")
    if not isinstance(payload["opt"], dict) or not payload["opt"]:
        raise ValueError("checkpoint optimizer state is empty")
    if not isinstance(payload["scheduler"], dict):
        raise ValueError("checkpoint scheduler state is malformed")
    if tuple(payload["operation_order"]) != TRAIN_UPDATE_ORDER:
        raise ValueError("checkpoint operation order is stale")
    branch = payload["branch_state"]
    branch_fields = {
        "schema_version", "clip_mask", "decay_mask", "device",
        "dtype", "scheduler_phase",
    }
    if not isinstance(branch, dict) or set(branch) != branch_fields:
        raise ValueError("checkpoint branch state is incomplete")
    if branch["schema_version"] != "pldr-realized-branch-state-v3":
        raise ValueError("unknown checkpoint branch-state schema")
    clip_mask = branch["clip_mask"]
    decay_mask = branch["decay_mask"]
    if (
        not isinstance(clip_mask, dict)
        or not isinstance(decay_mask, dict)
        or set(clip_mask) != set(decay_mask)
        or not set(clip_mask).issubset(payload["model"])
    ):
        raise ValueError("checkpoint realized masks are malformed")
    for name in clip_mask:
        for mask in (clip_mask[name], decay_mask[name]):
            if (
                not torch.is_tensor(mask)
                or mask.dtype != torch.bool
                or tuple(mask.shape) != tuple(payload["model"][name].shape)
            ):
                raise ValueError(f"checkpoint mask {name} has wrong shape or dtype")
    if not isinstance(branch["device"], str) or not branch["device"]:
        raise ValueError("checkpoint branch device identity is missing")
    if not isinstance(branch["dtype"], str) or not branch["dtype"]:
        raise ValueError("checkpoint branch dtype is missing")
    if "run_identity" in payload:
        identity = payload["run_identity"]
        identity_fields = {
            "schema_version", "run_id", "lineage_root", "parent_run_id",
            "from_scratch", "source_checkpoint_sha256",
        }
        if (not isinstance(identity, dict) or set(identity) != identity_fields
                or identity["schema_version"] != "pldr-run-identity-v1"
                or not isinstance(identity["run_id"], str)
                or not identity["run_id"]
                or not isinstance(identity["lineage_root"], str)
                or not identity["lineage_root"]
                or not isinstance(identity["from_scratch"], bool)):
            raise ValueError("checkpoint run identity is malformed")
        if identity["from_scratch"]:
            if (identity["parent_run_id"] is not None
                    or identity["source_checkpoint_sha256"] is not None):
                raise ValueError("fresh checkpoint run identity has a parent")
        elif (not isinstance(identity["parent_run_id"], str)
                or not identity["parent_run_id"]
                or not _is_sha256(identity["source_checkpoint_sha256"])):
            raise ValueError("continuation checkpoint run identity lacks lineage")
    validate_measurement_registry(payload["measurement_registry"])
    if payload["data_state"].get("cursor_end") != payload["data_offset_end"]:
        raise ValueError("checkpoint data cursor disagrees with top-level state")
    expected = build_state_manifest(payload)
    if payload["state_manifest"] != expected:
        raise ValueError("complete checkpoint manifest does not replay")
    return True


def load_complete_checkpoint(
        path: str | Path, *, map_location: str | torch.device = "cpu",
) -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    validate_complete_checkpoint(payload)
    return payload


def flatten_tensor_mapping(mapping: dict[str, Any]) -> np.ndarray:
    values = []
    for name in sorted(mapping):
        value = mapping[name]
        if torch.is_tensor(value):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        if array.dtype.hasobject or not np.isfinite(array).all():
            raise ValueError(f"state tensor {name} is nonnumeric or nonfinite")
        values.append(np.asarray(array, dtype=np.float64).reshape(-1))
    if not values:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(values)


def scientific_record_digest(record: dict) -> str:
    unsigned = {
        key: value for key, value in record.items()
        if key not in {
            "record_sha256", "scientific_record_sha256",
            "resource_metadata",
        }
    }
    return digest_object(unsigned)


def seal_record(record: dict) -> dict:
    value = dict(record)
    value.pop("record_sha256", None)
    value.pop("scientific_record_sha256", None)
    value["scientific_record_sha256"] = scientific_record_digest(value)
    value["record_sha256"] = digest_object(value)
    return value


def verify_sealed_record(record: dict) -> bool:
    if not isinstance(record, dict):
        raise ValueError("sealed record must be a mapping")
    unsigned = dict(record)
    digest = unsigned.pop("record_sha256", None)
    if digest != digest_object(unsigned):
        raise ValueError("record digest does not replay")
    if record.get("scientific_record_sha256") != scientific_record_digest(
        record
    ):
        raise ValueError("scientific record digest does not replay")
    return True


def load_npz_strict(path: str | Path, required_arrays: set[str]) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(required_arrays):
            missing = sorted(set(required_arrays) - set(archive.files))
            unknown = sorted(set(archive.files) - set(required_arrays))
            raise ValueError(
                f"raw NPZ schema mismatch; missing={missing}, unknown={unknown}")
        values = {
            name: np.asarray(archive[name]) for name in sorted(archive.files)
        }
    for name, value in values.items():
        if value.dtype.hasobject:
            raise ValueError(f"raw array {name} uses object dtype")
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"raw array {name} contains a nonfinite value")
    return values


def validate_partition_manifest(value: dict) -> bool:
    required = {
        "schema_version", "construction_ids", "validation_ids",
        "application_ids", "dataset_sha256", "tokenizer_sha256",
        "measurement_registry_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("partition manifest has missing or unknown fields")
    if value["schema_version"] != "pldr-confirmation-partitions-v3":
        raise ValueError("unknown partition manifest schema")
    sets = {}
    for name in ("construction_ids", "validation_ids", "application_ids"):
        rows = value[name]
        if (
            not isinstance(rows, list)
            or any(not isinstance(item, str) or not item for item in rows)
            or len(set(rows)) != len(rows)
        ):
            raise ValueError(f"{name} is not a unique string list")
        sets[name] = set(rows)
    if (
        sets["construction_ids"] & sets["validation_ids"]
        or sets["construction_ids"] & sets["application_ids"]
        or sets["validation_ids"] & sets["application_ids"]
    ):
        raise ValueError("construction, validation, and application rows overlap")
    for name in (
        "dataset_sha256", "tokenizer_sha256", "measurement_registry_sha256",
    ):
        digest = value[name]
        if (
            not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{name} is not a lowercase SHA-256 digest")
    return True


CHECKPOINT_SET_SCHEMA_VERSION = "pldr-complete-checkpoint-set-v3"


def build_checkpoint_set_manifest(
        artifact_root: str | Path, relative_paths: list[str]) -> dict:
    root = Path(artifact_root).resolve()
    if not relative_paths:
        raise ValueError("checkpoint set cannot be empty")
    rows = []
    for raw in relative_paths:
        relative = Path(raw)
        if relative.is_absolute() or "." in relative.parts or ".." in relative.parts:
            raise ValueError("checkpoint-set paths must be normalized and relative")
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError("checkpoint-set path does not resolve inside root")
        checkpoint = load_complete_checkpoint(path)
        rows.append({
            "path": relative.as_posix(),
            "sha256": sha256_path(path),
            "step": checkpoint["step"],
            "complete_state_sha256":
                checkpoint["state_manifest"]["complete_state_sha256"],
        })
    value = {
        "schema_version": CHECKPOINT_SET_SCHEMA_VERSION,
        "checkpoints": rows,
    }
    value["manifest_sha256"] = digest_object(value)
    return value


def load_checkpoint_set(
        manifest_path: str | Path, artifact_root: str | Path) -> list[dict]:
    value = load_json_object(manifest_path)
    if (
        not isinstance(value, dict)
        or set(value) != {
            "schema_version", "checkpoints", "manifest_sha256"}
        or value["schema_version"] != CHECKPOINT_SET_SCHEMA_VERSION
    ):
        raise ValueError("checkpoint-set manifest is malformed")
    unsigned = dict(value)
    digest = unsigned.pop("manifest_sha256")
    if digest != digest_object(unsigned):
        raise ValueError("checkpoint-set manifest digest does not replay")
    root = Path(artifact_root).resolve()
    checkpoints = []
    for row in value["checkpoints"]:
        if set(row) != {
            "path", "sha256", "step", "complete_state_sha256"}:
            raise ValueError("checkpoint-set row is malformed")
        relative = Path(row["path"])
        if relative.is_absolute() or "." in relative.parts or ".." in relative.parts:
            raise ValueError("checkpoint-set path is not normalized")
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError("checkpoint-set path does not resolve")
        if sha256_path(path) != row["sha256"]:
            raise ValueError("checkpoint-set file digest mismatch")
        checkpoint = load_complete_checkpoint(path)
        if (
            checkpoint["step"] != row["step"]
            or checkpoint["state_manifest"]["complete_state_sha256"]
            != row["complete_state_sha256"]
        ):
            raise ValueError("checkpoint-set scientific state binding mismatch")
        checkpoints.append(checkpoint)
    return checkpoints
