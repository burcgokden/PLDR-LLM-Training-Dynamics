#!/usr/bin/env python3
"""Reduce validated endpoint Taylor models to signed energy-cell bounds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from confirmation_artifacts import digest_object  # noqa: E402
from contrast_energy import (  # noqa: E402
    cell_second_bounds,
    integrate_pair_energy,
    unordered_pair_chunks,
)
from contrast_energy_specs import (  # noqa: E402
    CAMPAIGN_ID as REGISTERED_CAMPAIGN_ID,
    LOCK_SCHEMA,
    TAYLOR_POLICY,
)


INPUT_SCHEMA = "pldr-contrast-energy-taylor-model-v1"
SOURCE_SCHEMA = "pldr-chronological-pair-ledger-v1"
OUTPUT_SCHEMA = "pldr-validated-contrast-energy-jets-v1"


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_lock(
    path: str | Path | None,
    source_metadata: dict[str, Any],
) -> dict[str, Any] | None:
    core = source_metadata["campaign_id"] == REGISTERED_CAMPAIGN_ID
    heldout = source_metadata["role"] == "heldout"
    source_lock = source_metadata["lock_sha256"]
    if path is None:
        if core and (heldout or source_lock):
            raise ValueError(
                "held-out Taylor reduction requires the construction lock")
        return None
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(value) if isinstance(value, dict) else {}
    recorded = unsigned.pop("lock_sha256", None)
    if (
        not core
        or not heldout
        or value.get("schema_version") != LOCK_SCHEMA
        or value.get("campaign_id") != REGISTERED_CAMPAIGN_ID
        or recorded != source_lock
        or recorded != digest_object(unsigned)
        or not _is_sha256(
            value.get("taylor_backend_manifest_sha256"))
    ):
        raise ValueError("Taylor reduction construction lock is invalid")
    return value


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar(record: Any, name: str, kind: type) -> Any:
    if name not in record:
        raise ValueError(f"Taylor model omits {name}")
    value = np.asarray(record[name])
    if value.shape != ():
        raise ValueError(f"{name} must be scalar")
    return kind(value.item())


def _array(record: Any, name: str, ndim: int, dtype: Any = np.float64) -> np.ndarray:
    if name not in record:
        raise ValueError(f"Taylor model omits {name}")
    value = np.asarray(record[name], dtype=dtype)
    if value.ndim != ndim or not np.isfinite(value).all():
        raise ValueError(f"{name} has invalid shape or values")
    return np.ascontiguousarray(value)


def _integer_array(record: Any, name: str, ndim: int) -> np.ndarray:
    if name not in record:
        raise ValueError(f"Taylor model omits {name}")
    value = np.asarray(record[name])
    if (
        value.ndim != ndim
        or not np.issubdtype(value.dtype, np.integer)
    ):
        raise ValueError(f"{name} must be an integer array")
    return np.ascontiguousarray(value.astype(np.int64, copy=False))


def _single(record: Any, name: str, kind: type) -> Any:
    if name not in record:
        raise ValueError(f"source artifact omits {name}")
    value = np.asarray(record[name])
    if value.shape != (1,):
        raise ValueError(f"{name} must contain one update value")
    return kind(value[0].item())


def _source_metadata(
    path: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as record:
        if _scalar(record, "schema_version", str) != SOURCE_SCHEMA:
            raise ValueError("unknown contrast-energy source schema")
        if _scalar(record, "successor_evaluated", bool):
            raise ValueError(
                "a deciding source artifact already contains a successor")
        step_start = _scalar(record, "step_start", int)
        optimizer_steps = _array(
            record, "optimizer_step_after", 1, np.int64)
        if (
            len(optimizer_steps) != 1
            or int(optimizer_steps[0]) != step_start + 1
        ):
            raise ValueError(
                "contrast-energy source must contain one native update")
        normalized = _array(record, "normalized_rows", 3)
        gate = _array(record, "gate", 2)
        physical = _array(record, "physical_rows", 3)
        physical_jvp = _array(record, "physical_row_jvp", 3)
        if (
            normalized.shape[0] != 1
            or gate.shape[0] != 1
            or physical.shape[0] != 1
            or physical_jvp.shape[0] != 1
            or physical.shape[1:] != physical_jvp.shape[1:]
        ):
            raise ValueError(
                "source artifact must contain exactly one source state")
        cursor_before = _scalar(record, "data_cursor_before", int)
        cursor_after = _scalar(record, "data_cursor_after", int)
        chunk_indices = _array(
            record, "update_batch_chunk_indices", 2, np.int64)
        if (
            cursor_after <= cursor_before
            or chunk_indices.shape[0] != 1
            or chunk_indices.shape[1] != cursor_after - cursor_before
        ):
            raise ValueError("source cursor and exact minibatch disagree")
        role = _scalar(record, "role", str)
        seed = _scalar(record, "seed", int)
        layer = _scalar(record, "layer", int)
        metadata = {
            "campaign_id": _scalar(record, "campaign_id", str),
            "checkpoint_sha256": _scalar(
                record, "checkpoint_sha256", str),
            "registry_sha256": _scalar(record, "registry_sha256", str),
            "lock_sha256": _scalar(record, "lock_sha256", str),
            "data_order_sha256": _scalar(
                record, "data_order_sha256", str),
            "role": role,
            "seed": seed,
            "trajectory": f"{role}:seed{seed}",
            "layer": layer,
            "anchor": step_start,
            "global_step_before": step_start,
            "global_step_after": int(optimizer_steps[0]),
            "data_cursor_before": cursor_before,
            "data_cursor_after": cursor_after,
            "update_batch_chunk_indices": chunk_indices[0].tolist(),
            "update_batch_sha256": _single(
                record, "update_batch_sha256", str),
            "source_loss": _single(record, "source_loss", float),
            "raw_gradient_sha256": _single(
                record, "raw_gradient_sha256", str),
            "clipped_gradient_sha256": _single(
                record, "clipped_gradient_sha256", str),
            "optimizer_state_sha256": _single(
                record, "optimizer_state_sha256", str),
            "source_parameter_sha256": _single(
                record, "source_parameter_sha256", str),
            "parameter_displacement_sha256": _single(
                record, "parameter_displacement_sha256", str),
            "predicted_successor_parameter_sha256": _single(
                record, "predicted_successor_parameter_sha256", str),
        }
        return (
            metadata,
            np.ascontiguousarray(physical[0], dtype=np.float64),
            np.ascontiguousarray(physical_jvp[0], dtype=np.float64),
        )


def _reconstruct_source_terms(
    rows: np.ndarray,
    direction: np.ndarray,
    pairs: np.ndarray,
    supplied_energy: np.ndarray,
    supplied_slope: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if rows.shape != direction.shape or rows.ndim != 2 or len(rows) < 2:
        raise ValueError("source physical rows and JVP disagree")
    expected_count = len(rows) * (len(rows) - 1) // 2
    if (
        pairs.shape != (expected_count, 2)
        or supplied_energy.shape != (expected_count,)
        or supplied_slope.shape != (expected_count,)
    ):
        raise ValueError("Taylor model does not cover the complete pair registry")
    authoritative_energy = np.empty(expected_count, dtype=np.float64)
    authoritative_slope = np.empty(expected_count, dtype=np.float64)
    offset = 0
    for expected_pairs in unordered_pair_chunks(len(rows), 32_768):
        stop = offset + len(expected_pairs)
        if not np.array_equal(pairs[offset:stop], expected_pairs):
            raise ValueError(
                "Taylor-model pairs are not the complete lexicographic registry")
        contrast = (
            rows[expected_pairs[:, 0]] - rows[expected_pairs[:, 1]])
        tangent = (
            direction[expected_pairs[:, 0]]
            - direction[expected_pairs[:, 1]])
        authoritative_energy[offset:stop] = np.einsum(
            "pd,pd->p", contrast, contrast, optimize=True)
        authoritative_slope[offset:stop] = 2.0 * np.einsum(
            "pd,pd->p", contrast, tangent, optimize=True)
        offset = stop
    if offset != expected_count:
        raise AssertionError("pair reconstruction did not exhaust the registry")
    if not np.array_equal(supplied_energy, authoritative_energy):
        raise ValueError(
            "Taylor-model source energies disagree with captured physical rows")
    if not np.array_equal(supplied_slope, authoritative_slope):
        raise ValueError(
            "Taylor-model source slopes disagree with captured physical JVP")
    return authoritative_energy, authoritative_slope


def produce(
    source_path: str | Path,
    model_path: str | Path,
    lock_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate parent binding and produce all source-only deciding arrays."""

    source = Path(source_path).resolve()
    model = Path(model_path).resolve()
    source_digest = sha256_path(source)
    source_metadata, source_rows, source_direction = _source_metadata(source)
    lock = _load_lock(lock_path, source_metadata)
    with np.load(model, allow_pickle=False) as record:
        if _scalar(record, "schema_version", str) != INPUT_SCHEMA:
            raise ValueError("unknown contrast-energy Taylor-model schema")
        if _scalar(record, "source_sha256", str) != source_digest:
            raise ValueError("Taylor model is not bound to the source artifact")
        if _scalar(record, "successor_values_present", bool):
            raise ValueError("a source Taylor model cannot contain successor values")
        pairs = _integer_array(record, "pairs", 2)
        source_energy = _array(record, "source_energy", 1)
        source_slope = _array(record, "source_slope", 1)
        knots = _array(record, "knots", 1)
        endpoint_lower = _array(record, "endpoint_second_lower", 2)
        endpoint_upper = _array(record, "endpoint_second_upper", 2)
        third_modulus = _array(record, "third_modulus", 2)
        source_charge = _array(record, "source_charge", 1)
        arithmetic_charge = _array(record, "arithmetic_charge", 1)
        producer_metadata = json.loads(_scalar(record, "metadata_json", str))
    if (
        pairs.shape != (len(source_energy), 2)
        or source_slope.shape != source_energy.shape
        or source_charge.shape != source_energy.shape
        or arithmetic_charge.shape != source_energy.shape
    ):
        raise ValueError("Taylor-model pair arrays disagree")
    source_energy, source_slope = _reconstruct_source_terms(
        source_rows,
        source_direction,
        pairs,
        source_energy,
        source_slope,
    )
    if not isinstance(producer_metadata, dict):
        raise ValueError("Taylor-model metadata must be an object")
    required_metadata = {
        "model_order", "interval_backend", "explicit_layernorm",
        "rope_interval_enclosure", "adaptive_subdivision_rule",
        "taylor_policy_sha256", "backend_manifest_sha256",
        "source_parameter_sha256", "parameter_displacement_sha256",
        "predicted_successor_parameter_sha256",
    }
    if (
        not required_metadata.issubset(producer_metadata)
        or int(producer_metadata["model_order"]) != 3
        or producer_metadata["explicit_layernorm"] is not True
        or producer_metadata["rope_interval_enclosure"] is not True
        or not _is_sha256(producer_metadata["taylor_policy_sha256"])
        or not _is_sha256(producer_metadata["backend_manifest_sha256"])
        or producer_metadata["source_parameter_sha256"]
        != source_metadata["source_parameter_sha256"]
        or producer_metadata["parameter_displacement_sha256"]
        != source_metadata["parameter_displacement_sha256"]
        or producer_metadata["predicted_successor_parameter_sha256"]
        != source_metadata["predicted_successor_parameter_sha256"]
    ):
        raise ValueError("Taylor-model metadata omits a required validated route")
    if source_metadata["campaign_id"] == REGISTERED_CAMPAIGN_ID:
        if (
            producer_metadata["taylor_policy_sha256"]
            != digest_object(TAYLOR_POLICY)
            or producer_metadata["adaptive_subdivision_rule"]
            != TAYLOR_POLICY["adaptive_rule"]
            or not (
                TAYLOR_POLICY["initial_subdivisions"]
                <= len(knots) - 1
                <= TAYLOR_POLICY["maximum_subdivisions"]
            )
        ):
            raise ValueError("Taylor model changed the registered path policy")
        if (
            lock is not None
            and producer_metadata["backend_manifest_sha256"]
            != lock["taylor_backend_manifest_sha256"]
        ):
            raise ValueError(
                "held-out Taylor backend differs from the construction lock")

    cells = cell_second_bounds(
        endpoint_lower, endpoint_upper, third_modulus, knots)
    enclosure = integrate_pair_energy(
        source_energy,
        source_slope,
        cells["cell_lower"],
        cells["cell_upper"],
        knots,
        source_charge=source_charge,
        arithmetic_charge=arithmetic_charge,
    )
    metadata = {
        **source_metadata,
        "schema_version": OUTPUT_SCHEMA,
        "source_path": str(source),
        "source_sha256": source_digest,
        "taylor_model_path": str(model),
        "taylor_model_sha256": sha256_path(model),
        "construction_lock_verified": lock is not None,
        "successor_values_present": False,
        "frozen_before_successor": True,
        "pair_count": int(len(pairs)),
        "source_terms_reconstructed": True,
        "cell_count": int(len(knots) - 1),
        "taylor_producer_metadata": producer_metadata,
    }
    return {
        "schema_version": np.asarray(OUTPUT_SCHEMA),
        "pairs": pairs,
        "knots": knots,
        "endpoint_second_lower": endpoint_lower,
        "endpoint_second_upper": endpoint_upper,
        "third_modulus": cells["third_modulus"],
        "modulus_charge": cells["modulus_charge"],
        "cell_lower": cells["cell_lower"],
        "cell_upper": cells["cell_upper"],
        **enclosure,
        "source_sha256": np.asarray(source_digest),
        "taylor_model_sha256": np.asarray(sha256_path(model)),
        "successor_values_present": np.asarray(False),
        "metadata_json": np.asarray(json.dumps(
            metadata, sort_keys=True, separators=(",", ":"))),
    }


def write_npz(path: str | Path, arrays: dict[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--taylor-model", required=True)
    parser.add_argument("--lock")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    arrays = produce(
        arguments.source, arguments.taylor_model, arguments.lock)
    write_npz(arguments.output, arrays)
    print(json.dumps({
        "output": str(Path(arguments.output).resolve()),
        "pair_count": int(len(arrays["pairs"])),
        "source_sha256": arrays["source_sha256"].item(),
        "successor_values_present": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
