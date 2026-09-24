#!/usr/bin/env python3
"""Assemble one causal physical-row edge from a chronological live ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS))

from confirm.chronological_collapse import (  # noqa: E402
    adamw_chronological_direction,
)


SOURCE_SCHEMA = "pldr-chronological-pair-ledger-v1"
CERTIFICATE_SCHEMA = "pldr-causal-remainder-certificate-v1"
OUTPUT_SCHEMA = "pldr-causal-row-ledger-v1"


def _scalar(record: Any, name: str, kind: type) -> Any:
    if name not in record:
        raise ValueError(f"source ledger omits {name}")
    value = np.asarray(record[name])
    if value.shape != ():
        raise ValueError(f"{name} must be scalar")
    return kind(value.item())


def _array(record: Any, name: str, ndim: int) -> np.ndarray:
    if name not in record:
        raise ValueError(f"source ledger omits {name}")
    value = np.asarray(record[name], dtype=np.float64)
    if value.ndim != ndim or not np.isfinite(value).all():
        raise ValueError(f"{name} has invalid shape or values")
    return np.ascontiguousarray(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _load_certificate(path: str | Path) -> dict[str, Any]:
    certificate = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(certificate, dict)
        or certificate.get("schema_version") != CERTIFICATE_SCHEMA
        or certificate.get("successor_values_present") is not False
        or certificate.get("frozen_before_successor") is not True
        or not isinstance(
            certificate.get("row_remainder_radius_float64_outward"), list)
    ):
        raise ValueError("causal remainder certificate is invalid")
    unsigned = dict(certificate)
    recorded = unsigned.pop("certificate_sha256", None)
    digest = hashlib.sha256(json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()
    if recorded != digest:
        raise ValueError("causal remainder certificate digest does not replay")
    return certificate


def assemble(
    source_path: str | Path,
    certificate_path: str | Path,
    *,
    step_index: int,
    include_successor: bool,
) -> dict[str, Any]:
    ledger_path = Path(source_path).resolve()
    certificate_file = Path(certificate_path).resolve()
    certificate = _load_certificate(certificate_file)
    with np.load(ledger_path, allow_pickle=False) as record:
        if _scalar(record, "schema_version", str) != SOURCE_SCHEMA:
            raise ValueError("unknown chronological source ledger schema")
        role = _scalar(record, "role", str)
        seed = _scalar(record, "seed", int)
        data_offset_chunks = _scalar(record, "data_offset_chunks", int)
        layer = _scalar(record, "layer", int)
        step_start = _scalar(record, "step_start", int)
        beta1 = _scalar(record, "beta1", float)
        beta2 = _scalar(record, "beta2", float)
        epsilon = _scalar(record, "epsilon", float)
        normalized_rows = _array(record, "normalized_rows", 3)
        gate = _array(record, "gate", 2)
        gradient_history = _array(
            record, "clipped_gradient_history", 3)
        first_tail = _array(record, "first_moment_tail", 2)
        second_before = _array(record, "second_moment_before", 2)
        shape_jvp = _array(record, "normalized_row_jvp", 3)
        learning_rate = _array(record, "learning_rate", 1)
        weight_decay = _array(record, "weight_decay", 1)
        decay_mask = _array(record, "decay_mask", 1)
        optimizer_step_after = np.asarray(record["optimizer_step_after"])
        registry_sha256 = _scalar(record, "registry_sha256", str)
        checkpoint_sha256 = _scalar(record, "checkpoint_sha256", str)
        lock_sha256 = _scalar(record, "lock_sha256", str)
    local = int(step_index)
    transitions = shape_jvp.shape[0]
    if (
        transitions < 1
        or normalized_rows.shape[0] not in {transitions, transitions + 1}
        or gate.shape[0] != normalized_rows.shape[0]
    ):
        raise ValueError("chronological source and tangent lengths disagree")
    if not 0 <= local < transitions:
        raise ValueError("step_index is outside the chronological ledger")
    row_count = normalized_rows.shape[1]
    width = normalized_rows.shape[2]
    shape_radius = np.asarray(
        certificate["row_remainder_radius_float64_outward"],
        dtype=np.float64,
    )
    if shape_radius.shape != (row_count,) or np.any(shape_radius < 0.0):
        raise ValueError("certificate row radii disagree with the source ledger")
    source_digest = _sha256(ledger_path)
    if certificate.get("source_state_digest") != source_digest:
        raise ValueError("certificate is not bound to this source ledger")
    chronology = adamw_chronological_direction(
        gradient_history[local],
        first_tail[local],
        second_before[local],
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        optimizer_step_after=int(optimizer_step_after[local]),
    )
    gamma = gate[local]
    decayed = (
        1.0 - learning_rate[local] * weight_decay[local] * decay_mask
    ) * gamma
    predicted_gate = (
        decayed - learning_rate[local] * chronology["direction"])
    delta_gate = predicted_gate - gamma
    shape = normalized_rows[local]
    local_shape_jvp = shape_jvp[local]
    source_rows = shape * gamma
    row_jvp = (
        shape * delta_gate + local_shape_jvp * gamma)
    product_charge = np.linalg.norm(
        local_shape_jvp * delta_gate, axis=1)
    row_radius = (
        product_charge
        + float(np.max(np.abs(predicted_gate))) * shape_radius
    )
    metadata = {
        "source_schema": SOURCE_SCHEMA,
        "source_ledger_path": str(ledger_path),
        "source_ledger_sha256": source_digest,
        "role": role,
        "seed": seed,
        "data_offset_chunks": data_offset_chunks,
        "layer": layer,
        "global_step_before": step_start + local,
        "optimizer_step_after": int(optimizer_step_after[local]),
        "registry_sha256": registry_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "lock_sha256": lock_sha256,
        "row_representation": (
            "final_layernorm_bias_quotient_gamma_times_normalized_shape"),
        "jvp_routes": (
            "gate_times_shape_jvp_plus_gate_jvp_times_source_shape"),
        "remainder_routes": (
            "delta_gate_times_shape_jvp_plus_successor_gate_times_"
            "certified_shape_remainder"),
        "gate_route": (
            "source_history_to_first_moment_to_adaptive_innovation_"
            "to_predicted_gate"),
    }
    output: dict[str, Any] = {
        "schema_version": np.asarray(OUTPUT_SCHEMA),
        "certificate_schema": np.asarray(CERTIFICATE_SCHEMA),
        "certificate_sha256": np.asarray(certificate["certificate_sha256"]),
        "certificate_frozen_before_successor": np.asarray(True),
        "source_rows": source_rows,
        "row_jvp": row_jvp,
        "row_remainder_radius": row_radius,
        "gate_source": gamma,
        "gate_predicted": predicted_gate,
        "gate_clipped_gradient_history": gradient_history[local],
        "gate_first_moment_tail": first_tail[local],
        "gate_second_moment_before": second_before[local],
        "gate_beta1": np.asarray(beta1),
        "gate_beta2": np.asarray(beta2),
        "gate_adam_epsilon": np.asarray(epsilon),
        "gate_learning_rate": np.asarray(learning_rate[local]),
        "gate_weight_decay": np.asarray(weight_decay[local]),
        "gate_decay_mask": decay_mask,
        "gate_optimizer_step_after": np.asarray(
            int(optimizer_step_after[local]), dtype=np.int64),
        "metadata_json": np.asarray(json.dumps(
            metadata, sort_keys=True, separators=(",", ":"))),
    }
    if include_successor:
        if normalized_rows.shape[0] != transitions + 1:
            raise ValueError("source-only ledger has no successor values")
        output["successor_rows"] = (
            normalized_rows[local + 1] * gate[local + 1])
    if source_rows.shape != (row_count, width):
        raise AssertionError("assembled physical-row shape changed")
    return output


def write_npz(path: str | Path, arrays: dict[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_ledger")
    parser.add_argument("certificate")
    parser.add_argument("--step-index", type=int, required=True)
    parser.add_argument("--include-successor", action="store_true")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    arrays = assemble(
        arguments.source_ledger,
        arguments.certificate,
        step_index=arguments.step_index,
        include_successor=arguments.include_successor,
    )
    write_npz(arguments.output, arrays)
    print(json.dumps({
        "certificate_sha256": arrays["certificate_sha256"].item(),
        "output": str(Path(arguments.output).resolve()),
        "successor_included": "successor_rows" in arrays,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
