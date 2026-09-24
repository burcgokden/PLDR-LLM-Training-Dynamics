#!/usr/bin/env python3
"""Independent replay and campaign decision for intervention ledgers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS))

from confirm.chronological_collapse import source_removal_response  # noqa: E402
from confirm.chronological_confirmation_specs import (  # noqa: E402
    CAMPAIGN_ID,
    REGISTRY,
    TRAJECTORIES,
)
from confirm.confirmation_artifacts import digest_object  # noqa: E402


RECORD_SCHEMA = "pldr-chronological-intervention-ledger-v1"
REPORT_SCHEMA = "pldr-chronological-intervention-analysis-v1"
LOCK_SCHEMA = "pldr-chronological-construction-lock-v1"
ARMS = (
    "baseline",
    "adaptive-source-removal",
    "decay-removal",
    "shape-direction-removal",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _scalar(record: Any, name: str, kind: type) -> Any:
    if name not in record:
        raise ValueError(f"intervention ledger omits {name}")
    value = np.asarray(record[name])
    if value.shape != ():
        raise ValueError(f"{name} must be scalar")
    return kind(value.item())


def _array(record: Any, name: str, ndim: int) -> np.ndarray:
    if name not in record:
        raise ValueError(f"intervention ledger omits {name}")
    value = np.asarray(record[name], dtype=np.float64)
    if value.ndim != ndim or not np.isfinite(value).all():
        raise ValueError(f"{name} has invalid shape or values")
    return value


def _integer_array(record: Any, name: str, ndim: int) -> np.ndarray:
    if name not in record:
        raise ValueError(f"intervention ledger omits {name}")
    value = np.asarray(record[name])
    if value.ndim != ndim or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"{name} must be an integer array")
    return value.astype(np.int64, copy=False)


def _string_array(record: Any, name: str, ndim: int) -> np.ndarray:
    if name not in record:
        raise ValueError(f"intervention ledger omits {name}")
    value = np.asarray(record[name])
    if value.ndim != ndim or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{name} must be a string array")
    return value.astype(str, copy=False)


def _load_lock(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != LOCK_SCHEMA
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("constant_enlargement_after_lock") is not False
    ):
        raise ValueError("unknown or mutable chronological construction lock")
    unsigned = dict(value)
    recorded = unsigned.pop("lock_sha256", None)
    if recorded != digest_object(unsigned):
        raise ValueError("chronological construction lock digest does not replay")
    if float(value.get("intervention_minimum_effect", 0.0)) <= 0.0:
        raise ValueError("intervention lock has no positive effect floor")
    return value


def _relative_error(left: float, right: float) -> float:
    return abs(left - right) / max(
        abs(left), abs(right), np.finfo(np.float64).tiny)


def analyze_record(
    path: str | Path, lock: dict[str, Any],
) -> dict[str, Any]:
    """Replay one four-arm intervention from raw ledger tensors."""

    record_path = Path(path).resolve()
    with np.load(record_path, allow_pickle=False) as record:
        if _scalar(record, "schema_version", str) != RECORD_SCHEMA:
            raise ValueError("unknown intervention ledger schema")
        if _scalar(record, "campaign_id", str) != CAMPAIGN_ID:
            raise ValueError("intervention campaign identity changed")
        unit_id = _scalar(record, "unit_id", str)
        ordinal = _scalar(record, "ordinal", int)
        rng_substream = _scalar(record, "rng_substream", int)
        seed = _scalar(record, "seed", int)
        data_offset = _scalar(record, "data_offset_chunks", int)
        anchor = _scalar(record, "anchor", int)
        future_start = _scalar(record, "future_update_start", int)
        update_count = _scalar(record, "update_count", int)
        layer = _scalar(record, "layer", int)
        context_index = _scalar(record, "context_index", int)
        head = _scalar(record, "head", int)
        row_left = _scalar(record, "row_left", int)
        row_right = _scalar(record, "row_right", int)
        learning_rate = _scalar(record, "learning_rate", float)
        stored_prediction = _scalar(
            record, "predicted_source_removal_response", float)
        minimum_effect = _scalar(
            record, "intervention_minimum_effect", float)
        checkpoint_sha256 = _scalar(record, "checkpoint_sha256", str)
        registry_sha256 = _scalar(record, "registry_sha256", str)
        lock_sha256 = _scalar(record, "lock_sha256", str)
        source_model_sha256 = _scalar(
            record, "source_model_sha256", str)
        source_optimizer_sha256 = _scalar(
            record, "source_optimizer_sha256", str)

        arm_names = tuple(_string_array(record, "arm_names", 1).tolist())
        source_gate = _array(record, "source_gate", 1)
        source_contrast = _array(record, "source_contrast", 1)
        baseline_gate = _array(record, "baseline_gate_successor", 1)
        adaptive_source = _array(
            record, "adaptive_source_direction", 1)
        energy = _array(record, "pair_energy_history", 2)
        batch_sha256 = _string_array(
            record, "update_batch_sha256", 2)
        optimizer_steps = _integer_array(
            record, "optimizer_step_after", 2)
        initial_model_sha256 = _string_array(
            record, "initial_model_sha256", 1)
        initial_optimizer_sha256 = _string_array(
            record, "initial_optimizer_sha256", 1)
        final_model_sha256 = _string_array(
            record, "final_model_sha256", 1)
        final_optimizer_sha256 = _string_array(
            record, "final_optimizer_sha256", 1)

    heldout = [
        row for row in TRAJECTORIES if row["role"] == "heldout"
    ]
    trajectory_index = next((
        index for index, row in enumerate(heldout)
        if int(row["seed"]) == seed
        and int(row["data_offset_chunks"]) == data_offset
    ), None)
    expected_steps = np.arange(
        anchor + future_start + 1,
        anchor + future_start + update_count + 1,
        dtype=np.int64,
    )
    expected_rng_substream = (
        3_800_000
        + (trajectory_index * REGISTRY["intervention_sites_per_seed"]
           if trajectory_index is not None else -10_000)
        + ordinal
    )
    digest_arrays = (
        initial_model_sha256,
        initial_optimizer_sha256,
        final_model_sha256,
        final_optimizer_sha256,
    )
    metadata_valid = (
        trajectory_index is not None
        and 0 <= ordinal < REGISTRY["intervention_sites_per_seed"]
        and unit_id == f"cci-s{seed}-{ordinal:02d}"
        and anchor == REGISTRY["intervention_anchor_step"]
        and future_start == ordinal * REGISTRY["intervention_steps"]
        and update_count == REGISTRY["intervention_steps"]
        and layer == ordinal % 3
        and context_index == ordinal % 16
        and head == context_index % 4
        and (row_left, row_right) == REGISTRY["intervention_pair"]
        and rng_substream == expected_rng_substream
        and lock_sha256 == lock["lock_sha256"]
        and minimum_effect == float(lock["intervention_minimum_effect"])
        and learning_rate > 0.0
        and _is_sha256(checkpoint_sha256)
        and _is_sha256(registry_sha256)
        and _is_sha256(source_model_sha256)
        and _is_sha256(source_optimizer_sha256)
    )
    dimensions_valid = (
        arm_names == ARMS
        and source_gate.shape == source_contrast.shape
        and source_gate.shape == baseline_gate.shape
        and source_gate.shape == adaptive_source.shape
        and source_gate.size == 64
        and energy.shape == (len(ARMS), update_count + 1)
        and batch_sha256.shape == (len(ARMS), update_count)
        and optimizer_steps.shape == (len(ARMS), update_count)
        and all(value.shape == (len(ARMS),) for value in digest_arrays)
    )
    if not metadata_valid or not dimensions_valid:
        raise ValueError("intervention ledger registration or dimensions changed")
    if not all(
        _is_sha256(value)
        for values in digest_arrays for value in values
    ):
        raise ValueError("intervention arm digests are invalid")
    if not all(
        _is_sha256(value) for value in batch_sha256.reshape(-1)
    ):
        raise ValueError("intervention minibatch digests are invalid")

    replay = source_removal_response(
        baseline_gate,
        source_contrast[None, :],
        adaptive_source,
        learning_rate=learning_rate,
    )
    replay_prediction = float(replay["predicted_energy_response"][0])
    prediction_residual = _relative_error(
        replay_prediction, stored_prediction)
    actual_response = float(energy[1, 1] - energy[0, 1])
    effect_residual = _relative_error(
        replay_prediction, actual_response)
    initial_scale = max(
        float(np.max(np.abs(energy[:, 0]), initial=0.0)), 1.0)
    initial_energy_residual = float(
        np.max(np.abs(energy[:, 0] - energy[0, 0]), initial=0.0)
        / initial_scale
    )
    same_batch_stream = all(
        np.array_equal(batch_sha256[index], batch_sha256[0])
        for index in range(1, len(ARMS))
    )
    same_step_stream = all(
        np.array_equal(optimizer_steps[index], expected_steps)
        for index in range(len(ARMS))
    )
    identical_initial_state = (
        all(value == source_model_sha256 for value in initial_model_sha256)
        and all(
            value == source_optimizer_sha256
            for value in initial_optimizer_sha256
        )
    )
    predicted_sign = int(np.sign(replay_prediction))
    actual_sign = int(np.sign(actual_response))
    effect_rule = (
        predicted_sign != 0
        and predicted_sign == actual_sign
        and abs(replay_prediction) >= minimum_effect
        and abs(actual_response) >= minimum_effect
    )
    checks = {
        "prediction_replay": prediction_residual <= 1e-12,
        "source_removal_effect_replay": (
            effect_residual <= 256.0 * np.finfo(np.float32).eps),
        "identical_initial_model_and_optimizer": identical_initial_state,
        "identical_initial_pair_energy": (
            initial_energy_residual <= 64.0 * np.finfo(np.float64).eps),
        "identical_update_minibatches": same_batch_stream,
        "exact_optimizer_step_counters": same_step_stream,
    }
    structure_qualified = all(checks.values())
    site_qualified = structure_qualified and effect_rule
    return {
        "schema_version": REPORT_SCHEMA,
        "source": {
            "path": str(record_path),
            "sha256": _sha256(record_path),
        },
        "unit_id": unit_id,
        "ordinal": ordinal,
        "seed": seed,
        "data_offset_chunks": data_offset,
        "anchor": anchor,
        "layer": layer,
        "context_index": context_index,
        "head": head,
        "row_pair": [row_left, row_right],
        "checkpoint_sha256": checkpoint_sha256,
        "registry_sha256": registry_sha256,
        "lock_sha256": lock_sha256,
        "minimum_effect": minimum_effect,
        "predicted_response": replay_prediction,
        "stored_prediction_relative_residual": prediction_residual,
        "actual_response": actual_response,
        "prediction_actual_relative_residual": effect_residual,
        "signed_aligned_effect": predicted_sign * actual_response,
        "decay_removal_first_response": float(
            energy[2, 1] - energy[0, 1]),
        "shape_direction_removal_first_response": float(
            energy[3, 1] - energy[0, 1]),
        "checks": checks,
        "effect_rule": effect_rule,
        "structure_qualified": structure_qualified,
        "decision": (
            "SITE_QUALIFIED" if site_qualified else "SITE_NOT_QUALIFIED"),
    }


def analyze_campaign(
    paths: list[str], lock_path: str | Path,
) -> dict[str, Any]:
    """Apply the fixed 10-of-12 rule to the complete held-out registry."""

    lock = _load_lock(lock_path)
    reports = [analyze_record(path, lock) for path in paths]
    heldout = [
        (int(row["seed"]), int(row["data_offset_chunks"]))
        for row in TRAJECTORIES if row["role"] == "heldout"
    ]
    expected = {
        (seed, offset, ordinal)
        for seed, offset in heldout
        for ordinal in range(REGISTRY["intervention_sites_per_seed"])
    }
    observed = {
        (row["seed"], row["data_offset_chunks"], row["ordinal"])
        for row in reports
    }
    successes = {
        seed: sum(
            row["decision"] == "SITE_QUALIFIED"
            for row in reports if row["seed"] == seed
        )
        for seed, _offset in heldout
    }
    aligned_seed_means = np.asarray([
        np.mean([
            row["signed_aligned_effect"]
            for row in reports if row["seed"] == seed
        ])
        for seed, _offset in heldout
    ], dtype=np.float64)
    grand_mean = float(np.mean(aligned_seed_means))
    standard_error = float(
        np.std(aligned_seed_means, ddof=1) / math.sqrt(len(heldout)))
    critical_95_df3 = 3.182446305
    checks = {
        "complete_48_unit_registry": (
            len(reports) == len(expected) and observed == expected),
        "all_arm_isolation_and_replays_valid": all(
            row["structure_qualified"] for row in reports),
        "ten_of_twelve_per_seed": all(
            count >= REGISTRY["intervention_positive_sites_required"]
            for count in successes.values()
        ),
    }
    return {
        "schema_version": REPORT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "lock_sha256": lock["lock_sha256"],
        "record_count": len(reports),
        "qualified_sites_by_seed": successes,
        "seed_clustered_signed_aligned_effect": {
            "seed_means": aligned_seed_means.tolist(),
            "mean": grand_mean,
            "standard_error": standard_error,
            "two_sided_95_interval": [
                grand_mean - critical_95_df3 * standard_error,
                grand_mean + critical_95_df3 * standard_error,
            ],
            "clusters": len(heldout),
        },
        "checks": checks,
        "reports": reports,
        "decision": "QUALIFIED" if all(checks.values()) else "NOT_QUALIFIED",
    }


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", nargs="+")
    parser.add_argument("--lock", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-pass", action="store_true")
    arguments = parser.parse_args()
    report = analyze_campaign(arguments.records, arguments.lock)
    write_json(arguments.output, report)
    print(json.dumps({
        "decision": report["decision"],
        "record_count": report["record_count"],
        "qualified_sites_by_seed": report["qualified_sites_by_seed"],
    }, sort_keys=True))
    if arguments.require_pass and report["decision"] != "QUALIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
