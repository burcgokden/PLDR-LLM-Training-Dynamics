#!/usr/bin/env python3
"""Replay registered complete-source-removal interventions."""

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

from confirm.causal_confirmation_specs import (  # noqa: E402
    CAMPAIGN_ID,
    LOCK_SCHEMA,
    REGISTRY,
    TRAJECTORIES,
)
from confirm.chronological_collapse import source_removal_response  # noqa: E402
from confirm.confirmation_artifacts import digest_object  # noqa: E402


RECORD_SCHEMA = "pldr-chronological-intervention-ledger-v1"
REPORT_SCHEMA = "pldr-causal-intervention-analysis-v1"
ARMS = (
    "baseline",
    "adaptive-source-removal",
    "decay-removal",
    "shape-direction-removal",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value)


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
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("causal construction lock must be an object")
    unsigned = dict(value)
    recorded = unsigned.pop("lock_sha256", None)
    if (
        value.get("schema_version") != LOCK_SCHEMA
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("constant_enlargement_after_lock") is not False
        or recorded != digest_object(unsigned)
        or float(value.get("intervention_minimum_effect", 0.0)) <= 0.0
    ):
        raise ValueError("causal construction lock is invalid")
    return value


def _relative_error(left: float, right: float) -> float:
    return abs(left - right) / max(
        abs(left), abs(right), np.finfo(np.float64).tiny)


def analyze_record(path: str | Path, lock: dict[str, Any]) -> dict[str, Any]:
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
        source_model_sha256 = _scalar(record, "source_model_sha256", str)
        source_optimizer_sha256 = _scalar(
            record, "source_optimizer_sha256", str)
        arm_names = tuple(_string_array(record, "arm_names", 1).tolist())
        source_gate = _array(record, "source_gate", 1)
        source_contrast = _array(record, "source_contrast", 1)
        baseline_gate = _array(record, "baseline_gate_successor", 1)
        adaptive_source = _array(record, "adaptive_source_direction", 1)
        energy = _array(record, "pair_energy_history", 2)
        batch_sha256 = _string_array(record, "update_batch_sha256", 2)
        optimizer_steps = _integer_array(record, "optimizer_step_after", 2)
        initial_model_sha256 = _string_array(
            record, "initial_model_sha256", 1)
        initial_optimizer_sha256 = _string_array(
            record, "initial_optimizer_sha256", 1)
        final_model_sha256 = _string_array(
            record, "final_model_sha256", 1)
        final_optimizer_sha256 = _string_array(
            record, "final_optimizer_sha256", 1)

    heldout = [row for row in TRAJECTORIES if row["role"] == "heldout"]
    trajectory_index = next((
        index for index, row in enumerate(heldout)
        if int(row["seed"]) == seed
        and int(row["data_offset_chunks"]) == data_offset
    ), None)
    anchors = tuple(REGISTRY["intervention_anchor_steps"])
    horizons = tuple(REGISTRY["intervention_horizons"])
    site_count = len(anchors) * len(horizons)
    expected_anchor = anchors[ordinal // len(horizons)] \
        if 0 <= ordinal < site_count else -1
    expected_horizon = horizons[ordinal % len(horizons)] \
        if 0 <= ordinal < site_count else -1
    expected_rng = 4_000_000 + (
        trajectory_index * site_count if trajectory_index is not None
        else -10_000) + ordinal
    expected_steps = np.arange(
        anchor + 1, anchor + update_count + 1, dtype=np.int64)
    digests = (
        initial_model_sha256, initial_optimizer_sha256,
        final_model_sha256, final_optimizer_sha256,
    )
    metadata_valid = (
        trajectory_index is not None
        and 0 <= ordinal < site_count
        and unit_id == f"cri-s{seed}-{ordinal:02d}"
        and anchor == expected_anchor
        and update_count == expected_horizon
        and future_start == 0
        and layer == ordinal % 3
        and context_index == ordinal % 16
        and head == context_index % 4
        and (row_left, row_right) == (0, 1)
        and rng_substream == expected_rng
        and lock_sha256 == lock["lock_sha256"]
        and minimum_effect == float(lock["intervention_minimum_effect"])
        and learning_rate > 0.0
        and all(_is_sha256(value) for value in (
            checkpoint_sha256, registry_sha256,
            source_model_sha256, source_optimizer_sha256))
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
        and all(value.shape == (len(ARMS),) for value in digests)
    )
    if not metadata_valid or not dimensions_valid:
        raise ValueError("intervention registration or dimensions changed")
    if not all(_is_sha256(value) for values in digests for value in values):
        raise ValueError("intervention arm digests are invalid")
    if not all(_is_sha256(value) for value in batch_sha256.reshape(-1)):
        raise ValueError("intervention minibatch digests are invalid")

    replay = source_removal_response(
        baseline_gate, source_contrast[None, :], adaptive_source,
        learning_rate=learning_rate,
    )
    predicted = float(replay["predicted_energy_response"][0])
    actual = float(energy[1, 1] - energy[0, 1])
    prediction_residual = _relative_error(predicted, stored_prediction)
    effect_residual = _relative_error(predicted, actual)
    scale = max(float(np.max(np.abs(energy[:, 0]), initial=0.0)), 1.0)
    initial_residual = float(np.max(
        np.abs(energy[:, 0] - energy[0, 0]), initial=0.0) / scale)
    same_batches = all(
        np.array_equal(batch_sha256[index], batch_sha256[0])
        for index in range(1, len(ARMS)))
    same_steps = all(
        np.array_equal(optimizer_steps[index], expected_steps)
        for index in range(len(ARMS)))
    identical_initial = (
        all(value == source_model_sha256 for value in initial_model_sha256)
        and all(value == source_optimizer_sha256
                for value in initial_optimizer_sha256)
    )
    predicted_sign = int(np.sign(predicted))
    actual_sign = int(np.sign(actual))
    effect_rule = (
        predicted_sign != 0
        and predicted_sign == actual_sign
        and abs(predicted) >= minimum_effect
        and abs(actual) >= minimum_effect
    )
    checks = {
        "prediction_replay": prediction_residual <= 1e-12,
        "source_removal_effect_replay": (
            effect_residual <= 256.0 * np.finfo(np.float32).eps),
        "identical_initial_model_and_optimizer": identical_initial,
        "identical_initial_pair_energy": (
            initial_residual <= 64.0 * np.finfo(np.float64).eps),
        "identical_update_minibatches": same_batches,
        "exact_optimizer_step_counters": same_steps,
    }
    qualified = all(checks.values()) and effect_rule
    return {
        "schema_version": REPORT_SCHEMA,
        "source": {"path": str(record_path), "sha256": _sha256(record_path)},
        "unit_id": unit_id,
        "ordinal": ordinal,
        "seed": seed,
        "data_offset_chunks": data_offset,
        "anchor": anchor,
        "horizon": update_count,
        "layer": layer,
        "context_index": context_index,
        "head": head,
        "row_pair": [row_left, row_right],
        "checkpoint_sha256": checkpoint_sha256,
        "registry_sha256": registry_sha256,
        "lock_sha256": lock_sha256,
        "predicted_response": predicted,
        "actual_response": actual,
        "prediction_actual_relative_residual": effect_residual,
        "signed_aligned_effect": predicted_sign * actual,
        "decay_removal_first_response": float(energy[2, 1] - energy[0, 1]),
        "shape_direction_removal_first_response": float(
            energy[3, 1] - energy[0, 1]),
        "checks": checks,
        "effect_rule": effect_rule,
        "decision": "SITE_QUALIFIED" if qualified else "SITE_NOT_QUALIFIED",
    }


def analyze_campaign(paths: list[str], lock_path: str | Path) -> dict[str, Any]:
    lock = _load_lock(lock_path)
    reports = [analyze_record(path, lock) for path in paths]
    heldout = [
        (int(row["seed"]), int(row["data_offset_chunks"]))
        for row in TRAJECTORIES if row["role"] == "heldout"
    ]
    site_count = (
        len(REGISTRY["intervention_anchor_steps"])
        * len(REGISTRY["intervention_horizons"])
    )
    expected = {
        (seed, offset, ordinal)
        for seed, offset in heldout for ordinal in range(site_count)
    }
    observed = {
        (row["seed"], row["data_offset_chunks"], row["ordinal"])
        for row in reports
    }
    successes = {
        seed: sum(row["decision"] == "SITE_QUALIFIED"
                  for row in reports if row["seed"] == seed)
        for seed, _offset in heldout
    }
    seed_means = np.asarray([
        np.mean([row["signed_aligned_effect"] for row in reports
                 if row["seed"] == seed])
        for seed, _offset in heldout
    ], dtype=np.float64)
    mean = float(np.mean(seed_means))
    standard_error = float(
        np.std(seed_means, ddof=1) / math.sqrt(len(seed_means)))
    critical_95_df3 = 3.182446305
    checks = {
        "complete_64_unit_registry": (
            len(reports) == len(expected) and observed == expected),
        "all_arm_isolation_and_replays_valid": all(
            all(row["checks"].values()) for row in reports),
        "all_source_removal_signs_confirmed": all(
            count == site_count for count in successes.values()),
    }
    return {
        "schema_version": REPORT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "lock_sha256": lock["lock_sha256"],
        "record_count": len(reports),
        "qualified_sites_by_seed": successes,
        "seed_clustered_signed_aligned_effect": {
            "seed_means": seed_means.tolist(),
            "mean": mean,
            "standard_error": standard_error,
            "two_sided_95_interval": [
                mean - critical_95_df3 * standard_error,
                mean + critical_95_df3 * standard_error,
            ],
            "clusters": len(seed_means),
        },
        "checks": checks,
        "reports": reports,
        "decision": "QUALIFIED" if all(checks.values()) else "NOT_QUALIFIED",
    }


def _write(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
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
    _write(arguments.output, report)
    print(json.dumps({
        "decision": report["decision"],
        "record_count": report["record_count"],
    }, sort_keys=True))
    if arguments.require_pass and report["decision"] != "QUALIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
