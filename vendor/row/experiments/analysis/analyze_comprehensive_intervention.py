#!/usr/bin/env python3
"""Analyze all four horizons of exact comprehensive intervention arms."""

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

from confirm.comprehensive_campaign import campaign_spec  # noqa: E402
from confirm.comprehensive_gate_shape import two_arm_shape_response  # noqa: E402
from confirm.comprehensive_lock import load_lock  # noqa: E402
from confirm.gate_shape_evidence import write_json_atomic  # noqa: E402


RECORD_SCHEMA = "pldr-comprehensive-intervention-ledger-v1"
REPORT_SCHEMA = "pldr-comprehensive-intervention-report-v1"


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _scalar(record: Any, name: str) -> Any:
    value = np.asarray(record[name])
    if value.ndim != 0:
        raise ValueError(f"{name} must be scalar")
    return value.item()


def analyze_record(path: str | Path, lock: dict[str, Any] | None) -> dict[str, Any]:
    spec = campaign_spec()
    source = Path(path).resolve()
    with np.load(source, allow_pickle=False) as record:
        if _scalar(record, "schema_version") != RECORD_SCHEMA:
            raise ValueError("unknown comprehensive intervention schema")
        if (
            _scalar(record, "campaign_id") != spec["campaign_id"]
            or _scalar(record, "campaign_spec_sha256") != spec["spec_sha256"]
        ):
            raise ValueError("intervention campaign identity changed")
        role = str(_scalar(record, "role"))
        seed = int(_scalar(record, "seed"))
        ordinal = int(_scalar(record, "ordinal"))
        layer = int(_scalar(record, "layer"))
        lock_sha256 = str(_scalar(record, "lock_sha256"))
        source_contrast = np.asarray(record["source_contrast"], dtype=np.float64)
        source_energy = float(_scalar(record, "source_energy"))
        gates = np.asarray(record["gate_history"], dtype=np.float64)
        contrasts = np.asarray(record["contrast_history"], dtype=np.float64)
        energies = np.asarray(record["pair_energy_history"], dtype=np.float64)
        batches = np.asarray(record["update_batch_sha256"]).astype(str)
        arms = np.asarray(record["arm_names"]).astype(str).tolist()
        resources = {
            name.removeprefix("resource_"): _scalar(record, name)
            for name in record.files if name.startswith("resource_")
        }
    if role not in {"construction", "heldout"}:
        raise ValueError("intervention role is invalid")
    if role == "heldout":
        if lock is None or lock_sha256 != lock["lock_sha256"]:
            raise ValueError("held-out intervention is not construction-lock bound")
    elif lock is not None or lock_sha256:
        raise ValueError("construction intervention must precede the lock")
    if (
        seed not in spec["trajectories"][role]
        or not 0 <= ordinal < 12
        or layer != ordinal % 3
        or arms != ["baseline", "adaptive_source_removal"]
        or gates.shape != (2, 5, spec["model"]["head_width"])
        or contrasts.shape != gates.shape
        or energies.shape != (2, 5)
        or batches.shape != (2, 4)
        or not np.array_equal(batches[0], batches[1])
    ):
        raise ValueError("intervention dimensions or arm binding changed")
    tolerance = float(spec["decisions"]["identity_relative_tolerance"])
    floor = float(spec["decisions"]["normalized_effect_floor"])
    horizons = []
    maximum_identity = 0.0
    maximum_shape_ratio = 0.0
    normalization = max(abs(source_energy), 1.0)
    for horizon in range(1, 5):
        value = two_arm_shape_response(
            source_contrast[None, :],
            gates[0, horizon],
            contrasts[0, horizon][None, :],
            gates[1, horizon],
            contrasts[1, horizon][None, :],
        )
        fixed = float(value["fixed_shape_response"][0])
        correction = float(value["shape_correction"][0])
        live = float(value["live_response"][0])
        residual = float(abs(value["identity_residual"][0]) / max(
            abs(fixed), abs(correction), abs(live), 1.0))
        energy_live = float(energies[1, horizon] - energies[0, horizon])
        energy_residual = abs(energy_live - live) / max(
            abs(energy_live), abs(live), 1.0)
        shape_ratio = abs(correction) / max(abs(fixed), floor)
        maximum_identity = max(maximum_identity, residual, energy_residual)
        maximum_shape_ratio = max(maximum_shape_ratio, shape_ratio)
        horizons.append({
            "horizon": horizon,
            "fixed_source_shape_response": fixed,
            "baseline_shape_work": float(value["baseline_shape_work"][0]),
            "removal_shape_work": float(value["removal_shape_work"][0]),
            "shape_correction": correction,
            "shape_correction_ratio": shape_ratio,
            "live_response": live,
            "normalized_live_response": live / normalization,
            "identity_relative_residual": max(residual, energy_residual),
        })
    return {
        "source": {"path": str(source), "sha256": _sha256(source)},
        "role": role,
        "seed": seed,
        "ordinal": ordinal,
        "layer": layer,
        "horizons": horizons,
        "maximum_identity_relative_residual": maximum_identity,
        "maximum_shape_correction_ratio": maximum_shape_ratio,
        "identity_qualified": maximum_identity <= tolerance,
        "resources": resources,
    }


def _cluster_interval(
    records: list[dict[str, Any]], *, samples: int = 20000,
) -> dict[str, Any]:
    seeds = sorted({row["seed"] for row in records})
    by_seed = []
    for seed in seeds:
        rows = [row for row in records if row["seed"] == seed]
        if len(rows) != 12:
            raise ValueError("each held-out seed must contribute 12 units")
        by_seed.append(np.mean([
            [h["normalized_live_response"] for h in row["horizons"]]
            for row in rows
        ], axis=0))
    matrix = np.asarray(by_seed, dtype=np.float64)
    generator = np.random.Generator(np.random.PCG64(390041))
    draws = np.empty((samples, 4), dtype=np.float64)
    for index in range(samples):
        selected = generator.integers(0, len(seeds), size=len(seeds))
        draws[index] = np.mean(matrix[selected], axis=0)
    alpha_per_horizon = 0.05 / 4.0
    lower = np.quantile(draws, alpha_per_horizon, axis=0)
    upper = np.quantile(draws, 1.0 - alpha_per_horizon, axis=0)
    return {
        "method": "seed-cluster PCG64 percentile with Bonferroni four-horizon adjustment",
        "samples": samples,
        "seed_count": len(seeds),
        "mean": np.mean(matrix, axis=0).tolist(),
        "simultaneous_lower": lower.tolist(),
        "simultaneous_upper": upper.tolist(),
    }


def analyze_campaign(
    paths: list[str], lock_path: str | Path | None,
) -> dict[str, Any]:
    spec = campaign_spec()
    lock = load_lock(lock_path) if lock_path else None
    records = [analyze_record(path, lock) for path in paths]
    if not records:
        raise ValueError("intervention analysis needs records")
    role = records[0]["role"]
    if any(row["role"] != role for row in records):
        raise ValueError("intervention analysis cannot mix roles")
    expected_seeds = spec["trajectories"][role]
    expected = {(seed, ordinal) for seed in expected_seeds for ordinal in range(12)}
    actual = {(row["seed"], row["ordinal"]) for row in records}
    complete = actual == expected and len(records) == len(expected)
    identities = all(row["identity_qualified"] for row in records)
    maximum_shape = max(
        row["maximum_shape_correction_ratio"] for row in records)
    checks = {"complete_registered_units": complete, "all_identities": identities}
    interval = None
    if role == "heldout":
        interval = _cluster_interval(records)
        floor = float(lock["normalized_effect_floor"])
        checks["shape_correction_lock"] = (
            maximum_shape <= float(lock["shape_correction_ratio_limit"]))
        checks["all_four_cluster_intervals_clear_effect_floor"] = all(
            bound > floor for bound in interval["simultaneous_lower"])
        decision = "QUALIFIED" if all(checks.values()) else "NOT_QUALIFIED"
    else:
        decision = (
            "CONSTRUCTION_SUMMARY" if all(checks.values())
            else "CONSTRUCTION_INVALID")
    return {
        "schema_version": REPORT_SCHEMA,
        "campaign_id": spec["campaign_id"],
        "campaign_spec_sha256": spec["spec_sha256"],
        "lock_sha256": lock["lock_sha256"] if lock is not None else "",
        "role": role,
        "record_count": len(records),
        "checks": checks,
        "maxima": {"shape_correction_ratio": maximum_shape},
        "cluster_interval": interval,
        "records": records,
        "decision": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", nargs="+")
    parser.add_argument("--lock")
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-pass", action="store_true")
    arguments = parser.parse_args()
    report = analyze_campaign(arguments.records, arguments.lock)
    write_json_atomic(arguments.output, report)
    print(json.dumps({
        "decision": report["decision"],
        "record_count": report["record_count"],
    }, sort_keys=True))
    if arguments.require_pass and report["decision"] not in {
        "QUALIFIED", "CONSTRUCTION_SUMMARY",
    }:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
