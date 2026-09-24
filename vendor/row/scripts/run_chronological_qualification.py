#!/usr/bin/env python3
"""Generate and analyze the deterministic rev38 algebra qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(EXPERIMENTS / "analysis"))

from analyze_chronological_confirmation import (  # noqa: E402
    RECORD_SCHEMA,
    analyze_record,
    write_json,
)
from confirm.chronological_collapse import (  # noqa: E402
    occupied_power_secant,
    rectangular_score_transfer,
    source_removal_response,
    unordered_pairs,
)
from confirm.chronological_confirmation_specs import CAMPAIGN_ID  # noqa: E402


def build_qualification_ledger(path: Path) -> None:
    rng = np.random.default_rng(380038)
    transitions = 4
    history_length = 48
    vertex_count = 10
    width = 6
    step_start = 2000
    beta1 = 0.9
    beta2 = 0.95
    epsilon = 1e-5
    learning_rate = np.full(transitions, 0.01)
    weight_decay = np.full(transitions, 0.1)
    decay_mask = np.ones(width)

    gradient = rng.uniform(0.2, 0.8, size=width)
    gradient_history = np.repeat(
        gradient[None, None, :], transitions, axis=0)
    gradient_history = np.repeat(
        gradient_history, history_length, axis=1)
    first_tail = np.empty((transitions, width))
    second_before = np.empty((transitions, width))
    for transition in range(transitions):
        step_after = step_start + transition + 1
        first_tail[transition] = (
            1.0 - beta1 ** (step_after - history_length)) * gradient
        second_before[transition] = (
            1.0 - beta2 ** (step_after - 1)) * gradient * gradient

    normalized_rows = np.empty((transitions + 1, vertex_count, width))
    normalized_rows[0] = rng.normal(scale=0.7, size=(vertex_count, width))
    row_jvp = np.empty((transitions, vertex_count, width))
    row_radius = np.zeros((transitions, vertex_count))
    for transition in range(transitions):
        row_jvp[transition] = -0.015 * normalized_rows[transition]
        normalized_rows[transition + 1] = (
            normalized_rows[transition] + row_jvp[transition])

    gate = np.empty((transitions + 1, width))
    gate[0] = rng.uniform(0.7, 1.3, size=width)
    direction = gradient / (np.abs(gradient) + epsilon)
    for transition in range(transitions):
        gate[transition + 1] = (
            (1.0 - learning_rate[transition] * weight_decay[transition])
            * gate[transition]
            - learning_rate[transition] * direction
        )

    pair_count = len(unordered_pairs(vertex_count))
    numerical_charge = np.full((transitions, pair_count), 1e-13)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray(RECORD_SCHEMA),
        campaign_id=np.asarray(CAMPAIGN_ID),
        role=np.asarray("development"),
        seed=np.asarray(7444, dtype=np.int64),
        data_offset_chunks=np.asarray(0, dtype=np.int64),
        step_start=np.asarray(step_start, dtype=np.int64),
        layer=np.asarray(0, dtype=np.int64),
        beta1=np.asarray(beta1),
        beta2=np.asarray(beta2),
        epsilon=np.asarray(epsilon),
        normalized_rows=normalized_rows,
        gate=gate,
        clipped_gradient_history=gradient_history,
        first_moment_tail=first_tail,
        second_moment_before=second_before,
        normalized_row_jvp=row_jvp,
        normalized_row_remainder_radius=row_radius,
        numerical_charge=numerical_charge,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        decay_mask=decay_mask,
        optimizer_step_after=np.arange(
            step_start + 1, step_start + transitions + 1, dtype=np.int64),
        update_batch_sha256=np.asarray(["a" * 64] * transitions),
        checkpoint_sha256=np.asarray("b" * 64),
        registry_sha256=np.asarray("c" * 64),
        lock_sha256=np.asarray(""),
        segment_grid=np.asarray([0.0, 0.5, 1.0]),
        shape_safety_factor=np.asarray(2.0),
        numerical_relative_coefficient=np.asarray(0.0),
        numerical_absolute_charge=np.asarray(1e-13),
        gate_update_max_abs_residual=np.zeros(transitions),
    )


def algebra_checks() -> dict[str, object]:
    rng = np.random.default_rng(380039)
    query_zero = rng.normal(size=(255, 64))
    query_one = rng.normal(size=(255, 64))
    generator_zero = rng.normal(size=(64, 64))
    generator_one = rng.normal(size=(64, 64))
    key_zero = rng.normal(size=(255, 64))
    key_one = rng.normal(size=(255, 64))
    score = rectangular_score_transfer(
        query_zero,
        query_one,
        generator_zero,
        generator_one,
        key_zero,
        key_one,
    )
    power = occupied_power_secant(
        np.asarray([0.2, 0.5, 1.0, 2.0]),
        np.asarray([0.3, 0.7, 1.0, 1.5]),
        np.asarray([-2.0, -0.5, 0.0, 2.5]),
    )
    contrasts = rng.normal(size=(9, 64))
    baseline = rng.normal(size=64)
    source = rng.normal(size=64)
    intervention = source_removal_response(
        baseline, contrasts, source, learning_rate=7.5e-4)
    return {
        "rectangular_score_shape": list(score["reference_score"].shape),
        "rectangular_score_relative_residual": (
            score["identity_relative_residual"]),
        "power_secant_relative_residual_max": float(
            np.max(power["identity_relative_residual"])),
        "intervention_nonzero_predictions": int(np.count_nonzero(
            intervention["predicted_magnitude"] > 0.0)),
        "checks": {
            "rectangular_score_float64": (
                score["identity_relative_residual"] <= 2e-14),
            "signed_power_float64": bool(
                np.max(power["identity_relative_residual"]) <= 2e-14),
            "source_removal_sign_and_magnitude": bool(np.all(
                intervention["predicted_magnitude"] >= 0.0)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--require-pass", action="store_true")
    arguments = parser.parse_args()
    output = Path(arguments.output_dir).resolve()
    ledger = output / "qualification-ledger.npz"
    report_path = output / "qualification.json"
    build_qualification_ledger(ledger)
    pair_report = analyze_record(ledger, pair_chunk_size=11)
    algebra = algebra_checks()
    qualified = (
        pair_report["decision"] == "QUALIFIED"
        and all(algebra["checks"].values())
    )
    report = {
        "schema_version": "pldr-chronological-qualification-v1",
        "ledger": {
            "path": str(ledger),
            "analysis": pair_report,
        },
        "algebra": algebra,
        "status": "PASS" if qualified else "FAIL",
    }
    write_json(report_path, report)
    print(json.dumps({
        "status": report["status"],
        "pair_decision": pair_report["decision"],
        "score_residual": algebra["rectangular_score_relative_residual"],
    }, sort_keys=True))
    if arguments.require_pass and not qualified:
        raise SystemExit("chronological qualification failed")


if __name__ == "__main__":
    main()
