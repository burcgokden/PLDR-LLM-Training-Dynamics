#!/usr/bin/env python3
"""Strict mapwise analysis for the rev45 source-resolved campaign."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
ROOT = EXPERIMENTS.parent
sys.path.insert(0, str(EXPERIMENTS))

from confirm.confirmation_artifacts import sha256_path  # noqa: E402
from confirm.source_resolved_live import _schema, write_json_atomic  # noqa: E402
from confirm.source_resolved_specs import (  # noqa: E402
    ANALYSIS_SCHEMA,
    CAMPAIGN_ID,
    FROZEN_POLICY,
    REGISTRY,
    SCIENTIFIC_OUTCOMES,
)
from confirm.strict_schema import load_json, validate  # noqa: E402


DEFAULT_PROTOCOL = (
    ROOT / "experiments" / "protocols" / "source_resolved_confirmation"
)


def _outcome_count(counter: Counter[str]) -> dict[str, int]:
    return {name: int(counter[name]) for name in SCIENTIFIC_OUTCOMES}


def _row(
    identity: dict[str, Any],
    stage: str,
    trajectory: str,
    criterion: str,
    observed: float,
    outcome: str,
) -> dict[str, Any]:
    if outcome not in SCIENTIFIC_OUTCOMES:
        raise ValueError("analysis produced an unknown outcome")
    return {
        "map_id": str(identity["map_id"]),
        "context": str(identity["context"]),
        "layer": int(identity["layer"]),
        "head": int(identity["head"]),
        "stage": stage,
        "trajectory": trajectory,
        "criterion": criterion,
        "observed": float(observed),
        "outcome": outcome,
    }


def _artifact_rows(
    path: Path,
    kind: str,
    protocol: Path,
) -> list[dict[str, Any]]:
    value = load_json(path)
    validate(value, _schema(protocol, f"{kind}.schema.json"))
    technical = bool(value["technical_valid"])
    rows = []
    if kind == "edge":
        trajectory = str(value["trajectory"])
        for record in value["map_records"]:
            outcome = (
                "technically_invalid" if not technical
                else "unresolved" if not record["gain_defined"]
                else "confirmed" if record["source_gain"] < 1.0
                else "not_confirmed"
            )
            rows.append(_row(
                record, "Q1", trajectory, "one-edge-physical-energy-gain",
                record["source_gain"], outcome,
            ))
    elif kind == "mask_response":
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in value["responses"]:
            grouped[record["map_id"]].append(record)
        for records in grouped.values():
            maximum = max(record["row_response"] for record in records)
            outcome = (
                "technically_invalid" if not technical
                else "unresolved" if not value["smooth_full_state_eligible"]
                else "confirmed" if maximum < 1.0
                else "not_confirmed"
            )
            rows.append(_row(
                records[0], "Q2", "source-checkpoint",
                "maximum-registered-one-update-response", maximum, outcome,
            ))
    elif kind == "rank_tube":
        for record in value["per_map_rank"]:
            outcome = (
                "technically_invalid" if not technical
                else "confirmed" if record["rank_threshold_pass"]
                else "not_confirmed" if record["constraint_dimension_exact"]
                else "unresolved"
            )
            rows.append(_row(
                record, "Q3", "source-block",
                f"rank:{record['operator_scope']}",
                record["smallest_relative_singular_estimate"], outcome,
            ))
        stacked = value["stacked_rank"]
        stacked_outcome = (
            "technically_invalid" if not technical
            else "confirmed" if stacked["rank_threshold_pass"]
            else "unresolved"
        )
        rows.append(_row(
            stacked, "Q3", "source-block", "stacked-full-adamw-rank",
            stacked["smallest_relative_singular_estimate"], stacked_outcome,
        ))
    elif kind == "timecourse":
        trajectory = str(value["trajectory"])
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in value["map_records"]:
            grouped[record["map_id"]].append(record)
        if len(grouped) != REGISTRY["map_count"]:
            raise ValueError("timecourse map population is incomplete")
        cap = value["plga_selected_cap"]
        stage = "Q4" if value["role"] == "construction" else "Q6"
        for records in grouped.values():
            records.sort(key=lambda record: record["update"])
            final = records[-1]
            cumulative_gain = final["cumulative_gain"]
            gain_outcome = (
                "technically_invalid" if not technical
                else "unresolved" if cumulative_gain is None
                else "confirmed" if cumulative_gain < 1.0
                else "not_confirmed"
            )
            rows.append(_row(
                final, stage, trajectory, "finite-horizon-cumulative-gain",
                0.0 if cumulative_gain is None else cumulative_gain,
                gain_outcome,
            ))
            maximum_ratio = max(
                record["plga_quotient_ratio"] for record in records
            )
            plga_outcome = (
                "technically_invalid" if not technical
                else "unresolved" if cap is None
                else "confirmed" if maximum_ratio <= cap
                else "not_confirmed"
            )
            rows.append(_row(
                final, stage, trajectory,
                "frozen-plga-quotient-cap-over-full-timecourse",
                maximum_ratio, plga_outcome,
            ))
    elif kind == "intervention":
        for record in value["records"]:
            prediction = float(record["source_predicted_energy_change"])
            native = float(record["native_energy_change"])
            scale = max(abs(prediction), abs(native), 1.0)
            relative_residual = abs(native - prediction) / scale
            outcome = (
                "technically_invalid" if not technical
                else "confirmed"
                if relative_residual
                <= FROZEN_POLICY["intervention_prediction_relative_tolerance"]
                else "not_confirmed"
            )
            rows.append(_row(
                record, "Q5", "construction-intervention",
                f"paired-prediction-residual:{record['window']}"
                f":{record['intervention']}",
                relative_residual, outcome,
            ))
    else:
        raise ValueError(f"unsupported source-resolved analysis kind {kind}")
    return rows


def analyze(
    inputs: list[tuple[Path, str]],
    *,
    protocol_directory: str | Path,
) -> dict[str, Any]:
    """Validate inputs first, retain every map outcome, then aggregate."""

    if not inputs:
        raise ValueError("source-resolved analysis needs at least one artifact")
    protocol = Path(protocol_directory).resolve()
    rows = []
    digests = []
    seen = set()
    for raw_path, kind in inputs:
        path = raw_path.resolve()
        if path in seen:
            raise ValueError("analysis input membership contains duplicates")
        seen.add(path)
        rows.extend(_artifact_rows(path, kind, protocol))
        digests.append(sha256_path(path))
    total = Counter(row["outcome"] for row in rows)
    layers: dict[str, Counter[str]] = {
        str(index): Counter() for index in range(3)
    }
    heads: dict[str, Counter[str]] = {
        str(index): Counter() for index in range(4)
    }
    contexts: dict[str, Counter[str]] = defaultdict(Counter)
    stages: dict[str, Counter[str]] = defaultdict(Counter)
    trajectories: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        layers[str(row["layer"])][row["outcome"]] += 1
        heads[str(row["head"])][row["outcome"]] += 1
        contexts[row["context"]][row["outcome"]] += 1
        stages[row["stage"]][row["outcome"]] += 1
        trajectories[row["trajectory"]][row["outcome"]] += 1

    def groups(values: dict[str, Counter[str]]) -> list[dict[str, Any]]:
        return [
            {"key": key, "counts": _outcome_count(values[key])}
            for key in sorted(values)
        ]

    result = {
        "schema_version": ANALYSIS_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "input_sha256": digests,
        "map_record_count": len(rows),
        "outcome_counts": _outcome_count(total),
        "outcomes_by_layer": {
            key: _outcome_count(value) for key, value in layers.items()
        },
        "outcomes_by_head": {
            key: _outcome_count(value) for key, value in heads.items()
        },
        "outcomes_by_context": groups(contexts),
        "outcomes_by_stage": groups(stages),
        "outcomes_by_trajectory": groups(trajectories),
        "outcome_records": rows,
        "finite_horizon_only": True,
        "negative_outcomes_retained": True,
        "technical_valid": True,
    }
    if sum(result["outcome_counts"].values()) != len(rows):
        raise AssertionError("analysis outcome accounting is incomplete")
    validate(result, _schema(protocol, "analysis.schema.json"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--edges", nargs="*", type=Path, default=[])
    parser.add_argument("--mask-responses", nargs="*", type=Path, default=[])
    parser.add_argument("--rank-tubes", nargs="*", type=Path, default=[])
    parser.add_argument("--timecourses", nargs="*", type=Path, default=[])
    parser.add_argument("--interventions", nargs="*", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-valid", action="store_true")
    arguments = parser.parse_args()
    inputs = []
    for kind, paths in (
        ("edge", arguments.edges),
        ("mask_response", arguments.mask_responses),
        ("rank_tube", arguments.rank_tubes),
        ("timecourse", arguments.timecourses),
        ("intervention", arguments.interventions),
    ):
        inputs.extend((path, kind) for path in paths)
    report = analyze(inputs, protocol_directory=arguments.protocol_dir)
    write_json_atomic(arguments.output, report)
    print(json.dumps({
        "technical_valid": report["technical_valid"],
        "map_record_count": report["map_record_count"],
        "outcome_counts": report["outcome_counts"],
        "output": str(arguments.output.resolve()),
    }, sort_keys=True))
    if arguments.require_valid and not report["technical_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
