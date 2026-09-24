#!/usr/bin/env python3
"""Replay every composite PLGA cell under the causal campaign contract."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS))

from analysis import analyze_chronological_plga as analyzer  # noqa: E402
from confirm.causal_confirmation_specs import (  # noqa: E402
    CAMPAIGN_ID,
    LOCK_SCHEMA,
    REGISTRY,
    TRAJECTORIES,
)


REPORT_SCHEMA = "pldr-causal-plga-analysis-v1"


def analyze_campaign(
    paths: list[str], lock_path: str | Path | None,
) -> dict[str, Any]:
    lock = analyzer._load_lock(lock_path)
    reports = [analyzer.analyze_record(path, lock) for path in paths]
    if not reports:
        raise ValueError("PLGA analysis needs at least one ledger")
    role = reports[0]["role"]
    if role not in {"construction", "heldout"} or any(
        row["role"] != role for row in reports
    ):
        raise ValueError("PLGA campaign analysis cannot mix roles")
    cells = {
        (row["seed"], row["data_offset_chunks"], row["anchor"], row["layer"])
        for row in reports
    }
    if len(cells) != len(reports):
        raise ValueError("PLGA campaign contains duplicate cells")
    trajectories = [row for row in TRAJECTORIES if row["role"] == role]
    expected = {
        (int(trajectory["seed"]), int(trajectory["data_offset_chunks"]),
         anchor, layer)
        for trajectory in trajectories
        for anchor in REGISTRY["anchor_steps"]
        for layer in range(3)
    }
    complete = cells == expected and len(reports) == len(expected)
    if role == "construction":
        checks = {
            "complete_51_cell_grid": complete,
            "all_identities_valid": all(
                row["decision"] == "CONSTRUCTION_SUMMARY"
                for row in reports),
        }
        decision = (
            "CONSTRUCTION_SUMMARY" if all(checks.values())
            else "CONSTRUCTION_INVALID")
    else:
        margin_by_seed = {
            int(trajectory["seed"]): sum(
                row["margin_invoked_cell_count"]
                for row in reports
                if row["seed"] == int(trajectory["seed"]))
            for trajectory in trajectories
        }
        checks = {
            "complete_204_cell_grid": complete,
            "all_cells_qualified": all(
                row["decision"] == "QUALIFIED" for row in reports),
            "nonempty_margin_set_per_seed": all(
                count > 0 for count in margin_by_seed.values()),
        }
        decision = "QUALIFIED" if all(checks.values()) else "NOT_QUALIFIED"
    return {
        "schema_version": REPORT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "role": role,
        "lock_sha256": lock["lock_sha256"] if lock is not None else None,
        "record_count": len(reports),
        "checks": checks,
        "maxima": {
            name: max(row["maxima"][name] for row in reports)
            for name in reports[0]["maxima"]
        },
        "reports": reports,
        "decision": decision,
    }


def main() -> None:
    analyzer.CAMPAIGN_ID = CAMPAIGN_ID
    analyzer.LOCK_SCHEMA = LOCK_SCHEMA
    analyzer.REGISTRY = REGISTRY
    analyzer.TRAJECTORIES = TRAJECTORIES
    analyzer.REPORT_SCHEMA = REPORT_SCHEMA
    analyzer.analyze_campaign = analyze_campaign
    analyzer.main()


if __name__ == "__main__":
    main()
