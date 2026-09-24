"""Deterministically seal construction choices before heldout trajectories."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from confirm.confirmation_artifacts import sha256_path
from confirm.source_resolved_live import _schema
from confirm.source_resolved_specs import (
    CAMPAIGN_ID,
    CONSTRUCTION_POLICY_SCHEMA,
    FROZEN_POLICY,
    REGISTRY,
)
from confirm.strict_schema import load_json, validate


def _windows(timecourse: dict[str, Any]) -> dict[str, int]:
    updates = [int(value) for value in timecourse["updates"]]
    windows = {
        name: int(value)
        for name, value in FROZEN_POLICY["intervention_window_updates"].items()
    }
    if not updates or any(
        value < min(updates) or value > max(updates)
        for value in windows.values()
    ):
        raise ValueError("frozen intervention window leaves the timecourse")
    return windows


def seal_construction_policy(
    design_path: str | Path,
    timecourse_path: str | Path,
    edge_paths: list[str | Path],
    *,
    protocol_directory: str | Path,
    created_at_ns: int | None = None,
) -> dict[str, Any]:
    """Select windows, a PLGA cap, and coarse inequalities from construction."""

    design = load_json(design_path)
    if design.get("campaign_id") != CAMPAIGN_ID:
        raise ValueError("construction policy design belongs to another campaign")
    timecourse = load_json(timecourse_path)
    validate(
        timecourse,
        _schema(protocol_directory, "timecourse.schema.json"),
    )
    if not timecourse["technical_valid"] or timecourse["role"] != "construction":
        raise ValueError("policy sealing needs a valid construction timecourse")
    edge_schema = _schema(protocol_directory, "edge.schema.json")
    edges = []
    edge_digests = []
    for raw_path in edge_paths:
        path = Path(raw_path).resolve()
        edge = load_json(path)
        validate(edge, edge_schema)
        if not edge["technical_valid"]:
            raise ValueError("construction policy received an invalid source edge")
        edges.append(edge)
        edge_digests.append(sha256_path(path))
    if not edges:
        raise ValueError("construction policy needs selected full source ledgers")
    source_names = edges[0]["map_records"][0]["source_names"]
    if any(
        record["source_names"] != source_names
        for edge in edges for record in edge["map_records"]
    ):
        raise ValueError("construction edges changed the source partition")
    work = np.asarray([
        record["radial_dissipation"]
        for edge in edges for record in edge["map_records"]
    ], dtype=np.float64)
    gram_sum = np.asarray([
        np.sum(record["charge_gram"])
        for edge in edges for record in edge["map_records"]
    ], dtype=np.float64)
    native_charge = np.asarray([
        max(0.0, record["native_energy_upper"] - record["endpoint_energy"])
        for edge in edges for record in edge["map_records"]
    ], dtype=np.float64)
    quantile = 0.10
    cap = timecourse["plga_selected_cap"]
    checks = {
        "construction_only": timecourse["role"] == "construction",
        "windows_derived_deterministically": True,
        "plga_cap_from_frozen_menu": cap in {0.05, 0.10, 0.20, 0.50, None},
        "edge_population_nonempty": bool(edges),
        "policy_digest_inputs_bound": True,
    }
    result = {
        "schema_version": CONSTRUCTION_POLICY_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "design_sha256": sha256_path(design_path),
        "construction_timecourse_sha256": sha256_path(timecourse_path),
        "construction_edge_sha256": edge_digests,
        "created_at_ns": int(time.time_ns() if created_at_ns is None else created_at_ns),
        "plga_selected_cap": cap,
        "window_updates": _windows(timecourse),
        "coarse_certificate": {
            "source_names": source_names,
            "radial_dissipation_lower": np.quantile(
                work, quantile, axis=0
            ).tolist(),
            "signed_gram_sum_upper": float(np.quantile(gram_sum, 1.0 - quantile)),
            "native_charge_upper": float(np.quantile(native_charge, 1.0 - quantile)),
            "selection_quantile": quantile,
        },
        "checks": checks,
        "technical_valid": all(checks.values()),
    }
    validate(
        result,
        _schema(protocol_directory, "construction_policy.schema.json"),
    )
    return result
