#!/usr/bin/env python3
"""Replay archived scalar E3 families with correlated primitive corners.

This utility is deliberately separate from the registered campaign analyzer.
It produces an experiment-design diagnostic and never changes a sealed
campaign decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
CONFIRM = HERE.parent / "confirm"
sys.path.insert(0, str(CONFIRM))

from pathwise_contraction import (  # noqa: E402
    cross_metric_family_certificate,
    structured_family_certificate,
)
from validated_interval import as_fraction  # noqa: E402
from validated_linear_algebra import graph_window_certificate  # noqa: E402


SCHEMA_VERSION = "pldr-structured-e3-archive-replay-v2"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _measurement_map(payload):
    return {
        row["name"]: float(row["value"])
        for row in payload["measurements"]
        if row.get("status") == "OBSERVED"
    }


def replay_record(path, campaign_directory, window_length):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("certificate_box") != "outer":
        raise ValueError(f"structured replay requires an outer record: {path}")
    entry_path = Path(str(path).replace(
        "-outer/measurement.json", "-nominal/measurement.json"))
    if entry_path == path or not entry_path.is_file():
        raise ValueError(f"structured replay omits the paired entry record: {path}")
    entry_payload = json.loads(entry_path.read_text(encoding="utf-8"))
    if entry_payload.get("certificate_box") != "nominal":
        raise ValueError(
            f"structured replay entry has the wrong certificate box: {entry_path}")
    nominal = payload["nominal_matrix"]
    family = payload["family_matrices"]
    beta1 = float(nominal[1][1])
    certificate = structured_family_certificate(
        nominal,
        family,
        beta1,
        q_matrix=payload["construction"]["q_matrix"],
        window_length=window_length,
    )
    entry_nominal = entry_payload["nominal_matrix"]
    entry_certificate = structured_family_certificate(
        entry_nominal,
        entry_payload["family_matrices"],
        float(entry_nominal[1][1]),
        q_matrix=entry_payload["construction"]["q_matrix"],
        window_length=window_length,
    )
    cross_metric = cross_metric_family_certificate(
        entry_certificate["corner_matrices"],
        entry_certificate["construction"]["metric"],
        certificate["construction"]["metric"],
    )
    paired_graph = graph_window_certificate([
        {
            "edge_id": "entry_to_successor",
            "source": "entry",
            "target": "successor",
            "gain_upper": cross_metric["gain_upper"],
        },
        {
            "edge_id": "successor_stay",
            "source": "successor",
            "target": "successor",
            "gain_upper": certificate[
                "exact_vertex_certificate"]["gain_upper"],
        },
    ], window_length)
    measurements = _measurement_map(payload)
    return {
        "unit": path.parent.name,
        "source": path.relative_to(campaign_directory).as_posix(),
        "source_sha256": _sha256(path),
        "paired_entry_source": entry_path.relative_to(
            campaign_directory).as_posix(),
        "paired_entry_source_sha256": _sha256(entry_path),
        "legacy_entrywise_radius_q": measurements["robust_q"],
        "archived_observed_h_gain": measurements["observed_h_gain"],
        "structured_q": certificate["structured_q"],
        "structured_slack": 1.0 - certificate["structured_q"],
        "maximum_corner_h_gain": certificate["maximum_corner_h_gain"],
        "maximum_reconstruction_residual": (
            certificate["maximum_reconstruction_residual"]
        ),
        "maximum_primitive_box_ratio": (
            certificate["maximum_primitive_box_ratio"]
        ),
        "window_length": certificate["window_length"],
        "maximum_window_gain": certificate["maximum_window_gain"],
        "strictly_contracting": certificate["strictly_contracting"],
        "entry_to_successor_gain": float(as_fraction(
            cross_metric["gain_upper"])),
        "paired_maximum_window_gain": float(as_fraction(
            paired_graph["maximum_window_gain"])),
        "paired_path_count": paired_graph["admissible_path_count"],
        "primitive_bounds": certificate["primitive_bounds"],
        "exact_vertex_certificate": certificate["exact_vertex_certificate"],
        "graph_window_certificate": certificate["graph_window_certificate"],
        "exact_cross_metric_certificate": cross_metric,
        "paired_graph_window_certificate": paired_graph,
    }


def replay_campaign(campaign_directory, window_length=4):
    campaign_directory = Path(campaign_directory).resolve()
    records = sorted((campaign_directory / "raw" / "E3").glob(
        "*-outer/measurement.json"))
    if not records:
        raise ValueError("campaign contains no outer E3 measurements")
    units = [
        replay_record(path, campaign_directory, window_length)
        for path in records
    ]
    maximum_q = max(row["structured_q"] for row in units)
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": (
            "experiment-design diagnostic only; sealed campaign records and "
            "registered decisions are unchanged"
        ),
        "source_campaign_id": campaign_directory.name,
        "window_length": int(window_length),
        "aggregate": {
            "paired_unit_count": len(units),
            "all_exact_correlated_corner_certificates_contract": all(
                row["strictly_contracting"] for row in units
            ),
            "all_exact_cross_metric_windows_contract": all(
                row["paired_maximum_window_gain"] < 1.0 for row in units
            ),
            "maximum_structured_q": maximum_q,
            "minimum_structured_slack": 1.0 - maximum_q,
            "maximum_corner_h_gain": max(
                row["maximum_corner_h_gain"] for row in units
            ),
            "maximum_window_gain": max(
                row["maximum_window_gain"] for row in units
            ),
            "maximum_cross_metric_edge_gain": max(
                row["entry_to_successor_gain"] for row in units
            ),
            "maximum_paired_window_gain": max(
                row["paired_maximum_window_gain"] for row in units
            ),
            "paired_path_counts": sorted(set(
                row["paired_path_count"] for row in units
            )),
            "maximum_multiaffine_reconstruction_residual": max(
                row["maximum_reconstruction_residual"] for row in units
            ),
            "maximum_archived_observed_h_gain": max(
                row["archived_observed_h_gain"] for row in units
            ),
            "maximum_legacy_entrywise_radius_q": max(
                row["legacy_entrywise_radius_q"] for row in units
            ),
        },
        "units": units,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--window-length", type=int, default=4)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.window_length < 1:
        raise SystemExit("--window-length must be positive")
    result = replay_campaign(args.campaign_dir, args.window_length)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(output)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
