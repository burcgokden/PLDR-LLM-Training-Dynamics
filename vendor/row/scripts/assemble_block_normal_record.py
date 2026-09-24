#!/usr/bin/env python3
"""Join a sealed source prediction to a later native endpoint record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from analysis.analyze_block_normal_confirmation import (  # noqa: E402
    write_json_atomic,
)
from confirm.block_normal import backward_metrics, block_prediction  # noqa: E402
from confirm.block_normal_specs import (  # noqa: E402
    BLOCK_RECORD_SCHEMA,
    CAMPAIGN_ID,
    NATIVE_EVIDENCE_SCHEMA,
    SOURCE_EVIDENCE_SCHEMA,
)
from confirm.native_determinism import sha256_path  # noqa: E402


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("block-normal artifact must contain one JSON object")
    return value


def assemble(
    source: dict[str, Any],
    native: dict[str, Any],
) -> dict[str, Any]:
    """Assemble one block record while checking temporal and content linkage."""

    source_required = {
        "schema_version",
        "campaign_id",
        "stage",
        "trajectory",
        "seed",
        "block_start",
        "block_end",
        "source_checkpoint_path",
        "source_checkpoint_sha256",
        "created_at_ns",
        "operators",
        "quadratic_coefficients",
        "forces",
        "radii",
        "initial_normal_upper",
        "comparison_bounds",
    }
    native_required = {
        "schema_version",
        "campaign_id",
        "source_prediction_sha256",
        "source_checkpoint_sha256",
        "native_checkpoint_path",
        "native_checkpoint_sha256",
        "opened_at_ns",
        "observations",
    }
    if set(source) != source_required or set(native) != native_required:
        raise ValueError("source or native block artifact has an invalid schema")
    if (
        source["schema_version"] != SOURCE_EVIDENCE_SCHEMA
        or native["schema_version"] != NATIVE_EVIDENCE_SCHEMA
        or source["campaign_id"] != CAMPAIGN_ID
        or native["campaign_id"] != CAMPAIGN_ID
    ):
        raise ValueError("block artifact campaign or schema is unknown")
    checkpoint = Path(source["source_checkpoint_path"]).resolve()
    native_checkpoint = Path(native["native_checkpoint_path"]).resolve()
    if (
        not checkpoint.is_file()
        or not native_checkpoint.is_file()
        or sha256_path(checkpoint) != source["source_checkpoint_sha256"]
        or sha256_path(native_checkpoint) != native["native_checkpoint_sha256"]
        or native["source_checkpoint_sha256"]
        != source["source_checkpoint_sha256"]
        or native["opened_at_ns"] <= source["created_at_ns"]
    ):
        raise ValueError("checkpoint linkage or source-before-native order fails")
    metrics = backward_metrics(
        source["operators"],
        np.eye(np.asarray(source["operators"][0]).shape[0]),
    )
    prediction = block_prediction(
        metrics["gains"],
        source["quadratic_coefficients"],
        source["forces"],
        source["radii"],
        source["initial_normal_upper"],
    )
    observations = {
        row["comparison_id"]: row for row in native["observations"]
    }
    if len(observations) != len(native["observations"]):
        raise ValueError("native comparison identifiers are duplicated")
    comparisons = []
    for bound in source["comparison_bounds"]:
        comparison_id = bound["comparison_id"]
        if comparison_id not in observations:
            raise ValueError("native endpoint misses a frozen comparison")
        observed = observations.pop(comparison_id)
        comparisons.append({
            "prediction": bound["prediction"],
            "map_id": bound["map_id"],
            "context": bound["context"],
            "layer": bound["layer"],
            "head": bound["head"],
            "observed_lower": observed["observed_lower"],
            "observed_upper": observed["observed_upper"],
            "bound_lower": bound["bound_lower"],
            "bound_upper": bound["bound_upper"],
        })
    if observations:
        raise ValueError("native endpoint contains an unregistered comparison")
    return {
        "schema_version": BLOCK_RECORD_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "stage": source["stage"],
        "trajectory": source["trajectory"],
        "seed": source["seed"],
        "block_start": source["block_start"],
        "block_end": source["block_end"],
        "source_checkpoint_path": str(checkpoint),
        "source_checkpoint_sha256": source["source_checkpoint_sha256"],
        "native_checkpoint_path": str(native_checkpoint),
        "native_checkpoint_sha256": native["native_checkpoint_sha256"],
        "source_prediction_sha256": native["source_prediction_sha256"],
        "prediction_created_at_ns": source["created_at_ns"],
        "native_endpoint_opened_at_ns": native["opened_at_ns"],
        "technical_valid": True,
        "source_values_only": True,
        "gains": metrics["gains"].tolist(),
        "quadratic_coefficients": list(source["quadratic_coefficients"]),
        "forces": list(source["forces"]),
        "radii": list(source["radii"]),
        "initial_normal_upper": source["initial_normal_upper"],
        "prefix_products": prediction["prefix_products"].tolist(),
        "force_contributions": (
            prediction["force_contributions"].tolist()
        ),
        "normal_envelope": prediction["envelope"].tolist(),
        "lyapunov_residuals": metrics["lyapunov_residuals"].tolist(),
        "metric_condition_numbers": (
            metrics["condition_numbers"].tolist()
        ),
        "invariant_radius_residuals": (
            prediction["invariant_radius_residuals"].tolist()
        ),
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    source = _load(arguments.source)
    native = _load(arguments.native)
    if native["source_prediction_sha256"] != sha256_path(arguments.source):
        raise ValueError("native artifact is not bound to the source file")
    record = assemble(source, native)
    write_json_atomic(arguments.output, record)
    print(json.dumps({
        "stage": record["stage"],
        "trajectory": record["trajectory"],
        "comparison_count": len(record["comparisons"]),
        "output": str(arguments.output.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
