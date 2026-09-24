#!/usr/bin/env python3
"""Open a trainer-native next checkpoint after the source prediction is frozen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS))

from analysis.analyze_source_restoring_confirmation import (  # noqa: E402
    EDGE_SCHEMA,
    PREDICTION_SCHEMA,
    write_json_atomic,
)
from confirm.confirmation_artifacts import (  # noqa: E402
    digest_object,
    sha256_path,
)
from confirm.finite_increment_live import (  # noqa: E402
    load_checkpoint,
    registered_tokens,
)
from confirm.source_restoring import (  # noqa: E402
    CERTIFICATE_SCHEMA,
    cover_native_successor,
)
from confirm.source_restoring_live import (  # noqa: E402
    capture_physical_maps,
    load_registry,
    validate_source_checkpoint,
)
from confirm.source_restoring_specs import CAMPAIGN_ID  # noqa: E402
from confirm.row_map_live import _model_from_raw  # noqa: E402


def load_certificate(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as record:
        if str(np.asarray(record["schema_version"]).item()) != CERTIFICATE_SCHEMA:
            raise ValueError("unknown source-restoring certificate schema")
        result = {name: np.asarray(record[name]) for name in record.files}
    result["schema_version"] = CERTIFICATE_SCHEMA
    result["metadata"] = json.loads(str(result.pop("metadata_json").item()))
    result["successor_values_used"] = bool(result["successor_values_used"].item())
    result["technical_valid"] = bool(result["technical_valid"].item())
    result["scientific_outcome"] = tuple(
        str(value) for value in result["scientific_outcome"]
    )
    for key in ("map_count", "rows_per_map", "feature_dimension", "pairs_per_map"):
        result[key] = int(result[key].item())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--successor-checkpoint", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    prediction = json.loads(arguments.prediction.read_text(encoding="utf-8"))
    recorded_prediction_digest = prediction.get("prediction_sha256")
    unsigned_prediction = dict(prediction)
    unsigned_prediction.pop("prediction_sha256", None)
    if (
        prediction.get("schema_version") != PREDICTION_SCHEMA
        or prediction.get("campaign_id") != CAMPAIGN_ID
        or prediction.get("successor_values_used") is not False
        or prediction.get("technical_valid") is not True
        or recorded_prediction_digest != digest_object(unsigned_prediction)
        or not isinstance(prediction.get("created_at_ns"), int)
    ):
        raise ValueError("native coverage needs a frozen source-only prediction")
    if prediction.get("certificate_sha256") != sha256_path(arguments.certificate):
        raise ValueError("prediction-bound certificate changed")
    certificate = load_certificate(arguments.certificate)
    certificate_metadata = certificate["metadata"]
    source_metadata = certificate_metadata.get("source_metadata", {})
    parents = source_metadata.get("parents", {})
    source_artifact_path = Path(prediction.get("source_path", ""))
    if (
        certificate_metadata.get("campaign_id") != CAMPAIGN_ID
        or certificate_metadata.get("source_sha256")
        != prediction.get("source_sha256")
        or certificate_metadata.get("source_path")
        != str(source_artifact_path)
        or not source_artifact_path.is_file()
        or sha256_path(source_artifact_path) != prediction.get("source_sha256")
        or parents.get("checkpoint_path")
        != str(arguments.source_checkpoint.resolve())
        or parents.get("checkpoint_sha256")
        != sha256_path(arguments.source_checkpoint)
        or not isinstance(certificate_metadata.get("created_at_ns"), int)
        or not (
            source_metadata.get("created_at_ns")
            < certificate_metadata["created_at_ns"]
            < prediction["created_at_ns"]
        )
    ):
        raise ValueError("certificate lineage or creation order is invalid")
    registry = load_registry(arguments.registry)
    if (
        source_metadata.get("registry_sha256") != registry["registry_sha256"]
        or parents.get("registry_sha256") != sha256_path(arguments.registry)
        or parents.get("tokens_sha256") != sha256_path(arguments.tokens)
    ):
        raise ValueError("certificate registry or token lineage changed")
    source = load_checkpoint(
        arguments.source_checkpoint, require_optimizer=True, device="cpu"
    )
    validate_source_checkpoint(source, registry, arguments.tokens)
    if (
        int(source["step"]) != int(source_metadata.get("step", -1))
        or int(source["data_offset_end"])
        != int(source_metadata.get("data_cursor_before", -1))
    ):
        raise ValueError("source checkpoint state changed after certification")
    successor_opened_at_ns = max(
        time.time_ns(), int(prediction["created_at_ns"]) + 1
    )
    successor = load_checkpoint(
        arguments.successor_checkpoint, require_optimizer=True, device="cpu"
    )
    validate_source_checkpoint(successor, registry, arguments.tokens)
    if int(successor["step"]) != int(source["step"]) + 1:
        raise ValueError("native checkpoints are not consecutive")
    if (
        int(successor["data_offset_end"])
        != int(source["data_offset_end"]) + int(source["config"]["batch"])
        or successor["data_state"].get("data_order_sha256")
        != source["data_state"].get("data_order_sha256")
        or successor.get("run_identity") != source.get("run_identity")
    ):
        raise ValueError("native checkpoint lineage is not consecutive")
    device = torch.device(arguments.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device.index)
    rows, _contexts = registered_tokens(registry, arguments.tokens, device)
    model = _model_from_raw(successor, device)
    physical = capture_physical_maps(model, rows)
    coverage = cover_native_successor(certificate, physical)
    if not coverage["technical_valid"]:
        raise RuntimeError(
            "source enclosure does not cover the trainer-native successor"
        )
    multiplier = []
    for value in certificate["source_upper_multiplier"]:
        multiplier.append(None if not np.isfinite(value) else float(value))
    edge = {
        "schema_version": EDGE_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "step": int(source["step"]),
        "source_artifact_path": str(source_artifact_path),
        "source_artifact_sha256": prediction["source_sha256"],
        "registry_sha256": registry["registry_sha256"],
        "data_order_sha256": source["data_state"]["data_order_sha256"],
        "certificate_created_at_ns": certificate_metadata["created_at_ns"],
        "prediction_created_at_ns": prediction["created_at_ns"],
        "successor_opened_at_ns": successor_opened_at_ns,
        "source_checkpoint_path": str(arguments.source_checkpoint.resolve()),
        "source_checkpoint_sha256": sha256_path(arguments.source_checkpoint),
        "native_successor_checkpoint_path": str(
            arguments.successor_checkpoint.resolve()
        ),
        "native_successor_checkpoint_sha256": sha256_path(
            arguments.successor_checkpoint
        ),
        "prediction_sha256": sha256_path(arguments.prediction),
        "certificate_sha256": sha256_path(arguments.certificate),
        "technical_valid": True,
        "scientific_outcome": list(certificate["scientific_outcome"]),
        "source_diameter_squared": [
            float(value) for value in certificate["source_diameter_squared"]
        ],
        "source_diameter_squared_lower": [
            float(value)
            for value in certificate["source_diameter_squared_lower"]
        ],
        "source_diameter_squared_upper": [
            float(value)
            for value in certificate["source_diameter_squared_upper"]
        ],
        "source_upper_multiplier": multiplier,
        "source_successor_diameter_squared_upper": [
            float(value)
            for value in certificate["predicted_diameter_squared_upper"]
        ],
        "native_successor_diameter_squared": [
            float(value) for value in coverage["native_diameter_squared"]
        ],
        "native_successor_diameter_squared_lower": [
            float(value)
            for value in coverage["native_diameter_squared_lower"]
        ],
        "native_successor_diameter_squared_upper": [
            float(value)
            for value in coverage["native_diameter_squared_upper"]
        ],
        "certificate_coverage_fraction": coverage["coverage_fraction"],
        "native_peak_memory_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device.index))
            if device.type == "cuda" else 0
        ),
    }
    write_json_atomic(arguments.output, edge)
    print(json.dumps({
        "output": str(arguments.output.resolve()),
        "step": edge["step"],
        "coverage_fraction": edge["certificate_coverage_fraction"],
        "native_successor_is_saved_checkpoint": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
