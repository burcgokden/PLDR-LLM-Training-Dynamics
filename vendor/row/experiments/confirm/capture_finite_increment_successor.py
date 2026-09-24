#!/usr/bin/env python3
"""Replay a prediction-gated native successor from its immutable source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS))

from confirm.confirmation_artifacts import sha256_path  # noqa: E402
from confirm.finite_increment_live import (  # noqa: E402
    SOURCE_SCHEMA,
    SUCCESSOR_SCHEMA,
    prepare_transition,
    source_metadata,
    write_npz_atomic,
)


PREDICTION_SCHEMA = "pldr-finite-increment-prediction-v1"


def _scalar(record: object, name: str, kind: type) -> object:
    if name not in record:
        raise ValueError(f"source artifact omits {name}")
    value = np.asarray(record[name])
    if value.shape != ():
        raise ValueError(f"{name} must be scalar")
    return kind(value.item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-order", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=("construction", "heldout"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    prediction_path = arguments.prediction.resolve()
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    if (
        prediction.get("schema_version") != PREDICTION_SCHEMA
        or prediction.get("valid") is not True
        or prediction.get("successor_values_used") is not False
    ):
        raise ValueError("successor replay needs a valid source-only prediction")
    source_path = Path(prediction["source_path"]).resolve()
    if sha256_path(source_path) != prediction.get("source_sha256"):
        raise ValueError("prediction-bound source artifact changed")
    with np.load(source_path, allow_pickle=False) as source:
        if _scalar(source, "schema_version", str) != SOURCE_SCHEMA:
            raise ValueError("prediction names an unknown source schema")
        recorded_metadata = json.loads(_scalar(source, "metadata_json", str))
        recorded_arrays = {
            name: np.asarray(source[name])
            for name in (
                "source_shape", "predicted_shape", "source_rows",
                "predicted_rows", "source_gate", "predicted_gate",
            )
        }
    transition = prepare_transition(
        arguments.checkpoint,
        arguments.registry,
        arguments.tokens,
        arguments.data_order,
        arguments.device,
        execute_successor=True,
    )
    replay_metadata = source_metadata(
        transition,
        role=arguments.role,
        seed=arguments.seed,
        step=arguments.step,
    )
    replay_identity = dict(replay_metadata)
    recorded_identity = dict(recorded_metadata)
    replay_identity.pop("resources", None)
    recorded_identity.pop("resources", None)
    if replay_identity != recorded_identity:
        raise ValueError("successor replay did not reproduce source metadata")
    for name, replay_name in (
        ("source_shape", "source_shape"),
        ("predicted_shape", "predicted_shape"),
        ("source_rows", "source_rows"),
        ("predicted_rows", "predicted_rows"),
        ("source_gate", "source_gate"),
        ("predicted_gate", "predicted_gate"),
    ):
        if not np.array_equal(recorded_arrays[name], transition[replay_name]):
            raise ValueError(f"successor replay changed {name}")
    if (
        transition["native_parameter_max_abs_residual"] != 0.0
        or transition["actual_parameter_sha256"]
        != transition["predicted_parameter_sha256"]
    ):
        raise ValueError("native optimizer endpoint differs from the frozen clone")
    actual_shape = np.asarray(transition["actual_shape"])
    actual_rows = np.asarray(transition["actual_rows"])
    shape_residual = float(np.max(np.abs(
        actual_shape - transition["predicted_shape"]
    )))
    row_residual = float(np.max(np.abs(
        actual_rows - transition["predicted_rows"]
    )))
    metadata = {
        **recorded_metadata,
        "successor_evaluated": True,
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256_path(prediction_path),
        "source_sha256": sha256_path(source_path),
        "actual_parameter_sha256": transition["actual_parameter_sha256"],
        "native_parameter_max_abs_residual": transition[
            "native_parameter_max_abs_residual"
        ],
        "native_shape_max_abs_residual": shape_residual,
        "native_row_max_abs_residual": row_residual,
        "resources": transition["resources"],
    }
    write_npz_atomic(
        arguments.output,
        schema_version=np.asarray(SUCCESSOR_SCHEMA),
        metadata_json=np.asarray(json.dumps(
            metadata, sort_keys=True, separators=(",", ":")
        )),
        actual_shape=actual_shape,
        actual_rows=actual_rows,
    )
    print(json.dumps({
        "output": str(arguments.output.resolve()),
        "sha256": sha256_path(arguments.output),
        "native_parameter_max_abs_residual": transition[
            "native_parameter_max_abs_residual"
        ],
        "native_row_max_abs_residual": row_residual,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
