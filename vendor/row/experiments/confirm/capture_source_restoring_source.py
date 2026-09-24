#!/usr/bin/env python3
"""Capture source rows and signed JVP blocks before opening a successor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS))

from confirm.confirmation_artifacts import digest_object, sha256_path  # noqa: E402
from confirm.finite_increment_live import write_npz_atomic  # noqa: E402
from confirm.source_restoring_specs import (  # noqa: E402
    CAMPAIGN_ID,
    MASK_CONTRACT,
)
from confirm.source_restoring_live import (  # noqa: E402
    SOURCE_SCHEMA,
    capture_source_state,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--data-order", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    captured = capture_source_state(
        arguments.checkpoint,
        arguments.registry,
        arguments.tokens,
        arguments.data_order,
        arguments.device,
    )
    checkpoint = captured["checkpoint"]
    metadata = {
        "schema_version": SOURCE_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "created_at_ns": time.time_ns(),
        "step": int(checkpoint["step"]),
        "data_cursor_before": int(checkpoint["data_offset_end"]),
        "source_parameter_sha256": captured["source_parameter_sha256"],
        "source_loss": captured["source_loss"],
        "trajectory": captured["trajectory"],
        "registry_sha256": captured["registry"]["registry_sha256"],
        "dataset_sha256": captured["registry"]["dataset_sha256"],
        "tokenizer_sha256": captured["registry"]["tokenizer_sha256"],
        "mask_contract_sha256": digest_object(MASK_CONTRACT),
        "successor_checkpoint_opened": False,
        "successor_values_used": False,
        "map_order": "context-major-layer-major-head-major-v1",
        "parameter_partition": {
            "gate": "target-final-postnorm-affine",
            "shape": "target-prechain-and-residual-row-map-except-final-affine",
            "input": "all-remaining-model-parameters",
        },
        "maximum_jvp_additivity_residual": captured[
            "maximum_jvp_additivity_residual"
        ],
        "update_chunks": captured["update_chunks"],
        "contexts": captured["contexts"],
        "parents": captured["parents"],
        "resources": captured["resources"],
    }
    write_npz_atomic(
        arguments.output,
        schema_version=np.asarray(SOURCE_SCHEMA),
        metadata_json=np.asarray(json.dumps(
            metadata, sort_keys=True, separators=(",", ":")
        )),
        source_rows=captured["source_rows"],
        gate_velocity_rows=captured["gate_velocity_rows"],
        shape_velocity_rows=captured["shape_velocity_rows"],
        input_velocity_rows=captured["input_velocity_rows"],
    )
    print(json.dumps({
        "output": str(arguments.output.resolve()),
        "sha256": sha256_path(arguments.output),
        "map_count": int(captured["source_rows"].shape[0]),
        "successor_checkpoint_opened": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
