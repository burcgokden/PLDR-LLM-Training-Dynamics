"""Deterministic subprocess fixture for the registered two-device runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-device", type=int, required=True)
    parser.add_argument("--logical-device", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", required=True)
    arguments = parser.parse_args()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(arguments.physical_device):
        raise RuntimeError("runner did not bind the requested physical device")
    if arguments.logical_device != "cuda:0":
        raise RuntimeError("fixture expects the isolated logical device cuda:0")

    output = Path(arguments.output)
    metadata_output = Path(arguments.metadata_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)

    map_ids = np.asarray(
        [
            "c0.L0.H0",
            "c1.L0.H0",
            "c0.L1.H0",
            "c1.L1.H0",
        ]
    )
    update = np.arange(arguments.steps, dtype=np.float64)
    log_edges = np.stack(
        [
            -0.020
            - 0.001 * arguments.physical_device
            + 0.002 * np.sin(0.71 * update + map_index)
            for map_index in range(len(map_ids))
        ],
        axis=1,
    )
    log_energies = np.concatenate(
        [
            np.zeros((1, len(map_ids)), dtype=np.float64),
            np.cumsum(log_edges, axis=0),
        ],
        axis=0,
    )
    scales = np.arange(1, len(map_ids) + 1, dtype=np.float64)
    energies = (np.exp(log_energies) * scales[None, :])[None, :, :]
    trajectory_id = f"fixture-trajectory-{arguments.physical_device}"
    np.savez_compressed(
        output,
        energies=energies,
        steps=np.arange(arguments.steps + 1, dtype=np.int64),
        trajectory_ids=np.asarray([trajectory_id]),
        map_ids=map_ids,
    )

    fixture_source = Path(__file__).resolve()
    fixture_digest = _sha256(fixture_source)
    intervention_rows = [
        {
            "map_id": str(map_id),
            "constant_input_defect_norm": 0.0,
            "defect_to_output_norm_ratio": 0.0,
            "realized_secant_gain": 0.5,
            "maximum_sampled_directional_gain": 0.75,
            "decomposition_relative_residual": 0.0,
        }
        for map_id in map_ids
    ]
    metadata = {
        "schema_version": "pldr-row-rg-energy-producer-metadata-v2",
        "physical_device": arguments.physical_device,
        "logical_device": arguments.logical_device,
        "cuda_visible_devices": visible,
        "visible_gpu_count": 1,
        "gpu": "deterministic-fixture",
        "seed": arguments.seed,
        "wall_seconds": 0.0,
        "optimizer": {"updates": arguments.steps},
        "sources": {
            "producer_path": str(fixture_source),
            "producer_sha256": fixture_digest,
            "reference_model_path": str(fixture_source),
            "reference_model_sha256": fixture_digest,
            "reference_attention_path": str(fixture_source),
            "reference_attention_sha256": fixture_digest,
        },
        "data": {
            "shard_path": str(fixture_source),
            "shard_sha256": fixture_digest,
        },
        "capture": {
            "trajectory_id": trajectory_id,
            "map_ids": map_ids.tolist(),
            "states": arguments.steps + 1,
            "gate_shape_maximum_relative_residual": 0.0,
        },
        "plga_interventions_at_final_state": {
            "map_count": len(intervention_rows),
            "rows": intervention_rows,
        },
        "energy_artifact": {
            "path": str(output.resolve()),
            "sha256": _sha256(output),
            "size_bytes": output.stat().st_size,
        },
    }
    metadata_output.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
