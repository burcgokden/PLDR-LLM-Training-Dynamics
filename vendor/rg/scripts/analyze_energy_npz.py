#!/usr/bin/env python3
"""Analyze a raw PLDR energy artifact under the exact RG contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from row_rgmap.analysis import (  # noqa: E402
    analyze_energy_array,
    file_sha256,
    seal_record,
)


REQUIRED_KEYS = {"energies", "steps", "trajectory_ids", "map_ids"}


def load(path: Path):
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != REQUIRED_KEYS:
            raise ValueError(
                f"artifact keys must be exactly {sorted(REQUIRED_KEYS)}"
            )
        energies = np.asarray(payload["energies"])
        steps = np.asarray(payload["steps"])
        trajectory_ids = np.asarray(payload["trajectory_ids"])
        map_ids = np.asarray(payload["map_ids"])
    if energies.dtype != np.dtype(np.float64):
        raise ValueError("energies must be native float64 values")
    if energies.ndim != 3:
        raise ValueError("energies must have shape (trajectory, time, map)")
    if energies.shape[0] < 1 or energies.shape[1] < 2 or energies.shape[2] < 1:
        raise ValueError("energies must contain trajectories, edges, and maps")
    if not np.isfinite(energies).all() or np.any(energies < 0.0):
        raise ValueError("energies must be finite and nonnegative")
    if steps.shape != (energies.shape[1],):
        raise ValueError("steps do not match the time dimension")
    if not np.issubdtype(steps.dtype, np.integer):
        raise ValueError("steps must use an integer dtype")
    if trajectory_ids.shape != (energies.shape[0],):
        raise ValueError("trajectory_ids do not match the trajectory dimension")
    if map_ids.shape != (energies.shape[2],):
        raise ValueError("map_ids do not match the map dimension")
    step_values = [int(value) for value in steps.tolist()]
    if any(right - left != 1 for left, right in zip(
        step_values, step_values[1:]
    )):
        raise ValueError("steps must be consecutive for exact fine-edge blocking")
    if len(np.unique(trajectory_ids)) != len(trajectory_ids):
        raise ValueError("trajectory_ids must be unique within an artifact")
    if len(np.unique(map_ids)) != len(map_ids):
        raise ValueError("map_ids must be unique")
    return energies, steps, trajectory_ids, map_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--block-sizes", default="1,2,4,8,16,32,64,128,256"
    )
    arguments = parser.parse_args()
    source = Path(arguments.input)
    energies, steps, trajectory_ids, map_ids = load(source)
    block_sizes = tuple(int(value) for value in arguments.block_sizes.split(","))
    analysis = analyze_energy_array(energies, block_sizes=block_sizes)
    record = seal_record(
        {
            "schema_version": "pldr-row-rg-energy-artifact-result-v2",
            "input": {
                "sha256": file_sha256(source),
                "first_step": int(steps[0]),
                "last_step": int(steps[-1]),
                "trajectory_ids": [
                    str(value) for value in trajectory_ids.tolist()
                ],
                "map_ids": [str(value) for value in map_ids.tolist()],
            },
            "analysis": analysis,
        }
    )
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
