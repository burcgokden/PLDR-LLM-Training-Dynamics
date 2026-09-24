#!/usr/bin/env python3
"""Capture a source-only native AdamW endpoint for every registered map."""

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
    prepare_transition,
    source_metadata,
    write_npz_atomic,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-order", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=("construction", "heldout"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    if arguments.step < 0:
        raise ValueError("source step must be nonnegative")
    bundle = arguments.bundle.resolve()
    if not bundle.is_dir():
        raise FileNotFoundError("the staged campaign bundle is missing")
    transition = prepare_transition(
        arguments.checkpoint,
        arguments.registry,
        arguments.tokens,
        arguments.data_order,
        arguments.device,
        execute_successor=False,
    )
    checkpoint_config = transition["checkpoint"]["config"]
    if int(checkpoint_config.get("seed", -1)) != arguments.seed:
        raise ValueError("checkpoint seed and requested trajectory disagree")
    if arguments.role not in str(checkpoint_config.get("name", "")):
        raise ValueError("checkpoint name and requested role disagree")
    if int(transition["checkpoint"]["step"]) != arguments.step:
        raise ValueError("checkpoint step and requested source step disagree")
    metadata = source_metadata(
        transition,
        role=arguments.role,
        seed=arguments.seed,
        step=arguments.step,
    )
    write_npz_atomic(
        arguments.output,
        schema_version=np.asarray(SOURCE_SCHEMA),
        metadata_json=np.asarray(json.dumps(
            metadata, sort_keys=True, separators=(",", ":")
        )),
        source_shape=transition["source_shape"],
        predicted_shape=transition["predicted_shape"],
        source_rows=transition["source_rows"],
        predicted_rows=transition["predicted_rows"],
        source_gate=transition["source_gate"],
        predicted_gate=transition["predicted_gate"],
    )
    print(json.dumps({
        "output": str(arguments.output.resolve()),
        "sha256": sha256_path(arguments.output),
        "maps": int(np.prod(transition["source_rows"].shape[:3])),
        "successor_evaluated": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
