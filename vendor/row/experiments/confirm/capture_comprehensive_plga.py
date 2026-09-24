#!/usr/bin/env python3
"""Capture occupied iSwiGLU, power, curvature, score, and logit endpoints."""

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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXPERIMENTS))

from confirm.capture_chronological_plga import (  # noqa: E402
    _load_registry,
    _resources,
)
from confirm.comprehensive_campaign import (  # noqa: E402
    campaign_spec,
    validate_checkpoint_campaign_binding,
)
from confirm.comprehensive_lock import load_lock  # noqa: E402
from confirm.confirmation_artifacts import sha256_path  # noqa: E402
from confirm.gate_shape_evidence import write_npz_atomic  # noqa: E402
from confirm.observable_margin_live import (  # noqa: E402
    _plga_arrays,
    _rows,
    _token_matrix,
)
from confirm.row_map_live import (  # noqa: E402
    _load_raw_checkpoint,
    _model_from_raw,
)


RECORD_SCHEMA = "pldr-comprehensive-plga-ledger-v1"


def produce(arguments: argparse.Namespace) -> None:
    started = time.monotonic()
    spec = campaign_spec()
    device = torch.device(arguments.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    checkpoint_path = Path(arguments.checkpoint).resolve()
    checkpoint = _load_raw_checkpoint(
        checkpoint_path, require_optimizer=True, device="cpu")
    validate_checkpoint_campaign_binding(checkpoint)
    registry = _load_registry(arguments.registry)
    lock = load_lock(arguments.lock) if arguments.lock else None
    if arguments.role == "heldout" and lock is None:
        raise ValueError("held-out PLGA capture requires the construction lock")
    if arguments.role == "construction" and lock is not None:
        raise ValueError("construction PLGA capture must precede the lock")
    if int(arguments.seed) not in spec["trajectories"][arguments.role]:
        raise ValueError("PLGA seed is outside its prospective role")
    if checkpoint.get("measurement_registry") != registry:
        raise ValueError("checkpoint and fixed registry disagree")
    if sha256_path(arguments.tokens) != registry["dataset_sha256"]:
        raise ValueError("token archive disagrees with the registry")
    if int(checkpoint["config"]["seed"]) != int(arguments.seed):
        raise ValueError("checkpoint seed disagrees with the PLGA job")
    if int(checkpoint["step"]) not in spec["trajectories"]["anchors"]:
        raise ValueError("PLGA checkpoint is outside the registered anchors")
    layer = int(arguments.layer)
    if layer not in spec["trajectories"]["layer_indices"]:
        raise ValueError("PLGA layer is outside the registered architecture")

    model = _model_from_raw(checkpoint, device)
    matrix = _token_matrix(arguments.tokens, registry["context_length"])
    role_key = "construction" if arguments.role == "construction" else "validation"
    context_rows = registry[role_key]
    contexts = _rows(
        matrix, [int(row["chunk_index"]) for row in context_rows], device)
    context_ids = [str(row["id"]) for row in context_rows]
    heads = np.arange(len(contexts), dtype=np.int64) % spec["model"]["heads"]
    arrays = _plga_arrays(
        model,
        contexts,
        context_ids,
        heads,
        layer,
        energy_at_node=0.0,
        resistance=0.0,
    )
    resources = _resources(device, started)
    payload = {
        "schema_version": np.asarray(RECORD_SCHEMA),
        "campaign_id": np.asarray(spec["campaign_id"]),
        "campaign_spec_sha256": np.asarray(spec["spec_sha256"]),
        "role": np.asarray(arguments.role),
        "seed": np.asarray(arguments.seed, dtype=np.int64),
        "anchor": np.asarray(int(checkpoint["step"]), dtype=np.int64),
        "layer": np.asarray(layer, dtype=np.int64),
        "checkpoint_sha256": np.asarray(sha256_path(checkpoint_path)),
        "registry_sha256": np.asarray(registry["registry_sha256"]),
        "data_order_sha256": np.asarray(
            checkpoint["data_state"].get("data_order_sha256") or ""),
        "lock_sha256": np.asarray(lock["lock_sha256"] if lock else ""),
        "positive_floor": np.asarray(1e-9, dtype=np.float64),
        **arrays,
        **{
            f"resource_{name}": np.asarray(value)
            for name, value in resources.items()
        },
    }
    write_npz_atomic(arguments.output, **payload)
    print(json.dumps({
        "output": str(Path(arguments.output).resolve()),
        "role": arguments.role,
        "seed": int(arguments.seed),
        "anchor": int(checkpoint["step"]),
        "layer": layer,
        "cells": len(contexts),
    }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--lock")
    parser.add_argument("--role", choices=["construction", "heldout"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    produce(parser.parse_args())


if __name__ == "__main__":
    main()
