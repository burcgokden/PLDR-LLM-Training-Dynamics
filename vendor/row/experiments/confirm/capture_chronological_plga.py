#!/usr/bin/env python3
"""Capture occupied-segment PLGA, rectangular-score, and logit tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXPERIMENTS))

from confirm.chronological_confirmation_specs import (  # noqa: E402
    ARCHITECTURE,
    CAMPAIGN_ID,
    REGISTRY,
)
from confirm.confirmation_artifacts import (  # noqa: E402
    digest_object,
    sha256_path,
    validate_measurement_registry,
)
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


RECORD_SCHEMA = "pldr-chronological-plga-ledger-v1"
LOCK_SCHEMA = "pldr-chronological-construction-lock-v1"


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("JSON input must be an object")
    return value


def _load_registry(path: str | Path) -> dict[str, Any]:
    value = _load_json(path)
    validate_measurement_registry(value, require_nonempty=True)
    if value.get("schema_version") != "pldr-chronological-probe-registry-v1":
        raise ValueError("PLGA capture needs the chronological registry")
    return value


def _load_lock(path: str | Path) -> dict[str, Any]:
    value = _load_json(path)
    if (
        value.get("schema_version") != LOCK_SCHEMA
        or value.get("campaign_id") != CAMPAIGN_ID
    ):
        raise ValueError("unknown chronological construction lock")
    unsigned = dict(value)
    recorded = unsigned.pop("lock_sha256", None)
    if recorded != digest_object(unsigned):
        raise ValueError("chronological construction lock digest does not replay")
    if value.get("constant_enlargement_after_lock") is not False:
        raise ValueError("chronological construction lock is mutable")
    return value


def _resources(device: torch.device, started: float) -> dict[str, Any]:
    cuda = device.type == "cuda"
    return {
        "elapsed_seconds": time.monotonic() - started,
        "device": str(device),
        "peak_gpu_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if cuda else 0),
        "peak_gpu_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if cuda else 0),
        "peak_host_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            * (1024 if sys.platform != "darwin" else 1),
    }


def produce(arguments: argparse.Namespace) -> None:
    started = time.monotonic()
    device = torch.device(arguments.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    checkpoint_path = Path(arguments.checkpoint).resolve()
    registry_path = Path(arguments.registry).resolve()
    checkpoint = _load_raw_checkpoint(
        checkpoint_path, require_optimizer=True, device="cpu")
    registry = _load_registry(registry_path)
    lock = _load_lock(arguments.lock) if arguments.lock else None
    if arguments.role == "heldout" and lock is None:
        raise ValueError("held-out PLGA capture requires the construction lock")
    if arguments.role == "construction" and lock is not None:
        raise ValueError("construction PLGA capture precedes the lock")
    if checkpoint.get("measurement_registry") != registry:
        raise ValueError("checkpoint and chronological registry disagree")
    if sha256_path(arguments.tokens) != registry["dataset_sha256"]:
        raise ValueError("token archive disagrees with the registry")
    if int(checkpoint["config"]["seed"]) != int(arguments.seed):
        raise ValueError("checkpoint seed disagrees with the PLGA job")
    if int(checkpoint["step"]) not in REGISTRY["anchor_steps"]:
        raise ValueError("PLGA capture checkpoint is not a registered anchor")
    layer = int(arguments.layer)
    if not 0 <= layer < ARCHITECTURE["layers"]:
        raise ValueError("PLGA layer is outside the registered architecture")

    model = _model_from_raw(checkpoint, device)
    matrix = _token_matrix(arguments.tokens, registry["context_length"])
    role_key = "construction" if arguments.role == "construction" else "validation"
    context_rows = registry[role_key]
    contexts = _rows(
        matrix, [int(row["chunk_index"]) for row in context_rows], device)
    context_ids = [str(row["id"]) for row in context_rows]
    heads = np.arange(len(contexts), dtype=np.int64) % ARCHITECTURE["heads"]
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
        "campaign_id": np.asarray(CAMPAIGN_ID),
        "role": np.asarray(arguments.role),
        "seed": np.asarray(arguments.seed, dtype=np.int64),
        "data_offset_chunks": np.asarray(
            int(checkpoint["config"]["data_offset"]), dtype=np.int64),
        "anchor": np.asarray(int(checkpoint["step"]), dtype=np.int64),
        "layer": np.asarray(layer, dtype=np.int64),
        "checkpoint_sha256": np.asarray(sha256_path(checkpoint_path)),
        "registry_sha256": np.asarray(registry["registry_sha256"]),
        "lock_sha256": np.asarray(
            lock["lock_sha256"] if lock is not None else ""),
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
        "peak_gpu_reserved_bytes": resources["peak_gpu_reserved_bytes"],
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
