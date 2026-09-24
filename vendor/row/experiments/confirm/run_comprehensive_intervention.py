#!/usr/bin/env python3
"""Run exact two-arm, four-horizon adaptive-source-removal units."""

from __future__ import annotations

import argparse
import copy
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

from confirm.capture_comprehensive_confirmation import _load_data_order  # noqa: E402
from confirm.comprehensive_campaign import (  # noqa: E402
    campaign_spec,
    validate_checkpoint_campaign_binding,
)
from confirm.comprehensive_lock import load_lock  # noqa: E402
from confirm.confirmation_artifacts import (  # noqa: E402
    capture_rng_states,
    restore_rng_states,
    sha256_path,
)
from confirm.gate_shape_evidence import write_npz_atomic  # noqa: E402
from confirm.row_map_live import (  # noqa: E402
    _gate_parameter_name,
    _load_raw_checkpoint,
    _model_from_raw,
)
from confirm.run_chronological_intervention import (  # noqa: E402
    _batch_digest,
    _current_gate_source,
    _load_registry,
    _optimizer,
    _pair_state,
    _resources,
    _step,
    _token_matrix,
)


RECORD_SCHEMA = "pldr-comprehensive-intervention-ledger-v1"
ARMS = ("baseline", "adaptive_source_removal")


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
        raise ValueError("held-out intervention requires the construction lock")
    if arguments.role == "construction" and lock is not None:
        raise ValueError("construction intervention must precede the lock")
    if int(arguments.seed) not in spec["trajectories"][arguments.role]:
        raise ValueError("intervention seed is outside its prospective role")
    if checkpoint.get("measurement_registry") != registry:
        raise ValueError("checkpoint and fixed registry disagree")
    if sha256_path(arguments.tokens) != registry["dataset_sha256"]:
        raise ValueError("token archive disagrees with the registry")
    if int(checkpoint["config"]["seed"]) != int(arguments.seed):
        raise ValueError("checkpoint seed disagrees with the intervention job")
    if int(checkpoint["step"]) != 8000:
        raise ValueError("intervention source must be the registered update 8000")
    if not 0 <= arguments.ordinal < 12:
        raise ValueError("intervention ordinal must be in 0 through 11")

    ordinal = int(arguments.ordinal)
    updates = 4
    future_start = ordinal * updates
    layer = ordinal % 3
    role_key = "construction" if arguments.role == "construction" else "validation"
    context_rows = registry[role_key]
    context_index = ordinal % len(context_rows)
    head = context_index % 4
    left, right = 0, 63
    unit_id = f"cgs-{arguments.role}-s{arguments.seed}-{ordinal:02d}"

    matrix = _token_matrix(arguments.tokens, registry["context_length"])
    order = _load_data_order(
        arguments.data_order, checkpoint, matrix_rows=len(matrix))
    cursor = int(checkpoint["data_offset_end"])
    batch_size = int(checkpoint["config"]["batch"])
    if cursor + (future_start + updates) * batch_size > len(order):
        raise ValueError("intervention continuation leaves the frozen order")
    context_chunk = int(context_rows[context_index]["chunk_index"])
    probe_tokens = torch.from_numpy(
        matrix[context_chunk:context_chunk + 1].astype(np.int64, copy=True)
    ).to(device)

    source_model = _model_from_raw(checkpoint, device)
    source_optimizer = _optimizer(source_model, checkpoint)
    restore_rng_states(checkpoint["rng_states"])
    clip = float(checkpoint["config"]["clip"])
    for local_step in range(future_start):
        start = cursor + local_step * batch_size
        indices = order[start:start + batch_size]
        rows = torch.from_numpy(
            matrix[indices].astype(np.int64, copy=True)).to(device)
        _step(source_model, source_optimizer, rows, clip)
        source_optimizer.step()

    source_rng = capture_rng_states()
    source_gate, source_contrast, source_energy = _pair_state(
        source_model, probe_tokens, head, layer, left, right)
    gate_histories = []
    contrast_histories = []
    energy_histories = []
    batch_digests = []
    optimizer_steps = []
    removal_directions = []
    learning_rates = []

    for arm in ARMS:
        model, optimizer = copy.deepcopy((source_model, source_optimizer))
        restore_rng_states(copy.deepcopy(source_rng))
        gate_history = [source_gate]
        contrast_history = [source_contrast]
        energy_history = [source_energy]
        local_digests = []
        local_steps = []
        local_directions = []
        local_rates = []
        for arm_step in range(updates):
            start = cursor + (future_start + arm_step) * batch_size
            indices = np.asarray(order[start:start + batch_size], dtype=np.int64)
            batch_array = matrix[indices].astype(np.int64, copy=True)
            local_digests.append(_batch_digest(batch_array))
            rows = torch.from_numpy(batch_array).to(device)
            _step(model, optimizer, rows, clip)
            gate_name = _gate_parameter_name(layer)
            gate = dict(model.named_parameters())[gate_name]
            source_direction, eta, _decay, step_after = _current_gate_source(
                gate, optimizer)
            optimizer.step()
            if arm == "adaptive_source_removal":
                with torch.no_grad():
                    gate.add_(eta * source_direction)
                local_directions.append(
                    source_direction.detach().cpu().numpy().astype(
                        np.float32, copy=False))
                local_rates.append(eta)
            else:
                local_directions.append(np.zeros(
                    gate.numel(), dtype=np.float32))
                local_rates.append(eta)
            gate_after, contrast_after, energy_after = _pair_state(
                model, probe_tokens, head, layer, left, right)
            gate_history.append(gate_after)
            contrast_history.append(contrast_after)
            energy_history.append(energy_after)
            local_steps.append(step_after)
        gate_histories.append(gate_history)
        contrast_histories.append(contrast_history)
        energy_histories.append(energy_history)
        batch_digests.append(local_digests)
        optimizer_steps.append(local_steps)
        removal_directions.append(local_directions)
        learning_rates.append(local_rates)

    if batch_digests[0] != batch_digests[1]:
        raise RuntimeError("intervention arms consumed different minibatches")
    resources = _resources(device, started)
    payload = {
        "schema_version": np.asarray(RECORD_SCHEMA),
        "campaign_id": np.asarray(spec["campaign_id"]),
        "campaign_spec_sha256": np.asarray(spec["spec_sha256"]),
        "role": np.asarray(arguments.role),
        "unit_id": np.asarray(unit_id),
        "seed": np.asarray(arguments.seed, dtype=np.int64),
        "ordinal": np.asarray(ordinal, dtype=np.int64),
        "anchor": np.asarray(8000, dtype=np.int64),
        "future_update_start": np.asarray(future_start, dtype=np.int64),
        "update_count": np.asarray(updates, dtype=np.int64),
        "layer": np.asarray(layer, dtype=np.int64),
        "context_index": np.asarray(context_index, dtype=np.int64),
        "head": np.asarray(head, dtype=np.int64),
        "row_left": np.asarray(left, dtype=np.int64),
        "row_right": np.asarray(right, dtype=np.int64),
        "arm_names": np.asarray(ARMS),
        "source_gate": np.asarray(source_gate, dtype=np.float32),
        "source_contrast": np.asarray(source_contrast, dtype=np.float32),
        "source_energy": np.asarray(source_energy, dtype=np.float64),
        "gate_history": np.asarray(gate_histories, dtype=np.float32),
        "contrast_history": np.asarray(contrast_histories, dtype=np.float32),
        "pair_energy_history": np.asarray(energy_histories, dtype=np.float64),
        "adaptive_source_direction": np.asarray(
            removal_directions, dtype=np.float32),
        "learning_rate": np.asarray(learning_rates, dtype=np.float64),
        "update_batch_sha256": np.asarray(batch_digests),
        "optimizer_step_after": np.asarray(optimizer_steps, dtype=np.int64),
        "checkpoint_sha256": np.asarray(sha256_path(checkpoint_path)),
        "registry_sha256": np.asarray(registry["registry_sha256"]),
        "data_order_sha256": np.asarray(sha256_path(arguments.data_order)),
        "lock_sha256": np.asarray(lock["lock_sha256"] if lock else ""),
        **{
            f"resource_{name}": np.asarray(value)
            for name, value in resources.items()
        },
    }
    write_npz_atomic(arguments.output, **payload)
    print(json.dumps({
        "output": str(Path(arguments.output).resolve()),
        "unit_id": unit_id,
        "four_horizon_live_response": (
            np.asarray(energy_histories[1][1:])
            - np.asarray(energy_histories[0][1:])
        ).tolist(),
    }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--data-order", required=True)
    parser.add_argument("--lock")
    parser.add_argument("--role", choices=["construction", "heldout"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ordinal", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    produce(parser.parse_args())


if __name__ == "__main__":
    main()
