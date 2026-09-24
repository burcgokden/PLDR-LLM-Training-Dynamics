#!/usr/bin/env python3
"""Q0 qualification for the implemented PLDR edge and affine RG algebra."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call, jacrev, jvp


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXPERIMENTS))

from full_stack import build_registry  # noqa: E402
from independent_energy_checker import check_record  # noqa: E402
from independent_rg_checker import check_rg_flow  # noqa: E402
from pldr_model_v510 import PLDR_Model  # noqa: E402
from program_energy import (  # noqa: E402
    SCHEMA_VERSION,
    construct_program_energy_edge,
    digest_object,
    runtime_reduction_bound,
)
from program_energy_rg import construct_rg_flow, q0_rg_fixture  # noqa: E402


class ImplementedRowMap(nn.Module):
    """The exact ResLayerA composition used by one PLDR decoder layer."""

    def __init__(self, units):
        super().__init__()
        self.units = units

    def forward(self, row):
        value = row
        for unit in self.units:
            value = unit.ResUnit(value)
        return value


def _state_bytes(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _mask(sequence_length, device):
    return torch.triu(
        torch.ones(
            sequence_length, sequence_length, dtype=torch.float64,
            device=device,
        ),
        diagonal=1,
    )[None, None]


def _capture_physical_row(model, tokens):
    captured = []
    layer = model.decoder.dec_layers[0].mha1.layernorm1
    handle = layer.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output.detach()))
    inputs = tokens[:, :-1]
    model([inputs, _mask(inputs.shape[1], inputs.device)])
    handle.remove()
    if len(captured) != 1:
        raise RuntimeError("Q0 failed to capture the physical row source")
    return captured[0].reshape(-1, captured[0].shape[-1])[0].detach()


def build_q0_primitives(device="cpu"):
    """Take one real clipped-AdamW step and derive the complete tiny fixture."""

    torch.manual_seed(29001)
    device = torch.device(device)
    model = PLDR_Model(
        num_layers=1, d_model=8, num_heads=1, dff=12,
        input_vocab_size=32, A_dff=6, num_reslayerA=1, num_denseA=1,
        max_seq_len=8, device=device,
    ).to(device=device, dtype=torch.float64)
    model.train()
    tokens = torch.tensor(
        [[1, 4, 7, 3, 2], [2, 8, 5, 9, 1]],
        dtype=torch.long, device=device,
    )
    row = _capture_physical_row(model, tokens)
    row_map = ImplementedRowMap(
        model.decoder.dec_layers[0].mha1.reslayerAs)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-5, betas=(0.9, 0.95),
        eps=1e-5, weight_decay=0.01,
    )

    source_model_state = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }
    source_parameters = {
        name: parameter.detach().clone()
        for name, parameter in row_map.named_parameters()
    }

    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]
    logits, _, _, _ = model([inputs, _mask(inputs.shape[1], device)])
    loss = F.cross_entropy(logits.reshape(-1, 32), targets.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    target_model_state = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }
    target_parameters = {
        name: parameter.detach().clone()
        for name, parameter in row_map.named_parameters()
    }
    delta_parameters = {
        name: target_parameters[name] - source_parameters[name]
        for name in source_parameters
    }

    def psi(parameters):
        def phi(row_argument):
            return functional_call(row_map, parameters, (row_argument,))
        return jacrev(phi)(row).reshape(-1)

    source_stack = psi(source_parameters).detach()
    target_stack = psi(target_parameters).detach()
    _, actual_jvp = jvp(
        psi, (source_parameters,), (delta_parameters,))
    dense_tree = jacrev(psi)(source_parameters)
    dense_blocks = [
        dense_tree[name].reshape(source_stack.numel(), -1)
        for name in source_parameters
    ]
    dense_derivative = torch.cat(dense_blocks, dim=1)
    flat_delta = torch.cat([
        delta_parameters[name].reshape(-1) for name in source_parameters
    ])
    dense_jvp = dense_derivative @ flat_delta
    dense_jvp_error = float(torch.linalg.vector_norm(
        dense_jvp - actual_jvp))

    remainder = target_stack - source_stack - actual_jvp
    remainder_norm = float(torch.linalg.vector_norm(remainder))

    def path(second_parameter):
        parameters = {
            name: source_parameters[name]
                + second_parameter * delta_parameters[name]
            for name in source_parameters
        }
        return psi(parameters)

    second_norms = []
    for point in (0.0, 0.25, 0.5, 0.75, 1.0):
        scalar = torch.tensor(point, dtype=torch.float64, device=device)

        def first(argument):
            return jvp(
                path, (argument,), (torch.ones_like(argument),))[1]

        second = jvp(
            first, (scalar,), (torch.ones_like(scalar),))[1]
        second_norms.append(float(torch.linalg.vector_norm(second)))
    sampled_integral_upper = math.nextafter(
        0.5 * max(second_norms)
        + runtime_reduction_bound(
            np.concatenate([
                source_stack.detach().cpu().numpy(),
                target_stack.detach().cpu().numpy(),
            ]),
            operation_count=max(1, dense_derivative.numel() * 4),
        ),
        math.inf,
    )

    source_digest = _state_bytes(source_model_state)
    target_digest = _state_bytes(target_model_state)
    owner_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    registry = build_registry({0: [("q0_physical_row", 8)]})
    completeness = [
        [0, "q0_physical_row", output, input_index]
        for output in range(8)
        for input_index in range(8)
    ]
    zero_error = {
        "observed_norm": 0.0,
        "certified_upper": runtime_reduction_bound(
            source_stack.detach().cpu().numpy(),
            operation_count=source_stack.numel() * 4,
        ),
        "derivation": "binary64_operation_count_and_absolute_input_sum",
    }
    primitives = {
        "schema_version": SCHEMA_VERSION,
        "edge_id": "q0-real-pldr-adamw-step",
        "source_state_sha256": source_digest,
        "target_state_sha256": target_digest,
        "successor_owner_sha256": owner_digest,
        "registry": registry,
        "source_stack": source_stack.detach().cpu().tolist(),
        "target_stack": target_stack.detach().cpu().tolist(),
        "actual_update_jvp": actual_jvp.detach().cpu().tolist(),
        "metric_blocks": [{
            "block_index": 0,
            "kind": "dense_tiny_fixture",
            "matrix": np.eye(64).tolist(),
        }],
        "primitive_errors": {
            "taylor_remainder": {
                "observed_norm": remainder_norm,
                "certified_upper": sampled_integral_upper,
                "derivation":
                    "dense_directional_second_derivative_segment_quadrature",
            },
            "row_cover_motion": dict(zero_error),
            "metric_conversion": dict(zero_error),
            "runtime_roundoff": {
                "observed_norm": dense_jvp_error,
                "certified_upper": runtime_reduction_bound(
                    np.concatenate([
                        dense_jvp.detach().cpu().numpy(),
                        actual_jvp.detach().cpu().numpy(),
                    ]),
                    operation_count=max(1, dense_derivative.numel() * 8),
                ),
                "derivation":
                    "dense_and_matrix_free_binary64_operation_count",
            },
            "exogenous_input": dict(zero_error),
        },
        "block_completeness": completeness,
        "construction_partition_sha256":
            hashlib.sha256(b"q0-construction-row").hexdigest(),
        "validation_partition_sha256":
            hashlib.sha256(b"q0-validation-row").hexdigest(),
    }
    diagnostics = {
        "dense_jvp_error": dense_jvp_error,
        "taylor_remainder_norm": remainder_norm,
        "sampled_integral_upper": sampled_integral_upper,
        "source_state_sha256": source_digest,
        "target_state_sha256": target_digest,
        "successor_owner_sha256": owner_digest,
    }
    return primitives, diagnostics


def qualify(device="cpu"):
    primitives, diagnostics = build_q0_primitives(device)
    record = construct_program_energy_edge(primitives)
    replay = check_record(record)
    rg_flow = construct_rg_flow(q0_rg_fixture(), block_factor=2)
    rg_replay = check_rg_flow(rg_flow)
    conditions = {
        "constructor_qualified": record["decision"] == "QUALIFIED",
        "independent_replay": replay["decision"] == "REPLAYED",
        "dense_matrix_free_enclosed":
            record["primitive_errors"]["runtime_roundoff"]["enclosed"],
        "actual_successor_bound":
            primitives["source_state_sha256"]
            != primitives["target_state_sha256"],
        "rg_affine_map_and_flow": rg_flow["decision"] == "QUALIFIED",
        "rg_independent_replay": rg_replay["decision"] == "REPLAYED",
        "rg_temporal_order_witness": bool(
            rg_flow["diagnostics"]["first_pair_order_sensitivity"]
            > rg_flow["diagnostics"]["semigroup_tolerance"]
        ),
    }
    return {
        "schema_version": "pldr-program-energy-q0-v2",
        "stage": "Q0",
        "time_semantics": "FINITE_IMPLEMENTED_SCHEDULE",
        "primitives": primitives,
        "edge_record": record,
        "independent_replay": replay,
        "rg_qualification": rg_flow,
        "rg_independent_replay": rg_replay,
        "diagnostics": diagnostics,
        "conditions": conditions,
        "decision": "QUALIFIED" if all(conditions.values()) else "REJECTED",
        "record_sha256": digest_object({
            "stage": "Q0",
            "time_semantics": "FINITE_IMPLEMENTED_SCHEDULE",
            "edge": record["record_sha256"],
            "rg_flow": rg_flow["record_sha256"],
            "conditions": conditions,
        }),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    arguments = parser.parse_args()
    result = qualify(arguments.device)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        Path(arguments.output).write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    if result["decision"] != "QUALIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
