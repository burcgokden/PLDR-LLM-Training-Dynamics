#!/usr/bin/env python3
"""Live all-row qualification for the normal-stability instrumentation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call, jacrev, jvp


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXPERIMENTS))

from branch_resolved_state import (  # noqa: E402
    UPDATE_OPERATION_ORDER,
    bind_owned_successor,
    build_complete_state,
    digest_object,
    tensor_record,
    verify_replayed_target,
)
from confirmation_artifacts import seal_record  # noqa: E402
from constructive_block_comparison import (  # noqa: E402
    construct_comparison,
    construct_positive_witness,
)
from pldr_model_v510 import PLDR_Model  # noqa: E402
from validated_directional_taylor_mixed import (  # noqa: E402
    certify_directional_row_jacobian_remainder,
)
from validated_interval import (  # noqa: E402
    as_fraction,
    nonnegative_sqrt_upper,
)
from validated_row_map import (  # noqa: E402
    model_row_map_specification,
)


class ImplementedRowMap(nn.Module):
    """The ResLayerA composition used by one PLDR decoder layer."""

    def __init__(self, units):
        super().__init__()
        self.units = units

    def forward(self, row):
        value = row
        for unit in self.units:
            value = unit.ResUnit(value)
        return value


def _mask(sequence_length, device):
    return torch.triu(
        torch.ones(
            sequence_length, sequence_length, dtype=torch.float64,
            device=device,
        ),
        diagonal=1,
    )[None, None]


def _capture(model, tokens):
    captured_inputs = []
    captured_outputs = []
    layer = model.decoder.dec_layers[0].mha1.layernorm1

    def hook(_module, inputs, output):
        captured_inputs.append(inputs[0].detach())
        captured_outputs.append(output.detach())

    handle = layer.register_forward_hook(hook)
    inputs = tokens[:, :-1]
    logits, _, _, _ = model([inputs, _mask(inputs.shape[1], inputs.device)])
    handle.remove()
    if len(captured_outputs) != 1 or len(captured_inputs) != 1:
        raise RuntimeError("failed to capture the complete physical row set")
    before = captured_inputs[0].reshape(-1, captured_inputs[0].shape[-1])
    rows = captured_outputs[0].reshape(-1, captured_outputs[0].shape[-1])
    centered = before - before.mean(dim=-1, keepdim=True)
    centered_floor = torch.linalg.vector_norm(centered, dim=-1).min().item()

    probabilities = torch.softmax(logits.detach(), dim=-1).reshape(
        -1, logits.shape[-1])
    quotient_lowers = []
    quotient_uppers = []
    projection_errors = []
    for probability in probabilities:
        hessian = torch.diag(probability) - torch.outer(
            probability, probability)
        eigenvalues = torch.linalg.eigvalsh(hessian)
        quotient_lowers.append(float(eigenvalues[1]))
        quotient_uppers.append(float(eigenvalues[-1]))
        projection_errors.append(float(torch.abs(hessian.sum(dim=0)).max()))
    return {
        "rows": rows.detach().clone(),
        "layernorm_centered_floor": centered_floor,
        "probability_floor": float(probabilities.min()),
        "softmax_quotient_lower": min(quotient_lowers),
        "softmax_quotient_upper": max(quotient_uppers),
        "softmax_projection_error": max(projection_errors),
    }


def _tensor_records(mapping):
    return [
        tensor_record(name, value)
        for name, value in sorted(mapping.items())
    ]


def _optimizer_tensor_records(model, optimizer):
    rows = {}
    names = {parameter: name for name, parameter in model.named_parameters()}
    for parameter, state in optimizer.state.items():
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if key in state:
                rows[f"{names[parameter]}.{key}"] = state[key]
    return _tensor_records(rows)


def _rng_state():
    return {
        "python_repr": repr(random.getstate()),
        "numpy_repr": repr(np.random.get_state()),
        "torch_cpu_hex": torch.get_rng_state().cpu().numpy().tobytes().hex(),
        "torch_cuda_hex": [
            state.cpu().numpy().tobytes().hex()
            for state in (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else []
            )
        ],
    }


def _code_manifest():
    files = (
        Path(__file__),
        HERE / "branch_resolved_state.py",
        HERE / "confirmation_artifacts.py",
        HERE / "constructive_block_comparison.py",
        HERE / "validated_directional_taylor_mixed.py",
        HERE / "validated_interval.py",
        HERE / "validated_row_map.py",
        EXPERIMENTS / "pldr_model_v510.py",
    )
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def _complete_state(
        *, model_tensors, optimizer_tensors, step, rows, clip_mask,
        rng_state, tokens, device):
    row_records = [
        tensor_record(f"layer0.row{index}", row)
        for index, row in enumerate(rows)
    ]
    row_names = [record["name"] for record in row_records]
    return build_complete_state(
        model_tensors=model_tensors,
        optimizer_tensors=optimizer_tensors,
        optimizer_step=step,
        optimizer_groups={
            "learning_rate": "1/100000",
            "beta1": "9/10",
            "beta2": "19/20",
            "epsilon": "1/100000",
            "weight_decay": "1/100",
            "clip_value": "1",
        },
        scheduler_state={"phase": "constant", "applied_index": step},
        amp_state={"enabled": False},
        gradient_accumulation={"count": 0, "buffers": []},
        rng_states=rng_state,
        data_state={
            "dataset_sha256": hashlib.sha256(
                tokens.detach().cpu().numpy().tobytes()).hexdigest(),
            "batch_indices": list(range(tokens.shape[0])),
            "token_ids": tokens.detach().cpu().tolist(),
            "cursor": step,
        },
        intervention_state={"arm": "nominal", "applied": False},
        branch_signature={
            "clip_mask": clip_mask,
            "scheduler_phase": "constant",
            "mask_route": "causal",
        },
        physical_rows=row_records,
        row_registry={
            "layers": [0],
            "rows": row_names,
            "row_count": len(row_names),
            "coordinate_order": "output_major_input_minor",
        },
        model_config={
            "width": 8, "depth": 1, "heads": 1,
            "dff": 12, "row_hidden": 6,
            "device": str(device), "dtype": "float64",
        },
        operation_order=list(UPDATE_OPERATION_ORDER),
        code_manifest=_code_manifest(),
    )


def _execute_live(device, seed=30001):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    model = PLDR_Model(
        num_layers=1, d_model=8, num_heads=1, dff=12,
        input_vocab_size=32, A_dff=6, num_reslayerA=1, num_denseA=1,
        max_seq_len=8, device=device,
    ).to(device=device, dtype=torch.float64)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-5, betas=(0.9, 0.95),
        eps=1e-5, weight_decay=0.01,
    )
    tokens = torch.tensor(
        [
            [1, 4, 7, 3, 2, 11, 6, 10, 5],
            [2, 8, 5, 9, 1, 12, 4, 14, 7],
            [3, 6, 10, 15, 5, 13, 2, 9, 4],
            [4, 12, 1, 7, 16, 3, 11, 5, 8],
            [5, 9, 14, 2, 6, 17, 4, 12, 3],
            [6, 13, 3, 11, 8, 1, 15, 5, 9],
            [7, 2, 16, 4, 10, 6, 18, 3, 12],
            [8, 15, 4, 13, 2, 9, 5, 17, 6],
        ],
        dtype=torch.long, device=device,
    )
    row_map = ImplementedRowMap(
        model.decoder.dec_layers[0].mha1.reslayerAs)
    source_capture = _capture(model, tokens)
    source_model = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    source_optimizer = _optimizer_tensor_records(model, optimizer)
    source_rng = _rng_state()
    source_parameters = {
        name: parameter.detach().clone()
        for name, parameter in row_map.named_parameters()
    }
    source_specification = model_row_map_specification(model, 0)
    parameter_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]
    logits, _, _, _ = model([inputs, _mask(inputs.shape[1], device)])
    loss = F.cross_entropy(logits.reshape(-1, 32), targets.reshape(-1))
    loss.backward()
    raw_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    clip_mask = []
    for name, gradient in raw_gradients.items():
        flat = gradient.reshape(-1)
        clip_mask.extend(
            f"{name}:{index}:"
            + (
                "LOW_CLIPPED" if value < -1
                else "HIGH_CLIPPED" if value > 1
                else "UNCLIPPED"
            )
            for index, value in enumerate(flat.tolist())
        )
    torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
    clipped_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    target_capture = _capture(model, tokens)
    target_model = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    target_parameters = {
        name: parameter.detach().clone()
        for name, parameter in row_map.named_parameters()
    }
    target_specification = model_row_map_specification(model, 0)
    target_optimizer = _optimizer_tensor_records(model, optimizer)
    target_rng = _rng_state()

    replay_errors = []
    negative_work = 0.0
    for name, parameter in model.named_parameters():
        if name not in clipped_gradients:
            continue
        theta = parameter_before[name]
        gradient = clipped_gradients[name]
        first = (1.0 - 0.9) * gradient
        second = (1.0 - 0.95) * gradient.square()
        first_hat = first / (1.0 - 0.9)
        second_hat = second / (1.0 - 0.95)
        expected = (
            theta * (1.0 - 1e-5 * 0.01)
            - 1e-5 * first_hat / (second_hat.sqrt() + 1e-5)
        )
        actual = parameter.detach()
        scale = max(
            float(torch.linalg.vector_norm(actual)),
            float(torch.linalg.vector_norm(expected)),
            float(torch.finfo(actual.dtype).tiny),
        )
        replay_errors.append(
            float(torch.linalg.vector_norm(actual - expected)) / scale)
        negative_work += float(torch.sum(
            raw_gradients[name] * (actual - theta)))

    source_state = _complete_state(
        model_tensors=_tensor_records(source_model),
        optimizer_tensors=source_optimizer,
        step=0,
        rows=source_capture["rows"],
        clip_mask=clip_mask,
        rng_state=source_rng,
        tokens=tokens,
        device=device,
    )
    target_state = _complete_state(
        model_tensors=_tensor_records(target_model),
        optimizer_tensors=target_optimizer,
        step=1,
        rows=target_capture["rows"],
        clip_mask=clip_mask,
        rng_state=target_rng,
        tokens=tokens,
        device=device,
    )
    return {
        "row_map": row_map,
        "source_rows": source_capture["rows"],
        "target_rows": target_capture["rows"],
        "source_parameters": source_parameters,
        "target_parameters": target_parameters,
        "source_specification": source_specification,
        "target_specification": target_specification,
        "source_state": source_state,
        "target_state": target_state,
        "clip_mask": clip_mask,
        "layernorm_centered_floor":
            source_capture["layernorm_centered_floor"],
        "probability_floor": source_capture["probability_floor"],
        "softmax_quotient_lower":
            source_capture["softmax_quotient_lower"],
        "softmax_quotient_upper":
            source_capture["softmax_quotient_upper"],
        "softmax_projection_error":
            source_capture["softmax_projection_error"],
        "adamw_replay_relative_error": max(replay_errors, default=0.0),
        "optimizer_gradient_work": negative_work,
        "batch_size": int(tokens.shape[0]),
        "context_length": int(tokens.shape[1] - 1),
    }


def _jvp_diagnostics(execution):
    row_map = execution["row_map"]
    source_parameters = execution["source_parameters"]
    target_parameters = execution["target_parameters"]
    delta_parameters = {
        name: target_parameters[name] - source_parameters[name]
        for name in source_parameters
    }
    source_stacks = []
    target_stacks = []
    composite_rows = []
    dense_errors = []
    decomposition_errors = []
    remainder_rows = []
    remainder_uppers = []
    derivative_certificates = []

    def psi(parameters, row):
        def phi(argument):
            return functional_call(row_map, parameters, (argument,))
        return jacrev(phi)(row).reshape(-1)

    for source_row, target_row in zip(
        execution["source_rows"], execution["target_rows"],
    ):
        delta_row = target_row - source_row
        source_stack = psi(source_parameters, source_row).detach()
        target_stack = psi(target_parameters, target_row).detach()
        _, composite = jvp(
            psi,
            (source_parameters, source_row),
            (delta_parameters, delta_row),
        )
        _, parameter_part = jvp(
            lambda parameters: psi(parameters, source_row),
            (source_parameters,), (delta_parameters,))
        _, row_part = jvp(
            lambda row: psi(source_parameters, row),
            (source_row,), (delta_row,))
        decomposition_errors.append(float(torch.linalg.vector_norm(
            composite - parameter_part - row_part)))

        dense_parameter_tree = jacrev(
            lambda parameters: psi(parameters, source_row))(source_parameters)
        dense_parameter = torch.cat([
            dense_parameter_tree[name].reshape(source_stack.numel(), -1)
            for name in source_parameters
        ], dim=1)
        flat_delta = torch.cat([
            delta_parameters[name].reshape(-1) for name in source_parameters
        ])
        dense_row = jacrev(
            lambda row: psi(source_parameters, row))(source_row)
        dense_action = dense_parameter @ flat_delta + dense_row @ delta_row
        dense_errors.append(float(torch.linalg.vector_norm(
            dense_action - composite)))
        remainder = target_stack - source_stack - composite

        derivative = certify_directional_row_jacobian_remainder(
            execution["source_specification"],
            execution["target_specification"],
            source_row.detach().cpu().tolist(),
            target_row.detach().cpu().tolist(),
        )
        remainder_upper = as_fraction(derivative["remainder_upper"])
        source_stacks.append(source_stack)
        target_stacks.append(target_stack)
        composite_rows.append(composite)
        remainder_rows.append(remainder)
        remainder_uppers.append(remainder_upper)
        derivative_certificates.append(derivative)

    source_stacks = torch.stack(source_stacks)
    target_stacks = torch.stack(target_stacks)
    composite_rows = torch.stack(composite_rows)
    remainder_rows = torch.stack(remainder_rows)
    source_normal = source_stacks - source_stacks.mean(dim=0, keepdim=True)
    target_normal = target_stacks - target_stacks.mean(dim=0, keepdim=True)
    energy = float(source_normal.square().sum())
    covariance = source_normal.T @ source_normal / source_normal.shape[0]
    energy_from_covariance = float(
        source_normal.shape[0] * torch.trace(covariance))
    total_upper = nonnegative_sqrt_upper(sum(
        (value ** 2 for value in remainder_uppers), as_fraction(0)))
    observed_remainder = float(torch.linalg.vector_norm(remainder_rows))
    taylor_rows = [
        {
            "row_index": index,
            "observed_remainder": float(torch.linalg.vector_norm(
                remainder_rows[index])),
            "certified_remainder_upper": float(remainder_uppers[index]),
        }
        for index in range(remainder_rows.shape[0])
    ]

    scale = max(
        float(torch.linalg.vector_norm(composite_rows)),
        float(torch.finfo(source_stacks.dtype).tiny),
    )
    action_error = max(dense_errors, default=0.0)
    decomposition_error = max(decomposition_errors, default=0.0)
    live_gain = max(action_error, decomposition_error) / scale
    live_forcing = observed_remainder / max(
        float(torch.linalg.vector_norm(source_stacks)),
        float(torch.finfo(source_stacks.dtype).tiny),
    )
    transformed = {}
    structural = []
    if live_gain == 0.0:
        structural.append(("instrument", "instrument"))
    else:
        transformed[("instrument", "instrument")] = live_gain
    comparison = construct_comparison(
        block_names=["instrument"],
        transformed_operator_norms=transformed,
        forcing_norms={"instrument": live_forcing},
        structural_zeros=structural,
    )
    witness = construct_positive_witness(comparison["comparison_matrix"])
    return {
        "row_count": source_stacks.shape[0],
        "batch_size": execution["batch_size"],
        "context_length": execution["context_length"],
        "source_stack_dimension": source_stacks.shape[1],
        "row_input_dimension": execution["source_rows"].shape[1],
        "operator_to_frobenius_factor": float(nonnegative_sqrt_upper(
            as_fraction(execution["source_rows"].shape[1]))),
        "dense_action_error": action_error,
        "moving_row_decomposition_error": decomposition_error,
        "physical_row_displacement": float(torch.linalg.vector_norm(
            execution["target_rows"] - execution["source_rows"])),
        "observed_remainder": observed_remainder,
        "certified_remainder_upper": float(total_upper),
        "remainder_enclosed":
            as_fraction(observed_remainder) <= total_upper,
        "taylor_rows": taylor_rows,
        "energy": energy,
        "energy_from_row_covariance": energy_from_covariance,
        "energy_identity_error": abs(energy - energy_from_covariance),
        "row_normal_work": float(torch.sum(
            source_normal * (target_normal - source_normal))),
        "live_comparison": comparison,
        "live_positive_witness": witness,
        "derivative_certificate_count": len(derivative_certificates),
        "taylor_certificate_inputs": {
            "schema_version": "pldr-q-taylor-certificate-inputs-v3",
            "source_specification": execution["source_specification"],
            "target_specification": execution["target_specification"],
            "source_rows": execution["source_rows"].detach().cpu().tolist(),
            "target_rows": execution["target_rows"].detach().cpu().tolist(),
        },
        "derivative_certificates": derivative_certificates,
        "direction_record": {
            "schema_version": "pldr-q-all-row-joint-direction-v2",
            "row_count": source_stacks.shape[0],
            "parameter_delta_sha256": digest_object(_tensor_records(
                delta_parameters)),
        },
    }


def qualify(device="cpu"):
    started = time.perf_counter()
    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("the selected CUDA device is unavailable")
    live = _execute_live(selected)
    replayed = _execute_live(selected)
    jvp_result = _jvp_diagnostics(live)
    direction = dict(jvp_result["direction_record"])
    direction["direction_sha256"] = digest_object(direction)
    jvp_result["direction_record"] = direction
    edge = bind_owned_successor(
        edge_id="q-width8-depth1-all-rows",
        source_state=live["source_state"],
        target_state=live["target_state"],
        update_direction_sha256=direction["direction_sha256"],
        constructor_sha256=hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        operation_trace=UPDATE_OPERATION_ORDER,
    )
    replay = verify_replayed_target(edge, replayed["target_state"])
    mutation_rejected = False
    corrupted = copy.deepcopy(replayed["target_state"])
    corrupted["state_sha256"] = "0" * 64
    try:
        verify_replayed_target(edge, corrupted)
    except ValueError:
        mutation_rejected = True

    tolerance = 256 * torch.finfo(torch.float64).eps
    witness = jvp_result["live_positive_witness"]
    taylor_ratio = max(
        row["certified_remainder_upper"] / max(
            row["observed_remainder"], torch.finfo(torch.float64).tiny)
        for row in jvp_result["taylor_rows"]
    )
    conditions = {
        "all_rows_registered":
            jvp_result["row_count"]
            == len(live["source_state"]["row_registry"]["rows"]),
        "frozen_job_geometry":
            jvp_result["batch_size"] == 8
            and jvp_result["context_length"] == 8,
        "taylor_enclosure_ratio": taylor_ratio <= 1e3,
        "source_replays":
            live["source_state"]["state_sha256"]
            == replayed["source_state"]["state_sha256"],
        "target_replays": replay["decision"] == "REPLAYED",
        "physical_row_moves": jvp_result["physical_row_displacement"] > 0,
        "moving_row_decomposition":
            jvp_result["moving_row_decomposition_error"] < tolerance,
        "dense_matrix_free_action":
            jvp_result["dense_action_error"] < tolerance,
        "certified_taylor_segment": jvp_result["remainder_enclosed"],
        "layernorm_floor_measured":
            live["layernorm_centered_floor"] > 0,
        "softmax_quotient_resolved":
            live["softmax_quotient_lower"] > 0
            and live["softmax_quotient_upper"] <= 0.5 + tolerance
            and live["softmax_projection_error"] < tolerance,
        "adamw_successor_exact":
            live["adamw_replay_relative_error"] < tolerance,
        "energy_identity":
            jvp_result["energy_identity_error"]
            <= tolerance * max(jvp_result["energy"], 1.0),
        "negative_work_measured":
            live["optimizer_gradient_work"] < 0,
        "positive_witness_constructed":
            witness["kappa"]["numerator"]
            < witness["kappa"]["denominator"],
        "provenance_mutation_rejected": mutation_rejected,
    }
    result = {
        "producer_code_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        "schema_version": "pldr-q-live-record-v2",
        "stage": "Q",
        "device": str(selected),
        "owned_edge": edge,
        "replay": replay,
        "jvp_and_taylor": jvp_result,
        "measured_geometry": {
            "layernorm_centered_floor":
                live["layernorm_centered_floor"],
            "probability_floor": live["probability_floor"],
            "softmax_quotient_lower":
                live["softmax_quotient_lower"],
            "softmax_quotient_upper":
                live["softmax_quotient_upper"],
            "softmax_projection_error":
                live["softmax_projection_error"],
            "optimizer_gradient_work":
                live["optimizer_gradient_work"],
            "adamw_replay_relative_error":
                live["adamw_replay_relative_error"],
        },
        "resource_metadata": {
            "wall_clock_seconds": time.perf_counter() - started,
            "model_state_bytes": sum(
                len(bytes.fromhex(row["data_hex"]))
                for row in live["target_state"]["model_tensors"]),
            "optimizer_state_bytes": sum(
                len(bytes.fromhex(row["data_hex"]))
                for row in live["target_state"]["optimizer_tensors"]),
        },
        "conditions": conditions,
        "decision": "QUALIFIED" if all(conditions.values()) else "REJECTED",
    }
    return seal_record(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    arguments = parser.parse_args()
    result = qualify(arguments.device)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        Path(arguments.output).parent.mkdir(parents=True, exist_ok=True)
        Path(arguments.output).write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    if result["decision"] != "QUALIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
