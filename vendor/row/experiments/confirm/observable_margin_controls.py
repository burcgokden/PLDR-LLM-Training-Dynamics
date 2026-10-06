#!/usr/bin/env python3
"""Compact live controls for observable-margin confirmation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
import torch
from torch.func import jacrev

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
ROOT = EXPERIMENTS.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(SCRIPTS))

from gate_shape import adamw_gate_block  # noqa: E402
from confirmation_artifacts import (  # noqa: E402
    capture_rng_states,
    nested_state_digest,
    restore_rng_states,
    tensor_mapping_digest,
)
from gate_shape_evidence import (  # noqa: E402
    digest_object,
    sha256_path,
    write_json_atomic,
)
from observable_margin import native_roundoff_charge  # noqa: E402
from observable_margin_live import (  # noqa: E402
    _load_deciding_checkpoint,
    _load_lock,
    _load_registry,
    _locked_union_graph,
    _probe_node,
    _source_actions,
    _bridge_directions,
    _token_matrix,
)
from observable_margin_specs import (  # noqa: E402
    ARCHITECTURE,
    ARITHMETIC,
    CAMPAIGN_ID,
    INTERVENTION_COMPARISON_NAMES,
    INTERVENTION_OPERATOR_BY_NAME,
    INTERVENTION_OPERATOR_CONTRACTS,
    INTERVENTION_OPERATOR_FIELDS,
    REGISTRY,
    RESOURCE_CAPS,
    STAGE_BY_ID,
    STATISTICS,
    ANCHORS,
    RUNS,
)
from row_map_dynamics import (  # noqa: E402
    dimensionless_state_scale,
    relative_matrix_residual,
    scaled_chronological_diagnostics,
)
from train_run import schedule_multiplier  # noqa: E402
from row_map_live import (  # noqa: E402
    _gate_loss_derivatives,
    _gate_microbatches,
    _gate_parameter_name,
    _model_from_raw,
    _weighted_gate_gradient_function,
    _loss,
    _source_normalized_shape,
)

ROB_SCHEMA = "pldr-observable-robustness-v1"
RESTART_SCHEMA = "pldr-observable-restart-replay-v1"
INTERVENTION_SCHEMA = "pldr-observable-intervention-decision-v1"
RELEASE_SCHEMA = "pldr-observable-release-verification-v1"


def _resource_record(device, started):
    elapsed = float(time.monotonic() - started)
    host = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        host *= 1024
    resolved = str(device)
    result = {
        "resolved_device": resolved,
        "resource_resolved_device": resolved,
        "gpu_device_seconds": elapsed if device.type == "cuda" else 0.0,
        "wall_clock_seconds": elapsed,
        "peak_host_rss_bytes": host,
        "peak_gpu_allocated_bytes": 0,
        "peak_gpu_reserved_bytes": 0,
    }
    if device.type == "cuda":
        result["peak_gpu_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(device))
        result["peak_gpu_reserved_bytes"] = int(
            torch.cuda.max_memory_reserved(device))
    return result

def _sealed_lock(path):
    lock = _load_lock(path)
    return lock


def _next_rows(checkpoint, tokens, count, device, *, offset=0):
    matrix = _token_matrix(tokens, int(checkpoint["config"]["ctx"]))
    begin = int(checkpoint["data_offset_end"]) + int(offset)
    stop = begin + int(count)
    if begin < 0 or stop > len(matrix):
        raise ValueError("registered continuation interval leaves token archive")
    return torch.from_numpy(
        matrix[begin:stop].astype(np.int64, copy=True)).to(device)


def produce_robustness(arguments):
    started = time.monotonic()
    lock = _sealed_lock(arguments.lock)
    checkpoint_path = str(arguments.checkpoint).replace(
        "{anchor}", str(lock["selected"]["anchor"]))
    checkpoint = _load_deciding_checkpoint(checkpoint_path, device="cpu")
    registry = _load_registry(arguments.registry)
    if checkpoint["measurement_registry"] != registry:
        raise ValueError("ROB checkpoint does not own the probe registry")
    if sha256_path(arguments.tokens) != registry["dataset_sha256"]:
        raise ValueError("ROB token archive changed")
    if int(checkpoint["config"]["seed"]) != 4444:
        raise ValueError("ROB must use the construction source")
    layer = int(lock["selected"]["layer"])
    device = torch.device(arguments.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = _model_from_raw(checkpoint, device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(checkpoint["config"]["lr"]),
        betas=tuple(checkpoint["optimizer_config"]["groups"][0]["betas"]),
        eps=float(checkpoint["optimizer_config"]["groups"][0]["epsilon"]),
        weight_decay=float(checkpoint["config"]["wd"]))
    optimizer.load_state_dict(checkpoint["opt"])
    gate_name = _gate_parameter_name(layer)
    gate = dict(model.named_parameters())[gate_name]
    state = optimizer.state[gate]
    group = next(value for value in optimizer.param_groups
                 if any(parameter is gate for parameter in value["params"]))
    batch = int(checkpoint["config"]["batch"])
    rows = _next_rows(checkpoint, arguments.tokens, batch, device)
    derivatives = _gate_loss_derivatives(
        model, _gate_microbatches(rows), layer)
    raw = derivatives["gradient_at_gate"].detach()
    hessian = derivatives["true_hessian"].detach()
    gamma = gate.detach()
    first = state["exp_avg"].detach()
    second = state["exp_avg_sq"].detach()
    beta1, beta2 = map(float, group["betas"])
    epsilon = float(group["eps"])
    learning_rate = float(group["lr"])
    weight_decay = float(group["weight_decay"])
    clip_value = float(checkpoint["config"]["clip"])
    step_before = int(state["step"].item())
    step_after = step_before + 1
    analytic = adamw_gate_block(
        hessian.cpu().numpy(), gamma.cpu().numpy(), first.cpu().numpy(),
        second.cpu().numpy(), raw.cpu().numpy(),
        learning_rate=learning_rate, beta1=beta1, beta2=beta2,
        epsilon=epsilon, optimizer_step=step_after,
        weight_decay=weight_decay, clip_value=clip_value,
        boundary_tolerance=ARITHMETIC["clipping_boundary_tolerance"])
    gradient_function = _weighted_gate_gradient_function(
        model, _gate_microbatches(rows), layer)
    dimension = gamma.numel()
    source_state = torch.cat((gamma, first, second))

    def successor(vector):
        local_gamma = vector[:dimension]
        local_first = vector[dimension:2 * dimension]
        local_second = vector[2 * dimension:]
        local_raw = gradient_function(local_gamma)
        clipped = torch.clamp(local_raw, -clip_value, clip_value)
        next_first = beta1 * local_first + (1.0 - beta1) * clipped
        next_second = beta2 * local_second + (1.0 - beta2) * clipped * clipped
        first_hat = next_first / (1.0 - beta1 ** step_after)
        second_hat = next_second / (1.0 - beta2 ** step_after)
        next_gamma = (
            (1.0 - learning_rate * weight_decay) * local_gamma
            - learning_rate * first_hat / (torch.sqrt(second_hat) + epsilon))
        return torch.cat((next_gamma, next_first, next_second))

    autodiff = jacrev(successor, chunk_size=1)(source_state).detach().cpu().numpy()
    operator = np.asarray(analytic["operator"], dtype=np.float64)
    operator_residual = relative_matrix_residual(operator, autodiff)
    source_second_hat = second.detach().double().cpu().numpy() / (
        1.0 - beta2 ** step_before)
    target_second_hat = np.asarray(analytic["second_hat"], dtype=np.float64)
    source_scale = dimensionless_state_scale(source_second_hat, epsilon=epsilon)
    target_scale = dimensionless_state_scale(target_second_hat, epsilon=epsilon)
    diagnostics = scaled_chronological_diagnostics(
        [operator], [source_scale, target_scale])
    boundary_distance = float(np.min(np.abs(
        np.abs(raw.detach().cpu().numpy()) - clip_value)))
    checks = {
        "full_3d_native_block": bool(
            operator.shape == (3 * dimension, 3 * dimension)
            and operator_residual <= ARITHMETIC["relative_identity_tolerance"]),
        "scaled_induced_norm": math.isfinite(diagnostics["scaled_norm"]),
        "spectral_radius": math.isfinite(diagnostics["spectral_radius"]),
        "endpoint_conditioning": math.isfinite(
            diagnostics["endpoint_scaling_condition"]),
        "clipping_and_dissipation": bool(
            boundary_distance > ARITHMETIC["clipping_boundary_tolerance"]
            and weight_decay > 0.0),
        "diagnostic_not_scalar_margin_gate": True,
        "terminal_record_required_but_pass_not_required_by_M": True,
    }
    if set(checks) != set(STAGE_BY_ID["ROB"]["required_checks"]):
        raise ArithmeticError("ROB stage-check names drifted from the protocol")
    value = {
        "schema_version": ROB_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "stage": "ROB",
        "job_id": arguments.job_id,
        "scientific_gate": False,
        "checkpoint_complete_state_sha256": checkpoint["state_manifest"][
            "complete_state_sha256"],
        "construction_lock_sha256": lock["lock_sha256"],
        "probe_registry_sha256": registry["registry_sha256"],
        "seed": 4444,
        "anchor": int(checkpoint["step"]),
        "layer": layer,
        "stage_checks": checks,
        "metrics": {
            "state_dimension": 3 * dimension,
            "analytic_autodiff_relative_residual": operator_residual,
            "scaled_induced_norm": diagnostics["scaled_norm"],
            "spectral_radius": diagnostics["spectral_radius"],
            "endpoint_scaling_condition": diagnostics[
                "endpoint_scaling_condition"],
            "clipping_boundary_distance": boundary_distance,
            "operator_sha256": digest_object(operator.tolist()),
            "autodiff_operator_sha256": digest_object(autodiff.tolist()),
        },
        "resources": _resource_record(device, started),
        "status": "PASS" if all(checks.values()) else "DIAGNOSTIC_FAIL",
    }
    value["report_sha256"] = digest_object(value)
    write_json_atomic(arguments.output, value)



def _parameter_digest(model, *, exclude=()):
    excluded = set(exclude)
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if name in excluded:
            continue
        array = np.ascontiguousarray(parameter.detach().cpu().numpy())
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()




def _optimizer_named_state_digest(model, optimizer, *, exclude=()):
    excluded = set(exclude)
    state = {
        name: optimizer.state.get(parameter, {})
        for name, parameter in model.named_parameters()
        if name not in excluded
    }
    return nested_state_digest(state)


def _tensor_signature(value):
    tensor = value.detach()
    return {
        "sha256": tensor_mapping_digest({"value": tensor}),
        "norm": float(torch.linalg.vector_norm(tensor.double()).cpu()),
    }


def _moment_transition_diagnostics(record):
    required = {
        "step_before", "step_after", "beta1", "beta2",
        "first_moment_before", "second_moment_before",
        "clipped_gradient", "clipped_gradient_sha256",
        "first_moment_after", "second_moment_after",
    }
    if set(record) != required:
        raise ValueError("native selected-gate moment record changed")
    first_before = np.asarray(
        record["first_moment_before"], dtype=np.float64)
    second_before = np.asarray(
        record["second_moment_before"], dtype=np.float64)
    gradient = np.asarray(record["clipped_gradient"], dtype=np.float64)
    first_after = np.asarray(
        record["first_moment_after"], dtype=np.float64)
    second_after = np.asarray(
        record["second_moment_after"], dtype=np.float64)
    if (
        first_before.ndim != 1
        or first_before.shape != second_before.shape
        or first_before.shape != gradient.shape
        or first_before.shape != first_after.shape
        or first_before.shape != second_after.shape
        or first_before.size == 0
        or not all(np.all(np.isfinite(value)) for value in (
            first_before, second_before, gradient,
            first_after, second_after))
    ):
        raise ValueError("native selected-gate moment arrays are malformed")
    beta1 = float(record["beta1"])
    beta2 = float(record["beta2"])
    if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
        raise ValueError("native Adam beta lies outside [0,1)")
    expected_first = beta1 * first_before + (1.0 - beta1) * gradient
    expected_second = (
        beta2 * second_before + (1.0 - beta2) * gradient * gradient)
    first_scale = max(
        float(np.linalg.norm(expected_first)),
        float(np.linalg.norm(first_after)), 1e-300)
    second_scale = max(
        float(np.linalg.norm(expected_second)),
        float(np.linalg.norm(second_after)), 1e-300)
    return {
        "clock_increment": (
            int(record["step_after"]) - int(record["step_before"])),
        "first_moment_relative_residual": float(
            np.linalg.norm(first_after - expected_first) / first_scale),
        "second_moment_relative_residual": float(
            np.linalg.norm(second_after - expected_second) / second_scale),
    }


def _native_moment_transition_record(
    state_before, state_after, clipped_gradient, group,
):
    signature = _tensor_signature(clipped_gradient)
    beta1, beta2 = map(float, group["betas"])
    step_before = state_before["step"]
    step_after = state_after["step"]
    record = {
        "step_before": int(
            step_before.item() if torch.is_tensor(step_before) else step_before),
        "step_after": int(
            step_after.item() if torch.is_tensor(step_after) else step_after),
        "beta1": beta1,
        "beta2": beta2,
        "first_moment_before": (
            state_before["exp_avg"].detach().double().cpu().reshape(-1).tolist()),
        "second_moment_before": (
            state_before["exp_avg_sq"].detach().double().cpu().reshape(-1).tolist()),
        "clipped_gradient": (
            clipped_gradient.detach().double().cpu().reshape(-1).tolist()),
        "clipped_gradient_sha256": signature["sha256"],
        "first_moment_after": (
            state_after["exp_avg"].detach().double().cpu().reshape(-1).tolist()),
        "second_moment_after": (
            state_after["exp_avg_sq"].detach().double().cpu().reshape(-1).tolist()),
    }
    return {**record, **_moment_transition_diagnostics(record)}


def _validate_native_moment_transition_path(arm, continuation_steps):
    rows = arm.get("native_selected_gate_moment_transition_by_step", [])
    clipped = arm.get("clipped_gradient_signature_by_step", [])
    if len(rows) != continuation_steps or len(clipped) != continuation_steps:
        raise ValueError("native selected-gate moment path is incomplete")
    tolerance = ARITHMETIC[
        "native_moment_transition_relative_tolerance"]
    valid = []
    previous = None
    for row, signature in zip(rows, clipped, strict=True):
        diagnostic_fields = {
            "clock_increment", "first_moment_relative_residual",
            "second_moment_relative_residual",
        }
        raw = {
            key: value for key, value in row.items()
            if key not in diagnostic_fields}
        replay = _moment_transition_diagnostics(raw)
        chained = bool(
            previous is None
            or (
                row["step_before"] == previous["step_after"]
                and row["first_moment_before"]
                == previous["first_moment_after"]
                and row["second_moment_before"]
                == previous["second_moment_after"]
            ))
        valid.append(bool(
            row["clipped_gradient_sha256"] == signature["sha256"]
            and all(row[field] == replay[field] for field in diagnostic_fields)
            and replay["clock_increment"] == 1
            and replay["first_moment_relative_residual"] <= tolerance
            and replay["second_moment_relative_residual"] <= tolerance
            and chained))
        previous = row
    return valid


def _set_unit_rng(seed, device):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)



def produce_restart_replay(arguments):
    """Replay each anchor successor and compare to its uninterrupted checkpoint."""

    started = time.monotonic()
    registry = _load_registry(arguments.registry)
    if sha256_path(arguments.tokens) != registry["dataset_sha256"]:
        raise ValueError("T restart tokens changed")
    run = next((row for row in RUNS if row["seed"] == int(arguments.seed)), None)
    if run is None or run["name"] != arguments.run_name:
        raise ValueError("T restart job has an unregistered run identity")
    device = torch.device(arguments.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    pairs = []
    for anchor in ANCHORS["complete_optimizer_steps"]:
        source_path = Path(arguments.run_root) / run["name"] / f"ckpt_{anchor}.pt"
        target_path = Path(arguments.run_root) / run["name"] / f"ckpt_{anchor + 1}.pt"
        source = _load_deciding_checkpoint(source_path, device="cpu")
        target = _load_deciding_checkpoint(
            target_path, device="cpu", allow_restart_successor=True)
        if (
            source["measurement_registry"] != registry
            or target["measurement_registry"] != registry
            or int(source["config"]["seed"]) != run["seed"]
            or int(target["config"]["seed"]) != run["seed"]
            or int(source["step"]) != anchor
            or int(target["step"]) != anchor + 1
            or int(target["data_offset_end"])
            != int(source["data_offset_end"]) + int(source["config"]["batch"])
        ):
            raise ValueError("T restart checkpoint identity/cursor binding failed")
        model, optimizer, scheduler = _optimizer_model_scheduler(
            source, device)
        restore_rng_states(source["rng_states"])
        restored_rng_exact = (
            nested_state_digest(capture_rng_states())
            == nested_state_digest(source["rng_states"]))
        rows = _next_rows(
            source, arguments.tokens, int(source["config"]["batch"]), device)
        minibatch_sha256 = digest_object(rows.detach().cpu().numpy().tolist())
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, rows)
        loss.backward()
        clip_value = float(source["config"]["clip"])
        replay_clip_mask = {}
        replay_decay_mask = {}
        parameter_groups = {
            id(parameter): group for group in optimizer.param_groups
            for parameter in group["params"]
        }
        for parameter_name, parameter in model.named_parameters():
            gradient = parameter.grad
            clipped_mask = (
                torch.zeros_like(parameter, dtype=torch.bool, device="cpu")
                if gradient is None or clip_value <= 0.0
                else (gradient.detach().abs() > clip_value).to(
                    device="cpu", dtype=torch.bool))
            decayed = bool(
                gradient is not None
                and float(parameter_groups[id(parameter)].get(
                    "weight_decay", 0.0)) != 0.0)
            replay_clip_mask[parameter_name] = clipped_mask
            replay_decay_mask[parameter_name] = torch.full_like(
                clipped_mask, decayed, dtype=torch.bool, device="cpu")
        torch.nn.utils.clip_grad_value_(
            model.parameters(), clip_value=clip_value)
        optimizer.step()
        scheduler.step()
        replay_model_digest = tensor_mapping_digest(model.state_dict())
        replay_optimizer_digest = nested_state_digest(optimizer.state_dict())
        model_match = replay_model_digest == tensor_mapping_digest(target["model"])
        optimizer_match = (
            replay_optimizer_digest == nested_state_digest(target["opt"]))
        scheduler_match = (
            nested_state_digest(scheduler.state_dict())
            == nested_state_digest(target["scheduler"]))
        successor_rng_match = (
            nested_state_digest(capture_rng_states())
            == nested_state_digest(target["rng_states"]))
        branch_masks_match = bool(
            nested_state_digest(replay_clip_mask)
            == nested_state_digest(target["branch_state"]["clip_mask"])
            and nested_state_digest(replay_decay_mask)
            == nested_state_digest(target["branch_state"]["decay_mask"])
            and str(device) == target["branch_state"]["device"]
            and str(next(model.parameters()).dtype)
            == target["branch_state"]["dtype"])
        source_rng_manifest = source["state_manifest"]["rng"]
        restored_rng_manifest = {
            "source_state_sha256": nested_state_digest(source["rng_states"]),
            "restored_exact_before_successor": restored_rng_exact,
        }
        pairs.append({
            "anchor": anchor,
            "source_complete_state_sha256": source["state_manifest"][
                "complete_state_sha256"],
            "uninterrupted_successor_complete_state_sha256": target[
                "state_manifest"]["complete_state_sha256"],
            "minibatch_sha256": minibatch_sha256,
            "cursor_start": int(source["data_offset_end"]),
            "cursor_end": int(target["data_offset_end"]),
            "source_rng_manifest": source_rng_manifest,
            "restored_rng_binding": restored_rng_manifest,
            "model_successor_exact": model_match,
            "optimizer_successor_exact": optimizer_match,
            "scheduler_successor_exact": scheduler_match,
            "rng_successor_exact": successor_rng_match,
            "branch_masks_exact": branch_masks_match,
            "loss": float(loss.detach().cpu()),
        })
    checks = {
        "native_float32_adamw": all(
            row["model_successor_exact"] and row["optimizer_successor_exact"]
            and row["scheduler_successor_exact"] and row["rng_successor_exact"]
            and row["branch_masks_exact"] for row in pairs),
        "actual_minibatches": all(row["minibatch_sha256"] for row in pairs),
        "five_complete_optimizer_anchors": len(pairs) == 5,
        "exact_restart_replay": all(
            row["model_successor_exact"] and row["optimizer_successor_exact"]
            and row["scheduler_successor_exact"] and row["rng_successor_exact"]
            and row["branch_masks_exact"] for row in pairs),
        "data_cursor_and_rng_state": all(
            row["cursor_end"] > row["cursor_start"]
            and row["source_rng_manifest"]
            and row["restored_rng_binding"][
                "restored_exact_before_successor"]
            and row["rng_successor_exact"]
            for row in pairs),
        "checkpoint_registry_binding": True,
    }
    if not set(checks).issubset(STAGE_BY_ID["T"]["required_checks"]):
        raise ArithmeticError("T restart stage-check names drifted")
    value = {
        "schema_version": RESTART_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "stage": "T",
        "job_id": arguments.job_id,
        "run_name": run["name"],
        "seed": run["seed"],
        "probe_registry_sha256": registry["registry_sha256"],
        "pairs": pairs,
        "stage_checks": checks,
        "resources": _resource_record(device, started),
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    value["report_sha256"] = digest_object(value)
    write_json_atomic(arguments.output, value)
    if value["status"] != "PASS":
        raise SystemExit("exact native restart replay failed")



def _optimizer_model_scheduler(checkpoint, device):
    """Restore native optimizer and registered scheduler without LR drift."""

    model = _model_from_raw(checkpoint, device)
    model.train()
    config = checkpoint["config"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]),
        betas=tuple(checkpoint["optimizer_config"]["groups"][0]["betas"]),
        eps=float(checkpoint["optimizer_config"]["groups"][0]["epsilon"]),
        weight_decay=float(config["wd"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: schedule_multiplier(
            step, int(checkpoint["schedule_total_steps"]),
            int(config["warmup"]), alpha=float(config["anneal_floor"]),
            const_lr=bool(config["const_lr"])),
        last_epoch=-1)
    optimizer.load_state_dict(copy.deepcopy(checkpoint["opt"]))
    scheduler.load_state_dict(copy.deepcopy(checkpoint["scheduler"]))
    return model, optimizer, scheduler


def _optimizer_model(checkpoint, device):
    model = _model_from_raw(checkpoint, device)
    model.train()
    config = checkpoint["config"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]),
        betas=tuple(checkpoint["optimizer_config"]["groups"][0]["betas"]),
        eps=float(checkpoint["optimizer_config"]["groups"][0]["epsilon"]),
        weight_decay=float(config["wd"]))
    optimizer.load_state_dict(copy.deepcopy(checkpoint["opt"]))
    return model, optimizer


def _operator_contract(row):
    return {
        field: row[field]
        for field in INTERVENTION_OPERATOR_FIELDS
    }


def _counterfactual_shadow_gate(
    gate_before, adaptive_direction, learning_rate, weight_decay, arm,
):
    """Exact float64 gate successor for the registered intervention arm."""

    gate_before64 = np.asarray(gate_before, dtype=np.float64)
    adaptive64 = np.asarray(adaptive_direction, dtype=np.float64)
    if gate_before64.shape != adaptive64.shape:
        raise ValueError("counterfactual gate and adaptive direction disagree")
    if arm == INTERVENTION_COMPARISON_NAMES[0]:
        return gate_before64 - learning_rate * weight_decay * gate_before64
    if arm == "decay_source_removal":
        return gate_before64 - learning_rate * adaptive64
    if arm in {"baseline", INTERVENTION_COMPARISON_NAMES[2]}:
        return gate_before64 - learning_rate * (
            weight_decay * gate_before64 + adaptive64)
    raise ValueError("unknown observable intervention arm")


def _compact_arm_continuation(
    checkpoint, registry, tokens, lock, union_graph, contexts, head_indices,
    *, unit, arm, device, source_model,
):
    model, optimizer, scheduler = _optimizer_model_scheduler(checkpoint, device)
    layer = int(lock["selected"]["layer"])
    gate_name = _gate_parameter_name(layer)
    gate = dict(model.named_parameters())[gate_name]
    initial_state = {
        "model_sha256": tensor_mapping_digest(model.state_dict()),
        "optimizer_sha256": nested_state_digest(optimizer.state_dict()),
        "scheduler_sha256": nested_state_digest(scheduler.state_dict()),
        "rng_sha256": nested_state_digest(capture_rng_states()),
        "cursor": int(checkpoint["data_offset_end"]),
    }
    initial_state_sha256 = digest_object(initial_state)
    fixed_source_before = tensor_mapping_digest(source_model.state_dict())
    energies = []
    fp_charges = []
    sector_norms = []
    batch_digests = []
    target_residual_signatures = {
        name: [] for name in INTERVENTION_COMPARISON_NAMES}
    raw_gradient_signatures = []
    clipped_gradient_signatures = []
    native_moment_transitions = []
    successor_gate_sha256 = []
    successor_model_sha256 = []
    successor_optimizer_sha256 = []
    successor_scheduler_sha256 = []
    actual_shape_signatures = []
    forced_shape_signatures = []
    effective_shape_signatures = []
    effective_actual_difference_norms = []
    fixed_source_model_bindings = []
    clamp_hook_removed_before_probe = []
    direct_parameter_bindings = {
        name: [] for name in INTERVENTION_COMPARISON_NAMES}
    direct_optimizer_bindings = {
        name: [] for name in INTERVENTION_COMPARISON_NAMES}
    selected_gate_optimizer_bindings = {
        name: [] for name in INTERVENTION_COMPARISON_NAMES}
    source_probe = _probe_node(
        model, contexts, layer, head_indices, union_graph)
    energies.append(float(source_probe["energy"][0]))
    batch = int(checkpoint["config"]["batch"])
    matrix = _token_matrix(tokens, registry["context_length"])
    initial_cursor = int(checkpoint["data_offset_end"])
    source_model.train()
    clamp_name = INTERVENTION_COMPARISON_NAMES[2]
    for local_step in range(int(unit["update_count"])):
        begin = initial_cursor + (
            int(unit["future_update_start"]) + local_step) * batch
        stop = begin + batch
        if stop > len(matrix):
            raise ValueError("intervention continuation leaves token archive")
        row_array = matrix[begin:stop].astype(np.int64, copy=True)
        batch_digests.append(digest_object(row_array.tolist()))
        rows = torch.from_numpy(row_array).to(device)
        state = optimizer.state[gate]
        state_before = {
            name: value.detach().clone() if torch.is_tensor(value) else value
            for name, value in state.items()}
        gate_before = gate.detach().clone()
        group = next(value for value in optimizer.param_groups
                     if any(parameter is gate for parameter in value["params"]))

        # The source checkpoint is never advanced. Recompute its normalized
        # activation on the exact arm minibatch without advancing arm RNG.
        rng_before_shape_probe = capture_rng_states()
        source_state_before = tensor_mapping_digest(source_model.state_dict())
        forced_shape = _source_normalized_shape(source_model, rows, layer)
        source_state_after = tensor_mapping_digest(source_model.state_dict())
        fixed_source_model_bindings.append({
            "before": source_state_before, "after": source_state_after})
        restore_rng_states(rng_before_shape_probe)
        observed_actual = []
        observed_effective = []
        shape_module = model.decoder.dec_layers[
            layer].mha1.reslayerAs[-1].layernormA

        def shape_hook(local_module, inputs, output):
            value = inputs[0]
            centered = value - value.mean(dim=-1, keepdim=True)
            actual = centered / torch.sqrt(
                local_module.eps + torch.mean(
                    centered * centered, dim=-1, keepdim=True))
            effective = forced_shape if arm == clamp_name else actual
            if effective.shape != output.shape:
                raise ValueError("shape clamp source and target disagree")
            observed_actual.append(actual.detach())
            observed_effective.append(effective.detach())
            if arm == clamp_name:
                # forced_shape is detached, so gamma and beta remain trainable
                # while gradients through the LayerNorm input are blocked.
                return local_module.bias + local_module.weight * effective
            return None

        shape_parameter_before = _parameter_digest(model)
        shape_optimizer_before = _optimizer_named_state_digest(model, optimizer)
        shape_gate_optimizer_before = nested_state_digest(
            optimizer.state.get(gate, {}))
        hook = shape_module.register_forward_hook(shape_hook)
        optimizer.zero_grad(set_to_none=True)
        try:
            loss = _loss(model, rows)
            loss.backward()
        finally:
            hook.remove()
        if len(observed_actual) != 1 or len(observed_effective) != 1:
            raise RuntimeError(
                "intervention shape observer did not fire exactly once")
        shape_parameter_after = _parameter_digest(model)
        shape_optimizer_after = _optimizer_named_state_digest(model, optimizer)
        shape_gate_optimizer_after = nested_state_digest(
            optimizer.state.get(gate, {}))
        actual_shape = observed_actual[0]
        effective_shape = observed_effective[0]
        actual_shape_signatures.append(_tensor_signature(actual_shape))
        forced_shape_signatures.append(_tensor_signature(forced_shape))
        effective_shape_signatures.append(_tensor_signature(effective_shape))
        effective_actual_difference_norms.append(float(
            torch.linalg.vector_norm(
                effective_shape.double() - actual_shape.double()).cpu()))
        raw_gradient = gate.grad.detach().clone()
        raw_gradient_signatures.append(_tensor_signature(raw_gradient))
        source_actions = _source_actions(
            model, rows, layer, raw_gradient)
        torch.nn.utils.clip_grad_value_(
            model.parameters(), clip_value=float(checkpoint["config"]["clip"]))
        clipped_gradient = gate.grad.detach().clone()
        clipped_gradient_signatures.append(_tensor_signature(clipped_gradient))
        learning_rate = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        optimizer.step()
        scheduler.step()
        state_after = optimizer.state[gate]
        native_moment_transitions.append(_native_moment_transition_record(
            state_before, state_after, clipped_gradient, group))
        shadow = _bridge_directions(
            gate_before, raw_gradient, clipped_gradient, state_before,
            int(state_after["step"].item()), group, source_actions)
        native_gate = gate.detach().clone()
        decay_only = (1.0 - learning_rate * weight_decay) * gate_before
        decay_vector = learning_rate * weight_decay * gate_before
        decay_free_gate = native_gate + decay_vector

        gate_parameter_before = _parameter_digest(model, exclude=(gate_name,))
        gate_optimizer_before = _optimizer_named_state_digest(
            model, optimizer, exclude=(gate_name,))
        selected_gate_optimizer_before = nested_state_digest(
            optimizer.state.get(gate, {}))
        if arm == INTERVENTION_COMPARISON_NAMES[0]:
            # Suppress the complete selected-gate adaptive displacement only.
            # The native Adam moments just written by optimizer.step remain.
            with torch.no_grad():
                gate.copy_(decay_only)
        elif arm == "decay_source_removal":
            with torch.no_grad():
                gate.add_(decay_vector)
        elif arm not in {"baseline", clamp_name}:
            raise ValueError("unknown observable intervention arm")
        gate_parameter_after = _parameter_digest(model, exclude=(gate_name,))
        gate_optimizer_after = _optimizer_named_state_digest(
            model, optimizer, exclude=(gate_name,))
        selected_gate_optimizer_after = nested_state_digest(
            optimizer.state.get(gate, {}))
        measured_target_residuals = {
            INTERVENTION_COMPARISON_NAMES[0]: gate.detach() - decay_only,
            "decay_source_removal": decay_free_gate - gate.detach(),
            clamp_name: effective_shape.double() - forced_shape.double(),
        }
        for name, value in measured_target_residuals.items():
            target_residual_signatures[name].append(_tensor_signature(value))
        for name in INTERVENTION_COMPARISON_NAMES[:2]:
            direct_parameter_bindings[name].append({
                "before": gate_parameter_before,
                "after": gate_parameter_after,
            })
            direct_optimizer_bindings[name].append({
                "before": gate_optimizer_before,
                "after": gate_optimizer_after,
            })
            selected_gate_optimizer_bindings[name].append({
                "before": selected_gate_optimizer_before,
                "after": selected_gate_optimizer_after,
            })
        direct_parameter_bindings[clamp_name].append({
            "before": shape_parameter_before,
            "after": shape_parameter_after,
        })
        direct_optimizer_bindings[clamp_name].append({
            "before": shape_optimizer_before,
            "after": shape_optimizer_after,
        })
        selected_gate_optimizer_bindings[clamp_name].append({
            "before": shape_gate_optimizer_before,
            "after": shape_gate_optimizer_after,
        })

        # The clamp hook has been removed, so this is an unclamped post-update
        # probe of the executed successor.
        hook_absent = hook.id not in shape_module._forward_hooks
        if not hook_absent:
            raise RuntimeError(
                "normalized-shape clamp remained active during successor probe")
        clamp_hook_removed_before_probe.append(hook_absent)
        target_probe = _probe_node(
            model, contexts, layer, head_indices, union_graph)
        q_next = np.asarray(target_probe["q"][0], dtype=np.float64)
        exact_shadow_gate = _counterfactual_shadow_gate(
            gate_before.detach().double().cpu().numpy(),
            shadow["adaptive_direction"].detach().cpu().numpy(),
            learning_rate, weight_decay, arm)
        fp = native_roundoff_charge(
            exact_shadow_gate, gate.detach().double().cpu().numpy(), q_next)
        fp_charges.append(float(fp["floating_point_charge"]))
        sector_norms.append(np.linalg.norm(
            shadow["sector_directions"].detach().cpu().numpy(), axis=1).tolist())
        successor_gate_sha256.append(tensor_mapping_digest({
            "gate": gate.detach()}))
        successor_model_sha256.append(tensor_mapping_digest(model.state_dict()))
        successor_optimizer_sha256.append(nested_state_digest(
            optimizer.state_dict()))
        successor_scheduler_sha256.append(nested_state_digest(
            scheduler.state_dict()))
        energies.append(float(target_probe["energy"][0]))
    fixed_source_after = tensor_mapping_digest(source_model.state_dict())
    return {
        "arm": arm,
        "initial_state": initial_state,
        "initial_state_sha256": initial_state_sha256,
        "energy": energies,
        "floating_point_charge": fp_charges,
        "sector_norm_by_step": sector_norms,
        "intervention_target_residual_by_comparison": (
            target_residual_signatures),
        "raw_gradient_signature_by_step": raw_gradient_signatures,
        "clipped_gradient_signature_by_step": (
            clipped_gradient_signatures),
        "native_selected_gate_moment_transition_by_step": (
            native_moment_transitions),
        "successor_gate_sha256_by_step": successor_gate_sha256,
        "successor_model_sha256_by_step": successor_model_sha256,
        "successor_optimizer_sha256_by_step": successor_optimizer_sha256,
        "successor_scheduler_sha256_by_step": successor_scheduler_sha256,
        "pre_clamp_actual_normalized_activation_signature_by_step": (
            actual_shape_signatures),
        "forced_fixed_source_normalized_activation_signature_by_step": (
            forced_shape_signatures),
        "effective_executed_normalized_activation_signature_by_step": (
            effective_shape_signatures),
        "effective_minus_actual_normalized_activation_norm_by_step": (
            effective_actual_difference_norms),
        "fixed_source_reference_model_sha256_before": fixed_source_before,
        "fixed_source_reference_model_sha256_after": fixed_source_after,
        "fixed_source_model_binding_by_step": fixed_source_model_bindings,
        "clamp_hook_removed_before_probe_by_step": (
            clamp_hook_removed_before_probe),
        "direct_non_target_parameter_binding_by_comparison": (
            direct_parameter_bindings),
        "direct_non_target_optimizer_binding_by_comparison": (
            direct_optimizer_bindings),
        "selected_gate_optimizer_binding_by_comparison": (
            selected_gate_optimizer_bindings),
        "minibatch_sha256_by_step": batch_digests,
    }


def _derive_normalized_shape_activation_binding(
    baseline, modified, comparison_name, continuation_steps,
):
    actual_key = "pre_clamp_actual_normalized_activation_signature_by_step"
    forced_key = "forced_fixed_source_normalized_activation_signature_by_step"
    effective_key = (
        "effective_executed_normalized_activation_signature_by_step")
    norm_key = "effective_minus_actual_normalized_activation_norm_by_step"
    paths = {
        "baseline_actual": baseline.get(actual_key, []),
        "baseline_forced": baseline.get(forced_key, []),
        "baseline_effective": baseline.get(effective_key, []),
        "modified_actual": modified.get(actual_key, []),
        "modified_forced": modified.get(forced_key, []),
        "modified_effective": modified.get(effective_key, []),
        "baseline_difference_norm": baseline.get(norm_key, []),
        "modified_difference_norm": modified.get(norm_key, []),
    }
    if any(len(value) != continuation_steps for value in paths.values()):
        raise ValueError("normalized-shape activation evidence is incomplete")
    baseline_source_bindings = baseline.get(
        "fixed_source_model_binding_by_step", [])
    modified_source_bindings = modified.get(
        "fixed_source_model_binding_by_step", [])
    baseline_unclamped_probe = baseline.get(
        "clamp_hook_removed_before_probe_by_step", [])
    modified_unclamped_probe = modified.get(
        "clamp_hook_removed_before_probe_by_step", [])
    if not (
        len(baseline_source_bindings) == len(modified_source_bindings)
        == len(baseline_unclamped_probe) == len(modified_unclamped_probe)
        == continuation_steps
    ):
        raise ValueError(
            "fixed-source or unclamped-probe evidence is incomplete")
    tolerance = ARITHMETIC["absolute_identity_tolerance"]
    same_reference = [
        left["sha256"] == right["sha256"]
        for left, right in zip(
            paths["baseline_forced"], paths["modified_forced"], strict=True)
    ]
    baseline_identity = [
        actual["sha256"] == effective["sha256"]
        and float(delta) <= tolerance
        for actual, effective, delta in zip(
            paths["baseline_actual"], paths["baseline_effective"],
            paths["baseline_difference_norm"], strict=True)
    ]
    clamp_name = INTERVENTION_COMPARISON_NAMES[2]
    expected_modified = (
        paths["modified_forced"] if comparison_name == clamp_name
        else paths["modified_actual"])
    modified_binding = [
        expected["sha256"] == effective["sha256"]
        for expected, effective in zip(
            expected_modified, paths["modified_effective"], strict=True)
    ]
    changed = [
        actual["sha256"] != effective["sha256"]
        and float(delta) > tolerance
        for actual, effective, delta in zip(
            paths["modified_actual"], paths["modified_effective"],
            paths["modified_difference_norm"], strict=True)
    ]
    if comparison_name != clamp_name:
        modified_binding = [
            value and float(delta) <= tolerance
            for value, delta in zip(
                modified_binding, paths["modified_difference_norm"], strict=True)
        ]
    source_digest = baseline.get(
        "fixed_source_reference_model_sha256_before")
    fixed_source_step_binding = [
        left.get("before") == left.get("after") == source_digest
        and right.get("before") == right.get("after") == source_digest
        for left, right in zip(
            baseline_source_bindings, modified_source_bindings, strict=True)
    ]
    fixed_source_unchanged = bool(
        source_digest
        == baseline.get("fixed_source_reference_model_sha256_after")
        == modified.get("fixed_source_reference_model_sha256_before")
        == modified.get("fixed_source_reference_model_sha256_after")
        == baseline.get("initial_state", {}).get("model_sha256")
        == modified.get("initial_state", {}).get("model_sha256")
        and all(fixed_source_step_binding))
    successor_probe_unclamped = [
        left is True and right is True
        for left, right in zip(
            baseline_unclamped_probe, modified_unclamped_probe, strict=True)
    ]
    clamp_changed = any(changed) if comparison_name == clamp_name else False
    binding_valid = bool(
        fixed_source_unchanged and all(same_reference)
        and all(successor_probe_unclamped)
        and all(baseline_identity) and all(modified_binding)
        and (clamp_changed if comparison_name == clamp_name else not any(changed)))
    return {
        "fixed_source_reference_unchanged": fixed_source_unchanged,
        "fixed_source_model_unchanged_by_step": (
            fixed_source_step_binding),
        "successor_probe_unclamped_by_step": successor_probe_unclamped,
        "same_minibatch_fixed_source_reference_by_step": same_reference,
        "baseline_effective_equals_actual_by_step": baseline_identity,
        "modified_effective_matches_operator_by_step": modified_binding,
        "modified_effective_change_norm_by_step": [
            float(value) for value in paths["modified_difference_norm"]],
        "modified_effective_changed_beyond_tolerance_by_step": changed,
        "clamp_changed_beyond_tolerance": clamp_changed,
        "binding_valid": binding_valid,
    }


def validate_intervention_units(comparisons, lock):
    """Derive deterministic seed-stratified S3 summaries from raw evidence."""

    expected_names = list(INTERVENTION_COMPARISON_NAMES)
    if [row.get("name") for row in comparisons] != expected_names:
        raise ValueError("intervention comparison order changed")
    locked_list = lock["intervention"]["comparisons"]
    if [row.get("name") for row in locked_list] != expected_names:
        raise ValueError("locked intervention comparison order changed")
    for contract, locked in zip(
        INTERVENTION_OPERATOR_CONTRACTS, locked_list, strict=True,
    ):
        if _operator_contract(locked) != dict(contract):
            raise ValueError("locked intervention operator contract changed")
    locked_rows = {row["name"]: row for row in locked_list}
    continuation_steps = int(lock["intervention"].get(
        "short_continuation_steps", REGISTRY["intervention_steps"]))
    summaries = []
    clamp_name = INTERVENTION_COMPARISON_NAMES[2]
    for comparison in comparisons:
        units = comparison.get("units", [])
        if len(units) != STATISTICS["intervention_sites"]:
            raise ValueError("an intervention comparison does not have 24 sites")
        name = comparison["name"]
        locked = locked_rows[name]
        operator_contract = dict(INTERVENTION_OPERATOR_BY_NAME[name])
        differences = []
        seeds = []
        substreams = []
        intervals_by_seed = {
            seed: [] for seed in STATISTICS["heldout_seeds"]}
        differences_by_seed = {
            seed: [] for seed in STATISTICS["heldout_seeds"]}
        derived_claims = []
        for unit in units:
            required = {
                "unit_id", "seed", "rng_substream", "future_update_interval",
                "operator_contract", "operator_contract_unchanged",
                "common_initial_state", "common_minibatch_sequence",
                "intended_intervention_target_changed",
                "executed_successor_changed",
                "direct_operator_non_target_invariant",
                "selected_gate_optimizer_state_retained",
                "native_selected_gate_moment_transition_valid",
                "registered_scheduler_sequence_unchanged",
                "per_step_execution_evidence_complete",
                "normalized_shape_activation_binding_valid",
                "clamp_raw_gate_gradient_changed", "no_op_rejected",
                "locked_direction", "locked_effect_lower_threshold",
                "locked_direction_unchanged",
                "locked_lower_effect_interval_unchanged", "difference",
                "intended_intervention_target_difference_norm_by_step",
                "intended_intervention_target_changed_by_step",
                "raw_gradient_changed_by_step",
                "direct_non_target_parameter_invariant_by_step",
                "direct_non_target_optimizer_invariant_by_step",
                "selected_gate_optimizer_state_retained_by_step",
                "native_selected_gate_moment_transition_valid_by_step",
                "registered_scheduler_sequence_unchanged_by_step",
                "normalized_shape_activation_binding",
                "baseline", "modified",
            }
            if set(unit) != required:
                raise ValueError("intervention unit schema changed")
            baseline = unit["baseline"]
            modified = unit["modified"]
            if (
                baseline.get("arm") != "baseline"
                or modified.get("arm") != name
                or len(baseline.get("energy", [])) != continuation_steps + 1
                or len(modified.get("energy", [])) != continuation_steps + 1
            ):
                raise ValueError("intervention arm or successor-energy path changed")
            common_initial = bool(
                baseline.get("initial_state_sha256")
                == modified.get("initial_state_sha256")
                and baseline.get("initial_state") == modified.get("initial_state"))
            common_batches = bool(
                baseline.get("minibatch_sha256_by_step")
                == modified.get("minibatch_sha256_by_step")
                and len(baseline.get("minibatch_sha256_by_step", []))
                == continuation_steps)
            baseline_target = baseline[
                "intervention_target_residual_by_comparison"][name]
            modified_target = modified[
                "intervention_target_residual_by_comparison"][name]
            if (
                len(baseline_target) != continuation_steps
                or len(modified_target) != continuation_steps
            ):
                raise ValueError(
                    "intervention target-residual path is incomplete")
            difference_norm = [float(row["norm"]) for row in baseline_target]
            target_changed_by_step = [
                bool(
                    left["sha256"] != right["sha256"]
                    and difference > ARITHMETIC["absolute_identity_tolerance"]
                    and float(right["norm"])
                    <= ARITHMETIC["absolute_identity_tolerance"])
                for left, right, difference in zip(
                    baseline_target, modified_target, difference_norm,
                    strict=True)
            ]
            baseline_gradients = baseline.get(
                "raw_gradient_signature_by_step", [])
            modified_gradients = modified.get(
                "raw_gradient_signature_by_step", [])
            if (
                len(baseline_gradients) != continuation_steps
                or len(modified_gradients) != continuation_steps
            ):
                raise ValueError("raw gate-gradient evidence is incomplete")
            gradient_changed = [
                left["sha256"] != right["sha256"]
                for left, right in zip(
                    baseline_gradients, modified_gradients, strict=True)
            ]
            parameter_binding = []
            optimizer_binding = []
            selected_optimizer_binding = []
            for left, right in zip(
                baseline[
                    "direct_non_target_parameter_binding_by_comparison"][name],
                modified[
                    "direct_non_target_parameter_binding_by_comparison"][name],
                strict=True,
            ):
                parameter_binding.append(bool(
                    left["before"] == left["after"]
                    and right["before"] == right["after"]))
            for left, right in zip(
                baseline[
                    "direct_non_target_optimizer_binding_by_comparison"][name],
                modified[
                    "direct_non_target_optimizer_binding_by_comparison"][name],
                strict=True,
            ):
                optimizer_binding.append(bool(
                    left["before"] == left["after"]
                    and right["before"] == right["after"]))
            for left, right in zip(
                baseline["selected_gate_optimizer_binding_by_comparison"][name],
                modified["selected_gate_optimizer_binding_by_comparison"][name],
                strict=True,
            ):
                selected_optimizer_binding.append(bool(
                    left["before"] == left["after"]
                    and right["before"] == right["after"]))
            if not (
                len(parameter_binding) == len(optimizer_binding)
                == len(selected_optimizer_binding) == continuation_steps
            ):
                raise ValueError("intervention per-step binding path is incomplete")
            shape_binding = _derive_normalized_shape_activation_binding(
                baseline, modified, name, continuation_steps)
            baseline_moment_valid = _validate_native_moment_transition_path(
                baseline, continuation_steps)
            modified_moment_valid = _validate_native_moment_transition_path(
                modified, continuation_steps)
            moment_transition_valid_by_step = [
                left and right for left, right in zip(
                    baseline_moment_valid, modified_moment_valid, strict=True)
            ]
            clamp_gradient_changed = bool(
                any(gradient_changed) if name == clamp_name else True)
            successor_keys = (
                "successor_gate_sha256_by_step",
                "successor_model_sha256_by_step",
                "successor_optimizer_sha256_by_step",
                "successor_scheduler_sha256_by_step",
            )
            if any(
                len(baseline.get(key, [])) != continuation_steps
                or len(modified.get(key, [])) != continuation_steps
                for key in successor_keys
            ):
                raise ValueError("executed successor-state evidence is incomplete")
            scheduler_unchanged_by_step = [
                left == right for left, right in zip(
                    baseline["successor_scheduler_sha256_by_step"],
                    modified["successor_scheduler_sha256_by_step"],
                    strict=True)
            ]
            per_step_evidence_complete = True
            for arm_record in (baseline, modified):
                charges = arm_record.get("floating_point_charge", [])
                sectors = arm_record.get("sector_norm_by_step", [])
                if not (
                    len(charges) == len(sectors) == continuation_steps
                    and all(math.isfinite(float(value)) and float(value) >= 0.0
                            for value in charges)
                    and all(np.all(np.isfinite(np.asarray(row, dtype=np.float64)))
                            for row in sectors)
                ):
                    per_step_evidence_complete = False
            energy_scale = max(
                abs(float(baseline["energy"][-1])),
                abs(float(modified["energy"][-1])), 1e-300)
            successor_changed = bool(
                abs(float(modified["energy"][-1])
                    - float(baseline["energy"][-1]))
                > ARITHMETIC["relative_identity_tolerance"] * energy_scale
                and modified["successor_gate_sha256_by_step"][-1]
                != baseline["successor_gate_sha256_by_step"][-1]
                and modified["successor_model_sha256_by_step"][-1]
                != baseline["successor_model_sha256_by_step"][-1])
            target_changed = any(target_changed_by_step)
            non_target = all(parameter_binding) and all(optimizer_binding)
            selected_moments_retained = all(selected_optimizer_binding)
            native_moment_transition_valid = all(
                moment_transition_valid_by_step)
            scheduler_unchanged = all(scheduler_unchanged_by_step)
            contract_unchanged = bool(
                unit["operator_contract"] == operator_contract
                == _operator_contract(locked))
            shape_binding_valid = bool(shape_binding["binding_valid"])
            no_op_rejected = bool(
                target_changed and successor_changed and contract_unchanged
                and selected_moments_retained
                and native_moment_transition_valid and scheduler_unchanged
                and per_step_evidence_complete and shape_binding_valid
                and clamp_gradient_changed)
            locked_direction = float(unit["locked_direction"])
            locked_threshold = float(unit["locked_effect_lower_threshold"])
            direction_unchanged = (
                locked_direction == float(locked["direction"]))
            threshold_unchanged = (
                locked_threshold == float(locked["effect_lower_threshold"]))
            difference = locked_direction * (
                float(modified["energy"][-1])
                - float(baseline["energy"][-1]))
            difference_scale = max(
                abs(difference), abs(float(unit["difference"])), 1e-300)
            difference_exact = bool(
                abs(float(unit["difference"]) - difference) / difference_scale
                <= ARITHMETIC["relative_identity_tolerance"])
            expected_claims = {
                "operator_contract_unchanged": contract_unchanged,
                "common_initial_state": common_initial,
                "common_minibatch_sequence": common_batches,
                "intended_intervention_target_changed": target_changed,
                "executed_successor_changed": successor_changed,
                "direct_operator_non_target_invariant": non_target,
                "selected_gate_optimizer_state_retained": (
                    selected_moments_retained),
                "native_selected_gate_moment_transition_valid": (
                    native_moment_transition_valid),
                "registered_scheduler_sequence_unchanged": (
                    scheduler_unchanged),
                "per_step_execution_evidence_complete": (
                    per_step_evidence_complete),
                "normalized_shape_activation_binding_valid": (
                    shape_binding_valid),
                "clamp_raw_gate_gradient_changed": clamp_gradient_changed,
                "no_op_rejected": no_op_rejected,
                "locked_direction_unchanged": direction_unchanged,
                "locked_lower_effect_interval_unchanged": threshold_unchanged,
            }
            if (
                any(bool(unit[field]) != bool(value)
                    for field, value in expected_claims.items())
                or unit["operator_contract"] != operator_contract
                or unit[
                    "intended_intervention_target_difference_norm_by_step"
                ] != difference_norm
                or unit[
                    "intended_intervention_target_changed_by_step"
                ] != target_changed_by_step
                or unit["raw_gradient_changed_by_step"] != gradient_changed
                or unit["direct_non_target_parameter_invariant_by_step"]
                != parameter_binding
                or unit["direct_non_target_optimizer_invariant_by_step"]
                != optimizer_binding
                or unit["selected_gate_optimizer_state_retained_by_step"]
                != selected_optimizer_binding
                or unit[
                    "native_selected_gate_moment_transition_valid_by_step"
                ] != moment_transition_valid_by_step
                or unit[
                    "registered_scheduler_sequence_unchanged_by_step"
                ] != scheduler_unchanged_by_step
                or unit["normalized_shape_activation_binding"] != shape_binding
                or not difference_exact
                or not all(expected_claims.values())
            ):
                failed_claims = sorted(
                    field for field, value in expected_claims.items()
                    if bool(unit[field]) != bool(value) or not bool(value))
                raise ValueError(
                    "intervention unit failed causal bindings: "
                    + ", ".join(
                        failed_claims or ["structured evidence mismatch"]))
            seed = int(unit["seed"])
            interval = list(map(int, unit["future_update_interval"]))
            if seed not in intervals_by_seed or len(interval) != 2:
                raise ValueError("intervention seed or interval is unregistered")
            seeds.append(seed)
            substreams.append(int(unit["rng_substream"]))
            intervals_by_seed[seed].append(interval)
            differences_by_seed[seed].append(float(unit["difference"]))
            differences.append(float(unit["difference"]))
            derived_claims.append(expected_claims)
        per_seed_count = STATISTICS["sites_per_heldout_seed"]
        if (
            any(seeds.count(seed) != per_seed_count
                for seed in STATISTICS["heldout_seeds"])
            or len(substreams) != len(set(substreams))
        ):
            raise ValueError("intervention seed split or RNG substream changed")
        for intervals in intervals_by_seed.values():
            ordered_intervals = sorted(intervals)
            if any(
                left[1] > right[0]
                for left, right in zip(
                    ordered_intervals, ordered_intervals[1:], strict=False)
            ):
                raise ValueError("intervention future-token intervals overlap")
        positive_by_seed = {
            str(seed): sum(value > 0.0 for value in differences_by_seed[seed])
            for seed in STATISTICS["heldout_seeds"]
        }
        order_statistic = STATISTICS[
            "lower_effect_order_statistic_per_seed"]
        lower_effect_by_seed = {
            str(seed): sorted(differences_by_seed[seed])[order_statistic - 1]
            for seed in STATISTICS["heldout_seeds"]
        }
        threshold = float(locked["effect_lower_threshold"])
        sign_coverage_pass = all(
            count >= STATISTICS["positive_sites_required_per_seed"]
            for count in positive_by_seed.values())
        effect_floor_pass = all(
            value > 0.0 and value >= threshold
            for value in lower_effect_by_seed.values())
        directional_pass = bool(sign_coverage_pass and effect_floor_pass)
        execution_fields = (
            "operator_contract_unchanged",
            "intended_intervention_target_changed",
            "executed_successor_changed",
            "direct_operator_non_target_invariant",
            "selected_gate_optimizer_state_retained",
            "native_selected_gate_moment_transition_valid",
            "registered_scheduler_sequence_unchanged",
            "per_step_execution_evidence_complete",
            "normalized_shape_activation_binding_valid",
            "clamp_raw_gate_gradient_changed",
            "no_op_rejected",
        )
        execution_evidence_pass = all(
            all(row[field] for field in execution_fields)
            for row in derived_claims)
        summaries.append({
            "name": name,
            "sites": len(units),
            "sites_by_seed": {
                str(seed): seeds.count(seed)
                for seed in STATISTICS["heldout_seeds"]},
            "positive_direction_sites_by_seed": positive_by_seed,
            "positive_sites_required_per_seed": STATISTICS[
                "positive_sites_required_per_seed"],
            "lower_effect_order_statistic_per_seed": order_statistic,
            "lower_effect_by_seed": lower_effect_by_seed,
            "locked_effect_lower_threshold": threshold,
            "seed_stratified_sign_coverage_pass": sign_coverage_pass,
            "seed_stratified_effect_floor_pass": effect_floor_pass,
            "directional_prediction_pass": directional_pass,
            "registered_intervention_execution_pass": (
                execution_evidence_pass),
            "operator_contract_pass": all(
                row["operator_contract_unchanged"] for row in derived_claims),
            "intended_intervention_target_change_pass": all(
                row["intended_intervention_target_changed"]
                for row in derived_claims),
            "executed_successor_change_pass": all(
                row["executed_successor_changed"] for row in derived_claims),
            "direct_operator_non_target_invariance_pass": all(
                row["direct_operator_non_target_invariant"]
                for row in derived_claims),
            "selected_gate_optimizer_state_retention_pass": all(
                row["selected_gate_optimizer_state_retained"]
                for row in derived_claims),
            "native_selected_gate_moment_transition_pass": all(
                row["native_selected_gate_moment_transition_valid"]
                for row in derived_claims),
            "registered_scheduler_sequence_pass": all(
                row["registered_scheduler_sequence_unchanged"]
                for row in derived_claims),
            "per_step_execution_evidence_pass": all(
                row["per_step_execution_evidence_complete"]
                for row in derived_claims),
            "normalized_shape_activation_binding_pass": all(
                row["normalized_shape_activation_binding_valid"]
                for row in derived_claims),
            "clamp_raw_gate_gradient_change_pass": all(
                row["clamp_raw_gate_gradient_changed"]
                for row in derived_claims),
            "no_op_rejection_pass": all(
                row["no_op_rejected"] for row in derived_claims),
            "locked_direction_pass": all(
                row["locked_direction_unchanged"] for row in derived_claims),
            "locked_lower_effect_interval_pass": all(
                row["locked_lower_effect_interval_unchanged"]
                for row in derived_claims),
        })
    return summaries


def produce_intervention(arguments):
    started = time.monotonic()
    lock = _sealed_lock(arguments.lock)
    registry = _load_registry(arguments.registry)
    if sha256_path(arguments.tokens) != registry["dataset_sha256"]:
        raise ValueError("I token archive changed")
    device = torch.device(arguments.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    union_graph, _resistance = _locked_union_graph(lock)
    layer = int(lock["selected"]["layer"])
    head_indices = np.arange(
        REGISTRY["heldout_contexts"], dtype=np.int64) % ARCHITECTURE["heads"]
    comparison_names = list(INTERVENTION_COMPARISON_NAMES)
    comparisons = {name: [] for name in comparison_names}
    checkpoints = {}
    for seed in STATISTICS["heldout_seeds"]:
        template = (
            arguments.checkpoint_5555 if seed == 5555
            else arguments.checkpoint_6666)
        path = str(template).replace(
            "{anchor}", str(lock["selected"]["anchor"]))
        checkpoint = _load_deciding_checkpoint(path, device="cpu")
        if (
            int(checkpoint["config"]["seed"]) != seed
            or checkpoint["measurement_registry"] != registry
        ):
            raise ValueError("I checkpoint identity or registry changed")
        checkpoints[seed] = checkpoint
    locked_comparisons = {
        row["name"]: row for row in lock["intervention"]["comparisons"]}
    for unit in registry["intervention_units"]:
        seed = int(unit["seed"])
        checkpoint = checkpoints[seed]
        context_rows = [
            torch.from_numpy(_token_matrix(
                arguments.tokens, registry["context_length"])[
                    row["chunk_index"]:row["chunk_index"] + 1
                ].astype(np.int64, copy=True)).to(device)
            for row in registry["validation"]]
        source_model = _model_from_raw(checkpoint, device)
        source_model.train()
        _set_unit_rng(unit["rng_substream"], device)
        baseline = _compact_arm_continuation(
            checkpoint, registry, arguments.tokens, lock, union_graph,
            context_rows, head_indices, unit=unit, arm="baseline",
            device=device, source_model=source_model)
        for name in comparison_names:
            _set_unit_rng(unit["rng_substream"], device)
            modified = _compact_arm_continuation(
                checkpoint, registry, arguments.tokens, lock, union_graph,
                context_rows, head_indices, unit=unit, arm=name,
                device=device, source_model=source_model)
            locked = locked_comparisons[name]
            operator_contract = dict(INTERVENTION_OPERATOR_BY_NAME[name])
            contract_unchanged = bool(
                _operator_contract(locked) == operator_contract)
            direction = float(locked["direction"])
            threshold = float(locked["effect_lower_threshold"])
            difference = direction * (
                modified["energy"][-1] - baseline["energy"][-1])
            energy_scale = max(
                abs(modified["energy"][-1]), abs(baseline["energy"][-1]),
                1e-300)
            baseline_target = baseline[
                "intervention_target_residual_by_comparison"][name]
            modified_target = modified[
                "intervention_target_residual_by_comparison"][name]
            target_difference_norm = [
                float(row["norm"]) for row in baseline_target]
            target_changed_by_step = [
                bool(
                    left["sha256"] != right["sha256"]
                    and delta > ARITHMETIC["absolute_identity_tolerance"]
                    and float(right["norm"])
                    <= ARITHMETIC["absolute_identity_tolerance"])
                for left, right, delta in zip(
                    baseline_target, modified_target, target_difference_norm,
                    strict=True)
            ]
            gradient_changed_by_step = [
                left["sha256"] != right["sha256"]
                for left, right in zip(
                    baseline["raw_gradient_signature_by_step"],
                    modified["raw_gradient_signature_by_step"], strict=True)
            ]
            parameter_invariant_by_step = [
                left["before"] == left["after"]
                and right["before"] == right["after"]
                for left, right in zip(
                    baseline[
                        "direct_non_target_parameter_binding_by_comparison"][name],
                    modified[
                        "direct_non_target_parameter_binding_by_comparison"][name],
                    strict=True)
            ]
            optimizer_invariant_by_step = [
                left["before"] == left["after"]
                and right["before"] == right["after"]
                for left, right in zip(
                    baseline[
                        "direct_non_target_optimizer_binding_by_comparison"][name],
                    modified[
                        "direct_non_target_optimizer_binding_by_comparison"][name],
                    strict=True)
            ]
            selected_optimizer_retained_by_step = [
                left["before"] == left["after"]
                and right["before"] == right["after"]
                for left, right in zip(
                    baseline[
                        "selected_gate_optimizer_binding_by_comparison"][name],
                    modified[
                        "selected_gate_optimizer_binding_by_comparison"][name],
                    strict=True)
            ]
            continuation_steps = int(unit["update_count"])
            shape_binding = _derive_normalized_shape_activation_binding(
                baseline, modified, name, continuation_steps)
            baseline_moment_valid = _validate_native_moment_transition_path(
                baseline, continuation_steps)
            modified_moment_valid = _validate_native_moment_transition_path(
                modified, continuation_steps)
            moment_transition_valid_by_step = [
                left and right for left, right in zip(
                    baseline_moment_valid, modified_moment_valid, strict=True)
            ]
            scheduler_unchanged_by_step = [
                left == right for left, right in zip(
                    baseline["successor_scheduler_sha256_by_step"],
                    modified["successor_scheduler_sha256_by_step"],
                    strict=True)
            ]
            per_step_evidence_complete = all(
                len(arm_record["floating_point_charge"])
                == len(arm_record["sector_norm_by_step"])
                == continuation_steps
                and all(
                    math.isfinite(float(value)) and float(value) >= 0.0
                    for value in arm_record["floating_point_charge"])
                and all(
                    np.all(np.isfinite(np.asarray(row, dtype=np.float64)))
                    for row in arm_record["sector_norm_by_step"])
                for arm_record in (baseline, modified))
            common_initial = (
                baseline["initial_state_sha256"]
                == modified["initial_state_sha256"]
                and baseline["initial_state"] == modified["initial_state"])
            same_batches = (
                modified["minibatch_sha256_by_step"]
                == baseline["minibatch_sha256_by_step"])
            target_changed = any(target_changed_by_step)
            successor_changed = bool(
                abs(modified["energy"][-1] - baseline["energy"][-1])
                > ARITHMETIC["relative_identity_tolerance"] * energy_scale
                and modified["successor_gate_sha256_by_step"][-1]
                != baseline["successor_gate_sha256_by_step"][-1]
                and modified["successor_model_sha256_by_step"][-1]
                != baseline["successor_model_sha256_by_step"][-1])
            direct_operator_non_target_invariant = bool(
                all(parameter_invariant_by_step)
                and all(optimizer_invariant_by_step))
            selected_optimizer_retained = all(
                selected_optimizer_retained_by_step)
            native_moment_transition_valid = all(
                moment_transition_valid_by_step)
            scheduler_unchanged = all(scheduler_unchanged_by_step)
            clamp_gradient_changed = bool(
                any(gradient_changed_by_step)
                if name == INTERVENTION_COMPARISON_NAMES[2] else True)
            binding_valid = bool(shape_binding["binding_valid"])
            no_op_rejected = bool(
                target_changed and successor_changed and contract_unchanged
                and selected_optimizer_retained
                and native_moment_transition_valid and scheduler_unchanged
                and per_step_evidence_complete and binding_valid
                and clamp_gradient_changed)
            comparisons[name].append({
                "unit_id": unit["id"],
                "seed": seed,
                "rng_substream": unit["rng_substream"],
                "future_update_interval": [
                    unit["future_update_start"],
                    unit["future_update_start"] + unit["update_count"]],
                "operator_contract": operator_contract,
                "operator_contract_unchanged": contract_unchanged,
                "common_initial_state": common_initial,
                "common_minibatch_sequence": same_batches,
                "intended_intervention_target_changed": target_changed,
                "executed_successor_changed": successor_changed,
                "direct_operator_non_target_invariant": (
                    direct_operator_non_target_invariant),
                "selected_gate_optimizer_state_retained": (
                    selected_optimizer_retained),
                "native_selected_gate_moment_transition_valid": (
                    native_moment_transition_valid),
                "registered_scheduler_sequence_unchanged": (
                    scheduler_unchanged),
                "per_step_execution_evidence_complete": (
                    per_step_evidence_complete),
                "normalized_shape_activation_binding_valid": binding_valid,
                "clamp_raw_gate_gradient_changed": clamp_gradient_changed,
                "no_op_rejected": no_op_rejected,
                "locked_direction": direction,
                "locked_effect_lower_threshold": threshold,
                "locked_direction_unchanged": (
                    direction == locked["direction"]),
                "locked_lower_effect_interval_unchanged": (
                    threshold == locked["effect_lower_threshold"]),
                "difference": float(difference),
                "intended_intervention_target_difference_norm_by_step": (
                    target_difference_norm),
                "intended_intervention_target_changed_by_step": (
                    target_changed_by_step),
                "raw_gradient_changed_by_step": gradient_changed_by_step,
                "direct_non_target_parameter_invariant_by_step": (
                    parameter_invariant_by_step),
                "direct_non_target_optimizer_invariant_by_step": (
                    optimizer_invariant_by_step),
                "selected_gate_optimizer_state_retained_by_step": (
                    selected_optimizer_retained_by_step),
                "native_selected_gate_moment_transition_valid_by_step": (
                    moment_transition_valid_by_step),
                "registered_scheduler_sequence_unchanged_by_step": (
                    scheduler_unchanged_by_step),
                "normalized_shape_activation_binding": shape_binding,
                "baseline": baseline,
                "modified": modified,
            })
    raw_comparisons = [
        {"name": name, "units": comparisons[name]}
        for name in comparison_names]
    summaries = validate_intervention_units(raw_comparisons, lock)
    checks = {
        "twenty_four_preregistered_sites": all(
            row["sites"] == STATISTICS["intervention_sites"]
            for row in summaries),
        "three_comparisons": len(summaries) == 3,
        "common_native_source": all(
            unit["common_initial_state"] for row in raw_comparisons
            for unit in row["units"]),
        "common_actual_minibatch_sequence": all(
            unit["common_minibatch_sequence"] for row in raw_comparisons
            for unit in row["units"]),
        "registered_intervention_execution_evidence": all(
            row["registered_intervention_execution_pass"]
            and row["operator_contract_pass"]
            and row["intended_intervention_target_change_pass"]
            and row["executed_successor_change_pass"]
            and row["direct_operator_non_target_invariance_pass"]
            and row["selected_gate_optimizer_state_retention_pass"]
            and row["native_selected_gate_moment_transition_pass"]
            and row["registered_scheduler_sequence_pass"]
            and row["per_step_execution_evidence_pass"]
            and row["normalized_shape_activation_binding_pass"]
            and row["clamp_raw_gate_gradient_change_pass"]
            and row["no_op_rejection_pass"]
            for row in summaries),
        "seed_stratified_sign_coverage": all(
            row["seed_stratified_sign_coverage_pass"] for row in summaries),
        "seed_stratified_effect_floor": all(
            row["seed_stratified_effect_floor_pass"] for row in summaries),
    }
    if set(checks) != set(STAGE_BY_ID["I"]["required_checks"]):
        raise ArithmeticError("I stage-check names drifted from the protocol")
    value = {
        "schema_version": INTERVENTION_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "stage": "I",
        "job_id": arguments.job_id,
        "construction_lock_sha256": lock["lock_sha256"],
        "probe_registry_sha256": registry["registry_sha256"],
        "comparisons": summaries,
        "units": raw_comparisons,
        "stage_checks": checks,
        "resources": _resource_record(device, started),
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    value["decision_sha256"] = digest_object(value)
    write_json_atomic(arguments.output, value)


def _executor_owned_staged_runtime_path(relative):
    relative = Path(relative)
    if relative.as_posix() in {
        "campaign-attempt-ledger.json",
        "campaign-attempt-ledger.json.lock",
    }:
        return True
    return bool(
        len(relative.parts) > 1
        and relative.parts[0] in {"results", "records", "summaries"}
    )


def _canonical_checksum_relative(value):
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\\" in value
    ):
        raise ValueError("checksum manifest target name is malformed")
    relative = Path(value)
    if (
        relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.as_posix() != value
    ):
        raise ValueError("checksum manifest target name is noncanonical")
    return relative


def _verify_checksum_manifest(root):
    requested_root = Path(root).absolute()
    if requested_root.is_symlink() or not requested_root.is_dir():
        raise ValueError("staged bundle root is missing, symbolic, or nondirectory")
    root = requested_root.resolve(strict=True)
    manifest = root / "CHECKSUMS.sha256"
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError(
            "staged checksum manifest is missing, symbolic, or nonregular")

    immutable_files = set()
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise ValueError(
                "staged bundle contains symbolic link: "
                + candidate.relative_to(root).as_posix())
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise ValueError(
                "staged bundle contains nonregular node: "
                + candidate.relative_to(root).as_posix())
        relative = candidate.relative_to(root)
        if relative == Path("CHECKSUMS.sha256"):
            continue
        if _executor_owned_staged_runtime_path(relative):
            continue
        immutable_files.add(relative.as_posix())

    try:
        lines = manifest.read_bytes().decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("checksum manifest is not ASCII") from error
    seen = {}
    for line in lines:
        digest, separator, raw_relative = line.partition("  ")
        if (
            separator != "  "
            or len(digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in digest
            )
        ):
            raise ValueError("checksum manifest line is malformed")
        relative = _canonical_checksum_relative(raw_relative)
        relative_name = relative.as_posix()
        if relative_name == "CHECKSUMS.sha256":
            raise ValueError("checksum manifest cannot list itself")
        if _executor_owned_staged_runtime_path(relative):
            raise ValueError(
                "checksum manifest cannot list executor-owned runtime paths")
        if relative_name in seen:
            raise ValueError(
                f"checksum manifest has duplicate target: {relative_name}")
        target = root / relative
        if target.is_symlink() or not target.is_file():
            raise ValueError(
                "checksum manifest target is missing, symbolic, or nonregular: "
                + relative_name)
        if sha256_path(target) != digest:
            raise ValueError(f"checksum mismatch for {relative_name}")
        seen[relative_name] = digest

    if not seen:
        raise ValueError("checksum manifest is empty")
    observed_files = set(seen)
    if observed_files != immutable_files:
        omitted = sorted(immutable_files - observed_files)
        invented = sorted(observed_files - immutable_files)
        raise ValueError(
            "checksum manifest coverage mismatch; omitted="
            + ",".join(omitted)
            + "; invented="
            + ",".join(invented))
    return len(seen)


def _result_manifest_targets(results_root, manifest_path, *, excluded=()):
    results_root = Path(results_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_relative_to(results_root):
        raise ValueError("results manifest must stay below the results root")
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    excluded_paths = {Path(value).resolve() for value in excluded}
    if any(not value.is_relative_to(results_root) for value in excluded_paths):
        raise ValueError("excluded result artifact leaves the results root")
    targets = {}
    for candidate in sorted(results_root.rglob("*")):
        if candidate.is_symlink():
            raise ValueError("results manifest refuses symbolic links")
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        if resolved in {manifest_path, temporary} | excluded_paths:
            continue
        if not resolved.is_relative_to(results_root):
            raise ValueError("result artifact leaves the results root")
        relative = resolved.relative_to(results_root).as_posix()
        targets[relative] = resolved
    if not targets:
        raise ValueError("results manifest cannot be empty")
    return targets


def _write_results_manifest(results_root, manifest_path, *, excluded=()):
    """Write the detached manifest after all pre-release outputs exist."""

    manifest_path = Path(manifest_path).resolve()
    targets = _result_manifest_targets(
        results_root, manifest_path, excluded=excluded)
    lines = [
        f"{sha256_path(targets[relative])}  {relative}"
        for relative in sorted(targets)]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="ascii")
    temporary.replace(manifest_path)
    return len(lines)


def _verify_results_manifest(results_root, manifest_path, *, excluded=()):
    """Independently replay a complete, omission-sensitive result manifest."""

    results_root = Path(results_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    expected = _result_manifest_targets(
        results_root, manifest_path, excluded=excluded)
    observed = {}
    for line in manifest_path.read_text(encoding="ascii").splitlines():
        digest, separator, relative = line.partition("  ")
        candidate = Path(relative)
        if (
            separator != "  " or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or candidate.is_absolute() or not candidate.parts
            or any(part in {"", ".", ".."} for part in candidate.parts)
            or relative in observed
        ):
            raise ValueError("results manifest line is malformed")
        observed[relative] = digest
    if set(observed) != set(expected):
        raise ValueError("results manifest omits or invents an artifact")
    for relative, target in expected.items():
        if sha256_path(target) != observed[relative]:
            raise ValueError(f"results checksum mismatch for {relative}")
    return len(observed)


def _verify_staged_live_sources(staged_root):
    """Bind staged executable sources to the live repository bytes."""

    code_root = (Path(staged_root).resolve() / "code").resolve()
    if not code_root.is_dir():
        raise FileNotFoundError(code_root)
    checked = 0
    for staged in sorted(code_root.rglob("*")):
        if staged.is_symlink():
            raise ValueError("staged source bundle contains a symbolic link")
        if not staged.is_file():
            continue
        relative = staged.resolve().relative_to(code_root)
        live = (ROOT / relative).resolve()
        if not live.is_relative_to(ROOT) or not live.is_file():
            raise ValueError(f"staged source has no live repository peer: {relative}")
        if sha256_path(staged) != sha256_path(live):
            raise ValueError(f"staged/live source drift for {relative}")
        checked += 1
    if checked == 0:
        raise ValueError("staged source bundle is empty")
    return checked


def _binding_is_current(binding):
    try:
        target = Path(binding["path"]).resolve()
        return bool(
            target.is_file()
            and target.stat().st_size == int(binding["bytes"])
            and sha256_path(target) == binding["sha256"])
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _pre_release_records_match_plan(plan, records, release_record_key):
    """Bind every latest pre-release attempt to its canonical job and artifacts."""

    jobs = {row["record_key"]: row for row in plan.get("jobs", [])}
    raw_environment_path = plan.get("environment_attestation")
    if not isinstance(raw_environment_path, str) or not Path(
        raw_environment_path).is_absolute():
        return False, 0
    environment_path = Path(raw_environment_path).resolve()
    if (
        release_record_key not in jobs
        or len(jobs) != len(plan.get("jobs", []))
        or any(row.get("record_key") not in jobs for row in records)
    ):
        return False, 0
    latest = {}
    for record in records:
        key = record["record_key"]
        if key == release_record_key:
            continue
        previous = latest.get(key)
        if previous is None or record["attempt"] > previous["attempt"]:
            latest[key] = record
    expected = set(jobs) - {release_record_key}
    if set(latest) != expected:
        return False, len(latest)
    for key in sorted(expected):
        job = jobs[key]
        record = latest[key]
        allowed_status = (
            {"PASS", "FAIL"} if job["stage"] == "ROB" else {"PASS"})
        expected_native = (
            "float32" if job["stage"] in {"T", "B", "H", "ROB", "M", "I"}
            else "not-applicable")
        expected_deciding = (
            "float64-shadow"
            if job["stage"] in {"B", "H", "ROB", "M", "I", "F"}
            else "not-applicable")
        plan_inputs = [
            binding for binding in record["inputs"]
            if binding.get("role") == "execution-plan"]
        environment_inputs = [
            binding for binding in record["inputs"]
            if binding.get("role") == "runtime-environment-attestation"]
        identity_pass = bool(
            record["status"] in allowed_status
            and record["stage"] == job["stage"]
            and record["job_id"] == job["job_id"]
            and record["role"] == job["role"]
            and record["command"] == job["command"]
            and set(record["checks"]) == set(job["owned_checks"])
            and record["native_arithmetic"] == expected_native
            and record["deciding_arithmetic"] == expected_deciding
            and len(plan_inputs) == 1
            and len(environment_inputs) == 1
            and Path(environment_inputs[0]["path"]).resolve()
            == environment_path
            and record["resources"]["requested_device"] == job["device"])
        bindings = [*record["inputs"], *record["outputs"]]
        artifacts_pass = bool(
            bindings and all(_binding_is_current(binding) for binding in bindings))
        primary_bound = bool(
            record["status"] != "PASS"
            or any(
                Path(binding["path"]).resolve() == Path(job["output"]).resolve()
                for binding in record["outputs"]))
        if not (identity_pass and artifacts_pass and primary_bound):
            return False, len(latest)
    return True, len(latest)


def _no_raw_node_artifacts(results_root):
    forbidden = {
        "raw_model_node", "raw_optimizer_node", "raw_parameter_node",
        "normalized_shape", "normalized_shape_node",
    }
    checked = 0
    for path in Path(results_root).resolve().rglob("*.npz"):
        with np.load(path, allow_pickle=False) as archive:
            names = set(archive.files)
            if forbidden & names or any(
                name.startswith("raw_") and name.endswith("_node")
                for name in names
            ):
                raise ValueError(f"raw parameter-node array retained by {path}")
            if "normalized_edge_contrast" not in names:
                raise ValueError(f"compact ledger omits edge contrasts: {path}")
        checked += 1
    if checked == 0:
        raise ValueError("release verification found no compact ledgers")
    return checked

def _empty_repository_snapshot(error=""):
    return {
        "pass": False,
        "target_count": 0,
        "manifest_sha256": "",
        "staged_manifest_sha256": "",
        "staged_manifest_equal": False,
        "target_aggregate_sha256": "",
        "error": str(error),
    }


def _empty_export_snapshot(error=""):
    return {
        "pass": False,
        "file_count": 0,
        "manifest_entry_count": 0,
        "manifest_sha256": "",
        "staged_manifest_sha256": "",
        "staged_manifest_equal": False,
        "target_aggregate_sha256": "",
        "error": str(error),
    }


def _strict_regular_bytes(path, label):
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing, symbolic, or nonregular")
    return path.read_bytes()






def _resolved_absolute_path(value):
    if not isinstance(value, (str, Path)):
        raise ValueError("release provenance path has an invalid type")
    if isinstance(value, str) and not value:
        raise ValueError("release provenance path is empty")
    if not Path(value).is_absolute():
        raise ValueError("release provenance path is not absolute")
    return Path(value).resolve(strict=False)











