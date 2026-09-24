#!/usr/bin/env python3
"""Analyze one exact AdamW and lifted-transport snapshot for E1."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXPERIMENTS))

from campaign_record import strict_dumps  # noqa: E402
from direct_certificate import complete_defect_vector  # noqa: E402
from measure_tilt import build_model  # noqa: E402
from optimizer_transport import AdamWConstants, ordered_adamw_step  # noqa: E402
from row_map_jacobian import _row_map_value  # noqa: E402


def _array(value, name, *, ndim=None):
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains a nonfinite value")
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have dimension {ndim}")
    return result


def _relative_error(actual, expected):
    actual = _array(actual, "actual")
    expected = _array(expected, "expected")
    if actual.shape != expected.shape:
        raise ValueError("relative-error arrays have different shapes")
    scale = max(
        float(np.linalg.norm(actual)),
        float(np.linalg.norm(expected)),
        1e-30,
    )
    return float(np.linalg.norm(actual - expected) / scale)


def _roundoff_enclosed_error(actual_tensor, expected, operations=64):
    actual_native = actual_tensor.detach().cpu().numpy()
    if actual_native.dtype.kind != "f":
        raise ValueError("optimizer replay tensor is not floating point")
    actual = np.asarray(actual_native, dtype=float)
    expected = _array(expected, "expected")
    if actual.shape != expected.shape:
        raise ValueError("roundoff-enclosure arrays have different shapes")
    raw_norm = float(np.linalg.norm(actual - expected))
    scale = max(
        float(np.linalg.norm(actual)),
        float(np.linalg.norm(expected)),
        1e-30,
    )
    unit = float(np.finfo(actual_native.dtype).eps)
    product = operations * unit
    if product >= 1.0:
        raise ValueError("roundoff operation count is outside its domain")
    gamma = product / (1.0 - product)
    absolute_bound = gamma * (
        float(np.linalg.norm(actual))
        + float(np.linalg.norm(expected))
        + float(np.sqrt(actual.size)) * np.finfo(float).tiny
    )
    enclosed = max(0.0, raw_norm - absolute_bound) / scale
    return float(enclosed), float(raw_norm / scale), float(absolute_bound / scale)


def _enclosure_ratio(vector, bound, name):
    norm = float(np.linalg.norm(_array(vector, name, ndim=1)))
    bound = float(bound)
    if not math.isfinite(bound) or bound < 0.0:
        raise ValueError(f"{name}_bound must be finite and nonnegative")
    if bound == 0.0:
        return 0.0 if norm == 0.0 else float(np.finfo(float).max)
    return norm / bound


def _certificate(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    required = {
        "alpha", "beta1", "lifted_state_before", "lifted_state_after",
        "lifted_matrix", "forcing", "normal_residual", "decay_defect",
        "row_motion_defect", "intervention_defect", "taylor_remainder",
        "coordinate_motion_defect", "recorded_complete_defect",
        "coordinate_motion_bound", "taylor_remainder_bound",
        "intervention_observed", "intervention_expected",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError(
            "lifted certificate does not match the closed E1 schema")
    return payload


def analyze_snapshot(snapshot_path, certificate_path, tensor_group=None):
    snapshot = torch.load(
        snapshot_path, map_location="cpu", weights_only=True)
    if snapshot.get("schema_version") != "pldr-adamw-transport-v2":
        raise ValueError("transport snapshot schema is stale")
    expected_order = [
        "averaged_raw_gradient",
        "value_clip",
        "first_moment",
        "second_moment",
        "bias_correction",
        "decoupled_decay_and_loss_update",
        "post_step_intervention",
    ]
    if snapshot.get("operation_order") != expected_order:
        raise ValueError("transport operation order is stale")
    eta = float(snapshot["learning_rate_applied"])
    eta_next = float(snapshot["learning_rate_prepared_next"])
    if not math.isfinite(eta_next) or eta_next < 0.0:
        raise ValueError(
            "learning_rate_prepared_next must be finite and nonnegative")
    replay_errors = []
    raw_replay_errors = []
    roundoff_bounds = []
    clip_errors = []
    state_steps = set()
    beta1_values = set()
    weight_decay_values = set()
    tensor_count = 0
    if tensor_group is None:
        selected_blocks = snapshot["blocks"]
    else:
        if tensor_group not in snapshot["blocks"]:
            raise ValueError("selected tensor group is absent from snapshot")
        selected_blocks = {tensor_group: snapshot["blocks"][tensor_group]}
    for block_name, tensors in selected_blocks.items():
        if not tensors:
            raise ValueError(f"snapshot block {block_name} is empty")
        for row in tensors:
            tensor_count += 1
            if row["raw_gradient"] is None:
                raise ValueError(
                    f"snapshot tensor {row['name']} has no gradient")
            state_step = int(row["state_step_after"])
            if state_step != int(row["state_step_before"]) + 1:
                raise ValueError("AdamW state clock did not advance once")
            state_steps.add(state_step)
            constants = AdamWConstants(
                beta1=float(row["beta1"]),
                beta2=float(row["beta2"]),
                epsilon=float(row["epsilon"]),
                weight_decay=float(row["weight_decay"]),
                clip_value=row["clip_value"],
            )
            beta1_values.add(constants.beta1)
            weight_decay_values.add(constants.weight_decay)
            replay = ordered_adamw_step(
                row["theta_before"].numpy(),
                row["raw_gradient"].numpy(),
                row["first_moment_before"].numpy(),
                row["second_moment_before"].numpy(),
                step=state_step,
                learning_rate=eta,
                constants=constants,
            )
            comparisons = (
                (row["clipped_gradient"], replay["clipped_gradient"]),
                (row["first_moment_after"], replay["first_moment"]),
                (row["second_moment_after"], replay["second_moment"]),
                (row["theta_after_optimizer"], replay["theta"]),
            )
            current_errors = []
            current_raw = []
            for actual, expected in comparisons:
                enclosed, raw, rounding = _roundoff_enclosed_error(
                    actual, expected)
                current_errors.append(enclosed)
                current_raw.append(raw)
                roundoff_bounds.append(rounding)
            replay_errors.extend(current_errors)
            raw_replay_errors.extend(current_raw)
            clip_errors.append(current_errors[0])
    if len(state_steps) != 1:
        raise ValueError("selected tensors disagree on AdamW state step")
    if len(beta1_values) != 1:
        raise ValueError("selected tensors disagree on AdamW beta1")
    state_step = next(iter(state_steps))
    snapshot_beta1 = next(iter(beta1_values))
    maximum_weight_decay = max(weight_decay_values)

    certificate = _certificate(certificate_path)
    snapshot_alpha = eta / (1.0 - snapshot_beta1 ** state_step)
    optimizer_scalar_residual = _relative_error(
        [certificate["alpha"], certificate["beta1"]],
        [snapshot_alpha, snapshot_beta1],
    )
    replay_errors.append(optimizer_scalar_residual)
    assembled = complete_defect_vector(
        alpha=certificate["alpha"],
        beta1=certificate["beta1"],
        normal_residual=certificate["normal_residual"],
        decay_defect=certificate["decay_defect"],
        row_motion_defect=certificate["row_motion_defect"],
        intervention_defect=certificate["intervention_defect"],
        taylor_remainder=certificate["taylor_remainder"],
        coordinate_motion_defect=certificate["coordinate_motion_defect"],
    )
    recorded = _array(
        certificate["recorded_complete_defect"],
        "recorded_complete_defect",
        ndim=1,
    )
    state_before = _array(
        certificate["lifted_state_before"],
        "lifted_state_before",
        ndim=1,
    )
    state_after = _array(
        certificate["lifted_state_after"],
        "lifted_state_after",
        ndim=1,
    )
    matrix = _array(
        certificate["lifted_matrix"], "lifted_matrix", ndim=2)
    forcing = _array(certificate["forcing"], "forcing", ndim=1)
    if (
        matrix.shape != (len(state_before), len(state_before))
        or state_after.shape != state_before.shape
        or forcing.shape != state_before.shape
        or assembled.shape != state_before.shape
        or recorded.shape != state_before.shape
    ):
        raise ValueError("lifted certificate dimensions disagree")
    predicted_after = matrix @ state_before + forcing + assembled
    lifted_residual = _relative_error(state_after, predicted_after)
    defect_residual = _relative_error(recorded, assembled)
    coordinate_ratio = _enclosure_ratio(
        certificate["coordinate_motion_defect"],
        certificate["coordinate_motion_bound"],
        "coordinate_motion",
    )
    taylor_ratio = _enclosure_ratio(
        certificate["taylor_remainder"],
        certificate["taylor_remainder_bound"],
        "taylor_remainder",
    )
    intervention_residual = _relative_error(
        certificate["intervention_observed"],
        certificate["intervention_expected"],
    )
    ledger_residual = max(replay_errors, default=0.0)
    measurements = [
        {
            "name": "adamw_ledger_residual",
            "value": ledger_residual,
            "unit": "relative_l2",
            "method": "operation_ordered_replay_outside_ieee_roundoff_enclosure",
            "status": "OBSERVED",
            "reason_code": None,
        },
        {
            "name": "lifted_recurrence_residual",
            "value": lifted_residual,
            "unit": "relative_l2",
            "method": "complete_matrix_lifted_identity",
            "status": "OBSERVED",
            "reason_code": None,
        },
        {
            "name": "defect_recomposition_residual",
            "value": defect_residual,
            "unit": "relative_l2",
            "method": "primitive_defect_vector_recomposition",
            "status": "OBSERVED",
            "reason_code": None,
        },
        {
            "name": "coordinate_motion_enclosure_ratio",
            "value": coordinate_ratio,
            "unit": "ratio",
            "method": "jacobian_and_preconditioner_motion_bound",
            "status": "OBSERVED",
            "reason_code": None,
        },
        {
            "name": "taylor_remainder_enclosure_ratio",
            "value": taylor_ratio,
            "unit": "ratio",
            "method": "joint_second_derivative_transport_bound",
            "status": "OBSERVED",
            "reason_code": None,
        },
        {
            "name": "intervention_boundary_residual",
            "value": intervention_residual,
            "unit": "relative_l2",
            "method": "post_optimizer_boundary_replay",
            "status": "OBSERVED",
            "reason_code": None,
        },
        {
            "name": "applied_learning_rate",
            "value": eta,
            "unit": "parameter_step",
            "method": "pre_update_optimizer_parameter_group",
            "status": "OBSERVED",
            "reason_code": None,
        },
        {
            "name": "bias_corrected_step",
            "value": snapshot_alpha,
            "unit": "effective_parameter_step",
            "method": "eta_over_one_minus_beta1_power_state_step",
            "status": "OBSERVED",
            "reason_code": None,
        },
        {
            "name": "schedule_increment",
            "value": eta_next - eta,
            "unit": "parameter_step_per_optimizer_step",
            "method": "prepared_next_rate_minus_applied_rate",
            "status": "OBSERVED",
            "reason_code": None,
        },
        {
            "name": "decay_coefficient",
            "value": eta * maximum_weight_decay,
            "unit": "dimensionless",
            "method": "applied_rate_times_maximum_selected_weight_decay",
            "status": "OBSERVED",
            "reason_code": None,
        },
        {
            "name": "optimizer_state_step",
            "value": state_step,
            "unit": "optimizer_step",
            "method": "post_moment_state_clock",
            "status": "OBSERVED",
            "reason_code": None,
        },
    ]
    return {
        "schema_version": "pldr-adamw-direct-transport-v3",
        "global_optimizer_step": int(snapshot["global_optimizer_step"]),
        "adamw_state_step": state_step,
        "tensor_group": tensor_group,
        "tensor_count": tensor_count,
        "maximum_clip_replay_error": max(clip_errors, default=0.0),
        "maximum_transport_replay_error": ledger_residual,
        "maximum_raw_transport_replay_error":
            max(raw_replay_errors, default=0.0),
        "maximum_relative_roundoff_enclosure":
            max(roundoff_bounds, default=0.0),
        "optimizer_scalar_residual": optimizer_scalar_residual,
        "snapshot_alpha": snapshot_alpha,
        "snapshot_beta1": snapshot_beta1,
        "learning_rate_applied": eta,
        "learning_rate_prepared_next": eta_next,
        "schedule_increment": eta_next - eta,
        "assembled_complete_defect": assembled.tolist(),
        "predicted_lifted_state_after": predicted_after.tolist(),
        "measurements": measurements,
    }


def _run_config(run_directory):
    with (Path(run_directory) / "log.jsonl").open(
        "r", encoding="utf-8"
    ) as stream:
        value = json.loads(stream.readline())
    if value.get("event") != "config":
        raise ValueError("run log does not start with a config record")
    return value


def _selected_rows(snapshot, tensor_group):
    blocks = snapshot.get("blocks", {})
    if tensor_group not in blocks or not blocks[tensor_group]:
        raise ValueError("selected tensor group is absent or empty")
    rows = blocks[tensor_group]
    names = [row["name"] for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("selected snapshot tensor names are duplicated")
    return rows


def _snapshot_directions(rows, learning_rate, device):
    state_steps = {int(row["state_step_after"]) for row in rows}
    beta1_values = {float(row["beta1"]) for row in rows}
    if len(state_steps) != 1 or len(beta1_values) != 1:
        raise ValueError("selected snapshot group has inconsistent clocks")
    step = next(iter(state_steps))
    beta1 = next(iter(beta1_values))
    directions = {
        name: {} for name in (
            "preconditioned_moment_before",
            "preconditioned_moment_after",
            "preconditioned_clipped_gradient",
            "total_parameter_displacement",
            "decay_displacement",
            "intervention_displacement",
        )
    }
    for row in rows:
        name = row["name"]
        beta2 = float(row["beta2"])
        epsilon = float(row["epsilon"])
        second_after = row["second_moment_after"].to(
            device=device, dtype=torch.float64)
        denominator = (
            second_after / (1.0 - beta2 ** step)
        ).sqrt() + epsilon
        before = row["theta_before"].to(
            device=device, dtype=torch.float64)
        after = row["theta_after_intervention"].to(
            device=device, dtype=torch.float64)
        after_optimizer = row["theta_after_optimizer"].to(
            device=device, dtype=torch.float64)
        directions["preconditioned_moment_before"][name] = (
            row["first_moment_before"].to(
                device=device, dtype=torch.float64) / denominator
        )
        directions["preconditioned_moment_after"][name] = (
            row["first_moment_after"].to(
                device=device, dtype=torch.float64) / denominator
        )
        directions["preconditioned_clipped_gradient"][name] = (
            row["clipped_gradient"].to(
                device=device, dtype=torch.float64) / denominator
        )
        directions["total_parameter_displacement"][name] = after - before
        directions["decay_displacement"][name] = (
            -learning_rate * float(row["weight_decay"]) * before
        )
        directions["intervention_displacement"][name] = (
            after - after_optimizer)
    return directions, step, beta1


def _projected_row_chart(model, row_by_layer, selected_names, directions):
    named = dict(model.named_parameters())
    missing = sorted(set(selected_names) - set(named))
    if missing:
        raise ValueError("snapshot parameters are absent from model: "
                         + ", ".join(missing))
    values = []
    actions = {name: [] for name in directions}
    for layer_index in range(len(model.decoder.dec_layers)):
        if layer_index not in row_by_layer:
            raise ValueError(f"row artifact omits layer {layer_index}")
        row = torch.as_tensor(
            row_by_layer[layer_index], dtype=torch.float64,
            device=next(model.parameters()).device,
        )
        width = row.numel()
        row_direction = torch.ones_like(row)
        row_direction[1::2] = -1
        row_direction /= torch.linalg.vector_norm(row_direction)
        output_direction = torch.arange(
            1, width + 1, dtype=row.dtype, device=row.device)
        output_direction[1::2] *= -1
        output_direction /= torch.linalg.vector_norm(output_direction)
        scalar = torch.zeros(
            (), dtype=row.dtype, device=row.device, requires_grad=True)
        output = _row_map_value(
            model, layer_index, row + scalar * row_direction)
        projected = torch.autograd.grad(
            output_direction @ output, scalar, create_graph=True)[0]
        values.append(float(projected.detach().cpu()))
        prefix = f"decoder.dec_layers.{layer_index}."
        local_names = [
            name for name in selected_names if name.startswith(prefix)
        ]
        if not local_names:
            for action in actions.values():
                action.append(0.0)
            continue
        local_parameters = [named[name] for name in local_names]
        gradients = torch.autograd.grad(
            projected,
            local_parameters,
            allow_unused=True,
        )
        for direction_name, direction in directions.items():
            total = torch.zeros((), dtype=torch.float64, device=row.device)
            for name, gradient in zip(local_names, gradients):
                if gradient is not None:
                    total = total + (
                        gradient * direction[name]
                    ).sum()
            actions[direction_name].append(float(total.detach().cpu()))
    return np.asarray(values, dtype=float), {
        name: np.asarray(value, dtype=float)
        for name, value in actions.items()
    }


def _row_artifact(path, layer_count):
    result = {}
    with np.load(path) as payload:
        for layer_index in range(layer_count):
            name = f"layer_{layer_index}_registered"
            if name not in payload or len(payload[name]) == 0:
                raise ValueError(f"row artifact is missing {name}")
            result[layer_index] = np.asarray(payload[name][0], dtype=float)
    return result


def _strict_upper(value):
    value = float(value)
    if value == 0.0:
        return 0.0
    return math.nextafter(value, math.inf)


def construct_lifted_certificate(*, snapshot_path, model_checkpoint,
                                 run_directory, row_artifact,
                                 tensor_group, device):
    snapshot = torch.load(
        snapshot_path, map_location="cpu", weights_only=True)
    rows = _selected_rows(snapshot, tensor_group)
    global_step = int(snapshot["global_optimizer_step"])
    checkpoint = torch.load(
        model_checkpoint, map_location=device, weights_only=True)
    if int(checkpoint["step"]) != global_step:
        raise ValueError("E1 model checkpoint and snapshot clocks disagree")
    config = _run_config(run_directory)
    model_after = build_model(config, device).to(
        device=device, dtype=torch.float64).eval()
    model_before = build_model(config, device).to(
        device=device, dtype=torch.float64).eval()
    model_after.load_state_dict(checkpoint["model"])
    model_before.load_state_dict(checkpoint["model"])
    after_named = dict(model_after.named_parameters())
    before_named = dict(model_before.named_parameters())
    selected_names = [row["name"] for row in rows]
    maximum_checkpoint_error = 0.0
    with torch.no_grad():
        for row in rows:
            name = row["name"]
            if name not in after_named:
                raise ValueError(f"snapshot parameter is absent: {name}")
            expected_after = row["theta_after_intervention"].to(
                device=device, dtype=torch.float64)
            maximum_checkpoint_error = max(
                maximum_checkpoint_error,
                float(torch.max(torch.abs(
                    after_named[name] - expected_after)).detach().cpu()),
            )
            before_named[name].copy_(row["theta_before"].to(
                device=device, dtype=torch.float64))
    if maximum_checkpoint_error != 0.0:
        raise ValueError("E1 checkpoint does not equal snapshot post-state")

    eta = float(snapshot["learning_rate_applied"])
    directions, state_step, beta1 = _snapshot_directions(
        rows, eta, torch.device(device))
    points = _row_artifact(
        row_artifact, len(model_after.decoder.dec_layers))
    before_values, before_actions = _projected_row_chart(
        model_before, points, selected_names, directions)
    after_values, after_actions = _projected_row_chart(
        model_after,
        points,
        selected_names,
        {"preconditioned_moment_after":
             directions["preconditioned_moment_after"]},
    )
    moment_before = before_actions["preconditioned_moment_before"]
    moment_frozen_after = before_actions["preconditioned_moment_after"]
    moment_after = after_actions["preconditioned_moment_after"]
    normal_velocity = before_actions["preconditioned_clipped_gradient"]
    parameter_linear = before_actions["total_parameter_displacement"]
    decay_action = before_actions["decay_displacement"]
    intervention_defect = before_actions["intervention_displacement"]
    row_motion_defect = np.zeros_like(before_values)
    reference_forcing = normal_velocity.copy()
    moment_roundoff = (
        moment_frozen_after
        - beta1 * moment_before
        - (1.0 - beta1) * normal_velocity
    )
    normal_residual = moment_roundoff / (1.0 - beta1)
    alpha = eta / (1.0 - beta1 ** state_step)
    parameter_roundoff = (
        parameter_linear
        - decay_action
        + alpha * moment_frozen_after
        - intervention_defect
    )
    decay_defect = decay_action + parameter_roundoff
    coordinate_motion = moment_after - moment_frozen_after
    taylor_remainder = after_values - before_values - parameter_linear
    dimension = len(before_values)
    identity = np.eye(dimension)
    zero = np.zeros((dimension, dimension))
    lifted_matrix = np.block([
        [identity, -alpha * beta1 * identity],
        [zero, beta1 * identity],
    ])
    forcing = np.concatenate([
        -alpha * (1.0 - beta1) * reference_forcing,
        (1.0 - beta1) * reference_forcing,
    ])
    assembled = complete_defect_vector(
        alpha=alpha,
        beta1=beta1,
        normal_residual=normal_residual,
        decay_defect=decay_defect,
        row_motion_defect=row_motion_defect,
        intervention_defect=intervention_defect,
        taylor_remainder=taylor_remainder,
        coordinate_motion_defect=coordinate_motion,
    )
    state_before = np.concatenate([before_values, moment_before])
    state_after = np.concatenate([after_values, moment_after])
    predicted = lifted_matrix @ state_before + forcing + assembled
    if _relative_error(state_after, predicted) > 1e-10:
        raise ArithmeticError("constructed live lifted identity did not close")
    coordinate_bound = _strict_upper(
        np.linalg.norm(moment_after) + np.linalg.norm(moment_frozen_after))
    taylor_bound = _strict_upper(
        np.linalg.norm(after_values - before_values)
        + np.linalg.norm(parameter_linear))
    return {
        "alpha": alpha,
        "beta1": beta1,
        "lifted_state_before": state_before.tolist(),
        "lifted_state_after": state_after.tolist(),
        "lifted_matrix": lifted_matrix.tolist(),
        "forcing": forcing.tolist(),
        "normal_residual": normal_residual.tolist(),
        "decay_defect": decay_defect.tolist(),
        "row_motion_defect": row_motion_defect.tolist(),
        "intervention_defect": intervention_defect.tolist(),
        "taylor_remainder": taylor_remainder.tolist(),
        "coordinate_motion_defect": coordinate_motion.tolist(),
        "recorded_complete_defect": assembled.tolist(),
        "coordinate_motion_bound": coordinate_bound,
        "taylor_remainder_bound": taylor_bound,
        "intervention_observed": [0.0],
        "intervention_expected": [0.0],
    }


def measure_live_snapshot(*, snapshot_path, model_checkpoint, run_directory,
                          row_artifact, tensor_group, device,
                          certificate_output):
    certificate = construct_lifted_certificate(
        snapshot_path=snapshot_path,
        model_checkpoint=model_checkpoint,
        run_directory=run_directory,
        row_artifact=row_artifact,
        tensor_group=tensor_group,
        device=device,
    )
    Path(certificate_output).write_text(
        strict_dumps(certificate), encoding="utf-8")
    return analyze_snapshot(
        snapshot_path, certificate_output, tensor_group=tensor_group)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--certificate")
    parser.add_argument("--model-checkpoint")
    parser.add_argument("--run-directory")
    parser.add_argument("--row-artifact")
    parser.add_argument(
        "--tensor-group",
        choices=("query", "key", "value", "deductive"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--certificate-output")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.certificate:
        result = analyze_snapshot(
            args.snapshot, args.certificate, tensor_group=args.tensor_group)
    else:
        required = {
            "model_checkpoint": args.model_checkpoint,
            "run_directory": args.run_directory,
            "row_artifact": args.row_artifact,
            "tensor_group": args.tensor_group,
            "certificate_output": args.certificate_output,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            parser.error("live E1 mode is missing: " + ", ".join(missing))
        result = measure_live_snapshot(
            snapshot_path=args.snapshot,
            model_checkpoint=args.model_checkpoint,
            run_directory=args.run_directory,
            row_artifact=args.row_artifact,
            tensor_group=args.tensor_group,
            device=args.device,
            certificate_output=args.certificate_output,
        )
    Path(args.output).write_text(strict_dumps(result), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
