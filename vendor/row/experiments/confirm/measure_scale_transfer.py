#!/usr/bin/env python3
"""Live held-out scale-transfer, observable-bridge, and RG estimator for E8."""

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


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXPERIMENTS))

import campaign_design  # noqa: E402
from campaign_record import strict_dumps  # noqa: E402
from direct_certificate import deductive_output_order_bridge  # noqa: E402
from measure_intervention import measure_avalanches  # noqa: E402
from measure_rg_closure import measure as measure_rg  # noqa: E402
from measure_tilt import build_model  # noqa: E402
from train_run import PROBE_RESERVE_CHUNKS, create_masks  # noqa: E402


ARMS = tuple(campaign_design.E8["model_arms"])


def _load(value):
    if isinstance(value, dict):
        result = value
    else:
        with Path(value).open("r", encoding="utf-8") as stream:
            result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError("E8 input artifact must be a JSON object")
    return result


def _finite(value, name, *, nonnegative=False):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _measurement_values(value):
    rows = value.get("measurements")
    if not isinstance(rows, list):
        raise ValueError("measurement artifact has no measurement array")
    result = {row["name"]: _finite(row["value"], row["name"])
              for row in rows}
    if len(result) != len(rows):
        raise ValueError("measurement artifact contains duplicate names")
    return result


def _measurement(name, value, unit, method):
    return {
        "name": name,
        "value": _finite(value, name),
        "unit": unit,
        "method": method,
        "status": "OBSERVED",
        "reason_code": None,
    }


def _run_config(run_directory):
    with (Path(run_directory) / "log.jsonl").open(
        "r", encoding="utf-8",
    ) as stream:
        value = json.loads(stream.readline())
    if value.get("event") != "config":
        raise ValueError("run log does not begin with a config record")
    return value


def _log_rows(run_directory):
    with (Path(run_directory) / "log.jsonl").open(
        "r", encoding="utf-8",
    ) as stream:
        return [json.loads(line) for line in stream]


def _terminal_loss(run_directory, checkpoint):
    rows = [
        row for row in _log_rows(run_directory)
        if int(row.get("step", -1)) == int(checkpoint)
        and "val_loss" in row
    ]
    if len(rows) != 1:
        raise ValueError("E8 run must contain one terminal validation loss")
    return _finite(rows[0]["val_loss"], "heldout_loss", nonnegative=True)


def _inference_batch(config, token_path, prompt_offset, device):
    tokens = np.memmap(token_path, dtype=np.uint16, mode="r")
    context = int(config["ctx"])
    chunks = tokens[:len(tokens) // context * context].reshape(-1, context)
    probe_base = len(chunks) - PROBE_RESERVE_CHUNKS
    starts = [probe_base + int(prompt_offset), probe_base + int(prompt_offset) + 37]
    if starts[-1] >= len(chunks):
        raise ValueError("E8 prompt offset lies outside the reserved region")
    inputs = torch.from_numpy(
        np.asarray(chunks[starts], dtype=np.int64)).to(device)
    return inputs, create_masks(inputs, device)


def _rotated_queries(layer, hidden):
    batch = hidden.shape[0]
    query = layer.mha1.split_heads(layer.mha1.wq(hidden), batch)
    return layer.mha1.rotary_embedding(query)


def _apply_deductive_map(layer, density, epsilon):
    normalizer = layer.mha1.layernorm1
    normalized = F.layer_norm(
        density,
        (density.shape[-1],),
        normalizer.weight,
        normalizer.bias,
        float(epsilon),
    )
    result = normalized
    for unit in layer.mha1.reslayerAs:
        result = unit([result])
    return normalized, result


def _block_tensor(value, block_size):
    length = value.shape[1]
    if length % block_size:
        raise ValueError("tensor sequence is not divisible by block size")
    return value.reshape(
        value.shape[0], length // block_size, block_size, *value.shape[2:]
    ).mean(dim=2)


def _tensor_norm(value):
    return float(torch.linalg.vector_norm(value.detach().double()).cpu())


def _downstream_lipschitz_bound(model):
    log_bound = 0.0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            norm = float(torch.linalg.matrix_norm(
                module.weight.detach().double(), ord="fro").cpu())
            log_bound += math.log(max(norm, 1.0))
        elif isinstance(module, torch.nn.LayerNorm):
            gamma = (
                float(module.weight.detach().abs().max().cpu())
                if module.elementwise_affine else 1.0
            )
            log_bound += math.log(max(
                gamma / math.sqrt(float(module.eps)), 1.0))
    return math.exp(min(log_bound, math.log(1e150)))


def collect_live_tensors(*, run_directory, checkpoint, token_path,
                         device, prompt_offset, block_size,
                         direct_row_map_upper):
    config = _run_config(run_directory)
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if int(payload["step"]) != int(campaign_design.E8["checkpoint"]):
        raise ValueError("E8 model checkpoint is not at the registered step")
    model = build_model(config, device)
    model.load_state_dict(payload["model"])
    model = model.to(device).eval()
    inputs, mask = _inference_batch(
        config, token_path, prompt_offset, device)
    with torch.no_grad():
        logits, _, attention, _ = model([inputs, mask])
        hidden = model.decoder.layernorm1(
            model.decoder.embedding(inputs)
            * math.sqrt(float(model.d_model))
        )
        layer = model.decoder.dec_layers[0]
        rotated = _rotated_queries(layer, hidden)
        query = rotated.permute(0, 2, 1, 3)
        fine_density = torch.matmul(
            query.transpose(-1, -2), query)
        block_query = _block_tensor(rotated, int(block_size))
        coarse_query = block_query.permute(0, 2, 1, 3)
        coarse_density = torch.matmul(
            coarse_query.transpose(-1, -2), coarse_query)
        epsilon = float(layer.mha1.layernorm1.eps)
        fine_input, fine_deductive = _apply_deductive_map(
            layer, fine_density, epsilon)
        coarse_input, coarse_deductive = _apply_deductive_map(
            layer, coarse_density, epsilon / int(block_size) ** 2)

        fine_output, fine_attention, _ = layer([hidden[:1], mask[:1]])
        coarse_hidden = _block_tensor(hidden[:1], int(block_size))
        coarse_tokens = torch.ones(
            (1, coarse_hidden.shape[1]),
            dtype=torch.long, device=device)
        coarse_mask = create_masks(coarse_tokens, device)
        coarse_output, coarse_attention, _ = layer(
            [coarse_hidden, coarse_mask])
        fine_attention_output, _, _ = layer.mha1(
            [hidden[:1], hidden[:1], hidden[:1], mask[:1]])
        coarse_attention_output, _, _ = layer.mha1([
            coarse_hidden, coarse_hidden, coarse_hidden, coarse_mask,
        ])

    observed_deductive = _tensor_norm(
        fine_deductive[0, 0] - coarse_deductive[0, 0])
    blocked_fine_output = _block_tensor(
        fine_output, int(block_size))
    observed_full = _tensor_norm(
        blocked_fine_output - coarse_output
    ) / math.sqrt(float(coarse_output.shape[1]))
    blocked_fine_query = _block_tensor(
        rotated[:1, :, :1, :], int(block_size))
    coarse_rotated = _rotated_queries(layer, coarse_hidden)[:, :, :1, :]
    projection_bound = (
        _tensor_norm(blocked_fine_query) + _tensor_norm(coarse_rotated)
    )
    score_bound = (
        _tensor_norm(fine_attention[5])
        + _tensor_norm(coarse_attention[5])
    )
    value_bound = (
        _tensor_norm(_block_tensor(
            fine_attention_output, int(block_size)))
        + _tensor_norm(coarse_attention_output)
    )
    residual_bound = (
        _tensor_norm(blocked_fine_output) + _tensor_norm(coarse_output)
    )
    gamma = layer.mha1.layernorm1.weight.detach().cpu().double().numpy()
    beta = layer.mha1.layernorm1.bias.detach().cpu().double().numpy()
    layernorm_lipschitz = (
        float(np.max(np.abs(gamma)))
        / math.sqrt(epsilon / int(block_size) ** 2)
    )
    attention_logits = (
        fine_attention[5][0, 0, -1].detach().cpu().double().numpy()
    )
    rg_input = {
        "rotated_queries":
            rotated[0, :, 0].detach().cpu().double().numpy().tolist(),
        "block_size": int(block_size),
        "repeat_block_size": 2,
        "metric_layernorm_epsilon": epsilon,
        "metric_layernorm_gamma": gamma.tolist(),
        "metric_layernorm_beta": beta.tolist(),
        "layernorm_lipschitz": layernorm_lipschitz,
        "direct_row_map_upper": float(direct_row_map_upper),
        "deductive_downstream_lipschitz": 1.0,
        "observed_deductive_difference": observed_deductive,
        "attention_logits": attention_logits.tolist(),
        "full_layer_other_defect_bounds": {
            "input_projection_rope": projection_bound,
            "scores": score_bound,
            "mask_softmax": 2.0,
            "values_output": value_bound,
            "residual_ffn": residual_bound,
        },
        # Other-stage defects above are already measured in their final
        # comparison norm, so their transport multipliers are one.
        "full_layer_stage_lipschitz": {
            "input_projection_rope": 1.0,
            "deductive_map": 1.0,
            "scores": 1.0,
            "mask_softmax": 1.0,
            "values_output": 1.0,
            "residual_ffn": 1.0,
        },
        "observed_full_layer_difference": observed_full,
    }
    rg = measure_rg(rg_input)

    row_a = fine_input[0, 0, 0].detach().cpu().double().numpy()
    other_query = rotated[1].permute(1, 0, 2)
    other_density = torch.matmul(
        other_query.transpose(-1, -2), other_query)
    other_input, _ = _apply_deductive_map(
        layer, other_density[None], epsilon)
    row_b = other_input[0, 0, 0].detach().cpu().double().numpy()
    downstream = _downstream_lipschitz_bound(model)
    bridge = deductive_output_order_bridge(
        output_a=logits[0:1, -1].detach().cpu().double().numpy(),
        output_b=logits[1:2, -1].detach().cpu().double().numpy(),
        rows_a=row_a[None],
        rows_b=row_b[None],
        direct_row_map_upper=float(direct_row_map_upper),
        downstream_lipschitz=downstream,
    )
    return {
        "schema_version": "pldr-live-e8-tensors-v1",
        "block_size": int(block_size),
        "prompt_offset": int(prompt_offset),
        "layer_index": 0,
        "head_index": 0,
        "rg_input": rg_input,
        "rg_measurement": rg,
        "bridge": bridge,
    }


def _feature_vector(row, arm):
    if arm == "constant_rate":
        return [1.0, float(row["maximum_learning_rate"])]
    if arm == "loss_only":
        return [1.0, float(row["heldout_loss"])]
    if arm == "unconstrained_trend":
        return [
            1.0,
            float(row["model_width"]),
            float(row["model_depth"]),
            float(row["maximum_learning_rate"]),
            float(row["warmup_steps"]),
            float(row["anneal_floor"]),
            float(row["heldout_loss"]),
        ]
    raise ValueError("theory has no fitted feature vector")


def fit_baseline_models(training_rows):
    if not isinstance(training_rows, list) or not training_rows:
        raise ValueError("E8 training partition must be nonempty")
    targets = np.asarray([
        _finite(row["observed_entry_step"], "observed_entry_step")
        for row in training_rows
    ])
    coefficients = {}
    for arm in ARMS:
        if arm == "theory":
            continue
        matrix = np.asarray([
            _feature_vector(row, arm) for row in training_rows
        ], dtype=float)
        coefficient, _, _, _ = np.linalg.lstsq(
            matrix, targets, rcond=None)
        coefficients[arm] = coefficient.tolist()
    payload = {
        "schema_version": "pldr-e8-frozen-baselines-v1",
        "split_sha256": campaign_design.E8["split"]["sha256"],
        "training_seeds": campaign_design.E8["split"]["training_seeds"],
        "fit_order": campaign_design.E8["baseline_fit_order"],
        "coefficients": coefficients,
        "training_row_count": len(training_rows),
    }
    payload["fit_sha256"] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    return payload


def model_prediction(fit, physical, arm):
    if arm == "theory":
        return _finite(
            physical["predicted_entry_step"], "predicted_entry_step")
    fit = _load(fit)
    if fit.get("split_sha256") != campaign_design.E8["split"]["sha256"]:
        raise ValueError("E8 baseline fit has the wrong held-out split")
    coefficient = np.asarray(fit["coefficients"][arm], dtype=float)
    features = np.asarray(_feature_vector(physical, arm), dtype=float)
    return float(features @ coefficient)


def physical_summary(*, run_directory, e0_measurement, e6_measurement,
                     architecture_cell, schedule_cell):
    if architecture_cell not in campaign_design.ARCHITECTURES:
        raise ValueError("unknown E8 architecture cell")
    if schedule_cell not in campaign_design.SCHEDULES:
        raise ValueError("unknown E8 schedule cell")
    architecture = campaign_design.ARCHITECTURES[architecture_cell]
    schedule = campaign_design.SCHEDULES[schedule_cell]
    e6 = _measurement_values(_load(e6_measurement))
    return {
        "model_width": architecture["model_width"],
        "model_depth": architecture["model_depth"],
        "maximum_learning_rate": schedule["maximum_learning_rate"],
        "warmup_steps": schedule["warmup_steps"],
        "anneal_floor": schedule["anneal_floor"],
        "heldout_loss": _terminal_loss(
            run_directory, campaign_design.E8["checkpoint"]),
        "predicted_entry_step": e6["predicted_entry_step"],
        "observed_entry_step": e6["observed_entry_step"],
        "direct_row_map_upper": _measurement_values(
            _load(e0_measurement))["direct_row_map_upper"],
    }


def measure_live_e8(*, run_directory, checkpoint, token_path, device,
                    e0_measurement, e6_measurement, fit,
                    architecture_cell, schedule_cell, block_size, arm,
                    tensor_measurement=None):
    if arm not in ARMS:
        raise ValueError("unknown E8 model arm")
    physical = physical_summary(
        run_directory=run_directory,
        e0_measurement=e0_measurement,
        e6_measurement=e6_measurement,
        architecture_cell=architecture_cell,
        schedule_cell=schedule_cell,
    )
    if tensor_measurement is None:
        tensor = collect_live_tensors(
            run_directory=run_directory,
            checkpoint=checkpoint,
            token_path=token_path,
            device=device,
            prompt_offset=campaign_design.E8["prompt_offset"],
            block_size=int(block_size),
            direct_row_map_upper=physical["direct_row_map_upper"],
        )
    else:
        tensor = _load(tensor_measurement)
        if tensor.get("schema_version") != "pldr-live-e8-tensors-v1":
            raise ValueError("cached E8 tensor measurement schema is stale")
        if int(tensor.get("block_size", -1)) != int(block_size):
            raise ValueError("cached E8 tensor block size disagrees")
    prediction = model_prediction(fit, physical, arm)
    observed = physical["observed_entry_step"]
    model_score = abs(prediction - observed) / max(abs(observed), 1.0)
    e0 = _measurement_values(_load(e0_measurement))
    theorem_ratio = (
        0.0 if physical["direct_row_map_upper"] == 0.0
        else e0["grid_jacobian_upper"] / physical["direct_row_map_upper"]
    )
    bridge = tensor["bridge"]
    rg_values = _measurement_values(tensor["rg_measurement"])
    checkpoint_step = int(campaign_design.E8["checkpoint"])
    avalanche = measure_avalanches(
        source_run_directory=run_directory,
        branch_run_directory=run_directory,
        source_step=checkpoint_step
            - int(campaign_design.E7_INTERVENTIONS[
                "avalanche_window_updates"]),
        outcome_step=checkpoint_step,
    )
    sizes = [row["size"] for row in avalanche["events"]]
    tail = (
        float(np.max(sizes) / max(np.mean(sizes), 1e-30))
        if sizes else 0.0
    )
    scaling_error = abs(
        math.log1p(avalanche["mean_size"])
        / math.log(max(float(physical["model_width"]), 2.0))
        - math.log1p(avalanche["mean_duration"])
        / math.log(max(float(physical["model_depth"]) + 1.0, 2.0))
    )
    values = {
        **physical,
        "criterion": float(campaign_design.SCALAR_DIAGNOSTIC_RULES[
            "finite_entry"]["criterion"]),
        "baseline_indicator": 0.0 if arm == "theory" else 1.0,
        "model_score": model_score,
        "theorem_bound_ratio": theorem_ratio,
        "source_order_parameter": bridge["source_order_parameter"],
        "source_rmse": bridge["source_rmse"],
        "tensor_mean_abs": bridge["tensor_mean_abs"],
        "row_input_spread": bridge["row_input_spread"],
        "downstream_lipschitz": bridge["downstream_lipschitz"],
        "order_parameter_upper": bridge["order_parameter_upper"],
        "order_parameter_bridge_ratio":
            bridge["order_parameter_bridge_ratio"],
        "avalanche_event_rate": avalanche["event_rate"],
        "avalanche_mean_size": avalanche["mean_size"],
        "avalanche_mean_duration": avalanche["mean_duration"],
        "avalanche_mean_support": avalanche["mean_support"],
        "avalanche_tail_comparison": tail,
        "finite_size_scaling_error": scaling_error,
        **rg_values,
    }
    units = {
        name: "dimensionless" for name in values
    }
    units.update({
        "model_width": "features",
        "model_depth": "decoder_layers",
        "direct_row_map_upper": "output_row_per_input_row",
        "criterion": "output_row_per_input_row",
        "predicted_entry_step": "global_optimizer_step",
        "observed_entry_step": "global_optimizer_step",
        "heldout_loss": "nats_per_token",
        "maximum_learning_rate": "parameter_step",
        "warmup_steps": "global_optimizer_step",
        "source_rmse": "logit_unit",
        "tensor_mean_abs": "logit_unit",
        "row_input_spread": "physical_row_unit",
        "downstream_lipschitz": "logit_per_deductive_row",
        "order_parameter_upper": "dimensionless",
        "avalanche_event_rate": "events_per_update",
        "avalanche_mean_size": "threshold_normalized_gradient_sum",
        "avalanche_mean_duration": "optimizer_updates",
        "avalanche_mean_support": "parameter_blocks",
        "rg_block_size": "fine_tokens_per_coarse_token",
        "rg_within_covariance_trace": "query_squared",
        "rg_deductive_closure_upper": "deductive_tensor_frobenius",
        "rg_full_layer_closure_upper": "layer_output_frobenius",
    })
    measurements = [
        _measurement(
            name, values[name], units[name],
            "live_heldout_scale_bridge_rg_and_frozen_model_score",
        )
        for name in (
            "model_width", "model_depth", "direct_row_map_upper",
            "criterion", "predicted_entry_step", "observed_entry_step",
            "heldout_loss", "baseline_indicator", "model_score",
            "theorem_bound_ratio", "maximum_learning_rate",
            "warmup_steps", "anneal_floor", "source_order_parameter",
            "source_rmse", "tensor_mean_abs", "row_input_spread",
            "downstream_lipschitz", "order_parameter_upper",
            "order_parameter_bridge_ratio", "avalanche_event_rate",
            "avalanche_mean_size", "avalanche_mean_duration",
            "avalanche_mean_support", "avalanche_tail_comparison",
            "finite_size_scaling_error", "rg_block_size",
            "rg_density_identity_residual",
            "rg_layernorm_scale_residual",
            "rg_within_covariance_trace",
            "rg_deductive_closure_upper", "rg_deductive_closure_ratio",
            "rg_softmax_aggregation_residual",
            "rg_repeated_block_semigroup_residual",
            "rg_full_layer_closure_upper", "rg_full_layer_closure_ratio",
        )
    ]
    return {
        "schema_version": "pldr-live-e8-measurement-v1",
        "protocol_id": "E8",
        "architecture_cell": architecture_cell,
        "schedule_cell": schedule_cell,
        "block_size": int(block_size),
        "arm": arm,
        "model_prediction": prediction,
        "fit_sha256": _load(fit)["fit_sha256"],
        "tensor_measurement": tensor,
        "avalanche": avalanche,
        "measurements": measurements,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with Path(args.input).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    result = measure_live_e8(**value)
    Path(args.output).write_text(strict_dumps(result), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
