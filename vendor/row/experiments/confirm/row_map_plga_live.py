#!/usr/bin/env python3
"""Raw-checkpoint PLGA signed-power secant and margin producer."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXPERIMENTS))

from gate_shape import plga_secant_chain  # noqa: E402
from gate_shape_evidence import sha256_path, write_npz_atomic  # noqa: E402
from row_map_live import (  # noqa: E402
    SCHEMA,
    _causal_mask,
    _load_raw_checkpoint,
    _load_registry,
    _model_from_raw,
    _resource_record,
    _token_chunks,
)


def _run(model, rows, gcaches=None):
    device = next(model.parameters()).device
    inputs = rows[:, :-1].to(device)
    mask = _causal_mask(
        inputs.shape[1], device, next(model.parameters()).dtype)
    with torch.no_grad():
        return model([inputs, mask], Gcachelst=gcaches)


def produce(arguments):
    checkpoint = _load_raw_checkpoint(arguments.checkpoint)
    registry = _load_registry(arguments.registry)
    if sha256_path(arguments.tokens) != registry["tokens_sha256"]:
        raise ValueError("PLGA tokens disagree with the registry")
    candidates = [
        row for row in registry["application_pairs"]
        if row["id"] == arguments.pair_id
        and row["ownership"] == arguments.role
    ]
    if len(candidates) != 1:
        raise ValueError("PLGA capture needs one role-owned registered pair")
    pair = candidates[0]
    device = torch.device(arguments.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    model = _model_from_raw(checkpoint, device, dtype=torch.float64)
    left = _token_chunks(
        arguments.tokens, [pair["left_chunk"]],
        registry["context_length"], device)[0]
    right = _token_chunks(
        arguments.tokens, [pair["right_chunk"]],
        registry["context_length"], device)[0]
    left_result = _run(model, left)
    right_result = _run(model, right)
    layer = int(pair["layer"])
    head = int(pair["head"])
    left_attention = left_result[2]
    right_attention = right_result[2]
    left_cache = left_result[3]
    right_cache = right_result[3]
    generator_left = left_cache[layer][2][0, head].detach().cpu().numpy()
    generator_right = right_cache[layer][2][0, head].detach().cpu().numpy()
    module = model.decoder.dec_layers[layer].mha1.plgatt_layer
    weight = module.Wlst[head].detach().cpu().numpy()
    bias = module.blst[head].detach().cpu().numpy()
    powers = module.pwlst[head].detach().cpu().numpy()
    coupling = module.alst[head].detach().cpu().numpy()
    coupling_bias = module.balst[head].detach().cpu().numpy()
    replay = plga_secant_chain(
        generator_left, generator_right, weight, bias, powers, coupling,
        coupling_bias)
    if np.min(replay["metric_left"]) <= 0.0 or np.min(
            replay["metric_right"]) <= 0.0:
        raise ArithmeticError("signed real powers require positive bases")
    actual_left = left_attention[layer][4][0, head].detach().cpu().numpy()
    actual_right = right_attention[layer][4][0, head].detach().cpu().numpy()
    actual_difference = actual_right - actual_left
    curvature_residual = float(np.linalg.norm(
        actual_difference - replay["realized_difference"]))

    left_gcache = [
        [row[0].detach(), row[4].detach()] for row in left_attention]
    candidate_gcache = [list(row) for row in left_gcache]
    candidate_gcache[layer] = [
        right_attention[layer][0].detach(),
        right_attention[layer][4].detach(),
    ]
    reference_logits = _run(model, left, left_gcache)[0][0, -1]
    candidate_logits = _run(model, left, candidate_gcache)[0][0, -1]
    reference = reference_logits.detach().cpu().numpy()
    candidate = candidate_logits.detach().cpu().numpy()
    reference_centered = reference - np.mean(reference)
    candidate_centered = candidate - np.mean(candidate)
    winner = int(np.argmax(reference_centered))
    margin = float(
        reference_centered[winner]
        - np.max(np.delete(reference_centered, winner)))
    difference_norm = float(np.linalg.norm(
        candidate_centered - reference_centered))
    qualified = bool(np.sqrt(2.0) * difference_norm < margin)
    write_npz_atomic(
        arguments.output,
        schema_version=np.asarray(SCHEMA),
        stage=np.asarray("A"),
        checkpoint_sha256=np.asarray(checkpoint["content_sha256"]),
        run_name=np.asarray(checkpoint["config"]["name"]),
        seed=np.asarray(checkpoint["config"]["seed"], dtype=np.int64),
        registry_sha256=np.asarray(registry["registry_sha256"]),
        tokens_sha256=np.asarray(registry["tokens_sha256"]),
        role=np.asarray(arguments.role),
        pair_id=np.asarray(pair["id"]),
        layer=np.asarray(layer, dtype=np.int64),
        head=np.asarray(head, dtype=np.int64),
        generator_left=generator_left,
        generator_right=generator_right,
        weight=weight,
        bias=bias,
        powers=powers,
        coupling=coupling,
        coupling_bias=coupling_bias,
        metric_left=replay["metric_left"],
        metric_right=replay["metric_right"],
        activation_secant=replay["activation_secant"],
        power_secant=replay["power_secant"],
        curvature_difference=actual_difference,
        secant_prediction=replay["predicted_difference"],
        secant_identity_residual=np.asarray(replay["identity_residual"]),
        implementation_replay_residual=np.asarray(curvature_residual),
        secant_operator_bound=np.asarray(replay["operator_bound"]),
        reference_centered_logits=reference_centered,
        candidate_centered_logits=candidate_centered,
        centered_logit_difference=candidate_centered - reference_centered,
        reference_winner=np.asarray(winner, dtype=np.int64),
        candidate_winner=np.asarray(
            int(np.argmax(candidate_centered)), dtype=np.int64),
        reference_margin=np.asarray(margin),
        centered_logit_difference_norm=np.asarray(difference_norm),
        sharp_margin_qualified=np.asarray(qualified),
        signed_exponent_min=np.asarray(float(np.min(powers))),
        signed_exponent_max=np.asarray(float(np.max(powers))),
        **{
            f"resource_{name}": np.asarray(value)
            for name, value in _resource_record(device, started).items()
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument(
        "--role", choices=("construction", "validation"), required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    produce(parser.parse_args())


if __name__ == "__main__":
    main()
