#!/usr/bin/env python3
"""Checkpoint-bound exact PLGA secant and centered-logit producer."""

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

from confirmation_artifacts import load_complete_checkpoint  # noqa: E402
from gate_shape import plga_secant_chain  # noqa: E402
from gate_shape_evidence import sha256_path, write_npz_atomic  # noqa: E402
from gate_shape_live import (  # noqa: E402
    ARCHITECTURE,
    PRODUCER_SCHEMA,
    _causal_mask,
    _model_from_checkpoint,
    _resource_record,
    load_registry,
)


def _tokens(path, context_length, chunk_index):
    data = np.memmap(path, dtype=np.uint16, mode="r")
    start = int(chunk_index) * int(context_length)
    stop = start + int(context_length)
    if start < 0 or stop > len(data):
        raise ValueError("application chunk lies outside the token archive")
    return torch.from_numpy(
        np.asarray(data[start:stop], dtype=np.int64).copy())[None]


def _run(model, tokens, gcaches=None):
    device = next(model.parameters()).device
    inputs = tokens[:, :-1].to(device)
    mask = _causal_mask(
        inputs.shape[1], device, next(model.parameters()).dtype)
    with torch.no_grad():
        return model([inputs, mask], Gcachelst=gcaches)


def produce(arguments):
    checkpoint = load_complete_checkpoint(arguments.checkpoint)
    registry = load_registry(arguments.registry)
    if checkpoint["measurement_registry"] != registry:
        raise ValueError("application checkpoint does not own the registry")
    if sha256_path(arguments.tokens) != registry["dataset_sha256"]:
        raise ValueError("application tokens disagree with the registry")
    prefix = "ga-c" if arguments.registry_role == "construction" else "ga-v"
    layer = int(arguments.layer)
    head = int(arguments.head)
    if not (
        0 <= layer < ARCHITECTURE["layers"]
        and 0 <= head < ARCHITECTURE["heads"]
    ):
        raise ValueError("application layer or head lies outside the model")
    expected_pair_id = f"{prefix}{layer * ARCHITECTURE['heads'] + head:03d}"
    if arguments.pair_id != expected_pair_id:
        raise ValueError(
            "application pair does not match the registered layer/head map")
    candidates = [
        row for row in registry["application_pairs"]
        if row["id"].startswith(prefix)
        and (not arguments.pair_id or row["id"] == arguments.pair_id)
    ]
    if len(candidates) != 1:
        raise ValueError("application capture needs exactly one registered pair")
    pair = candidates[0]
    device = torch.device(arguments.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    model = _model_from_checkpoint(checkpoint, device).double()
    left_tokens = _tokens(
        arguments.tokens, registry["context_length"], pair["left_chunk"])
    right_tokens = _tokens(
        arguments.tokens, registry["context_length"], pair["right_chunk"])
    left_result = _run(model, left_tokens)
    right_result = _run(model, right_tokens)
    left_attention = left_result[2]
    right_attention = right_result[2]
    left_cache = left_result[3]
    right_cache = right_result[3]
    if not (0 <= layer < len(left_cache)):
        raise ValueError("application layer lies outside the model")
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
    actual_left = left_attention[layer][4][0, head].detach().cpu().numpy()
    actual_right = right_attention[layer][4][0, head].detach().cpu().numpy()
    curvature_residual = float(np.linalg.norm(
        actual_right - actual_left - replay["realized_difference"]))

    left_gcache = [[row[0].detach(), row[4].detach()] for row in left_attention]
    right_gcache = [[row[0].detach(), row[4].detach()] for row in left_attention]
    right_gcache[layer] = [
        right_attention[layer][0].detach(), right_attention[layer][4].detach()]
    reference_logits = _run(model, left_tokens, left_gcache)[0][0, -1]
    candidate_logits = _run(model, left_tokens, right_gcache)[0][0, -1]
    reference_numpy = reference_logits.detach().cpu().numpy()
    candidate_numpy = candidate_logits.detach().cpu().numpy()
    reference_centered = reference_numpy - np.mean(reference_numpy)
    candidate_centered = candidate_numpy - np.mean(candidate_numpy)
    reference_winner = int(np.argmax(reference_centered))
    candidate_winner = int(np.argmax(candidate_centered))
    reference_margin = float(
        reference_centered[reference_winner]
        - np.max(np.delete(reference_centered, reference_winner))
    )
    resources = _resource_record(device, started)
    write_npz_atomic(
        arguments.output,
        schema_version=np.asarray(PRODUCER_SCHEMA),
        stage=np.asarray("A"),
        producer_sha256=np.asarray(sha256_path(Path(__file__))),
        checkpoint_sha256=np.asarray(
            checkpoint["state_manifest"]["complete_state_sha256"]),
        tokens_sha256=np.asarray(sha256_path(arguments.tokens)),
        registry_sha256=np.asarray(registry["registry_sha256"]),
        registry_role=np.asarray(arguments.registry_role),
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
        activation_secant=replay["activation_secant"],
        power_secant=replay["power_secant"],
        curvature_difference=actual_right - actual_left,
        secant_prediction=replay["predicted_difference"],
        curvature_replay_residual=np.asarray(curvature_residual),
        reference_logits=reference_numpy,
        candidate_logits=candidate_numpy,
        reference_centered_logits=reference_centered,
        candidate_centered_logits=candidate_centered,
        centered_logit_difference=candidate_centered - reference_centered,
        reference_winner=np.asarray(reference_winner, dtype=np.int64),
        candidate_winner=np.asarray(candidate_winner, dtype=np.int64),
        reference_margin=np.asarray(reference_margin),
        centered_logit_difference_norm=np.asarray(
            np.linalg.norm(candidate_centered - reference_centered)),
        **{
            f"resource_{name}": np.asarray(value)
            for name, value in resources.items()
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument(
        "--registry-role", choices=("construction", "validation"), required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--head", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    produce(parser.parse_args())


if __name__ == "__main__":
    main()
