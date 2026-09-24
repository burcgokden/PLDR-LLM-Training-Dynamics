#!/usr/bin/env python3
"""Build raw, decision-free artifacts for Q/T/N/I/A/R analysis."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ANALYSIS = HERE.parent / "analysis"
sys.path.insert(0, str(ANALYSIS))

from confirmation_artifacts import (  # noqa: E402
    build_checkpoint_set_manifest,
    load_checkpoint_set,
    sha256_path,
    write_json_atomic,
)


def _normalized_relative(value):
    path = Path(value)
    if path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise ValueError("paths must be normalized and relative to artifact root")
    return path


def _write_npz(path, values):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **values)
    temporary.replace(destination)


def checkpoint_set(arguments):
    value = build_checkpoint_set_manifest(
        arguments.artifact_root, arguments.checkpoint)
    write_json_atomic(arguments.output, value)


def file_set(arguments):
    root = Path(arguments.artifact_root).resolve()
    files = []
    seen = set()
    for raw in arguments.file:
        relative = _normalized_relative(raw)
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError("file-set entry does not resolve inside artifact root")
        if relative in seen:
            raise ValueError("file-set entries must be unique")
        seen.add(relative)
        files.append({
            "path": relative.as_posix(),
            "sha256": sha256_path(path),
        })
    if not files:
        raise ValueError("file set cannot be empty")
    write_json_atomic(arguments.output, {
        "schema_version": "pldr-bound-file-set-v1",
        "files": files,
    })


def partition_manifest(arguments):
    def identifiers(raw, name):
        values = [value for value in raw.split(",") if value]
        if not values or len(values) != len(set(values)):
            raise ValueError(f"{name} must contain unique identifiers")
        return values

    value = {
        "schema_version": "pldr-confirmation-partitions-v2",
        "construction_ids": identifiers(
            arguments.construction_ids, "construction_ids"),
        "validation_ids": identifiers(
            arguments.validation_ids, "validation_ids"),
        "application_ids": identifiers(
            arguments.application_ids, "application_ids"),
        "dataset_sha256": sha256_path(arguments.dataset),
        "tokenizer_sha256": sha256_path(arguments.tokenizer),
    }
    write_json_atomic(arguments.output, value)


def resource_metadata(arguments):
    seconds = [
        float(value) for value in arguments.gpu_device_seconds.split(",")
        if value
    ]
    values = [
        arguments.work_units,
        arguments.peak_gpu_bytes_each,
        arguments.peak_host_bytes,
        arguments.retained_artifact_bytes,
        arguments.wall_clock_seconds,
        *seconds,
    ]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("resource metadata must be finite and nonnegative")
    write_json_atomic(arguments.output, {
        "schema_version": "pldr-resource-metadata-v2",
        "work_units": arguments.work_units,
        "gpu_device_seconds": seconds,
        "peak_gpu_bytes_each": arguments.peak_gpu_bytes_each,
        "peak_host_bytes": arguments.peak_host_bytes,
        "retained_artifact_bytes": arguments.retained_artifact_bytes,
        "wall_clock_seconds": arguments.wall_clock_seconds,
    })


def _schedule_rate(checkpoint):
    config = checkpoint["config"]
    step = int(checkpoint["step"])
    base = (
        float(config["lr"])
        * float(config.get("learning_rate_multiplier", 1.0))
    )
    warmup = max(1, int(config["warmup"]))
    if config.get("const_lr", False):
        factor = min(1.0, (step + 1) / warmup)
    else:
        total = int(checkpoint["schedule_total_steps"])
        floor = float(config.get("anneal_floor", 0.1))
        hold = int(config.get("hold_until", -1))
        if hold >= 0:
            if step <= warmup and warmup > 1:
                factor = step / warmup
            elif step <= hold:
                factor = 1.0
            else:
                decay = (min(step, total) - hold) / (total - hold)
                factor = (
                    (1 - floor) * 0.5
                    * (1 + math.cos(math.pi * decay)) + floor
                )
        elif step <= warmup:
            factor = step / warmup
        else:
            decay = (min(step, total) - warmup) / (total - warmup)
            factor = (
                (1 - floor) * 0.5
                * (1 + math.cos(math.pi * decay)) + floor
            )
    return base * factor


def training_observations(arguments):
    landmarks = load_checkpoint_set(
        arguments.landmark_manifest, arguments.artifact_root)
    replays = load_checkpoint_set(
        arguments.replay_manifest, arguments.artifact_root)
    if len(landmarks) != 8 or len(replays) != 2:
        raise ValueError("training observations need eight landmarks and two replays")
    steps = np.asarray(
        [checkpoint["step"] for checkpoint in landmarks],
        dtype=np.int64).reshape(2, 4)
    if not np.all(np.diff(steps, axis=1) > 0):
        raise ValueError("training landmarks are not ordered within each seed")
    seeds = np.asarray([
        landmarks[0]["config"]["seed"],
        landmarks[4]["config"]["seed"],
    ], dtype=np.int64)
    if len(set(seeds.tolist())) != 2:
        raise ValueError("training checkpoint sets do not contain two seeds")
    for seed_index, seed in enumerate(seeds):
        for checkpoint in landmarks[4 * seed_index:4 * seed_index + 4]:
            if int(checkpoint["config"]["seed"]) != int(seed):
                raise ValueError("landmark checkpoint is assigned to wrong seed")
    source_indices = (0, 4)
    target_indices = (3, 7)
    source_digests = np.asarray([
        landmarks[index]["state_manifest"]["model_sha256"]
        for index in source_indices
    ])
    target_digests = np.asarray([
        landmarks[index]["state_manifest"]["model_sha256"]
        for index in target_indices
    ])
    replay_digests = np.asarray([
        checkpoint["state_manifest"]["model_sha256"]
        for checkpoint in replays
    ])
    prepared_rates = np.asarray([
        checkpoint["optimizer_config"]["groups"][0]["learning_rate"]
        for checkpoint in landmarks
    ], dtype=np.float64).reshape(2, 4)
    recomputed_rates = np.asarray([
        _schedule_rate(checkpoint) for checkpoint in landmarks
    ], dtype=np.float64).reshape(2, 4)
    first = landmarks[0]["model_config"]
    model_shape = np.asarray([
        first["width"], first["layers"], first["heads"],
        landmarks[0]["schedule_total_steps"],
    ], dtype=np.int64)
    values = {
        "source_model_digests": source_digests,
        "target_model_digests": target_digests,
        "replayed_target_model_digests": replay_digests,
        "applied_learning_rates": prepared_rates,
        "recomputed_learning_rates": recomputed_rates,
        "landmark_steps": steps,
        "seed_ids": seeds,
        "data_cursors": np.asarray([
            checkpoint["data_offset_end"] for checkpoint in landmarks
        ], dtype=np.int64),
        "model_shape": model_shape,
        "optimizer_steps": np.asarray([
            checkpoint["step"] for checkpoint in landmarks
        ], dtype=np.int64),
    }
    _write_npz(arguments.output, values)


def replay_manifest(arguments):
    root = Path(arguments.artifact_root).resolve()
    files = []
    for raw in arguments.file:
        relative = _normalized_relative(raw)
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError("R manifest file does not resolve inside root")
        files.append({
            "path": relative.as_posix(),
            "sha256": sha256_path(path),
            "scientific": True,
        })
    pairs = []
    for raw in arguments.pair:
        pieces = raw.split(":", 2)
        if len(pieces) != 3 or pieces[0] not in {
            "record", "table", "figure",
        }:
            raise ValueError("rebuild pair must be KIND:FIRST:SECOND")
        first = _normalized_relative(pieces[1])
        second = _normalized_relative(pieces[2])
        for relative in (first, second):
            path = (root / relative).resolve()
            if root not in path.parents or not path.is_file():
                raise ValueError("R rebuild-pair path does not resolve")
        pairs.append({
            "kind": pieces[0],
            "first": first.as_posix(),
            "second": second.as_posix(),
        })
    write_json_atomic(arguments.output, {
        "schema_version": "pldr-r-raw-observations-v2",
        "files": files,
        "rebuild_pairs": pairs,
    })


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    checkpoint_parser = sub.add_parser("checkpoint-set")
    checkpoint_parser.add_argument("--artifact-root", required=True)
    checkpoint_parser.add_argument(
        "--checkpoint", action="append", required=True)
    checkpoint_parser.add_argument("--output", required=True)

    file_parser = sub.add_parser("file-set")
    file_parser.add_argument("--artifact-root", required=True)
    file_parser.add_argument("--file", action="append", required=True)
    file_parser.add_argument("--output", required=True)

    partition_parser = sub.add_parser("partition-manifest")
    partition_parser.add_argument("--dataset", required=True)
    partition_parser.add_argument("--tokenizer", required=True)
    partition_parser.add_argument("--construction-ids", required=True)
    partition_parser.add_argument("--validation-ids", required=True)
    partition_parser.add_argument("--application-ids", required=True)
    partition_parser.add_argument("--output", required=True)

    resource_parser = sub.add_parser("resource-metadata")
    resource_parser.add_argument("--work-units", type=float, required=True)
    resource_parser.add_argument("--gpu-device-seconds", default="")
    resource_parser.add_argument(
        "--peak-gpu-bytes-each", type=float, required=True)
    resource_parser.add_argument(
        "--peak-host-bytes", type=float, required=True)
    resource_parser.add_argument(
        "--retained-artifact-bytes", type=float, required=True)
    resource_parser.add_argument(
        "--wall-clock-seconds", type=float, required=True)
    resource_parser.add_argument("--output", required=True)

    training_parser = sub.add_parser("training-observations")
    training_parser.add_argument("--artifact-root", required=True)
    training_parser.add_argument("--landmark-manifest", required=True)
    training_parser.add_argument("--replay-manifest", required=True)
    training_parser.add_argument("--output", required=True)


    replay_parser = sub.add_parser("replay-manifest")
    replay_parser.add_argument("--artifact-root", required=True)
    replay_parser.add_argument("--file", action="append", default=[])
    replay_parser.add_argument("--pair", action="append", default=[])
    replay_parser.add_argument("--output", required=True)

    arguments = parser.parse_args()
    commands = {
        "checkpoint-set": checkpoint_set,
        "file-set": file_set,
        "partition-manifest": partition_manifest,
        "resource-metadata": resource_metadata,
        "training-observations": training_observations,
        "replay-manifest": replay_manifest,
    }
    commands[arguments.command](arguments)


if __name__ == "__main__":
    main()
