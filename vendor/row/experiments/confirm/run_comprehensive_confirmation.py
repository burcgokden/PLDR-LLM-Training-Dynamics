#!/usr/bin/env python3
"""Plan, seal, and lock the comprehensive collapse confirmation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shlex
import sys


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ANALYSIS = ROOT / "experiments" / "analysis"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ANALYSIS))

from comprehensive_evidence import (  # noqa: E402
    load_request,
    seal_request,
)
from comprehensive_protocol_specs import (  # noqa: E402
    ARCHITECTURE,
    PREDICTIONS,
    RAW_REQUIRED,
    RESOURCE_CAPS,
    STAGES,
)
from confirmation_artifacts import (  # noqa: E402
    digest_object,
    load_complete_checkpoint,
    sha256_path,
    write_json_atomic,
)


PROTOCOL_ROOT = (
    ROOT / "experiments" / "protocols" / "comprehensive_collapse_confirmation")


def _training_command(
        *, name, run_id, seed, device, tokens, tokenizer, registry,
        output_root):
    snapshots = {
        step
        for window in ARCHITECTURE["optimizer_snapshot_windows"]
        for step in window
    }
    landmarks = {256, 768, 3072, 5632, 6144, *snapshots}
    values = [
        "python3", "experiments/train_run.py",
        "--name", name,
        "--run-id", run_id,
        "--lineage-root", run_id,
        "--lr", "0.0003",
        "--warmup", "512",
        "--steps", str(ARCHITECTURE["updates"]),
        "--schedule_total_steps", str(ARCHITECTURE["updates"]),
        "--batch", str(ARCHITECTURE["batch_size"]),
        "--ctx", str(ARCHITECTURE["context_length"]),
        "--layers", str(ARCHITECTURE["layers"]),
        "--heads", str(ARCHITECTURE["heads"]),
        "--dk", str(ARCHITECTURE["head_width"]),
        "--adff", "48",
        "--seed", str(seed),
        "--device", device,
        "--tokens", str(tokens),
        "--tok_model", str(tokenizer),
        "--confirmation_registry", str(registry),
        "--outdir", str(output_root),
        "--ckpt_steps", ",".join(map(str, sorted(landmarks))),
        "--optimizer_ckpt_steps", ",".join(map(str, sorted(landmarks))),
        "--confirmation_snapshot_steps", ",".join(map(str, sorted(snapshots))),
        "--confirmation_snapshot_blocks",
        "phi,plga,query,key,value,attn,ffn",
        "--probe_region", "global",
        "--skip_generation",
        "--terminal_validation",
    ]
    return " ".join(shlex.quote(value) for value in values)


def plan():
    return {
        "schema_version": "pldr-comprehensive-execution-plan-v4",
        "stage_order": [stage["id"] for stage in STAGES],
        "stages": list(STAGES),
        "architecture": ARCHITECTURE,
        "resource_caps": RESOURCE_CAPS,
        "predictions": list(PREDICTIONS),
        "launch_gate": {
            "qualification_cuda0": "SEALED_Q_REQUIRED",
            "qualification_cuda1": "SEALED_Q_REQUIRED",
            "normal_engineering": "CONTEXT_256_BELOW_20_GIB_REQUIRED",
            "prediction_lock": "SEALED_CONSTRUCTION_LOCK_REQUIRED",
        },
    }


def training_commands(arguments):
    devices = [value for value in arguments.devices.split(",") if value]
    if len(devices) != 2:
        raise ValueError("the registered program needs exactly two devices")
    commands = []
    for index, (seed, device) in enumerate(zip(
            ARCHITECTURE["trajectory_seeds"], devices)):
        name = f"rev34-seed-{seed}"
        commands.append({
            "role": "primary" if index == 0 else "matched_replication",
            "run_id": name,
            "command": _training_command(
                name=name, run_id=name, seed=seed, device=device,
                tokens=arguments.tokens, tokenizer=arguments.tokenizer,
                registry=arguments.registry, output_root=arguments.output_root),
        })
    replay_name = f"rev34-seed-{ARCHITECTURE['trajectory_seeds'][0]}-replay"
    commands.append({
        "role": "same_device_from_scratch_replay",
        "run_id": replay_name,
        "command": _training_command(
            name=replay_name, run_id=replay_name,
            seed=ARCHITECTURE["trajectory_seeds"][0], device=devices[0],
            tokens=arguments.tokens, tokenizer=arguments.tokenizer,
            registry=arguments.registry,
            output_root=Path(arguments.output_root).with_name(
                Path(arguments.output_root).name + "-replay")),
    })
    value = {
        "schema_version": "pldr-comprehensive-training-commands-v4",
        "commands": commands,
        "replay_semantics": (
            "same device, from scratch, distinct run identity and lineage; "
            "scientific tensors, clocks, masks, registry, next minibatch, and RNG compare"
        ),
    }
    write_json_atomic(arguments.output, value)


def _rate_continuation_command(
        *, source, source_path, name, multiplier, device, tokens, tokenizer,
        registry, output_root):
    config = source["config"]
    model = source["model_config"]
    source_step = int(source["step"])
    nodes = list(range(source_step + 1, source_step + 17))
    values = [
        "python3", "experiments/train_run.py",
        "--name", name,
        "--run-id", name,
        "--lr", str(float(config["lr"])),
        "--learning_rate_multiplier", str(float(multiplier)),
        "--warmup", str(int(config["warmup"])),
        "--steps", "16",
        "--schedule_total_steps", str(int(source["schedule_total_steps"])),
        "--batch", str(int(config["batch"])),
        "--ctx", str(int(config["ctx"])),
        "--layers", str(int(model["layers"])),
        "--heads", str(int(model["heads"])),
        "--dk", str(int(model["head_width"])),
        "--adff", str(int(model["row_hidden_width"])),
        "--seed", str(int(config["seed"])),
        "--device", str(device),
        "--tokens", str(tokens),
        "--tok_model", str(tokenizer),
        "--outdir", str(output_root),
        "--init_from", str(source_path),
        "--confirmation_registry", str(registry),
        "--ckpt_steps", ",".join(map(str, nodes)),
        "--optimizer_ckpt_steps", ",".join(map(str, nodes)),
        "--probe_region", str(config.get("probe_region", "global")),
        "--optimizer", str(config.get("optimizer", "adamw")),
        "--wd", str(float(config.get("wd", 0.1))),
        "--clip", str(float(config.get("clip", 1.0))),
        "--loss_mode", str(config.get("loss_mode", "block")),
        "--accum", str(int(config.get("accum", 1))),
        "--anneal_floor", str(float(config.get("anneal_floor", 0.1))),
        "--hold_until", str(int(config.get("hold_until", -1))),
        "--skip_generation",
        "--terminal_validation",
    ]
    if bool(config.get("const_lr", False)):
        values.append("--const_lr")
    return " ".join(shlex.quote(value) for value in values)


def intervention_commands(arguments):
    source_path = Path(arguments.source_checkpoint).resolve()
    source = load_complete_checkpoint(source_path)
    tokens = Path(arguments.tokens).resolve()
    tokenizer = Path(arguments.tokenizer).resolve()
    registry_path = Path(arguments.registry).resolve()
    with registry_path.open("r", encoding="utf-8") as stream:
        registry = json.load(stream)
    if source["measurement_registry"] != registry:
        raise ValueError("intervention source checkpoint and registry disagree")
    if (
        sha256_path(tokens) != registry.get("dataset_sha256")
        or sha256_path(tokenizer) != registry.get("tokenizer_sha256")
    ):
        raise ValueError("intervention token inputs disagree with the registry")
    model = source["model_config"]
    expected_shape = (
        ARCHITECTURE["width"], ARCHITECTURE["layers"],
        ARCHITECTURE["heads"], ARCHITECTURE["head_width"],
        ARCHITECTURE["context_length"], ARCHITECTURE["batch_size"],
    )
    actual_shape = (
        int(model["width"]), int(model["layers"]), int(model["heads"]),
        int(model["head_width"]), int(source["config"]["ctx"]),
        int(source["config"]["batch"]),
    )
    if actual_shape != expected_shape:
        raise ValueError("intervention source is outside the frozen architecture")
    if (
        source["optimizer_config"].get("name") != "adamw"
        or int(source["schedule_total_steps"]) != ARCHITECTURE["updates"]
        or int(source["step"]) + 16 > int(source["schedule_total_steps"])
    ):
        raise ValueError("intervention source lacks a sixteen-step AdamW horizon")
    forbidden = {
        "dag": "", "sdpa": False, "freeze_g_at": -1,
        "freeze_v_at": -1, "freeze_plga_at": -1,
        "freeze_attn_at": -1, "freeze_ffn_at": -1,
        "pulse_at": -1, "pulse_train": "", "mode_block": "",
    }
    changed = {
        name: source["config"].get(name, expected)
        for name, expected in forbidden.items()
        if source["config"].get(name, expected) != expected
    }
    if changed:
        raise ValueError(
            "intervention source carries another intervention: "
            + ", ".join(sorted(changed)))
    multipliers = [
        float(value) for value in arguments.rate_multipliers.split(",")
        if value]
    if (
        len(multipliers) != 2
        or not all(math.isfinite(value) and value > 0 for value in multipliers)
        or not multipliers[0] < multipliers[1]
    ):
        raise ValueError("rate multipliers must be two increasing positive values")
    commands = []
    for role, multiplier in zip(("low_rate", "high_rate"), multipliers):
        label = format(multiplier, ".6g").replace(".", "p")
        name = f"rev34-{role}-s{source['step']}-x{label}"
        branch_root = Path(arguments.output_root).resolve() / name
        checkpoint_steps = list(range(
            int(source["step"]) + 1, int(source["step"]) + 17))
        commands.append({
            "role": role,
            "run_id": name,
            "rate_multiplier": multiplier,
            "run_directory": str(branch_root),
            "event_ledger": str(branch_root / "log.jsonl"),
            "checkpoint_steps": checkpoint_steps,
            "checkpoint_paths": [
                str(branch_root / f"ckpt_{step}.pt")
                for step in checkpoint_steps],
            "command": _rate_continuation_command(
                source=source, source_path=source_path, name=name,
                multiplier=multiplier, device=arguments.device,
                tokens=tokens, tokenizer=tokenizer, registry=registry_path,
                output_root=Path(arguments.output_root).resolve()),
        })
    write_json_atomic(arguments.output, {
        "schema_version": "pldr-comprehensive-intervention-commands-v4",
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": sha256_path(source_path),
        "source_complete_state_sha256": source["state_manifest"][
            "complete_state_sha256"],
        "source_step": int(source["step"]),
        "tokens_sha256": sha256_path(tokens),
        "tokenizer_sha256": sha256_path(tokenizer),
        "registry_sha256": registry["registry_sha256"],
        "commands": commands,
        "matching_rule": (
            "same complete source, RNG, minibatches, optimizer moments, "
            "schedule phase, and non-rate configuration; only the registered "
            "learning-rate multiplier and run identity differ"
        ),
    })


def seal_stage_request(arguments):
    roles = [value.partition("=")[0] for value in arguments.artifact]
    expected = set(RAW_REQUIRED[arguments.stage])
    if len(roles) != len(set(roles)) or set(roles) != expected:
        raise ValueError(
            f"{arguments.stage} artifact roles must be: "
            + ", ".join(sorted(expected)))
    protocol = PROTOCOL_ROOT / f"{arguments.stage.lower()}_protocol.json"
    request = seal_request(
        arguments.stage, arguments.campaign_id, arguments.artifact_root,
        arguments.artifact, protocol)
    write_json_atomic(arguments.output, request)


def seal_construction_request(arguments):
    protocol = PROTOCOL_ROOT / "construction_protocol.json"
    request = seal_request(
        "CONSTRUCTION", arguments.campaign_id, arguments.artifact_root,
        [f"construction_summary={arguments.summary}"], protocol)
    write_json_atomic(arguments.output, request)


def lock_predictions(arguments):
    summaries = []
    request_digests = []
    protocol_digests = set()
    partition_digests = set()
    code_digests = set()
    for raw_request in arguments.request:
        request, resolved, _protocol = load_request(
            Path(arguments.artifact_root) / raw_request)
        if request["campaign_id"] != arguments.campaign_id:
            raise ValueError("construction requests belong to another campaign")
        if request["stage"] != "CONSTRUCTION" or set(resolved) != {
                "construction_summary"}:
            raise ValueError(
                "prediction locking needs sealed construction-summary evidence")
        with resolved["construction_summary"].open("r", encoding="utf-8") as stream:
            summary = json.load(stream)
        required = {
            "schema_version", "code_sha256", "partition_sha256", "schedule",
            "selected_edges", "intervals", "path_metrics", "radii",
            "direct_conversion", "application",
        }
        if not isinstance(summary, dict) or set(summary) != required:
            raise ValueError("construction summary has an invalid closed schema")
        summaries.append(summary)
        request_digests.append(request["request_sha256"])
        protocol_digests.add(request["protocol"]["sha256"])
        partition_digests.add(summary["partition_sha256"])
        code_digests.add(summary["code_sha256"])
    if not summaries:
        raise ValueError("prediction lock needs construction evidence")
    if len(partition_digests) != 1 or len(code_digests) != 1:
        raise ValueError("construction summaries disagree on code or partition")
    lock = {
        "schema_version": "pldr-prediction-lock-v4",
        "campaign_id": arguments.campaign_id,
        "code_sha256": next(iter(code_digests)),
        "protocol_sha256": digest_object(sorted(protocol_digests)),
        "partition_sha256": next(iter(partition_digests)),
        "construction_request_sha256": sorted(request_digests),
        "schedule": [summary["schedule"] for summary in summaries],
        "selected_edges": [summary["selected_edges"] for summary in summaries],
        "intervals": [summary["intervals"] for summary in summaries],
        "path_metrics": [summary["path_metrics"] for summary in summaries],
        "radii": [summary["radii"] for summary in summaries],
        "direct_conversion": [
            summary["direct_conversion"] for summary in summaries],
        "application": [summary["application"] for summary in summaries],
    }
    lock["lock_sha256"] = digest_object(lock)
    write_json_atomic(arguments.output, lock)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--output", required=True)

    training = subparsers.add_parser("training-commands")
    training.add_argument("--tokens", required=True)
    training.add_argument("--tokenizer", required=True)
    training.add_argument("--registry", required=True)
    training.add_argument("--output-root", required=True)
    training.add_argument("--devices", default="cuda:0,cuda:1")
    training.add_argument("--output", required=True)

    intervention = subparsers.add_parser("intervention-commands")
    intervention.add_argument("--source-checkpoint", required=True)
    intervention.add_argument("--tokens", required=True)
    intervention.add_argument("--tokenizer", required=True)
    intervention.add_argument("--registry", required=True)
    intervention.add_argument("--output-root", required=True)
    intervention.add_argument("--device", default="cuda:0")
    intervention.add_argument("--rate-multipliers", default="0.75,1.25")
    intervention.add_argument("--output", required=True)

    seal = subparsers.add_parser("seal-request")
    seal.add_argument("--stage", choices=[stage["id"] for stage in STAGES],
                      required=True)
    seal.add_argument("--campaign-id", required=True)
    seal.add_argument("--artifact-root", required=True)
    seal.add_argument("--artifact", action="append", required=True)
    seal.add_argument("--output", required=True)

    construction = subparsers.add_parser("seal-construction")
    construction.add_argument("--campaign-id", required=True)
    construction.add_argument("--artifact-root", required=True)
    construction.add_argument("--summary", required=True)
    construction.add_argument("--output", required=True)

    lock = subparsers.add_parser("lock-predictions")
    lock.add_argument("--campaign-id", required=True)
    lock.add_argument("--artifact-root", required=True)
    lock.add_argument("--request", action="append", required=True)
    lock.add_argument("--output", required=True)

    arguments = parser.parse_args()
    if arguments.command == "plan":
        write_json_atomic(arguments.output, plan())
    elif arguments.command == "training-commands":
        training_commands(arguments)
    elif arguments.command == "intervention-commands":
        intervention_commands(arguments)
    elif arguments.command == "seal-request":
        seal_stage_request(arguments)
    elif arguments.command == "seal-construction":
        seal_construction_request(arguments)
    elif arguments.command == "lock-predictions":
        lock_predictions(arguments)


if __name__ == "__main__":
    main()
