#!/usr/bin/env python3
"""Plan, launch, seal, qualify, and analyze the confirmation campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ANALYSIS = ROOT / "experiments" / "analysis"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ANALYSIS))

from confirmation_artifacts import (  # noqa: E402
    REQUEST_SCHEMA_VERSION,
    sha256_path,
    write_json_atomic,
)
from program_energy_protocol_specs import (  # noqa: E402
    ARCHITECTURE_RUNS,
    ARTIFACT_ROLES,
    BLOCKING_QUALIFICATION,
    RESOURCE_CAPS,
    STAGES,
)
from composite_confirmation_analysis import analyze_stage  # noqa: E402
from analyze_program_energy_confirmation import analyze  # noqa: E402


def _training_command(
        run, tokens, tokenizer, registry, output_root, device):
    windows = sorted(
        set(run["landmarks"])
        | {
            step for anchor in run["normal_window_starts"]
            for step in range(anchor, anchor + 9)
        }
    )
    snapshots = sorted({
        step for anchor in run["normal_window_starts"]
        for step in range(anchor + 1, anchor + 9)
    })
    checkpoint_text = ",".join(str(value) for value in windows)
    snapshot_text = ",".join(str(value) for value in snapshots)
    command = [
        "python3", "experiments/train_run.py",
        "--name", run["run"],
        "--lr", "0.0003",
        "--warmup", "512",
        "--steps", str(run["updates"]),
        "--schedule_total_steps", str(run["updates"]),
        "--batch", "32",
        "--ctx", "256",
        "--layers", "4",
        "--heads", "4",
        "--dk", "16",
        "--adff", "48",
        "--seed", str(run["seed"]),
        "--device", device,
        "--tokens", str(tokens),
        "--tok_model", str(tokenizer),
        "--confirmation_registry", str(registry),
        "--outdir", str(output_root),
        "--ckpt_steps", checkpoint_text,
        "--optimizer_ckpt_steps", checkpoint_text,
        "--confirmation_snapshot_steps", snapshot_text,
        "--confirmation_snapshot_blocks",
        "phi,plga,query,key,value,attn,ffn",
        "--probe_region", "global",
        "--skip_generation",
        "--terminal_validation",
    ]
    return " ".join(shlex.quote(value) for value in command)


PRIMARY_RUNS = tuple(
    run for run in ARCHITECTURE_RUNS if run["gate"] == "PRIMARY")
GATED_RUNS = tuple(
    run for run in ARCHITECTURE_RUNS if run["gate"] != "PRIMARY")


def plan(
        tokens=None, tokenizer=None, registry=None, output_root=None,
        devices=None):
    devices = devices or ("cuda:0",)
    commands = []
    supplied = (tokens, tokenizer, registry, output_root)
    if any(value is not None for value in supplied) and any(
        value is None for value in supplied
    ):
        raise ValueError(
            "tokens, tokenizer, registry, and output root are one launch set")
    if all(value is not None for value in supplied):
        if len(devices) != len(PRIMARY_RUNS):
            raise ValueError(
                "one device must be supplied per primary architecture run")
        for run, device in zip(PRIMARY_RUNS, devices):
            commands.append(_training_command(
                run, tokens, tokenizer, registry, output_root, device))
    return {
        "schema_version": "pldr-collapse-execution-plan-v3",
        "stage_order": [row["id"] for row in STAGES],
        "stages": list(STAGES),
        "architecture_runs": list(ARCHITECTURE_RUNS),
        "resource_caps": RESOURCE_CAPS,
        "ordered_affine_blocking": BLOCKING_QUALIFICATION,
        "protocol_root": str(
            ROOT / "experiments" / "protocols"
            / "program_bound_confirmation"),
        "training_commands": commands,
        "gated_replication": {
            "release_condition": "CONFIRMED_Q_T_N_SEALED_RECORDS",
            "architecture_runs": list(GATED_RUNS),
        },
        "commands": {
            "qualify": "run_program_energy_campaign.py qualify --device DEVICE",
            "training_commands": (
                "run_program_energy_campaign.py training-commands "
                "--tokens TOKENS --tokenizer TOKENIZER "
                "--registry REGISTRY --output-root CAMPAIGN/T"
            ),
            "gated_replication_command": (
                "run_program_energy_campaign.py gated-replication-command "
                "--records Q_RECORD T_RECORD N_RECORD --tokens TOKENS "
                "--tokenizer TOKENIZER --registry REGISTRY "
                "--output-root CAMPAIGN/T_REPLICATION"
            ),
            "seal_request": (
                "run_program_energy_campaign.py seal-request --stage STAGE "
                "--campaign-id ID --artifact-root ROOT "
                "--artifact ROLE=PATH ... --output REQUEST"
            ),
            "stage_analyze": (
                "run_program_energy_campaign.py stage-analyze --stage STAGE "
                "--request REQUEST --artifact-root ROOT --output RECORD"
            ),
            "campaign_analyze": (
                "run_program_energy_campaign.py analyze "
                "--records Q T N I A R --output REPORT"
            ),
        },
    }


def gated_replication_command(
        *, records, tokens, tokenizer, registry, output_root,
        device="cuda:0"):
    gate = analyze(records)
    required = ("Q", "T", "N")
    if any(
        gate["stages"][stage]["verdict"] != "CONFIRMED"
        for stage in required
    ):
        raise ValueError(
            "gated replication requires confirmed sealed Q, T, and N records")
    if len(GATED_RUNS) != 1:
        raise ValueError("the frozen protocol must define one gated replication")
    command = _training_command(
        GATED_RUNS[0], tokens, tokenizer, registry, output_root, device)
    return {
        "schema_version": "pldr-collapse-gated-replication-launch-v3",
        "gate_record_sha256": gate["record_sha256"],
        "required_confirmed_stages": list(required),
        "architecture_run": GATED_RUNS[0],
        "training_command": command,
    }


def seal_request(*, stage, campaign_id, time_semantics, artifact_root,
                 bindings):
    if stage not in {row["id"] for row in STAGES}:
        raise ValueError("unknown stage")
    parsed = {}
    for binding in bindings:
        if "=" not in binding:
            raise ValueError("artifact binding must be ROLE=RELATIVE_PATH")
        role, raw_path = binding.split("=", 1)
        if role in parsed:
            raise ValueError("artifact role is duplicated")
        relative = Path(raw_path)
        if (
            role not in ARTIFACT_ROLES or relative.is_absolute()
            or "." in relative.parts or ".." in relative.parts
        ):
            raise ValueError("artifact role or normalized relative path is invalid")
        path = Path(artifact_root).resolve() / relative
        if not path.is_file():
            raise ValueError(f"artifact does not exist: {relative}")
        parsed[role] = {
            "role": role,
            "path": relative.as_posix(),
            "sha256": sha256_path(path),
        }
    if set(parsed) != set(ARTIFACT_ROLES):
        raise ValueError("every frozen artifact role must be supplied once")
    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "stage": stage,
        "time_semantics": time_semantics,
        "artifacts": [
            parsed[role] for role in ARTIFACT_ROLES
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument("--output")

    qualify_parser = sub.add_parser("qualify")
    qualify_parser.add_argument("--device", default="cpu")
    qualify_parser.add_argument("--output")

    train_parser = sub.add_parser("training-commands")
    train_parser.add_argument("--tokens", required=True)
    train_parser.add_argument("--tokenizer", required=True)
    train_parser.add_argument("--registry", required=True)
    train_parser.add_argument("--output-root", required=True)
    train_parser.add_argument("--devices", default="cuda:0")
    train_parser.add_argument("--output")

    replication_parser = sub.add_parser("gated-replication-command")
    replication_parser.add_argument("--records", nargs=3, required=True)
    replication_parser.add_argument("--tokens", required=True)
    replication_parser.add_argument("--tokenizer", required=True)
    replication_parser.add_argument("--registry", required=True)
    replication_parser.add_argument("--output-root", required=True)
    replication_parser.add_argument("--device", default="cuda:0")
    replication_parser.add_argument("--output")

    seal_parser = sub.add_parser("seal-request")
    seal_parser.add_argument("--stage", required=True)
    seal_parser.add_argument("--campaign-id", required=True)
    seal_parser.add_argument(
        "--time-semantics", default="FINITE_IMPLEMENTED_SCHEDULE")
    seal_parser.add_argument("--artifact-root", required=True)
    seal_parser.add_argument("--artifact", action="append", default=[])
    seal_parser.add_argument("--output", required=True)

    stage_parser = sub.add_parser("stage-analyze")
    stage_parser.add_argument("--stage", required=True)
    stage_parser.add_argument("--request", required=True)
    stage_parser.add_argument("--artifact-root", required=True)
    stage_parser.add_argument("--output", required=True)

    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--records", nargs="+", required=True)
    analyze_parser.add_argument("--output", required=True)

    arguments = parser.parse_args()
    if arguments.command == "plan":
        value = plan()
        if arguments.output:
            write_json_atomic(arguments.output, value)
        else:
            print(json.dumps(value, indent=2, sort_keys=True))
    elif arguments.command == "qualify":
        from qualification_normal_stability import qualify
        value = qualify(arguments.device)
        if arguments.output:
            write_json_atomic(arguments.output, value)
        else:
            print(json.dumps(value, indent=2, sort_keys=True))
        if value["decision"] != "QUALIFIED":
            raise SystemExit(1)
    elif arguments.command == "training-commands":
        devices = tuple(
            value for value in arguments.devices.split(",") if value)
        if len(devices) != len(PRIMARY_RUNS):
            raise ValueError("one device must be supplied per primary run")
        value = plan(
            arguments.tokens, arguments.tokenizer, arguments.registry,
            arguments.output_root, devices)
        if arguments.output:
            write_json_atomic(arguments.output, value)
        else:
            for command in value["training_commands"]:
                print(command)
    elif arguments.command == "gated-replication-command":
        value = gated_replication_command(
            records=arguments.records,
            tokens=arguments.tokens,
            tokenizer=arguments.tokenizer,
            registry=arguments.registry,
            output_root=arguments.output_root,
            device=arguments.device,
        )
        if arguments.output:
            write_json_atomic(arguments.output, value)
        else:
            print(value["training_command"])
    elif arguments.command == "seal-request":
        value = seal_request(
            stage=arguments.stage,
            campaign_id=arguments.campaign_id,
            time_semantics=arguments.time_semantics,
            artifact_root=arguments.artifact_root,
            bindings=arguments.artifact,
        )
        write_json_atomic(arguments.output, value)
    elif arguments.command == "stage-analyze":
        with Path(arguments.request).open("r", encoding="utf-8") as stream:
            request = json.load(stream)
        value = analyze_stage(
            stage_id=arguments.stage,
            request=request,
            artifact_root=arguments.artifact_root,
        )
        write_json_atomic(arguments.output, value)
        if value["decision"] != "CONFIRMED":
            raise SystemExit(1)
    else:
        write_json_atomic(arguments.output, analyze(arguments.records))


if __name__ == "__main__":
    main()
