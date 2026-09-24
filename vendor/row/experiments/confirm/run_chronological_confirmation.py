#!/usr/bin/env python3
"""Create the registry, execution plan, and immutable construction lock."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shlex
import sys
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(HERE))

from confirm.chronological_confirmation_specs import (  # noqa: E402
    ARCHITECTURE,
    CAMPAIGN_ID,
    OPTIMIZER,
    REGISTRY,
    TRAJECTORIES,
)
from confirm.confirmation_artifacts import (  # noqa: E402
    digest_object,
    sha256_path,
    validate_measurement_registry,
)
from confirm.gate_shape_evidence import write_json_atomic  # noqa: E402


PLAN_SCHEMA = "pldr-chronological-live-plan-v1"
LOCK_SCHEMA = "pldr-chronological-construction-lock-v1"


def create_registry(
    tokens: str | Path,
    tokenizer: str | Path,
    output: str | Path,
    *,
    reserved_start: int = -1,
) -> dict[str, Any]:
    """Create the 24-context, 48-intervention checkpoint-owned registry."""

    tokens = Path(tokens).resolve()
    tokenizer = Path(tokenizer).resolve()
    if not tokens.is_file() or not tokenizer.is_file():
        raise FileNotFoundError("tokens and tokenizer must be regular files")
    data = np.memmap(tokens, dtype=np.uint16, mode="r")
    chunk_count = len(data) // REGISTRY["context_length"]
    context_count = REGISTRY["context_count"]
    start = int(reserved_start)
    if start < 0:
        start = chunk_count - context_count - 64
    maximum_capture_stop = max(
        int(row["data_offset_chunks"])
        + (int(row["updates"]) + max(REGISTRY["construction_block_lengths"]))
        * int(OPTIMIZER["batch_size"])
        for row in TRAJECTORIES
    )
    if start < maximum_capture_stop or start + context_count > chunk_count:
        raise ValueError("fixed contexts overlap training or continuation ranges")

    construction = [
        {"id": f"ccc{index:03d}", "chunk_index": start + index}
        for index in range(8)
    ]
    validation = [
        {"id": f"cch{index:03d}", "chunk_index": start + 8 + index}
        for index in range(16)
    ]
    intervention_units = []
    substream = 3_800_000
    heldout = [row for row in TRAJECTORIES if row["role"] == "heldout"]
    for trajectory in heldout:
        for ordinal in range(REGISTRY["intervention_sites_per_seed"]):
            intervention_units.append({
                "id": f"cci-s{trajectory['seed']}-{ordinal:02d}",
                "seed": int(trajectory["seed"]),
                "ordinal": ordinal,
                "future_update_start": ordinal * REGISTRY["intervention_steps"],
                "update_count": REGISTRY["intervention_steps"],
                "rng_substream": substream,
            })
            substream += 1
    registry = {
        "schema_version": "pldr-chronological-probe-registry-v1",
        "context_length": REGISTRY["context_length"],
        "construction": construction,
        "validation": validation,
        "anchors": list(REGISTRY["anchor_steps"]),
        "context_head_block_rule": (
            "head-equals-context-index-modulo-four-v1"),
        "block_row_rule": (
            "complete-ordered-generator-rows-0-through-63-v1"),
        "history_length": REGISTRY["history_length"],
        "intervention_units": intervention_units,
        "intervention_unit_rule": (
            "twelve-fixed-four-update-sites-per-heldout-seed-v1"),
        "dataset_sha256": sha256_path(tokens),
        "tokenizer_sha256": sha256_path(tokenizer),
    }
    registry["registry_sha256"] = digest_object(registry)
    validate_measurement_registry(registry, require_nonempty=True)
    write_json_atomic(output, registry)
    return registry


def _training_command(
    trajectory: dict[str, Any],
    *,
    device: str,
    tokens: Path,
    tokenizer: Path,
    registry: Path,
    run_root: Path,
) -> list[str]:
    anchors = (
        REGISTRY["development_anchor_steps"]
        if trajectory["role"] == "development"
        else REGISTRY["anchor_steps"]
    )
    checkpoints = ",".join(map(str, [*anchors, trajectory["updates"]]))
    return [
        sys.executable,
        str(ROOT / "experiments" / "train_run.py"),
        "--name", trajectory["name"],
        "--run-id", trajectory["name"],
        "--lr", str(OPTIMIZER["learning_rate"]),
        "--warmup", str(OPTIMIZER["warmup_updates"]),
        "--const_lr",
        "--steps", str(trajectory["updates"]),
        "--batch", str(OPTIMIZER["batch_size"]),
        "--ctx", str(OPTIMIZER["context_length"]),
        "--layers", str(ARCHITECTURE["layers"]),
        "--heads", str(ARCHITECTURE["heads"]),
        "--dk", str(ARCHITECTURE["head_width"]),
        "--adff", "170",
        "--seed", str(trajectory["seed"]),
        "--device", device,
        "--tokens", str(tokens),
        "--tok_model", str(tokenizer),
        "--outdir", str(run_root),
        "--data_offset", str(trajectory["data_offset_chunks"]),
        "--confirmation_registry", str(registry),
        "--probe_region", "global",
        "--optimizer", "adamw",
        "--wd", str(OPTIMIZER["weight_decay"]),
        "--clip", str(OPTIMIZER["gradient_clip_value"]),
        "--chronological_history_length", str(REGISTRY["history_length"]),
        "--ckpt_steps", checkpoints,
        "--optimizer_ckpt_steps", checkpoints,
        "--skip_generation",
        "--terminal_validation",
    ]


def create_plan(
    tokens: str | Path,
    tokenizer: str | Path,
    registry: str | Path,
    run_root: str | Path,
    evidence_root: str | Path,
    output: str | Path,
    *,
    devices: tuple[str, ...],
) -> dict[str, Any]:
    """Resolve every training and all-cell capture job without execution."""

    if not devices or len(set(devices)) != len(devices):
        raise ValueError("plan devices must be nonempty and unique")
    tokens = Path(tokens).resolve()
    tokenizer = Path(tokenizer).resolve()
    registry_path = Path(registry).resolve()
    run_root = Path(run_root).resolve()
    evidence_root = Path(evidence_root).resolve()
    with registry_path.open("r", encoding="utf-8") as stream:
        registry_value = json.load(stream)
    validate_measurement_registry(registry_value, require_nonempty=True)
    jobs = []
    capture_index = 0
    for trajectory_index, trajectory in enumerate(TRAJECTORIES):
        device = devices[trajectory_index % len(devices)]
        jobs.append({
            "id": f"train-{trajectory['name']}",
            "stage": "D" if trajectory["role"] == "development"
            else "C" if trajectory["role"] == "construction" else "H",
            "role": trajectory["role"],
            "device": device,
            "command": _training_command(
                trajectory,
                device=device,
                tokens=tokens,
                tokenizer=tokenizer,
                registry=registry_path,
                run_root=run_root,
            ),
        })
        anchors = (
            REGISTRY["development_anchor_steps"]
            if trajectory["role"] == "development"
            else REGISTRY["anchor_steps"]
        )
        blocks: tuple[int | str, ...] = (
            REGISTRY["construction_block_lengths"]
            if trajectory["role"] in {"development", "construction"}
            else ("{construction_primary_block}",)
        )
        for layer in range(ARCHITECTURE["layers"]):
            for anchor in anchors:
                for block in blocks:
                    device = devices[capture_index % len(devices)]
                    capture_index += 1
                    stem = (
                        f"{trajectory['name']}-l{layer}-a{anchor}-b{block}"
                    )
                    capture_command = [
                        sys.executable,
                        str(HERE / "capture_chronological_confirmation.py"),
                        "--checkpoint",
                        str(run_root / trajectory["name"] / f"ckpt_{anchor}.pt"),
                        "--registry", str(registry_path),
                        "--tokens", str(tokens),
                        "--layer", str(layer),
                        "--updates", str(block),
                        "--role", trajectory["role"],
                        "--seed", str(trajectory["seed"]),
                        "--device", device,
                        "--output", str(evidence_root / f"{stem}.npz"),
                    ]
                    if trajectory["role"] == "heldout":
                        capture_command.extend([
                            "--lock", "{construction_lock}",
                            "--shape-safety-factor",
                            "{shape_safety_factor}",
                            "--numerical-relative-coefficient",
                            "{numerical_relative_coefficient}",
                            "--numerical-absolute-charge",
                            "{numerical_absolute_charge}",
                        ])
                    jobs.append({
                        "id": f"capture-{stem}",
                        "stage": "D" if trajectory["role"] == "development"
                        else "C" if trajectory["role"] == "construction" else "H",
                        "role": trajectory["role"],
                        "device": device,
                        "requires_construction_lock": (
                            trajectory["role"] == "heldout"),
                        "command": capture_command,
                    })
        if trajectory["role"] in {"construction", "heldout"}:
            for layer in range(ARCHITECTURE["layers"]):
                for anchor in REGISTRY["anchor_steps"]:
                    device = devices[capture_index % len(devices)]
                    capture_index += 1
                    stem = (
                        f"plga-{trajectory['name']}-l{layer}-a{anchor}"
                    )
                    command = [
                        sys.executable,
                        str(HERE / "capture_chronological_plga.py"),
                        "--checkpoint",
                        str(run_root / trajectory["name"] / f"ckpt_{anchor}.pt"),
                        "--registry", str(registry_path),
                        "--tokens", str(tokens),
                        "--role", trajectory["role"],
                        "--seed", str(trajectory["seed"]),
                        "--layer", str(layer),
                        "--device", device,
                        "--output", str(evidence_root / f"{stem}.npz"),
                    ]
                    if trajectory["role"] == "heldout":
                        command.extend([
                            "--lock", "{construction_lock}",
                        ])
                    jobs.append({
                        "id": f"capture-{stem}",
                        "stage": (
                            "C" if trajectory["role"] == "construction"
                            else "P"
                        ),
                        "role": trajectory["role"],
                        "device": device,
                        "requires_construction_lock": (
                            trajectory["role"] == "heldout"),
                        "command": command,
                    })
        if trajectory["role"] == "heldout":
            units = [
                row for row in registry_value["intervention_units"]
                if int(row["seed"]) == int(trajectory["seed"])
            ]
            if len(units) != REGISTRY["intervention_sites_per_seed"]:
                raise ValueError("intervention registry is incomplete for a seed")
            for unit in units:
                device = devices[capture_index % len(devices)]
                capture_index += 1
                stem = str(unit["id"])
                jobs.append({
                    "id": f"intervene-{stem}",
                    "stage": "I",
                    "role": "heldout",
                    "device": device,
                    "requires_construction_lock": True,
                    "command": [
                        sys.executable,
                        str(HERE / "run_chronological_intervention.py"),
                        "--checkpoint",
                        str(
                            run_root / trajectory["name"]
                            / f"ckpt_{REGISTRY['intervention_anchor_step']}.pt"
                        ),
                        "--registry", str(registry_path),
                        "--tokens", str(tokens),
                        "--lock", "{construction_lock}",
                        "--unit-id", stem,
                        "--seed", str(trajectory["seed"]),
                        "--device", device,
                        "--output", str(
                            evidence_root / f"intervention-{stem}.npz"),
                    ],
                })
    payload = {
        "schema_version": PLAN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "registry": {
            "path": str(registry_path),
            "sha256": sha256_path(registry_path),
        },
        "tokens_sha256": sha256_path(tokens),
        "tokenizer_sha256": sha256_path(tokenizer),
        "devices": list(devices),
        "jobs": jobs,
        "analysis_commands": {
            "pair_record_prelock": [
                sys.executable,
                str(ROOT / "experiments" / "analysis"
                    / "analyze_chronological_confirmation.py"),
                "{pair_ledger}",
                "--output", "{pair_report}",
                "--require-pass",
            ],
            "pair_record_heldout": [
                sys.executable,
                str(ROOT / "experiments" / "analysis"
                    / "analyze_chronological_confirmation.py"),
                "{pair_ledger}",
                "--lock", "{construction_lock}",
                "--output", "{pair_report}",
                "--require-pass",
            ],
            "construction_plga": [
                sys.executable,
                str(ROOT / "experiments" / "analysis"
                    / "analyze_chronological_plga.py"),
                "{construction_plga_ledgers}",
                "--output", "{construction_plga_report}",
                "--require-pass",
            ],
            "heldout_plga": [
                sys.executable,
                str(ROOT / "experiments" / "analysis"
                    / "analyze_chronological_plga.py"),
                "{heldout_plga_ledgers}",
                "--lock", "{construction_lock}",
                "--output", "{heldout_plga_report}",
                "--require-pass",
            ],
            "interventions": [
                sys.executable,
                str(ROOT / "experiments" / "analysis"
                    / "analyze_chronological_intervention.py"),
                "{intervention_ledgers}",
                "--lock", "{construction_lock}",
                "--output", "{intervention_report}",
                "--require-pass",
            ],
        },
    }
    payload["plan_sha256"] = digest_object(payload)
    write_json_atomic(output, payload)
    return payload


def _load_reports(paths: list[str]) -> list[dict[str, Any]]:
    reports = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as stream:
            report = json.load(stream)
        if report.get("schema_version") != "pldr-chronological-analysis-v1":
            raise ValueError(f"unexpected analysis report schema: {path}")
        report["report_path"] = str(Path(path).resolve())
        report["report_sha256"] = sha256_path(path)
        reports.append(report)
    return reports


def _load_plga_construction_report(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    if (
        not isinstance(report, dict)
        or report.get("schema_version")
        != "pldr-chronological-plga-analysis-v1"
        or report.get("role") != "construction"
        or report.get("decision") != "CONSTRUCTION_SUMMARY"
        or report.get("record_count") != 15
        or not all(report.get("checks", {}).values())
    ):
        raise ValueError("PLGA construction report is incomplete or invalid")
    report["report_path"] = str(Path(path).resolve())
    report["report_sha256"] = sha256_path(path)
    return report


def create_lock(
    development_paths: list[str],
    construction_paths: list[str],
    output: str | Path,
    *,
    plga_construction_path: str | Path,
    shape_safety_factor: float,
    numerical_relative_coefficient: float,
    numerical_absolute_charge: float,
    intervention_minimum_effect: float,
    plga_power_secant_bound: float,
    centered_logit_remainder_bound: float,
    plga_transfer_coefficient_bound: float,
    centered_logit_response_coefficient: float,
) -> dict[str, Any]:
    """Freeze the smallest qualifying development block and all constants."""

    development = _load_reports(development_paths)
    construction = _load_reports(construction_paths)
    plga_construction = _load_plga_construction_report(
        plga_construction_path)
    if len(development) != 27 or len(construction) != 45:
        raise ValueError(
            "construction lock needs all 27 development and 45 construction cells"
        )
    if (
        any(
            row["role"] != "development"
            or row["seed"] != 7444
            or row["data_offset_chunks"] != 0
            for row in development
        )
        or any(
            row["role"] != "construction"
            or row["seed"] != 8444
            or row["data_offset_chunks"] != 8192
            for row in construction
        )
    ):
        raise ValueError(
            "construction lock report roles or trajectories disagree")
    development_cells = {
        (row["layer"], row["step_start"], row["step_stop"] - row["step_start"])
        for row in development
    }
    construction_cells = {
        (row["layer"], row["step_start"], row["step_stop"] - row["step_start"])
        for row in construction
    }
    expected_development = {
        (layer, anchor, block)
        for layer in range(ARCHITECTURE["layers"])
        for anchor in REGISTRY["development_anchor_steps"]
        for block in REGISTRY["construction_block_lengths"]
    }
    expected_construction = {
        (layer, anchor, block)
        for layer in range(ARCHITECTURE["layers"])
        for anchor in REGISTRY["anchor_steps"]
        for block in REGISTRY["construction_block_lengths"]
    }
    if (
        development_cells != expected_development
        or construction_cells != expected_construction
    ):
        raise ValueError(
            "development or construction report-cell registry is incomplete")

    qualifying = []
    for block in REGISTRY["construction_block_lengths"]:
        block_rows = [
            row for row in development
            if row["step_stop"] - row["step_start"] == block
        ]
        if len(block_rows) == 9 and all(
            row["decision"] == "QUALIFIED" for row in block_rows
        ):
            qualifying.append(block)
    if not qualifying:
        raise ValueError(
            "no block qualifies across all development layer-anchor cells")
    primary = min(qualifying)
    primary_construction = [
        row for row in construction
        if row["step_stop"] - row["step_start"] == primary
    ]
    if len(primary_construction) != 15 or any(
        row["decision"] != "QUALIFIED" for row in primary_construction
    ):
        raise ValueError(
            "the primary block does not qualify across construction cells")

    constants = (
        shape_safety_factor,
        numerical_relative_coefficient,
        numerical_absolute_charge,
        intervention_minimum_effect,
        plga_power_secant_bound,
        centered_logit_remainder_bound,
        plga_transfer_coefficient_bound,
        centered_logit_response_coefficient,
    )
    if (
        not all(math.isfinite(value) for value in constants)
        or shape_safety_factor < 1.0
        or numerical_relative_coefficient < 0.0
        or numerical_absolute_charge < 0.0
        or intervention_minimum_effect <= 0.0
        or plga_power_secant_bound <= 0.0
        or centered_logit_remainder_bound < 0.0
        or plga_transfer_coefficient_bound <= 0.0
        or centered_logit_response_coefficient <= 0.0
    ):
        raise ValueError("construction constants are invalid")
    plga_maxima = plga_construction["maxima"]
    if (
        plga_power_secant_bound < float(plga_maxima["power_secant"])
        or plga_transfer_coefficient_bound
        < float(plga_maxima["transfer_coefficient"])
        or centered_logit_response_coefficient
        < float(plga_maxima["centered_response_ratio"])
    ):
        raise ValueError(
            "PLGA lock constants do not cover the construction report")
    payload = {
        "schema_version": LOCK_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "primary_block_length": primary,
        "history_length": REGISTRY["history_length"],
        "segment_grid": [0.0, 0.5, 1.0],
        "shape_safety_factor": shape_safety_factor,
        "numerical_relative_coefficient": numerical_relative_coefficient,
        "numerical_absolute_charge": numerical_absolute_charge,
        "intervention_minimum_effect": intervention_minimum_effect,
        "plga_power_secant_bound": plga_power_secant_bound,
        "centered_logit_remainder_bound": centered_logit_remainder_bound,
        "constant_enlargement_after_lock": False,
        "plga_transfer_coefficient_bound": plga_transfer_coefficient_bound,
        "centered_logit_response_coefficient": centered_logit_response_coefficient,
        "development_reports": [
            {"path": row["report_path"], "sha256": row["report_sha256"]}
            for row in development
        ],
        "construction_reports": [
            {"path": row["report_path"], "sha256": row["report_sha256"]}
            for row in construction
        ],
        "plga_construction_report": {
            "path": plga_construction["report_path"],
            "sha256": plga_construction["report_sha256"],
        },
    }
    payload["lock_sha256"] = digest_object(payload)
    write_json_atomic(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    registry_parser = subparsers.add_parser("registry")
    registry_parser.add_argument("--tokens", required=True)
    registry_parser.add_argument("--tokenizer", required=True)
    registry_parser.add_argument("--reserved-start", type=int, default=-1)
    registry_parser.add_argument("--output", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--tokens", required=True)
    plan_parser.add_argument("--tokenizer", required=True)
    plan_parser.add_argument("--registry", required=True)
    plan_parser.add_argument("--run-root", required=True)
    plan_parser.add_argument("--evidence-root", required=True)
    plan_parser.add_argument("--devices", default="cuda:0,cuda:1")
    plan_parser.add_argument("--output", required=True)
    lock_parser = subparsers.add_parser("lock")
    lock_parser.add_argument("--development-report", action="append", default=[])
    lock_parser.add_argument("--construction-report", action="append", default=[])
    lock_parser.add_argument("--plga-construction-report", required=True)
    lock_parser.add_argument("--shape-safety-factor", type=float, required=True)
    lock_parser.add_argument(
        "--numerical-relative-coefficient", type=float, required=True)
    lock_parser.add_argument(
        "--numerical-absolute-charge", type=float, required=True)
    lock_parser.add_argument(
        "--intervention-minimum-effect", type=float, required=True)
    lock_parser.add_argument(
        "--plga-power-secant-bound", type=float, required=True)
    lock_parser.add_argument(
        "--centered-logit-remainder-bound", type=float, required=True)
    lock_parser.add_argument(
        "--plga-transfer-coefficient-bound", type=float, required=True)
    lock_parser.add_argument(
        "--centered-logit-response-coefficient", type=float, required=True)
    lock_parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    if arguments.command == "registry":
        create_registry(
            arguments.tokens, arguments.tokenizer, arguments.output,
            reserved_start=arguments.reserved_start)
    elif arguments.command == "plan":
        devices = tuple(value for value in arguments.devices.split(",") if value)
        payload = create_plan(
            arguments.tokens, arguments.tokenizer, arguments.registry,
            arguments.run_root, arguments.evidence_root, arguments.output,
            devices=devices)
        print(f"chronological plan: {len(payload['jobs'])} jobs")
    else:
        create_lock(
            arguments.development_report,
            arguments.construction_report,
            arguments.output,
            plga_construction_path=arguments.plga_construction_report,
            shape_safety_factor=arguments.shape_safety_factor,
            numerical_relative_coefficient=arguments.numerical_relative_coefficient,
            numerical_absolute_charge=arguments.numerical_absolute_charge,
            intervention_minimum_effect=arguments.intervention_minimum_effect,
            plga_power_secant_bound=arguments.plga_power_secant_bound,
            centered_logit_remainder_bound=arguments.centered_logit_remainder_bound,
            plga_transfer_coefficient_bound=(
                arguments.plga_transfer_coefficient_bound),
            centered_logit_response_coefficient=(
                arguments.centered_logit_response_coefficient),
        )
    if arguments.command in {"registry", "lock"}:
        print(shlex.join(["wrote", str(Path(arguments.output).resolve())]))


if __name__ == "__main__":
    main()
