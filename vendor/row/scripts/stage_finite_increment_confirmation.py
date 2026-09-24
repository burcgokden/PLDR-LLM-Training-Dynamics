#!/usr/bin/env python3
"""Stage the all-layer finite-increment confirmation campaign."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from confirm.confirmation_artifacts import digest_object, sha256_path  # noqa: E402
from confirm.finite_increment_specs import (  # noqa: E402
    ARCHITECTURE,
    CAMPAIGN_ID,
    CONSECUTIVE_EDGES,
    FINITE_INCREMENT_POLICY,
    HELDOUT_ANCHORS,
    OPTIMIZER,
    PLAN_SCHEMA,
    REGISTRY,
    REGISTRY_SCHEMA,
    RESOURCE_BUDGET,
    SCHEMA_VERSION,
    STAGES,
    TRAJECTORIES,
)
from train_run import (  # noqa: E402
    PROBE_RESERVE_CHUNKS,
    training_chunk_limit,
    validate_data_order,
)


DEFAULT_OUTPUT = (
    ROOT.parent / "experiment-data" / "manuscript-revisions" / "rev42"
    / "finite-increment-collapse-confirmation"
)
DEFAULT_TOKENS = (
    ROOT.parent / "experiment-data" / "shared" / "datasets"
    / "refinedweb-100m-tokens" / "refinedweb_tokens.npy"
)
DEFAULT_TOKENIZER = ROOT / "experiments" / "data" / "tokenizer.model"

ENTRY_POINTS = (
    "experiments/train_run.py",
    "experiments/pldr_model_v510.py",
    "experiments/power_law_attention_layer_v510.py",
    "experiments/optimizer_ledger.py",
    "experiments/instrument.py",
    "experiments/confirm/finite_increment.py",
    "experiments/confirm/finite_increment_specs.py",
    "experiments/confirm/capture_finite_increment_source.py",
    "experiments/confirm/produce_finite_increment_certificate.py",
    "experiments/confirm/capture_finite_increment_successor.py",
    "experiments/analysis/analyze_finite_increment_confirmation.py",
    "scripts/run_finite_increment_qualification.py",
    "scripts/execute_finite_increment_plan.py",
    "scripts/stage_finite_increment_confirmation.py",
    "scripts/retain_model_only_checkpoint.py",
)

DYNAMIC_IMPORTS = (
    "experiments/confirm/confirmation_artifacts.py",
    "experiments/confirm/gate_shape_evidence.py",
    "experiments/confirm/row_map_live.py",
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _file_record(path: Path, base: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(base).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_path(path),
    }


def _resolve_import(module: str) -> Path | None:
    parts = module.split(".")
    candidates = [
        ROOT.joinpath(*parts).with_suffix(".py"),
        ROOT.joinpath(*parts, "__init__.py"),
        EXPERIMENTS.joinpath(*parts).with_suffix(".py"),
        EXPERIMENTS.joinpath(*parts, "__init__.py"),
        EXPERIMENTS.joinpath("confirm", *parts).with_suffix(".py"),
        EXPERIMENTS.joinpath("analysis", *parts).with_suffix(".py"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def source_import_closure() -> list[Path]:
    """Return the local static import closure plus declared dynamic entries."""

    pending = [ROOT / relative for relative in ENTRY_POINTS + DYNAMIC_IMPORTS]
    selected: set[Path] = set()
    while pending:
        path = pending.pop().resolve()
        if path in selected:
            continue
        try:
            path.relative_to(ROOT)
        except ValueError as error:
            raise ValueError("a source entry escapes the repository") from error
        if not path.is_file():
            raise FileNotFoundError(f"campaign source is missing: {path}")
        selected.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.append(node.module)
        for module in modules:
            resolved = _resolve_import(module)
            if resolved is not None and resolved not in selected:
                pending.append(resolved)
    return sorted(selected)


def create_registry(tokens: Path, tokenizer: Path) -> dict[str, Any]:
    data = np.memmap(tokens, dtype=np.uint16, mode="r")
    total_chunks = len(data) // int(REGISTRY["context_length"])
    reserve_start = training_chunk_limit(total_chunks)
    context_count = int(REGISTRY["context_count"])
    if reserve_start + context_count > total_chunks:
        raise ValueError("the trainer probe tail cannot hold the registry")
    construction = []
    validation = []
    for index in range(context_count):
        ownership = (
            "construction"
            if index < int(REGISTRY["construction_context_count"])
            else "validation"
        )
        row = {
            "id": f"{ownership[0]}{index:02d}",
            "chunk_index": reserve_start + index,
        }
        if ownership == "construction":
            construction.append(row)
        else:
            validation.append(row)
    value = {
        "schema_version": REGISTRY_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "context_length": REGISTRY["context_length"],
        "construction": construction,
        "validation": validation,
        "dataset_sha256": sha256_path(tokens),
        "tokenizer_sha256": sha256_path(tokenizer),
        "total_chunks": total_chunks,
        "trainer_probe_reserve_chunks": PROBE_RESERVE_CHUNKS,
        "trainer_training_limit": reserve_start,
        "heads": list(range(ARCHITECTURE["heads"])),
        "layers": list(REGISTRY["layers"]),
        "rows_per_map": REGISTRY["rows_per_map"],
        "pairs_per_map": REGISTRY["pairs_per_map"],
        "pair_rule": REGISTRY["pair_rule"],
        "cross_map_pairs_forbidden": True,
    }
    value["registry_sha256"] = digest_object(value)
    return value


def campaign_spec(registry: dict[str, Any]) -> dict[str, Any]:
    value = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "architecture": ARCHITECTURE,
        "optimizer": OPTIMIZER,
        "trajectories": list(TRAJECTORIES),
        "registry_sha256": registry["registry_sha256"],
        "finite_increment_policy": FINITE_INCREMENT_POLICY,
        "resource_budget": RESOURCE_BUDGET,
        "stages": list(STAGES),
        "policy_locked_before_heldout": True,
        "prediction_precedes_independent_successor": True,
        "scientific_gate": (
            "complete_all_layer_coverage_and_positive_state_dominance"
        ),
    }
    value["spec_sha256"] = digest_object(value)
    return value


def _order(seed: int, total_chunks: int) -> np.ndarray:
    limit = training_chunk_limit(total_chunks)
    generated = np.random.Generator(np.random.PCG64(seed)).permutation(limit)
    return validate_data_order(generated, total_chunks)


def _node(
    node_id: str,
    stage: str,
    argv: list[str],
    *,
    depends_on: tuple[str, ...] = (),
    outputs: tuple[str, ...] = (),
    device: str = "cpu",
    predicted_output_bytes: int = 0,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "stage": stage,
        "device": device,
        "depends_on": list(depends_on),
        "argv": argv,
        "outputs": list(outputs),
        "predicted_output_bytes": int(predicted_output_bytes),
    }


def _edge_nodes(
    trajectory: dict[str, Any],
    step: int,
    stage: str,
    *,
    dependency: str,
) -> list[dict[str, Any]]:
    role = str(trajectory["role"])
    seed = int(trajectory["seed"])
    key = f"{role}-s{seed}-a{step}"
    checkpoint = f"work/{key}/ckpt_{step}.pt"
    successor_checkpoint = f"work/{key}/ckpt_{step + 1}.pt"
    source = f"artifacts/source-{key}.npz"
    certificate = f"artifacts/finite-{key}.json"
    prediction = f"reports/predict-{key}.json"
    successor = f"artifacts/successor-{key}.npz"
    coverage = f"reports/coverage-{key}.json"
    source_id = f"source-{key}"
    finite_id = f"finite-{key}"
    predict_id = f"predict-{key}"
    successor_id = f"successor-{key}"
    coverage_id = f"coverage-{key}"
    common = [
        "--bundle", "{bundle}",
        "--registry", "protocol/registry.json",
        "--tokens", "{tokens}",
    ]
    return [
        _node(
            source_id,
            stage,
            [
                "python3", "source/experiments/confirm/capture_finite_increment_source.py",
                *common,
                "--checkpoint", checkpoint,
                "--data-order", f"orders/{role}-seed{seed}-chunk-order.npy",
                "--output", source,
                "--role", role,
                "--seed", str(seed),
                "--step", str(step),
            ],
            depends_on=(dependency,),
            outputs=(source,),
            device="cuda:0" if seed % 2 == 0 else "cuda:1",
            predicted_output_bytes=48_000_000,
        ),
        _node(
            finite_id,
            stage,
            [
                "python3", "source/experiments/confirm/produce_finite_increment_certificate.py",
                "--source", source,
                "--output", certificate,
            ],
            depends_on=(source_id,),
            outputs=(certificate,),
            predicted_output_bytes=2_000_000,
        ),
        _node(
            predict_id,
            stage,
            [
                "python3", "source/experiments/analysis/analyze_finite_increment_confirmation.py",
                "predict", "--certificate", certificate, "--output", prediction,
            ],
            depends_on=(finite_id,),
            outputs=(prediction,),
            predicted_output_bytes=2_000_000,
        ),
        _node(
            successor_id,
            stage,
            [
                "python3", "source/experiments/confirm/capture_finite_increment_successor.py",
                *common,
                "--checkpoint", successor_checkpoint,
                "--prediction", prediction,
                "--output", successor,
            ],
            depends_on=(predict_id,),
            outputs=(successor,),
            device="cuda:0" if seed % 2 == 0 else "cuda:1",
            predicted_output_bytes=16_000_000,
        ),
        _node(
            coverage_id,
            stage,
            [
                "python3", "source/experiments/analysis/analyze_finite_increment_confirmation.py",
                "coverage", "--certificate", certificate,
                "--successor", successor, "--prediction", prediction,
                "--output", coverage, "--require-coverage",
            ],
            depends_on=(successor_id,),
            outputs=(coverage,),
            predicted_output_bytes=2_000_000,
        ),
    ]


def build_plan() -> dict[str, Any]:
    nodes = [
        _node(
            "qualification-e0",
            "E0",
            [
                "python3", "source/scripts/run_finite_increment_qualification.py",
                "--output", "reports/qualification-e0.json", "--require-pass",
            ],
            outputs=("reports/qualification-e0.json",),
            predicted_output_bytes=100_000,
        )
    ]
    for device in RESOURCE_BUDGET["devices"]:
        suffix = device.replace(":", "")
        nodes.append(_node(
            f"qualification-real-{suffix}",
            "E1",
            [
                "python3", "source/scripts/run_finite_increment_qualification.py",
                "--output", f"reports/qualification-real-{suffix}.json",
                "--device", device,
                "--real-checkpoint", "{qualification_checkpoint}",
                "--tokens", "{tokens}",
                "--require-pass",
            ],
            depends_on=("qualification-e0",),
            outputs=(f"reports/qualification-real-{suffix}.json",),
            device=device,
            predicted_output_bytes=2_000_000,
        ))
    nodes.append(_node(
        "construction-ready",
        "E1",
        ["python3", "-c", "print('construction-ready')"],
        depends_on=("qualification-real-cuda0", "qualification-real-cuda1"),
    ))

    construction = TRAJECTORIES[0]
    for step in CONSECUTIVE_EDGES:
        stage = "E2" if step == REGISTRY["construction_anchor"] else "E3"
        nodes.extend(_edge_nodes(
            construction, step, stage, dependency="construction-ready"
        ))
    construction_coverages = tuple(
        f"coverage-construction-s{construction['seed']}-a{step}"
        for step in CONSECUTIVE_EDGES
    )
    nodes.append(_node(
        "aggregate-consecutive-all-layer",
        "E3",
        [
            "python3", "source/experiments/analysis/analyze_finite_increment_confirmation.py",
            "aggregate", "--reports-glob", "reports/coverage-construction-*.json",
            "--output", "reports/consecutive-all-layer.json",
        ],
        depends_on=construction_coverages,
        outputs=("reports/consecutive-all-layer.json",),
        predicted_output_bytes=2_000_000,
    ))
    nodes.append(_node(
        "construction-lock",
        "E3",
        [
            "python3", "source/experiments/analysis/analyze_finite_increment_confirmation.py",
            "lock", "--aggregate", "reports/consecutive-all-layer.json",
            "--output", "protocol/construction-lock.json",
        ],
        depends_on=("aggregate-consecutive-all-layer",),
        outputs=("protocol/construction-lock.json",),
        predicted_output_bytes=200_000,
    ))

    for trajectory in TRAJECTORIES:
        role = str(trajectory["role"])
        seed = int(trajectory["seed"])
        nodes.append(_node(
            f"diameter-time-course-{role}-s{seed}",
            "E4",
            [
                "python3", "source/experiments/analysis/analyze_finite_increment_confirmation.py",
                "time-course", "--run-root", f"work/{role}-s{seed}",
                "--registry", "protocol/registry.json",
                "--output", f"reports/time-course-{role}-s{seed}.json",
            ],
            depends_on=("construction-lock",),
            outputs=(f"reports/time-course-{role}-s{seed}.json",),
            predicted_output_bytes=4_000_000,
        ))

    for trajectory in TRAJECTORIES[1:]:
        for step in HELDOUT_ANCHORS:
            nodes.extend(_edge_nodes(
                trajectory, step, "E5", dependency="construction-lock"
            ))

    known = {node["id"] for node in nodes}
    if len(known) != len(nodes):
        raise AssertionError("execution-plan node identifiers are not unique")
    if any(set(node["depends_on"]) - known for node in nodes):
        raise AssertionError("execution-plan dependency is unknown")
    value = {
        "schema_version": PLAN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "status": "ready_for_e0_then_real_path_qualification",
        "source_only_prediction": True,
        "nodes": nodes,
        "node_count": len(nodes),
        "theorem_node_bindings": {
            "finite-source-transport": "source-construction-s7444-a1000",
            "two-variable-plga": "finite-construction-s7444-a1000",
            "source-state-contraction": "predict-construction-s7444-a1000",
            "product-convolution": "aggregate-consecutive-all-layer",
            "heldout-coverage": "coverage-heldout-s8444-a1000",
        },
    }
    value["plan_sha256"] = digest_object(value)
    return value


def _trajectory_device(trajectory: dict[str, Any]) -> str:
    index = next(
        index for index, row in enumerate(TRAJECTORIES)
        if int(row["seed"]) == int(trajectory["seed"])
    )
    return str(RESOURCE_BUDGET["devices"][index % len(RESOURCE_BUDGET["devices"])])


def _training_node(
    trajectory: dict[str, Any],
    *,
    dependency: str,
    optimizer_checkpoints: tuple[int, ...],
) -> dict[str, Any]:
    role = str(trajectory["role"])
    seed = int(trajectory["seed"])
    name = str(trajectory["name"])
    device = _trajectory_device(trajectory)
    return _node(
        f"train-{role}-s{seed}",
        "E4" if role == "construction" else "E5",
        [
            "python3", "source/experiments/train_run.py",
            "--name", name,
            "--run-id", name,
            "--lineage-root", name,
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
            "--seed", str(seed),
            "--device", device,
            "--tokens", "{tokens}",
            "--tok_model", "{tokenizer}",
            "--outdir", "work",
            "--data_offset", "0",
            "--data_order", f"orders/{role}-seed{seed}-chunk-order.npy",
            "--confirmation_registry", "protocol/registry.json",
            "--campaign_spec", "protocol/campaign-spec.json",
            "--probe_region", "global",
            "--optimizer", "adamw",
            "--wd", str(OPTIMIZER["weight_decay"]),
            "--clip", str(OPTIMIZER["gradient_clip_value"]),
            "--probe_every", "100",
            "--sharp_every", "100000",
            "--sharp_pre_every", "100000",
            "--sharp_full", "0",
            "--val_batches", "1",
            "--ckpt_steps", "",
            "--optimizer_ckpt_steps", ",".join(
                str(step) for step in optimizer_checkpoints
            ),
            "--skip_generation",
            "--terminal_validation",
        ],
        depends_on=(dependency,),
        outputs=(f"work/{name}",),
        device=device,
        predicted_output_bytes=(
            (len(optimizer_checkpoints) + 1) * 260_000_000
        ),
    )


def _finite_edge_nodes(
    trajectory: dict[str, Any],
    step: int,
    stage: str,
    *,
    dependency: str,
) -> list[dict[str, Any]]:
    role = str(trajectory["role"])
    seed = int(trajectory["seed"])
    name = str(trajectory["name"])
    device = _trajectory_device(trajectory)
    key = f"{role}-s{seed}-a{step}"
    checkpoint = f"work/{name}/ckpt_{step}.pt"
    source = f"artifacts/source-{key}.npz"
    certificate = f"artifacts/finite-{key}.json"
    prediction = f"reports/predict-{key}.json"
    successor = f"artifacts/successor-{key}.npz"
    coverage_path = f"reports/coverage-{key}.json"
    retention = f"reports/retention-{key}.json"
    source_id = f"source-{key}"
    finite_id = f"finite-{key}"
    predict_id = f"predict-{key}"
    successor_id = f"successor-{key}"
    coverage_id = f"coverage-{key}"
    retention_id = f"retain-{key}"
    order = f"orders/{role}-seed{seed}-chunk-order.npy"
    common = [
        "--bundle", "{bundle}",
        "--registry", "protocol/registry.json",
        "--tokens", "{tokens}",
        "--checkpoint", checkpoint,
        "--data-order", order,
        "--role", role,
        "--seed", str(seed),
        "--step", str(step),
        "--device", device,
    ]
    retention_argv = [
        "python3", "source/scripts/retain_model_only_checkpoint.py",
        "--checkpoint", checkpoint,
        "--report", retention,
    ]
    retention_outputs = [checkpoint, retention]
    if step == int(trajectory["updates"]):
        final_checkpoint = f"work/{name}/ckpt_final.pt"
        retention_argv.extend(["--also", final_checkpoint])
        retention_outputs.append(final_checkpoint)
    return [
        _node(
            source_id,
            stage,
            [
                "python3",
                "source/experiments/confirm/capture_finite_increment_source.py",
                *common,
                "--output", source,
            ],
            depends_on=(dependency,),
            outputs=(source,),
            device=device,
            predicted_output_bytes=50_000_000,
        ),
        _node(
            finite_id,
            stage,
            [
                "python3",
                "source/experiments/confirm/produce_finite_increment_certificate.py",
                "--source", source,
                "--output", certificate,
            ],
            depends_on=(source_id,),
            outputs=(certificate,),
            predicted_output_bytes=2_000_000,
        ),
        _node(
            predict_id,
            stage,
            [
                "python3",
                "source/experiments/analysis/analyze_finite_increment_confirmation.py",
                "predict", "--certificate", certificate,
                "--output", prediction,
            ],
            depends_on=(finite_id,),
            outputs=(prediction,),
            predicted_output_bytes=2_000_000,
        ),
        _node(
            successor_id,
            stage,
            [
                "python3",
                "source/experiments/confirm/capture_finite_increment_successor.py",
                *common,
                "--prediction", prediction,
                "--output", successor,
            ],
            depends_on=(predict_id,),
            outputs=(successor,),
            device=device,
            predicted_output_bytes=20_000_000,
        ),
        _node(
            coverage_id,
            stage,
            [
                "python3",
                "source/experiments/analysis/analyze_finite_increment_confirmation.py",
                "coverage", "--certificate", certificate,
                "--successor", successor,
                "--prediction", prediction,
                "--output", coverage_path,
                "--require-coverage",
            ],
            depends_on=(successor_id,),
            outputs=(coverage_path,),
            predicted_output_bytes=2_000_000,
        ),
        _node(
            retention_id,
            stage,
            retention_argv,
            depends_on=(coverage_id,),
            outputs=tuple(retention_outputs),
            predicted_output_bytes=90_000_000,
        ),
    ]


def build_plan() -> dict[str, Any]:
    nodes = [_node(
        "qualification-e0",
        "E0",
        [
            "python3", "source/scripts/run_finite_increment_qualification.py",
            "--output", "reports/qualification-e0.json",
            "--require-pass",
        ],
        outputs=("reports/qualification-e0.json",),
        predicted_output_bytes=100_000,
    )]
    qualification_ids = []
    for device in RESOURCE_BUDGET["devices"]:
        suffix = device.replace(":", "")
        identifier = f"qualification-real-{suffix}"
        qualification_ids.append(identifier)
        nodes.append(_node(
            identifier,
            "E1",
            [
                "python3", "source/scripts/run_finite_increment_qualification.py",
                "--output", f"reports/qualification-real-{suffix}.json",
                "--device", device,
                "--real-checkpoint", "{qualification_checkpoint}",
                "--tokens", "{tokens}",
                "--require-pass",
            ],
            depends_on=("qualification-e0",),
            outputs=(f"reports/qualification-real-{suffix}.json",),
            device=device,
            predicted_output_bytes=2_000_000,
        ))
    nodes.append(_node(
        "construction-ready",
        "E1",
        ["python3", "-c", "print('construction-ready')"],
        depends_on=tuple(qualification_ids),
    ))

    construction = TRAJECTORIES[0]
    construction_train = f"train-construction-s{construction['seed']}"
    nodes.append(_training_node(
        construction,
        dependency="construction-ready",
        optimizer_checkpoints=tuple(CONSECUTIVE_EDGES),
    ))
    construction_coverages = []
    construction_retentions = []
    for step in CONSECUTIVE_EDGES:
        stage = "E2" if step == REGISTRY["construction_anchor"] else "E3"
        edge = _finite_edge_nodes(
            construction, step, stage, dependency=construction_train
        )
        nodes.extend(edge)
        construction_coverages.append(edge[4]["id"])
        construction_retentions.append(edge[5]["id"])
    nodes.append(_node(
        "aggregate-consecutive-all-layer",
        "E3",
        [
            "python3",
            "source/experiments/analysis/analyze_finite_increment_confirmation.py",
            "aggregate",
            "--reports-glob", "reports/coverage-construction-*.json",
            "--output", "reports/consecutive-all-layer.json",
        ],
        depends_on=tuple(construction_coverages),
        outputs=("reports/consecutive-all-layer.json",),
        predicted_output_bytes=4_000_000,
    ))
    nodes.append(_node(
        "construction-lock",
        "E3",
        [
            "python3",
            "source/experiments/analysis/analyze_finite_increment_confirmation.py",
            "lock",
            "--aggregate", "reports/consecutive-all-layer.json",
            "--output", "protocol/construction-lock.json",
        ],
        depends_on=("aggregate-consecutive-all-layer",),
        outputs=("protocol/construction-lock.json",),
        predicted_output_bytes=200_000,
    ))

    for trajectory in TRAJECTORIES:
        role = str(trajectory["role"])
        seed = int(trajectory["seed"])
        train_id = f"train-{role}-s{seed}"
        if role == "heldout":
            nodes.append(_training_node(
                trajectory,
                dependency="construction-lock",
                optimizer_checkpoints=tuple(HELDOUT_ANCHORS),
            ))
        timecourse_id = f"diameter-time-course-{role}-s{seed}"
        nodes.append(_node(
            timecourse_id,
            "E4" if role == "construction" else "E5",
            [
                "python3",
                "source/experiments/analysis/analyze_finite_increment_confirmation.py",
                "time-course",
                "--run-root", f"work/{trajectory['name']}",
                "--registry", "protocol/registry.json",
                "--output", f"reports/time-course-{role}-s{seed}.json",
            ],
            depends_on=(train_id,),
            outputs=(f"reports/time-course-{role}-s{seed}.json",),
            predicted_output_bytes=5_000_000,
        ))
        if role == "construction":
            nodes.append(_node(
                "retain-final-construction",
                "E4",
                [
                    "python3", "source/scripts/retain_model_only_checkpoint.py",
                    "--checkpoint",
                    f"work/{trajectory['name']}/ckpt_final.pt",
                    "--report", "reports/retention-final-construction.json",
                ],
                depends_on=(timecourse_id, *construction_retentions),
                outputs=(
                    f"work/{trajectory['name']}/ckpt_final.pt",
                    "reports/retention-final-construction.json",
                ),
                predicted_output_bytes=90_000_000,
            ))
        else:
            for step in HELDOUT_ANCHORS:
                nodes.extend(_finite_edge_nodes(
                    trajectory, step, "E5", dependency=train_id
                ))

    known = {node["id"] for node in nodes}
    if len(known) != len(nodes):
        raise AssertionError("execution-plan node identifiers are not unique")
    if any(set(node["depends_on"]) - known for node in nodes):
        raise AssertionError("execution-plan dependency is unknown")
    value = {
        "schema_version": PLAN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "status": "ready_for_qualification_then_chronological_execution",
        "source_only_prediction": True,
        "successor_replays_same_complete_source": True,
        "training_orders_use_trainer_validator": True,
        "nodes": nodes,
        "node_count": len(nodes),
        "theorem_node_bindings": {
            "finite-source-transport": "source-construction-s7444-a1000",
            "two-variable-plga": "finite-construction-s7444-a1000",
            "source-state-contraction": "predict-construction-s7444-a1000",
            "product-convolution": "aggregate-consecutive-all-layer",
            "heldout-coverage": "coverage-heldout-s8444-a1000",
        },
    }
    value["plan_sha256"] = digest_object(value)
    return value


def _copy_source(output: Path) -> None:
    records = []
    for source in source_import_closure():
        relative = source.relative_to(ROOT)
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append(_file_record(destination, output))
    manifest = {
        "schema_version": "pldr-import-closed-source-manifest-v1",
        "entry_points": list(ENTRY_POINTS),
        "dynamic_import_allowlist": list(DYNAMIC_IMPORTS),
        "files": records,
    }
    manifest["manifest_sha256"] = digest_object(manifest)
    _write_json(output / "source" / "SOURCE_MANIFEST.json", manifest)


def _manifest(output: Path) -> None:
    records = [
        _file_record(path, output)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "MANIFEST.sha256"
    ]
    payload = "".join(
        f"{record['sha256']}  {record['path']}\n" for record in records
    )
    (output / "MANIFEST.sha256").write_text(payload, encoding="ascii")


def stage(output: Path, tokens: Path, tokenizer: Path) -> None:
    if not tokens.is_file() or not tokenizer.is_file():
        raise FileNotFoundError("campaign tokens or tokenizer are missing")
    output.mkdir(parents=True, exist_ok=True)
    registry = create_registry(tokens, tokenizer)
    _write_json(output / "protocol" / "registry.json", registry)
    _write_json(output / "protocol" / "campaign-spec.json", campaign_spec(registry))
    plan = build_plan()
    _write_json(output / "protocol" / "execution-plan.json", plan)
    order_records = []
    for trajectory in TRAJECTORIES:
        role = str(trajectory["role"])
        seed = int(trajectory["seed"])
        value = _order(seed, int(registry["total_chunks"]))
        relative = Path("orders") / f"{role}-seed{seed}-chunk-order.npy"
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("wb") as stream:
            np.save(stream, value, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        validate_data_order(
            np.load(path, allow_pickle=False), int(registry["total_chunks"])
        )
        order_records.append({
            "role": role,
            "seed": seed,
            **_file_record(path, output),
            "minimum_chunk": int(value.min()),
            "maximum_chunk": int(value.max()),
            "trainer_training_limit": int(registry["trainer_training_limit"]),
        })
    _write_json(output / "protocol" / "order-manifest.json", order_records)
    _copy_source(output)
    _manifest(output)


def _tree_digest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_path(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def check(output: Path, tokens: Path, tokenizer: Path) -> None:
    if not output.is_dir():
        raise FileNotFoundError("the staged finite-increment campaign is missing")
    with tempfile.TemporaryDirectory(prefix="pldr-finite-stage-check-") as directory:
        expected = Path(directory) / "bundle"
        stage(expected, tokens, tokenizer)
        observed_digest = _tree_digest(output)
        expected_digest = _tree_digest(expected)
        missing = sorted(set(expected_digest) - set(observed_digest))
        changed = sorted(
            path for path in set(expected_digest) & set(observed_digest)
            if expected_digest[path] != observed_digest[path]
        )
        mutable_prefixes = (
            "artifacts/", "logs/", "reports/", "work/",
        )
        permitted_protocol_outputs = {
            "protocol/construction-lock.json",
        }
        extra = sorted(
            path for path in set(observed_digest) - set(expected_digest)
            if not path.startswith(mutable_prefixes)
            and path not in permitted_protocol_outputs
        )
        if missing or changed or extra:
            raise ValueError(
                f"staged campaign differs: missing={missing}, extra={extra}, "
                f"changed={changed}"
            )
    print(f"finite-increment staging: verified {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if arguments.check:
        check(arguments.output, arguments.tokens, arguments.tokenizer)
    else:
        stage(arguments.output, arguments.tokens, arguments.tokenizer)
        print(f"finite-increment staging: wrote {arguments.output}")


if __name__ == "__main__":
    main()
