#!/usr/bin/env python3
"""Stage the prospective causal row-map confirmation and dependency graph."""

from __future__ import annotations
from companion_paths import legacy_path, data_root, resolve_row_arguments

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from confirm.causal_confirmation_specs import (  # noqa: E402
    ANCHORS,
    ARCHITECTURE,
    CAMPAIGN_ID,
    NUMERICAL_POLICIES,
    OPTIMIZER,
    REGISTRY,
    RESOURCE_BUDGET,
    SCHEMA_VERSION,
    TRAJECTORIES,
)
from confirm.comprehensive_campaign import (  # noqa: E402
    DATASET_SHA256,
    PROBE_RESERVE_CHUNKS,
    TOKENIZER_SHA256,
)
from confirm.comprehensive_gate_shape import (  # noqa: E402
    permuted_chunk_order,
)
from confirm.confirmation_artifacts import (  # noqa: E402
    digest_object,
    sha256_path,
    validate_measurement_registry,
)


DEFAULT_OUTPUT = Path(
    legacy_path('/pldr-data/row/rev40/causal-row-map-confirmation')
)
DEFAULT_TOKENS = data_root('row') / 'inputs/refinedweb_tokens.npy'
DEFAULT_TOKENIZER = ROOT / "experiments" / "data" / "tokenizer.model"
PLAN_SCHEMA = "pldr-causal-row-map-execution-plan-v1"


SOURCE_FILES = (
    "requirements.txt",
    "experiments/train_run.py",
    "experiments/pldr_model_v510.py",
    "experiments/power_law_attention_layer_v510.py",
    "experiments/confirm/confirmation_artifacts.py",
    "experiments/confirm/capture_chronological_confirmation.py",
    "experiments/confirm/capture_causal_source.py",
    "experiments/confirm/causal_confirmation_specs.py",
    "experiments/confirm/assemble_causal_ledger.py",
    "experiments/confirm/attach_causal_successor.py",
    "experiments/confirm/causal_row_map.py",
    "experiments/confirm/postnorm_geometry.py",
    "experiments/confirm/composite_plga.py",
    "experiments/confirm/capture_causal_plga.py",
    "experiments/confirm/capture_chronological_plga.py",
    "experiments/confirm/run_causal_intervention.py",
    "experiments/confirm/run_chronological_intervention.py",
    "experiments/confirm/directional_taylor.py",
    "experiments/confirm/chronological_collapse.py",
    "experiments/confirm/chronological_confirmation_specs.py",
    "experiments/confirm/gate_shape.py",
    "experiments/confirm/validated_interval.py",
    "experiments/confirm/validated_directional_taylor_mixed.py",
    "experiments/analysis/analyze_causal_confirmation.py",
    "experiments/analysis/analyze_causal_plga.py",
    "experiments/analysis/analyze_chronological_plga.py",
    "experiments/analysis/analyze_causal_intervention.py",
    "experiments/analysis/finalize_causal_lock.py",
    "experiments/analysis/finalize_causal_campaign.py",
    "scripts/build_causal_remainder_certificate.py",
    "scripts/run_causal_row_map_qualification.py",
    "scripts/stage_causal_row_map_confirmation.py",
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256_path(path),
        "bytes": path.stat().st_size,
    }


def campaign_spec() -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "status": "prospective_unexecuted",
        "scientific_target": (
            "source-state prediction of finite-registry physical row-map "
            "contraction and re-expansion under executed AdamW"),
        "data": {
            "dataset": "tiiuae/falcon-refinedweb",
            "split": "train",
            "token_dtype": "uint16",
            "packing": "nonoverlapping_contiguous_256_token_chunks",
            "dataset_sha256": DATASET_SHA256,
            "tokenizer_kind": "SentencePiece unigram",
            "tokenizer_vocabulary": 32_000,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "order_generator": "NumPy PCG64",
            "probe_reserve_chunks": PROBE_RESERVE_CHUNKS,
        },
        "architecture": ARCHITECTURE,
        "optimizer": OPTIMIZER,
        "trajectories": list(TRAJECTORIES),
        "registry": {
            **REGISTRY,
            "anchor_steps": list(ANCHORS),
        },
        "numerical_policies": NUMERICAL_POLICIES,
        "resource_budget": RESOURCE_BUDGET,
        "chronology": [
            "seal source checkpoint, executed displacement, JVP, and interval bounds",
            "write source-only physical-row ledger",
            "write source-only maximum prediction",
            "authorize and rerun the one-step successor",
            "attach successor for coverage without changing the prediction",
        ],
        "validated_jet_input_contract": {
            "schema_version": "pldr-causal-validated-jet-bounds-v1",
            "complete_parameter_segment": True,
            "segment_grid": [0.0, 0.5, 1.0],
            "segment_subdivisions": 2,
            "successor_values_present": False,
            "source_state_digest": "sha256 of chronological source NPZ",
            "required_per_row_fields": [
                "point_second_directional_uppers",
                "third_directional_modulus",
                "jvp_error_radius",
                "subdivisions",
            ],
        },
        "decision_rules": {
            "contraction": "maximum_successor_upper_squared < source_diameter_lower_squared",
            "reexpansion": "maximum_successor_lower_squared > source_diameter_upper_squared",
            "unresolved": "otherwise",
            "successor_is_decisive": False,
        },
    }
    value["campaign_sha256"] = digest_object(value)
    return json.loads(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False))


def create_registry(tokens: Path, tokenizer: Path) -> dict[str, Any]:
    data = np.memmap(tokens, dtype=np.uint16, mode="r")
    chunk_count = len(data) // int(REGISTRY["context_length"])
    start = chunk_count - PROBE_RESERVE_CHUNKS
    maximum_training = max(
        int(row["data_offset_chunks"])
        + int(row["updates"]) * int(OPTIMIZER["batch_size"])
        for row in TRAJECTORIES
    )
    if start <= maximum_training or start + 24 > chunk_count:
        raise ValueError("probe reservation overlaps a training trajectory")
    construction = [
        {"id": f"crc{index:03d}", "chunk_index": start + index}
        for index in range(8)
    ]
    validation = [
        {"id": f"crh{index:03d}", "chunk_index": start + 8 + index}
        for index in range(16)
    ]
    units = []
    substream = 4_000_000
    heldout = [row for row in TRAJECTORIES if row["role"] == "heldout"]
    for trajectory in heldout:
        for ordinal in range(16):
            units.append({
                "id": f"cri-s{trajectory['seed']}-{ordinal:02d}",
                "seed": int(trajectory["seed"]),
                "ordinal": ordinal,
                "future_update_start": 0,
                "update_count": int(REGISTRY["intervention_horizons"][
                    ordinal % len(REGISTRY["intervention_horizons"])]),
                "rng_substream": substream,
            })
            substream += 1
    value: dict[str, Any] = {
        "schema_version": "pldr-causal-probe-registry-v1",
        "context_length": int(REGISTRY["context_length"]),
        "construction": construction,
        "validation": validation,
        "anchors": list(ANCHORS),
        "context_head_block_rule": "head-equals-context-index-modulo-four-v1",
        "block_row_rule": "complete-ordered-generator-rows-0-through-63-v1",
        "history_length": int(REGISTRY["history_length"]),
        "intervention_units": units,
        "intervention_unit_rule": (
            "four-anchors-by-four-horizons-per-heldout-seed-v1"),
        "dataset_sha256": sha256_path(tokens),
        "tokenizer_sha256": sha256_path(tokenizer),
    }
    value["registry_sha256"] = digest_object(value)
    validate_measurement_registry(value, require_nonempty=True)
    return value


def _training_command(
    trajectory: dict[str, Any], device: str, tokens: Path,
    tokenizer: Path, registry: Path, order: Path, runs: Path,
) -> list[str]:
    checkpoints = ",".join(map(str, ANCHORS))
    return [
        sys.executable, str(EXPERIMENTS / "train_run.py"),
        "--name", str(trajectory["name"]),
        "--run-id", str(trajectory["name"]),
        "--lineage-root", str(trajectory["name"]),
        "--lr", str(OPTIMIZER["learning_rate"]),
        "--warmup", str(OPTIMIZER["warmup_updates"]), "--const_lr",
        "--steps", str(trajectory["updates"]),
        "--batch", str(OPTIMIZER["batch_size"]),
        "--ctx", str(OPTIMIZER["context_length"]),
        "--layers", str(ARCHITECTURE["layers"]),
        "--heads", str(ARCHITECTURE["heads"]),
        "--dk", str(ARCHITECTURE["head_width"]), "--adff", "170",
        "--seed", str(trajectory["seed"]), "--device", device,
        "--tokens", str(tokens), "--tok_model", str(tokenizer),
        "--outdir", str(runs),
        "--data_offset", str(trajectory["data_offset_chunks"]),
        "--data_order", str(order),
        "--confirmation_registry", str(registry),
        "--campaign_spec", str(registry.parent / "campaign-spec.json"),
        "--probe_region", "global", "--optimizer", "adamw",
        "--wd", str(OPTIMIZER["weight_decay"]),
        "--clip", str(OPTIMIZER["gradient_clip_value"]),
        "--chronological_history_length", str(REGISTRY["history_length"]),
        "--ckpt_steps", checkpoints,
        "--optimizer_ckpt_steps", checkpoints,
        "--skip_generation", "--terminal_validation",
    ]


def _node(
    identifier: str, stage: str, role: str, depends: list[str],
    argv: list[str], outputs: list[Path], device: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "id": identifier,
        "stage": stage,
        "role": role,
        "depends_on": depends,
        "argv": argv,
        "outputs": [str(path) for path in outputs],
    }
    if device is not None:
        value["device"] = device
    return value


def build_plan(
    output: Path, tokens: Path, tokenizer: Path, registry: Path,
) -> dict[str, Any]:
    runs = output / "runs"
    nodes: list[dict[str, Any]] = []
    qualification = output / "reports" / "qualification-report.json"
    nodes.append(_node(
        "qualification", "implementation-qualification", "qualification",
        [],
        [sys.executable,
         str(ROOT / "scripts" / "run_causal_row_map_qualification.py"),
         "--device", "cuda:0", "--width", "64", "--hidden", "170",
         "--output", str(qualification), "--require-pass"],
        [qualification], "cuda:0",
    ))
    construction_prediction_ids = []
    construction_prediction_paths = []
    all_prediction_paths = []
    all_coverage_paths = []
    all_coverage_ids = []
    construction_plga_paths = []
    construction_plga_ids = []
    heldout_plga_paths = []
    heldout_plga_ids = []
    intervention_paths = []
    intervention_ids = []
    lock_path = output / "protocol" / "construction-lock.json"
    for trajectory_index, trajectory in enumerate(TRAJECTORIES):
        role = str(trajectory["role"])
        seed = int(trajectory["seed"])
        name = str(trajectory["name"])
        device = RESOURCE_BUDGET["devices"][trajectory_index % 2]
        train_id = f"train-{name}"
        order = output / "orders" / f"{role}-seed{seed}-chunk-order.npy"
        train_dependencies = ["qualification"]
        if role == "heldout":
            train_dependencies.append("construction-lock")
        nodes.append(_node(
            train_id, "training", role, train_dependencies,
            _training_command(
                trajectory, device, tokens, tokenizer, registry, order, runs),
            [runs / name], device,
        ))
        capture_dependencies = [train_id]
        if role == "heldout":
            capture_dependencies.append("construction-lock")
        for layer in range(int(ARCHITECTURE["layers"])):
            for anchor in ANCHORS:
                stem = f"{role}-s{seed}-a{anchor}-l{layer}"
                checkpoint = runs / name / f"ckpt_{anchor}.pt"
                chrono_source = output / "raw" / "source" / f"{stem}.npz"
                jet_bounds = output / "raw" / "validated-jets" / f"{stem}.json"
                certificate = output / "certificates" / f"{stem}.json"
                causal_source = output / "raw" / "causal" / f"{stem}-source.npz"
                prediction = output / "reports" / "prediction" / f"{stem}.json"
                chrono_successor = output / "raw" / "successor" / f"{stem}.npz"
                causal_covered = output / "raw" / "causal" / f"{stem}-covered.npz"
                coverage = output / "reports" / "coverage" / f"{stem}.json"
                source_id = f"source-{stem}"
                certificate_id = f"certificate-{stem}"
                assembly_id = f"assemble-{stem}"
                prediction_id = f"predict-{stem}"
                successor_id = f"successor-{stem}"
                attach_id = f"attach-{stem}"
                lock_args = ["--lock", str(lock_path)] if role == "heldout" else []
                source_argv = [
                    sys.executable,
                    str(EXPERIMENTS / "confirm" / "capture_causal_source.py"),
                    "--checkpoint", str(checkpoint),
                    "--registry", str(registry), "--tokens", str(tokens),
                    "--layer", str(layer), "--updates", "1",
                    "--role", role, "--seed", str(seed), "--device", device,
                    "--defer-successor", "--output", str(chrono_source),
                    *lock_args,
                ]
                nodes.append(_node(
                    source_id, "source-capture", role,
                    list(capture_dependencies), source_argv,
                    [chrono_source], device,
                ))
                nodes.append({
                    "id": f"validated-jets-{stem}",
                    "stage": "validated-interval-input",
                    "role": role,
                    "depends_on": [source_id],
                    "input_contract": (
                        "pldr-causal-validated-jet-bounds-v1; exact rational "
                        "bounds on the complete parameter segment; outward JVP-error radius; no successor"),
                    "source_state_digest": f"{{sha256:{chrono_source}}}",
                    "outputs": [str(jet_bounds)],
                })
                nodes.append(_node(
                    certificate_id, "remainder-certificate", role,
                    [source_id, f"validated-jets-{stem}"],
                    [sys.executable, str(ROOT / "scripts"
                     / "build_causal_remainder_certificate.py"),
                     str(jet_bounds), "--output", str(certificate)],
                    [certificate],
                ))
                nodes.append(_node(
                    assembly_id, "source-ledger", role,
                    [source_id, certificate_id],
                    [sys.executable, str(EXPERIMENTS / "confirm"
                     / "assemble_causal_ledger.py"), str(chrono_source),
                     str(certificate), "--step-index", "0", "--output",
                     str(causal_source)], [causal_source],
                ))
                nodes.append(_node(
                    prediction_id, "source-prediction", role, [assembly_id],
                    [sys.executable, str(EXPERIMENTS / "analysis"
                     / "analyze_causal_confirmation.py"), str(causal_source),
                     "--pair-chunk-size", str(REGISTRY["pair_chunk_size"]),
                     "--output", str(prediction), "--require-pass"],
                    [prediction],
                ))
                all_prediction_paths.append(prediction)
                successor_argv = [
                    sys.executable,
                    str(EXPERIMENTS / "confirm" / "capture_causal_source.py"),
                    "--checkpoint", str(checkpoint),
                    "--registry", str(registry), "--tokens", str(tokens),
                    "--layer", str(layer), "--updates", "1",
                    "--role", role, "--seed", str(seed), "--device", device,
                    "--prediction-report", str(prediction),
                    "--expected-source-sha256", f"{{sha256:{causal_source}}}",
                    "--output", str(chrono_successor), *lock_args,
                ]
                nodes.append(_node(
                    successor_id, "successor-capture", role, [prediction_id],
                    successor_argv, [chrono_successor], device,
                ))
                nodes.append(_node(
                    attach_id, "successor-attachment", role,
                    [prediction_id, successor_id],
                    [sys.executable, str(EXPERIMENTS / "confirm"
                     / "attach_causal_successor.py"), str(causal_source),
                     str(prediction), str(chrono_successor), "--output",
                     str(causal_covered)], [causal_covered],
                ))
                nodes.append(_node(
                    f"coverage-{stem}", "coverage-analysis", role, [attach_id],
                    [sys.executable, str(EXPERIMENTS / "analysis"
                     / "analyze_causal_confirmation.py"), str(causal_covered),
                     "--pair-chunk-size", str(REGISTRY["pair_chunk_size"]),
                     "--output", str(coverage), "--require-pass"], [coverage],
                ))
                all_coverage_paths.append(coverage)
                all_coverage_ids.append(f"coverage-{stem}")
                if role in {"development", "construction"}:
                    construction_prediction_ids.append(prediction_id)
                    construction_prediction_paths.append(prediction)
        if role in {"construction", "heldout"}:
            for layer in range(int(ARCHITECTURE["layers"])):
                for anchor in ANCHORS:
                    stem = f"plga-{role}-s{seed}-a{anchor}-l{layer}"
                    raw = output / "raw" / "plga" / f"{stem}.npz"
                    argv = [
                        sys.executable, str(EXPERIMENTS / "confirm"
                        / "capture_causal_plga.py"),
                        "--checkpoint", str(runs / name / f"ckpt_{anchor}.pt"),
                        "--registry", str(registry), "--tokens", str(tokens),
                        "--role", role, "--seed", str(seed),
                        "--layer", str(layer), "--device", device,
                        "--output", str(raw),
                    ]
                    if role == "heldout":
                        argv.extend(["--lock", str(lock_path)])
                    nodes.append(_node(
                        stem, "composite-plga", role,
                        list(capture_dependencies), argv, [raw], device,
                    ))
                    if role == "construction":
                        construction_plga_paths.append(raw)
                        construction_plga_ids.append(stem)
                    else:
                        heldout_plga_paths.append(raw)
                        heldout_plga_ids.append(stem)
        if role == "heldout":
            for ordinal in range(16):
                anchor = REGISTRY["intervention_anchor_steps"][ordinal // 4]
                unit = f"cri-s{seed}-{ordinal:02d}"
                raw = output / "raw" / "intervention" / f"{unit}.npz"
                nodes.append(_node(
                    f"intervene-{unit}", "complete-source-removal", role,
                    list(capture_dependencies),
                    [sys.executable, str(EXPERIMENTS / "confirm"
                     / "run_causal_intervention.py"),
                     "--checkpoint", str(runs / name / f"ckpt_{anchor}.pt"),
                     "--registry", str(registry), "--tokens", str(tokens),
                     "--lock", str(lock_path), "--unit-id", unit,
                     "--seed", str(seed), "--device", device,
                     "--output", str(raw)], [raw], device,
                ))
                intervention_paths.append(raw)
                intervention_ids.append(f"intervene-{unit}")
    construction_plga_report = (
        output / "reports" / "construction-plga.json")
    heldout_plga_report = output / "reports" / "heldout-plga.json"
    intervention_report = output / "reports" / "heldout-intervention.json"
    final_report = output / "reports" / "final-report.json"
    nodes.append(_node(
        "analyze-construction-plga", "plga-analysis", "construction",
        construction_plga_ids,
        [sys.executable, str(EXPERIMENTS / "analysis"
         / "analyze_causal_plga.py"),
         *map(str, construction_plga_paths),
         "--output", str(construction_plga_report), "--require-pass"],
        [construction_plga_report],
    ))
    nodes.append(_node(
        "construction-lock", "construction-lock", "construction",
        construction_prediction_ids + ["analyze-construction-plga"],
        [sys.executable, str(EXPERIMENTS / "analysis"
         / "finalize_causal_lock.py"),
         *map(str, construction_prediction_paths),
         "--plga-report", str(construction_plga_report),
         "--require-complete-grid",
         "--output", str(lock_path)], [lock_path],
    ))
    nodes.append(_node(
        "analyze-heldout-plga", "plga-analysis", "heldout",
        heldout_plga_ids,
        [sys.executable, str(EXPERIMENTS / "analysis"
         / "analyze_causal_plga.py"),
         *map(str, heldout_plga_paths), "--lock", str(lock_path),
         "--output", str(heldout_plga_report), "--require-pass"],
        [heldout_plga_report],
    ))
    nodes.append(_node(
        "analyze-heldout-intervention", "intervention-analysis", "heldout",
        intervention_ids,
        [sys.executable, str(EXPERIMENTS / "analysis"
         / "analyze_causal_intervention.py"),
         *map(str, intervention_paths), "--lock", str(lock_path),
         "--output", str(intervention_report), "--require-pass"],
        [intervention_report],
    ))
    nodes.append(_node(
        "finalize-causal-campaign", "campaign-analysis", "heldout",
        all_coverage_ids + [
            "qualification",
            "analyze-construction-plga", "analyze-heldout-plga",
            "analyze-heldout-intervention",
        ],
        [sys.executable, str(EXPERIMENTS / "analysis"
         / "finalize_causal_campaign.py"),
         "--predictions", *map(str, all_prediction_paths),
         "--coverage", *map(str, all_coverage_paths),
         "--construction-plga", str(construction_plga_report),
         "--heldout-plga", str(heldout_plga_report),
         "--intervention", str(intervention_report),
         "--lock", str(lock_path),
         "--qualification", str(qualification),
         "--output", str(final_report), "--require-pass"],
        [final_report],
    ))
    ids = [node["id"] for node in nodes]
    if len(ids) != len(set(ids)):
        raise AssertionError("causal execution-plan node ids are not unique")
    known = set(ids)
    if any(set(node["depends_on"]) - known for node in nodes):
        raise AssertionError("causal execution plan has an unknown dependency")
    value: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "status": "ready_pending_validated_interval_inputs",
        "devices": list(RESOURCE_BUDGET["devices"]),
        "strict_dependency_rule": (
            "every successor-capture node depends on its source-prediction node"),
        "construction_lock_precedes_heldout_training": True,
        "nodes": nodes,
    }
    value["plan_sha256"] = digest_object(value)
    return value


def _write_orders(output: Path, total_chunks: int, reserved: np.ndarray) -> list[dict[str, Any]]:
    manifest = []
    for trajectory in TRAJECTORIES:
        role = str(trajectory["role"])
        seed = int(trajectory["seed"])
        generated = permuted_chunk_order(
            total_chunks, seed, reserved_chunks=reserved)
        path = output / "orders" / f"{role}-seed{seed}-chunk-order.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, generated["order"], allow_pickle=False)
        manifest.append({
            "role": role, "seed": seed, "path": str(path.relative_to(output)),
            "array_sha256": generated["sha256"],
            "file_sha256": sha256_path(path),
            "chunk_count": generated["chunk_count"],
        })
    return manifest


def stage(output: Path, tokens: Path, tokenizer: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if sha256_path(tokens) != DATASET_SHA256:
        raise ValueError("token archive digest differs from the campaign")
    if sha256_path(tokenizer) != TOKENIZER_SHA256:
        raise ValueError("tokenizer digest differs from the campaign")
    registry_value = create_registry(tokens, tokenizer)
    registry_path = output / "protocol" / "registry.json"
    _write_json(output / "protocol" / "campaign-spec.json", campaign_spec())
    _write_json(registry_path, registry_value)
    data = np.memmap(tokens, dtype=np.uint16, mode="r")
    total_chunks = len(data) // int(REGISTRY["context_length"])
    reserved = np.asarray([
        row["chunk_index"]
        for row in registry_value["construction"] + registry_value["validation"]
    ], dtype=np.int64)
    order_manifest = _write_orders(output, total_chunks, reserved)
    _write_json(output / "protocol" / "order-manifest.json", order_manifest)
    plan = build_plan(output, tokens, tokenizer, registry_path)
    _write_json(output / "protocol" / "execution-plan.json", plan)
    source_records = []
    for relative in SOURCE_FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"campaign source is missing: {relative}")
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_records.append(_file_record(destination, output))
    _write_json(output / "source" / "SOURCE_MANIFEST.json", source_records)
    qualification = output / "reports" / "qualification-report.json"
    _write_json(output / "protocol" / "qualification-command.json", {
        "argv": [
            sys.executable,
            str(ROOT / "scripts" / "run_causal_row_map_qualification.py"),
            "--device", "cuda:0", "--width", "64", "--hidden", "170",
            "--output", str(qualification), "--require-pass",
        ],
        "output": str(qualification),
    })
    records = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.sha256":
            records.append(_file_record(path, output))
    manifest = "".join(
        f"{record['sha256']}  {record['path']}\n" for record in records)
    (output / "MANIFEST.sha256").write_text(manifest, encoding="ascii")


def check(
    output: Path, tokens: Path, tokenizer: Path,
) -> None:
    required = (
        output / "protocol" / "campaign-spec.json",
        output / "protocol" / "registry.json",
        output / "protocol" / "execution-plan.json",
        output / "protocol" / "order-manifest.json",
        output / "source" / "SOURCE_MANIFEST.json",
        output / "MANIFEST.sha256",
    )
    if any(path.is_symlink() or not path.is_file() for path in required):
        raise ValueError("causal campaign staging is incomplete")
    for path in output.rglob("*"):
        if path.is_symlink():
            raise ValueError("causal campaign staging contains a symlink")

    spec = json.loads(required[0].read_text(encoding="utf-8"))
    registry = json.loads(required[1].read_text(encoding="utf-8"))
    plan = json.loads(required[2].read_text(encoding="utf-8"))
    orders = json.loads(required[3].read_text(encoding="utf-8"))
    source_records = json.loads(required[4].read_text(encoding="utf-8"))
    if spec != campaign_spec():
        raise ValueError("causal campaign specification is stale")
    expected_registry = create_registry(tokens, tokenizer)
    if registry != expected_registry:
        raise ValueError("causal measurement registry is stale")
    if (
        registry.get("dataset_sha256") != spec["data"]["dataset_sha256"]
        or registry.get("tokenizer_sha256")
        != spec["data"]["tokenizer_sha256"]
    ):
        raise ValueError("causal registry digests differ from the specification")
    validate_measurement_registry(registry, require_nonempty=True)
    expected_plan = build_plan(output, tokens, tokenizer, required[1])
    if plan != expected_plan:
        raise ValueError("causal execution plan is stale")

    contexts = registry["construction"] + registry["validation"]
    reserved = np.asarray(
        [row["chunk_index"] for row in contexts], dtype=np.int64)
    data = np.memmap(tokens, dtype=np.uint16, mode="r")
    total_chunks = len(data) // int(REGISTRY["context_length"])
    if not isinstance(orders, list) or len(orders) != len(TRAJECTORIES):
        raise ValueError("causal order manifest has an invalid count")
    for record, trajectory in zip(orders, TRAJECTORIES, strict=True):
        role = str(trajectory["role"])
        seed = int(trajectory["seed"])
        relative = Path(
            "orders") / f"{role}-seed{seed}-chunk-order.npy"
        path = output / relative
        if (
            not isinstance(record, dict)
            or set(record) != {
                "role", "seed", "path", "array_sha256",
                "file_sha256", "chunk_count",
            }
            or record["role"] != role
            or int(record["seed"]) != seed
            or record["path"] != str(relative)
            or not path.is_file()
            or sha256_path(path) != record["file_sha256"]
        ):
            raise ValueError("causal order manifest record is invalid")
        order = np.load(path, allow_pickle=False)
        regenerated = permuted_chunk_order(
            total_chunks, seed, reserved_chunks=reserved)
        if (
            order.dtype != np.dtype(np.int64)
            or order.ndim != 1
            or int(record["chunk_count"]) != len(order)
            or record["array_sha256"] != regenerated["sha256"]
            or not np.array_equal(order, regenerated["order"])
        ):
            raise ValueError("causal PCG64 order does not replay")

    if not isinstance(source_records, list) or len(source_records) != len(
        SOURCE_FILES
    ):
        raise ValueError("causal source snapshot has an invalid count")
    for relative, record in zip(SOURCE_FILES, source_records, strict=True):
        source = ROOT / relative
        staged = output / "source" / relative
        expected = _file_record(staged, output)
        if (
            not source.is_file()
            or not staged.is_file()
            or source.read_bytes() != staged.read_bytes()
            or record != expected
        ):
            raise ValueError(f"staged campaign source drifted: {relative}")

    records = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path != required[5]:
            records.append(_file_record(path, output))
    expected_manifest = "".join(
        f"{record['sha256']}  {record['path']}\n"
        for record in records
    )
    if required[5].read_text(encoding="ascii") != expected_manifest:
        raise ValueError("causal campaign manifest is not the exact tree")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--tokens", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Report inputs and outputs without acquisition")
    arguments = parser.parse_args()
    if not resolve_row_arguments(arguments, parser, defaults={'tokens': DEFAULT_TOKENS, 'tokenizer': DEFAULT_TOKENIZER},
            identities={'tokens': globals().get('DATASET_SHA256'), 'tokenizer': globals().get('TOKENIZER_SHA256')}):
        return
    output = Path(arguments.output).resolve()
    if arguments.check:
        check(
            output, Path(arguments.tokens).resolve(),
            Path(arguments.tokenizer).resolve())
        print(f"causal row-map stage: verified {output}")
        return
    stage(
        output, Path(arguments.tokens).resolve(),
        Path(arguments.tokenizer).resolve())
    print(f"causal row-map stage: wrote {output}")


if __name__ == "__main__":
    main()
