#!/usr/bin/env python3
"""Stage the prospective Rev41 contrast-energy confirmation campaign."""

from __future__ import annotations
from companion_paths import legacy_path, data_root, resolve_row_arguments

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from confirm.comprehensive_campaign import (  # noqa: E402
    DATASET_SHA256,
    PROBE_RESERVE_CHUNKS,
    TOKENIZER_SHA256,
)
from confirm.comprehensive_gate_shape import permuted_chunk_order  # noqa: E402
from confirm.confirmation_artifacts import (  # noqa: E402
    digest_object,
    sha256_path,
    validate_measurement_registry,
)
from confirm.contrast_energy_specs import (  # noqa: E402
    ANCHORS,
    ARCHITECTURE,
    CAMPAIGN_ID,
    NUMERICAL_POLICIES,
    OPTIMIZER,
    REGISTRY,
    RESOURCE_BUDGET,
    SCHEMA_VERSION,
    TAYLOR_POLICY,
    TRAJECTORIES,
)


DEFAULT_OUTPUT = Path(
    legacy_path('/pldr-data/row/rev41/contrast-energy-cocycle-confirmation')
)
DEFAULT_TOKENS = data_root('row') / 'inputs/refinedweb_tokens.npy'
DEFAULT_TOKENIZER = ROOT / "experiments" / "data" / "tokenizer.model"
PLAN_SCHEMA = "pldr-contrast-energy-execution-plan-v1"

SOURCE_FILES = (
    "requirements.txt",
    "experiments/train_run.py",
    "experiments/pldr_model_v510.py",
    "experiments/power_law_attention_layer_v510.py",
    "experiments/optimizer_ledger.py",
    "experiments/confirm/confirmation_artifacts.py",
    "experiments/confirm/chronological_history.py",
    "experiments/confirm/chronological_collapse.py",
    "experiments/confirm/capture_chronological_confirmation.py",
    "experiments/confirm/capture_contrast_energy_source.py",
    "experiments/confirm/attach_contrast_energy_successor.py",
    "experiments/confirm/contrast_energy_specs.py",
    "experiments/confirm/contrast_energy.py",
    "experiments/confirm/explicit_layernorm.py",
    "experiments/confirm/produce_validated_contrast_energy_jets.py",
    "experiments/confirm/validated_interval.py",
    "experiments/confirm/validated_directional_taylor_mixed.py",
    "experiments/confirm/validated_row_map.py",
    "experiments/confirm/composite_plga.py",
    "experiments/analysis/analyze_contrast_energy_confirmation.py",
    "experiments/analysis/aggregate_diameter_cocycle.py",
    "scripts/finalize_contrast_energy_lock.py",
    "scripts/finalize_contrast_energy_resources.py",
    "scripts/retain_model_only_checkpoint.py",
    "scripts/run_contrast_energy_qualification.py",
    "scripts/stage_contrast_energy_confirmation.py",
    "scripts/execute_contrast_energy_plan.py",
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False
    ) + "\n", encoding="utf-8")
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
            "source-only signed contrast-energy bounds and chronological "
            "diameter multipliers for all three PLDR row-map layers"
        ),
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
        "registry": {**REGISTRY, "anchor_steps": list(ANCHORS),
                     "layers": list(REGISTRY["layers"])},
        "numerical_policies": NUMERICAL_POLICIES,
        "taylor_policy": TAYLOR_POLICY,
        "resource_budget": RESOURCE_BUDGET,
        "stages": {
            "A": "committed analytic preflight without campaign authorization",
            "B": "first real-path all-layer anchor on each registered device",
            "C": "102 development and construction layer edges",
            "D": "204 locked held-out layer edges",
            "E": (
                "separate planned occupied PLGA and causal-score extension "
                "outside the core graph"
            ),
        },
        "chronology": [
            "seal checkpoint optimizer state data order cursor and next batch",
            "capture source rows native displacement and complete parameter JVP",
            "materialize validated third-order Taylor models without a successor",
            "freeze direct pair-energy prediction",
            "authorize native successor replay",
            "attach successor only for coverage",
            "aggregate ordered lower and upper diameter multipliers",
        ],
        "decision_rules": {
            "contraction": (
                "maximum_successor_upper_squared < "
                "maximum_source_lower_squared"
            ),
            "reexpansion": (
                "maximum_successor_lower_squared > "
                "maximum_source_upper_squared"
            ),
            "unresolved": "otherwise",
            "successor_is_decisive": False,
        },
        "taylor_model_contract": {
            "schema_version": "pldr-contrast-energy-taylor-model-v1",
            "producer_must_be_executable": True,
            "materialization_status": (
                "required_external_artifact_not_generated_by_core_plan"
            ),
            "source_sha256_required": True,
            "taylor_policy_sha256_required": True,
            "backend_manifest_sha256_required": True,
            "source_displacement_endpoint_digests_required": True,
            "successor_values_present": False,
            "endpoint_second_derivative_intervals": True,
            "positive_cellwise_third_modulus": True,
            "explicit_layernorm": True,
            "rope_interval_enclosure": True,
        },
    }
    value["spec_sha256"] = digest_object(value)
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
    value: dict[str, Any] = {
        "schema_version": "pldr-contrast-energy-registry-v1",
        "context_length": int(REGISTRY["context_length"]),
        "construction": [
            {"id": f"cec{index:03d}", "chunk_index": start + index}
            for index in range(8)
        ],
        "validation": [
            {"id": f"ceh{index:03d}", "chunk_index": start + 8 + index}
            for index in range(16)
        ],
        "anchors": list(ANCHORS),
        "context_head_block_rule": (
            "head-equals-context-index-modulo-four-v1"),
        "block_row_rule": (
            "complete-ordered-generator-rows-0-through-63-v1"),
        "history_length": int(REGISTRY["history_length"]),
        "pair_rule": "all-unordered-pairs-lexicographic-v1",
        "layers": list(REGISTRY["layers"]),
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
    lock_path = output / "protocol" / "construction-lock.json"
    resource_lock_path = output / "protocol" / "stage-b-resource-lock.json"
    nodes: list[dict[str, Any]] = []
    required_taylor_models: list[dict[str, Any]] = []
    qualification = output / "reports" / "qualification-report.json"
    nodes.append(_node(
        "qualification", "A", "qualification", [],
        [sys.executable, str(ROOT / "scripts" /
         "run_contrast_energy_qualification.py"),
         "--device", "cpu", "--output", str(qualification),
         "--require-pass"],
        [qualification], "cpu",
    ))
    construction_coverage_ids: list[str] = []
    construction_reports: list[Path] = []
    prediction_ids: list[str] = []
    prediction_paths: list[Path] = []
    coverage_ids: list[str] = []
    stage_b_coverage_ids: list[str] = []
    stage_b_resource_node_ids: list[str] = []
    for trajectory_index, trajectory in enumerate(TRAJECTORIES):
        role = str(trajectory["role"])
        seed = int(trajectory["seed"])
        name = str(trajectory["name"])
        device = RESOURCE_BUDGET["devices"][trajectory_index % 2]
        order = output / "orders" / f"{role}-seed{seed}-chunk-order.npy"
        train_id = f"train-{name}"
        train_dependencies = ["qualification"]
        if role == "heldout":
            train_dependencies.append("construction-lock")
        nodes.append(_node(
            train_id, "C" if role != "heldout" else "D", role,
            train_dependencies,
            _training_command(
                trajectory, device, tokens, tokenizer, registry, order, runs),
            [runs / name], device,
        ))
        coverage_by_anchor = {anchor: [] for anchor in ANCHORS}
        for layer in REGISTRY["layers"]:
            for anchor in ANCHORS:
                stem = f"{role}-s{seed}-a{anchor}-l{layer}"
                checkpoint = runs / name / f"ckpt_{anchor}.pt"
                source = output / "raw" / "source" / f"{stem}.npz"
                taylor = output / "raw" / "taylor-models" / f"{stem}.npz"
                jets = output / "certificates" / f"{stem}.npz"
                prediction = output / "reports" / "prediction" / f"{stem}.json"
                successor = output / "raw" / "successor" / f"{stem}.npz"
                attachment = output / "raw" / "coverage" / f"{stem}.npz"
                coverage = output / "reports" / "coverage" / f"{stem}.json"
                source_id = f"source-{stem}"
                jet_id = f"validated-jets-{stem}"
                prediction_id = f"predict-{stem}"
                successor_id = f"successor-{stem}"
                attachment_id = f"attach-{stem}"
                lock_args = (
                    ["--lock", str(lock_path)] if role == "heldout" else [])
                edge_stage = (
                    "B"
                    if role in {"development", "construction"}
                    and anchor == ANCHORS[0]
                    else "C" if role != "heldout" else "D"
                )
                source_dependencies = [train_id]
                if edge_stage == "C":
                    source_dependencies.append("stage-b-resource-lock")
                nodes.append(_node(
                    source_id, edge_stage, role,
                    source_dependencies,
                    [sys.executable, str(EXPERIMENTS / "confirm" /
                     "capture_contrast_energy_source.py"),
                     "--checkpoint", str(checkpoint),
                     "--registry", str(registry), "--tokens", str(tokens),
                     "--data-order", str(order), "--layer", str(layer),
                     "--updates", "1", "--role", role,
                     "--seed", str(seed), "--device", device,
                     "--defer-successor", "--output", str(source),
                     *lock_args],
                    [source], device,
                ))
                required_taylor_models.append({
                    "path": str(taylor),
                    "schema_version": "pldr-contrast-energy-taylor-model-v1",
                    "source_node": source_id,
                    "source_sha256_binding": f"{{sha256:{source}}}",
                    "successor_values_present": False,
                    "producer_contract": "validated-third-order-taylor-v1",
                    "taylor_policy_sha256": digest_object(TAYLOR_POLICY),
                    "backend_manifest_sha256_required": True,
                })
                nodes.append(_node(
                    jet_id, edge_stage, role,
                    [source_id],
                    [sys.executable, str(EXPERIMENTS / "confirm" /
                     "produce_validated_contrast_energy_jets.py"),
                     "--source", str(source), "--taylor-model", str(taylor),
                     "--output", str(jets), *lock_args],
                    [jets], device,
                ))
                nodes.append(_node(
                    prediction_id, edge_stage, role,
                    [jet_id],
                    [sys.executable, str(EXPERIMENTS / "analysis" /
                     "analyze_contrast_energy_confirmation.py"), str(jets),
                     "--output", str(prediction)],
                    [prediction], "cpu",
                ))
                nodes.append(_node(
                    successor_id, edge_stage, role,
                    [prediction_id],
                    [sys.executable, str(EXPERIMENTS / "confirm" /
                     "capture_contrast_energy_source.py"),
                     "--checkpoint", str(checkpoint),
                     "--registry", str(registry), "--tokens", str(tokens),
                     "--data-order", str(order), "--layer", str(layer),
                     "--updates", "1", "--role", role,
                     "--seed", str(seed), "--device", device,
                     "--prediction-report", str(prediction),
                     "--expected-source-sha256", f"{{sha256:{source}}}",
                     "--output", str(successor), *lock_args],
                    [successor], device,
                ))
                nodes.append(_node(
                    attachment_id, edge_stage, role,
                    [prediction_id, successor_id],
                    [sys.executable, str(EXPERIMENTS / "confirm" /
                     "attach_contrast_energy_successor.py"),
                     str(source), str(prediction), str(successor),
                     "--output", str(attachment)],
                    [attachment], "cpu",
                ))
                coverage_id = f"coverage-{stem}"
                nodes.append(_node(
                    coverage_id, edge_stage, role,
                    [attachment_id],
                    [sys.executable, str(EXPERIMENTS / "analysis" /
                     "analyze_contrast_energy_confirmation.py"), str(jets),
                     "--successor", str(attachment),
                     "--output", str(coverage), "--require-coverage"],
                    [coverage], "cpu",
                ))
                prediction_ids.append(prediction_id)
                prediction_paths.append(prediction)
                coverage_ids.append(coverage_id)
                coverage_by_anchor[anchor].append(coverage_id)
                if edge_stage == "B":
                    stage_b_coverage_ids.append(coverage_id)
                    stage_b_resource_node_ids.extend(
                        [source_id, successor_id])
                if role in {"development", "construction"}:
                    construction_coverage_ids.append(coverage_id)
                    construction_reports.append(coverage)
        for anchor in ANCHORS:
            checkpoint = runs / name / f"ckpt_{anchor}.pt"
            retention = (
                output / "reports" / "retention"
                / f"{role}-s{seed}-a{anchor}.json"
            )
            retention_argv = [
                sys.executable,
                str(ROOT / "scripts" / "retain_model_only_checkpoint.py"),
                "--checkpoint", str(checkpoint),
                "--report", str(retention),
            ]
            retention_outputs = [checkpoint, retention]
            if anchor == ANCHORS[-1]:
                final_checkpoint = runs / name / "ckpt_final.pt"
                retention_argv.extend(["--also", str(final_checkpoint)])
                retention_outputs.append(final_checkpoint)
            retention_stage = (
                "B"
                if role in {"development", "construction"}
                and anchor == ANCHORS[0]
                else "C" if role != "heldout" else "D"
            )
            nodes.append(_node(
                f"retain-{role}-s{seed}-a{anchor}",
                retention_stage,
                role,
                coverage_by_anchor[anchor],
                retention_argv,
                retention_outputs,
                "cpu",
            ))
    nodes.append(_node(
        "stage-b-resource-lock", "B", "qualification",
        stage_b_coverage_ids,
        [sys.executable, str(ROOT / "scripts" /
         "finalize_contrast_energy_resources.py"),
         "--log-dir", str(output / "logs"),
         *sum((["--node-id", identifier]
               for identifier in stage_b_resource_node_ids), []),
         "--provisional-peak-mib",
         str(int(RESOURCE_BUDGET["provisional_peak_gib_each"] * 1024)),
         "--enlargement", str(RESOURCE_BUDGET["stage_b_enlargement"]),
         "--output", str(resource_lock_path), "--require-pass"],
        [resource_lock_path], "cpu",
    ))
    nodes.append(_node(
        "construction-lock", "C", "construction",
        ["stage-b-resource-lock", *construction_coverage_ids],
        [sys.executable, str(ROOT / "scripts" /
         "finalize_contrast_energy_lock.py"),
         "--campaign-spec", str(registry.parent / "campaign-spec.json"),
         "--resource-lock", str(resource_lock_path),
         "--reports", *map(str, construction_reports),
         "--output", str(lock_path), "--require-pass"],
        [lock_path], "cpu",
    ))
    aggregate = output / "reports" / "diameter-cocycle.json"
    nodes.append(_node(
        "aggregate-diameter-cocycle", "D", "all",
        prediction_ids,
        [sys.executable, str(EXPERIMENTS / "analysis" /
         "aggregate_diameter_cocycle.py"),
         *map(str, prediction_paths), "--output", str(aggregate)],
        [aggregate], "cpu",
    ))
    ids = [node["id"] for node in nodes]
    if len(ids) != len(set(ids)):
        raise AssertionError("execution-plan node ids are not unique")
    known = set(ids)
    if any(set(node["depends_on"]) - known for node in nodes):
        raise AssertionError("execution plan has an unknown dependency")
    if any(not node.get("argv") for node in nodes):
        raise AssertionError("every execution-plan node needs an argv vector")
    value: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "status": "ready_pending_validated_taylor_models",
        "devices": list(RESOURCE_BUDGET["devices"]),
        "source_only_prediction": True,
        "successor_dependency_rule": (
            "each successor node depends on its immutable prediction node"),
        "construction_lock_precedes_heldout_training": True,
        "required_taylor_models": required_taylor_models,
        "nodes": nodes,
    }
    value["plan_sha256"] = digest_object(value)
    return value


def _write_orders(
    output: Path, total_chunks: int, reserved: np.ndarray,
) -> list[dict[str, Any]]:
    manifest = []
    for trajectory in TRAJECTORIES:
        role = str(trajectory["role"])
        seed = int(trajectory["seed"])
        generated = permuted_chunk_order(
            total_chunks, seed, reserved_chunks=reserved)
        relative = Path("orders") / f"{role}-seed{seed}-chunk-order.npy"
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.save(stream, generated["order"], allow_pickle=False)
        manifest.append({
            "role": role,
            "seed": seed,
            "path": str(relative),
            "array_sha256": generated["sha256"],
            "file_sha256": sha256_path(path),
            "chunk_count": len(generated["order"]),
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
        row["chunk_index"] for row in
        registry_value["construction"] + registry_value["validation"]
    ], dtype=np.int64)
    orders = _write_orders(output, total_chunks, reserved)
    _write_json(output / "protocol" / "order-manifest.json", orders)
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
            str(ROOT / "scripts" / "run_contrast_energy_qualification.py"),
            "--device", "cpu", "--output", str(qualification),
            "--require-pass",
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


def check(output: Path, tokens: Path, tokenizer: Path) -> None:
    required = (
        output / "protocol" / "campaign-spec.json",
        output / "protocol" / "registry.json",
        output / "protocol" / "execution-plan.json",
        output / "protocol" / "order-manifest.json",
        output / "source" / "SOURCE_MANIFEST.json",
        output / "MANIFEST.sha256",
    )
    if any(path.is_symlink() or not path.is_file() for path in required):
        raise ValueError("contrast-energy campaign staging is incomplete")
    if any(path.is_symlink() for path in output.rglob("*")):
        raise ValueError("contrast-energy campaign staging contains a symlink")
    spec = json.loads(required[0].read_text(encoding="utf-8"))
    registry = json.loads(required[1].read_text(encoding="utf-8"))
    plan = json.loads(required[2].read_text(encoding="utf-8"))
    orders = json.loads(required[3].read_text(encoding="utf-8"))
    source_records = json.loads(required[4].read_text(encoding="utf-8"))
    if spec != campaign_spec():
        raise ValueError("campaign specification is stale")
    if registry != create_registry(tokens, tokenizer):
        raise ValueError("measurement registry is stale")
    validate_measurement_registry(registry, require_nonempty=True)
    if plan != build_plan(output, tokens, tokenizer, required[1]):
        raise ValueError("execution plan is stale")
    contexts = registry["construction"] + registry["validation"]
    reserved = np.asarray(
        [row["chunk_index"] for row in contexts], dtype=np.int64)
    data = np.memmap(tokens, dtype=np.uint16, mode="r")
    total_chunks = len(data) // int(REGISTRY["context_length"])
    if not isinstance(orders, list) or len(orders) != len(TRAJECTORIES):
        raise ValueError("order manifest has an invalid count")
    for record, trajectory in zip(orders, TRAJECTORIES, strict=True):
        role = str(trajectory["role"])
        seed = int(trajectory["seed"])
        relative = Path("orders") / f"{role}-seed{seed}-chunk-order.npy"
        path = output / relative
        regenerated = permuted_chunk_order(
            total_chunks, seed, reserved_chunks=reserved)
        if (
            record.get("role") != role or record.get("seed") != seed
            or record.get("path") != str(relative)
            or not path.is_file()
            or record.get("file_sha256") != sha256_path(path)
            or record.get("array_sha256") != regenerated["sha256"]
            or record.get("chunk_count") != len(regenerated["order"])
            or not np.array_equal(
                np.load(path, allow_pickle=False), regenerated["order"])
        ):
            raise ValueError("PCG64 order manifest does not replay")
    if not isinstance(source_records, list) or len(source_records) != len(SOURCE_FILES):
        raise ValueError("source snapshot has an invalid count")
    for relative, record in zip(SOURCE_FILES, source_records, strict=True):
        source = ROOT / relative
        staged = output / "source" / relative
        if (
            not source.is_file() or not staged.is_file()
            or source.read_bytes() != staged.read_bytes()
            or record != _file_record(staged, output)
        ):
            raise ValueError(f"staged campaign source drifted: {relative}")
    records = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path != required[5]:
            records.append(_file_record(path, output))
    expected_manifest = "".join(
        f"{record['sha256']}  {record['path']}\n" for record in records)
    if required[5].read_text(encoding="ascii") != expected_manifest:
        raise ValueError("campaign manifest is not the exact tree")


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
    tokens = Path(arguments.tokens).resolve()
    tokenizer = Path(arguments.tokenizer).resolve()
    if arguments.check:
        check(output, tokens, tokenizer)
        print(f"contrast-energy stage: verified {output}")
    else:
        stage(output, tokens, tokenizer)
        print(f"contrast-energy stage: wrote {output}")


if __name__ == "__main__":
    main()
