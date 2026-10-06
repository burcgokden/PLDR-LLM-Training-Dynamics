#!/usr/bin/env python3
"""Stage the complete prospective gate-shape confirmation campaign."""

from __future__ import annotations
from companion_paths import configured_path, data_root, resolve_row_arguments

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

from confirm.comprehensive_campaign import (  # noqa: E402
    DATASET_SHA256,
    TOKENIZER_SHA256,
    campaign_spec,
    write_chunk_orders,
)
from confirm.confirmation_artifacts import (  # noqa: E402
    sha256_path,
    validate_measurement_registry,
)
from confirm.gate_shape_evidence import write_json_atomic  # noqa: E402
from confirm.run_chronological_confirmation import create_registry  # noqa: E402


DEFAULT_OUTPUT = Path(
    configured_path('data:row/comprehensive-gate-shape/comprehensive-gate-shape-confirmation')
)
DEFAULT_TOKENS = data_root('row') / 'inputs/refinedweb_tokens.npy'
DEFAULT_TOKENIZER = ROOT / "experiments" / "data" / "tokenizer.model"


SOURCE_FILES = (
    "requirements.txt",
    "experiments/train_run.py",
    "experiments/pldr_model_v510.py",
    "experiments/power_law_attention_layer_v510.py",
    "experiments/instrument.py",
    "experiments/optimizer_ledger.py",
    "experiments/confirm/confirmation_artifacts.py",
    "experiments/confirm/gate_shape.py",
    "experiments/confirm/gate_shape_evidence.py",
    "experiments/confirm/row_map_live.py",
    "experiments/confirm/observable_margin_live.py",
    "experiments/confirm/chronological_confirmation_specs.py",
    "experiments/confirm/capture_chronological_confirmation.py",
    "experiments/confirm/capture_chronological_plga.py",
    "experiments/confirm/run_chronological_confirmation.py",
    "experiments/confirm/run_chronological_intervention.py",
    "experiments/confirm/comprehensive_gate_shape.py",
    "experiments/confirm/comprehensive_campaign.py",
    "experiments/confirm/comprehensive_lock.py",
    "experiments/confirm/capture_comprehensive_confirmation.py",
    "experiments/confirm/run_comprehensive_intervention.py",
    "experiments/confirm/capture_comprehensive_plga.py",
    "experiments/analysis/analyze_comprehensive_confirmation.py",
    "experiments/analysis/analyze_comprehensive_intervention.py",
    "experiments/analysis/analyze_comprehensive_plga.py",
    "experiments/analysis/finalize_comprehensive_lock.py",
    "experiments/analysis/finalize_comprehensive_report.py",
    "scripts/run_comprehensive_gate_shape_qualification.py",
    "scripts/stage_comprehensive_gate_shape_confirmation.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _trajectory_rows(spec: dict[str, Any]) -> list[tuple[str, int]]:
    return [
        (role, int(seed))
        for role in ("development", "construction", "heldout")
        for seed in spec["trajectories"][role]
    ]


def _name(role: str, seed: int) -> str:
    return f"cgs-{role}-s{seed}"


def _training_command(
    output: Path,
    tokens: Path,
    tokenizer: Path,
    registry: Path,
    role: str,
    seed: int,
    device: str,
) -> list[str]:
    spec = campaign_spec()
    anchors = (
        [2000, 4000, 8000]
        if role == "development" else spec["trajectories"]["anchors"]
    )
    steps = 9001 if role == "development" else 24001
    checkpoints = ",".join(map(str, anchors))
    order = output / "orders" / f"{role}-seed{seed}-chunk-order.npy"
    return [
        sys.executable,
        str(ROOT / "experiments" / "train_run.py"),
        "--name", _name(role, seed),
        "--run-id", _name(role, seed),
        "--lineage-root", _name(role, seed),
        "--lr", "0.00075",
        "--warmup", "250",
        "--const_lr",
        "--steps", str(steps),
        "--batch", "32",
        "--ctx", "256",
        "--layers", "3",
        "--heads", "4",
        "--dk", "64",
        "--adff", "170",
        "--seed", str(seed),
        "--device", device,
        "--tokens", str(tokens),
        "--tok_model", str(tokenizer),
        "--outdir", str(output / "runs"),
        "--data_offset", "0",
        "--data_order", str(order),
        "--probe_region", "global",
        "--confirmation_registry", str(registry),
        "--campaign_spec",
        str(output / "protocol" / "campaign-spec.json"),
        "--ckpt_steps", checkpoints,
        "--optimizer_ckpt_steps", checkpoints,
        "--probe_every", "1000",
        "--sharp_every", "1000000",
        "--sharp_pre_every", "1000000",
        "--skip_generation",
    ]


def build_execution_plan(
    output: Path, tokens: Path, tokenizer: Path,
) -> dict[str, Any]:
    spec = campaign_spec()
    registry = output / "protocol" / "registry.json"
    trajectories = _trajectory_rows(spec)
    training = []
    endpoint_capture = []
    endpoint_analysis = []
    plga_capture = []
    for index, (role, seed) in enumerate(trajectories):
        device = f"cuda:{index % 2}"
        training.append({
            "id": f"train-{_name(role, seed)}",
            "role": role,
            "seed": seed,
            "device": device,
            "argv": _training_command(
                output, tokens, tokenizer, registry, role, seed, device),
        })
        anchors = (
            [2000, 4000, 8000]
            if role == "development" else spec["trajectories"]["anchors"]
        )
        updates = (
            spec["trajectories"]["development_block_updates"]
            if role == "development"
            else spec["trajectories"]["confirmation_block_updates"]
        )
        order = output / "orders" / f"{role}-seed{seed}-chunk-order.npy"
        for anchor in anchors:
            checkpoint = output / "runs" / _name(role, seed) / f"ckpt_{anchor}.pt"
            for layer in spec["trajectories"]["layer_indices"]:
                stem = f"{role}-s{seed}-a{anchor}-l{layer}"
                raw = output / "raw" / "endpoint" / f"{stem}.npz"
                report = output / "reports" / "endpoint" / f"{stem}.json"
                endpoint_capture.append({
                    "id": f"capture-{stem}",
                    "role": role,
                    "argv": [
                        sys.executable,
                        str(ROOT / "experiments" / "confirm"
                            / "capture_comprehensive_confirmation.py"),
                        "--checkpoint", str(checkpoint),
                        "--registry", str(registry),
                        "--tokens", str(tokens),
                        "--data-order", str(order),
                        "--layer", str(layer),
                        "--updates", str(updates),
                        "--role", role,
                        "--seed", str(seed),
                        "--device", device,
                        "--context-chunk-size", "8",
                        "--output", str(raw),
                    ],
                })
                analysis_argv = [
                    sys.executable,
                    str(ROOT / "experiments" / "analysis"
                        / "analyze_comprehensive_confirmation.py"),
                    "--ledger", str(raw),
                    "--output", str(report),
                ]
                if role == "heldout":
                    analysis_argv.extend([
                        "--lock", str(output / "protocol" / "construction-lock.json")
                    ])
                endpoint_analysis.append({
                    "id": f"analyze-{stem}", "role": role, "argv": analysis_argv})
                if role != "development":
                    plga_raw = output / "raw" / "plga" / f"{stem}.npz"
                    plga_argv = [
                        sys.executable,
                        str(ROOT / "experiments" / "confirm"
                            / "capture_comprehensive_plga.py"),
                        "--checkpoint", str(checkpoint),
                        "--registry", str(registry),
                        "--tokens", str(tokens),
                        "--role", role,
                        "--seed", str(seed),
                        "--layer", str(layer),
                        "--device", device,
                        "--output", str(plga_raw),
                    ]
                    if role == "heldout":
                        plga_argv.extend([
                            "--lock", str(output / "protocol" / "construction-lock.json")
                        ])
                    plga_capture.append({
                        "id": f"plga-{stem}", "role": role, "argv": plga_argv})

    intervention_capture = []
    for role in ("construction", "heldout"):
        for seed in spec["trajectories"][role]:
            device = f"cuda:{seed % 2}"
            checkpoint = output / "runs" / _name(role, seed) / "ckpt_8000.pt"
            order = output / "orders" / f"{role}-seed{seed}-chunk-order.npy"
            for ordinal in range(12):
                raw = (
                    output / "raw" / "intervention"
                    / f"{role}-s{seed}-u{ordinal:02d}.npz"
                )
                argv = [
                    sys.executable,
                    str(ROOT / "experiments" / "confirm"
                        / "run_comprehensive_intervention.py"),
                    "--checkpoint", str(checkpoint),
                    "--registry", str(registry),
                    "--tokens", str(tokens),
                    "--data-order", str(order),
                    "--role", role,
                    "--seed", str(seed),
                    "--ordinal", str(ordinal),
                    "--device", device,
                    "--output", str(raw),
                ]
                if role == "heldout":
                    argv.extend([
                        "--lock", str(output / "protocol" / "construction-lock.json")
                    ])
                intervention_capture.append({
                    "id": f"intervention-{role}-s{seed}-u{ordinal:02d}",
                    "role": role,
                    "argv": argv,
                })

    construction_endpoint_reports = [
        str(output / "reports" / "endpoint"
            / f"construction-s8444-a{anchor}-l{layer}.json")
        for anchor in spec["trajectories"]["anchors"]
        for layer in spec["trajectories"]["layer_indices"]
    ]
    development_endpoint_reports = [
        str(output / "reports" / "endpoint"
            / f"development-s7444-a{anchor}-l{layer}.json")
        for anchor in (2000, 4000, 8000)
        for layer in spec["trajectories"]["layer_indices"]
    ]
    heldout_endpoint_reports = [
        str(output / "reports" / "endpoint"
            / f"heldout-s{seed}-a{anchor}-l{layer}.json")
        for seed in spec["trajectories"]["heldout"]
        for anchor in spec["trajectories"]["anchors"]
        for layer in spec["trajectories"]["layer_indices"]
    ]
    all_endpoint_reports = [
        *development_endpoint_reports, *construction_endpoint_reports,
        *heldout_endpoint_reports,
    ]
    construction_interventions = [
        str(output / "raw" / "intervention"
            / f"construction-s8444-u{ordinal:02d}.npz")
        for ordinal in range(12)
    ]
    heldout_interventions = [
        str(output / "raw" / "intervention"
            / f"heldout-s{seed}-u{ordinal:02d}.npz")
        for seed in spec["trajectories"]["heldout"]
        for ordinal in range(12)
    ]
    construction_plga = [
        str(output / "raw" / "plga"
            / f"construction-s8444-a{anchor}-l{layer}.npz")
        for anchor in spec["trajectories"]["anchors"]
        for layer in spec["trajectories"]["layer_indices"]
    ]
    heldout_plga = [
        str(output / "raw" / "plga"
            / f"heldout-s{seed}-a{anchor}-l{layer}.npz")
        for seed in spec["trajectories"]["heldout"]
        for anchor in spec["trajectories"]["anchors"]
        for layer in spec["trajectories"]["layer_indices"]
    ]
    aggregate = {
        "construction_intervention": [
            sys.executable,
            str(ROOT / "experiments" / "analysis"
                / "analyze_comprehensive_intervention.py"),
            *construction_interventions,
            "--output", str(output / "reports" / "construction-intervention.json"),
            "--require-pass",
        ],
        "construction_plga": [
            sys.executable,
            str(ROOT / "experiments" / "analysis"
                / "analyze_comprehensive_plga.py"),
            *construction_plga,
            "--output", str(output / "reports" / "construction-plga.json"),
            "--require-pass",
        ],
        "freeze_lock": [
            sys.executable,
            str(ROOT / "experiments" / "analysis"
                / "finalize_comprehensive_lock.py"),
            "--development-reports", *development_endpoint_reports,
            "--construction-reports", *construction_endpoint_reports,
            "--intervention-report",
            str(output / "reports" / "construction-intervention.json"),
            "--plga-report", str(output / "reports" / "construction-plga.json"),
            "--output", str(output / "protocol" / "construction-lock.json"),
        ],
        "heldout_intervention": [
            sys.executable,
            str(ROOT / "experiments" / "analysis"
                / "analyze_comprehensive_intervention.py"),
            *heldout_interventions,
            "--lock", str(output / "protocol" / "construction-lock.json"),
            "--output", str(output / "reports" / "heldout-intervention.json"),
            "--require-pass",
        ],
        "heldout_plga": [
            sys.executable,
            str(ROOT / "experiments" / "analysis"
                / "analyze_comprehensive_plga.py"),
            *heldout_plga,
            "--lock", str(output / "protocol" / "construction-lock.json"),
            "--output", str(output / "reports" / "heldout-plga.json"),
            "--require-pass",
        ],
        "finalize_campaign": [
            sys.executable,
            str(ROOT / "experiments" / "analysis"
                / "finalize_comprehensive_report.py"),
            "--endpoint-reports", *all_endpoint_reports,
            "--intervention-reports",
            str(output / "reports" / "construction-intervention.json"),
            str(output / "reports" / "heldout-intervention.json"),
            "--plga-reports",
            str(output / "reports" / "construction-plga.json"),
            str(output / "reports" / "heldout-plga.json"),
            "--lock", str(output / "protocol" / "construction-lock.json"),
            "--campaign-root", str(output),
            "--output", str(output / "reports" / "final-report.json"),
            "--require-pass",
        ],
    }
    return {
        "schema_version": "pldr-comprehensive-execution-plan-v1",
        "campaign_id": spec["campaign_id"],
        "campaign_spec_sha256": spec["spec_sha256"],
        "stage_order": ["Q", "D", "C", "H", "I", "P", "F"],
        "qualification": [
            sys.executable,
            str(ROOT / "scripts"
                / "run_comprehensive_gate_shape_qualification.py"),
            "--tokens", str(tokens),
            "--tokenizer", str(tokenizer),
            "--registry", str(registry),
            "--campaign-spec",
            str(output / "protocol" / "campaign-spec.json"),
            "--data-order", str(
                output / "orders" / "development-seed7444-chunk-order.npy"),
            "--device", "cuda:0",
            "--output", str(
                output / "reports" / "qualification-report.json"),
            "--require-pass",
        ],
        "training": training,
        "endpoint_capture": endpoint_capture,
        "endpoint_analysis": endpoint_analysis,
        "intervention_capture": intervention_capture,
        "plga_capture": plga_capture,
        "aggregate_commands": aggregate,
        "no_unresolved_placeholders": True,
    }


def _source_snapshot(output: Path) -> dict[str, str]:
    manifest = {}
    for relative_text in SOURCE_FILES:
        relative = Path(relative_text)
        source = ROOT / relative
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"missing source snapshot input: {source}")
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest[(Path("source") / relative).as_posix()] = _sha256(target)
    return manifest


def _tree_manifest(output: Path) -> str:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"staging tree contains symlink: {path}")
        relative = path.relative_to(output)
        if not path.is_file() or path.name == "MANIFEST.sha256":
            continue
        if relative.parts[0] in {"raw", "reports", "runs"}:
            continue
        if relative == Path("protocol/construction-lock.json"):
            continue
        rows.append(f"{_sha256(path)}  {relative.as_posix()}\n")
    return "".join(rows)


def _runbook(spec: dict[str, Any]) -> str:
    return f"""# Comprehensive gate-shape confirmation

Campaign: `{spec['campaign_id']}`

This directory is a prospective, unexecuted campaign package. Execute the
argv arrays in `protocol/execution-plan.json` in stage order Q, D, C, H, I,
P, F. Every path is resolved and no command contains a template placeholder.

Primary scientific decisions use native endpoints and independent float64
analysis. Observed JVP remainders and native float32 replay discrepancies are
diagnostics only. Construction creates `protocol/construction-lock.json`;
held-out analyzers and counterfactual producers require its digest.

`MANIFEST.sha256` binds the immutable protocol, orders, source snapshot, and
runbook. Generated checkpoints, raw ledgers, reports, and the construction
lock are excluded because they are produced only after staging; their own
digests and cross-bindings are verified by the campaign finalizer.

The hard resource limits are {spec['resource_contract']['gpu_hour_cap']:.1f}
aggregate GPU-hours and {spec['resource_contract']['compressed_output_cap_bytes']}
compressed output bytes.
"""


def stage(output: Path, tokens: Path, tokenizer: Path) -> None:
    if output.exists():
        raise FileExistsError(
            f"refusing to replace existing campaign directory: {output}")
    if sha256_path(tokens) != DATASET_SHA256:
        raise ValueError("token archive digest disagrees with the campaign")
    if sha256_path(tokenizer) != TOKENIZER_SHA256:
        raise ValueError("tokenizer digest disagrees with the campaign")
    output.mkdir(parents=True)
    spec = campaign_spec()
    write_json_atomic(output / "protocol" / "campaign-spec.json", spec)
    token_data = np.memmap(tokens, dtype=np.uint16, mode="r")
    total_chunks = len(token_data) // int(spec["optimizer"]["context_length"])
    orders = write_chunk_orders(output / "orders", total_chunks=total_chunks)
    write_json_atomic(output / "protocol" / "order-manifest.json", {
        "schema_version": "pldr-comprehensive-order-manifest-v1",
        "orders": orders,
    })
    registry = create_registry(
        tokens,
        tokenizer,
        output / "protocol" / "registry.json",
        reserved_start=total_chunks - 128,
    )
    validate_measurement_registry(registry, require_nonempty=True)
    plan = build_execution_plan(output, tokens.resolve(), tokenizer.resolve())
    write_json_atomic(output / "protocol" / "execution-plan.json", plan)
    _write_text(output / "RUNBOOK.md", _runbook(spec))
    source_manifest = _source_snapshot(output)
    write_json_atomic(output / "protocol" / "source-manifest.json", {
        "schema_version": "pldr-comprehensive-source-manifest-v1",
        "files": source_manifest,
    })
    _write_text(output / "MANIFEST.sha256", _tree_manifest(output))


def check(output: Path, tokens: Path, tokenizer: Path) -> None:
    if not output.is_dir() or output.is_symlink():
        raise ValueError("staged campaign directory is absent or invalid")
    if sha256_path(tokens) != DATASET_SHA256 or sha256_path(tokenizer) != TOKENIZER_SHA256:
        raise ValueError("current data bindings changed")
    manifest_path = output / "MANIFEST.sha256"
    if manifest_path.read_text(encoding="utf-8") != _tree_manifest(output):
        raise ValueError("staged campaign manifest or membership is stale")
    spec = json.loads((output / "protocol" / "campaign-spec.json").read_text())
    if spec != campaign_spec():
        raise ValueError("staged campaign specification is stale")
    registry = json.loads((output / "protocol" / "registry.json").read_text())
    validate_measurement_registry(registry, require_nonempty=True)
    expected_plan = build_execution_plan(output, tokens.resolve(), tokenizer.resolve())
    actual_plan = json.loads((output / "protocol" / "execution-plan.json").read_text())
    if actual_plan != expected_plan:
        raise ValueError("staged execution plan is stale")
    order_manifest = json.loads(
        (output / "protocol" / "order-manifest.json").read_text())
    for row in order_manifest["orders"]:
        path = output / "orders" / row["path"]
        if _sha256(path) != row["file_sha256"]:
            raise ValueError("staged chunk order digest changed")
    for relative_text in SOURCE_FILES:
        relative = Path(relative_text)
        if (output / "source" / relative).read_bytes() != (ROOT / relative).read_bytes():
            raise ValueError(f"staged source snapshot is stale: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tokens", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Report inputs and outputs without acquisition")
    arguments = parser.parse_args()
    if not resolve_row_arguments(arguments, parser, defaults={'tokens': DEFAULT_TOKENS, 'tokenizer': DEFAULT_TOKENIZER},
            identities={'tokens': globals().get('DATASET_SHA256'), 'tokenizer': globals().get('TOKENIZER_SHA256')}):
        return
    if arguments.check:
        check(arguments.output.resolve(), arguments.tokens.resolve(),
              arguments.tokenizer.resolve())
        print(f"comprehensive gate-shape staging: verified {arguments.output}")
    else:
        stage(arguments.output.resolve(), arguments.tokens.resolve(),
              arguments.tokenizer.resolve())
        print(f"comprehensive gate-shape staging: wrote {arguments.output}")


if __name__ == "__main__":
    main()
