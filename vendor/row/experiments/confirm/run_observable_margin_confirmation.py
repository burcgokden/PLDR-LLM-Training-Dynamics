#!/usr/bin/env python3
"""Plan and gate the fresh observable-margin confirmation campaign.

The planner never opens held-out evidence while choosing the construction
cell.  GPU instrumentation writes compact ledgers matching the generated
schema; ``await-ledger`` then invokes the independent analyzer in a fresh
process.  Missing fresh checkpoints or ledgers are reported as readiness
states, not silently replaced by retained trajectories.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
sys.pycache_prefix = "/dev/null"

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ANALYZER = ROOT / "experiments" / "analysis" / "analyze_observable_margin.py"
LIVE = ROOT / "experiments" / "confirm" / "observable_margin_live.py"
CONTROLS = ROOT / "experiments" / "confirm" / "observable_margin_controls.py"
DEFAULT_MANUSCRIPT_EXPORT = (
    ROOT.parent / "paper-outputs" / "paper-outputs-rev37")
sys.path.insert(0, str(HERE))

from gate_shape_evidence import (  # noqa: E402
    digest_object,
    load_npz,
    sha256_path,
    write_json_atomic,
)
from observable_margin import unpack_union_graph  # noqa: E402
from observable_margin_specs import (  # noqa: E402
    ANCHORS,
    ARCHITECTURE,
    ARITHMETIC,
    CAMPAIGN_ID,
    INTERVENTION_COMPARISON_NAMES,
    INTERVENTION_OPERATOR_BY_NAME,
    INTERVENTION_OPERATOR_CONTRACTS,
    INTERVENTION_OPERATOR_FIELDS,
    OBSERVABLE_CONTRACT,
    REGISTRY,
    RESOURCE_CAPS,
    RUNS,
    STAGE_BY_ID,
    STAGES,
    STATISTICS,
    STOP_GATES,
    TRANSFER,
)


PLAN_SCHEMA = "pldr-observable-margin-execution-plan-v1"
READINESS_SCHEMA = "pldr-observable-margin-readiness-v1"
LOCK_SCHEMA = "pldr-observable-margin-construction-lock-v1"
FINAL_SCHEMA = "pldr-observable-margin-final-decision-v1"


def _json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _load_sealed(path, schema, digest_field):
    value = _json(path)
    if value.get("schema_version") != schema:
        raise ValueError(f"unexpected schema in {path}")
    unsigned = dict(value)
    recorded = unsigned.pop(digest_field, None)
    if recorded != digest_object(unsigned):
        raise ValueError(f"digest does not replay for {path}")
    return value


def _command(values):
    return [str(value) for value in values]


def _display(values):
    return shlex.join([str(value) for value in values])



def _training_chunk_stop():
    """Exclusive chunk boundary consumed by the 24,001-update T run."""

    return (max(ANCHORS["complete_optimizer_steps"]) + 1) * ARCHITECTURE[
        "batch_size"]


def create_registry(tokens, tokenizer, output, *, reserved_start=-1):
    """Create the checkpoint-owned probe and intervention-unit registry."""

    tokens = Path(tokens).resolve()
    tokenizer = Path(tokenizer).resolve()
    if not tokens.is_file() or not tokenizer.is_file():
        raise FileNotFoundError("tokens and tokenizer must exist")
    context = REGISTRY["context_length"]
    data = np.memmap(tokens, dtype=np.uint16, mode="r")
    chunk_count = len(data) // context
    construction_count = REGISTRY["construction_contexts"]
    heldout_count = REGISTRY["heldout_contexts"]
    required = construction_count + heldout_count
    start = int(reserved_start)
    if start < 0:
        start = chunk_count - required - 64
    if start < 0 or start + required > chunk_count:
        raise ValueError("fixed probe registry leaves the token archive")
    training_stop = _training_chunk_stop()
    if training_stop > start:
        raise ValueError("fresh training range overlaps the fixed probe registry")
    cursor = start

    def contexts(prefix, count):
        nonlocal cursor
        result = [
            {"id": f"{prefix}{index:03d}", "chunk_index": cursor + index}
            for index in range(count)
        ]
        cursor += count
        return result

    construction = contexts("omc", construction_count)
    validation = contexts("omh", heldout_count)
    intervention_units = []
    substream = 370000
    update_count = int(REGISTRY["intervention_steps"])
    for seed in STATISTICS["heldout_seeds"]:
        for ordinal in range(STATISTICS["sites_per_heldout_seed"]):
            intervention_units.append({
                "id": f"omu-s{seed}-{ordinal:02d}",
                "seed": seed,
                "ordinal": ordinal,
                "future_update_start": ordinal * update_count,
                "update_count": update_count,
                "rng_substream": substream,
            })
            substream += 1
    registry = {
        "schema_version": "pldr-observable-probe-registry-v1",
        "context_length": context,
        "construction": construction,
        "validation": validation,
        "anchors": ANCHORS["complete_optimizer_steps"],
        "context_head_block_rule": REGISTRY["context_head_block_rule"],
        "block_row_rule": REGISTRY["block_row_rule"],
        "graph_transfer": REGISTRY["graph_transfer"],
        "plga_binding_rule": (
            "same-node-context-block-anchor-row-zero-b-locks-layer-node-v1"),
        "intervention_units": intervention_units,
        "intervention_unit_rule": REGISTRY["intervention_unit_rule"],
        "dataset_sha256": sha256_path(tokens),
        "tokenizer_sha256": sha256_path(tokenizer),
    }
    registry["registry_sha256"] = digest_object(registry)
    # Use the checkpoint validator's canonical registry implementation.
    from confirmation_artifacts import validate_measurement_registry
    validate_measurement_registry(registry, require_nonempty=True)
    write_json_atomic(output, registry)
    return registry


def _checkpoint(run_root, run, step):
    return Path(run_root) / run["name"] / ANCHORS["checkpoint_pattern"].format(
        step=step)


def _readiness_checkpoint(path, *, run, step, registry):
    result = {
        "run": run["name"],
        "seed": run["seed"],
        "role": run["ownership"],
        "step": int(step),
        "path": str(Path(path).resolve()),
        "present": Path(path).is_file(),
        "bytes": Path(path).stat().st_size if Path(path).is_file() else 0,
        "complete_state_valid": False,
        "registry_bound": False,
        "cursor_and_rng_bound": False,
    }
    if not result["present"]:
        return result
    try:
        from confirmation_artifacts import load_complete_checkpoint
        payload = load_complete_checkpoint(path, map_location="cpu")
        identity = payload.get("run_identity", {})
        result["complete_state_valid"] = bool(
            int(payload["step"]) == int(step)
            and int(payload["config"]["seed"]) == int(run["seed"])
            and identity.get("run_id") == run["name"]
            and identity.get("lineage_root") == run["name"]
            and identity.get("from_scratch") is True)
        result["registry_bound"] = (
            payload["measurement_registry"] == registry)
        result["cursor_and_rng_bound"] = bool(
            payload.get("rng_states")
            and payload["data_state"].get("cursor_end")
            == payload.get("data_offset_end"))
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        pass
    return result


def readiness(
    run_root, tokens, tokenizer, registry_path, *, check_stage="R",
):
    """Return typed preflight or post-training anchor readiness.

    READY_FOR_T deliberately does not require checkpoints that stage T creates.
    READY_FOR_B additionally validates every scientific anchor and its exact
    one-successor replay checkpoint with the canonical complete-state loader.
    """

    import torch
    from confirmation_artifacts import validate_measurement_registry

    run_root = Path(run_root).resolve()
    tokens = Path(tokens).resolve()
    tokenizer = Path(tokenizer).resolve()
    registry_path = Path(registry_path).resolve()
    files = {
        "tokens": tokens.is_file(),
        "tokenizer": tokenizer.is_file(),
        "registry": registry_path.is_file(),
    }
    registry = None
    registry_checks = {
        "schema_valid": False,
        "digest_replays": False,
        "dataset_bound": False,
        "tokenizer_bound": False,
        "partition_counts": False,
        "training_probe_disjoint": False,
    }
    if all(files.values()):
        try:
            registry = _json(registry_path)
            unsigned = dict(registry)
            recorded = unsigned.pop("registry_sha256", None)
            validate_measurement_registry(registry, require_nonempty=True)
            chunks = [
                int(row["chunk_index"])
                for role in ("construction", "validation")
                for row in registry[role]
            ]
            training_stop = _training_chunk_stop()
            registry_checks = {
                "schema_valid": True,
                "digest_replays": recorded == digest_object(unsigned),
                "dataset_bound": (
                    registry.get("dataset_sha256") == sha256_path(tokens)),
                "tokenizer_bound": (
                    registry.get("tokenizer_sha256")
                    == sha256_path(tokenizer)),
                "partition_counts": (
                    len(registry.get("construction", []))
                    == REGISTRY["construction_contexts"]
                    and len(registry.get("validation", []))
                    == REGISTRY["heldout_contexts"]),
                "training_probe_disjoint": bool(
                    chunks and len(chunks) == len(set(chunks))
                    and training_stop <= min(chunks)),
            }
        except (KeyError, OSError, TypeError, ValueError):
            registry = None
    device_checks = {}
    for name in RESOURCE_CAPS["registered_devices"]:
        passed = False
        try:
            index = int(name.split(":", 1)[1])
            if torch.cuda.is_available() and index < torch.cuda.device_count():
                device = torch.device(name)
                torch.cuda.reset_peak_memory_stats(device)
                sample = torch.ones(32, dtype=torch.float32, device=device)
                torch.cuda.synchronize(device)
                resolved_index = sample.device.index
                allocated = int(torch.cuda.max_memory_allocated(device))
                reserved = int(torch.cuda.max_memory_reserved(device))
                passed = bool(
                    resolved_index == index
                    and allocated >= sample.numel() * sample.element_size()
                    and reserved >= allocated > 0)
                del sample
                torch.cuda.empty_cache()
        except (RuntimeError, TypeError, ValueError):
            passed = False
        device_checks[name] = passed
    resource_meter_ready = bool(
        device_checks and all(device_checks.values())
        and hasattr(torch.cuda, "max_memory_allocated")
        and hasattr(torch.cuda, "max_memory_reserved")
        and hasattr(torch.cuda, "synchronize"))
    preflight_ready = bool(
        all(files.values()) and all(registry_checks.values())
        and all(device_checks.values()) and resource_meter_ready)
    if check_stage not in {"R", "T"}:
        raise ValueError("readiness check stage must be R or T")
    anchors = []
    successors = []
    if check_stage == "T":
        checkpoint_registry = registry if registry is not None else {}
        for run in RUNS:
            for step in ANCHORS["complete_optimizer_steps"]:
                anchors.append(_readiness_checkpoint(
                    _checkpoint(run_root, run, step), run=run, step=step,
                    registry=checkpoint_registry))
                successors.append(_readiness_checkpoint(
                    _checkpoint(run_root, run, step + 1), run=run,
                    step=step + 1, registry=checkpoint_registry))
    checkpoint_rows = [*anchors, *successors]
    anchor_ready = bool(
        check_stage == "T" and preflight_ready and checkpoint_rows
        and all(
            row["present"] and row["complete_state_valid"]
            and row["registry_bound"] and row["cursor_and_rng_bound"]
            for row in checkpoint_rows))
    status = (
        ("READY_FOR_B" if anchor_ready else "WAITING_FOR_ANCHORS")
        if check_stage == "T"
        else ("READY_FOR_T" if preflight_ready else "WAITING_FOR_PREFLIGHT"))
    r_checks = {
        "fresh_run_names": len({run["name"] for run in RUNS}) == 3,
        "dataset_and_tokenizer_digests": bool(
            registry_checks["dataset_bound"]
            and registry_checks["tokenizer_bound"]),
        "fixed_probe_registry": bool(
            registry_checks["schema_valid"]
            and registry_checks["digest_replays"]
            and registry_checks["partition_counts"]),
        "training_probe_disjointness": registry_checks[
            "training_probe_disjoint"],
        "native_update_shadow_arithmetic": bool(
            "float32" in ARITHMETIC["native_update"]
            and ARITHMETIC["deciding_shadow_dtype"] == "float64"),
        "registered_devices": all(device_checks.values()),
        "compact_evidence_only": (
            OBSERVABLE_CONTRACT["raw_node_policy"] == "forbidden"),
        "resource_meter_ready": resource_meter_ready,
    }
    t_checks = {
        "five_complete_optimizer_anchors": bool(anchor_ready),
        "data_cursor_and_rng_state": bool(anchor_ready and all(
            row["cursor_and_rng_bound"] for row in checkpoint_rows)),
        "checkpoint_registry_binding": bool(anchor_ready and all(
            row["registry_bound"] for row in checkpoint_rows)),
    }
    value = {
        "schema_version": READINESS_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "status": status,
        "required_level_semantics": {
            "T": "preflight-valid-without-requiring-future-checkpoints",
            "B": "all-anchor-and-anchor-plus-one-complete-states-valid",
        },
        "files": files,
        "registry_checks": registry_checks,
        "device_checks": device_checks,
        "device_check_rule": (
            "allocate-float32-resolve-index-synchronize-and-replay-"
            "peak-allocated-reserved-v1"),
        "resource_meter_ready": resource_meter_ready,
        "complete_optimizer_anchors": anchors,
        "exact_restart_successors": successors,
        "old_trajectory_substitution_allowed": False,
        "next_stage": (
            "B" if anchor_ready else "T" if check_stage == "R" else None),
        "stage_checks": r_checks if check_stage == "R" else t_checks,
    }
    value["readiness_sha256"] = digest_object(value)
    return value


def _job(
    stage, job_id, role, device, action, command, output, *, dependencies=(),
    reserved_gpu_hours=0.0, reserved_wall_clock_hours=0.25,
    reserved_output_bytes=16 * 1024 ** 2, timeout_seconds=900,
    owned_checks=(),
):
    return {
        "stage": stage,
        "job_id": job_id,
        "record_key": f"{stage}:{job_id}",
        "role": role,
        "device": device,
        "action": action,
        "dependencies": list(dependencies),
        "stage_gpu_hour_cap": RESOURCE_CAPS[
            "planned_stage_gpu_hours"][stage],
        "output": str(output),
        "command": _command(command),
        "command_display": _display(command),
        "reserved_gpu_hours": float(reserved_gpu_hours),
        "reserved_wall_clock_hours": float(reserved_wall_clock_hours),
        "reserved_output_bytes": int(reserved_output_bytes),
        "timeout_seconds": int(timeout_seconds),
        "owned_checks": list(owned_checks),
    }


def _validate_serial_resource_envelope(jobs):
    """Validate reservations for the strictly serial campaign executor."""

    stage_budget = sum(
        float(value)
        for value in RESOURCE_CAPS["planned_stage_gpu_hours"].values())
    planned_gpu = float(RESOURCE_CAPS["planned_gpu_hours"])
    if not math.isclose(stage_budget, planned_gpu, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "aggregate stage GPU budget does not equal its stage caps")
    job_gpu = sum(float(row["reserved_gpu_hours"]) for row in jobs)
    job_wall = sum(float(row["reserved_wall_clock_hours"]) for row in jobs)
    if job_gpu > planned_gpu + 1e-12:
        raise ValueError(
            "serial job GPU reservations exceed the aggregate stage budget")
    if job_wall > float(RESOURCE_CAPS["hard_wall_clock_hours"]) + 1e-12:
        raise ValueError(
            "serial job wall reservations exceed the hard wall cap")
    return {
        "aggregate_stage_gpu_budget": planned_gpu,
        "job_gpu_reservations": job_gpu,
        "job_wall_reservations": job_wall,
        "hard_wall_clock_hours": float(
            RESOURCE_CAPS["hard_wall_clock_hours"]),
    }


def execution_plan(
        run_root, tokens, tokenizer, output_root, registry_path,
        environment_attestation, manuscript_export_root=None):
    run_root = Path(run_root).resolve()
    tokens = Path(tokens).resolve()
    tokenizer = Path(tokenizer).resolve()
    output_root = Path(output_root).resolve()
    registry_path = Path(registry_path).resolve()
    environment_attestation = Path(environment_attestation).resolve()
    manuscript_export_root = Path(
        manuscript_export_root or DEFAULT_MANUSCRIPT_EXPORT).resolve()
    campaign_ledger = output_root.parent / "campaign-attempt-ledger.json"
    repository_manifest = ROOT / "MANIFEST.sha256"
    staged_repository_manifest = (
        output_root.parent / "preflight" / "repository-MANIFEST.sha256")
    manuscript_export_manifest = manuscript_export_root / "MANIFEST.sha256"
    staged_manuscript_export_manifest = (
        output_root.parent / "preflight"
        / "manuscript-export-MANIFEST.sha256")
    jobs = []
    readiness_path = output_root / "R" / "registry-validation.json"
    jobs.append(_job(
        "R", "registry-validation", "shared", "cpu", "registry-validation",
        [
            "python3", __file__, "readiness", "--run-root", run_root,
            "--tokens", tokens, "--tokenizer", tokenizer,
            "--registry", registry_path, "--check-stage", "R",
            "--require-level", "T", "--output", readiness_path,
        ], readiness_path,
        reserved_wall_clock_hours=0.1, reserved_output_bytes=8 * 1024 ** 2,
        timeout_seconds=600, owned_checks=tuple(
            STAGE_BY_ID["R"]["required_checks"]),
    ))
    anchor_steps = ANCHORS["complete_optimizer_steps"]
    replay_steps = sorted({*anchor_steps, *(step + 1 for step in anchor_steps)})
    anchor_text = ",".join(map(str, replay_steps))
    restart_keys = []
    for run in RUNS:
        output = run_root / run["name"] / "ckpt_24000.pt"
        train_key = f"T:train-s{run['seed']}"
        jobs.append(_job(
            "T", f"train-s{run['seed']}", run["ownership"], run["device"],
            "native-training",
            [
                "python3", ROOT / "experiments" / "train_run.py",
                "--name", run["name"], "--run-id", run["name"],
                "--lineage-root", run["name"], "--lr", 7.5e-4,
                "--warmup", 250, "--steps", 24001, "--batch", 32,
                "--ctx", 256, "--layers", 3, "--heads", 4, "--dk", 64,
                "--adff", 170, "--seed", run["seed"], "--device",
                run["device"], "--tokens", tokens, "--tok_model", tokenizer,
                "--outdir", run_root, "--ckpt_steps", ",",
                "--optimizer_ckpt_steps", anchor_text,
                "--confirmation_registry", registry_path, "--const_lr",
                "--probe_region", "global", "--optimizer", "adamw",
                "--wd", 0.1, "--clip", 1.0,
                "--probe_every", ARCHITECTURE["training_instrumentation"][
                    "probe_every"],
                "--sharp_every", ARCHITECTURE["training_instrumentation"][
                    "sharp_every"],
                "--sharp_pre_every", ARCHITECTURE[
                    "training_instrumentation"]["sharp_pre_every"],
                "--sharp_full", ARCHITECTURE["training_instrumentation"][
                    "sharp_full"],
                "--sharp_block_every", ARCHITECTURE[
                    "training_instrumentation"]["sharp_block_every"],
                "--val_batches", ARCHITECTURE["training_instrumentation"][
                    "validation_batches"],
                "--skip_generation",
            ], output, dependencies=("R:registry-validation",),
            reserved_gpu_hours=1.8, reserved_wall_clock_hours=5.0,
            reserved_output_bytes=3 * 1024 ** 3, timeout_seconds=18000,
            owned_checks=(
                "three_fresh_seeds", "native_float32_adamw",
                "actual_minibatches", "five_complete_optimizer_anchors",
                "data_cursor_and_rng_state", "checkpoint_registry_binding"),
        ))
        restart_output = output_root / "T" / f"restart-s{run['seed']}.json"
        restart_id = f"restart-s{run['seed']}"
        restart_key = f"T:{restart_id}"
        restart_keys.append(restart_key)
        jobs.append(_job(
            "T", restart_id, run["ownership"], run["device"],
            "exact-native-restart-replay",
            [
                "python3", CONTROLS, "restart-replay", "--run-root", run_root,
                "--run-name", run["name"], "--seed", run["seed"],
                "--tokens", tokens, "--registry", registry_path,
                "--device", run["device"], "--job-id", restart_id,
                "--output", restart_output,
            ], restart_output, dependencies=(train_key,),
            reserved_gpu_hours=0.18, reserved_wall_clock_hours=0.5,
            reserved_output_bytes=8 * 1024 ** 2, timeout_seconds=1800,
            owned_checks=(
                "native_float32_adamw", "actual_minibatches",
                "five_complete_optimizer_anchors", "exact_restart_replay",
                "data_cursor_and_rng_state", "checkpoint_registry_binding"),
        ))
    anchor_readiness = output_root / "T" / "anchor-readiness.json"
    jobs.append(_job(
        "T", "anchors-ready", "shared", "cpu", "post-training-readiness",
        [
            "python3", __file__, "readiness", "--run-root", run_root,
            "--tokens", tokens, "--tokenizer", tokenizer,
            "--registry", registry_path, "--check-stage", "T",
            "--require-level", "B", "--output", anchor_readiness,
        ], anchor_readiness, dependencies=tuple(restart_keys),
        reserved_wall_clock_hours=0.5, reserved_output_bytes=8 * 1024 ** 2,
        timeout_seconds=1800, owned_checks=(
            "five_complete_optimizer_anchors", "data_cursor_and_rng_state",
            "checkpoint_registry_binding"),
    ))

    construction = RUNS[0]
    b_reports = []
    b_analysis_keys = []
    block_reservations = {
        16: (0.08, 128 * 1024 ** 2, 1800),
        64: (0.15, 384 * 1024 ** 2, 3600),
        256: (0.40, 768 * 1024 ** 2, 7200),
    }
    for anchor in ANCHORS["construction_window_starts"]:
        for layer in ANCHORS["layers"]:
            for block in ANCHORS["candidate_block_lengths"]:
                tag = f"s4444-a{anchor}-l{layer}-b{block}"
                ledger = output_root / "B" / f"{tag}.npz"
                report = output_root / "B" / f"{tag}-analysis.json"
                capture_id = f"capture-{tag}"
                analysis_id = f"analyze-{tag}"
                b_reports.append(report)
                b_analysis_keys.append(f"B:{analysis_id}")
                gpu_hours, output_bytes, timeout = block_reservations[block]
                jobs.append(_job(
                    "B", capture_id, "construction", construction["device"],
                    "native-fixed-probe-ledger",
                    [
                        "python3", LIVE, "--stage", "B", "--checkpoint",
                        _checkpoint(run_root, construction, anchor), "--tokens", tokens,
                        "--registry", registry_path, "--role", "construction",
                        "--seed", construction["seed"], "--layer", layer,
                        "--updates", block, "--device", construction["device"],
                        "--job-id", capture_id, "--output", ledger,
                    ], ledger, dependencies=("T:anchors-ready",),
                    reserved_gpu_hours=gpu_hours,
                    reserved_wall_clock_hours=max(gpu_hours + 0.1, 0.25),
                    reserved_output_bytes=output_bytes, timeout_seconds=timeout,
                    owned_checks=("fixed_probe_at_every_node",),
                ))
                jobs.append(_job(
                    "B", analysis_id, "construction", "cpu",
                    "independent-ledger-analysis",
                    [
                        "python3", ANALYZER, ledger, "--output", report,
                        "--require-pass",
                    ], report, dependencies=(f"B:{capture_id}",),
                    reserved_wall_clock_hours=0.5,
                    reserved_output_bytes=16 * 1024 ** 2,
                    timeout_seconds=1800,
                    owned_checks=tuple(
                        name for name in STAGE_BY_ID["B"]["required_checks"]
                        if name != "one_locked_candidate_cell"),
                ))
    lock_path = output_root / "B" / "construction-lock.json"
    jobs.append(_job(
        "B", "lock-candidate", "construction", "cpu", "construction-lock",
        [
            "python3", __file__, "lock", "--reports", *b_reports,
            "--output", lock_path,
        ], lock_path, dependencies=tuple(b_analysis_keys),
        reserved_wall_clock_hours=0.5, reserved_output_bytes=32 * 1024 ** 2,
        timeout_seconds=1800,
        owned_checks=("construction_only", "one_locked_candidate_cell"),
    ))
    locked_checkpoint_construction = (
        run_root / construction["name"] / "ckpt_{anchor}.pt")
    h_ledger = output_root / "H" / "locked-bridge.npz"
    h_report = output_root / "H" / "locked-bridge-analysis.json"
    jobs.append(_job(
        "H", "capture-locked-bridge", "construction", "cuda:0",
        "native-fixed-probe-ledger",
        [
            "python3", LIVE, "--stage", "H", "--checkpoint",
            locked_checkpoint_construction, "--tokens", tokens, "--registry",
            registry_path, "--role", "construction", "--seed", 4444,
            "--lock", lock_path, "--device", "cuda:0", "--job-id",
            "capture-locked-bridge", "--output", h_ledger,
        ], h_ledger, dependencies=("B:lock-candidate",),
        reserved_gpu_hours=5.5, reserved_wall_clock_hours=6.0,
        reserved_output_bytes=1024 ** 3, timeout_seconds=21600,
        owned_checks=("float64_shadow",),
    ))
    jobs.append(_job(
        "H", "analyze-locked-bridge", "construction", "cpu",
        "independent-ledger-analysis",
        [
            "python3", ANALYZER, h_ledger, "--output", h_report,
            "--lock", lock_path, "--require-pass", "--require-positive",
        ], h_report, dependencies=("H:capture-locked-bridge",),
        reserved_wall_clock_hours=0.5, reserved_output_bytes=16 * 1024 ** 2,
        timeout_seconds=1800,
        owned_checks=tuple(STAGE_BY_ID["H"]["required_checks"]),
    ))
    rob_report = output_root / "ROB" / "locked-coordinate-diagnostic.json"
    jobs.append(_job(
        "ROB", "locked-coordinate", "control", "cuda:1",
        "native-3d-coordinate-diagnostic",
        [
            "python3", CONTROLS, "robustness", "--checkpoint",
            locked_checkpoint_construction, "--tokens", tokens,
            "--registry", registry_path, "--lock", lock_path,
            "--device", "cuda:1", "--job-id", "locked-coordinate",
            "--output", rob_report,
        ], rob_report, dependencies=("H:analyze-locked-bridge",),
        reserved_gpu_hours=1.8, reserved_wall_clock_hours=2.0,
        reserved_output_bytes=32 * 1024 ** 2, timeout_seconds=7200,
        owned_checks=tuple(STAGE_BY_ID["ROB"]["required_checks"]),
    ))
    heldout_reports = []
    for run in RUNS[1:]:
        tag = f"s{run['seed']}-locked"
        ledger = output_root / "M" / f"{tag}.npz"
        report = output_root / "M" / f"{tag}-analysis.json"
        heldout_reports.append(report)
        jobs.append(_job(
            "M", f"capture-{tag}", "heldout", run["device"],
            "native-fixed-probe-ledger",
            [
                "python3", LIVE, "--stage", "M", "--checkpoint",
                run_root / run["name"] / "ckpt_{anchor}.pt", "--tokens", tokens,
                "--registry", registry_path, "--role", "heldout",
                "--seed", run["seed"], "--lock", lock_path,
                "--device", run["device"], "--job-id", f"capture-{tag}",
                "--output", ledger,
            ], ledger, dependencies=(
                "H:analyze-locked-bridge", "ROB:locked-coordinate"),
            reserved_gpu_hours=4.0, reserved_wall_clock_hours=4.5,
            reserved_output_bytes=1024 ** 3, timeout_seconds=16200,
            owned_checks=("fixed_probe_at_every_node",),
        ))
        jobs.append(_job(
            "M", f"analyze-{tag}", "heldout", "cpu",
            "independent-ledger-analysis",
            [
                "python3", ANALYZER, ledger, "--output", report,
                "--lock", lock_path, "--require-pass", "--require-positive",
            ], report, dependencies=(f"M:capture-{tag}",),
            reserved_wall_clock_hours=0.5,
            reserved_output_bytes=16 * 1024 ** 2, timeout_seconds=1800,
            owned_checks=tuple(STAGE_BY_ID["M"]["required_checks"]),
        ))
    plga = output_root / "M" / "plga-centered-logit-margin.json"
    jobs.append(_job(
        "M", "plga-centered-logit", "heldout", "cpu",
        "independent-plga-margin-decision",
        [
            "python3", __file__, "plga-finalize", "--lock", lock_path,
            "--heldout", *heldout_reports, "--output", plga,
            "--require-pass",
        ], plga, dependencies=(
            "M:analyze-s5555-locked", "M:analyze-s6666-locked"),
        reserved_wall_clock_hours=0.25,
        reserved_output_bytes=16 * 1024 ** 2, timeout_seconds=900,
        owned_checks=("positive_base_real_exponent_plga",),
    ))
    intervention = output_root / "I" / "intervention-decision.json"
    jobs.append(_job(
        "I", "paired-interventions", "heldout", "cuda:0",
        "native-paired-short-continuations",
        [
            "python3", CONTROLS, "intervention", "--checkpoint-5555",
            run_root / RUNS[1]["name"] / "ckpt_{anchor}.pt",
            "--checkpoint-6666",
            run_root / RUNS[2]["name"] / "ckpt_{anchor}.pt",
            "--tokens", tokens, "--registry", registry_path,
            "--lock", lock_path, "--device", "cuda:0",
            "--job-id", "paired-interventions", "--output", intervention,
        ], intervention, dependencies=("M:plga-centered-logit",),
        reserved_gpu_hours=7.5, reserved_wall_clock_hours=8.0,
        reserved_output_bytes=512 * 1024 ** 2, timeout_seconds=28800,
        owned_checks=tuple(STAGE_BY_ID["I"]["required_checks"]),
    ))
    final = output_root / "F" / "final-decision.json"
    jobs.append(_job(
        "F", "s1-s5", "shared", "cpu", "final-decision",
        [
            "python3", __file__, "finalize", "--lock", lock_path,
            "--heldout", *heldout_reports, "--intervention", intervention,
            "--plga", plga, "--rob", rob_report, "--output", final,
            "--require-pass",
        ], final, dependencies=("I:paired-interventions",),
        reserved_wall_clock_hours=0.25,
        reserved_output_bytes=16 * 1024 ** 2, timeout_seconds=900,
        owned_checks=("fresh_process_reanalysis", "schema_and_digest_replay"),
    ))
    release = output_root / "F" / "release-verification.json"
    tests = [
        ROOT / "experiments" / "tests" / "test_observable_margin.py",
        ROOT / "experiments" / "tests" / "test_observable_margin_pipeline.py",
        ROOT / "experiments" / "tests" / "test_observable_margin_executor.py",
        ROOT / "experiments" / "tests" / "test_observable_margin_environment.py",
        ROOT / "experiments" / "tests" / "test_current_release_gate.py",
    ]
    jobs.append(_job(
        "F", "release-verification", "shared", "cpu",
        "release-verification",
        [
            "python3", CONTROLS, "release", "--plan",
            output_root.parent / "execution_plan.json",
            "--environment-attestation", environment_attestation,
            "--campaign-ledger",
            campaign_ledger, "--staged-root", output_root.parent,
            "--results-root", output_root, "--final", final, "--lock",
            lock_path, "--heldout", *heldout_reports, "--intervention",
            intervention, "--plga", plga, "--rob", rob_report,
            "--repository-manifest", repository_manifest,
            "--staged-repository-manifest", staged_repository_manifest,
            "--manuscript-export-root", manuscript_export_root,
            "--manuscript-export-manifest", manuscript_export_manifest,
            "--staged-manuscript-export-manifest",
            staged_manuscript_export_manifest,
            "--tests", *tests, "--repository-release-command",
            "make", "release-check-current",
            "--job-id", "release-verification",
            "--output", release,
        ], release, dependencies=("F:s1-s5",),
        reserved_wall_clock_hours=2.0,
        reserved_output_bytes=64 * 1024 ** 2, timeout_seconds=7200,
        owned_checks=tuple(STAGE_BY_ID["F"]["required_checks"]),
    ))
    _validate_serial_resource_envelope(jobs)
    plan = {
        "schema_version": PLAN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "status": "READY_FOR_T",
        "repository_root": str(ROOT),
        "repository_manifest": str(repository_manifest),
        "staged_repository_manifest": str(staged_repository_manifest),
        "run_root": str(run_root),
        "tokens": str(tokens),
        "tokenizer": str(tokenizer),
        "registry": str(registry_path),
        "environment_attestation": str(environment_attestation),
        "manuscript_export_root": str(manuscript_export_root),
        "manuscript_export_manifest": str(manuscript_export_manifest),
        "staged_manuscript_export_manifest": str(
            staged_manuscript_export_manifest),
        "output_root": str(output_root),
        "campaign_attempt_ledger": str(campaign_ledger),
        "stage_order": [stage["id"] for stage in STAGES],
        "stage_dependencies": {
            stage["id"]: stage["dependencies"] for stage in STAGES},
        "dependency_semantics": {
            "ROB_to_M": (
                "M waits for any terminal ROB record for provenance ordering; "
                "ROB PASS is not required and ROB cannot enter S1-S5."),
            "executor_order": (
                "strictly-serial-topological-order-with-cumulative-resource-"
                "accounting-v1"),
        },
        "native_update": "registered-float32-adamw-actual-minibatches",
        "deciding_arithmetic": "float64-shadow-with-measured-roundoff-charge",
        "compact_ledger_schema": "pldr-observable-margin-ledger-v1",
        "instrumentation_contract": (
            "The GPU producer evaluates one hash-bound probe registry at every "
            "node while actual minibatches drive updates, stores normalized "
            "edge contrasts but no raw parameter-node or full normalized-node "
            "arrays, and never telescopes measurements from changing probes."
        ),
        "resource_caps": RESOURCE_CAPS,
        "stop_gates": list(STOP_GATES),
        "jobs": jobs,
    }
    keys = [job["record_key"] for job in jobs]
    if len(keys) != len(set(keys)):
        raise ArithmeticError("execution plan repeats a stage/job record key")
    plan["plan_sha256"] = digest_object(plan)
    return plan

def select_construction(reports):
    candidates = []
    for path in reports:
        report = _load_sealed(
            path, "pldr-observable-margin-analysis-v1", "analysis_sha256")
        if report.get("role") != "construction" or report.get("seed") != 4444:
            raise ValueError("construction lock attempted to open nonconstruction evidence")
        cell = (report["anchor"], report["layer"], report["block_length"])
        if (
            cell[0] not in ANCHORS["construction_window_starts"]
            or cell[1] not in ANCHORS["layers"]
            or cell[2] not in ANCHORS["candidate_block_lengths"]
        ):
            raise ValueError("construction report lies outside the frozen grid")
        if report.get("decision") == "QUALIFIED":
            candidates.append((
                -float(report["metrics"]["minimum_relative_margin_sum"]),
                cell[2], cell[0], cell[1], report, str(Path(path).resolve()),
            ))
    if not candidates:
        raise ValueError("no construction cell satisfies the registered margin")
    _, block, anchor, layer, report, path = min(candidates)
    ledger_path = Path(report["source"]["path"]).resolve()
    if sha256_path(ledger_path) != report["source"]["sha256"]:
        raise ValueError("selected construction ledger changed after analysis")
    raw = load_npz(ledger_path, required=(
        "affine_factor", "affine_forcing", "roundoff_charge",
        "effective_resistance_max", "producer_energy", "vertex_diameter",
        "adaptive_direction",
        "bridge_sector_direction", "union_graph_sha256",
        "graph_rule", "graph_transfer_mapping", "graph_vertex_count",
        "graph_edges", "graph_weights", "block_vertex_offsets",
        "block_head_index", "union_source_physical_sha256",
        "plga_pair_id", "plga_context_index", "plga_head",
        "plga_anchor_row", "plga_secant_operator_bound",
        "plga_downstream_score_ratio", "plga_generator_difference_norm",
        "plga_query_operator_norm", "plga_key_operator_norm",
        "probe_registry_sha256",
    ))
    safety = TRANSFER["construction_safety_multiplier"]
    factor_upper = np.max(np.asarray(raw["affine_factor"], dtype=np.float64), axis=1)
    forcing_upper = safety * np.max(
        np.asarray(raw["affine_forcing"], dtype=np.float64), axis=1)
    roundoff_upper = safety * np.max(
        np.asarray(raw["roundoff_charge"], dtype=np.float64), axis=1)
    graph_digest = str(np.asarray(raw["union_graph_sha256"]).item())
    graph_rule = str(np.asarray(raw["graph_rule"]).item())
    graph_mapping = str(np.asarray(raw["graph_transfer_mapping"]).item())
    union_graph = unpack_union_graph(
        int(np.asarray(raw["graph_vertex_count"]).item()),
        raw["graph_edges"], raw["graph_weights"], rule=graph_rule,
        neighbors=REGISTRY["neighbors"], expected_sha256=graph_digest)
    resistance_values = np.asarray(
        raw["effective_resistance_max"], dtype=np.float64)
    if resistance_values.shape != (1,):
        raise ValueError("construction union resistance is incomplete")
    resistance_upper = float(resistance_values[0])
    source_energy_upper = float(np.max(np.asarray(raw["producer_energy"])[0]))
    energy_path = [source_energy_upper]
    for factor_value, forcing_value in zip(
        factor_upper, forcing_upper, strict=True,
    ):
        energy_path.append(float(
            factor_value * energy_path[-1] + forcing_value))
    diameter_path = [math.sqrt(max(resistance_upper * value, 0.0))
                     for value in energy_path]
    plga_context = np.asarray(raw["plga_context_index"], dtype=np.int64)
    plga_head = np.asarray(raw["plga_head"], dtype=np.int64)
    block_heads = np.asarray(raw["block_head_index"], dtype=np.int64)
    block_offsets = np.asarray(raw["block_vertex_offsets"], dtype=np.int64)
    block_count = REGISTRY["construction_contexts"]
    if (
        not np.array_equal(plga_context, np.arange(block_count))
        or not np.array_equal(plga_head, block_heads)
        or not np.array_equal(
            block_offsets,
            np.arange(block_count + 1) * ARCHITECTURE["row_width"])
    ):
        raise ValueError("construction PLGA blocks are not the union subsets")
    plga_operator_upper = safety * np.asarray(
        raw["plga_secant_operator_bound"], dtype=np.float64)
    plga_downstream_upper = safety * np.asarray(
        raw["plga_downstream_score_ratio"], dtype=np.float64)
    plga_query_upper = safety * np.asarray(
        raw["plga_query_operator_norm"], dtype=np.float64)
    plga_key_upper = safety * np.asarray(
        raw["plga_key_operator_norm"], dtype=np.float64)
    plga_generator_upper = np.full(
        block_count,
        math.sqrt(max(
            ARCHITECTURE["row_width"] * resistance_upper * energy_path[-1],
            0.0)),
        dtype=np.float64)
    plga_centered_upper = (
        plga_downstream_upper * plga_query_upper * plga_key_upper
        * plga_operator_upper * plga_generator_upper
        / math.sqrt(ARCHITECTURE["row_width"]))
    adaptive_upper = safety * np.linalg.norm(
        np.asarray(raw["adaptive_direction"], dtype=np.float64), axis=1)
    sector_upper = safety * np.linalg.norm(
        np.asarray(raw["bridge_sector_direction"], dtype=np.float64), axis=2)
    lock = {
        "schema_version": LOCK_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "source_owned": True,
        "heldout_artifacts_opened": [],
        "selected": {"anchor": anchor, "layer": layer, "block_length": block},
        "selection_metric": "minimum_relative_margin_sum",
        "selection_value": report["metrics"]["minimum_relative_margin_sum"],
        "selection_rule": ANCHORS["selection_rule"],
        "lower_bound_rule": TRANSFER["lower_bound_rule"],
        "safety_multiplier": safety,
        "enlargement_allowed": False,
        "numerical_nonvacuity": {
            "relative_margin_floor": float(
                report["metrics"]["deciding_relative_margin_floor"]),
            "rule": (
                "multiplier-times-block-length-times-relative-plus-"
                "absolute-over-source-energy-v1"),
        },
        "diameter_nonvacuity": {
            "construction_source_vertex_diameter": float(np.asarray(
                raw["vertex_diameter"], dtype=np.float64)[0, 0]),
            "endpoint_usefulness_target": float(
                report["metrics"]["diameter_endpoint_usefulness_target"]),
            "endpoint_fraction_of_construction_source": TRANSFER[
                "diameter_endpoint_fraction_of_construction_source"],
            "bound_to_observed_ratio_max": TRANSFER[
                "diameter_bound_to_observed_ratio_max"],
        },
        "graph": {
            "rule": graph_rule,
            "neighbors": REGISTRY["neighbors"],
            "mapping": graph_mapping,
            "union_graph_sha256": graph_digest,
            "vertex_count": int(union_graph["vertex_count"]),
            "edges": np.asarray(raw["graph_edges"], dtype=np.int64).tolist(),
            "weights": np.asarray(
                raw["graph_weights"], dtype=np.float64).tolist(),
            "effective_resistance": resistance_upper,
            "source_physical_sha256": str(np.asarray(
                raw["union_source_physical_sha256"]).item()),
        },
        "intervention": {
            "unit_registry_sha256": str(
                np.asarray(raw["probe_registry_sha256"]).item()),
            "sites_per_comparison": STATISTICS["intervention_sites"],
            "heldout_seed_site_count": {
                str(seed): STATISTICS["sites_per_heldout_seed"]
                for seed in STATISTICS["heldout_seeds"]},
            "short_continuation_steps": REGISTRY["intervention_steps"],
            "comparisons": [
                {
                    **dict(contract),
                    "direction": 1.0,
                    "effect_lower_threshold": float(
                        report["metrics"]["deciding_relative_margin_floor"]
                        * source_energy_upper),
                }
                for contract in INTERVENTION_OPERATOR_CONTRACTS
            ],
            "lower_effect_order_statistic_per_seed": STATISTICS[
                "lower_effect_order_statistic_per_seed"],
            "positive_sites_required_per_seed": STATISTICS[
                "positive_sites_required_per_seed"],
            "criterion": STATISTICS["deterministic_criterion"],
        },
        "plga": {
            "pair_rule": "same-context-collapsed-anchor-union-subset-v1",
            "selected_layer": layer,
            "selected_node": anchor + block,
            "anchor_row": 0,
            "ordinal_context_head_binding": [
                {"ordinal": int(index), "head": int(head), "anchor_row": 0}
                for index, head in enumerate(block_heads)
            ],
            "construction_comparison_id_by_ordinal": np.asarray(
                raw["plga_pair_id"], dtype=np.str_).tolist(),
            "block_vertex_offsets": block_offsets.tolist(),
            "block_head_index": block_heads.tolist(),
            "secant_operator_upper_by_context_head": (
                plga_operator_upper.tolist()),
            "downstream_response_envelope_by_context_head": (
                plga_downstream_upper.tolist()),
            "stacked_generator_upper_by_context_head": (
                plga_generator_upper.tolist()),
            "query_operator_upper_by_context_head": (
                plga_query_upper.tolist()),
            "key_operator_upper_by_context_head": (
                plga_key_upper.tolist()),
            "centered_logit_remainder_upper_by_context_head": (
                plga_centered_upper.tolist()),
            "exact_score_rule": "Q-deltaG-K-transpose-over-sqrt-d-v1",
        },
        "frozen_envelopes": {
            "affine_factor_upper_by_step": factor_upper.tolist(),
            "affine_forcing_upper_by_step": forcing_upper.tolist(),
            "roundoff_charge_upper_by_step": roundoff_upper.tolist(),
            "adaptive_direction_norm_upper_by_step": adaptive_upper.tolist(),
            "bridge_sector_norm_upper_by_step": sector_upper.tolist(),
            "source_energy_upper": source_energy_upper,
            "source_diameter_bound_upper": diameter_path[0],
            "effective_resistance_upper": resistance_upper,
            "energy_upper_path": energy_path,
            "diameter_upper_path": diameter_path,
        },
        "bindings": {
            "probe_registry_sha256": str(
                np.asarray(raw["probe_registry_sha256"]).item()),
            "source_ledger": {
                "path": str(ledger_path), "sha256": sha256_path(ledger_path)},
        },
        "source_report": {
            "path": path,
            "sha256": report["analysis_sha256"],
            "affine_bound_nonvacuity": report["affine_bound_nonvacuity"],
        },
    }
    lock["lock_sha256"] = digest_object(lock)
    return lock


def plga_decision(lock_path, heldout_paths):
    lock = _load_sealed(lock_path, LOCK_SCHEMA, "lock_sha256")
    reports = [
        _load_sealed(path, "pldr-observable-margin-analysis-v1", "analysis_sha256")
        for path in heldout_paths]
    by_seed = {}
    qualified = []
    for report in reports:
        seed = int(report["seed"])
        rows = report["metrics"]["plga_pairs"]
        local = []
        for row in rows:
            value = dict(row)
            value["seed"] = seed
            value["role"] = report["role"]
            if value.get("qualified") is True:
                local.append(value)
                qualified.append(value)
        by_seed[str(seed)] = {
            "registered_pair_count": len(rows),
            "qualified_pair_count": len(local),
            "nonempty": bool(local),
            "analysis_sha256": report["analysis_sha256"],
        }
    expected = set(STATISTICS["heldout_seeds"])
    passed = bool(
        set(map(int, by_seed)) == expected
        and all(value["nonempty"] for value in by_seed.values()))
    value = {
        "schema_version": "pldr-observable-plga-margin-v1",
        "campaign_id": CAMPAIGN_ID,
        "construction_lock_sha256": lock["lock_sha256"],
        "pair_rule": lock["plga"]["pair_rule"],
        "roles": by_seed,
        "qualified_centered_logit_pairs": qualified,
        "status": "PASS" if passed else "FAIL",
    }
    value["margin_sha256"] = digest_object(value)
    return value


def final_decision(lock_path, heldout_paths, intervention_path, plga_path, rob_path):
    lock = _load_sealed(lock_path, LOCK_SCHEMA, "lock_sha256")
    selected = lock["selected"]
    heldout = [
        _load_sealed(path, "pldr-observable-margin-analysis-v1", "analysis_sha256")
        for path in heldout_paths
    ]
    heldout_seeds = sorted(report.get("seed") for report in heldout)
    unchanged = all(
        report.get("role") == "heldout"
        and report.get("anchor") == selected["anchor"]
        and report.get("layer") == selected["layer"]
        and report.get("block_length") == selected["block_length"]
        for report in heldout
    )
    intervention = _load_sealed(
        intervention_path, "pldr-observable-intervention-decision-v1",
        "decision_sha256")
    comparisons = intervention.get("comparisons", [])
    unit_groups = intervention.get("units", [])
    detailed_units_pass = bool(
        [group.get("name") for group in unit_groups]
        == list(INTERVENTION_COMPARISON_NAMES)
        and all(
            len(group.get("units", [])) == STATISTICS["intervention_sites"]
            and {unit.get("seed") for unit in group.get("units", [])}
            == set(STATISTICS["heldout_seeds"])
            and all(
                sum(unit.get("seed") == seed for unit in group.get("units", []))
                == STATISTICS["sites_per_heldout_seed"]
                for seed in STATISTICS["heldout_seeds"])
            and all(
                unit.get("operator_contract")
                == dict(INTERVENTION_OPERATOR_BY_NAME[group["name"]])
                and unit.get("operator_contract_unchanged") is True
                and unit.get("common_initial_state") is True
                and unit.get("common_minibatch_sequence") is True
                and unit.get("intended_intervention_target_changed") is True
                and unit.get("executed_successor_changed") is True
                and unit.get("direct_operator_non_target_invariant") is True
                and unit.get(
                    "selected_gate_optimizer_state_retained") is True
                and unit.get(
                    "native_selected_gate_moment_transition_valid") is True
                and unit.get(
                    "registered_scheduler_sequence_unchanged") is True
                and unit.get(
                    "per_step_execution_evidence_complete") is True
                and unit.get(
                    "normalized_shape_activation_binding_valid") is True
                and unit.get("clamp_raw_gate_gradient_changed") is True
                and unit.get("no_op_rejected") is True
                and unit.get("locked_direction_unchanged") is True
                and unit.get(
                    "locked_lower_effect_interval_unchanged") is True
                for unit in group.get("units", []))
            for group in unit_groups))
    replayed_comparisons = None
    if detailed_units_pass:
        from observable_margin_controls import validate_intervention_units
        replayed_comparisons = validate_intervention_units(unit_groups, lock)
    expected_stage_checks = set(STAGE_BY_ID["I"]["required_checks"])
    intervention_pass = bool(
        intervention.get("status") == "PASS"
        and intervention.get("construction_lock_sha256") == lock["lock_sha256"]
        and detailed_units_pass
        and comparisons == replayed_comparisons
        and set(intervention.get("stage_checks", {})) == expected_stage_checks
        and all(intervention["stage_checks"].values())
        and len(comparisons) == STATISTICS["intervention_comparisons"]
        and [row.get("name") for row in comparisons]
        == list(INTERVENTION_COMPARISON_NAMES)
        and all(
            row.get("sites") == STATISTICS["intervention_sites"]
            and row.get("sites_by_seed") == {
                str(seed): STATISTICS["sites_per_heldout_seed"]
                for seed in STATISTICS["heldout_seeds"]}
            and all(
                count >= STATISTICS["positive_sites_required_per_seed"]
                for count in row.get(
                    "positive_direction_sites_by_seed", {}).values())
            and row.get("lower_effect_order_statistic_per_seed")
            == STATISTICS["lower_effect_order_statistic_per_seed"]
            and all(
                value >= row.get(
                    "locked_effect_lower_threshold", math.inf)
                and value > 0.0
                for value in row.get("lower_effect_by_seed", {}).values())
            and row.get("seed_stratified_sign_coverage_pass") is True
            and row.get("seed_stratified_effect_floor_pass") is True
            and row.get("directional_prediction_pass") is True
            and row.get("registered_intervention_execution_pass") is True
            and row.get("operator_contract_pass") is True
            and row.get(
                "intended_intervention_target_change_pass") is True
            and row.get("executed_successor_change_pass") is True
            and row.get(
                "direct_operator_non_target_invariance_pass") is True
            and row.get(
                "selected_gate_optimizer_state_retention_pass") is True
            and row.get(
                "native_selected_gate_moment_transition_pass") is True
            and row.get(
                "registered_scheduler_sequence_pass") is True
            and row.get(
                "per_step_execution_evidence_pass") is True
            and row.get(
                "normalized_shape_activation_binding_pass") is True
            and row.get("clamp_raw_gate_gradient_change_pass") is True
            and row.get("no_op_rejection_pass") is True
            and row.get("locked_direction_pass") is True
            and row.get("locked_lower_effect_interval_pass") is True
            for row in comparisons))
    plga = _load_sealed(
        plga_path, "pldr-observable-plga-margin-v1", "margin_sha256")
    qualified_pairs = plga.get("qualified_centered_logit_pairs", [])
    plga_pass = bool(
        plga.get("status") == "PASS"
        and set(map(int, plga.get("roles", {})))
        == set(STATISTICS["heldout_seeds"])
        and all(value.get("nonempty") is True
                for value in plga.get("roles", {}).values())
        and qualified_pairs
    ) and all(
        row.get("positive_bases") is True
        and row.get("real_exponent") is True
        and row.get("centered_logit_gap", 0.0)
        > 2.0 ** 0.5 * row.get("remainder_bound", float("inf"))
        and row.get("heldout_transfer_without_enlargement") is True
        for row in qualified_pairs
    )
    source_report = _load_sealed(
        lock["source_report"]["path"],
        "pldr-observable-margin-analysis-v1", "analysis_sha256")
    heldout_lower_bound = min(
        float(report["metrics"]["minimum_relative_margin_sum"])
        for report in heldout
    )
    transfer_rows = [report.get("construction_lock_transfer") for report in heldout]
    algebra_prerequisite = bool(all(
        report.get("algebra_status") == "PASS"
        and report["checks"].get("deciding_signed_credit_absolute_charge_rule")
        and report["checks"].get("optimizer_decay_mask_verified")
        for report in [source_report, *heldout]
    ))
    s1 = bool(
        algebra_prerequisite
        and heldout_seeds == STATISTICS["heldout_seeds"]
        and all(report.get("positive_margin") is True for report in heldout)
        and heldout_lower_bound > lock["numerical_nonvacuity"][
            "relative_margin_floor"]
        and lock["lower_bound_rule"] == TRANSFER["lower_bound_rule"]
    )
    frozen = lock["frozen_envelopes"]
    frozen_nonvacuous = bool(
        frozen["energy_upper_path"][-1] < frozen["energy_upper_path"][0]
        and frozen["diameter_upper_path"][-1] < frozen["diameter_upper_path"][0]
    )
    s2 = bool(
        algebra_prerequisite and frozen_nonvacuous
        and all(row is not None for row in transfer_rows)
        and all(
            row["energy_coverage_relative_excess"]
            <= ARITHMETIC["relative_identity_tolerance"]
            and row["diameter_coverage_relative_excess"]
            <= ARITHMETIC["relative_identity_tolerance"]
            for row in transfer_rows
        )
        and all(report.get("diameter_bound_useful") is True
                for report in heldout)
    )
    s3 = intervention_pass
    s4 = bool(
        algebra_prerequisite
        and intervention.get("construction_lock_sha256") == lock["lock_sha256"]
        and detailed_units_pass
        and all(
            row.get("locked_direction_pass") is True
            and row.get("locked_lower_effect_interval_pass") is True
            for row in comparisons)
        and heldout_seeds == STATISTICS["heldout_seeds"] and unchanged
        and all(row is not None and row["enlargement_allowed"] is False
                for row in transfer_rows)
        and all(
            report["checks"].get("construction_source_envelopes_not_enlarged")
            and report["checks"].get(
                "construction_numerical_margin_floor_not_enlarged")
            and report["checks"].get("frozen_affine_arrays_unchanged")
            and report["checks"].get("frozen_energy_and_diameter_paths_cover")
            and report["checks"].get("construction_probe_binding_unchanged")
            and report["checks"].get("construction_graph_rule_unchanged")
            and report["checks"].get("construction_graph_mapping_unchanged")
            and report["checks"].get("construction_union_graph_digest_unchanged")
            for report in heldout
        )
    )
    s5 = plga_pass
    rob = _json(rob_path) if Path(rob_path).is_file() else {
        "status": "MISSING", "scientific_gate": False}
    decision = {
        "schema_version": FINAL_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "construction_lock_sha256": lock["lock_sha256"],
        "selected": selected,
        "algebraic_prerequisite": algebra_prerequisite,
        "heldout_simultaneous_relative_margin_lower_bound": heldout_lower_bound,
        "statements": {
            "S1_positive_heldout_margin": s1,
            "S2_nonvacuous_frozen_energy_and_diameter_bound": s2,
            "S3_causal_source_response": s3,
            "S4_fresh_seed_transfer_without_enlargement": s4,
            "S5_nonempty_centered_logit_plga_margin": s5,
        },
        "robustness": {
            "scientific_gate": False,
            "terminal_record_present": Path(rob_path).is_file(),
            "reported_status": rob.get("status", rob.get("decision", "UNKNOWN")),
        },
        "stage_checks": {
            "fresh_process_reanalysis": True,
            "schema_and_digest_replay": True,
        },
        "heldout_report_sha256": [row["analysis_sha256"] for row in heldout],
        "intervention_decision_sha256": intervention["decision_sha256"],
        "plga_margin_sha256": plga["margin_sha256"],
        "status": (
            "CONFIRMED" if algebra_prerequisite
            and all((s1, s2, s3, s4, s5)) else "NOT_CONFIRMED"),
    }
    decision["final_sha256"] = digest_object(decision)
    return decision




