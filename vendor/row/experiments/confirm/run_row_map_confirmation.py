#!/usr/bin/env python3
"""Build deterministic execution commands for row-map confirmation."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
import shlex
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from gate_shape_evidence import digest_object, write_json_atomic  # noqa: E402
from row_map_confirmation_specs import (  # noqa: E402
    CHECKPOINTS,
    FIXED_CONSTANTS,
    RESOURCE_CAPS,
    RUNS,
    STAGES,
)


LIVE = "experiments/confirm/row_map_live.py"
PLGA = "experiments/confirm/row_map_plga_live.py"
ANALYZER = "experiments/analysis/analyze_row_map_confirmation.py"


def _command(values):
    return shlex.join([str(value) for value in values])


def _checkpoint(run_root, run_name, step):
    return Path(run_root) / run_name / CHECKPOINTS[
        "checkpoint_file_by_step"][step]


def _job(stage, job_id, device, command, output):
    return {
        "stage": stage,
        "job_id": job_id,
        "device": device,
        "output": str(output),
        "command": command,
    }


def _load_signed_json(path, schema, digest_field):
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if value.get("schema_version") != schema:
        raise ValueError(f"unexpected schema in {path}")
    unsigned = dict(value)
    recorded = unsigned.pop(digest_field, None)
    if recorded != digest_object(unsigned):
        raise ValueError(f"digest does not replay for {path}")
    return value


def _analysis_job(stage, kind, inputs, output, *, require_pass=False):
    values = [
        "python3", ANALYZER, kind, *inputs, "--output", output,
    ]
    if require_pass:
        values.append("--require-pass")
    return _job(
        stage, f"analyze-{kind}", "cpu", _command(values), output)


def execution_plan(arguments):
    run_root = Path(arguments.run_root).resolve()
    tokens = Path(arguments.tokens).resolve()
    tokenizer = Path(arguments.tokenizer).resolve()
    output_root = Path(arguments.output_root).resolve()
    registry = output_root / "registry.json"
    inventory = output_root / "inventory.json"
    devices = [value for value in arguments.devices.split(",") if value]
    if len(devices) != 2 or len(set(devices)) != 2:
        raise ValueError("the plan requires two distinct registered devices")
    jobs = [
        _job(
            "setup", "inventory", "cpu",
            _command([
                "python3", LIVE, "inventory", "--run-root", run_root,
                "--output", inventory,
            ]), inventory),
        _job(
            "setup", "registry", "cpu",
            _command([
                "python3", LIVE, "registry", "--tokens", tokens,
                "--tokenizer", tokenizer, "--construction-contexts",
                FIXED_CONSTANTS["construction_contexts"],
                "--validation-contexts",
                FIXED_CONSTANTS["validation_contexts"], "--output", registry,
            ]), registry),
    ]

    analysis_root = output_root / "R"
    analysis_outputs = {
        "geometry": analysis_root / "geometry-analysis.json",
        "loss": analysis_root / "loss-analysis.json",
        "blocks": analysis_root / "block-analysis.json",
        "dynamics": analysis_root / "dynamics-analysis.json",
        "plga": analysis_root / "plga-analysis.json",
    }

    construction = RUNS[0]
    q_outputs = []
    for device in devices:
        tag = device.replace(":", "-")
        output = output_root / "Q" / f"qualification-{tag}.npz"
        q_outputs.append(output)
        jobs.append(_job(
            "Q", f"qualification-{tag}", device,
            _command([
                "python3", LIVE, "optimizer-block", "--checkpoint",
                _checkpoint(run_root, construction["name"], 24000),
                "--tokens", tokens, "--registry", registry, "--layer", 2,
                "--qualification", "--device", device, "--output", output,
            ]), output))
    qualification_analysis = output_root / "Q" / "qualification-analysis.json"
    jobs.append(_analysis_job(
        "Q", "blocks", q_outputs, qualification_analysis, require_pass=True))

    geometry_outputs = []
    loss_outputs = []
    for run_index, run in enumerate(RUNS):
        role = run["ownership"]
        for layer in range(3):
            for step_index, step in enumerate(CHECKPOINTS["geometry_steps"]):
                output = (
                    output_root / "G"
                    / f"{run['name']}-step{step}-layer{layer}-{role}.npz")
                geometry_outputs.append(output)
                device = devices[(run_index + layer + step_index) % 2]
                values = [
                    "python3", LIVE, "geometry", "--checkpoint",
                    _checkpoint(run_root, run["name"], step), "--tokens", tokens,
                    "--registry", registry, "--role", role, "--layer", layer,
                    "--device", device, "--output", output,
                ]
                if step_index + 1 < len(CHECKPOINTS["geometry_steps"]):
                    target = CHECKPOINTS["geometry_steps"][step_index + 1]
                    values.extend([
                        "--target-checkpoint",
                        _checkpoint(run_root, run["name"], target),
                    ])
                jobs.append(_job(
                    "G", f"{run['name']}-{step}-l{layer}", device,
                    _command(values), output))
            for anchor_index, step in enumerate(
                    CHECKPOINTS["loss_sector_steps"]):
                output = (
                    output_root / "H"
                    / f"{run['name']}-step{step}-layer{layer}-{role}.npz")
                loss_outputs.append(output)
                device = devices[(run_index + layer + anchor_index) % 2]
                jobs.append(_job(
                    "H", f"{run['name']}-{step}-l{layer}", device,
                    _command([
                        "python3", LIVE, "geometry", "--checkpoint",
                        _checkpoint(run_root, run["name"], step),
                        "--tokens", tokens, "--registry", registry,
                        "--role", role, "--layer", layer, "--loss-sector",
                        "--device", device, "--output", output,
                    ]), output))
    jobs.append(_analysis_job(
        "G", "geometry", geometry_outputs, analysis_outputs["geometry"]))
    jobs.append(_analysis_job(
        "H", "loss", loss_outputs, analysis_outputs["loss"]))

    block_outputs = list(q_outputs)
    dynamics_outputs = []
    for run_index, run in enumerate(RUNS):
        for layer in range(3):
            device = devices[(run_index + layer) % 2]
            output = (
                output_root / "D"
                / f"{run['name']}-step24000-layer{layer}.npz")
            block_outputs.append(output)
            jobs.append(_job(
                "D", f"{run['name']}-24000-l{layer}", device,
                _command([
                    "python3", LIVE, "optimizer-block", "--checkpoint",
                    _checkpoint(run_root, run["name"], 24000),
                    "--tokens", tokens, "--registry", registry,
                    "--layer", layer, "--device", device, "--output", output,
                ]), output))
            continuation_output = (
                output_root / "D" / "continuations"
                / run["name"] / f"layer{layer}" / "baseline_continue")
            dynamics_outputs.append(continuation_output / "observables.npz")
            jobs.append(_job(
                "D", f"{run['name']}-24000-l{layer}-window", device,
                _command([
                    "python3", LIVE, "continue", "--checkpoint",
                    _checkpoint(run_root, run["name"], 24000),
                    "--tokens", tokens, "--registry", registry,
                    "--arm", "baseline_continue", "--layer", layer,
                    "--updates", FIXED_CONSTANTS["continuation_updates"],
                    "--shape-jvp", "--capture-blocks", "--device", device,
                    "--output", continuation_output,
                ]), continuation_output))
    jobs.append(_analysis_job(
        "D", "blocks", block_outputs, analysis_outputs["blocks"]))
    jobs.append(_analysis_job(
        "D", "dynamics", dynamics_outputs, analysis_outputs["dynamics"]))

    plga_outputs = []
    for run_index, run in enumerate(RUNS):
        role = run["ownership"]
        for pair_index in range(12):
            pair_id = f"plga-{role[0]}-{pair_index:02d}"
            device = devices[(run_index + pair_index) % 2]
            output = output_root / "A" / f"{run['name']}-{pair_id}.npz"
            plga_outputs.append(output)
            jobs.append(_job(
                "A", f"{run['name']}-{pair_id}", device,
                _command([
                    "python3", PLGA, "--checkpoint",
                    _checkpoint(run_root, run["name"], 24000),
                    "--tokens", tokens, "--registry", registry,
                    "--role", role, "--pair-id", pair_id,
                    "--device", device, "--output", output,
                ]), output))
    jobs.append(_analysis_job(
        "A", "plga", plga_outputs, analysis_outputs["plga"]))

    plan = {
        "schema_version": "pldr-row-map-execution-plan-v1",
        "status": "READY_FOR_GATED_EXECUTION",
        "repository_root": str(ROOT),
        "run_root": str(run_root),
        "tokens": str(tokens),
        "tokenizer": str(tokenizer),
        "output_root": str(output_root),
        "registry": str(registry),
        "stage_order": ["setup", "Q", "G", "H", "D", "A"],
        "required_live_stage_sequence": [
            "setup", "Q", "G", "H", "D", "A",
        ],
        "stage_dependencies": {
            "setup": [],
            "Q": ["setup"],
            "G": ["Q"],
            "H": ["Q"],
            "D": ["G", "H"],
            "A": ["G", "H"],
        },
        "resource_caps": RESOURCE_CAPS,
        "jobs": jobs,
        "deferred_interventions": {
            "reason": (
                "Stage I is generated from digest-sealed construction "
                "geometry after every base-stage record is complete."),
            "selection_schema": "pldr-row-map-intervention-selection-v1",
            "selection_rule": (
                "all construction layers with an energy ratio at most one "
                "third; otherwise the unique minimum-ratio layer with "
                "lower-index tie breaking"),
            "final_campaign_analysis": "DEFERRED_UNTIL_STAGE_I",
        },
        "stop_conditions": [
            "aggregate_gpu_hours_above_2.4",
            "wall_clock_hours_above_2.0",
            "nonfinite_state", "digest_mismatch", "schema_failure",
            "qualification_failure",
        ],
    }
    plan["plan_sha256"] = digest_object(plan)
    return plan


def select_interventions(arguments):
    base_plan = _load_signed_json(
        arguments.base_plan, "pldr-row-map-execution-plan-v1", "plan_sha256")
    geometry = _load_signed_json(
        arguments.geometry_analysis, "pldr-row-map-analysis-v1",
        "analysis_sha256")
    if geometry.get("kind") != "geometry":
        raise ValueError("intervention selection needs geometry analysis")
    expected = [
        job for job in base_plan["jobs"]
        if job["job_id"] == "analyze-geometry"]
    if len(expected) != 1 or (
            Path(expected[0]["output"]).resolve()
            != Path(arguments.geometry_analysis).resolve()):
        raise ValueError("geometry analysis is not bound to the base plan")
    required_exact = (
        "all_node_identities", "all_graph_bounds",
        "all_covered_domain_bounds",
    )
    if not all(geometry["checks"].get(name, False) for name in required_exact):
        raise ValueError("construction selection requires exact geometry replay")
    construction_seed = int(RUNS[0]["seed"])
    layer_ratios = {}
    for row in geometry["results"].get("intervals", []):
        if row.get("role") != "construction" or int(row["seed"]) != construction_seed:
            continue
        ratio = row.get("energy_ratio")
        if ratio is None:
            continue
        ratio = float(ratio)
        if not math.isfinite(ratio) or ratio < 0.0:
            raise ValueError("construction energy ratio is invalid")
        layer = int(row["layer"])
        layer_ratios.setdefault(layer, []).append(ratio)
    if set(layer_ratios) != {0, 1, 2}:
        raise ValueError("construction analysis does not cover every layer")
    threshold = 1.0 / float(FIXED_CONSTANTS["material_energy_factor"])
    layers = sorted(
        layer for layer, ratios in layer_ratios.items()
        if min(ratios) <= threshold)
    fallback = not layers
    if fallback:
        layers = [min(
            layer_ratios,
            key=lambda layer: (min(layer_ratios[layer]), layer))]
    minimum_ratios = {
        str(layer): min(values)
        for layer, values in sorted(layer_ratios.items())
    }
    selection = {
        "schema_version": "pldr-row-map-intervention-selection-v1",
        "selection_status": "SEALED_FROM_CONSTRUCTION_GEOMETRY",
        "base_plan_sha256": base_plan["plan_sha256"],
        "geometry_analysis_sha256": geometry["analysis_sha256"],
        "construction_seed": construction_seed,
        "material_energy_ratio_threshold": threshold,
        "minimum_construction_ratio_by_layer": minimum_ratios,
        "fallback_minimum_ratio_rule_used": fallback,
        "layers": layers,
        "selection_rule": (
            "select every construction layer with minimum interval energy "
            "ratio at most one third; if the set is empty, select the "
            "minimum-ratio layer with lower-index tie breaking"),
    }
    selection["selection_sha256"] = digest_object(selection)
    return selection


def intervention_plan(arguments):
    selection = _load_signed_json(
        arguments.selection, "pldr-row-map-intervention-selection-v1",
        "selection_sha256")
    base_plan = _load_signed_json(
        arguments.base_plan, "pldr-row-map-execution-plan-v1", "plan_sha256")
    prior_summary = _load_signed_json(
        arguments.prior_summary, "pldr-row-map-execution-summary-v1",
        "summary_sha256")
    geometry = _load_signed_json(
        arguments.geometry_analysis, "pldr-row-map-analysis-v1",
        "analysis_sha256")
    if selection.get("selection_status") != (
            "SEALED_FROM_CONSTRUCTION_GEOMETRY"):
        raise ValueError("intervention selection is not sealed")
    if (
        selection.get("base_plan_sha256") != base_plan["plan_sha256"]
        or selection.get("geometry_analysis_sha256")
        != geometry["analysis_sha256"]
    ):
        raise ValueError("selection provenance disagrees with the campaign")
    if prior_summary.get("plan_sha256") != base_plan["plan_sha256"]:
        raise ValueError("resource summary is not for the base plan")
    expected_stages = ["setup", "Q", "G", "H", "D", "A"]
    if (
        prior_summary.get("selected_stages") != expected_stages
        or prior_summary.get("dry_run")
    ):
        raise ValueError("prior summary must cover the complete live base plan")
    expected_jobs = Counter(
        (job["stage"], job["job_id"]) for job in base_plan["jobs"])
    summary_rows = prior_summary.get("results", [])
    actual_jobs = Counter(
        (row.get("stage"), row.get("job_id")) for row in summary_rows
        if row.get("status") in {"PASS", "SKIPPED_VERIFIED"})
    if actual_jobs != expected_jobs or len(summary_rows) != sum(expected_jobs.values()):
        raise ValueError("prior summary does not certify every base job")
    if geometry.get("kind") != "geometry":
        raise ValueError("selection provenance is not geometry analysis")
    expected_geometry = [
        job for job in base_plan["jobs"]
        if job["job_id"] == "analyze-geometry"]
    if len(expected_geometry) != 1 or (
            Path(expected_geometry[0]["output"]).resolve()
            != Path(arguments.geometry_analysis).resolve()):
        raise ValueError("geometry analysis path disagrees with the base plan")

    layers = selection.get("layers")
    if not isinstance(layers, list) or not layers or any(
            not isinstance(layer, int) or layer not in {0, 1, 2}
            for layer in layers):
        raise ValueError("selection needs a nonempty list of distinct layers")
    if len(set(layers)) != len(layers):
        raise ValueError("intervention selection repeats a layer")

    run_root = Path(arguments.run_root).resolve()
    tokens = Path(arguments.tokens).resolve()
    registry = Path(arguments.registry).resolve()
    output_root = Path(arguments.output_root).resolve()
    analysis_root = Path(arguments.analysis_root).resolve()
    if (
        run_root != Path(base_plan["run_root"]).resolve()
        or tokens != Path(base_plan["tokens"]).resolve()
        or registry != Path(base_plan["registry"]).resolve()
        or analysis_root != Path(base_plan["output_root"]).resolve() / "R"
    ):
        raise ValueError("intervention paths disagree with the base plan")
    devices = [value for value in arguments.devices.split(",") if value]
    if len(devices) != 2 or len(set(devices)) != 2:
        raise ValueError("interventions require two distinct devices")

    prior_gpu = float(prior_summary["aggregate_gpu_hours"])
    prior_wall = float(prior_summary["wall_clock_hours"])
    hard_gpu = float(RESOURCE_CAPS["hard_gpu_hours"])
    hard_wall = float(RESOURCE_CAPS["hard_wall_clock_hours"])
    if not (
        math.isfinite(prior_gpu) and math.isfinite(prior_wall)
        and 0.0 <= prior_gpu < hard_gpu and 0.0 <= prior_wall < hard_wall
    ):
        raise ValueError("base execution exhausted or invalidated resource caps")
    remaining_caps = dict(RESOURCE_CAPS)
    remaining_caps["hard_gpu_hours"] = hard_gpu - prior_gpu
    remaining_caps["hard_wall_clock_hours"] = hard_wall - prior_wall

    arms = (
        "baseline_continue", "freeze_final_gate", "disable_final_gate_decay",
        "freeze_normalized_shape", "replay_physical_rows",
    )
    jobs = []
    observable_outputs = []
    for run_index, run in enumerate(RUNS):
        for layer in layers:
            device = devices[(run_index + layer) % 2]
            for arm in arms:
                output = output_root / run["name"] / f"layer{layer}" / arm
                observable_outputs.append(output / "observables.npz")
                jobs.append(_job(
                    "I", f"{run['name']}-l{layer}-{arm}", device,
                    _command([
                        "python3", LIVE, "continue", "--checkpoint",
                        _checkpoint(run_root, run["name"], 24000),
                        "--tokens", tokens, "--registry", registry,
                        "--arm", arm, "--layer", layer,
                        "--updates", FIXED_CONSTANTS["continuation_updates"],
                        "--device", device, "--output", output,
                    ]), output))
    intervention_analysis = analysis_root / "intervention-analysis.json"
    jobs.append(_analysis_job(
        "R", "interventions", observable_outputs, intervention_analysis))
    campaign_output = analysis_root / "campaign-analysis.json"
    component_analyses = [
        analysis_root / "geometry-analysis.json",
        analysis_root / "loss-analysis.json",
        analysis_root / "block-analysis.json",
        analysis_root / "dynamics-analysis.json",
        intervention_analysis,
        analysis_root / "plga-analysis.json",
    ]
    jobs.append(_analysis_job(
        "R", "campaign", component_analyses, campaign_output))
    value = {
        "schema_version": "pldr-row-map-intervention-plan-v1",
        "base_plan_sha256": base_plan["plan_sha256"],
        "prior_summary_sha256": prior_summary["summary_sha256"],
        "geometry_analysis_sha256": geometry["analysis_sha256"],
        "selection_sha256": selection["selection_sha256"],
        "selection": selection,
        "stage_order": ["I", "R"],
        "required_live_stage_sequence": ["I", "R"],
        "stage_dependencies": {"I": [], "R": ["I"]},
        "resource_caps": remaining_caps,
        "campaign_resource_accounting": {
            "campaign_hard_gpu_hours": hard_gpu,
            "campaign_hard_wall_clock_hours": hard_wall,
            "base_gpu_hours": prior_gpu,
            "base_wall_clock_hours": prior_wall,
            "remaining_gpu_hours": hard_gpu - prior_gpu,
            "remaining_wall_clock_hours": hard_wall - prior_wall,
        },
        "jobs": jobs,
    }
    value["plan_sha256"] = digest_object(value)
    return value


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    command = subparsers.add_parser("plan")
    command.add_argument("--run-root", required=True)
    command.add_argument("--tokens", required=True)
    command.add_argument("--tokenizer", required=True)
    command.add_argument("--output-root", required=True)
    command.add_argument("--devices", default="cuda:0,cuda:1")
    command.add_argument("--output", required=True)

    command = subparsers.add_parser("select-interventions")
    command.add_argument("--base-plan", required=True)
    command.add_argument("--geometry-analysis", required=True)
    command.add_argument("--output", required=True)

    command = subparsers.add_parser("interventions")
    command.add_argument("--selection", required=True)
    command.add_argument("--base-plan", required=True)
    command.add_argument("--prior-summary", required=True)
    command.add_argument("--geometry-analysis", required=True)
    command.add_argument("--run-root", required=True)
    command.add_argument("--tokens", required=True)
    command.add_argument("--registry", required=True)
    command.add_argument("--output-root", required=True)
    command.add_argument("--analysis-root", required=True)
    command.add_argument("--devices", default="cuda:0,cuda:1")
    command.add_argument("--output", required=True)

    arguments = parser.parse_args()
    if arguments.command == "plan":
        value = execution_plan(arguments)
    elif arguments.command == "select-interventions":
        value = select_interventions(arguments)
    else:
        value = intervention_plan(arguments)
    write_json_atomic(arguments.output, value)


if __name__ == "__main__":
    main()
