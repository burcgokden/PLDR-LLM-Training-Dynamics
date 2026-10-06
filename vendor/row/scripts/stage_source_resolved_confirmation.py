#!/usr/bin/env python3
"""Stage the immutable, producer-first source-resolved campaign."""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from confirm.confirmation_artifacts import digest_object, sha256_path  # noqa: E402
from confirm.source_resolved_specs import (  # noqa: E402
    CAMPAIGN_ID,
    FROZEN_POLICY,
    INTERVENTIONS,
    RESOURCE_BUDGET,
    TRAJECTORIES,
    campaign_design,
)


DEFAULT_PARENT = (
    ROOT.parent / "experiment-data" / "campaigns" / "source-restoring"
    / "source-restoring-collapse-confirmation"
)
DEFAULT_OUTPUT = (
    ROOT.parent / "experiment-data" / "campaigns" / "source-resolved"
    / "source-resolved-row-map-confirmation"
)
DEFAULT_TOKENS = (
    ROOT.parent / "experiment-data" / "shared" / "datasets"
    / "refinedweb-100m-tokens" / "refinedweb_tokens.npy"
)
DEFAULT_TOKENIZER = ROOT / "experiments" / "data" / "tokenizer.model"
PROTOCOL_SOURCE = (
    ROOT / "experiments" / "protocols" / "source_resolved_confirmation"
)

ENTRY_POINTS = (
    "experiments/train_run.py",
    "experiments/pldr_model_v510.py",
    "experiments/power_law_attention_layer_v510.py",
    "experiments/confirm/source_resolved_energy.py",
    "experiments/confirm/source_resolved_live.py",
    "experiments/confirm/source_resolved_response.py",
    "experiments/confirm/source_resolved_observer.py",
    "experiments/confirm/source_resolved_timecourse.py",
    "experiments/confirm/source_resolved_block_response.py",
    "experiments/confirm/source_resolved_rank_tube.py",
    "experiments/confirm/source_resolved_intervention.py",
    "experiments/confirm/source_resolved_policy.py",
    "experiments/confirm/source_resolved_specs.py",
    "experiments/confirm/strict_schema.py",
    "experiments/confirm/resource_executor.py",
    "experiments/confirm/resource_worker.py",
    "experiments/confirm/campaign_budget.py",
    "experiments/confirm/campaign_plan_executor.py",
    "experiments/analysis/analyze_source_resolved_confirmation.py",
    "scripts/run_source_resolved_producer.py",
    "scripts/execute_source_resolved_plan.py",
    "scripts/stage_source_resolved_confirmation.py",
)


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def _verify_checksum_manifest(root: Path, manifest: Path) -> None:
    root = root.resolve()
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        path = (root / relative).resolve()
        path.relative_to(root)
        if not path.is_file() or sha256_path(path) != digest:
            raise ValueError(f"checksum manifest does not replay: {relative}")


def _resolve_import(module: str) -> Path | None:
    parts = module.split(".")
    candidates = (
        ROOT.joinpath(*parts).with_suffix(".py"),
        ROOT.joinpath(*parts, "__init__.py"),
        EXPERIMENTS.joinpath(*parts).with_suffix(".py"),
        EXPERIMENTS.joinpath(*parts, "__init__.py"),
        EXPERIMENTS.joinpath("confirm", *parts).with_suffix(".py"),
        EXPERIMENTS.joinpath("analysis", *parts).with_suffix(".py"),
        ROOT.joinpath("scripts", *parts).with_suffix(".py"),
    )
    return next((path.resolve() for path in candidates if path.is_file()), None)


def source_import_closure() -> list[Path]:
    pending = [ROOT / relative for relative in ENTRY_POINTS]
    selected: set[Path] = set()
    while pending:
        path = pending.pop().resolve()
        if path in selected:
            continue
        path.relative_to(ROOT)
        if not path.is_file():
            raise FileNotFoundError(f"campaign source is missing: {path}")
        selected.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        modules = []
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


def _node(
    node_id: str,
    stage: str,
    kind: str,
    command: list[str],
    outputs: list[Path],
    *,
    dependencies: list[str] | None = None,
    inputs: list[Path] | None = None,
    device: str = "cpu",
    wall_seconds: float = 1800.0,
    projected_output_bytes: int = 16 * 1024**2,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "stage": stage,
        "kind": kind,
        "device": device,
        "depends_on": [] if dependencies is None else dependencies,
        "required_inputs": [str(path) for path in (inputs or [])],
        "command": command,
        "expected_outputs": [str(path) for path in outputs],
        "projected_output_bytes": int(projected_output_bytes),
        "caps": {
            "wall_seconds": float(wall_seconds),
            "host_rss_bytes": 48 * 1024**3,
            "gpu_reserved_bytes": (
                RESOURCE_BUDGET["process_reserved_memory_cap_bytes"]
                if device.startswith("cuda") else 0
            ),
            "output_bytes": RESOURCE_BUDGET["persistent_output_cap_bytes"],
            "poll_seconds": RESOURCE_BUDGET["poll_seconds"],
        },
    }


def _trainer_common(
    train: Path,
    name: str,
    run_id: str,
    seed: int,
    device: str,
    tokens: Path,
    tokenizer: Path,
    output: Path,
    registry: Path,
    order: Path,
) -> list[str]:
    return [
        sys.executable, str(train),
        "--name", name,
        "--run-id", run_id,
        "--lr", "7.5e-4",
        "--warmup", "250",
        "--const_lr",
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
        "--probe_every", "1000000",
        "--sharp_every", "1000000",
        "--sharp_pre_every", "1000000",
        "--sharp_full", "0",
        "--ckpt_steps", ",",
        "--confirmation_registry", str(registry),
        "--data_order", str(order),
        "--probe_region", "global",
        "--skip_generation",
    ]


def _producer(
    script: Path, protocol: Path, command: str, *arguments: str
) -> list[str]:
    return [
        sys.executable, str(script), "--protocol-dir", str(protocol),
        command, *arguments,
    ]


def launch_plan(
    output: Path,
    tokens: Path,
    tokenizer: Path,
    registry: Path,
    orders: dict[str, Path],
) -> dict[str, Any]:
    source = output / "source"
    protocol = output / "protocol"
    producer = source / "scripts" / "run_source_resolved_producer.py"
    train = source / "experiments" / "train_run.py"
    nodes: list[dict[str, Any]] = []
    records = output / "records"
    reports = output / "reports"
    pilot = output / "preliminary" / "checkpoints"

    q0_ids = []
    for suffix, device in (("cpu", "cpu"), ("cuda0", "cuda:0"), ("cuda1", "cuda:1")):
        target = records / "Q0" / f"qualification-{suffix}.json"
        node_id = f"q0-real-{suffix}"
        q0_ids.append(node_id)
        nodes.append(_node(
            node_id, "Q0", "real-checkpoint-producer",
            _producer(
                producer, protocol, "qualify",
                "--device", device,
                "--checkpoint", str(pilot / "ckpt_1000.pt"),
                "--tokens", str(tokens),
                "--tokenizer", str(tokenizer),
                "--data-order", str(orders["construction"]),
                "--output", str(target),
                "--require-pass",
            ),
            [target],
            inputs=[pilot / "ckpt_1000.pt", tokens, tokenizer, orders["construction"]],
            device=device,
            wall_seconds=600.0,
        ))

    bindings: dict[int, Path] = {}
    bind_ids: dict[int, str] = {}
    for step in range(1000, 1005):
        target = records / "Q1" / f"binding-{step}.json"
        node_id = f"q1-bind-{step}"
        bindings[step] = target
        bind_ids[step] = node_id
        nodes.append(_node(
            node_id, "Q1", "checkpoint-binding-producer",
            _producer(
                producer, protocol, "bind-checkpoint",
                "--checkpoint", str(pilot / f"ckpt_{step}.pt"),
                "--tokens", str(tokens),
                "--tokenizer", str(tokenizer),
                "--data-order", str(orders["construction"]),
                "--output", str(target),
            ),
            [target], dependencies=q0_ids,
            inputs=[pilot / f"ckpt_{step}.pt", tokens, tokenizer, orders["construction"]],
            wall_seconds=600.0,
        ))

    edges: dict[int, Path] = {}
    edge_ids: dict[int, str] = {}
    for step in range(1000, 1004):
        target = records / "Q1" / f"edge-{step}-{step + 1}.json"
        node_id = f"q1-edge-{step}-{step + 1}"
        edges[step] = target
        edge_ids[step] = node_id
        device = f"cuda:{step % 2}"
        nodes.append(_node(
            node_id, "Q1", "finite-source-producer",
            _producer(
                producer, protocol, "edge",
                "--source-binding", str(bindings[step]),
                "--endpoint-binding", str(bindings[step + 1]),
                "--registry", str(registry),
                "--tokens", str(tokens),
                "--data-order", str(orders["construction"]),
                "--device", device,
                "--output", str(target),
            ),
            [target], dependencies=[bind_ids[step], bind_ids[step + 1]],
            inputs=[bindings[step], bindings[step + 1], registry, tokens, orders["construction"]],
            device=device,
            wall_seconds=900.0,
            projected_output_bytes=8 * 1024**2,
        ))

    q2_outputs = []
    q2_ids = []
    for half, device, indices, random_count in (
        ("a", "cuda:0", "0-143", 8),
        ("b", "cuda:1", "144-287", 0),
    ):
        expanded = ",".join(
            str(value) for value in (
                range(0, 144) if half == "a" else range(144, 288)
            )
        )
        target = records / "Q2" / f"mask-response-{half}.json"
        node_id = f"q2-mask-response-{half}"
        q2_outputs.append(target)
        q2_ids.append(node_id)
        nodes.append(_node(
            node_id, "Q2", "masked-adamw-producer",
            _producer(
                producer, protocol, "mask-response",
                "--binding", str(bindings[1000]),
                "--registry", str(registry),
                "--tokens", str(tokens),
                "--data-order", str(orders["construction"]),
                "--device", device,
                "--map-indices", expanded,
                "--random-directions", str(random_count),
                "--random-seed", "45101",
                "--output", str(target),
            ),
            [target], dependencies=[edge_ids[1000]],
            inputs=[bindings[1000], registry, tokens, orders["construction"]],
            device=device,
            wall_seconds=1800.0,
            projected_output_bytes=32 * 1024**2,
        ))

    q3_run = "q3-native-continuation"
    q3_run_dir = output / "runs" / "source-construction-s7444"
    q3_command = _trainer_common(
        train, "source-construction-s7444", "q3-source-continuation",
        7444, "cuda:0", tokens, tokenizer, output, registry,
        orders["construction"],
    ) + [
        "--steps", "12",
        "--schedule_total_steps", "1025",
        "--init_from", str(pilot / "ckpt_1004.pt"),
        "--optimizer_ckpt_steps", ",".join(str(step) for step in range(1005, 1016)),
    ]
    nodes.append(_node(
        q3_run, "Q3", "registered-trainer-producer", q3_command,
        [
            *[q3_run_dir / f"ckpt_{step}.pt"
              for step in range(1005, 1016)],
            q3_run_dir / "ckpt_final.pt",
        ], dependencies=q2_ids,
        inputs=[pilot / "ckpt_1004.pt", registry, tokens, tokenizer, orders["construction"]],
        device="cuda:0", wall_seconds=1800.0,
        projected_output_bytes=4 * 1024**3,
    ))
    for step in range(1005, 1017):
        target = records / "Q3" / f"binding-{step}.json"
        node_id = f"q3-bind-{step}"
        bindings[step] = target
        bind_ids[step] = node_id
        checkpoint = (
            q3_run_dir / "ckpt_final.pt"
            if step == 1016 else q3_run_dir / f"ckpt_{step}.pt"
        )
        nodes.append(_node(
            node_id, "Q3", "checkpoint-binding-producer",
            _producer(
                producer, protocol, "bind-checkpoint",
                "--checkpoint", str(checkpoint),
                "--tokens", str(tokens), "--tokenizer", str(tokenizer),
                "--data-order", str(orders["construction"]),
                "--output", str(target),
            ),
            [target], dependencies=[q3_run],
            inputs=[checkpoint, tokens, tokenizer, orders["construction"]],
            wall_seconds=600.0,
        ))
    for step in range(1004, 1016):
        target = records / "Q3" / f"edge-{step}-{step + 1}.json"
        node_id = f"q3-edge-{step}-{step + 1}"
        edges[step] = target
        edge_ids[step] = node_id
        source_binding = bindings[step]
        endpoint_binding = bindings[step + 1]
        device = f"cuda:{step % 2}"
        nodes.append(_node(
            node_id, "Q3", "finite-source-producer",
            _producer(
                producer, protocol, "edge",
                "--source-binding", str(source_binding),
                "--endpoint-binding", str(endpoint_binding),
                "--registry", str(registry), "--tokens", str(tokens),
                "--data-order", str(orders["construction"]),
                "--device", device, "--output", str(target),
            ),
            [target], dependencies=[bind_ids[step], bind_ids[step + 1]],
            inputs=[source_binding, endpoint_binding, registry, tokens, orders["construction"]],
            device=device, wall_seconds=900.0,
            projected_output_bytes=8 * 1024**2,
        ))
    all_edges = [edges[step] for step in range(1000, 1016)]
    block_bindings = [bindings[step] for step in range(1000, 1017)]
    block_output = records / "Q3" / "lifted-block-response.json"
    q3_block = "q3-lifted-block-response"
    nodes.append(_node(
        q3_block, "Q3", "lifted-adamw-block-producer",
        _producer(
            producer, protocol, "block-response",
            "--bindings", *[str(path) for path in block_bindings],
            "--edges", *[str(path) for path in all_edges],
            "--mask-response", str(q2_outputs[0]),
            "--registry", str(registry), "--tokens", str(tokens),
            "--data-order", str(orders["construction"]),
            "--device", "cuda:0", "--random-seed", "45301",
            "--output", str(block_output),
        ),
        [block_output],
        dependencies=[
            *q2_ids, *[edge_ids[step] for step in range(1000, 1016)]
        ],
        inputs=[
            *block_bindings, *all_edges, q2_outputs[0], registry, tokens,
            orders["construction"],
        ],
        device="cuda:0", wall_seconds=10800.0,
        projected_output_bytes=128 * 1024**2,
    ))

    rank_output = records / "Q3" / "rank-tube-block.json"
    q3_rank = "q3-rank-tube-block"
    nodes.append(_node(
        q3_rank, "Q3", "rank-tube-block-producer",
        _producer(
            producer, protocol, "rank-tube-block",
            "--source-binding", str(bindings[1000]),
            "--endpoint-binding", str(bindings[1016]),
            "--mask-responses", *[str(path) for path in q2_outputs],
            "--edges", *[str(path) for path in all_edges],
            "--block-response", str(block_output),
            "--registry", str(registry), "--tokens", str(tokens),
            "--device", "cuda:1", "--output", str(rank_output),
        ),
        [rank_output],
        dependencies=[q3_block],
        inputs=[
            bindings[1000], bindings[1016], *q2_outputs, *all_edges,
            block_output, registry, tokens,
        ],
        device="cuda:1", wall_seconds=3600.0,
        projected_output_bytes=64 * 1024**2,
    ))

    design_spec = protocol / "training_campaign_spec.json"
    construction = TRAJECTORIES[0]
    q4_name = str(construction["name"])
    q4_run = "q4-construction-train"
    q4_dir = output / "runs" / q4_name
    q4_command = _trainer_common(
        train, q4_name, q4_name, int(construction["seed"]), "cuda:0",
        tokens, tokenizer, output, registry, orders["construction"],
    ) + [
        "--lineage-root", q4_name,
        "--steps", "9000",
        "--optimizer_ckpt_steps", "1000,4000,8000",
        "--campaign_spec", str(design_spec),
    ]
    nodes.append(_node(
        q4_run, "Q4", "registered-trainer-producer", q4_command,
        [
            *[q4_dir / f"ckpt_{step}.pt" for step in (1000, 4000, 8000)],
            q4_dir / "ckpt_final.pt", q4_dir / "log.jsonl",
        ],
        dependencies=[edge_ids[1003]],
        inputs=[registry, design_spec, tokens, tokenizer, orders["construction"]],
        device="cuda:0", wall_seconds=3600.0,
        projected_output_bytes=2 * 1024**3,
    ))
    q4_bindings = []
    q4_bind_ids = {}
    for step in (1000, 4000, 8000, 9000):
        target = records / "Q4" / f"binding-{step}.json"
        node_id = f"q4-bind-{step}"
        q4_bindings.append(target)
        q4_bind_ids[step] = node_id
        nodes.append(_node(
            node_id, "Q4", "checkpoint-binding-producer",
            _producer(
                producer, protocol, "bind-checkpoint",
                "--checkpoint", str(
                    q4_dir / "ckpt_final.pt"
                    if step == 9000 else q4_dir / f"ckpt_{step}.pt"
                ),
                "--tokens", str(tokens), "--tokenizer", str(tokenizer),
                "--data-order", str(orders["construction"]),
                "--output", str(target),
            ), [target], dependencies=[q4_run],
            inputs=[
                q4_dir / "ckpt_final.pt"
                if step == 9000 else q4_dir / f"ckpt_{step}.pt",
                tokens, tokenizer, orders["construction"],
            ],
            wall_seconds=600.0,
        ))
    q4_timecourse = records / "Q4" / "construction-timecourse.json"
    q4_time_id = "q4-construction-timecourse"
    nodes.append(_node(
        q4_time_id, "Q4", "timecourse-producer",
        _producer(
            producer, protocol, "timecourse",
            "--log", str(q4_dir / "log.jsonl"),
            "--registry", str(registry),
            "--bindings", *[str(path) for path in q4_bindings],
            "--policy", str(protocol / "campaign_design.json"),
            "--role", "construction", "--output", str(q4_timecourse),
        ), [q4_timecourse], dependencies=list(q4_bind_ids.values()),
        inputs=[q4_dir / "log.jsonl", *q4_bindings, registry, protocol / "campaign_design.json"],
        wall_seconds=1200.0, projected_output_bytes=512 * 1024**2,
    ))
    policy = records / "Q4" / "construction-policy.json"
    policy_id = "q4-seal-policy"
    nodes.append(_node(
        policy_id, "Q4", "construction-seal",
        _producer(
            producer, protocol, "seal-policy",
            "--design", str(protocol / "campaign_design.json"),
            "--timecourse", str(q4_timecourse),
            "--edges", *[str(path) for path in all_edges],
            "--output", str(policy),
        ), [policy], dependencies=[q4_time_id, *[edge_ids[step] for step in range(1000, 1016)]],
        inputs=[protocol / "campaign_design.json", q4_timecourse, *all_edges],
        wall_seconds=600.0,
    ))

    q5_results = []
    q5_result_ids = []
    flag_by_intervention = {
        "final-gate-freeze": ["--freeze_final_gate_at", "0"],
        "row-program-weight-freeze": ["--freeze_row_program_at", "0"],
        "upstream-shape-freeze": ["--freeze_upstream_shape_at", "0"],
        "moment-reset": ["--reset_opt"],
        "learning-rate-half": ["--learning_rate_multiplier", "0.5"],
        "learning-rate-double": ["--learning_rate_multiplier", "2.0"],
    }
    for window, step in FROZEN_POLICY["intervention_window_updates"].items():
        source_binding = records / "Q4" / f"binding-{step}.json"
        prediction_ids = []
        prediction_paths = {}
        for intervention_name in INTERVENTIONS:
            safe = intervention_name.replace("-", "_")
            prediction_path = records / "Q5" / f"prediction-{window}-{safe}.json"
            prediction_id = f"q5-predict-{window}-{safe}"
            prediction_ids.append(prediction_id)
            prediction_paths[intervention_name] = prediction_path
            nodes.append(_node(
                prediction_id, "Q5", "source-only-intervention-producer",
                _producer(
                    producer, protocol, "intervention-predict",
                    "--source-binding", str(source_binding),
                    "--registry", str(registry), "--tokens", str(tokens),
                    "--data-order", str(orders["construction"]),
                    "--intervention", intervention_name,
                    "--window", window,
                    "--device", "cuda:1", "--output", str(prediction_path),
                ), [prediction_path], dependencies=[policy_id, q4_bind_ids[step]],
                inputs=[source_binding, policy, registry, tokens, orders["construction"]],
                device="cuda:1", wall_seconds=900.0,
                projected_output_bytes=8 * 1024**2,
            ))
        control_name = f"q5-control-{window}"
        control_dir = output / "runs" / control_name
        control_command = _trainer_common(
            train, control_name, control_name, 7444, "cuda:0", tokens,
            tokenizer, output, registry, orders["construction"],
        ) + [
            "--steps", "1", "--init_from", str(q4_dir / f"ckpt_{step}.pt"),
            "--optimizer_ckpt_steps", ",",
            "--campaign_spec", str(design_spec),
            "--schedule_total_steps", "9000",
        ]
        nodes.append(_node(
            control_name, "Q5", "registered-native-control", control_command,
            [control_dir / "ckpt_final.pt"], dependencies=prediction_ids,
            inputs=[q4_dir / f"ckpt_{step}.pt", design_spec, registry, tokens, tokenizer, orders["construction"]],
            device="cuda:0", wall_seconds=600.0,
            projected_output_bytes=384 * 1024**2,
        ))
        control_binding = records / "Q5" / f"binding-control-{window}.json"
        control_bind_id = f"q5-bind-control-{window}"
        nodes.append(_node(
            control_bind_id, "Q5", "checkpoint-binding-producer",
            _producer(
                producer, protocol, "bind-checkpoint",
                "--checkpoint", str(control_dir / "ckpt_final.pt"),
                "--tokens", str(tokens), "--tokenizer", str(tokenizer),
                "--data-order", str(orders["construction"]),
                "--output", str(control_binding),
            ), [control_binding], dependencies=[control_name],
            inputs=[control_dir / "ckpt_final.pt", tokens, tokenizer, orders["construction"]],
            wall_seconds=600.0,
        ))
        for index, intervention_name in enumerate(INTERVENTIONS):
            safe = intervention_name.replace("-", "_")
            arm_name = f"q5-{window}-{safe}"
            arm_dir = output / "runs" / arm_name
            device = f"cuda:{index % 2}"
            arm_command = _trainer_common(
                train, arm_name, arm_name, 7444, device, tokens, tokenizer,
                output, registry, orders["construction"],
            ) + [
                "--steps", "1", "--init_from", str(q4_dir / f"ckpt_{step}.pt"),
                "--optimizer_ckpt_steps", ",",
                "--campaign_spec", str(design_spec),
                "--schedule_total_steps", "9000",
                *flag_by_intervention[intervention_name],
            ]
            nodes.append(_node(
                arm_name, "Q5", "registered-native-intervention", arm_command,
                [arm_dir / "ckpt_final.pt"],
                dependencies=[f"q5-predict-{window}-{safe}"],
                inputs=[q4_dir / f"ckpt_{step}.pt", design_spec, registry, tokens, tokenizer, orders["construction"]],
                device=device, wall_seconds=600.0,
                projected_output_bytes=384 * 1024**2,
            ))
            arm_binding = records / "Q5" / f"binding-{window}-{safe}.json"
            arm_bind_id = f"q5-bind-{window}-{safe}"
            nodes.append(_node(
                arm_bind_id, "Q5", "checkpoint-binding-producer",
                _producer(
                    producer, protocol, "bind-checkpoint",
                    "--checkpoint", str(arm_dir / "ckpt_final.pt"),
                    "--tokens", str(tokens), "--tokenizer", str(tokenizer),
                    "--data-order", str(orders["construction"]),
                    "--output", str(arm_binding),
                ), [arm_binding], dependencies=[arm_name],
                inputs=[arm_dir / "ckpt_final.pt", tokens, tokenizer, orders["construction"]],
                wall_seconds=600.0,
            ))
            result_path = records / "Q5" / f"result-{window}-{safe}.json"
            result_id = f"q5-result-{window}-{safe}"
            q5_results.append(result_path)
            q5_result_ids.append(result_id)
            nodes.append(_node(
                result_id, "Q5", "native-intervention-result-producer",
                _producer(
                    producer, protocol, "intervention",
                    "--prediction", str(prediction_paths[intervention_name]),
                    "--control-binding", str(control_binding),
                    "--arm-binding", str(arm_binding),
                    "--registry", str(registry), "--tokens", str(tokens),
                    "--policy", str(policy), "--device", device,
                    "--output", str(result_path),
                ), [result_path], dependencies=[control_bind_id, arm_bind_id],
                inputs=[prediction_paths[intervention_name], control_binding, arm_binding, registry, tokens, policy],
                device=device, wall_seconds=900.0,
                projected_output_bytes=8 * 1024**2,
            ))

    heldout_timecourses = []
    heldout_ids = []
    for trajectory in TRAJECTORIES[1:]:
        role_name = str(trajectory["name"])
        seed = int(trajectory["seed"])
        device = str(trajectory["device"])
        order_key = "heldout8444" if seed == 8444 else "heldout9444"
        run_id = f"q6-train-{seed}"
        run_dir = output / "runs" / role_name
        command = _trainer_common(
            train, role_name, role_name, seed, device, tokens, tokenizer,
            output, registry, orders[order_key],
        ) + [
            "--lineage-root", role_name, "--steps", "9000",
            "--optimizer_ckpt_steps", ",",
            "--campaign_spec", str(design_spec),
        ]
        nodes.append(_node(
            run_id, "Q6", "registered-heldout-trainer-producer", command,
            [run_dir / "ckpt_final.pt", run_dir / "log.jsonl"],
            dependencies=[q3_rank, policy_id, *q5_result_ids],
            inputs=[registry, design_spec, policy, tokens, tokenizer, orders[order_key]],
            device=device, wall_seconds=3600.0,
            projected_output_bytes=768 * 1024**2,
        ))
        binding_path = records / "Q6" / f"binding-{seed}-9000.json"
        binding_id = f"q6-bind-{seed}-9000"
        nodes.append(_node(
            binding_id, "Q6", "checkpoint-binding-producer",
            _producer(
                producer, protocol, "bind-checkpoint",
                "--checkpoint", str(run_dir / "ckpt_final.pt"),
                "--tokens", str(tokens), "--tokenizer", str(tokenizer),
                "--data-order", str(orders[order_key]),
                "--output", str(binding_path),
            ), [binding_path], dependencies=[run_id],
            inputs=[run_dir / "ckpt_final.pt", tokens, tokenizer, orders[order_key]],
            wall_seconds=600.0,
        ))
        timecourse_path = records / "Q6" / f"timecourse-{seed}.json"
        timecourse_id = f"q6-timecourse-{seed}"
        heldout_timecourses.append(timecourse_path)
        heldout_ids.append(timecourse_id)
        nodes.append(_node(
            timecourse_id, "Q6", "timecourse-producer",
            _producer(
                producer, protocol, "timecourse",
                "--log", str(run_dir / "log.jsonl"),
                "--registry", str(registry), "--bindings", str(binding_path),
                "--policy", str(policy), "--role", "heldout",
                "--output", str(timecourse_path),
            ), [timecourse_path], dependencies=[binding_id, policy_id],
            inputs=[run_dir / "log.jsonl", binding_path, registry, policy],
            wall_seconds=1200.0, projected_output_bytes=512 * 1024**2,
        ))

    final_report = reports / "final-analysis.json"
    analyzer = source / "experiments" / "analysis" / "analyze_source_resolved_confirmation.py"
    nodes.append(_node(
        "q6-final-analysis", "Q6", "strict-mapwise-analysis",
        [
            sys.executable, str(analyzer), "--protocol-dir", str(protocol),
            "--edges", *[str(path) for path in all_edges],
            "--mask-responses", *[str(path) for path in q2_outputs],
            "--rank-tubes", str(rank_output),
            "--timecourses", str(q4_timecourse), *[str(path) for path in heldout_timecourses],
            "--interventions", *[str(path) for path in q5_results],
            "--output", str(final_report), "--require-valid",
        ], [final_report],
        dependencies=[q3_rank, q4_time_id, *q5_result_ids, *heldout_ids],
        inputs=[*all_edges, *q2_outputs, rank_output, q4_timecourse, *heldout_timecourses, *q5_results],
        wall_seconds=1800.0, projected_output_bytes=512 * 1024**2,
    ))

    return {
        "schema_version": "pldr-source-resolved-launch-plan-v1",
        "campaign_id": CAMPAIGN_ID,
        "bundle_root": str(output),
        "dataset": {"path": str(tokens), "sha256": sha256_path(tokens)},
        "tokenizer": {"path": str(tokenizer), "sha256": sha256_path(tokenizer)},
        "registry": {"path": str(registry), "sha256": sha256_path(registry)},
        "orders": {
            name: {"path": str(path), "sha256": sha256_path(path)}
            for name, path in orders.items()
        },
        "resource_budget": RESOURCE_BUDGET,
        "nodes": nodes,
        "final_analysis": str(final_report),
        "producer_precedes_every_scientific_consumer": True,
        "scientific_outcomes_control_execution": False,
    }


def _immutable(path: Path, output: Path) -> bool:
    relative = path.relative_to(output)
    generated_bytecode = (
        "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}
    )
    return not generated_bytecode and (
        relative.parts[0] in {"source", "protocol", "orders", "preliminary"}
        or relative == Path("README.md")
    )


def _write_manifest(output: Path) -> None:
    members = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path.name != "MANIFEST.sha256" and _immutable(path, output)
    )
    payload = "".join(
        f"{sha256_path(path)}  {path.relative_to(output).as_posix()}\n"
        for path in members
    )
    (output / "MANIFEST.sha256").write_text(payload, encoding="ascii")


def build(
    parent: Path,
    output: Path,
    tokens: Path,
    tokenizer: Path,
    pilot_checkpoint_directory: Path,
    pilot_artifacts: list[Path],
) -> None:
    parent = parent.resolve()
    output = output.resolve()
    tokens = tokens.resolve()
    tokenizer = tokenizer.resolve()
    pilot_checkpoint_directory = pilot_checkpoint_directory.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "source-resolved staged bundle already exists; use --check"
        )
    for required in (
        parent / "MANIFEST.sha256", PROTOCOL_SOURCE / "CHECKSUMS.sha256",
        tokens, tokenizer,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    _verify_checksum_manifest(parent, parent / "MANIFEST.sha256")
    _verify_checksum_manifest(
        PROTOCOL_SOURCE, PROTOCOL_SOURCE / "CHECKSUMS.sha256"
    )
    if len({path.resolve() for path in pilot_artifacts}) != len(pilot_artifacts):
        raise ValueError("pilot artifact membership contains duplicates")
    for directory in (
        "source", "protocol", "orders", "preliminary/checkpoints",
        "preliminary/records", "records/Q0", "records/Q1", "records/Q2",
        "records/Q3", "records/Q4", "records/Q5", "records/Q6",
        "reports", "runtime", "runs",
    ):
        (output / directory).mkdir(parents=True, exist_ok=True)

    for source_path in source_import_closure():
        copy_file(source_path, output / "source" / source_path.relative_to(ROOT))
    for source_path in PROTOCOL_SOURCE.iterdir():
        if source_path.is_file():
            copy_file(source_path, output / "protocol" / source_path.name)
    registry_source = parent / "protocol" / "registry.json"
    copy_file(registry_source, output / "protocol" / "registry.json")
    order_sources = {
        "construction": parent / "orders" / "construction-seed7444-chunk-order.npy",
        "heldout8444": parent / "orders" / "heldout-seed8444-chunk-order.npy",
        "heldout9444": parent / "orders" / "heldout-seed9444-chunk-order.npy",
    }
    orders = {}
    for name, source_path in order_sources.items():
        target = output / "orders" / source_path.name
        copy_file(source_path, target)
        orders[name] = target
    for step in range(1000, 1005):
        source_path = pilot_checkpoint_directory / f"ckpt_{step}.pt"
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        copy_file(source_path, output / "preliminary" / "checkpoints" / source_path.name)
    for source_path in pilot_artifacts:
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        copy_file(source_path, output / "preliminary" / "records" / source_path.name)

    registry = json.loads(registry_source.read_text(encoding="utf-8"))
    design = campaign_design()
    training_spec = {
        "schema_version": "pldr-source-resolved-training-spec-v1",
        "campaign_id": CAMPAIGN_ID,
        "registry_sha256": registry["registry_sha256"],
        "design": design,
    }
    training_spec["spec_sha256"] = digest_object(training_spec)
    (output / "protocol" / "training_campaign_spec.json").write_bytes(
        canonical(training_spec)
    )
    plan = launch_plan(
        output, tokens, tokenizer, output / "protocol" / "registry.json", orders
    )
    (output / "protocol" / "launch_plan.json").write_bytes(canonical(plan))
    (output / "README.md").write_text(
        "# Source-resolved row-map confirmation\n\n"
        "This immutable bundle stages real checkpoint producers for Q0 through "
        "Q6. Run `source/scripts/execute_source_resolved_plan.py --plan "
        "protocol/launch_plan.json --dry-run` from any directory to inspect "
        "the producer-first graph. Scientific nonconfirmation never prunes a "
        "later stage; technical invalidity does.\n",
        encoding="utf-8",
    )
    _write_manifest(output)


def verify(output: Path) -> None:
    output = output.resolve()
    manifest = output / "MANIFEST.sha256"
    if not manifest.is_file():
        raise FileNotFoundError("Source-resolved staged manifest is missing")
    expected = set()
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        path = output / relative
        expected.add(relative)
        if not path.is_file() or sha256_path(path) != digest:
            raise ValueError(f"staged immutable file changed: {relative}")
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "MANIFEST.sha256" and _immutable(path, output)
    }
    if actual != expected:
        raise ValueError("staged immutable membership changed")
    plan = json.loads((output / "protocol" / "launch_plan.json").read_text())
    if plan.get("campaign_id") != CAMPAIGN_ID or not plan.get("nodes"):
        raise ValueError("staged launch plan is invalid")
    ids = [node["id"] for node in plan["nodes"]]
    if len(ids) != len(set(ids)):
        raise ValueError("staged launch plan has duplicate node identifiers")
    node_ids = set(ids)
    if any(not set(node["depends_on"]).issubset(node_ids) for node in plan["nodes"]):
        raise ValueError("staged launch plan has an unknown dependency")
    print(f"source-resolved staging: verified ({len(expected)} immutable files, {len(ids)} nodes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--pilot-checkpoint-dir", type=Path)
    parser.add_argument("--pilot-artifact", action="append", type=Path, default=[])
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if arguments.check:
        verify(arguments.output)
        return
    if arguments.pilot_checkpoint_dir is None:
        raise ValueError("initial staging requires --pilot-checkpoint-dir")
    build(
        arguments.parent, arguments.output, arguments.tokens,
        arguments.tokenizer, arguments.pilot_checkpoint_dir,
        arguments.pilot_artifact,
    )
    verify(arguments.output)


if __name__ == "__main__":
    main()
