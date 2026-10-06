#!/usr/bin/env python3
"""Stage the immutable orbitwise confirmation bundle and launch plan."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from confirm.confirmation_artifacts import sha256_path  # noqa: E402
from confirm.orbitwise_specs import (  # noqa: E402
    CAMPAIGN_ID,
    INTERVENTION_LENGTH,
    INTERVENTION_UPDATES,
    LAUNCH_PLAN_SCHEMA,
    OPTIMIZER,
    PERMANENT_CHECKPOINT_UPDATES,
    PROBE_RESERVE_CHUNKS,
    REGISTRY,
    RESOURCE_BUDGET,
    TERMINAL_UPDATE,
    TRAJECTORIES,
    construction_checkpoint_updates,
    source_edges,
)
from scripts.gen_orbitwise_protocols import (  # noqa: E402
    OUTPUT as PROTOCOL_SOURCE,
    generated_files,
)


DEFAULT_OUTPUT = (
    ROOT.parent
    / "experiment-data"
    / "campaigns"
    / "orbitwise"
    / "orbitwise-row-map-confirmation"
)
DEFAULT_TOKENS = (
    ROOT.parent
    / "experiment-data"
    / "shared"
    / "datasets"
    / "refinedweb-100m-tokens"
    / "refinedweb_tokens.npy"
)
DEFAULT_TOKENIZER = ROOT / "experiments" / "data" / "tokenizer.model"
DEFAULT_REGISTRY = (
    ROOT.parent
    / "experiment-data"
    / "campaigns"
    / "source-resolved"
    / "source-resolved-row-map-confirmation"
    / "protocol"
    / "registry.json"
)
BINDING_PROTOCOL_SOURCE = (
    ROOT / "experiments" / "protocols" / "source_resolved_confirmation"
)

ENTRY_POINTS = (
    "experiments/train_run.py",
    "experiments/analysis/analyze_orbitwise_confirmation.py",
    "scripts/run_orbitwise_producer.py",
    "scripts/run_source_resolved_producer.py",
    "scripts/execute_orbitwise_plan.py",
    "scripts/stage_orbitwise_confirmation.py",
)


def _resolve_import(module: str) -> Path | None:
    parts = module.split(".")
    candidates = (
        ROOT.joinpath(*parts).with_suffix(".py"),
        ROOT.joinpath(*parts, "__init__.py"),
        ROOT.joinpath("experiments", *parts).with_suffix(".py"),
        ROOT.joinpath("experiments", *parts, "__init__.py"),
        ROOT.joinpath("experiments", "confirm", *parts).with_suffix(".py"),
        ROOT.joinpath("experiments", "analysis", *parts).with_suffix(".py"),
        ROOT.joinpath("scripts", *parts).with_suffix(".py"),
    )
    return next((path.resolve() for path in candidates if path.is_file()), None)


def source_import_closure() -> list[Path]:
    """Return the deterministic local import closure of campaign entry points."""

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
        modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module
            ):
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
        "required_inputs": [str(path.resolve()) for path in (inputs or [])],
        "command": [str(value) for value in command],
        "expected_outputs": [str(path.resolve()) for path in outputs],
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
    bundle: Path,
    *,
    name: str,
    run_id: str,
    seed: int,
    device: str,
    tokens: Path,
    tokenizer: Path,
    registry: Path,
    order: Path,
) -> list[str]:
    train = bundle / "source" / "experiments" / "train_run.py"
    return [
        sys.executable,
        str(train),
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
        "--outdir", str(bundle / "runs"),
        "--probe_every", "1000000",
        "--sharp_every", "1000000",
        "--sharp_pre_every", "1000000",
        "--sharp_full", "0",
        "--ckpt_steps", ",",
        "--confirmation_registry", str(registry),
        "--data_order", str(order),
        "--probe_region", "global",
        "--skip_generation",
        "--campaign_spec", str(bundle / "protocol" / "training_campaign_spec.json"),
    ]


def _orbitwise_producer(
    bundle: Path,
    command: str,
    *arguments: str,
) -> list[str]:
    return [
        sys.executable,
        str(bundle / "source" / "scripts" / "run_orbitwise_producer.py"),
        command,
        *arguments,
        "--protocol-dir", str(bundle / "protocol"),
        "--binding-protocol-dir", str(bundle / "binding_protocol"),
    ]


def _binding_producer(
    bundle: Path,
    command: str,
    *arguments: str,
) -> list[str]:
    return [
        sys.executable,
        str(bundle / "source" / "scripts" / "run_source_resolved_producer.py"),
        "--protocol-dir", str(bundle / "binding_protocol"),
        command,
        *arguments,
    ]


def launch_plan(
    bundle: Path,
    tokens: Path,
    tokenizer: Path,
    registry: Path,
    orders: dict[str, Path],
) -> dict[str, Any]:
    """Build the complete producer-first launch graph."""

    nodes: list[dict[str, Any]] = []
    records = bundle / "records"
    reports = bundle / "reports"

    q0_ids = []
    for suffix, device in (
        ("cpu", "cpu"),
        ("cuda0", "cuda:0"),
        ("cuda1", "cuda:1"),
    ):
        node_id = f"c0-qualify-{suffix}"
        output = records / "C0" / f"qualification-{suffix}.json"
        q0_ids.append(node_id)
        nodes.append(_node(
            node_id,
            "C0",
            "orbitwise-kernel-device-qualification",
            _orbitwise_producer(
                bundle,
                "qualify",
                "--device", device,
                "--output", str(output),
                "--require-pass",
            ),
            [output],
            inputs=[bundle / "protocol" / "qualification.schema.json"],
            device=device,
            wall_seconds=600.0,
        ))

    training_ids: dict[str, str] = {}
    run_directories: dict[str, Path] = {}
    for trajectory in TRAJECTORIES:
        name = str(trajectory["name"])
        role = str(trajectory["role"])
        seed = int(trajectory["seed"])
        device = str(trajectory["device"])
        order = orders[str(trajectory["order_key"])]
        node_id = f"c1-train-{seed}"
        run = bundle / "runs" / name
        run_directories[name] = run
        training_ids[name] = node_id
        checkpoints = (
            construction_checkpoint_updates()
            if role == "construction"
            else list(PERMANENT_CHECKPOINT_UPDATES)
        )
        intermediate = [
            step for step in checkpoints if step != TERMINAL_UPDATE
        ]
        command = _trainer_common(
            bundle,
            name=name,
            run_id=name,
            seed=seed,
            device=device,
            tokens=tokens,
            tokenizer=tokenizer,
            registry=registry,
            order=order,
        ) + [
            "--lineage-root", name,
            "--steps", str(TERMINAL_UPDATE),
            "--optimizer_ckpt_steps", ",".join(map(str, intermediate)),
        ]
        outputs = [run / "log.jsonl", run / "ckpt_final.pt"] + [
            run / f"ckpt_{step}.pt" for step in intermediate
        ]
        dependencies = list(q0_ids)
        if role == "transfer":
            dependencies.append(training_ids[TRAJECTORIES[0]["name"]])
        nodes.append(_node(
            node_id,
            "C1",
            "registered-orbitwise-trainer",
            command,
            outputs,
            dependencies=dependencies,
            inputs=[
                tokens,
                tokenizer,
                registry,
                order,
                bundle / "protocol" / "training_campaign_spec.json",
            ],
            device=device,
            wall_seconds=5400.0,
            projected_output_bytes=(
                5 * 1024**3 if role == "construction" else 2 * 1024**3
            ),
        ))

    construction = TRAJECTORIES[0]
    construction_name = str(construction["name"])
    construction_run = run_directories[construction_name]
    construction_order = orders[str(construction["order_key"])]
    source_links: list[Path] = []
    source_link_ids: list[str] = []
    source_bindings: dict[int, tuple[str, Path]] = {}
    for source_step, endpoint_step in source_edges():
        if source_step not in source_bindings:
            bind_id = f"c2-bind-source-{source_step}"
            binding = records / "C2" / f"binding-source-{source_step}.json"
            source_bindings[source_step] = (bind_id, binding)
            nodes.append(_node(
                bind_id,
                "C2",
                "complete-source-checkpoint-binding",
                _binding_producer(
                    bundle,
                    "bind-checkpoint",
                    "--checkpoint", str(
                        construction_run / f"ckpt_{source_step}.pt"
                    ),
                    "--tokens", str(tokens),
                    "--tokenizer", str(tokenizer),
                    "--data-order", str(construction_order),
                    "--output", str(binding),
                ),
                [binding],
                dependencies=[training_ids[construction_name]],
                inputs=[
                    construction_run / f"ckpt_{source_step}.pt",
                    tokens,
                    tokenizer,
                    construction_order,
                ],
                wall_seconds=600.0,
            ))
        source_bind_id, source_binding = source_bindings[source_step]
        edge_name = f"{source_step}-{endpoint_step}"
        prediction_id = f"c2-predict-{edge_name}"
        prediction = records / "C2" / f"prediction-{edge_name}.json"
        device = f"cuda:{source_step % 2}"
        nodes.append(_node(
            prediction_id,
            "C2",
            "source-only-reopening-prediction",
            _orbitwise_producer(
                bundle,
                "source-predict",
                "--source-binding", str(source_binding),
                "--registry", str(registry),
                "--tokens", str(tokens),
                "--data-order", str(construction_order),
                "--device", device,
                "--output", str(prediction),
            ),
            [prediction],
            dependencies=[source_bind_id],
            inputs=[source_binding, registry, tokens, construction_order],
            device=device,
            wall_seconds=900.0,
            projected_output_bytes=16 * 1024**2,
        ))

        replay_name = f"source-replay-{edge_name}"
        replay_id = f"c2-replay-{edge_name}"
        replay_run = bundle / "runs" / replay_name
        replay_command = _trainer_common(
            bundle,
            name=replay_name,
            run_id=replay_name,
            seed=int(construction["seed"]),
            device=device,
            tokens=tokens,
            tokenizer=tokenizer,
            registry=registry,
            order=construction_order,
        ) + [
            "--steps", "1",
            "--init_from", str(
                construction_run / f"ckpt_{source_step}.pt"
            ),
            "--optimizer_ckpt_steps", ",",
            "--schedule_total_steps", str(TERMINAL_UPDATE),
        ]
        nodes.append(_node(
            replay_id,
            "C2",
            "prospective-native-one-step-replay",
            replay_command,
            [replay_run / "ckpt_final.pt", replay_run / "log.jsonl"],
            dependencies=[prediction_id],
            inputs=[
                prediction,
                construction_run / f"ckpt_{source_step}.pt",
                tokens,
                tokenizer,
                registry,
                construction_order,
            ],
            device=device,
            wall_seconds=900.0,
            projected_output_bytes=384 * 1024**2,
        ))
        endpoint_bind_id = f"c2-bind-endpoint-{edge_name}"
        endpoint_binding = (
            records / "C2" / f"binding-endpoint-{edge_name}.json"
        )
        nodes.append(_node(
            endpoint_bind_id,
            "C2",
            "complete-replay-endpoint-binding",
            _binding_producer(
                bundle,
                "bind-checkpoint",
                "--checkpoint", str(replay_run / "ckpt_final.pt"),
                "--tokens", str(tokens),
                "--tokenizer", str(tokenizer),
                "--data-order", str(construction_order),
                "--output", str(endpoint_binding),
            ),
            [endpoint_binding],
            dependencies=[replay_id],
            inputs=[
                replay_run / "ckpt_final.pt",
                tokens,
                tokenizer,
                construction_order,
            ],
            wall_seconds=600.0,
        ))
        link_id = f"c2-link-{edge_name}"
        link = records / "C2" / f"source-link-{edge_name}.json"
        source_links.append(link)
        source_link_ids.append(link_id)
        nodes.append(_node(
            link_id,
            "C2",
            "source-to-native-reopening-link",
            _orbitwise_producer(
                bundle,
                "source-link",
                "--prediction", str(prediction),
                "--source-binding", str(source_binding),
                "--endpoint-binding", str(endpoint_binding),
                "--registry", str(registry),
                "--tokens", str(tokens),
                "--data-order", str(construction_order),
                "--device", device,
                "--output", str(link),
            ),
            [link],
            dependencies=[prediction_id, endpoint_bind_id],
            inputs=[
                prediction,
                source_binding,
                endpoint_binding,
                registry,
                tokens,
                construction_order,
            ],
            device=device,
            wall_seconds=1200.0,
            projected_output_bytes=16 * 1024**2,
        ))

    gate_results: list[Path] = []
    gate_result_ids: list[str] = []
    for index, source_step in enumerate(INTERVENTION_UPDATES):
        source_bind_id, source_binding = source_bindings[source_step]
        prediction_id = f"c3-seal-gate-{source_step}"
        prediction = records / "C3" / f"gate-prediction-{source_step}.json"
        nodes.append(_node(
            prediction_id,
            "C3",
            "sealed-gate-direction-prediction",
            _orbitwise_producer(
                bundle,
                "seal-gate",
                "--source-binding", str(source_binding),
                "--output", str(prediction),
            ),
            [prediction],
            dependencies=[source_bind_id],
            inputs=[source_binding],
            wall_seconds=300.0,
        ))
        branch_ids = []
        branch_runs = {}
        for arm_index, arm in enumerate(("control", "frozen")):
            branch_id = f"c3-{arm}-{source_step}"
            branch_ids.append(branch_id)
            name = f"gate-{arm}-{source_step}"
            run = bundle / "runs" / name
            branch_runs[arm] = run
            device = f"cuda:{(index + arm_index) % 2}"
            command = _trainer_common(
                bundle,
                name=name,
                run_id=name,
                seed=int(construction["seed"]),
                device=device,
                tokens=tokens,
                tokenizer=tokenizer,
                registry=registry,
                order=construction_order,
            ) + [
                "--steps", str(INTERVENTION_LENGTH),
                "--init_from", str(
                    construction_run / f"ckpt_{source_step}.pt"
                ),
                "--optimizer_ckpt_steps", ",",
                "--schedule_total_steps", str(TERMINAL_UPDATE),
            ]
            if arm == "frozen":
                command.extend([
                    "--freeze_final_gate_at", str(source_step)
                ])
            nodes.append(_node(
                branch_id,
                "C3",
                f"native-gate-{arm}-branch",
                command,
                [run / "ckpt_final.pt", run / "log.jsonl"],
                dependencies=[prediction_id],
                inputs=[
                    prediction,
                    construction_run / f"ckpt_{source_step}.pt",
                    tokens,
                    tokenizer,
                    registry,
                    construction_order,
                ],
                device=device,
                wall_seconds=1200.0,
                projected_output_bytes=512 * 1024**2,
            ))
        result_id = f"c3-gate-result-{source_step}"
        result = records / "C3" / f"gate-result-{source_step}.json"
        gate_result_ids.append(result_id)
        gate_results.append(result)
        nodes.append(_node(
            result_id,
            "C3",
            "mapwise-gate-intervention-result",
            _orbitwise_producer(
                bundle,
                "gate-result",
                "--prediction", str(prediction),
                "--control-log", str(branch_runs["control"] / "log.jsonl"),
                "--frozen-log", str(branch_runs["frozen"] / "log.jsonl"),
                "--registry", str(registry),
                "--output", str(result),
            ),
            [result],
            dependencies=branch_ids,
            inputs=[
                prediction,
                branch_runs["control"] / "log.jsonl",
                branch_runs["frozen"] / "log.jsonl",
                registry,
            ],
            wall_seconds=600.0,
            projected_output_bytes=8 * 1024**2,
        ))

    analysis = reports / "final-analysis.json"
    analyzer = (
        bundle
        / "source"
        / "experiments"
        / "analysis"
        / "analyze_orbitwise_confirmation.py"
    )
    trajectory_arguments = []
    for trajectory in TRAJECTORIES:
        name = str(trajectory["name"])
        trajectory_arguments.extend([
            "--trajectory",
            (
                f"{name}:{trajectory['role']}:{trajectory['seed']}:"
                f"{run_directories[name] / 'log.jsonl'}"
            ),
        ])
    analysis_command = [
        sys.executable,
        str(analyzer),
        *trajectory_arguments,
        "--source-links", *map(str, source_links),
        "--gate-interventions", *map(str, gate_results),
        "--protocol-dir", str(bundle / "protocol"),
        "--output", str(analysis),
        "--require-valid",
    ]
    nodes.append(_node(
        "c4-final-analysis",
        "C4",
        "strict-orbitwise-analysis",
        analysis_command,
        [analysis],
        dependencies=[
            *training_ids.values(),
            *source_link_ids,
            *gate_result_ids,
        ],
        inputs=[
            *[run_directories[str(item["name"])] / "log.jsonl"
              for item in TRAJECTORIES],
            *source_links,
            *gate_results,
            bundle / "protocol" / "analysis.schema.json",
        ],
        wall_seconds=1800.0,
        projected_output_bytes=32 * 1024**2,
    ))

    plan = {
        "schema_version": LAUNCH_PLAN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "bundle_root": str(bundle.resolve()),
        "dataset": {
            "path": str(tokens.resolve()),
            "sha256": sha256_path(tokens),
        },
        "tokenizer": {
            "path": str(tokenizer.resolve()),
            "sha256": sha256_path(tokenizer),
        },
        "registry": {
            "path": str(registry.resolve()),
            "sha256": sha256_path(registry),
        },
        "orders": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_path(path),
            }
            for name, path in orders.items()
        },
        "resource_budget": RESOURCE_BUDGET,
        "nodes": nodes,
        "final_analysis": str(analysis.resolve()),
        "producer_precedes_every_scientific_consumer": True,
        "scientific_outcomes_control_execution": False,
    }
    validate_plan(plan)
    return plan


def validate_plan(plan: dict[str, Any]) -> None:
    """Validate exact graph ordering, membership, and global caps."""

    if (
        not isinstance(plan, dict)
        or plan.get("schema_version") != LAUNCH_PLAN_SCHEMA
        or plan.get("campaign_id") != CAMPAIGN_ID
        or not isinstance(plan.get("nodes"), list)
        or len(plan["nodes"]) != 59
    ):
        raise ValueError("orbitwise launch plan has an invalid header or count")
    ids = [node["id"] for node in plan["nodes"]]
    if len(ids) != len(set(ids)):
        raise ValueError("orbitwise launch plan has duplicate node ids")
    positions = {node_id: position for position, node_id in enumerate(ids)}
    expected_outputs: set[str] = set()
    for position, node in enumerate(plan["nodes"]):
        if (
            not node["command"]
            or any(
                dependency not in positions
                or positions[dependency] >= position
                for dependency in node["depends_on"]
            )
            or any(output in expected_outputs for output in node["expected_outputs"])
            or node["projected_output_bytes"] < 0
            or node["caps"]["wall_seconds"] <= 0.0
            or node["caps"]["gpu_reserved_bytes"]
            > RESOURCE_BUDGET["process_reserved_memory_cap_bytes"]
        ):
            raise ValueError(f"orbitwise launch node is invalid: {node['id']}")
        expected_outputs.update(node["expected_outputs"])
    if (
        sum(node["projected_output_bytes"] for node in plan["nodes"])
        > RESOURCE_BUDGET["persistent_output_cap_bytes"]
        or plan["scientific_outcomes_control_execution"]
        or not plan["producer_precedes_every_scientific_consumer"]
    ):
        raise ValueError("orbitwise plan violates a global evidence cap")


def _write_order(path: Path, seed: int, chunks: int) -> None:
    training_chunks = chunks - PROBE_RESERVE_CHUNKS
    required_chunks = TERMINAL_UPDATE * int(OPTIMIZER["batch_size"])
    if training_chunks < required_chunks:
        raise ValueError(
            "registered training region is shorter than the campaign"
        )
    order = np.random.default_rng(seed).permutation(training_chunks).astype(
        np.int64, copy=False
    )
    np.save(path, order, allow_pickle=False)


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _manifest(bundle: Path) -> None:
    immutable_roots = (
        bundle / "source",
        bundle / "protocol",
        bundle / "binding_protocol",
        bundle / "orders",
    )
    members = sorted(
        path
        for root in immutable_roots
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    readme = bundle / "README.md"
    if readme.is_file():
        members.append(readme)
    payload = "".join(
        f"{sha256_path(path)}  {path.relative_to(bundle).as_posix()}\n"
        for path in sorted(set(members))
    )
    (bundle / "MANIFEST.sha256").write_text(payload, encoding="ascii")


def build(
    output: Path,
    tokens: Path,
    tokenizer: Path,
    registry_source: Path,
) -> dict[str, Any]:
    """Create one new immutable campaign bundle."""

    output = output.resolve()
    tokens = tokens.resolve()
    tokenizer = tokenizer.resolve()
    registry_source = registry_source.resolve()
    for path in (tokens, tokenizer, registry_source):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite nonempty campaign bundle {output}"
        )
    for directory in (
        "source",
        "protocol",
        "binding_protocol",
        "orders",
        "records/C0",
        "records/C1",
        "records/C2",
        "records/C3",
        "records/C4",
        "reports",
        "runtime",
        "runs",
    ):
        (output / directory).mkdir(parents=True, exist_ok=True)

    for source in source_import_closure():
        _copy(source, output / "source" / source.relative_to(ROOT))
    for relative, payload in generated_files().items():
        target = output / "protocol" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    for source in BINDING_PROTOCOL_SOURCE.iterdir():
        if source.is_file():
            _copy(source, output / "binding_protocol" / source.name)
    registry = output / "protocol" / "registry.json"
    _copy(registry_source, registry)

    registry_value = json.loads(registry.read_text(encoding="utf-8"))
    if (
        registry_value.get("registry_sha256")
        != "4000f1afc8af6a0b80f19e0517a1b6d4484808841a1735a2e1fe3e82c703e627"
    ):
        raise ValueError("orbitwise registry content identifier changed")
    chunks = tokens.stat().st_size // np.dtype(np.uint16).itemsize // 256
    if chunks != int(registry_value["total_chunks"]):
        raise ValueError("dataset chunk count differs from registry")
    orders = {}
    for trajectory in TRAJECTORIES:
        key = str(trajectory["order_key"])
        path = output / "orders" / f"{trajectory['name']}-chunk-order.npy"
        _write_order(path, int(trajectory["seed"]), chunks)
        orders[key] = path

    plan = launch_plan(output, tokens, tokenizer, registry, orders)
    plan_path = output / "protocol" / "launch_plan.json"
    plan_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# Orbitwise row-map confirmation bundle\n\n"
        "Run a preflight from this directory with:\n\n"
        "    python3 source/scripts/execute_orbitwise_plan.py "
        "--plan protocol/launch_plan.json --dry-run\n\n"
        "The scientific launch graph has 59 producer-first nodes. "
        "Its outcomes never control later execution.\n",
        encoding="utf-8",
    )
    _manifest(output)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if arguments.check:
        expected = generated_files()
        actual = {
            path.relative_to(PROTOCOL_SOURCE): path.read_bytes()
            for path in PROTOCOL_SOURCE.rglob("*")
            if path.is_file()
        } if PROTOCOL_SOURCE.is_dir() else {}
        if actual != expected:
            raise SystemExit("orbitwise protocols are stale before staging")
        with tempfile.TemporaryDirectory(
            prefix="pldr-orbitwise-stage-check-"
        ) as raw:
            plan = build(
                Path(raw) / "bundle",
                arguments.tokens,
                arguments.tokenizer,
                arguments.registry,
            )
        print(
            f"orbitwise staging: verified ({len(plan['nodes'])} nodes)"
        )
        return
    plan = build(
        arguments.output,
        arguments.tokens,
        arguments.tokenizer,
        arguments.registry,
    )
    print(json.dumps({
        "bundle": str(arguments.output.resolve()),
        "campaign_id": CAMPAIGN_ID,
        "node_count": len(plan["nodes"]),
        "projected_output_bytes": sum(
            node["projected_output_bytes"] for node in plan["nodes"]
        ),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
