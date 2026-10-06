#!/usr/bin/env python3
"""Stage the mixed-collapse campaign and relative launch plan."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from confirm.confirmation_artifacts import (  # noqa: E402
    digest_object,
    sha256_path,
    validate_measurement_registry,
)
from confirm.mixed_collapse_specs import (  # noqa: E402
    CAMPAIGN_ID,
    INTERVENTION_LENGTH,
    INTERVENTION_MODES,
    INTERVENTION_UPDATES,
    OPTIMIZER,
    PERMANENT_CHECKPOINT_UPDATES,
    PROBE_RESERVE_CHUNKS,
    SOURCE_RADIUS_UPDATES,
    TERMINAL_UPDATE,
    TRAJECTORIES,
)
from scripts.gen_mixed_collapse_protocols import (  # noqa: E402
    OUTPUT as PROTOCOL_SOURCE,
    canonical,
    generated_files,
)


DEFAULT_OUTPUT = (
    PROJECTS / "experiment-data" / "campaigns" / "mixed-collapse"
    / "mixed-row-map-collapse-confirmation"
)
DEFAULT_TOKENS = (
    PROJECTS / "experiment-data" / "shared" / "datasets"
    / "refinedweb-538m-prefix-locked" / "refinedweb_tokens.npy"
)
DEFAULT_PREDECESSOR_TOKENS = (
    PROJECTS / "experiment-data" / "shared" / "datasets"
    / "refinedweb-100m-tokens" / "refinedweb_tokens.npy"
)
DEFAULT_TOKENIZER = ROOT / "experiments" / "data" / "tokenizer.model"
DEFAULT_REGISTRY = (
    PROJECTS / "experiment-data" / "campaigns" / "orbitwise"
    / "orbitwise-row-map-confirmation" / "protocol" / "registry.json"
)
PREDECESSOR_ORDERS = (
    PROJECTS / "experiment-data" / "campaigns" / "orbitwise"
    / "orbitwise-row-map-confirmation" / "orders"
)

ENTRY_POINTS = (
    "scripts/archive_refinedweb_continuation.py",
    "experiments/train_run.py",
    "experiments/analysis/analyze_mixed_collapse_confirmation.py",
    "experiments/confirm/source_frozen_radius.py",
    "scripts/execute_mixed_collapse_plan.py",
    "scripts/gen_mixed_collapse_protocols.py",
    "scripts/run_source_frozen_radius.py",
    "scripts/stage_mixed_collapse_confirmation.py",
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
    """Return the deterministic local import closure of campaign entrypoints."""

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
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.append(node.module)
        for module in modules:
            resolved = _resolve_import(module)
            if resolved is not None and resolved not in selected:
                pending.append(resolved)
    return sorted(selected)


def predecessor_order_path(order_key: str) -> Path:
    names = {
        "construction": "construction-seed10444-chunk-order.npy",
        "transfer11444": "transfer-seed11444-chunk-order.npy",
        "transfer12444": "transfer-seed12444-chunk-order.npy",
    }
    return PREDECESSOR_ORDERS / names[order_key]


def build_unique_extension_order(
    base_order: np.ndarray, seed: int, training_chunks: int, required_rows: int
) -> np.ndarray:
    """Preserve the predecessor permutation and append disjoint new chunks."""

    base = np.asarray(base_order)
    if base.ndim != 1 or base.dtype.kind not in "iu" or len(base) == 0:
        raise ValueError("predecessor order is not a nonempty integer vector")
    base = np.asarray(base, dtype=np.int64)
    if (
        int(base.min()) != 0
        or int(base.max()) != len(base) - 1
        or len(np.unique(base)) != len(base)
    ):
        raise ValueError("predecessor order is not a full training permutation")
    if len(base) > training_chunks or required_rows > training_chunks:
        raise ValueError("enlarged training pool is shorter than the campaign")
    extension = np.arange(len(base), training_chunks, dtype=np.int64)
    generator = np.random.Generator(
        np.random.PCG64(int(seed) + 47_000_003)
    )
    generator.shuffle(extension)
    result = np.concatenate((base, extension))[:required_rows]
    if (
        len(result) != required_rows
        or len(np.unique(result)) != len(result)
        or int(result.min()) < 0
        or int(result.max()) >= training_chunks
    ):
        raise RuntimeError("unique extension order violated its contract")
    return result


def relocate_registry(
    registry_source: Path, predecessor_tokens: Path, tokens: Path, tokenizer: Path
) -> dict[str, Any]:
    """Move the predecessor probe registry to the identical enlarged tail."""

    value = json.loads(registry_source.read_text(encoding="utf-8"))
    validate_measurement_registry(value, require_nonempty=True)
    old = np.memmap(predecessor_tokens, dtype=np.uint16, mode="r")
    new = np.memmap(tokens, dtype=np.uint16, mode="r")
    context_length = int(value["context_length"])
    old_total = len(old) // context_length
    new_total = len(new) // context_length
    old_limit = int(value["trainer_training_limit"])
    new_limit = new_total - PROBE_RESERVE_CHUNKS
    if (
        len(old) % context_length
        or len(new) % context_length
        or old_total != int(value["total_chunks"])
        or value["dataset_sha256"] != sha256_path(predecessor_tokens)
        or value["tokenizer_sha256"] != sha256_path(tokenizer)
        or new_limit != TERMINAL_UPDATE * int(OPTIMIZER["batch_size"])
    ):
        raise ValueError("archive or predecessor registry binding disagrees")
    relocated = json.loads(json.dumps(value))
    for group in ("construction", "validation"):
        for record in relocated[group]:
            old_index = int(record["chunk_index"])
            offset = old_index - old_limit
            if not 0 <= offset < PROBE_RESERVE_CHUNKS:
                raise ValueError("registry context lies outside predecessor reserve")
            new_index = new_limit + offset
            old_row = old[
                old_index * context_length:(old_index + 1) * context_length
            ]
            new_row = new[
                new_index * context_length:(new_index + 1) * context_length
            ]
            if not np.array_equal(old_row, new_row):
                raise ValueError("relocated registry context changed token content")
            record["chunk_index"] = new_index
    relocated["dataset_sha256"] = sha256_path(tokens)
    relocated["total_chunks"] = new_total
    relocated["trainer_training_limit"] = new_limit
    relocated.pop("registry_sha256")
    relocated["registry_sha256"] = digest_object(relocated)
    validate_measurement_registry(relocated, require_nonempty=True)
    return relocated


def _trainer_common(trajectory: dict[str, Any]) -> list[str]:
    order_key = str(trajectory["order_key"])
    return [
        "{python}",
        "{bundle}/source/experiments/train_run.py",
        "--lr", str(OPTIMIZER["learning_rate"]),
        "--warmup", str(OPTIMIZER["warmup_updates"]),
        "--const_lr",
        "--schedule_total_steps", str(TERMINAL_UPDATE),
        "--batch", str(OPTIMIZER["batch_size"]),
        "--ctx", str(OPTIMIZER["context_length"]),
        "--layers", "3",
        "--heads", "4",
        "--dk", "64",
        "--adff", "170",
        "--seed", str(trajectory["seed"]),
        "--tokens", "{tokens}",
        "--tok_model", "{tokenizer}",
        "--outdir", "{bundle}/runs",
        "--probe_every", "1000000",
        "--sharp_every", "1000000",
        "--sharp_pre_every", "1000000",
        "--sharp_full", "0",
        "--ckpt_steps", ",",
        "--confirmation_registry", "{bundle}/protocol/registry.json",
        "--campaign_spec", "{bundle}/protocol/training_campaign_spec.json",
        "--data_order", f"{{bundle}}/orders/{order_key}-chunk-order.npy",
        "--probe_region", "global",
        "--skip_generation",
    ]


def launch_plan() -> dict[str, Any]:
    """Return a path-relative plan whose outcomes never control later nodes."""

    nodes = []
    for trajectory in TRAJECTORIES:
        name = str(trajectory["name"])
        run_id = f"mixed-collapse-{name}"
        intermediate = [
            step for step in PERMANENT_CHECKPOINT_UPDATES
            if step != TERMINAL_UPDATE
        ]
        nodes.append({
            "id": f"train-{trajectory['seed']}",
            "stage": "trajectory",
            "device": str(trajectory["preferred_device"]),
            "depends_on": [],
            "command": _trainer_common(trajectory) + [
                "--name", name,
                "--run-id", run_id,
                "--lineage-root", run_id,
                "--device", str(trajectory["preferred_device"]),
                "--steps", str(TERMINAL_UPDATE),
                "--optimizer_ckpt_steps", ",".join(map(str, intermediate)),
                "--final_checkpoint", "complete",
            ],
            "expected_outputs": [
                f"runs/{name}/log.jsonl",
                f"runs/{name}/ckpt_final.pt",
            ] + [f"runs/{name}/ckpt_{step}.pt" for step in intermediate],
        })
    for trajectory in TRAJECTORIES:
        name = str(trajectory["name"])
        order_key = str(trajectory["order_key"])
        lineage = f"mixed-collapse-{name}"
        for source_step in INTERVENTION_UPDATES:
            for mode in INTERVENTION_MODES:
                arm = f"gate-{trajectory['seed']}-{source_step}-{mode}"
                nodes.append({
                    "id": arm,
                    "stage": "gate-factorial",
                    "device": str(trajectory["preferred_device"]),
                    "depends_on": [f"train-{trajectory['seed']}"],
                    "command": _trainer_common(trajectory) + [
                        "--name", arm,
                        "--run-id", f"mixed-collapse-{arm}",
                        "--lineage-root", lineage,
                        "--device", str(trajectory["preferred_device"]),
                        "--init_from",
                        f"{{bundle}}/runs/{name}/ckpt_{source_step}.pt",
                        "--steps", str(INTERVENTION_LENGTH),
                        "--final_gate_step_mode", mode,
                        "--final_checkpoint", "none",
                    ],
                    "expected_outputs": [f"runs/{arm}/log.jsonl"],
                })
        for source_step in SOURCE_RADIUS_UPDATES:
            node_id = f"source-radius-{trajectory['seed']}-{source_step}"
            nodes.append({
                "id": node_id,
                "stage": "source-radius",
                "device": str(trajectory["preferred_device"]),
                "depends_on": [f"train-{trajectory['seed']}"],
                "command": [
                    "{python}",
                    "{bundle}/source/scripts/run_source_frozen_radius.py",
                    "--checkpoint",
                    f"{{bundle}}/runs/{name}/ckpt_{source_step}.pt",
                    "--registry", "{bundle}/protocol/registry.json",
                    "--tokens", "{tokens}",
                    "--data-order",
                    f"{{bundle}}/orders/{order_key}-chunk-order.npy",
                    "--prediction-device", str(trajectory["preferred_device"]),
                    "--comparison-device", "cpu",
                    "--protocol-dir", "{bundle}/protocol",
                    "--output", f"{{bundle}}/records/{node_id}.json",
                ],
                "expected_outputs": [f"records/{node_id}.json"],
            })
    nodes.append({
        "id": "analyze",
        "stage": "analysis",
        "device": "cpu",
        "depends_on": [node["id"] for node in nodes],
        "command": [
            "{python}",
            "{bundle}/source/experiments/analysis/"
            "analyze_mixed_collapse_confirmation.py",
            "--bundle", "{bundle}",
        ],
        "expected_outputs": [
            "reports/final-analysis.json",
            "reports/manuscript-summary.json",
            "reports/confirmation_results.tex",
        ],
    })
    return {
        "schema_version": "pldr-mixed-collapse-launch-plan-v1",
        "campaign_id": CAMPAIGN_ID,
        "paths_are_bundle_relative": True,
        "scientific_outcomes_control_execution": False,
        "nodes": nodes,
    }


def _source_binding(paths: list[Path]) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        stdout=subprocess.PIPE, text=True,
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--"]
        + [str(path.relative_to(ROOT)) for path in paths],
        cwd=ROOT, check=True, stdout=subprocess.PIPE,
    ).stdout
    return {
        "schema_version": "pldr-source-snapshot-binding-v1",
        "git_head_before_execution": head,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "source_files": {
            str(path.relative_to(ROOT)): sha256_path(path) for path in paths
        },
    }


def _manifest(root: Path) -> bytes:
    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name != "MANIFEST.sha256"
    )
    lines = [
        f"{sha256_path(path)}  {path.relative_to(root).as_posix()}"
        for path in paths
    ]
    return ("\n".join(lines) + "\n").encode("ascii")


def _verify_source_snapshot(output: Path) -> None:
    """Verify exact membership and bytes of the frozen executable source."""

    binding_path = output / "protocol" / "source_binding.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    recorded = binding.get("source_files")
    if (
        binding.get("schema_version")
        != "pldr-source-snapshot-binding-v1"
        or not isinstance(recorded, dict)
        or not recorded
    ):
        raise ValueError("staged source binding is invalid")
    source_root = output / "source"
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError("staged source root is not a regular directory")
    actual = set()
    for path in source_root.rglob("*"):
        relative = path.relative_to(source_root).as_posix()
        if path.is_symlink():
            raise ValueError(f"staged source contains a symlink: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"staged source is not regular: {relative}")
        actual.add(relative)
    if actual != set(recorded):
        raise ValueError("staged source membership drifted")
    for relative, digest in recorded.items():
        candidate = PurePosixPath(relative)
        if (
            candidate.is_absolute()
            or not candidate.parts
            or any(part in {"", ".", ".."} for part in candidate.parts)
            or candidate.as_posix() != relative
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("staged source binding contains an unsafe entry")
        target = source_root.joinpath(*candidate.parts)
        if sha256_path(target) != digest:
            raise ValueError(f"staged source digest drifted: {relative}")


def _archive_manifest(tokens: Path) -> tuple[Path, dict[str, Any]]:
    path = Path(str(tokens) + ".manifest.json")
    if not path.is_file():
        raise FileNotFoundError(f"token archive manifest is absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema_version")
        != "pldr-refinedweb-prefix-locked-archive-v1"
        or value.get("output_sha256") != sha256_path(tokens)
        or value.get("training_chunks")
        != TERMINAL_UPDATE * int(OPTIMIZER["batch_size"])
        or value.get("reserve_chunks") != PROBE_RESERVE_CHUNKS
    ):
        raise ValueError("token archive manifest or digest disagrees")
    return path, value


def stage(
    output: Path,
    tokens: Path,
    tokenizer: Path,
    registry: Path,
    predecessor_tokens: Path,
) -> None:
    """Create a new immutable-ready bundle; refuse to overwrite one."""

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty bundle: {output}")
    required = (tokens, tokenizer, registry, predecessor_tokens)
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("campaign input file is absent")
    archive_manifest, archive_value = _archive_manifest(tokens)
    registry_value = relocate_registry(
        registry, predecessor_tokens, tokens, tokenizer
    )
    output.mkdir(parents=True, exist_ok=True)
    protocol = output / "protocol"
    protocol.mkdir()
    files = generated_files()
    for relative, payload in files.items():
        destination = protocol / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    (protocol / "registry.json").write_bytes(canonical(registry_value))

    source_paths = source_import_closure()
    for path in source_paths:
        destination = output / "source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    (protocol / "source_binding.json").write_bytes(
        canonical(_source_binding(source_paths))
    )

    required_rows = TERMINAL_UPDATE * int(OPTIMIZER["batch_size"])
    training_chunks = (
        tokens.stat().st_size // np.dtype(np.uint16).itemsize
        // 256 - PROBE_RESERVE_CHUNKS
    )
    orders = output / "orders"
    orders.mkdir()
    for trajectory in TRAJECTORIES:
        order_key = str(trajectory["order_key"])
        source_order = predecessor_order_path(order_key)
        base = np.load(source_order, allow_pickle=False)
        order = build_unique_extension_order(
            base, int(trajectory["seed"]), training_chunks, required_rows
        )
        np.save(orders / f"{order_key}-chunk-order.npy", order, allow_pickle=False)

    (protocol / "launch_plan.json").write_bytes(canonical(launch_plan()))
    input_binding = {
        "schema_version": "pldr-mixed-collapse-input-binding-v1",
        "tokens_path": str(tokens.resolve()),
        "tokens_sha256": archive_value["output_sha256"],
        "token_archive_manifest_path": str(archive_manifest.resolve()),
        "token_archive_manifest_sha256": sha256_path(archive_manifest),
        "predecessor_tokens_path": str(predecessor_tokens.resolve()),
        "predecessor_tokens_sha256": sha256_path(predecessor_tokens),
        "tokenizer_path": str(tokenizer.resolve()),
        "tokenizer_sha256": sha256_path(tokenizer),
        "source_registry_sha256": sha256_path(registry),
        "registry_sha256": sha256_path(protocol / "registry.json"),
        "predecessor_orders": {
            str(trajectory["order_key"]): sha256_path(
                predecessor_order_path(str(trajectory["order_key"]))
            )
            for trajectory in TRAJECTORIES
        },
    }
    (protocol / "input_binding.json").write_bytes(canonical(input_binding))
    for directory in ("runs", "records", "reports", "runtime", "incidents"):
        (output / directory).mkdir()
    (output / "README.md").write_text(
        "# Mixed row-map collapse confirmation\n\n"
        "Stage with `scripts/stage_mixed_collapse_confirmation.py`; execute "
        "the relative launch graph with `scripts/execute_mixed_collapse_plan.py`.\n",
        encoding="utf-8",
    )
    (output / "MANIFEST.sha256").write_bytes(_manifest(output))


def check(
    output: Path,
    tokens: Path,
    tokenizer: Path,
    registry: Path,
    predecessor_tokens: Path,
) -> None:
    """Verify staged static inputs without requiring run outputs."""

    expected_protocol = generated_files()
    for relative, payload in expected_protocol.items():
        path = output / "protocol" / relative
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"staged protocol drifted: {relative}")
    plan_path = output / "protocol" / "launch_plan.json"
    if json.loads(plan_path.read_text(encoding="utf-8")) != launch_plan():
        raise ValueError("staged launch plan drifted")
    expected_registry = relocate_registry(
        registry, predecessor_tokens, tokens, tokenizer
    )
    actual_registry = json.loads(
        (output / "protocol" / "registry.json").read_text(encoding="utf-8")
    )
    if actual_registry != expected_registry:
        raise ValueError("staged relocated registry drifted")
    training_chunks = int(expected_registry["trainer_training_limit"])
    expected_rows = TERMINAL_UPDATE * int(OPTIMIZER["batch_size"])
    for trajectory in TRAJECTORIES:
        order_key = str(trajectory["order_key"])
        path = output / "orders" / f"{order_key}-chunk-order.npy"
        order = np.load(path, allow_pickle=False)
        base = np.load(
            predecessor_order_path(order_key), allow_pickle=False
        )
        expected = build_unique_extension_order(
            base, int(trajectory["seed"]), training_chunks, expected_rows
        )
        if not np.array_equal(order, expected):
            raise ValueError(f"staged order drifted: {path.name}")
    _verify_source_snapshot(output)
    _archive_manifest(tokens)
    print("mixed-collapse staging: verified")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--predecessor-tokens", type=Path, default=DEFAULT_PREDECESSOR_TOKENS
    )
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if arguments.check:
        check(
            arguments.output.resolve(),
            arguments.tokens.resolve(),
            arguments.tokenizer.resolve(),
            arguments.registry.resolve(),
            arguments.predecessor_tokens.resolve(),
        )
    else:
        stage(
            arguments.output.resolve(),
            arguments.tokens.resolve(),
            arguments.tokenizer.resolve(),
            arguments.registry.resolve(),
            arguments.predecessor_tokens.resolve(),
        )
        print(f"mixed-collapse staging: created {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
