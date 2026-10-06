#!/usr/bin/env python3
"""Stage the digest-bound block-normal Q0-Q5 campaign."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOUNDATION = (
    ROOT.parent
    / "experiment-data"
    / "shared"
    / "block-normal-registration-foundation"
)
DEFAULT_OUTPUT = (
    ROOT.parent
    / "experiment-data"
    / "campaigns"
    / "block-normal"
    / "block-normal-collapse-confirmation"
)
DEFAULT_DATASET = (
    ROOT.parent
    / "experiment-data"
    / "shared"
    / "datasets"
    / "refinedweb-100m-tokens"
    / "refinedweb_tokens.npy"
)
DEFAULT_TOKENIZER = ROOT / "experiments" / "data" / "tokenizer.model"
PROTOCOL_SOURCE = (
    ROOT / "experiments" / "protocols" / "block_normal_confirmation"
)
sys.path.insert(0, str(ROOT / "experiments"))

from confirm.block_normal_specs import (  # noqa: E402
    CAMPAIGN_ID,
    REGISTRY_SCHEMA,
    RESOURCE_BUDGET,
)
from confirm.native_determinism import sha256_path  # noqa: E402


SOURCE_SEEDS = (
    "experiments/train_run.py",
    "experiments/pldr_model_v510.py",
    "experiments/power_law_attention_layer_v510.py",
    "experiments/confirm/block_normal_specs.py",
    "experiments/confirm/finite_transport.py",
    "experiments/confirm/normal_projection.py",
    "experiments/confirm/block_normal.py",
    "experiments/confirm/native_determinism.py",
    "experiments/confirm/resource_executor.py",
    "experiments/confirm/resource_worker.py",
    "experiments/confirm/campaign_budget.py",
    "experiments/confirm/campaign_plan_executor.py",
    "experiments/confirm/checkpoint_retention.py",
    "experiments/confirm/evidence_seal.py",
    "experiments/confirm/capture_chronological_confirmation.py",
    "experiments/confirm/streamed_parameter_normal.py",
    "experiments/analysis/analyze_block_normal_confirmation.py",
    "scripts/gen_block_normal_protocols.py",
    "scripts/run_block_normal_qualification.py",
    "scripts/assemble_block_normal_record.py",
    "scripts/assemble_block_normal_stage.py",
    "scripts/stage_block_normal_confirmation.py",
    "scripts/freeze_block_normal_foundation.py",
    "scripts/execute_block_normal_plan.py",
    "scripts/seal_block_normal_evidence.py",
)

STAGE_COUNTS = {
    "Q1": 4,
    "Q2": 91,
    "Q3": 9,
    "Q4": 18,
    "Q5": 182,
}


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def digest_object(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)


def _resolve_import(module: str) -> Path | None:
    parts = module.split(".")
    experiments = ROOT / "experiments"
    candidates = (
        ROOT.joinpath(*parts).with_suffix(".py"),
        ROOT.joinpath(*parts, "__init__.py"),
        experiments.joinpath(*parts).with_suffix(".py"),
        experiments.joinpath(*parts, "__init__.py"),
        experiments.joinpath("confirm", *parts).with_suffix(".py"),
        experiments.joinpath("analysis", *parts).with_suffix(".py"),
        ROOT.joinpath("scripts", *parts).with_suffix(".py"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def source_import_closure() -> list[Path]:
    pending = [ROOT / relative for relative in SOURCE_SEEDS]
    selected: set[Path] = set()
    while pending:
        path = pending.pop().resolve()
        if path in selected:
            continue
        try:
            path.relative_to(ROOT)
        except ValueError as error:
            raise ValueError("source entry escapes the repository") from error
        if not path.is_file():
            raise FileNotFoundError(f"source closure is missing {path}")
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


def _stage_node(
    output: Path,
    python: str,
    stage: str,
    dependencies: list[str],
) -> dict[str, Any]:
    source = output / "evidence" / stage / "source"
    native = output / "evidence" / stage / "native"
    records = output / "records" / stage
    report = output / "reports" / f"{stage.lower()}-analysis.json"
    command = [
        python,
        str(output / "source" / "scripts"
            / "assemble_block_normal_stage.py"),
        "--source-dir", str(source),
        "--native-dir", str(native),
        "--output-dir", str(records),
        "--expected-count", str(STAGE_COUNTS[stage]),
        "--analysis-output", str(report),
    ]
    targets = {
        "Q1": 0.3,
        "Q2": 0.8,
        "Q3": 2.5,
        "Q4": 1.8,
        "Q5": 2.5,
    }
    return {
        "id": f"{stage.lower()}-assemble-analyze",
        "stage": stage,
        "device": "cpu",
        "depends_on": dependencies,
        "required_inputs": [str(source), str(native)],
        "command": command,
        "expected_outputs": [str(report)],
        "caps": {
            "wall_seconds": targets[stage] * 3600.0,
            "host_rss_bytes": 32 * 1024**3,
            "gpu_reserved_bytes": 0,
            "output_bytes": (
                RESOURCE_BUDGET["persistent_output_cap_bytes"]
            ),
            "poll_seconds": RESOURCE_BUDGET["poll_seconds"],
        },
    }


def launch_plan(
    output: Path,
    dataset: Path,
    tokenizer: Path,
    registry: Path,
    orders: list[dict[str, Any]],
) -> dict[str, Any]:
    python = str(Path(sys.executable).resolve())
    qualification = output / "source" / "scripts" / (
        "run_block_normal_qualification.py"
    )
    nodes = []
    for suffix, device in (
        ("cpu", "cpu"),
        ("cuda0", "cuda:0"),
        ("cuda1", "cuda:1"),
    ):
        report = output / "reports" / f"q0-{suffix}.json"
        nodes.append({
            "id": f"q0-{suffix}",
            "stage": "Q0",
            "device": device,
            "depends_on": [],
            "required_inputs": [],
            "command": [
                python,
                str(qualification),
                "--device", device,
                "--output", str(report),
                "--require-pass",
            ],
            "expected_outputs": [str(report)],
            "caps": {
                "wall_seconds": 360.0,
                "host_rss_bytes": 16 * 1024**3,
                "gpu_reserved_bytes": (
                    0 if device == "cpu" else
                    RESOURCE_BUDGET["process_reserved_memory_cap_bytes"]
                ),
                "output_bytes": 128 * 1024**2,
                "poll_seconds": 0.25,
            },
        })
    previous = ["q0-cpu", "q0-cuda0", "q0-cuda1"]
    for stage in ("Q1", "Q2", "Q3", "Q4", "Q5"):
        node = _stage_node(output, python, stage, previous)
        nodes.append(node)
        previous = [node["id"]]
    analyzer = (
        output / "source" / "experiments" / "analysis"
        / "analyze_block_normal_confirmation.py"
    )
    return {
        "schema_version": "pldr-block-normal-launch-plan-v1",
        "campaign_id": CAMPAIGN_ID,
        "bundle_root": str(output),
        "dataset": {
            "path": str(dataset),
            "sha256": sha256_path(dataset),
        },
        "tokenizer": {
            "path": str(tokenizer),
            "sha256": sha256_path(tokenizer),
        },
        "registry": {
            "path": str(registry),
            "sha256": sha256_path(registry),
        },
        "orders": orders,
        "resource_budget": RESOURCE_BUDGET,
        "stage_record_counts": STAGE_COUNTS,
        "nodes": nodes,
        "analysis_command": [
            python,
            str(analyzer),
            "--record-dirs",
            *[
                str(output / "records" / stage)
                for stage in ("Q1", "Q2", "Q3", "Q4", "Q5")
            ],
            "--output",
            str(output / "reports" / "final-analysis.json"),
            "--require-valid",
        ],
        "evidence_contract": {
            "source_file_suffix": ".source.json",
            "native_file_suffix": ".native.json",
            "source_must_precede_native": True,
            "native_binds_source_file_sha256": True,
            "checkpoint_paths_and_digests_required": True,
        },
    }


def _immutable_files(output: Path) -> list[Path]:
    selected = [output / "README.md"]
    for directory in ("orders", "protocol", "source"):
        selected.extend(
            path for path in (output / directory).rglob("*") if path.is_file()
        )
    return sorted(selected)


def _write_manifest(output: Path) -> None:
    lines = [
        f"{sha256_path(path)}  {path.relative_to(output).as_posix()}\n"
        for path in _immutable_files(output)
    ]
    (output / "MANIFEST.sha256").write_text("".join(lines), encoding="ascii")


def _validate_foundation(foundation: Path) -> None:
    manifest = foundation / "MANIFEST.sha256"
    if not manifest.is_file():
        raise FileNotFoundError("registration-foundation manifest is missing")
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        path = foundation / relative
        if not path.is_file() or sha256_path(path) != digest:
            raise ValueError(f"registration-foundation file changed: {relative}")


def build(
    foundation: Path,
    output: Path,
    dataset: Path,
    tokenizer: Path,
) -> None:
    foundation = foundation.resolve()
    output = output.resolve()
    dataset = dataset.resolve()
    tokenizer = tokenizer.resolve()
    if (
        not (foundation / "MANIFEST.sha256").is_file()
        or not (PROTOCOL_SOURCE / "CHECKSUMS.sha256").is_file()
        or not dataset.is_file()
        or not tokenizer.is_file()
    ):
        raise FileNotFoundError(
            "foundation, protocol, dataset, or tokenizer is missing"
        )
    _validate_foundation(foundation)
    for directory in (
        "orders", "protocol", "source", "evidence", "records", "reports",
        "runtime", "runs",
    ):
        (output / directory).mkdir(parents=True, exist_ok=True)
    for stage in ("Q1", "Q2", "Q3", "Q4", "Q5"):
        for directory in (
            output / "evidence" / stage / "source",
            output / "evidence" / stage / "native",
            output / "records" / stage,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    orders = []
    for source in sorted((foundation / "orders").glob("*.npy")):
        target = output / "orders" / source.name
        copy_file(source, target)
        orders.append({
            "path": str(target),
            "sha256": sha256_path(target),
            "foundation_path": str(source.resolve()),
            "foundation_sha256": sha256_path(source),
        })
    if len(orders) != 3:
        raise ValueError("the staged campaign requires exactly three orders")

    foundation_registry = foundation / "registry.json"
    registry = json.loads(foundation_registry.read_text(encoding="utf-8"))
    registry["schema_version"] = REGISTRY_SCHEMA
    registry["campaign_id"] = CAMPAIGN_ID
    registry["map_count"] = 288
    registry["within_map_pair_count"] = 580_608
    registry.pop("registry_sha256", None)
    registry["registry_sha256"] = digest_object(registry)
    registry_path = output / "protocol" / "registry.json"
    registry_path.write_bytes(canonical(registry))
    if (
        registry["dataset_sha256"] != sha256_path(dataset)
        or registry["tokenizer_sha256"] != sha256_path(tokenizer)
        or len(registry["construction"]) != 8
        or len(registry["validation"]) != 16
    ):
        raise ValueError(
            "registration foundation does not bind the registered population"
        )

    for source in sorted(PROTOCOL_SOURCE.iterdir()):
        if source.is_file():
            copy_file(source, output / "protocol" / source.name)
    (output / "protocol" / "order_manifest.json").write_bytes(
        canonical({
            "schema_version": "pldr-block-normal-order-manifest-v1",
            "campaign_id": CAMPAIGN_ID,
            "orders": orders,
        })
    )

    source_rows = []
    for source in source_import_closure():
        relative = source.relative_to(ROOT)
        target = output / "source" / relative
        copy_file(source, target)
        source_rows.append({
            "path": relative.as_posix(),
            "sha256": sha256_path(target),
            "bytes": target.stat().st_size,
        })
    source_manifest = {
        "schema_version": "pldr-block-normal-source-manifest-v1",
        "campaign_id": CAMPAIGN_ID,
        "files": source_rows,
    }
    source_manifest["manifest_sha256"] = digest_object(source_manifest)
    (output / "source" / "SOURCE_MANIFEST.json").write_bytes(
        canonical(source_manifest)
    )

    plan = launch_plan(output, dataset, tokenizer, registry_path, orders)
    (output / "protocol" / "launch_plan.json").write_bytes(canonical(plan))
    provenance = {
        "schema_version": "pldr-block-normal-stage-provenance-v1",
        "campaign_id": CAMPAIGN_ID,
        "foundation_path": str(foundation),
        "foundation_manifest_sha256": sha256_path(
            foundation / "MANIFEST.sha256"
        ),
        "repository_path": str(ROOT),
        "dataset_path": str(dataset),
        "dataset_sha256": sha256_path(dataset),
        "tokenizer_path": str(tokenizer),
        "tokenizer_sha256": sha256_path(tokenizer),
        "launch_plan_sha256": sha256_path(
            output / "protocol" / "launch_plan.json"
        ),
    }
    (output / "protocol" / "PROVENANCE.json").write_bytes(
        canonical(provenance)
    )
    (output / "README.md").write_text(
        "Block-normal Q0-Q5 campaign bundle. Immutable protocol, orders, "
        "and source are covered by MANIFEST.sha256. Evidence, records, "
        "reports, runtime records, and runs are append-only campaign data. "
        "Q0 is deterministic fixture qualification, not scientific "
        "confirmation. Checkpoint-backed implemented-model measurements "
        "begin at Q1.\n",
        encoding="utf-8",
    )
    _write_manifest(output)
    validate(output)


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def validate(output: Path) -> None:
    output = output.resolve()
    manifest = output / "MANIFEST.sha256"
    if not manifest.is_file():
        raise FileNotFoundError("staged campaign manifest is missing")
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        path = output / relative
        if not path.is_file() or sha256_path(path) != digest:
            raise ValueError(f"staged immutable file changed: {relative}")
    plan_path = output / "protocol" / "launch_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        plan.get("schema_version") != "pldr-block-normal-launch-plan-v1"
        or plan.get("campaign_id") != CAMPAIGN_ID
        or len(plan.get("nodes", [])) != 8
        or any(
            "{" in value or "}" in value or "PLACEHOLDER" in value
            for value in _strings(plan)
        )
        or any(
            not Path(plan[name]["path"]).is_absolute()
            for name in ("dataset", "tokenizer", "registry")
        )
    ):
        raise ValueError("staged launch plan is incomplete or contains placeholders")
    ids = [node["id"] for node in plan["nodes"]]
    if len(ids) != len(set(ids)):
        raise ValueError("launch node identifiers are duplicated")
    print(
        "block-normal stage: verified "
        f"({len(ids)} nodes, {len(_immutable_files(output))} immutable files)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--foundation", type=Path, default=DEFAULT_FOUNDATION
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if arguments.check:
        validate(arguments.output)
    else:
        build(
            arguments.foundation,
            arguments.output,
            arguments.dataset,
            arguments.tokenizer,
        )


if __name__ == "__main__":
    main()
