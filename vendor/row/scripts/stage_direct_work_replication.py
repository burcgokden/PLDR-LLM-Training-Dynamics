#!/usr/bin/env python3
"""Stage the frozen temporal-context direct-work replication."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

from bundle_hygiene import assert_clean_tree, is_excluded_path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from confirm.direct_work_replication_specs import (  # noqa: E402
    CAMPAIGN_ID,
    LAYERS,
    RELEASE_ID,
    SOURCE_STEPS,
    TRAJECTORIES,
    campaign_design,
)


SOURCE_SCRIPT_FILES = (
    Path("scripts/bundle_hygiene.py"),
    Path("scripts/stage_direct_work_replication.py"),
    Path("scripts/execute_direct_work_replication.py"),
    Path("scripts/seal_direct_work_replication.py"),
)
ANALYSIS_FILES = (
    Path("experiments/analysis/analyze_direct_work_precision.py"),
    Path("experiments/analysis/analyze_direct_work_replication.py"),
    Path("experiments/analysis/sign_semantics.py"),
)
GENERATED_RELEASE = {
    "schema_version": "pldr-release-v1",
    "release_id": RELEASE_ID,
    "manuscript_components": [],
    "artifact": {
        "name": f"pldr-curvature-sandpile-{RELEASE_ID}",
        "export_directory": f"paper-outputs-{RELEASE_ID}",
        "export_target": f"export-{RELEASE_ID}",
    },
}
REGISTRY_CHUNKS = tuple(range(2_097_176, 2_097_200))
TERMINAL_BATCH_CHUNKS = tuple(range(2_097_200, 2_097_232))


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_file_bytes(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def digest_object(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def source_files() -> tuple[Path, ...]:
    files = {
        path.relative_to(ROOT)
        for path in EXPERIMENTS.glob("*.py")
        if path.is_file()
    }
    files.update(
        path.relative_to(ROOT)
        for path in (EXPERIMENTS / "confirm").glob("*.py")
        if path.is_file()
    )
    files.update(ANALYSIS_FILES)
    files.update(SOURCE_SCRIPT_FILES)
    excluded = [relative for relative in files if is_excluded_path(relative)]
    if excluded:
        raise ValueError(
            f"replication source selection includes bytecode: {excluded}"
        )
    missing = [relative for relative in files if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"replication source files are missing: {missing}")
    return tuple(sorted(files))


def bound_file(path: Path, binding_root: Path) -> Path:
    resolved = path.resolve(strict=True)
    if (
        not resolved.is_relative_to(binding_root)
        or resolved.is_symlink()
        or not resolved.is_file()
    ):
        raise ValueError(f"invalid bound file: {resolved}")
    return resolved


def new_registry(prior: dict[str, Any]) -> dict[str, Any]:
    old_chunks = {
        int(row["chunk_index"])
        for group in ("construction", "validation")
        for row in prior[group]
    }
    if old_chunks & set(REGISTRY_CHUNKS):
        raise ValueError("replication registry overlaps the earlier registry")
    registry = dict(prior)
    registry["campaign_id"] = CAMPAIGN_ID
    registry["construction"] = [
        {"id": f"r{index:02d}", "chunk_index": chunk}
        for index, chunk in enumerate(REGISTRY_CHUNKS[:8])
    ]
    registry["validation"] = [
        {"id": f"r{index:02d}", "chunk_index": chunk}
        for index, chunk in enumerate(REGISTRY_CHUNKS[8:], start=8)
    ]
    registry.pop("registry_sha256", None)
    registry["registry_sha256"] = digest_object(registry)
    return registry


def terminal_batch(registry: dict[str, Any]) -> dict[str, Any]:
    if set(REGISTRY_CHUNKS) & set(TERMINAL_BATCH_CHUNKS):
        raise ValueError("terminal update batch overlaps measurement contexts")
    payload = {
        "schema_version": "pldr-reserve-update-batch-v1",
        "context_length": int(registry["context_length"]),
        "batch_size": len(TERMINAL_BATCH_CHUNKS),
        "dataset_sha256": registry["dataset_sha256"],
        "chunk_indices": list(TERMINAL_BATCH_CHUNKS),
    }
    payload["content_sha256"] = digest_object(payload)
    return payload


def checkpoint_file(prior: Path, trajectory: dict[str, Any], step: int) -> Path:
    name = "ckpt_final.pt" if step == 65_536 else f"ckpt_{step}.pt"
    return prior / "runs" / trajectory["name"] / name


def producer_command(
    source: Path,
    *,
    checkpoint: Path,
    registry: Path,
    tokens: Path,
    order: Path,
    trajectory: dict[str, Any],
    step: int,
    layer: int,
    role: str,
    device: str,
    output: Path,
    terminal_batch_path: Path,
    qualification: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        str(source / "experiments/confirm/direct_work_replication_live.py"),
        "--checkpoint",
        str(checkpoint),
        "--registry",
        str(registry),
        "--tokens",
        str(tokens),
        "--data-order",
        str(order),
        "--trajectory",
        str(trajectory["label"]),
        "--seed",
        str(trajectory["seed"]),
        "--step",
        str(step),
        "--layer",
        str(layer),
        "--role",
        role,
        "--device",
        device,
        "--allow-disjoint-registry",
        "--output",
        str(output),
    ]
    if step == 65_536:
        command.extend([
            "--update-batch-registry",
            str(terminal_batch_path),
        ])
    if qualification:
        command.append("--qualification")
    return command


def staged_objects(
    binding_root: Path, bundle: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    prior = bound_file(
        binding_root
        / "experiment-data/manuscript-revisions/rev47"
        / "mixed-row-map-collapse-confirmation/protocol/registry.json",
        binding_root,
    ).parents[1]
    prior_registry_path = prior / "protocol/registry.json"
    prior_registry = load_json(prior_registry_path)
    construction_incident = bound_file(
        binding_root
        / "experiment-data/manuscript-revisions/rev52/incidents"
        / "direct-work-replication-construction-rounding-bound-20260831T1818Z"
        / "incident.json",
        binding_root,
    )
    registry = new_registry(prior_registry)
    reserve_batch = terminal_batch(registry)
    tokens = bound_file(
        binding_root
        / "experiment-data/shared/datasets/refinedweb-538m-prefix-locked"
        / "refinedweb_tokens.npy",
        binding_root,
    )
    source = bundle / "source"
    protocol = bundle / "protocol"
    records = bundle / "records"
    registry_path = protocol / "registry.json"
    terminal_batch_path = protocol / "terminal-update-batch.json"

    generated_release_bytes = canonical_file_bytes(GENERATED_RELEASE)
    sources = source_files()
    source_digests = {
        relative.as_posix(): sha256_path(ROOT / relative)
        for relative in sources
    }
    source_digests["release.json"] = hashlib.sha256(
        generated_release_bytes
    ).hexdigest()
    input_digests = {
        "tokens": sha256_path(tokens),
        "prior_registry": sha256_path(prior_registry_path),
        "construction_bound_incident": sha256_path(construction_incident),
        "replication_registry": hashlib.sha256(
            canonical_file_bytes(registry)
        ).hexdigest(),
        "terminal_update_batch": hashlib.sha256(
            canonical_file_bytes(reserve_batch)
        ).hexdigest(),
    }
    bound_inputs: dict[str, str] = {
        "tokens": str(tokens),
        "prior_registry": str(prior_registry_path),
        "construction_bound_incident": str(construction_incident),
        "replication_registry": str(registry_path),
        "terminal_update_batch": str(terminal_batch_path),
    }
    for trajectory in TRAJECTORIES:
        order = bound_file(prior / "orders" / trajectory["order"], binding_root)
        input_digests[f"order-{trajectory['label']}"] = sha256_path(order)
        bound_inputs[f"order-{trajectory['label']}"] = str(order)
        for step in SOURCE_STEPS:
            checkpoint = bound_file(
                checkpoint_file(prior, trajectory, step), binding_root
            )
            input_digests[f"checkpoint-{trajectory['label']}-{step}"] = (
                sha256_path(checkpoint)
            )
            bound_inputs[f"checkpoint-{trajectory['label']}-{step}"] = str(
                checkpoint
            )

    nodes: list[dict[str, Any]] = []
    construction = TRAJECTORIES[0]
    q_checkpoint = checkpoint_file(prior, construction, 18_000)
    q_order = prior / "orders" / construction["order"]
    for device_index in (0, 1):
        identifier = f"qualification-cuda{device_index}"
        output = records / f"{identifier}.npz"
        nodes.append({
            "node_id": identifier,
            "phase": "qualification",
            "role": "development",
            "device": f"cuda:{device_index}",
            "command": producer_command(
                source,
                checkpoint=q_checkpoint,
                registry=registry_path,
                tokens=tokens,
                order=q_order,
                trajectory=construction,
                step=18_000,
                layer=0,
                role="development",
                device=f"cuda:{device_index}",
                output=output,
                terminal_batch_path=terminal_batch_path,
                qualification=True,
            ),
            "expected_outputs": [str(output)],
        })
    ordinal = 0
    for trajectory in TRAJECTORIES:
        phase = (
            "construction"
            if trajectory["role"] == "construction"
            else "heldout"
        )
        for step in SOURCE_STEPS:
            for layer in LAYERS:
                device = f"cuda:{ordinal % 2}"
                ordinal += 1
                output = records / f"{trajectory['label']}-S{step}-L{layer}.npz"
                nodes.append({
                    "node_id": (
                        f"{phase}-{trajectory['label']}-S{step}-L{layer}"
                    ),
                    "phase": phase,
                    "role": trajectory["role"],
                    "device": device,
                    "command": producer_command(
                        source,
                        checkpoint=checkpoint_file(prior, trajectory, step),
                        registry=registry_path,
                        tokens=tokens,
                        order=prior / "orders" / trajectory["order"],
                        trajectory=trajectory,
                        step=step,
                        layer=layer,
                        role=trajectory["role"],
                        device=device,
                        output=output,
                        terminal_batch_path=terminal_batch_path,
                    ),
                    "expected_outputs": [str(output)],
                })
    analyzer = source / "experiments/analysis/analyze_direct_work_replication.py"
    lock_output = records / "construction-lock.json"
    nodes.append({
        "node_id": "construction-lock",
        "phase": "lock",
        "role": "analysis",
        "device": "cpu",
        "command": [
            sys.executable,
            str(analyzer),
            "lock",
            "--campaign-root",
            str(bundle),
        ],
        "expected_outputs": [str(lock_output)],
    })
    analysis_outputs = [
        bundle / "analysis/summary.json",
        bundle / "analysis/direct_work_replication_macros.tex",
        bundle / "analysis/direct_work_replication_results.tex",
    ]
    final_command = [
        sys.executable,
        str(analyzer),
        "final",
        "--campaign-root",
        str(bundle),
    ]
    nodes.append({
        "node_id": "final-analysis",
        "phase": "analysis",
        "role": "analysis",
        "device": "cpu",
        "command": final_command,
        "check_command": [*final_command, "--check"],
        "expected_outputs": [str(path) for path in analysis_outputs],
    })
    design = campaign_design()
    design.update({
        "binding_root": str(binding_root),
        "campaign_root": str(bundle),
        "source_sha256": source_digests,
        "input_sha256": input_digests,
        "bound_inputs": bound_inputs,
        "registry_sha256": registry["registry_sha256"],
        "terminal_update_batch_sha256": reserve_batch["content_sha256"],
        "source_file_count": len(source_digests),
    })
    plan = {
        "schema_version": "pldr-direct-work-replication-launch-v1",
        "campaign_id": CAMPAIGN_ID,
        "binding_root": str(binding_root),
        "campaign_root": str(bundle),
        "node_count": len(nodes),
        "nodes": nodes,
    }
    return design, plan, registry, reserve_batch


def payloads(binding_root: Path, bundle: Path) -> dict[Path, bytes]:
    design, plan, registry, reserve_batch = staged_objects(
        binding_root, bundle
    )
    readme = (
        "# Temporal-context direct-work replication\n\n"
        "This campaign executes a paired one-step intervention at two unused "
        "chronological checkpoints and one terminal counterfactual "
        "continuation. It uses 24 registered measurement contexts disjoint "
        "from the earlier observer registry. The terminal checkpoint has no "
        "remaining entry in its historical training order, so its successor "
        "uses a separately digested 32-context reserve batch and is never "
        "described as a historical training successor.\n\n"
        "A construction-only protocol check found that a cancellation "
        "residual had been bounded by derived small quantities instead of "
        "the primitive coordinates that generated it. No heldout node was "
        "opened. The retained incident is bound as an external input, and "
        "this campaign uses primitive-coordinate and finite-sum backward "
        "error bounds.\n\n"
        "Primary eligibility uses native row-map energy, paired-state "
        "invariants, a magnitude-aware native work-charge bound, the frozen "
        "effect floor, and exact four-source reconstruction. The magnitude "
        "of the native-versus-factorized implementation defect is a "
        "secondary result and is not an eligibility filter.\n"
    ).encode("utf-8")
    return {
        Path("protocol/design.json"): canonical_file_bytes(design),
        Path("protocol/launch-plan.json"): canonical_file_bytes(plan),
        Path("protocol/registry.json"): canonical_file_bytes(registry),
        Path("protocol/terminal-update-batch.json"): canonical_file_bytes(
            reserve_batch
        ),
        Path("README.md"): readme,
    }


def stage(binding_root: Path, bundle: Path, *, check: bool) -> None:
    binding_root = binding_root.resolve(strict=True)
    bundle = bundle.resolve(strict=False)
    if not bundle.is_relative_to(binding_root):
        raise ValueError("replication bundle escapes binding root")
    assert_clean_tree(bundle)
    generated = payloads(binding_root, bundle)
    sources = source_files()
    generated_release = canonical_file_bytes(GENERATED_RELEASE)
    if check:
        stale = [
            str(relative) for relative, payload in generated.items()
            if not (bundle / relative).is_file()
            or (bundle / relative).read_bytes() != payload
        ]
        for relative in sources:
            target = bundle / "source" / relative
            if (
                target.is_symlink()
                or not target.is_file()
                or target.read_bytes() != (ROOT / relative).read_bytes()
            ):
                stale.append(f"source/{relative}")
        if (
            not (bundle / "source/release.json").is_file()
            or (bundle / "source/release.json").read_bytes() != generated_release
        ):
            stale.append("source/release.json")
        if stale:
            raise SystemExit(
                "direct-work replication staging is stale: "
                + ", ".join(stale)
            )
        print("direct-work replication staging: current")
        return
    if bundle.exists():
        raise FileExistsError(f"refusing to overwrite replication: {bundle}")
    for relative, payload in generated.items():
        write_bytes(bundle / relative, payload)
    for relative in sources:
        target = bundle / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    write_bytes(bundle / "source/release.json", generated_release)
    assert_clean_tree(bundle)
    print(
        "direct-work replication staging: wrote "
        f"{len(staged_objects(binding_root, bundle)[1]['nodes'])} nodes"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    binding_root = arguments.binding_root.resolve(strict=True)
    bundle = (
        arguments.campaign_root.resolve(strict=False)
        if arguments.campaign_root is not None
        else binding_root
        / "experiment-data/manuscript-revisions"
        / RELEASE_ID
        / "direct-work-temporal-context-replication"
    )
    stage(binding_root, bundle, check=arguments.check)


if __name__ == "__main__":
    main()
