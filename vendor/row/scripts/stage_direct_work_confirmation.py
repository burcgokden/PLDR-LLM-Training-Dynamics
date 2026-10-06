#!/usr/bin/env python3
"""Stage the digest-bound direct-work confirmation graph."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from confirm.direct_work_specs import (  # noqa: E402
    CAMPAIGN_ID,
    LAYERS,
    RELEASE_ID,
    SOURCE_STEPS,
    TRAJECTORIES,
    campaign_design,
)


SOURCE_FILES = (
    Path("release.json"),
    Path("experiments/confirm/direct_work.py"),
    Path("experiments/confirm/resource_executor.py"),
    Path("experiments/confirm/resource_worker.py"),
    Path("experiments/confirm/direct_work_specs.py"),
    Path("experiments/confirm/direct_work_live.py"),
    Path("experiments/analysis/analyze_direct_work_confirmation.py"),
    Path("scripts/stage_direct_work_confirmation.py"),
    Path("scripts/execute_direct_work_plan.py"),
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _bound_path(path: Path, binding_root: Path) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(binding_root):
        raise ValueError(f"input escapes binding root: {resolved}")
    return resolved


def _node(
    *,
    identifier: str,
    phase: str,
    role: str,
    trajectory: dict[str, Any],
    step: int,
    layer: int,
    device: str,
    checkpoint: Path,
    order: Path,
    registry: Path,
    tokens: Path,
    output: Path,
    qualification: bool = False,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str((ROOT / "experiments/confirm/direct_work_live.py").resolve()),
        "--checkpoint", str(checkpoint),
        "--registry", str(registry),
        "--tokens", str(tokens),
        "--data-order", str(order),
        "--trajectory", str(trajectory["label"]),
        "--seed", str(trajectory["seed"]),
        "--step", str(step),
        "--layer", str(layer),
        "--role", role,
        "--device", device,
        "--output", str(output),
    ]
    if qualification:
        command.append("--qualification")
    return {
        "node_id": identifier,
        "phase": phase,
        "role": role,
        "device": device,
        "command": command,
        "output": str(output),
    }


def stage(binding_root: Path, campaign_root: Path) -> dict[str, Any]:
    binding_root = binding_root.resolve(strict=True)
    campaign_root = campaign_root.resolve(strict=False)
    if not campaign_root.is_relative_to(binding_root):
        raise ValueError("campaign root escapes binding root")
    prior = _bound_path(
        binding_root
        / "experiment-data/campaigns/mixed-collapse/"
        "mixed-row-map-collapse-confirmation",
        binding_root,
    )
    registry = _bound_path(prior / "protocol/registry.json", binding_root)
    tokens = _bound_path(
        binding_root
        / "experiment-data/shared/datasets/refinedweb-538m-prefix-locked/"
        "refinedweb_tokens.npy",
        binding_root,
    )
    source_digests = {
        path.as_posix(): sha256_path(_bound_path(ROOT / path, binding_root))
        for path in SOURCE_FILES
    }
    input_digests = {
        "registry": sha256_path(registry),
        "tokens": sha256_path(tokens),
    }
    for trajectory in TRAJECTORIES:
        order = _bound_path(prior / "orders" / trajectory["order"], binding_root)
        input_digests[f"order-{trajectory['label']}"] = sha256_path(order)
        for step in ((1_024,) if trajectory["label"] == "A" else ()) + SOURCE_STEPS:
            checkpoint = _bound_path(
                prior / "runs" / trajectory["name"] / f"ckpt_{step}.pt",
                binding_root,
            )
            input_digests[f"checkpoint-{trajectory['label']}-{step}"] = (
                sha256_path(checkpoint)
            )

    records = campaign_root / "records"
    nodes: list[dict[str, Any]] = []
    construction = TRAJECTORIES[0]
    nodes.append(_node(
        identifier="q0-A-S1024-L0",
        phase="qualification",
        role="development",
        trajectory=construction,
        step=1_024,
        layer=0,
        device="cuda:0",
        checkpoint=prior / "runs" / construction["name"] / "ckpt_1024.pt",
        order=prior / "orders" / construction["order"],
        registry=registry,
        tokens=tokens,
        output=records / "qualification-A-S1024-L0.npz",
        qualification=True,
    ))
    ordinal = 0
    for trajectory in TRAJECTORIES:
        phase = "construction" if trajectory["role"] == "construction" else "heldout"
        for step in SOURCE_STEPS:
            for layer in LAYERS:
                device = f"cuda:{ordinal % 2}"
                ordinal += 1
                identifier = f"{phase}-{trajectory['label']}-S{step}-L{layer}"
                nodes.append(_node(
                    identifier=identifier,
                    phase=phase,
                    role=trajectory["role"],
                    trajectory=trajectory,
                    step=step,
                    layer=layer,
                    device=device,
                    checkpoint=(
                        prior / "runs" / trajectory["name"] / f"ckpt_{step}.pt"
                    ),
                    order=prior / "orders" / trajectory["order"],
                    registry=registry,
                    tokens=tokens,
                    output=records / f"{trajectory['label']}-S{step}-L{layer}.npz",
                ))
    analyzer = str(
        (ROOT / "experiments/analysis/analyze_direct_work_confirmation.py").resolve()
    )
    nodes.extend((
        {
            "node_id": "construction-lock",
            "phase": "lock",
            "role": "analysis",
            "device": "cpu",
            "command": [
                sys.executable, analyzer, "lock",
                "--campaign-root", str(campaign_root),
            ],
            "output": str(records / "construction-lock.json"),
        },
        {
            "node_id": "final-analysis",
            "phase": "analysis",
            "role": "analysis",
            "device": "cpu",
            "command": [
                sys.executable, analyzer, "final",
                "--campaign-root", str(campaign_root),
                "--tex-output", str(ROOT / "docs/figures"),
            ],
            "output": str(campaign_root / "analysis/summary.json"),
        },
    ))
    design = campaign_design()
    design["source_sha256"] = source_digests
    design["input_sha256"] = input_digests
    design["binding_root"] = str(binding_root)
    plan = {
        "schema_version": "pldr-direct-work-launch-plan-v1",
        "campaign_id": CAMPAIGN_ID,
        "binding_root": str(binding_root),
        "campaign_root": str(campaign_root),
        "node_count": len(nodes),
        "nodes": nodes,
    }
    return {"design": design, "plan": plan}


def materialize(bundle: dict[str, Any], campaign_root: Path) -> None:
    write_bytes(
        campaign_root / "protocol/design.json", canonical_bytes(bundle["design"])
    )
    write_bytes(
        campaign_root / "protocol/launch-plan.json",
        canonical_bytes(bundle["plan"]),
    )
    for relative in SOURCE_FILES:
        target = campaign_root / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)


def check(bundle: dict[str, Any], campaign_root: Path) -> None:
    expected = {
        campaign_root / "protocol/design.json": canonical_bytes(bundle["design"]),
        campaign_root / "protocol/launch-plan.json": canonical_bytes(bundle["plan"]),
    }
    for path, payload in expected.items():
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"staged direct-work protocol is stale: {path}")
    for relative in SOURCE_FILES:
        target = campaign_root / "source" / relative
        if not target.is_file() or target.read_bytes() != (ROOT / relative).read_bytes():
            raise ValueError(f"staged direct-work source is stale: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    binding_root = arguments.binding_root.resolve(strict=True)
    campaign_root = (
        arguments.campaign_root.resolve(strict=False)
        if arguments.campaign_root is not None
        else binding_root
        / "experiment-data/campaigns"
        / RELEASE_ID
        / "direct-work-source-intervention"
    )
    bundle = stage(binding_root, campaign_root)
    if arguments.check:
        check(bundle, campaign_root)
        print("direct-work staging: verified")
    else:
        materialize(bundle, campaign_root)
        print(f"direct-work staging: wrote {bundle['plan']['node_count']} nodes")


if __name__ == "__main__":
    main()
