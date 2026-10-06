#!/usr/bin/env python3
"""Stage an immutable-ready observable full-state cocycle confirmation bundle."""

from __future__ import annotations
from companion_paths import configured_path

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from confirm.observable_cocycle_specs import (  # noqa: E402
    CAMPAIGN_ID,
    SOURCE_STEPS,
    TRAJECTORIES,
)
from scripts.gen_observable_cocycle_protocols import (  # noqa: E402
    canonical,
    generated_files,
    record_inventory,
)


DEFAULT_INPUT = (
    PROJECTS
    / "experiment-data"
    / "campaigns"
    / "mixed-collapse"
    / "mixed-row-map-collapse-confirmation"
)
DEFAULT_OUTPUT = (
    PROJECTS
    / "experiment-data"
    / "campaigns"
    / "observable-cocycle"
    / "observable-full-state-cocycle-confirmation"
)
DEFAULT_TOKENS = (
    PROJECTS
    / "experiment-data"
    / "shared"
    / "datasets"
    / "refinedweb-538m-prefix-locked"
    / "refinedweb_tokens.npy"
)

ENTRYPOINTS = (
    "experiments/confirm/observable_cocycle_live.py",
    "experiments/analysis/analyze_observable_cocycle_confirmation.py",
    "scripts/execute_observable_cocycle_plan.py",
    "scripts/gen_observable_cocycle_protocols.py",
    "scripts/stage_observable_cocycle_confirmation.py",
)

LAUNCH_AUTHORIZATION = (
    ROOT / "experiments" / "protocols" / "observable_cocycle_confirmation"
    / "launch-authorization.json"
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    """Return the exact local import closure of campaign entrypoints."""

    pending = [(ROOT / relative).resolve() for relative in ENTRYPOINTS]
    selected: set[Path] = set()
    while pending:
        path = pending.pop()
        path.relative_to(ROOT)
        if path in selected:
            continue
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


def _common_live(
    command: str,
    trajectory: dict[str, Any],
    step: int,
    output: str,
    *,
    order: bool = False,
    layer: int | None = None,
) -> list[str]:
    label = str(trajectory["label"])
    result = [
        "{python}",
        "{bundle}/source/experiments/confirm/observable_cocycle_live.py",
        command,
        "--checkpoint",
        f"{{checkpoint:{label}:{step}}}",
        "--tokens",
        "{tokens}",
        "--registry",
        "{bundle}/protocol/registry.json",
    ]
    if order:
        result.extend(
            ["--order", f"{{bundle}}/orders/{trajectory['order']}"]
        )
    if layer is not None:
        result.extend(["--layer", str(layer)])
    result.extend(
        ["--device", str(trajectory["device"]), "--output", f"{{bundle}}/{output}"]
    )
    return result


def launch_plan() -> dict[str, Any]:
    """Return the complete outcome-independent execution DAG."""

    inventory = record_inventory()
    nodes: list[dict[str, Any]] = []
    trajectories = {str(row["label"]): row for row in TRAJECTORIES}

    qualification = inventory["qualification"]
    qualification_id = "qualification-A-L0-S4096"
    qualification_command = _common_live(
        "qualify",
        trajectories["A"],
        int(qualification["step"]),
        qualification["path"],
        order=True,
        layer=int(qualification["layer"]),
    )
    qualification_command.append("--require-pass")
    nodes.append(
        {
            "id": qualification_id,
            "stage": "qualification",
            "device": trajectories["A"]["device"],
            "depends_on": [],
            "command": qualification_command,
            "expected_outputs": [qualification["path"]],
        }
    )

    calibration_ids = []
    for record in inventory["calibration"]:
        label = str(record["trajectory"])
        node_id = (
            f"calibration-{label}-L{record['layer']}-S{record['step']}"
        )
        calibration_ids.append(node_id)
        nodes.append(
            {
                "id": node_id,
                "stage": "calibration",
                "device": trajectories[label]["device"],
                "depends_on": [qualification_id],
                "command": _common_live(
                    "calibrate",
                    trajectories[label],
                    int(record["step"]),
                    record["path"],
                    order=True,
                    layer=int(record["layer"]),
                ),
                "expected_outputs": [record["path"]],
            }
        )

    lock_id = "construction-lock"
    nodes.append(
        {
            "id": lock_id,
            "stage": "lock",
            "device": "cpu",
            "depends_on": calibration_ids,
            "command": [
                "{python}",
                (
                    "{bundle}/source/experiments/analysis/"
                    "analyze_observable_cocycle_confirmation.py"
                ),
                "lock",
                "--bundle",
                "{bundle}",
                "--output",
                f"{{bundle}}/{inventory['lock_output']}",
            ],
            "expected_outputs": [inventory["lock_output"]],
        }
    )

    prediction_ids = {}
    for record in inventory["prediction"]:
        label = str(record["trajectory"])
        stem = f"{label}-L{record['layer']}-S{record['step']}"
        node_id = f"prediction-{stem}"
        prediction_ids[(label, int(record["layer"]), int(record["step"]))] = node_id
        command = _common_live(
            "predict",
            trajectories[label],
            int(record["step"]),
            record["path"],
            order=True,
            layer=int(record["layer"]),
        )
        command.extend(["--lock", f"{{bundle}}/{inventory['lock_output']}"])
        nodes.append(
            {
                "id": node_id,
                "stage": "prediction",
                "device": trajectories[label]["device"],
                "depends_on": [lock_id],
                "command": command,
                "expected_outputs": [record["path"]],
            }
        )

    validation_ids = []
    for record in inventory["validation"]:
        label = str(record["trajectory"])
        key = (label, int(record["layer"]), int(record["step"]))
        stem = f"{label}-L{record['layer']}-S{record['step']}"
        node_id = f"validation-{stem}"
        validation_ids.append(node_id)
        command = _common_live(
            "validate",
            trajectories[label],
            int(record["step"]),
            record["path"],
            order=True,
            layer=int(record["layer"]),
        )
        command.extend(
            ["--prediction", f"{{bundle}}/{record['prediction_path']}"]
        )
        nodes.append(
            {
                "id": node_id,
                "stage": "validation",
                "device": trajectories[label]["device"],
                "depends_on": [prediction_ids[key]],
                "command": command,
                "expected_outputs": [record["path"]],
            }
        )

    stratum_ids = []
    for record in inventory["stratum"]:
        label = str(record["trajectory"])
        node_id = f"stratum-{label}-L{record['layer']}"
        stratum_ids.append(node_id)
        nodes.append(
            {
                "id": node_id,
                "stage": "stratum",
                "device": trajectories[label]["device"],
                "depends_on": [qualification_id],
                "command": _common_live(
                    "stratum",
                    trajectories[label],
                    int(record["step"]),
                    record["path"],
                    order=True,
                    layer=int(record["layer"]),
                ),
                "expected_outputs": [record["path"]],
            }
        )

    nodes.append(
        {
            "id": "analysis",
            "stage": "analysis",
            "device": "cpu",
            "depends_on": [
                qualification_id,
                lock_id,
                *validation_ids,
                *stratum_ids,
            ],
            "command": [
                "{python}",
                (
                    "{bundle}/source/experiments/analysis/"
                    "analyze_observable_cocycle_confirmation.py"
                ),
                "final",
                "--bundle",
                "{bundle}",
            ],
            "expected_outputs": inventory["analysis_outputs"],
        }
    )
    return {
        "schema_version": "pldr-observable-cocycle-launch-plan-v1",
        "campaign_id": CAMPAIGN_ID,
        "paths_are_bundle_relative": True,
        "scientific_outcomes_control_execution": False,
        "nodes": nodes,
    }


def _git_clean_head() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    if status:
        raise ValueError("campaign staging requires a clean committed source tree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()

def _validated_launch_authorization(head: str) -> dict[str, Any]:
    if not LAUNCH_AUTHORIZATION.is_file():
        raise FileNotFoundError(
            "committed launch authorization is absent; freeze it before staging"
        )
    value = json.loads(LAUNCH_AUTHORIZATION.read_text(encoding="utf-8"))
    expected_command = [
        "python3",
        "scripts/execute_observable_cocycle_plan.py",
        "--bundle",
        configured_path('data:row/observable-cocycle/observable-full-state-cocycle-confirmation'),
        "--all",
    ]
    if (
        not isinstance(value, dict)
        or value.get("schema_version")
        != "pldr-observable-cocycle-launch-authorization-v1"
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("authorized") is not True
        or value.get("execution_command") != expected_command
        or value.get("source_code_commit") == head
        or not isinstance(value.get("source_code_commit"), str)
        or len(value["source_code_commit"]) != 40
    ):
        raise ValueError("launch authorization is incomplete or retrospective")
    ancestor = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            value["source_code_commit"],
            head,
        ],
        cwd=ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("authorized code commit is not an ancestor of staged HEAD")
    return value


def _checkpoint_path(input_bundle: Path, name: str, step: int) -> Path:
    filename = "ckpt_final.pt" if step == 65_536 else f"ckpt_{step}.pt"
    return input_bundle / "runs" / name / filename


def _manifest(root: Path) -> bytes:
    excluded = {"MANIFEST.sha256"}
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name not in excluded
    )
    return (
        "\n".join(
            f"{sha256_path(path)}  {path.relative_to(root).as_posix()}"
            for path in paths
        )
        + "\n"
    ).encode("ascii")


def _safe_relative(value: str) -> Path:
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"unsafe manifest path: {value}")
    return Path(*candidate.parts)


def stage(output: Path, input_bundle: Path, tokens: Path) -> None:
    """Create the static bundle while refusing overwrite and uncommitted source."""

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty bundle: {output}")
    if not tokens.is_file() or not input_bundle.is_dir():
        raise FileNotFoundError("token archive or checkpoint bundle is absent")
    head = _git_clean_head()
    authorization = _validated_launch_authorization(head)
    output.mkdir(parents=True, exist_ok=True)
    protocol = output / "protocol"
    protocol.mkdir()
    for relative, payload in generated_files().items():
        destination = protocol / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    (protocol / "launch-authorization.json").write_bytes(
        canonical(authorization)
    )
    registry_source = input_bundle / "protocol" / "registry.json"
    shutil.copy2(registry_source, protocol / "registry.json")
    orders = output / "orders"
    orders.mkdir()
    for trajectory in TRAJECTORIES:
        source = input_bundle / "orders" / str(trajectory["order"])
        shutil.copy2(source, orders / source.name)

    source_paths = source_import_closure()
    source_files = {}
    for path in source_paths:
        relative = path.relative_to(ROOT)
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        source_files[relative.as_posix()] = sha256_path(path)
    (protocol / "source_binding.json").write_bytes(
        canonical(
            {
                "schema_version": "pldr-observable-cocycle-source-binding-v1",
                "git_commit": head,
                "source_files": source_files,
            }
        )
    )

    checkpoints: dict[str, dict[str, dict[str, str]]] = {}
    for trajectory in TRAJECTORIES:
        label = str(trajectory["label"])
        checkpoints[label] = {}
        for step in SOURCE_STEPS:
            path = _checkpoint_path(input_bundle, str(trajectory["name"]), int(step))
            if not path.is_file():
                raise FileNotFoundError(f"checkpoint is absent: {path}")
            checkpoints[label][str(step)] = {
                "path": str(path.resolve()),
                "sha256": sha256_path(path),
            }
    input_binding = {
        "schema_version": "pldr-observable-cocycle-input-binding-v1",
        "tokens_path": str(tokens.resolve()),
        "tokens_sha256": sha256_path(tokens),
        "registry_sha256": sha256_path(protocol / "registry.json"),
        "launch_authorization_sha256": sha256_path(
            protocol / "launch-authorization.json"
        ),
        "orders": {
            str(row["label"]): {
                "path": f"orders/{row['order']}",
                "sha256": sha256_path(orders / str(row["order"])),
            }
            for row in TRAJECTORIES
        },
        "checkpoints": checkpoints,
    }
    (protocol / "input_binding.json").write_bytes(canonical(input_binding))
    (protocol / "launch_plan.json").write_bytes(canonical(launch_plan()))
    for directory in ("records", "reports", "runtime", "incidents"):
        (output / directory).mkdir()
    (output / "README.md").write_text(
        "# Observable full-state row-map cocycle confirmation\n\n"
        "The static protocol, source snapshot, checkpoint hashes, data order hashes, "
        "and complete outcome-independent launch graph are bound before execution.\n",
        encoding="utf-8",
    )
    (output / "MANIFEST.sha256").write_bytes(_manifest(output))


def check(output: Path, input_bundle: Path, tokens: Path) -> None:
    """Verify the immutable static files without rejecting later run outputs."""

    for relative, payload in generated_files().items():
        target = output / "protocol" / relative
        if not target.is_file() or target.read_bytes() != payload:
            raise ValueError(f"staged protocol drifted: {relative}")
    if json.loads((output / "protocol" / "launch_plan.json").read_text()) != launch_plan():
        raise ValueError("staged launch plan drifted")
    binding = json.loads((output / "protocol" / "input_binding.json").read_text())
    if sha256_path(tokens) != binding["tokens_sha256"]:
        raise ValueError("token archive digest drifted")
    staged_authorization = output / "protocol" / "launch-authorization.json"
    if (
        staged_authorization.read_bytes()
        != canonical(json.loads(LAUNCH_AUTHORIZATION.read_text(encoding="utf-8")))
        or sha256_path(staged_authorization)
        != binding["launch_authorization_sha256"]
    ):
        raise ValueError("staged launch authorization drifted")
    for trajectory in TRAJECTORIES:
        label = str(trajectory["label"])
        order = output / binding["orders"][label]["path"]
        if sha256_path(order) != binding["orders"][label]["sha256"]:
            raise ValueError(f"data order digest drifted: {label}")
        for step, record in binding["checkpoints"][label].items():
            path = Path(record["path"])
            expected = _checkpoint_path(input_bundle, str(trajectory["name"]), int(step))
            if path != expected.resolve() or sha256_path(path) != record["sha256"]:
                raise ValueError(f"checkpoint binding drifted: {label}/{step}")
    source_binding = json.loads(
        (output / "protocol" / "source_binding.json").read_text()
    )
    source_root = output / "source"
    actual = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_file()
    }
    if actual != set(source_binding["source_files"]):
        raise ValueError("source snapshot membership drifted")
    for relative, digest in source_binding["source_files"].items():
        if sha256_path(source_root / _safe_relative(relative)) != digest:
            raise ValueError(f"source snapshot digest drifted: {relative}")
    for line in (output / "MANIFEST.sha256").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        if sha256_path(output / _safe_relative(relative)) != digest:
            raise ValueError(f"static manifest drifted: {relative}")
    print("observable-cocycle staging: verified")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--input-bundle", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    input_bundle = arguments.input_bundle.resolve()
    tokens = arguments.tokens.resolve()
    if arguments.check:
        check(output, input_bundle, tokens)
    else:
        stage(output, input_bundle, tokens)
        print(f"observable-cocycle staging: created {output}")


if __name__ == "__main__":
    main()
