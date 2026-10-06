#!/usr/bin/env python3
"""Stage the immutable finite-increment observable-balance campaign."""

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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from confirm.observable_balance_specs import (  # noqa: E402
    CAMPAIGN_ID,
    SOURCE_STEPS,
    TRAJECTORIES,
    campaign_design,
)
from scripts.gen_observable_balance_protocols import (  # noqa: E402
    canonical,
    digest_object,
    generated_files,
    record_inventory,
)


ENTRYPOINTS = (
    "experiments/confirm/observable_balance_live.py",
    "experiments/analysis/analyze_observable_balance_confirmation.py",
    "experiments/analysis/summarize_observable_balance_manuscript.py",
    "scripts/execute_observable_balance_plan.py",
    "scripts/gen_observable_balance_protocols.py",
    "scripts/seal_observable_balance_provenance.py",
    "scripts/stage_observable_balance_confirmation.py",
)

LAUNCH_AUTHORIZATION = (
    ROOT / "experiments" / "protocols" / "observable_balance_confirmation"
    / "launch-authorization.json"
)


def default_paths(binding_root: Path) -> dict[str, Path]:
    """Derive all external campaign paths only from an explicit binding root."""
    return {
        "input_bundle": (
            binding_root
            / "experiment-data/campaigns/mixed-collapse"
            / "mixed-row-map-collapse-confirmation"
        ),
        "output": (
            binding_root
            / "experiment-data/campaigns/observable-balance"
            / "finite-increment-observable-balance"
        ),
        "tokens": (
            binding_root
            / "experiment-data/shared/datasets"
            / "refinedweb-538m-prefix-locked/refinedweb_tokens.npy"
        ),
    }


def bound_path(path: Path, binding_root: Path, label: str) -> Path:
    """Resolve one external path and reject escape from its binding root."""
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(binding_root):
        raise ValueError(f"{label} escapes binding root: {resolved}")
    return resolved


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


def _science_device(label: str, layer: int, step: int) -> str:
    """Return the frozen accelerator assignment for one scientific node."""

    if label == "B":
        return "cuda:0"
    if label == "C":
        return "cuda:1"
    step_index = SOURCE_STEPS.index(step)
    return f"cuda:{(3 * layer + step_index) % 2}"


def _common_live(
    command: str,
    trajectory: dict[str, Any],
    step: int,
    layer: int,
    output: str,
    device: str,
) -> list[str]:
    label = str(trajectory["label"])
    return [
        "{python}",
        "{bundle}/source/experiments/confirm/observable_balance_live.py",
        command,
        "--checkpoint",
        f"{{checkpoint:{label}:{step}}}",
        "--tokens",
        "{tokens}",
        "--order",
        f"{{bundle}}/orders/{trajectory['order']}",
        "--measurement-registry",
        "{bundle}/protocol/measurement-registry.json",
        "--device",
        device,
        "--layer",
        str(layer),
        "--output",
        f"{{bundle}}/{output}",
    ]


def _node(
    node_id: str,
    role: str,
    device: str,
    depends_on: list[str],
    command: list[str],
    expected_outputs: list[str],
) -> dict[str, Any]:
    return {
        "id": node_id,
        "stage": role,
        "role": role,
        "device": device,
        "depends_on": depends_on,
        "command": command,
        "expected_outputs": expected_outputs,
    }


def launch_plan() -> dict[str, Any]:
    """Return the complete outcome-independent execution DAG."""

    inventory = record_inventory()
    trajectories = {str(row["label"]): row for row in TRAJECTORIES}
    nodes: list[dict[str, Any]] = []

    qualification_id = "qualification-A-L0-S4096"
    qualification_path = str(inventory["qualification"])
    qualification_command = _common_live(
        "qualify",
        trajectories["A"],
        4_096,
        0,
        qualification_path,
        "cuda:0",
    )
    qualification_command.append("--require-pass")
    nodes.append(
        _node(
            qualification_id,
            "qualification",
            "cuda:0",
            [],
            qualification_command,
            [qualification_path],
        )
    )

    construction_ids: list[str] = []
    for layer in range(3):
        for step in SOURCE_STEPS:
            node_id = f"construction-A-L{layer}-S{step}"
            path = f"records/construction-A-L{layer}-S{step}.npz"
            device = _science_device("A", layer, step)
            construction_ids.append(node_id)
            nodes.append(
                _node(
                    node_id,
                    "construction",
                    device,
                    [qualification_id],
                    _common_live(
                        "construct", trajectories["A"], step, layer, path, device
                    ),
                    [path],
                )
            )

    lock_id = "construction-lock"
    lock_path = str(inventory["lock"])
    nodes.append(
        _node(
            lock_id,
            "lock",
            "cpu",
            construction_ids,
            [
                "{python}",
                (
                    "{bundle}/source/experiments/analysis/"
                    "analyze_observable_balance_confirmation.py"
                ),
                "lock",
                "--bundle",
                "{bundle}",
                "--output",
                f"{{bundle}}/{lock_path}",
            ],
            [lock_path],
        )
    )

    prediction_ids: dict[tuple[str, int, int], str] = {}
    for trajectory in TRAJECTORIES:
        if trajectory["role"] != "heldout":
            continue
        label = str(trajectory["label"])
        for layer in range(3):
            for step in SOURCE_STEPS:
                key = (label, layer, step)
                node_id = f"prediction-{label}-L{layer}-S{step}"
                path = f"records/prediction-{label}-L{layer}-S{step}.npz"
                device = _science_device(label, layer, step)
                command = _common_live(
                    "predict", trajectory, step, layer, path, device
                )
                command.extend(["--lock", f"{{bundle}}/{lock_path}"])
                prediction_ids[key] = node_id
                nodes.append(
                    _node(
                        node_id,
                        "prediction",
                        device,
                        [lock_id],
                        command,
                        [path],
                    )
                )

    validation_ids: list[str] = []
    for trajectory in TRAJECTORIES:
        if trajectory["role"] != "heldout":
            continue
        label = str(trajectory["label"])
        for layer in range(3):
            for step in SOURCE_STEPS:
                key = (label, layer, step)
                node_id = f"validation-{label}-L{layer}-S{step}"
                path = f"records/validation-{label}-L{layer}-S{step}.npz"
                prediction_path = f"records/prediction-{label}-L{layer}-S{step}.npz"
                device = _science_device(label, layer, step)
                command = _common_live(
                    "validate", trajectory, step, layer, path, device
                )
                command.extend(
                    ["--prediction", f"{{bundle}}/{prediction_path}"]
                )
                validation_ids.append(node_id)
                nodes.append(
                    _node(
                        node_id,
                        "validation",
                        device,
                        [prediction_ids[key]],
                        command,
                        [path],
                    )
                )

    stratum_ids: list[str] = []
    for trajectory in TRAJECTORIES:
        label = str(trajectory["label"])
        for layer in range(3):
            node_id = f"stratum-control-{label}-L{layer}-S4096"
            path = f"records/stratum-control-{label}-L{layer}-S4096.npz"
            device = (
                f"cuda:{layer % 2}"
                if label == "A"
                else ("cuda:0" if label == "B" else "cuda:1")
            )
            stratum_ids.append(node_id)
            nodes.append(
                _node(
                    node_id,
                    "stratum-control",
                    device,
                    [qualification_id],
                    _common_live(
                        "stratum-control",
                        trajectory,
                        4_096,
                        layer,
                        path,
                        device,
                    ),
                    [path],
                )
            )

    nodes.append(
        _node(
            "analysis",
            "analysis",
            "cpu",
            [qualification_id, lock_id, *validation_ids, *stratum_ids],
            [
                "{python}",
                (
                    "{bundle}/source/experiments/analysis/"
                    "analyze_observable_balance_confirmation.py"
                ),
                "final",
                "--bundle",
                "{bundle}",
            ],
            list(inventory["analysis_outputs"]),
        )
    )
    expected = int(inventory["expected_counts"]["total_plan_nodes"])
    if len(nodes) != expected:
        raise AssertionError(f"launch plan has {len(nodes)} nodes, expected {expected}")
    return {
        "schema_version": "pldr-observable-balance-launch-plan-v1",
        "campaign_id": CAMPAIGN_ID,
        "design_sha256": digest_object(campaign_design()),
        "paths_are_bundle_relative": True,
        "scientific_outcomes_control_execution": False,
        "planned_devices_are_immutable": True,
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


def _validated_launch_authorization(head: str, binding_root: Path) -> dict[str, Any]:
    if not LAUNCH_AUTHORIZATION.is_file():
        raise FileNotFoundError(
            "committed launch authorization is absent; freeze it before staging"
        )
    value = json.loads(LAUNCH_AUTHORIZATION.read_text(encoding="utf-8"))
    expected_command = [
        "python3",
        "scripts/execute_observable_balance_plan.py",
        "--bundle",
        str(default_paths(binding_root)["output"]),
        "--all",
    ]
    if (
        not isinstance(value, dict)
        or value.get("schema_version")
        != "pldr-observable-balance-launch-authorization-v1"
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("design_sha256") != digest_object(campaign_design())
        or value.get("authorized") is not True
        or value.get("execution_command") != expected_command
        or value.get("source_code_commit") == head
        or not isinstance(value.get("source_code_commit"), str)
        or len(value["source_code_commit"]) != 40
    ):
        raise ValueError("launch authorization is incomplete or retrospective")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", value["source_code_commit"], head],
        cwd=ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("authorized code commit is not an ancestor of staged HEAD")
    launch_notes = ROOT / "experiments" / "confirm" / "LAUNCH_NOTES.md"
    if CAMPAIGN_ID not in launch_notes.read_text(encoding="utf-8"):
        raise ValueError("campaign is absent from the committed launch notes")
    return value


def _checkpoint_path(input_bundle: Path, name: str, step: int) -> Path:
    filename = "ckpt_final.pt" if step == 65_536 else f"ckpt_{step}.pt"
    return input_bundle / "runs" / name / filename


def _manifest(root: Path) -> bytes:
    excluded_names = {"MANIFEST.sha256"}
    mutable_roots = {"records", "reports", "runtime", "incidents"}
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in excluded_names
        and path.relative_to(root).parts[0] not in mutable_roots
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


def stage(
    output: Path, input_bundle: Path, tokens: Path, binding_root: Path,
) -> None:
    """Create the static bundle while refusing overwrite and uncommitted source."""

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty bundle: {output}")
    if not tokens.is_file() or not input_bundle.is_dir():
        raise FileNotFoundError("token archive or checkpoint bundle is absent")
    head = _git_clean_head()
    authorization = _validated_launch_authorization(head, binding_root)
    output.mkdir(parents=True, exist_ok=True)
    protocol = output / "protocol"
    protocol.mkdir()
    for relative, payload in generated_files().items():
        destination = protocol / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    (protocol / "launch-authorization.json").write_bytes(canonical(authorization))

    measurement_source = input_bundle / "protocol" / "registry.json"
    if not measurement_source.is_file():
        raise FileNotFoundError("measurement registry is absent from the input bundle")
    shutil.copy2(measurement_source, protocol / "measurement-registry.json")
    measurement_digest = sha256_path(protocol / "measurement-registry.json")

    orders = output / "orders"
    orders.mkdir()
    for trajectory in TRAJECTORIES:
        source = input_bundle / "orders" / str(trajectory["order"])
        if not source.is_file():
            raise FileNotFoundError(f"registered data order is absent: {source}")
        shutil.copy2(source, orders / source.name)

    source_paths = source_import_closure()
    source_files: dict[str, str] = {}
    for path in source_paths:
        relative = path.relative_to(ROOT)
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        source_files[relative.as_posix()] = sha256_path(path)
    source_binding = {
        "schema_version": "pldr-observable-balance-source-binding-v1",
        "science_source_commit": authorization["source_code_commit"],
        "staging_commit": head,
        "source_files": source_files,
    }
    (protocol / "source-binding.json").write_bytes(canonical(source_binding))

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

    plan = launch_plan()
    (protocol / "launch-plan.json").write_bytes(canonical(plan))
    authorization_digest = sha256_path(protocol / "launch-authorization.json")
    registry_unsigned = {
        "schema_version": "pldr-observable-balance-campaign-registry-v1",
        "campaign_id": CAMPAIGN_ID,
        "design_sha256": digest_object(campaign_design()),
        "measurement_registry_sha256": measurement_digest,
        "science_source_commit": authorization["source_code_commit"],
        "staging_commit": head,
        "source_binding_sha256": sha256_path(protocol / "source-binding.json"),
        "launch_authorization_sha256": authorization_digest,
        "launch_plan_sha256": sha256_path(protocol / "launch-plan.json"),
    }
    campaign_registry = {
        **registry_unsigned,
        "registry_sha256": digest_object(registry_unsigned),
    }
    (protocol / "campaign-registry.json").write_bytes(canonical(campaign_registry))

    input_binding = {
        "schema_version": "pldr-observable-balance-input-binding-v1",
        "campaign_id": CAMPAIGN_ID,
        "input_bundle_path": str(input_bundle.resolve()),
        "tokens_path": str(tokens.resolve()),
        "tokens_sha256": sha256_path(tokens),
        "measurement_registry_sha256": measurement_digest,
        "campaign_registry_sha256": sha256_path(
            protocol / "campaign-registry.json"
        ),
        "launch_authorization_sha256": authorization_digest,
        "orders": {
            str(row["label"]): {
                "path": f"orders/{row['order']}",
                "sha256": sha256_path(orders / str(row["order"])),
            }
            for row in TRAJECTORIES
        },
        "checkpoints": checkpoints,
    }
    (protocol / "input-binding.json").write_bytes(canonical(input_binding))
    for directory in ("records", "reports", "runtime", "incidents"):
        (output / directory).mkdir()
    (output / "incidents" / "index.json").write_bytes(
        canonical(
            {
                "schema_version": "pldr-observable-balance-incident-index-v1",
                "campaign_id": CAMPAIGN_ID,
                "incidents": [],
            }
        )
    )
    (output / "README.md").write_text(
        "# Finite-increment observable-balance confirmation\n\n"
        "Static protocol, independent measurement and campaign registries, source "
        "snapshot, checkpoint hashes, data-order hashes, and the full 57-node "
        "outcome-independent graph were bound before execution. Runtime and "
        "incident membership is sealed after all nodes complete.\n",
        encoding="utf-8",
    )
    (output / "MANIFEST.sha256").write_bytes(_manifest(output))


def check(output: Path, input_bundle: Path, tokens: Path) -> None:
    """Verify a staged bundle using only its frozen bindings."""

    manifest_lines = (output / "MANIFEST.sha256").read_text().splitlines()
    manifest_paths: set[str] = set()
    for line in manifest_lines:
        digest, relative = line.split("  ", 1)
        path = output / _safe_relative(relative)
        if sha256_path(path) != digest:
            raise ValueError(f"static manifest drifted: {relative}")
        manifest_paths.add(relative)
    mutable_roots = {"records", "reports", "runtime", "incidents"}
    actual_static = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
        and path.name not in {"MANIFEST.sha256", "RUNTIME.sha256"}
        and path.relative_to(output).parts[0] not in mutable_roots
    }
    if actual_static != manifest_paths:
        raise ValueError("static bundle membership drifted")

    protocol = output / "protocol"
    binding = json.loads((protocol / "input-binding.json").read_text())
    if binding.get("campaign_id") != CAMPAIGN_ID:
        raise ValueError("input binding campaign identity drifted")
    if Path(binding["input_bundle_path"]) != input_bundle.resolve():
        raise ValueError("input bundle path drifted")
    if Path(binding["tokens_path"]) != tokens.resolve() or sha256_path(tokens) != binding[
        "tokens_sha256"
    ]:
        raise ValueError("token archive binding drifted")
    if sha256_path(protocol / "measurement-registry.json") != binding[
        "measurement_registry_sha256"
    ]:
        raise ValueError("measurement registry digest drifted")
    if sha256_path(protocol / "campaign-registry.json") != binding[
        "campaign_registry_sha256"
    ]:
        raise ValueError("campaign registry file digest drifted")
    if sha256_path(protocol / "launch-authorization.json") != binding[
        "launch_authorization_sha256"
    ]:
        raise ValueError("launch authorization digest drifted")

    campaign_registry = json.loads(
        (protocol / "campaign-registry.json").read_text(encoding="utf-8")
    )
    registry_unsigned = dict(campaign_registry)
    registered_digest = registry_unsigned.pop("registry_sha256", None)
    if registered_digest != digest_object(registry_unsigned):
        raise ValueError("campaign registry self-digest drifted")
    plan = json.loads((protocol / "launch-plan.json").read_text(encoding="utf-8"))
    if (
        plan.get("campaign_id") != CAMPAIGN_ID
        or campaign_registry["launch_plan_sha256"]
        != sha256_path(protocol / "launch-plan.json")
        or plan.get("design_sha256") != campaign_registry["design_sha256"]
        or len(plan.get("nodes", [])) != 57
    ):
        raise ValueError("launch plan and campaign registry disagree")

    for trajectory in TRAJECTORIES:
        label = str(trajectory["label"])
        order = output / _safe_relative(binding["orders"][label]["path"])
        if sha256_path(order) != binding["orders"][label]["sha256"]:
            raise ValueError(f"data order digest drifted: {label}")
        for step, record in binding["checkpoints"][label].items():
            path = Path(record["path"])
            expected = _checkpoint_path(input_bundle, str(trajectory["name"]), int(step))
            if path != expected.resolve() or sha256_path(path) != record["sha256"]:
                raise ValueError(f"checkpoint binding drifted: {label}/{step}")

    source_binding = json.loads((protocol / "source-binding.json").read_text())
    if campaign_registry["source_binding_sha256"] != sha256_path(
        protocol / "source-binding.json"
    ):
        raise ValueError("source binding digest drifted")
    source_root = output / "source"
    actual_sources = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_file()
    }
    if actual_sources != set(source_binding["source_files"]):
        raise ValueError("source snapshot membership drifted")
    for relative, digest in source_binding["source_files"].items():
        if sha256_path(source_root / _safe_relative(relative)) != digest:
            raise ValueError(f"source snapshot digest drifted: {relative}")
    print("observable-balance staging: verified")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input-bundle", type=Path)
    parser.add_argument("--tokens", type=Path)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    binding_root = arguments.binding_root.resolve(strict=True)
    defaults = default_paths(binding_root)
    input_bundle = bound_path(
        arguments.input_bundle or defaults["input_bundle"],
        binding_root,
        "input bundle",
    )
    output = bound_path(
        arguments.output or defaults["output"], binding_root, "output bundle"
    )
    tokens = bound_path(
        arguments.tokens or defaults["tokens"], binding_root, "token archive"
    )
    if arguments.check:
        check(output, input_bundle, tokens)
    else:
        stage(output, input_bundle, tokens, binding_root)
        print(f"observable-balance staging: created {output}")


if __name__ == "__main__":
    main()
