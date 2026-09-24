#!/usr/bin/env python3
"""Execute the digest-sealed Rev41 plan with dependency and resource gates.

Archival execution interface: no validated interruption-safe cumulative campaign
budget guarantee. See the combined docs/RESOURCE_EXECUTION.md and
provenance/campaign-accounting.json for the inspected accounting boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from confirm.confirmation_artifacts import digest_object  # noqa: E402
from confirm.contrast_energy_specs import (  # noqa: E402
    CAMPAIGN_ID,
    RESOURCE_BUDGET,
)
from stage_contrast_energy_confirmation import PLAN_SCHEMA  # noqa: E402


LOG_SCHEMA = "pldr-contrast-energy-node-log-v1"
RESOURCE_LOCK_SCHEMA = "pldr-contrast-energy-resource-lock-v1"


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_plan(path: str | Path) -> tuple[Path, dict[str, Any]]:
    plan_path = Path(path).resolve()
    value = json.loads(plan_path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "campaign_id", "status", "devices",
        "source_only_prediction", "successor_dependency_rule",
        "construction_lock_precedes_heldout_training",
        "required_taylor_models", "nodes", "plan_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("execution plan has missing or unknown fields")
    unsigned = dict(value)
    recorded = unsigned.pop("plan_sha256")
    if (
        value["schema_version"] != PLAN_SCHEMA
        or value["campaign_id"] != CAMPAIGN_ID
        or recorded != digest_object(unsigned)
    ):
        raise ValueError("execution plan identity or digest does not replay")
    nodes = value["nodes"]
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("execution plan has no nodes")
    identifiers = [node.get("id") for node in nodes]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("execution plan repeats a node id")
    known = set(identifiers)
    seen = set()
    for node in nodes:
        if set(node) not in (
            {"id", "stage", "role", "depends_on", "argv", "outputs"},
            {"id", "stage", "role", "depends_on", "argv", "outputs", "device"},
        ):
            raise ValueError("execution-plan node fields are invalid")
        if (
            not isinstance(node["argv"], list) or len(node["argv"]) < 2
            or any(not isinstance(item, str) or not item for item in node["argv"])
            or set(node["depends_on"]) - known
        ):
            raise ValueError("node argv or topological dependencies are invalid")
        script = Path(node["argv"][1]).resolve()
        if (
            Path(node["argv"][0]).resolve() != Path(sys.executable).resolve()
            or not script.is_file() or not script.is_relative_to(ROOT)
        ):
            raise ValueError("node argv does not name a repository Python script")
        outputs = [Path(item).resolve() for item in node["outputs"]]
        if not outputs or any(
            not output.is_relative_to(plan_path.parents[1]) for output in outputs
        ):
            raise ValueError("node output leaves the staged campaign root")
        seen.add(node["id"])
    if seen != known:
        raise AssertionError("execution-plan validation lost a node")
    return plan_path, value


def _resolve_argument(value: str) -> str:
    prefix = "{sha256:"
    if not value.startswith(prefix) or not value.endswith("}"):
        return value
    path = Path(value[len(prefix):-1]).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"digest placeholder parent is missing: {path}")
    return sha256_path(path)


def _parent_records(node: dict[str, Any], by_id: dict[str, dict]) -> list[dict]:
    records = []
    for identifier in node["depends_on"]:
        for raw in by_id[identifier]["outputs"]:
            path = Path(raw).resolve()
            if not path.exists():
                raise FileNotFoundError(
                    f"dependency {identifier} output is missing: {path}")
            if path.is_file():
                records.append({
                    "node": identifier, "path": str(path),
                    "sha256": sha256_path(path), "bytes": path.stat().st_size,
                })
            elif path.is_dir():
                records.append({
                    "node": identifier, "path": str(path),
                    "sha256": None, "bytes": None,
                })
            else:
                raise ValueError("dependency output is neither file nor directory")
    return records


def _depends_on(
    node: dict[str, Any],
    target: str,
    by_id: dict[str, dict],
) -> bool:
    pending = list(node["depends_on"])
    seen: set[str] = set()
    while pending:
        identifier = pending.pop()
        if identifier == target:
            return True
        if identifier in seen:
            continue
        seen.add(identifier)
        pending.extend(by_id[identifier]["depends_on"])
    return False


def _resource_cap(
    node: dict[str, Any],
    by_id: dict[str, dict],
    campaign_root: Path,
) -> tuple[int | None, str, str | None]:
    device = str(node.get("device", "cpu"))
    if not device.startswith("cuda:"):
        return None, "not_applicable", None
    provisional = int(
        RESOURCE_BUDGET["provisional_peak_gib_each"] * 1024)
    if not _depends_on(node, "stage-b-resource-lock", by_id):
        return provisional, "provisional_stage_b", None

    path = campaign_root / "protocol" / "stage-b-resource-lock.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"post-Stage-B node lacks the measured resource lock: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(value) if isinstance(value, dict) else {}
    recorded = unsigned.pop("resource_lock_sha256", None)
    if (
        value.get("schema_version") != RESOURCE_LOCK_SCHEMA
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("status") != "measured_caps_sealed"
        or value.get("passed") is not True
        or value.get("provisional_peak_mib_each") != provisional
        or value.get("enlargement")
        != RESOURCE_BUDGET["stage_b_enlargement"]
        or recorded != digest_object(unsigned)
    ):
        raise ValueError("Stage B resource lock identity or digest is invalid")
    caps = value.get("sealed_peak_cap_mib")
    if (
        not isinstance(caps, dict)
        or set(caps) != set(RESOURCE_BUDGET["devices"])
    ):
        raise ValueError("Stage B resource lock has an invalid device set")
    cap = caps.get(device)
    if (
        isinstance(cap, bool)
        or not isinstance(cap, int)
        or cap <= 0
        or cap > provisional
    ):
        raise ValueError(f"Stage B resource lock has no valid cap for {device}")
    return cap, "measured_stage_b", recorded


def _validate_taylor_input(node: dict[str, Any]) -> None:
    if not node["id"].startswith("validated-jets-"):
        return
    argv = node["argv"]
    taylor = Path(argv[argv.index("--taylor-model") + 1]).resolve()
    source = Path(argv[argv.index("--source") + 1]).resolve()
    if not taylor.is_file() or not source.is_file():
        raise FileNotFoundError("validated-jet node lacks its source or Taylor model")
    with np.load(taylor, allow_pickle=False) as record:
        schema = np.asarray(record["schema_version"])
        bound = np.asarray(record["source_sha256"])
        successor = np.asarray(record["successor_values_present"])
        if (
            schema.shape != ()
            or str(schema.item()) != "pldr-contrast-energy-taylor-model-v1"
            or bound.shape != () or str(bound.item()) != sha256_path(source)
            or successor.shape != () or bool(successor.item())
        ):
            raise ValueError("Taylor model schema or source binding is invalid")


def _gpu_memory_mib(pid: int) -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return 0
    maximum = 0
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0].isdigit() and int(fields[0]) == pid:
            maximum = max(maximum, int(fields[1]))
    return maximum


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n", encoding="utf-8")
    temporary.replace(path)


def execute_node(
    node: dict[str, Any], by_id: dict[str, dict], campaign_root: Path,
    *, timeout_seconds: int, resume: bool,
) -> dict[str, Any]:
    log_path = campaign_root / "logs" / f"{node['id']}.json"
    outputs = [Path(item).resolve() for item in node["outputs"]]
    resolved_argv = [_resolve_argument(item) for item in node["argv"]]
    parents = _parent_records(node, by_id)
    gpu_cap_mib, gpu_cap_source, resource_lock_sha256 = _resource_cap(
        node, by_id, campaign_root)
    if resume and log_path.is_file() and all(path.exists() for path in outputs):
        old = json.loads(log_path.read_text(encoding="utf-8"))
        old_resources = old.get("resources", {})
        if (
            old.get("schema_version") == LOG_SCHEMA
            and old.get("exit_code") == 0
            and old.get("resolved_argv") == resolved_argv
            and old.get("parent_records") == parents
            and old_resources.get("gpu_memory_cap_mib") == gpu_cap_mib
            and old_resources.get("gpu_memory_cap_source") == gpu_cap_source
            and old_resources.get("resource_lock_sha256")
            == resource_lock_sha256
        ):
            return old
    _validate_taylor_input(node)
    stdout_path = campaign_root / "logs" / f"{node['id']}.stdout"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    peak_gpu_mib = 0
    with stdout_path.open("wb") as stream:
        process = subprocess.Popen(
            resolved_argv, cwd=ROOT, stdout=stream,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed > timeout_seconds:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise TimeoutError(f"node {node['id']} exceeded its wall timeout")
            if str(node.get("device", "")).startswith("cuda:"):
                peak_gpu_mib = max(peak_gpu_mib, _gpu_memory_mib(process.pid))
                if gpu_cap_mib is None:
                    raise AssertionError("CUDA node lacks a resource cap")
                if peak_gpu_mib > gpu_cap_mib:
                    process.terminate()
                    raise RuntimeError(
                        f"node {node['id']} exceeded its {gpu_cap_source} "
                        f"GPU cap of {gpu_cap_mib} MiB")
            time.sleep(0.5)
        exit_code = int(process.returncode)
    wall = float(time.monotonic() - started)
    missing = [str(path) for path in outputs if not path.exists()]
    output_records = []
    for path in outputs:
        if path.is_file():
            output_records.append({
                "path": str(path), "sha256": sha256_path(path),
                "bytes": path.stat().st_size,
            })
        elif path.is_dir():
            output_records.append({
                "path": str(path), "sha256": None, "bytes": None,
            })
    host = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    if sys.platform != "darwin":
        host *= 1024
    record = {
        "schema_version": LOG_SCHEMA,
        "node_id": node["id"],
        "stage": node["stage"],
        "role": node["role"],
        "device": node.get("device", "cpu"),
        "resolved_argv": resolved_argv,
        "parent_records": parents,
        "exit_code": exit_code,
        "missing_outputs": missing,
        "output_records": output_records,
        "stdout_path": str(stdout_path),
        "stdout_sha256": sha256_path(stdout_path),
        "resources": {
            "wall_seconds": wall,
            "peak_gpu_memory_mib": peak_gpu_mib,
            "peak_host_rss_bytes": host,
            "gpu_memory_cap_mib": gpu_cap_mib,
            "gpu_memory_cap_source": gpu_cap_source,
            "resource_lock_sha256": resource_lock_sha256,
        },
    }
    record["record_sha256"] = digest_object(record)
    _write_json(log_path, record)
    if exit_code != 0 or missing:
        raise RuntimeError(
            f"node {node['id']} failed with exit {exit_code}; missing={missing}")
    return record


def dependency_closure(targets: set[str], by_id: dict[str, dict]) -> set[str]:
    selected = set()
    pending = list(targets)
    while pending:
        identifier = pending.pop()
        if identifier in selected:
            continue
        if identifier not in by_id:
            raise ValueError(f"unknown requested node {identifier}")
        selected.add(identifier)
        pending.extend(by_id[identifier]["depends_on"])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--node", action="append", default=[])
    parser.add_argument("--stage", action="append", default=[])
    parser.add_argument("--timeout-seconds", type=int, default=21_600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    plan_path, plan = load_plan(arguments.plan)
    if arguments.check:
        print(f"contrast-energy executor: verified {len(plan['nodes'])} nodes")
        return
    if arguments.timeout_seconds < 1:
        raise ValueError("timeout must be positive")
    by_id = {node["id"]: node for node in plan["nodes"]}
    targets = set(arguments.node)
    if arguments.stage:
        targets.update(
            node["id"] for node in plan["nodes"]
            if node["stage"] in set(arguments.stage))
    selected = dependency_closure(targets, by_id) if targets else set(by_id)
    campaign_root = plan_path.parents[1]
    records = []
    completed = set()
    pending = set(selected)
    while pending:
        progress = False
        for node in plan["nodes"]:
            identifier = node["id"]
            if identifier not in pending:
                continue
            if not set(node["depends_on"]).issubset(completed):
                continue
            records.append(execute_node(
                node, by_id, campaign_root,
                timeout_seconds=arguments.timeout_seconds,
                resume=arguments.resume,
            ))
            pending.remove(identifier)
            completed.add(identifier)
            progress = True
        if not progress:
            raise RuntimeError(
                "selected execution graph has a dependency cycle")
    total_wall = sum(record["resources"]["wall_seconds"] for record in records)
    total_bytes = sum(
        item["bytes"] or 0 for record in records
        for item in record["output_records"])
    if total_wall > RESOURCE_BUDGET["wall_hours_upper"] * 3600:
        raise RuntimeError("selected execution exceeded the campaign wall envelope")
    if total_bytes > RESOURCE_BUDGET["aggregate_output_gib_cap"] * 2 ** 30:
        raise RuntimeError("selected execution exceeded the persistent-output cap")
    summary = {
        "schema_version": "pldr-contrast-energy-execution-summary-v1",
        "plan_sha256": plan["plan_sha256"],
        "node_count": len(records),
        "total_wall_seconds": total_wall,
        "record_sha256": [record["record_sha256"] for record in records],
    }
    _write_json(campaign_root / "logs" / "execution-summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
