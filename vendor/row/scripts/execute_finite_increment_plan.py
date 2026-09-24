#!/usr/bin/env python3
"""Execute the finite-increment plan with dependency and resource gates.

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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from confirm.confirmation_artifacts import digest_object, sha256_path  # noqa: E402
from confirm.finite_increment_specs import (  # noqa: E402
    CAMPAIGN_ID,
    PLAN_SCHEMA,
    RESOURCE_BUDGET,
)


LOG_SCHEMA = "pldr-finite-increment-node-log-v1"
FAILURE_SCHEMA = "pldr-finite-increment-failure-v1"
SUMMARY_SCHEMA = "pldr-finite-increment-execution-summary-v1"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _tree_bytes(root: Path) -> int:
    return sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )


def _record_output(path: Path, bundle: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError("campaign outputs may not be symbolic links")
    relative = path.relative_to(bundle).as_posix()
    if path.is_file():
        return {
            "path": relative,
            "kind": "file",
            "bytes": path.stat().st_size,
            "sha256": sha256_path(path),
        }
    if path.is_dir():
        return {
            "path": relative,
            "kind": "directory",
            "bytes": _tree_bytes(path),
            "sha256": None,
        }
    raise FileNotFoundError(f"declared output is missing: {path}")


def load_plan(path: str | Path) -> tuple[Path, dict[str, Any]]:
    plan_path = Path(path).resolve()
    value = json.loads(plan_path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "campaign_id", "status", "source_only_prediction",
        "successor_replays_same_complete_source",
        "training_orders_use_trainer_validator", "nodes", "node_count",
        "theorem_node_bindings", "plan_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("execution plan has missing or unknown fields")
    unsigned = dict(value)
    recorded = unsigned.pop("plan_sha256")
    if (
        value["schema_version"] != PLAN_SCHEMA
        or value["campaign_id"] != CAMPAIGN_ID
        or value["source_only_prediction"] is not True
        or value["successor_replays_same_complete_source"] is not True
        or value["training_orders_use_trainer_validator"] is not True
        or recorded != digest_object(unsigned)
    ):
        raise ValueError("execution plan identity or digest does not replay")
    nodes = value["nodes"]
    if not isinstance(nodes, list) or len(nodes) != value["node_count"]:
        raise ValueError("execution plan node count is invalid")
    identifiers = [node.get("id") for node in nodes]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("execution plan repeats a node identifier")
    known = set(identifiers)
    for node in nodes:
        if set(node) != {
            "id", "stage", "device", "depends_on", "argv", "outputs",
            "predicted_output_bytes",
        }:
            raise ValueError("execution-plan node fields are invalid")
        if (
            not isinstance(node["id"], str)
            or not isinstance(node["argv"], list)
            or len(node["argv"]) < 2
            or any(not isinstance(item, str) for item in node["argv"])
            or set(node["depends_on"]) - known
            or not isinstance(node["predicted_output_bytes"], int)
            or node["predicted_output_bytes"] < 0
        ):
            raise ValueError("execution-plan node structure is invalid")
    return plan_path, value


def _resolve_argv(
    argv: list[str],
    *,
    bundle: Path,
    tokens: Path,
    tokenizer: Path,
    qualification_checkpoint: Path,
) -> list[str]:
    replacements = {
        "{bundle}": str(bundle),
        "{tokens}": str(tokens),
        "{tokenizer}": str(tokenizer),
        "{qualification_checkpoint}": str(qualification_checkpoint),
    }
    resolved = [replacements.get(value, value) for value in argv]
    if resolved[0] != "python3":
        raise ValueError("campaign nodes must use the staged Python interpreter")
    resolved[0] = sys.executable
    if resolved[1] != "-c":
        script = (bundle / resolved[1]).resolve()
        source_root = (bundle / "source").resolve()
        if not script.is_file() or not script.is_relative_to(source_root):
            raise ValueError("campaign node script leaves the source snapshot")
        resolved[1] = str(script)
    return resolved


def _gpu_memory_mib(pid: int) -> int:
    result = subprocess.run(
        [
            "nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    maximum = 0
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0].isdigit() and int(fields[0]) == pid:
            maximum = max(maximum, int(fields[1]))
    return maximum


def _failure(
    bundle: Path,
    node: dict[str, Any],
    *,
    reason: str,
    resolved_argv: list[str],
    exit_code: int | None,
) -> None:
    value: dict[str, Any] = {
        "schema_version": FAILURE_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "node_id": node["id"],
        "stage": node["stage"],
        "device": node["device"],
        "reason": reason,
        "resolved_argv": resolved_argv,
        "exit_code": exit_code,
        "bundle_bytes": _tree_bytes(bundle),
        "recoverable": True,
    }
    value["failure_sha256"] = digest_object(value)
    _write_json(bundle / "reports" / "failures" / f"{node['id']}.json", value)


def _dependency_records(
    node: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
    bundle: Path,
) -> list[dict[str, Any]]:
    records = []
    for parent_id in node["depends_on"]:
        log = bundle / "logs" / f"{parent_id}.json"
        if not log.is_file():
            raise FileNotFoundError(f"dependency log is missing: {parent_id}")
        value = json.loads(log.read_text(encoding="utf-8"))
        if value.get("schema_version") != LOG_SCHEMA or value.get("exit_code") != 0:
            raise ValueError(f"dependency did not finish successfully: {parent_id}")
        for output in by_id[parent_id]["outputs"]:
            path = (bundle / output).resolve()
            if not path.is_relative_to(bundle) or not path.exists():
                raise FileNotFoundError(
                    f"dependency output is missing: {parent_id}: {output}"
                )
        records.append({
            "node_id": parent_id,
            "log_sha256": sha256_path(log),
            "record_sha256": value["record_sha256"],
        })
    return records


def execute_node(
    node: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
    *,
    bundle: Path,
    tokens: Path,
    tokenizer: Path,
    qualification_checkpoint: Path,
    timeout_seconds: int,
    resume: bool,
) -> dict[str, Any]:
    log_path = bundle / "logs" / f"{node['id']}.json"
    outputs = [(bundle / output).resolve() for output in node["outputs"]]
    if any(not path.is_relative_to(bundle) for path in outputs):
        raise ValueError("a declared output leaves the campaign bundle")
    if resume and log_path.is_file() and all(path.exists() for path in outputs):
        old = json.loads(log_path.read_text(encoding="utf-8"))
        if old.get("schema_version") == LOG_SCHEMA and old.get("exit_code") == 0:
            return old
    parents = _dependency_records(node, by_id, bundle)
    argv = _resolve_argv(
        node["argv"],
        bundle=bundle,
        tokens=tokens,
        tokenizer=tokenizer,
        qualification_checkpoint=qualification_checkpoint,
    )
    stdout_path = bundle / "logs" / f"{node['id']}.stdout"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    peak_gpu_mib = 0
    exit_code: int | None = None
    reason = None
    process: subprocess.Popen[str] | None = None
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout:
            process = subprocess.Popen(
                argv,
                cwd=bundle,
                stdout=stdout,
                stderr=subprocess.STDOUT,
                text=True,
            )
            next_storage_check = started
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed > timeout_seconds:
                    reason = f"node exceeded {timeout_seconds} seconds"
                    process.terminate()
                    break
                if str(node["device"]).startswith("cuda:"):
                    try:
                        peak_gpu_mib = max(
                            peak_gpu_mib, _gpu_memory_mib(process.pid)
                        )
                    except (FileNotFoundError, subprocess.SubprocessError) as error:
                        reason = f"GPU resource probe failed closed: {error}"
                        process.terminate()
                        break
                    if peak_gpu_mib > int(
                        float(RESOURCE_BUDGET["peak_reserved_gib_each"]) * 1024
                    ):
                        reason = "node exceeded the sealed GPU memory cap"
                        process.terminate()
                        break
                if time.monotonic() >= next_storage_check:
                    if _tree_bytes(bundle) > int(
                        float(RESOURCE_BUDGET["persistent_output_gib_cap"]) * 2 ** 30
                    ):
                        reason = "campaign exceeded the persistent-output cap"
                        process.terminate()
                        break
                    next_storage_check = time.monotonic() + 10.0
                time.sleep(1.0)
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
            exit_code = process.wait()
    except Exception as error:
        reason = f"{type(error).__name__}: {error}"
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
    if reason is None and exit_code != 0:
        reason = f"node exited with status {exit_code}"
    if reason is None:
        missing = [str(path) for path in outputs if not path.exists()]
        if missing:
            reason = f"node omitted declared outputs: {missing}"
    if reason is None and _tree_bytes(bundle) > int(
        float(RESOURCE_BUDGET["persistent_output_gib_cap"]) * 2 ** 30
    ):
        reason = "campaign exceeded the persistent-output cap after the node"
    if reason is not None:
        _failure(
            bundle,
            node,
            reason=reason,
            resolved_argv=argv,
            exit_code=exit_code,
        )
        raise RuntimeError(f"node {node['id']} failed: {reason}")
    output_records = [_record_output(path, bundle) for path in outputs]
    record: dict[str, Any] = {
        "schema_version": LOG_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "node_id": node["id"],
        "stage": node["stage"],
        "device": node["device"],
        "resolved_argv": argv,
        "dependencies": parents,
        "exit_code": int(exit_code),
        "stdout_path": stdout_path.relative_to(bundle).as_posix(),
        "stdout_sha256": sha256_path(stdout_path),
        "outputs": output_records,
        "resources": {
            "wall_seconds": time.monotonic() - started,
            "peak_gpu_memory_mib": peak_gpu_mib,
            "peak_host_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            ) * (1024 if sys.platform != "darwin" else 1),
            "bundle_bytes_after": _tree_bytes(bundle),
        },
    }
    record["record_sha256"] = digest_object(record)
    _write_json(log_path, record)
    return record


def dependency_closure(
    targets: set[str], by_id: dict[str, dict[str, Any]]
) -> set[str]:
    selected = set()
    pending = list(targets)
    while pending:
        identifier = pending.pop()
        if identifier in selected:
            continue
        if identifier not in by_id:
            raise ValueError(f"unknown requested node: {identifier}")
        selected.add(identifier)
        pending.extend(by_id[identifier]["depends_on"])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--qualification-checkpoint", type=Path, required=True)
    parser.add_argument("--node", action="append", default=[])
    parser.add_argument("--stage", action="append", default=[])
    parser.add_argument("--timeout-seconds", type=int, default=21_600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    plan_path, plan = load_plan(arguments.plan)
    bundle = plan_path.parents[1]
    tokens = arguments.tokens.resolve()
    tokenizer = arguments.tokenizer.resolve()
    qualification_checkpoint = arguments.qualification_checkpoint.resolve()
    for path, label in (
        (tokens, "tokens"),
        (tokenizer, "tokenizer"),
        (qualification_checkpoint, "qualification checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
    by_id = {node["id"]: node for node in plan["nodes"]}
    for node in plan["nodes"]:
        _resolve_argv(
            node["argv"],
            bundle=bundle,
            tokens=tokens,
            tokenizer=tokenizer,
            qualification_checkpoint=qualification_checkpoint,
        )
    if arguments.check:
        print(json.dumps({
            "status": "verified",
            "nodes": len(by_id),
            "plan_sha256": plan["plan_sha256"],
        }, sort_keys=True))
        return
    if arguments.timeout_seconds < 1:
        raise ValueError("node timeout must be positive")
    targets = set(arguments.node)
    if arguments.stage:
        stages = set(arguments.stage)
        targets.update(
            node["id"] for node in plan["nodes"] if node["stage"] in stages
        )
    selected = dependency_closure(targets, by_id) if targets else set(by_id)
    pending = set(selected)
    completed = set()
    records = []
    while pending:
        progressed = False
        for node in plan["nodes"]:
            identifier = node["id"]
            if identifier not in pending:
                continue
            selected_dependencies = set(node["depends_on"]) & selected
            if not selected_dependencies.issubset(completed):
                continue
            record = execute_node(
                node,
                by_id,
                bundle=bundle,
                tokens=tokens,
                tokenizer=tokenizer,
                qualification_checkpoint=qualification_checkpoint,
                timeout_seconds=arguments.timeout_seconds,
                resume=arguments.resume,
            )
            records.append(record)
            completed.add(identifier)
            pending.remove(identifier)
            progressed = True
        if not progressed:
            raise RuntimeError("selected execution graph contains a cycle")
    gpu_seconds = sum(
        record["resources"]["wall_seconds"]
        for record in records if str(record["device"]).startswith("cuda:")
    )
    if gpu_seconds > float(RESOURCE_BUDGET["aggregate_gpu_hours_cap"]) * 3600:
        raise RuntimeError("execution exceeded the aggregate GPU-hour cap")
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "plan_sha256": plan["plan_sha256"],
        "node_count": len(records),
        "aggregate_gpu_seconds": gpu_seconds,
        "bundle_bytes": _tree_bytes(bundle),
        "record_sha256": [record["record_sha256"] for record in records],
    }
    summary["summary_sha256"] = digest_object(summary)
    _write_json(bundle / "logs" / "execution-summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
