#!/usr/bin/env python3
"""Execute selected nodes from the relocatable mixed-collapse launch plan.

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
import subprocess
import sys
import time
from typing import Any


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _resolve_command(
    command: list[str],
    *,
    bundle: Path,
    tokens: Path,
    tokenizer: Path,
    device_override: str | None,
) -> list[str]:
    values = {
        "{python}": sys.executable,
        "{bundle}": str(bundle),
        "{tokens}": str(tokens),
        "{tokenizer}": str(tokenizer),
    }
    resolved = []
    for part in command:
        text = str(part)
        for marker, value in values.items():
            text = text.replace(marker, value)
        resolved.append(text)
    if device_override:
        for option in ("--device", "--prediction-device"):
            if option in resolved:
                index = resolved.index(option)
                resolved[index + 1] = device_override
    return resolved


def _resource_summary(log_path: Path) -> dict[str, Any] | None:
    if not log_path.is_file() or log_path.suffix != ".jsonl":
        return None
    result = None
    with log_path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if value.get("event") == "resource_summary":
                result = value
    return result


def _attempt_directory(runtime: Path, node_id: str) -> Path:
    root = runtime / node_id
    index = 1
    while (root / f"attempt-{index}").exists():
        index += 1
    result = root / f"attempt-{index}"
    result.mkdir(parents=True)
    return result


def _completed(
    bundle: Path, node: dict[str, Any], command: list[str]
) -> bool:
    record_path = bundle / "runtime" / node["id"] / "completed.json"
    if not record_path.is_file():
        return False
    record = _load_json(record_path)
    expected = [bundle / path for path in node["expected_outputs"]]
    if any(not path.is_file() for path in expected):
        return False
    return bool(
        record.get("command_sha256") == hashlib.sha256(_canonical(command)).hexdigest()
        and record.get("outputs")
        == {str(path.relative_to(bundle)): _sha256(path) for path in expected}
    )


def execute_node(
    bundle: Path,
    node: dict[str, Any],
    node_by_id: dict[str, dict[str, Any]],
    *,
    tokens: Path,
    tokenizer: Path,
    device_override: str | None,
) -> None:
    command = _resolve_command(
        node["command"],
        bundle=bundle,
        tokens=tokens,
        tokenizer=tokenizer,
        device_override=device_override,
    )
    if _completed(bundle, node, command):
        print(f"{node['id']}: already completed and digest-verified", flush=True)
        return
    for dependency in node["depends_on"]:
        dependency_node = node_by_id[dependency]
        dependency_command = _resolve_command(
            dependency_node["command"],
            bundle=bundle,
            tokens=tokens,
            tokenizer=tokenizer,
            device_override=None,
        )
        if not _completed(bundle, dependency_node, dependency_command):
            raise RuntimeError(
                f"{node['id']} has incomplete dependency {dependency}"
            )
    outputs = [bundle / path for path in node["expected_outputs"]]
    if any(path.exists() for path in outputs):
        raise FileExistsError(
            f"{node['id']} has unsealed pre-existing output; preserve it and "
            "resolve the incident before rerunning"
        )
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    attempt = _attempt_directory(bundle / "runtime", node["id"])
    stdout_path = attempt / "stdout.log"
    started_ns = time.time_ns()
    started = time.monotonic()
    environment = dict(os.environ)
    source = bundle / "source"
    pythonpath = os.pathsep.join([
        str(source / "experiments"), str(source),
        environment.get("PYTHONPATH", ""),
    ]).rstrip(os.pathsep)
    environment.update({
        "PYTHONPATH": pythonpath,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    })
    print(f"{node['id']}: starting on {device_override or node['device']}", flush=True)
    with stdout_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=bundle,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end="", flush=True)
        exit_code = process.wait()
    elapsed = time.monotonic() - started
    output_digests = {
        str(path.relative_to(bundle)): _sha256(path)
        for path in outputs if path.is_file()
    }
    resource_summary = next(
        (
            _resource_summary(path)
            for path in outputs
            if path.name == "log.jsonl"
        ),
        None,
    )
    record = {
        "schema_version": "pldr-mixed-collapse-execution-v1",
        "node_id": node["id"],
        "stage": node["stage"],
        "started_at_ns": started_ns,
        "elapsed_seconds": elapsed,
        "exit_code": exit_code,
        "command": command,
        "command_sha256": hashlib.sha256(_canonical(command)).hexdigest(),
        "stdout_sha256": _sha256(stdout_path),
        "outputs": output_digests,
        "producer_resource_summary": resource_summary,
    }
    (attempt / "execution.json").write_text(
        json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if exit_code != 0 or len(output_digests) != len(outputs):
        raise RuntimeError(
            f"{node['id']} failed with exit {exit_code}; preserved {attempt}"
        )
    completed_path = bundle / "runtime" / node["id"] / "completed.json"
    completed_path.write_text(
        json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"{node['id']}: completed in {elapsed:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--node", action="append", default=[])
    parser.add_argument(
        "--stage",
        choices=["trajectory", "gate-factorial", "source-radius", "analysis"],
    )
    parser.add_argument("--tokens", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--device")
    arguments = parser.parse_args()
    bundle = arguments.bundle.resolve()
    plan = _load_json(bundle / "protocol" / "launch_plan.json")
    if plan.get("scientific_outcomes_control_execution") is not False:
        raise ValueError("launch plan permits outcome-dependent execution")
    binding = _load_json(bundle / "protocol" / "input_binding.json")
    tokens = (arguments.tokens or Path(binding["tokens_path"])).resolve()
    tokenizer = (arguments.tokenizer or Path(binding["tokenizer_path"])).resolve()
    if _sha256(tokens) != binding["tokens_sha256"]:
        raise ValueError("token archive digest changed")
    if _sha256(tokenizer) != binding["tokenizer_sha256"]:
        raise ValueError("tokenizer digest changed")
    nodes = plan["nodes"]
    node_by_id = {node["id"]: node for node in nodes}
    if len(node_by_id) != len(nodes):
        raise ValueError("launch plan contains duplicate node ids")
    selected = list(arguments.node)
    if arguments.stage:
        selected.extend(
            node["id"] for node in nodes if node["stage"] == arguments.stage
        )
    if not selected:
        raise ValueError("select at least one --node or --stage")
    if len(set(selected)) != len(selected):
        raise ValueError("node selection contains duplicates")
    for node_id in selected:
        if node_id not in node_by_id:
            raise KeyError(f"unknown launch node: {node_id}")
        execute_node(
            bundle,
            node_by_id[node_id],
            node_by_id,
            tokens=tokens,
            tokenizer=tokenizer,
            device_override=arguments.device,
        )


if __name__ == "__main__":
    main()
