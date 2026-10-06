#!/usr/bin/env python3
"""Execute the source-resolved producer graph with a deadline and observed memory/storage admission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from confirm.confirmation_artifacts import sha256_path  # noqa: E402
from confirm.resource_executor import (  # noqa: E402
    ResourceCaps,
    record_is_admissible,
    resource_record_path,
    resource_records,
    canonical_digest,
    directory_bytes,
    run_capped,
    write_json_atomic,
)
from confirm.source_resolved_specs import (  # noqa: E402
    CAMPAIGN_ID as SOURCE_CAMPAIGN_ID,
)


CAMPAIGN_ID = SOURCE_CAMPAIGN_ID
LAUNCH_PLAN_SCHEMA = "pldr-source-resolved-launch-plan-v1"
RESOURCE_SCHEMA = "pldr-source-resolved-resource-record-v1"
PREFLIGHT_SCHEMA = "pldr-source-resolved-preflight-failure-v1"
EXECUTION_SCHEMA = "pldr-source-resolved-execution-report-v1"
CAMPAIGN_LABEL = "source-resolved"


def _load_plan(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != LAUNCH_PLAN_SCHEMA
        or value.get("campaign_id") != CAMPAIGN_ID
        or not isinstance(value.get("nodes"), list)
    ):
        raise ValueError(f"unknown {CAMPAIGN_LABEL} launch plan")
    bundle = Path(value["bundle_root"]).resolve()
    if path.resolve() != bundle / "protocol" / "launch_plan.json":
        raise ValueError("launch plan is outside its bound bundle")
    for name in ("dataset", "tokenizer", "registry"):
        record = value[name]
        artifact = Path(record["path"]).resolve()
        if not artifact.is_file() or sha256_path(artifact) != record["sha256"]:
            raise ValueError(f"bound launch input changed: {name}")
    for record in value["orders"].values():
        artifact = Path(record["path"]).resolve()
        if not artifact.is_file() or sha256_path(artifact) != record["sha256"]:
            raise ValueError("a frozen training order changed")
    ids = [node["id"] for node in value["nodes"]]
    if len(ids) != len(set(ids)):
        raise ValueError("launch plan has duplicate node identifiers")
    positions = {node_id: index for index, node_id in enumerate(ids)}
    for index, node in enumerate(value["nodes"]):
        dependencies = node.get("depends_on")
        command = node.get("command")
        if (
            not isinstance(dependencies, list)
            or any(dependency not in positions or positions[dependency] >= index for dependency in dependencies)
            or not isinstance(command, list)
            or not command
            or any(not isinstance(argument, str) or not argument for argument in command)
            or not isinstance(node.get("projected_output_bytes"), int)
            or node["projected_output_bytes"] < 0
        ):
            raise ValueError(f"launch node {node.get('id')} is malformed or unsorted")
        ResourceCaps(**node["caps"]).validate()
    return value


def _gpu_count(device: str) -> int:
    return len(set(re.findall(r"cuda:(\d+)", device)))


def _resource_path(bundle: Path, node_id: str) -> Path:
    return resource_record_path(bundle, node_id)


def _completed(bundle: Path, node: dict[str, Any]) -> bool:
    """Per-process compatibility check only; campaign admission uses its journal."""
    path = _resource_path(bundle, node["id"])
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    outputs = node["expected_outputs"]
    digests = record.get("expected_output_sha256")
    return bool(
        record.get("schema_version") == RESOURCE_SCHEMA
        and record.get("campaign_id") == CAMPAIGN_ID
        and record.get("node_id") == node["id"]
        and record_is_admissible(record)
        and record.get("command_sha256") == canonical_digest(node["command"])
        and isinstance(digests, dict)
        and set(digests) == set(outputs)
        and all(
            Path(output).is_file()
            and sha256_path(Path(output)) == digests[output]
            for output in outputs
        )
    )


def execute(plan, *, stages, node_ids, dry_run, continue_on_failure):
    """Apply persistent full-plan admission; stage selection never resets cost."""
    from confirm.campaign_plan_executor import execute_plan
    return execute_plan(plan, stages=stages, node_ids=node_ids, dry_run=dry_run,
                        continue_on_failure=continue_on_failure,
                        resource_schema=RESOURCE_SCHEMA, execution_schema=EXECUTION_SCHEMA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--stages", nargs="*")
    parser.add_argument("--nodes", nargs="*")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    plan = _load_plan(arguments.plan)
    report = execute(
        plan,
        stages=None if arguments.stages is None else set(arguments.stages),
        node_ids=None if arguments.nodes is None else set(arguments.nodes),
        dry_run=arguments.dry_run,
        continue_on_failure=arguments.continue_on_failure,
    )
    output = arguments.output or (
        Path(plan["bundle_root"]) / "reports" / "execution-report.json"
    )
    write_json_atomic(output, report)
    print(json.dumps({
        "dry_run": report["dry_run"],
        "campaign_decision": report["campaign_decision"],
        "statuses": {row["id"]: row["status"] for row in report["nodes"]},
        "output": str(output.resolve()),
    }, sort_keys=True))
    if report["campaign_decision"] != "admissible" or any(row["status"] not in {"passed", "already_completed", "ready"} for row in report["nodes"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
