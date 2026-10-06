#!/usr/bin/env python3
"""Seal or verify the fresh-context radial-predictor holdout."""

from __future__ import annotations
from companion_paths import acquisition_identity

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from bundle_hygiene import (
    assert_clean_tree,
    relative_payload_files,
)

CAMPAIGN_ID = "pldr-radial-context-holdout-v1"
RELEASE_ID = acquisition_identity('radial-context-acquisition-release')
SEAL_SCHEMA = "pldr-radial-context-holdout-seal-v1"
RESOURCE_SCHEMA = "pldr-radial-context-holdout-resource-v1"
ROOT_SEAL_FILES = (
    "PAYLOAD.sha256",
    "RUNTIME.sha256",
    "provenance-seal.json",
    "MANIFEST.sha256",
)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    write_bytes(path, canonical_bytes(value))


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_files(root: Path) -> list[Path]:
    return relative_payload_files(root)


def manifest_bytes(root: Path, paths: list[Path]) -> bytes:
    return "".join(
        f"{sha256_path(root / relative)}  {relative.as_posix()}\n"
        for relative in sorted(paths)
    ).encode("ascii")


def parse_manifest(path: Path) -> dict[Path, str]:
    rows: dict[Path, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        digest, separator, relative = line.partition("  ")
        candidate = Path(relative)
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or candidate.is_absolute()
            or ".." in candidate.parts
            or candidate in rows
        ):
            raise ValueError(f"malformed manifest line: {path}")
        rows[candidate] = digest
    return rows


def verify_manifest(path: Path, root: Path, expected: set[Path]) -> None:
    rows = parse_manifest(path)
    if set(rows) != expected:
        raise ValueError(f"manifest membership changed: {path.name}")
    for relative, expected_digest in rows.items():
        target = root / relative
        if target.is_symlink() or not target.is_file():
            raise ValueError(f"manifest target is absent: {relative}")
        if sha256_path(target) != expected_digest:
            raise ValueError(f"manifest target changed: {relative}")


def payload_paths(root: Path) -> list[Path]:
    return [
        relative for relative in relative_files(root)
        if relative.parts[0] in {
            "analysis", "protocol", "records", "source"
        }
        or relative in {Path("README.md"), Path("incident-index.json")}
    ]


def runtime_paths(root: Path) -> list[Path]:
    return [
        relative for relative in relative_files(root)
        if relative.parts[0] == "runtime"
    ]


def verify_sources_and_inputs(
    binding_root: Path, campaign_root: Path, design: dict[str, Any]
) -> None:
    source_digests = design.get("source_sha256")
    input_digests = design.get("input_sha256")
    bound_inputs = design.get("bound_inputs")
    if (
        not isinstance(source_digests, dict)
        or not source_digests
        or not isinstance(input_digests, dict)
        or not isinstance(bound_inputs, dict)
        or set(input_digests) != set(bound_inputs)
    ):
        raise ValueError("radial holdout source or input digest map is incomplete")
    for relative, expected in sorted(source_digests.items()):
        path = campaign_root / "source" / relative
        if path.is_symlink() or not path.is_file() or sha256_path(path) != expected:
            raise ValueError(f"frozen radial holdout source changed: {relative}")
    for name, expected in sorted(input_digests.items()):
        path = Path(bound_inputs[name]).resolve(strict=True)
        if not path.is_relative_to(binding_root) or path.is_symlink():
            raise ValueError(f"invalid radial holdout input binding: {name}")
        if sha256_path(path) != expected:
            raise ValueError(f"radial holdout input changed: {name}")


def verify_core(
    binding_root: Path, campaign_root: Path
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    assert_clean_tree(campaign_root)
    if not campaign_root.is_relative_to(binding_root):
        raise ValueError("radial holdout root escapes binding root")
    design = load_json(campaign_root / "protocol/design.json")
    plan = load_json(campaign_root / "protocol/launch-plan.json")
    if (
        design.get("campaign_id") != CAMPAIGN_ID
        or design.get("release_id") != RELEASE_ID
        or plan.get("campaign_id") != CAMPAIGN_ID
        or Path(design.get("binding_root", "")).resolve() != binding_root
        or Path(design.get("campaign_root", "")).resolve() != campaign_root
        or Path(plan.get("binding_root", "")).resolve() != binding_root
        or Path(plan.get("campaign_root", "")).resolve() != campaign_root
        or plan.get("node_count") != len(plan.get("nodes", []))
        or plan.get("node_count") != 31
    ):
        raise ValueError("radial holdout design or launch identity changed")
    verify_sources_and_inputs(binding_root, campaign_root, design)

    identifiers: set[str] = set()
    records: list[dict[str, Any]] = []
    for node in plan["nodes"]:
        identifier = str(node["node_id"])
        if identifier in identifiers:
            raise ValueError(f"duplicate radial holdout node: {identifier}")
        identifiers.add(identifier)
        outputs = [Path(path).resolve() for path in node["expected_outputs"]]
        if any(
            not path.is_relative_to(campaign_root)
            or path.is_symlink()
            or not path.is_file()
            for path in outputs
        ):
            raise ValueError(f"radial holdout output is absent: {identifier}")
        resource = campaign_root / "runtime" / identifier / "resource.json"
        value = load_json(resource)
        if (
            value.get("schema_version") != RESOURCE_SCHEMA
            or value.get("node_id") != identifier
            or value.get("technical_valid") is not True
            or "wall_seconds" in value
            or "attempt_elapsed_wall_seconds" not in value
            or "accelerator_reservation_wall_seconds" not in value
        ):
            raise ValueError(f"invalid radial holdout resource record: {identifier}")
        if str(node["device"]).startswith("cuda:"):
            if value.get("gpu_monitor_status") not in {
                "measured", "unavailable"
            }:
                raise ValueError(f"GPU monitor is inconsistent: {identifier}")
            if int(value.get("producer_peak_gpu_reserved_bytes", 0)) <= 0:
                raise ValueError(f"producer CUDA counter is absent: {identifier}")
        elif value.get("gpu_monitor_status") != "not_applicable":
            raise ValueError(f"CPU monitor status changed: {identifier}")
        if "check_command" in node and (
            value.get("analysis_check_status") != "passed"
            or value.get("analysis_check_exit_code") != 0
        ):
            raise ValueError("final radial holdout analysis did not reproduce")
        records.append(value)

    runtime_directories = {
        path.name for path in (campaign_root / "runtime").iterdir()
        if path.is_dir()
    }
    if runtime_directories != identifiers:
        raise ValueError("radial holdout runtime-node membership changed")
    return design, plan, records


def analysis_check(
    campaign_root: Path, plan: dict[str, Any]
) -> subprocess.CompletedProcess[str]:
    nodes = [node for node in plan["nodes"] if node["node_id"] == "final-analysis"]
    if len(nodes) != 1 or "check_command" not in nodes[0]:
        raise ValueError("radial holdout analysis check command is absent")
    result = subprocess.run(
        [str(value) for value in nodes[0]["check_command"]],
        cwd=campaign_root,
        env={
            **os.environ,
            "PYTHONPATH": str(campaign_root / "source/experiments"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": "/tmp/pldr-radial-context-holdout-pycache",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "radial holdout analysis check failed: " + result.stderr.strip()
        )
    return result


def incident_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    incidents = []
    for record in records:
        status = record.get("gpu_monitor_status")
        if status not in {"unavailable", "inconsistent"}:
            continue
        incidents.append({
            "node_id": record["node_id"],
            "device": record["device"],
            "status": status,
            "monitor_peak_gpu_reserved_bytes": record.get(
                "monitor_peak_gpu_reserved_bytes"
            ),
            "producer_peak_gpu_reserved_bytes": record.get(
                "producer_peak_gpu_reserved_bytes"
            ),
            "gpu_probe_kind": record.get("gpu_probe_kind"),
            "gpu_probe_error": record.get("gpu_probe_error"),
            "gpu_probe_observation": record.get("gpu_probe_observation"),
            "incident": record.get("gpu_monitor_incident"),
        })
    return {
        "schema_version": "pldr-incident-index-v1",
        "campaign_id": CAMPAIGN_ID,
        "incidents": incidents,
    }


def seal(binding_root: Path, campaign_root: Path) -> None:
    if any((campaign_root / name).exists() for name in ROOT_SEAL_FILES):
        raise FileExistsError("radial holdout already has seal material")
    design, plan, records = verify_core(binding_root, campaign_root)
    analysis_check(campaign_root, plan)
    write_json(campaign_root / "incident-index.json", incident_payload(records))
    write_bytes(
        campaign_root / "PAYLOAD.sha256",
        manifest_bytes(campaign_root, payload_paths(campaign_root)),
    )
    write_bytes(
        campaign_root / "RUNTIME.sha256",
        manifest_bytes(campaign_root, runtime_paths(campaign_root)),
    )
    summary = campaign_root / "analysis/radial_context_holdout_summary.json"
    provenance = {
        "schema_version": SEAL_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "release_id": RELEASE_ID,
        "design_sha256": sha256_path(campaign_root / "protocol/design.json"),
        "launch_plan_sha256": sha256_path(
            campaign_root / "protocol/launch-plan.json"
        ),
        "payload_manifest_sha256": sha256_path(
            campaign_root / "PAYLOAD.sha256"
        ),
        "runtime_manifest_sha256": sha256_path(
            campaign_root / "RUNTIME.sha256"
        ),
        "analysis_summary_sha256": sha256_path(summary),
        "source_sha256": design["source_sha256"],
        "input_sha256": design["input_sha256"],
        "node_count": len(records),
        "science_record_count": len(list((campaign_root / "records").glob("*.npz"))),
        "gpu_monitor_incident_count": len(
            incident_payload(records)["incidents"]
        ),
        "all_nodes_technical_valid": all(
            record["technical_valid"] is True for record in records
        ),
    }
    write_json(campaign_root / "provenance-seal.json", provenance)
    all_except_manifest = [
        relative for relative in relative_files(campaign_root)
        if relative != Path("MANIFEST.sha256")
    ]
    write_bytes(
        campaign_root / "MANIFEST.sha256",
        manifest_bytes(campaign_root, all_except_manifest),
    )
    verify(binding_root, campaign_root)
    print(
        "radial holdout seal: wrote and verified "
        f"{len(all_except_manifest)} files"
    )


def verify(binding_root: Path, campaign_root: Path) -> None:
    assert_clean_tree(campaign_root)
    design, plan, records = verify_core(binding_root, campaign_root)
    all_except_manifest = {
        relative for relative in relative_files(campaign_root)
        if relative != Path("MANIFEST.sha256")
    }
    verify_manifest(
        campaign_root / "MANIFEST.sha256",
        campaign_root,
        all_except_manifest,
    )
    verify_manifest(
        campaign_root / "PAYLOAD.sha256",
        campaign_root,
        set(payload_paths(campaign_root)),
    )
    verify_manifest(
        campaign_root / "RUNTIME.sha256",
        campaign_root,
        set(runtime_paths(campaign_root)),
    )
    expected_incidents = canonical_bytes(incident_payload(records))
    if (campaign_root / "incident-index.json").read_bytes() != expected_incidents:
        raise ValueError("radial holdout incident index changed")
    provenance = load_json(campaign_root / "provenance-seal.json")
    if (
        provenance.get("schema_version") != SEAL_SCHEMA
        or provenance.get("campaign_id") != CAMPAIGN_ID
        or provenance.get("design_sha256")
        != sha256_path(campaign_root / "protocol/design.json")
        or provenance.get("launch_plan_sha256")
        != sha256_path(campaign_root / "protocol/launch-plan.json")
        or provenance.get("payload_manifest_sha256")
        != sha256_path(campaign_root / "PAYLOAD.sha256")
        or provenance.get("runtime_manifest_sha256")
        != sha256_path(campaign_root / "RUNTIME.sha256")
        or provenance.get("analysis_summary_sha256")
        != sha256_path(campaign_root / "analysis/radial_context_holdout_summary.json")
        or provenance.get("source_sha256") != design["source_sha256"]
        or provenance.get("input_sha256") != design["input_sha256"]
        or provenance.get("node_count") != len(records)
        or provenance.get("gpu_monitor_incident_count")
        != len(incident_payload(records)["incidents"])
        or provenance.get("all_nodes_technical_valid") is not True
    ):
        raise ValueError("radial holdout provenance seal changed")
    analysis_check(campaign_root, plan)
    print(
        "radial holdout seal: verified "
        f"{len(records)} nodes and {len(all_except_manifest)} files"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("seal", "verify"))
    parser.add_argument("--binding-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    arguments = parser.parse_args()
    binding_root = arguments.binding_root.resolve(strict=True)
    campaign_root = arguments.campaign_root.resolve(strict=True)
    if arguments.command == "seal":
        seal(binding_root, campaign_root)
    else:
        verify(binding_root, campaign_root)


if __name__ == "__main__":
    main()
