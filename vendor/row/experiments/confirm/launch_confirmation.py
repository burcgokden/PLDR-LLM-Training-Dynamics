#!/usr/bin/env python3
"""Single launch, record, validation, and analysis boundary for E0-E8."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
ROOT = EXPERIMENTS.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXPERIMENTS / "analysis"))
sys.path.insert(0, str(SCRIPTS))

import campaign_record as record_contract  # noqa: E402
from assemble_confirmation import assemble  # noqa: E402
from certificate_graph import (  # noqa: E402
    digest_object,
    load_graph,
)
from estimator_manifest import load_manifest  # noqa: E402
from confirmation_analysis import (  # noqa: E402
    analyze, load_records, load_registration,
)
from confirmation_qualification import run_qualification  # noqa: E402
import gen_confirmation_protocols as protocol_generator  # noqa: E402


DEPENDENCIES = {
    pid: tuple(protocol_generator.SPECS[pid]["depends_on"])
    for pid in record_contract.PROTOCOLS
}
PROTOCOL_DIRECTORY = (
    EXPERIMENTS / "protocols" / "row_map_confirmation")


def _json_read(path):
    return record_contract.strict_load(path)


def _json_write(path, value, *, exclusive=False):
    mode = "x" if exclusive else "w"
    with Path(path).open(mode, encoding="utf-8") as stream:
        stream.write(record_contract.strict_dumps(value))


def _protocols_current():
    stale = []
    for path, expected in protocol_generator.generated_files().items():
        if not path.is_file() or path.read_text(encoding="utf-8") != expected:
            stale.append(str(path.relative_to(ROOT)))
    if stale:
        raise RuntimeError(
            "generated confirmation specifications are stale: "
            + ", ".join(stale))


def _git_value(arguments, directory):
    result = subprocess.run(
        ["git", "-C", str(directory), *arguments],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def _source_state(source_root):
    source_root = Path(source_root).resolve()
    commit = _git_value(["rev-parse", "HEAD"], source_root)
    clean = not bool(_git_value(["status", "--porcelain"], source_root))
    return commit, clean


def _campaign_metadata(campaign_directory):
    campaign_directory = Path(campaign_directory).resolve()
    metadata_path = campaign_directory / "campaign.json"
    if not metadata_path.is_file():
        raise RuntimeError("campaign.json is missing")
    metadata = _json_read(metadata_path)
    if metadata.get("campaign_id") != campaign_directory.name:
        raise RuntimeError("campaign identifier disagrees with its directory")
    if metadata.get("schema_version") != record_contract.SCHEMA_VERSION:
        raise RuntimeError("campaign schema is stale")
    return metadata


def qualify(output=None, device="cpu"):
    result = run_qualification(device=device)
    if output is not None:
        _json_write(output, result)
    if result["status"] != "PASS":
        raise RuntimeError("Q0 instrument qualification failed")
    return result


def initialize(campaign_root, campaign_id, binding_root, source_root,
               device="cpu"):
    _protocols_current()
    campaign_id = record_contract._identifier(
        campaign_id, "campaign_id")
    binding_root = Path(binding_root).resolve()
    source_root = Path(source_root).resolve()
    try:
        source_root.relative_to(binding_root)
    except ValueError as error:
        raise RuntimeError(
            "source_root must lie inside binding_root") from error
    commit, clean = _source_state(source_root)
    if not clean:
        raise RuntimeError(
            "campaign initialization requires a clean source tree")

    campaign_root = Path(campaign_root).resolve()
    campaign_root.mkdir(parents=True, exist_ok=True)
    campaign_directory = campaign_root / campaign_id
    campaign_directory.mkdir()
    try:
        (campaign_directory / "raw").mkdir()
        for pid in record_contract.PROTOCOLS:
            (campaign_directory / "raw" / pid).mkdir()
        (campaign_directory / "derived").mkdir()
        (campaign_directory / "launch").mkdir()
        frozen = campaign_directory / "frozen"
        frozen.mkdir()
        shutil.copytree(
            PROTOCOL_DIRECTORY, frozen / "protocols",
            copy_function=shutil.copy2,
        )
        q0 = qualify(campaign_directory / "q0.json", device=device)
        metadata = {
            "schema_version": record_contract.SCHEMA_VERSION,
            "campaign_id": campaign_id,
            "status": "INITIALIZED",
            "binding_root": str(binding_root),
            "source_root": str(source_root),
            "source_commit": commit,
            "source_tree_clean": clean,
            "qualification_status": q0["status"],
            "clock": "global_optimizer_step",
            "launcher": str(Path(__file__).resolve()),
            "dependencies": {
                pid: list(value) for pid, value in DEPENDENCIES.items()
            },
        }
        _json_write(
            campaign_directory / "campaign.json", metadata,
            exclusive=True,
        )
    except Exception:
        shutil.rmtree(campaign_directory)
        raise
    return campaign_directory


def _analysis_path(campaign_directory):
    return Path(campaign_directory) / "derived" / "analysis.json"


def _authorize(campaign_directory, protocol_id):
    if protocol_id not in record_contract.PROTOCOLS:
        raise ValueError("protocol_id must be E0 through E8")
    metadata = _campaign_metadata(campaign_directory)
    realized_commit, realized_clean = _source_state(metadata["source_root"])
    if realized_commit != metadata["source_commit"]:
        raise RuntimeError("source commit changed after campaign initialization")
    if not realized_clean:
        raise RuntimeError(
            "launch and record operations require a clean source tree"
        )
    q0 = _json_read(Path(campaign_directory) / "q0.json")
    if q0.get("status") != "PASS":
        raise RuntimeError("Q0 has not passed")
    dependencies = [
        value for value in DEPENDENCIES[protocol_id] if value != "Q0"
    ]
    if dependencies:
        analysis_path = _analysis_path(campaign_directory)
        if not analysis_path.is_file():
            raise RuntimeError(
                "dependency analysis is missing for " + protocol_id)
        analysis = _json_read(analysis_path)
        statuses = analysis.get("experiments", {})
        missing = [
            value for value in dependencies
            if statuses.get(value, {}).get("status") != "CONFIRMED"
        ]
        if missing:
            raise RuntimeError(
                "unconfirmed dependencies: " + ", ".join(missing))
    return metadata


def launch(campaign_directory, protocol_id, command, dry_run=False):
    metadata = _authorize(campaign_directory, protocol_id)
    if not command:
        raise ValueError("a command is required after --")
    campaign_directory = Path(campaign_directory).resolve()
    launch_directory = campaign_directory / "launch"
    sequence = len(list(launch_directory.glob("launch-*.json")))
    launch_record = {
        "schema_version": record_contract.SCHEMA_VERSION,
        "campaign_id": metadata["campaign_id"],
        "protocol_id": protocol_id,
        "sequence": sequence,
        "command": list(command),
        "dry_run": bool(dry_run),
        "status": "DRY_RUN" if dry_run else "STARTED",
    }
    path = launch_directory / f"launch-{sequence:05d}.json"
    _json_write(path, launch_record, exclusive=True)
    if dry_run:
        return 0
    environment = os.environ.copy()
    environment["PLDR_CONFIRMATION_CAMPAIGN"] = str(campaign_directory)
    environment["PLDR_CONFIRMATION_PROTOCOL"] = protocol_id
    completed = subprocess.run(
        list(command), cwd=ROOT, env=environment, check=False)
    launch_record["status"] = (
        "COMPLETED" if completed.returncode == 0 else "FAILED")
    launch_record["returncode"] = completed.returncode
    _json_write(path, launch_record)
    return completed.returncode


def _load_array_file(path, label):
    value = _json_read(path)
    if not isinstance(value, list):
        raise ValueError(f"{label} must contain a JSON array")
    return value


def _artifact_rows(paths, phase_directory):
    rows = []
    phase_directory = Path(phase_directory).resolve()
    for source in paths:
        source = Path(source).resolve()
        try:
            relative = source.relative_to(phase_directory)
        except ValueError as error:
            raise ValueError(
                "raw artifacts must lie inside the protocol raw directory"
            ) from error
        if not source.is_file():
            raise ValueError(f"artifact is not a file: {source}")
        rows.append({
            "name": record_contract._identifier(
                source.stem, "artifact name"),
            "path": relative.as_posix(),
            "sha256": record_contract.sha256_file(source),
            "media_type": (
                "application/json" if source.suffix == ".json"
                else "application/octet-stream"
            ),
        })
    return rows


def _bound_measurement_artifact(graph, artifact_paths, protocol_id):
    roots = [
        node for node in graph["nodes"]
        if node["derivation_kind"] == "root_artifact"
    ]
    if len(roots) != 1:
        raise ValueError(
            "measurement graph must contain exactly one root artifact")
    root_value = roots[0]["value"]
    if not isinstance(root_value, dict) or set(root_value) != {
        "protocol_id", "artifact_sha256", "schema_version",
    }:
        raise ValueError("measurement graph root artifact is not closed")
    if root_value["protocol_id"] != protocol_id:
        raise ValueError(
            "measurement graph root protocol disagrees with record")
    matches = []
    for path in artifact_paths:
        path = Path(path)
        if path.suffix != ".json":
            continue
        try:
            value = _json_read(path)
        except (OSError, ValueError, record_contract.RecordError):
            continue
        if digest_object(value) == root_value["artifact_sha256"]:
            matches.append((path, value))
    if len(matches) != 1:
        raise ValueError(
            "record must attach exactly one graph-bound measurement artifact")
    _, measurement = matches[0]
    if measurement.get("schema_version") != root_value["schema_version"]:
        raise ValueError(
            "measurement artifact schema disagrees with graph root")
    if not isinstance(measurement.get("measurements"), list):
        raise ValueError("bound measurement artifact has no measurement array")
    return measurement


def create_record(args):
    campaign_directory = Path(args.campaign_dir).resolve()
    metadata = _authorize(campaign_directory, args.protocol)
    binding_root = Path(metadata["binding_root"])
    phase_directory = campaign_directory / "raw" / args.protocol
    protocol_spec = next(
        (campaign_directory / "frozen" / "protocols").glob(
            f"{args.protocol.lower()}_*.md"
        ),
        None,
    )
    if protocol_spec is None:
        raise RuntimeError("frozen protocol file is missing")
    schema_path = (
        campaign_directory / "frozen" / "protocols"
        / "record_schema.json"
    )
    registration_path = (
        campaign_directory / "frozen" / "protocols" / "registration.json"
    )
    estimator_path = (
        campaign_directory / "frozen" / "protocols" / "estimator_manifest.json"
    )
    binding_paths = {
        "protocol": protocol_spec,
        "schema": schema_path,
        "registration": registration_path,
        "estimator_manifest": estimator_path,
        "source": Path(args.source_binding),
        "model": Path(args.model),
        "data_order": Path(args.data_order),
        "tokenizer": Path(args.tokenizer),
    }
    bindings = {
        name: record_contract.binding_for(path, binding_root)
        for name, path in binding_paths.items()
    }
    assembled = _json_read(args.assembled)
    if not isinstance(assembled, dict) or set(assembled) != {
        "protocol_id", "certificate_graph_sha256", "measurements",
        "node_bindings",
    }:
        raise ValueError("assembled measurement bundle is not closed")
    if assembled["protocol_id"] != args.protocol:
        raise ValueError("assembled measurement protocol disagrees with record")
    graph = load_graph(args.certificate_graph)
    _bound_measurement_artifact(graph, args.artifact, args.protocol)
    manifest = load_manifest(estimator_path, ROOT)
    replayed = assemble(
        args.protocol, graph, assembled["node_bindings"], manifest,
    )
    if replayed != assembled:
        raise ValueError("assembled measurement bundle fails exact replay")
    measurements = assembled["measurements"]
    tensor_groups = _load_array_file(args.tensor_groups, "tensor_groups")
    artifacts = _artifact_rows(
        [*args.artifact, args.assembled, args.certificate_graph],
        phase_directory,
    )

    import numpy
    import torch

    record = {
        "schema_version": record_contract.SCHEMA_VERSION,
        "campaign_id": metadata["campaign_id"],
        "protocol_id": args.protocol,
        "run_id": args.run_id,
        "seed": args.seed,
        "unit_key": {
            "seed": args.seed,
            "checkpoint": args.global_step,
            "layer_bundle": args.layer_bundle,
            "tensor_group": args.tensor_group,
            "certificate_box": args.certificate_box,
            "schedule_cell": args.schedule_cell,
            "architecture_cell": args.architecture_cell,
            "block_size": args.block_size,
            "arm": args.arm,
        },
        "clocks": {
            "global_optimizer_step": args.global_step,
            "segment_index": args.segment_index,
            "segment_local_step": args.segment_step,
            "segment_global_offset": args.segment_offset,
            "checkpoint_global_step": args.checkpoint_step,
        },
        "bindings": bindings,
        "tensor_groups": tensor_groups,
        "intervention": {
            "arm": args.arm,
            "assigned_dose": args.dose,
            "applied_after_optimizer": args.post_optimizer,
            "description": args.intervention_description,
        },
        "measurements": measurements,
        "artifacts": artifacts,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "device": args.device,
            "dtype": args.dtype,
            "git_commit": metadata["source_commit"],
            "source_tree_clean": metadata["source_tree_clean"],
        },
    }
    filename = (
        f"record-{args.run_id}-{args.global_step:09d}.json")
    destination = phase_directory / filename
    record_contract.write_strict(destination, record, exclusive=True)
    record_contract.load_and_verify(
        destination, binding_root=binding_root,
        artifact_root=phase_directory,
    )
    return destination


def validate_campaign(campaign_directory):
    metadata = _campaign_metadata(campaign_directory)
    loaded = load_records(
        Path(campaign_directory) / "raw", metadata["binding_root"])
    return {
        "status": "PASS",
        "record_count": len(loaded),
        "protocol_counts": {
            pid: sum(row["protocol_id"] == pid for row in loaded)
            for pid in record_contract.PROTOCOLS
        },
    }


def analyze_campaign(campaign_directory):
    metadata = _campaign_metadata(campaign_directory)
    campaign_directory = Path(campaign_directory)
    registration = load_registration(
        campaign_directory / "frozen" / "protocols"
        / "registration.json"
    )
    loaded = load_records(
        campaign_directory / "raw", metadata["binding_root"])
    result = analyze(loaded, registration)
    _json_write(_analysis_path(campaign_directory), result)
    return result


def parser():
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="action", required=True)

    q0 = sub.add_parser("qualify")
    q0.add_argument("--device", default="cpu")
    q0.add_argument("--output")

    init = sub.add_parser("init")
    init.add_argument("--campaign-root", required=True)
    init.add_argument("--campaign-id", required=True)
    init.add_argument("--binding-root", default=str(ROOT.parent))
    init.add_argument("--source-root", default=str(ROOT))
    init.add_argument("--device", default="cpu")

    run = sub.add_parser("run")
    run.add_argument("--campaign-dir", required=True)
    run.add_argument("--protocol", choices=record_contract.PROTOCOLS,
                     required=True)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("command", nargs=argparse.REMAINDER)

    record = sub.add_parser("record")
    record.add_argument("--campaign-dir", required=True)
    record.add_argument("--protocol", choices=record_contract.PROTOCOLS,
                        required=True)
    record.add_argument("--run-id", required=True)
    record.add_argument("--seed", type=int, required=True)
    record.add_argument("--global-step", type=int, required=True)
    record.add_argument("--segment-index", type=int, default=0)
    record.add_argument("--segment-step", type=int, required=True)
    record.add_argument("--segment-offset", type=int, required=True)
    record.add_argument("--checkpoint-step", type=int, required=True)
    record.add_argument("--source-binding", required=True)
    record.add_argument("--model", required=True)
    record.add_argument("--data-order", required=True)
    record.add_argument("--tokenizer", required=True)
    record.add_argument("--tensor-groups", required=True)
    record.add_argument("--assembled", required=True)
    record.add_argument("--certificate-graph", required=True)
    record.add_argument("--layer-bundle")
    record.add_argument("--tensor-group")
    record.add_argument("--certificate-box")
    record.add_argument("--schedule-cell")
    record.add_argument("--architecture-cell")
    record.add_argument("--block-size", type=int)
    record.add_argument("--artifact", action="append", default=[])
    record.add_argument("--arm", default="observational")
    record.add_argument("--dose", type=float)
    record.add_argument("--post-optimizer", action="store_true")
    record.add_argument(
        "--intervention-description", default="no intervention")
    record.add_argument("--device", required=True)
    record.add_argument("--dtype", default="float64")

    validate = sub.add_parser("validate")
    validate.add_argument("--campaign-dir", required=True)

    analysis = sub.add_parser("analyze")
    analysis.add_argument("--campaign-dir", required=True)
    return root


def main():
    args = parser().parse_args()
    if args.action == "qualify":
        result = qualify(args.output, args.device)
        print(result["status"])
        return
    if args.action == "init":
        path = initialize(
            args.campaign_root, args.campaign_id, args.binding_root,
            args.source_root, args.device,
        )
        print(path)
        return
    if args.action == "run":
        command = (
            args.command[1:] if args.command[:1] == ["--"]
            else args.command
        )
        raise SystemExit(launch(
            args.campaign_dir, args.protocol, command, args.dry_run))
    if args.action == "record":
        print(create_record(args))
        return
    if args.action == "validate":
        print(record_contract.strict_dumps(
            validate_campaign(args.campaign_dir)), end="")
        return
    if args.action == "analyze":
        print(record_contract.strict_dumps(
            analyze_campaign(args.campaign_dir)), end="")
        return
    raise AssertionError("unreachable")


if __name__ == "__main__":
    main()

