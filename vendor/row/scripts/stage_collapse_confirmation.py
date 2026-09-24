#!/usr/bin/env python3
"""Stage the immutable composite-collapse confirmation campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIRM = ROOT / "experiments" / "confirm"
sys.path.insert(0, str(CONFIRM))

from program_energy_protocol_specs import (  # noqa: E402
    ARCHITECTURE_RUNS,
    RESOURCE_CAPS,
    STAGES,
)


PROTOCOL_SOURCE = (
    ROOT / "experiments" / "protocols" / "program_bound_confirmation"
)
DEFAULT_OUTPUT = (
    ROOT.parent / "experiment-data" / "manuscript-revisions" / "rev33"
    / "confirmation-program"
)
MUTABLE_ROOT_OUTPUTS = {
    Path("execution_plan.json"),
    Path("measurement_registry.json"),
    Path("partition_manifest.json"),
    Path("q_qualification.json"),
    Path("gated_replication_command.json"),
    Path("training_commands.json"),
}


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _protocol_payloads():
    names = (
        [
            "README.md",
            "CHECKSUMS.sha256",
            "campaign_design.json",
            "request_schema.json",
            "resource_metadata_schema.json",
            "stage_record_schema.json",
            "raw_observation_schemas.json",
        ]
        + [f"{stage['id'].lower()}_protocol.json" for stage in STAGES]
    )
    paths = [PROTOCOL_SOURCE / name for name in names]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "generate the confirmation protocols first: "
            + ", ".join(missing)
        )
    return {
        Path("protocols") / path.name: path.read_bytes()
        for path in sorted(paths)
    }


def _bundle_digest(files):
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda value: value.as_posix()):
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(bytes([0]))
        digest.update(files[path])
        digest.update(bytes([0]))
    return digest.hexdigest()


README = """# PLDR row-map collapse confirmation staging

This immutable bundle owns Q, T, N, I, A, and R for the comprehensive
collapse theory. It qualifies every physical row, replays complete training
states, constructs the composite observable frame from live hooks, checks the
decay-aware optimizer corridor, transports the same-source bound to logits,
and independently replays the complete analysis.

The hard limits are 4.5 GPU-hours, two GPUs, and 12 wall-clock hours. A
measurement registry fixes disjoint construction, validation, application,
and reserve contexts before training. Scientific raw records can be created
only by the live model producers. The analyzer recomputes all capacities,
Gram matrices, charges, comparison operators, bounds, and decisions.

Run `./run_next_confirmation.sh help` for the exact commands. Campaign outputs
belong below `campaigns/CAMPAIGN_ID`; protocols and staging manifests remain
immutable.
"""


WRAPPER = """#!/bin/sh
set -eu

repo_root=/pldr-work/row-code
data_root=/pldr-data/row/rev33/confirmation-program
launch_dir=$(pwd)
command=${1-}

absolute_path() {
  case "$1" in
    /*) printf '%s\\n' "$1" ;;
    *) printf '%s/%s\\n' "$launch_dir" "$1" ;;
  esac
}

case "$command" in
  plan)
    cd "$repo_root"
    python3 experiments/confirm/run_program_energy_campaign.py plan \\
      --output "$(absolute_path "${OUTPUT:-$data_root/execution_plan.json}")"
    ;;
  registry)
    : "${DATASET:?}" "${TOKENIZER:?}"
    cd "$repo_root"
    python3 experiments/confirm/live_confirmation_producers.py \\
      measurement-registry \\
      --dataset "$(absolute_path "$DATASET")" \\
      --tokenizer "$(absolute_path "$TOKENIZER")" \\
      --context-length "${CONTEXT_LENGTH:-256}" \\
      --construction-count "${CONSTRUCTION_COUNT:-32}" \\
      --validation-count "${VALIDATION_COUNT:-32}" \\
      --application-count "${APPLICATION_COUNT:-32}" \\
      --reserve-chunks "${RESERVE_CHUNKS:-5120}" \\
      --reserved-start "${RESERVED_START:--1}" \\
      --output "$(absolute_path "${OUTPUT:-$data_root/measurement_registry.json}")" \\
      --partition-output "$(absolute_path "${PARTITION_OUTPUT:-$data_root/partition_manifest.json}")"
    ;;
  qualify)
    cd "$repo_root"
    python3 experiments/confirm/run_program_energy_campaign.py qualify \\
      --device "${DEVICE:-cuda:0}" \\
      --output "$(absolute_path "${OUTPUT:-$data_root/q_qualification.json}")"
    ;;
  training-commands)
    : "${TOKENS:?}" "${TOKENIZER:?}" "${REGISTRY:?}"
    cd "$repo_root"
    python3 experiments/confirm/run_program_energy_campaign.py \\
      training-commands --tokens "$(absolute_path "$TOKENS")" \\
      --tokenizer "$(absolute_path "$TOKENIZER")" \\
      --registry "$(absolute_path "$REGISTRY")" \\
      --output-root "$data_root/campaigns/${CAMPAIGN_ID:-campaign}/T" \\
      --devices "${DEVICES:-cuda:0}" \\
      --output "$(absolute_path "${OUTPUT:-$data_root/training_commands.json}")"
    ;;
  gated-replication-command)
    : "${TOKENS:?}" "${TOKENIZER:?}" "${REGISTRY:?}"
    : "${Q_RECORD:?}" "${T_RECORD:?}" "${N_RECORD:?}"
    cd "$repo_root"
    python3 experiments/confirm/run_program_energy_campaign.py \\
      gated-replication-command \\
      --records "$(absolute_path "$Q_RECORD")" \\
        "$(absolute_path "$T_RECORD")" "$(absolute_path "$N_RECORD")" \\
      --tokens "$(absolute_path "$TOKENS")" \\
      --tokenizer "$(absolute_path "$TOKENIZER")" \\
      --registry "$(absolute_path "$REGISTRY")" \\
      --output-root "$data_root/campaigns/${CAMPAIGN_ID:-campaign}/T-replication" \\
      --device "${DEVICE:-cuda:0}" \\
      --output "$(absolute_path "${OUTPUT:-$data_root/gated_replication_command.json}")"
    ;;
  checkpoint-set)
    : "${ARTIFACT_ROOT:?}" "${CHECKPOINTS:?}" "${OUTPUT:?}"
    cd "$repo_root"
    set --
    for checkpoint in $CHECKPOINTS; do
      set -- "$@" --checkpoint "$checkpoint"
    done
    python3 experiments/confirm/capture_confirmation_stage.py checkpoint-set \\
      --artifact-root "$(absolute_path "$ARTIFACT_ROOT")" "$@" \\
      --output "$(absolute_path "$OUTPUT")"
    ;;
  training-record)
    : "${ARTIFACT_ROOT:?}" "${LANDMARK_MANIFEST:?}"
    : "${REPLAY_MANIFEST:?}" "${OUTPUT:?}"
    cd "$repo_root"
    python3 experiments/confirm/live_confirmation_producers.py training \\
      --artifact-root "$(absolute_path "$ARTIFACT_ROOT")" \\
      --landmark-manifest "$(absolute_path "$LANDMARK_MANIFEST")" \\
      --replay-manifest "$(absolute_path "$REPLAY_MANIFEST")" \\
      --output "$(absolute_path "$OUTPUT")"
    ;;
  normal-record)
    : "${ARTIFACT_ROOT:?}" "${SOURCE_MANIFEST:?}"
    : "${TARGET_MANIFEST:?}" "${TOKENS:?}" "${OUTPUT:?}"
    cd "$repo_root"
    python3 experiments/confirm/live_confirmation_producers.py normal \\
      --artifact-root "$(absolute_path "$ARTIFACT_ROOT")" \\
      --source-manifest "$(absolute_path "$SOURCE_MANIFEST")" \\
      --target-manifest "$(absolute_path "$TARGET_MANIFEST")" \\
      --tokens "$(absolute_path "$TOKENS")" \\
      --device "${DEVICE:-cuda:0}" --layer "${LAYER:-0}" \\
      --rows-per-split "${ROWS_PER_SPLIT:-32}" \\
      --hessian-rows "${HESSIAN_ROWS:-8}" \\
      --output "$(absolute_path "$OUTPUT")"
    ;;
  intervention-record)
    : "${ARTIFACT_ROOT:?}" "${SOURCE_MANIFEST:?}"
    : "${LOW_MANIFEST:?}" "${HIGH_MANIFEST:?}"
    : "${NORMAL_OBSERVATIONS:?}" "${TOKENS:?}" "${OUTPUT:?}"
    cd "$repo_root"
    python3 experiments/confirm/live_confirmation_producers.py intervention \\
      --artifact-root "$(absolute_path "$ARTIFACT_ROOT")" \\
      --source-manifest "$(absolute_path "$SOURCE_MANIFEST")" \\
      --low-manifest "$(absolute_path "$LOW_MANIFEST")" \\
      --high-manifest "$(absolute_path "$HIGH_MANIFEST")" \\
      --normal-observations "$(absolute_path "$NORMAL_OBSERVATIONS")" \\
      --tokens "$(absolute_path "$TOKENS")" \\
      --device "${DEVICE:-cuda:0}" \\
      --rows-per-split "${ROWS_PER_SPLIT:-32}" \\
      --output "$(absolute_path "$OUTPUT")"
    ;;
  application-record)
    : "${ARTIFACT_ROOT:?}" "${CHECKPOINT_MANIFEST:?}"
    : "${TOKENS:?}" "${OUTPUT:?}"
    cd "$repo_root"
    python3 experiments/confirm/live_confirmation_producers.py application \\
      --artifact-root "$(absolute_path "$ARTIFACT_ROOT")" \\
      --checkpoint-manifest "$(absolute_path "$CHECKPOINT_MANIFEST")" \\
      --tokens "$(absolute_path "$TOKENS")" \\
      --device "${DEVICE:-cuda:0}" \\
      --construction-pairs "${CONSTRUCTION_PAIRS:-16}" \\
      --application-pairs "${APPLICATION_PAIRS:-32}" \\
      --output "$(absolute_path "$OUTPUT")"
    ;;
  resource)
    : "${WORK_UNITS:?}" "${PEAK_GPU_BYTES:?}" "${PEAK_HOST_BYTES:?}"
    : "${RETAINED_BYTES:?}" "${WALL_SECONDS:?}" "${OUTPUT:?}"
    cd "$repo_root"
    python3 experiments/confirm/capture_confirmation_stage.py resource-metadata \\
      --work-units "$WORK_UNITS" \\
      --gpu-device-seconds "${GPU_DEVICE_SECONDS:-}" \\
      --peak-gpu-bytes-each "$PEAK_GPU_BYTES" \\
      --peak-host-bytes "$PEAK_HOST_BYTES" \\
      --retained-artifact-bytes "$RETAINED_BYTES" \\
      --wall-clock-seconds "$WALL_SECONDS" \\
      --output "$(absolute_path "$OUTPUT")"
    ;;
  replay-manifest)
    : "${ARTIFACT_ROOT:?}" "${OUTPUT:?}"
    cd "$repo_root"
    set --
    for file in ${FILES:-}; do set -- "$@" --file "$file"; done
    for pair in ${PAIRS:-}; do set -- "$@" --pair "$pair"; done
    python3 experiments/confirm/capture_confirmation_stage.py replay-manifest \\
      --artifact-root "$(absolute_path "$ARTIFACT_ROOT")" "$@" \\
      --output "$(absolute_path "$OUTPUT")"
    ;;
  seal-request)
    : "${STAGE:?}" "${CAMPAIGN_ID:?}" "${ARTIFACT_ROOT:?}"
    : "${ARTIFACTS:?}" "${OUTPUT:?}"
    cd "$repo_root"
    set --
    for artifact in $ARTIFACTS; do set -- "$@" --artifact "$artifact"; done
    python3 experiments/confirm/run_program_energy_campaign.py seal-request \\
      --stage "$STAGE" --campaign-id "$CAMPAIGN_ID" \\
      --time-semantics "${TIME_SEMANTICS:-FINITE_IMPLEMENTED_SCHEDULE}" \\
      --artifact-root "$(absolute_path "$ARTIFACT_ROOT")" "$@" \\
      --output "$(absolute_path "$OUTPUT")"
    ;;
  stage-analyze)
    : "${STAGE:?}" "${REQUEST:?}" "${ARTIFACT_ROOT:?}" "${OUTPUT:?}"
    cd "$repo_root"
    python3 experiments/confirm/run_program_energy_campaign.py stage-analyze \\
      --stage "$STAGE" --request "$(absolute_path "$REQUEST")" \\
      --artifact-root "$(absolute_path "$ARTIFACT_ROOT")" \\
      --output "$(absolute_path "$OUTPUT")"
    ;;
  campaign-analyze)
    : "${RECORDS:?}" "${OUTPUT:?}"
    cd "$repo_root"
    set --
    for record in $RECORDS; do set -- "$@" "$(absolute_path "$record")"; done
    python3 experiments/confirm/run_program_energy_campaign.py analyze \\
      --records "$@" --output "$(absolute_path "$OUTPUT")"
    ;;
  help|*)
    echo "commands: plan registry qualify training-commands gated-replication-command checkpoint-set" >&2
    echo "          training-record normal-record intervention-record" >&2
    echo "          application-record resource replay-manifest seal-request" >&2
    echo "          stage-analyze campaign-analyze" >&2
    [ "$command" = help ] || exit 2
    ;;
esac
"""


def expected_payloads():
    protocols = _protocol_payloads()
    files = dict(protocols)
    files[Path("README.md")] = README.encode("utf-8")
    files[Path("confirmation_design.json")] = (
        json.dumps(
            {
                "schema_version": "pldr-collapse-confirmation-staging-v3",
                "protocol_bundle_sha256": _bundle_digest(protocols),
                "stage_order": [stage["id"] for stage in STAGES],
                "architecture_runs": list(ARCHITECTURE_RUNS),
                "resource_caps": RESOURCE_CAPS,
                "measurement_registry_builder": (
                    "experiments/confirm/live_confirmation_producers.py"
                ),
                "scientific_artifact_producer": (
                    "experiments/confirm/live_confirmation_producers.py"
                ),
                "stage_analyzer": (
                    "experiments/analysis/"
                    "composite_confirmation_analysis.py"
                ),
                "campaign_analyzer": (
                    "experiments/analysis/"
                    "analyze_program_energy_confirmation.py"
                ),
                "arbitrary_array_ingestion": False,
                "campaign_directory": "campaigns/CAMPAIGN_ID",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    files[Path("run_next_confirmation.sh")] = WRAPPER.encode("utf-8")
    manifest = "".join(
        f"{_sha256(files[path])}  {path.as_posix()}\n"
        for path in sorted(files, key=lambda value: value.as_posix())
    )
    files[Path("STAGING_MANIFEST.sha256")] = manifest.encode("ascii")
    return files


def _existing_files(output):
    if not output.exists():
        return set()
    return {
        path.relative_to(output)
        for path in output.rglob("*")
        if path.is_file()
    }


def _is_owned_mutable_output(path):
    return path in MUTABLE_ROOT_OUTPUTS or (
        bool(path.parts) and path.parts[0] == "campaigns"
    )

def stage(output, check=False):
    output = Path(output).resolve()
    expected = expected_payloads()
    if check:
        stale = [
            relative.as_posix()
            for relative, payload in expected.items()
            if not (output / relative).is_file()
            or (output / relative).read_bytes() != payload
        ]
        unknown = sorted(
            {
                path for path in _existing_files(output)
                if path not in expected and not _is_owned_mutable_output(path)
            },
            key=lambda value: value.as_posix(),
        )
        if stale or unknown:
            details = stale + [f"unknown:{path.as_posix()}" for path in unknown]
            raise SystemExit(
                "confirmation staging is stale: " + ", ".join(details)
            )
        print(f"confirmation staging: {len(expected)} files verified")
        return
    if output.exists():
        unknown = sorted(
            {
                path for path in _existing_files(output)
                if path not in expected and not _is_owned_mutable_output(path)
            },
            key=lambda value: value.as_posix(),
        )
        if unknown:
            raise SystemExit(
                "refusing to overwrite staging with unknown files: "
                + ", ".join(path.as_posix() for path in unknown)
            )
    for relative, payload in expected.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    (output / "run_next_confirmation.sh").chmod(0o755)
    print(f"confirmation staging: wrote {len(expected)} files to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    stage(arguments.output, arguments.check)


if __name__ == "__main__":
    main()
