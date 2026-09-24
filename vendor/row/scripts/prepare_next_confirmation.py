#!/usr/bin/env python3
"""Stage the normal-stability confirmation campaign in experiment-data."""

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
    ROOT.parent / "experiment-data" / "manuscript-revisions" / "rev32"
    / "confirmation-program"
)


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _protocol_payloads():
    if not PROTOCOL_SOURCE.is_dir():
        raise FileNotFoundError(
            "generate normal-stability protocols before staging the campaign")
    names = (
        ["README.md", "campaign_design.json", "request_schema.json",
         "resource_metadata_schema.json", "stage_record_schema.json",
         "raw_observation_schemas.json"]
        + [f"{stage['id'].lower()}_protocol.json" for stage in STAGES]
    )
    paths = [PROTOCOL_SOURCE / name for name in names]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "normal-stability protocol bundle is incomplete: "
            + ", ".join(missing))
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

This immutable bundle owns Q, T, N, I, A, and R: all-row qualification,
complete-state training, projected normal windows, corridor interventions,
the same-source application bridge, and independent replay. Hard limits are
32 GPU-hours, two GPUs, and 24 wall-clock hours.

Requests bind raw observations, complete checkpoint sets, disjoint partitions,
code, dataset, tokenizer, device, and dtype. The analyzer accepts no caller
decision, comparison matrix, force bound, threshold, or free-form scientific
measurement.

Start with plan, qualify, and training-commands. Then use checkpoint-set,
partition, resource, pack-observations, seal-request, stage-analyze, and
campaign-analyze. Campaign outputs belong below campaigns/CAMPAIGN_ID.
Protocol and staging manifest files remain immutable.
"""


WRAPPER = """#!/bin/sh
set -eu

repo_root=/pldr-work/row-code
data_root=/pldr-data/row/rev32/confirmation-program
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
  qualify)
    cd "$repo_root"
    python3 experiments/confirm/run_program_energy_campaign.py qualify \\
      --device "${DEVICE:-cpu}" \\
      --output "$(absolute_path "${OUTPUT:-$data_root/q_qualification.json}")"
    ;;
  training-commands)
    : "${TOKENS:?}" "${TOKENIZER:?}"
    cd "$repo_root"
    python3 experiments/confirm/run_program_energy_campaign.py \\
      training-commands --tokens "$(absolute_path "$TOKENS")" \\
      --tokenizer "$(absolute_path "$TOKENIZER")" \\
      --output-root "$data_root/campaigns/${CAMPAIGN_ID:-campaign}/T" \\
      --devices "${DEVICES:-cuda:0,cuda:1}" \\
      --output "$(absolute_path "${OUTPUT:-$data_root/training_commands.json}")"
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
  partition)
    : "${DATASET:?}" "${TOKENIZER:?}" "${CONSTRUCTION_IDS:?}"
    : "${VALIDATION_IDS:?}" "${APPLICATION_IDS:?}" "${OUTPUT:?}"
    cd "$repo_root"
    python3 experiments/confirm/capture_confirmation_stage.py \\
      partition-manifest --dataset "$(absolute_path "$DATASET")" \\
      --tokenizer "$(absolute_path "$TOKENIZER")" \\
      --construction-ids "$CONSTRUCTION_IDS" \\
      --validation-ids "$VALIDATION_IDS" \\
      --application-ids "$APPLICATION_IDS" \\
      --output "$(absolute_path "$OUTPUT")"
    ;;
  resource)
    : "${WORK_UNITS:?}" "${PEAK_GPU_BYTES:?}" "${PEAK_HOST_BYTES:?}"
    : "${RETAINED_BYTES:?}" "${WALL_SECONDS:?}" "${OUTPUT:?}"
    cd "$repo_root"
    python3 experiments/confirm/capture_confirmation_stage.py \\
      resource-metadata --work-units "$WORK_UNITS" \\
      --gpu-device-seconds "${GPU_DEVICE_SECONDS:-}" \\
      --peak-gpu-bytes-each "$PEAK_GPU_BYTES" \\
      --peak-host-bytes "$PEAK_HOST_BYTES" \\
      --retained-artifact-bytes "$RETAINED_BYTES" \\
      --wall-clock-seconds "$WALL_SECONDS" \\
      --output "$(absolute_path "$OUTPUT")"
    ;;
  pack-observations)
    : "${STAGE:?}" "${ARRAYS:?}" "${OUTPUT:?}"
    cd "$repo_root"
    set --
    for array in $ARRAYS; do
      set -- "$@" --array "$array"
    done
    python3 experiments/confirm/capture_confirmation_stage.py \\
      pack-observations --stage "$STAGE" "$@" \\
      --output "$(absolute_path "$OUTPUT")"
    ;;
  seal-request)
    : "${STAGE:?}" "${CAMPAIGN_ID:?}" "${ARTIFACT_ROOT:?}"
    : "${ARTIFACTS:?}" "${OUTPUT:?}"
    cd "$repo_root"
    set --
    for artifact in $ARTIFACTS; do
      set -- "$@" --artifact "$artifact"
    done
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
    for record in $RECORDS; do
      set -- "$@" "$(absolute_path "$record")"
    done
    python3 experiments/confirm/run_program_energy_campaign.py analyze \\
      --records "$@" --output "$(absolute_path "$OUTPUT")"
    ;;
  help|*)
    echo "commands: plan qualify training-commands checkpoint-set partition" >&2
    echo "          resource pack-observations seal-request stage-analyze" >&2
    echo "          campaign-analyze" >&2
    [ "$command" = help ] || exit 2
    ;;
esac
"""


def expected_payloads():
    protocols = _protocol_payloads()
    files = dict(protocols)
    files[Path("README.md")] = README.encode("utf-8")
    files[Path("confirmation_design.json")] = (
        json.dumps({
            "schema_version": "pldr-collapse-confirmation-staging-v2",
            "protocol_bundle_sha256": _bundle_digest(protocols),
            "stage_order": [stage["id"] for stage in STAGES],
            "architecture_runs": list(ARCHITECTURE_RUNS),
            "resource_caps": RESOURCE_CAPS,
            "artifact_builder":
                "experiments/confirm/capture_confirmation_stage.py",
            "stage_analyzer":
                "experiments/analysis/confirmation_stage_analysis.py",
            "campaign_analyzer":
                "experiments/analysis/analyze_program_energy_confirmation.py",
            "campaign_directory": "campaigns/CAMPAIGN_ID",
        }, indent=2, sort_keys=True) + chr(10)
    ).encode("utf-8")
    files[Path("run_next_confirmation.sh")] = WRAPPER.encode("utf-8")
    manifest = "".join(
        f"{_sha256(files[path])}  {path.as_posix()}" + chr(10)
        for path in sorted(files, key=lambda value: value.as_posix())
    )
    files[Path("STAGING_MANIFEST.sha256")] = manifest.encode("ascii")
    return files


def stage(output, check=False):
    output = Path(output).resolve()
    expected = expected_payloads()
    if check:
        stale = [
            relative.as_posix() for relative, payload in expected.items()
            if not (output / relative).is_file()
            or (output / relative).read_bytes() != payload
        ]
        if stale:
            raise SystemExit(
                "confirmation staging is stale: " + ", ".join(stale))
        print(f"confirmation staging: {len(expected)} files verified")
        return
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
