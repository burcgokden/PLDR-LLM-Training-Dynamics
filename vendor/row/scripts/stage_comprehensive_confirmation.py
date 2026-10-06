#!/usr/bin/env python3
"""Stage the immutable comprehensive-collapse confirmation bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIRM = ROOT / "experiments" / "confirm"
sys.path.insert(0, str(CONFIRM))

from comprehensive_protocol_specs import (  # noqa: E402
    ARCHITECTURE,
    RESOURCE_CAPS,
    STAGES,
)


PROTOCOL_ROOT = (
    ROOT / "experiments" / "protocols" / "comprehensive_collapse_confirmation")
DEFAULT_OUTPUT = (
    ROOT.parent / "experiment-data" / "campaigns" / "comprehensive-confirmation"
    / "confirmation-program")
MUTABLE = {Path("qualification"), Path("construction"), Path("campaigns")}


def _digest(payload):
    return hashlib.sha256(payload).hexdigest()


def _json(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _protocol_payloads():
    if not PROTOCOL_ROOT.is_dir():
        raise FileNotFoundError(
            "run scripts/gen_comprehensive_protocols.py before staging")
    files = {}
    for source in sorted(PROTOCOL_ROOT.iterdir()):
        if source.is_file():
            files[Path("protocols") / source.name] = source.read_bytes()
    if not files:
        raise FileNotFoundError("the comprehensive protocol bundle is empty")
    return files


README = f"""# PLDR-LLM comprehensive collapse confirmation

This immutable staging bundle owns the next Q, T, TR, N, D, I, A, S2N, and
R confirmation stages. The theorem object is the realizable parameter normal
of the finite row-map stencil. The loss object is the selected pairwise frame
and signed true cross-entropy response pulled through centered vocabulary
logits. The optimizer object is the full live AdamW successor on position,
first moment, and second moment.

The planned resource total is {RESOURCE_CAPS['planned_gpu_hours']:.2f}
GPU-hours, the engineering stop is {RESOURCE_CAPS['hard_gpu_hours']:.1f}
GPU-hours, and the wall-clock stop is
{RESOURCE_CAPS['hard_wall_clock_hours']:.1f} hours. Scientific stages are
locked until both target devices pass Q and one complete context-256
parameter-normal engineering run stays below the 20 GiB allocation cap.

Run `./run_next_confirmation.sh help` for the single generated launch path.
The `intervention-commands` entry creates both source-bound 16-update rate
arms from one complete checkpoint; the `intervention` entry measures their
full finite-stencil SVD normal coordinates. Use `snapshot-set` to bind the
exact early and late optimizer files before sealing D evidence. Before
sealing predictions, run `application` with
`REGISTRY_ROLE=construction` and pass that archive as `APPLICATION` to
`construction-summary`. Validation application runs require the resulting
prediction lock and use `REGISTRY_ROLE=validation`. Seal every completed
construction summary with `seal-construction`, then pass those request paths
to `lock-predictions`. Exact stage evidence-role names are listed in each
generated protocol. Raw campaigns belong below
`campaigns/CAMPAIGN_ID`. Protocols, the execution plan, readiness record, and
staging manifest are immutable.
"""


WRAPPER = """#!/bin/sh
set -eu

repo_root="${PLDR_ROW_CODE_ROOT:?Set PLDR_ROW_CODE_ROOT to the public vendor/row directory}"
data_root="${PLDR_ROW_RUN_ROOT:?Set PLDR_ROW_RUN_ROOT to the campaign directory}"
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
    python3 experiments/confirm/run_comprehensive_confirmation.py plan \\
      --output "$(absolute_path "${OUTPUT:-$data_root/execution_plan.runtime.json}")"
    ;;
  registry)
    : "${TOKENS:?}" "${TOKENIZER:?}"
    cd "$repo_root"
    python3 experiments/confirm/comprehensive_live_producers.py registry \\
      --tokens "$(absolute_path "$TOKENS")" \\
      --tokenizer "$(absolute_path "$TOKENIZER")" \\
      --output "$(absolute_path "${OUTPUT:-$data_root/construction/registry.json}")"
    ;;
  qualify)
    : "${DEVICE:?}" "${OUTPUT:?}"
    cd "$repo_root"
    python3 experiments/confirm/comprehensive_live_producers.py qualify \\
      --device "$DEVICE" --output "$(absolute_path "$OUTPUT")"
    ;;
  normal-engineering)
    : "${CHECKPOINT:?}" "${TOKENS:?}" "${REGISTRY:?}" "${OUTPUT_ROOT:?}"
    cd "$repo_root"
    python3 experiments/confirm/comprehensive_live_producers.py normal \\
      --checkpoint "$(absolute_path "$CHECKPOINT")" \\
      --tokens "$(absolute_path "$TOKENS")" \\
      --registry "$(absolute_path "$REGISTRY")" \\
      --device "${DEVICE:-cuda:0}" --engineering-only \\
      --context-limit 1 --action-contexts 1 --krylov-iterations 8 \\
      --output-root "$(absolute_path "$OUTPUT_ROOT")"
    ;;
  normal-construction)
    : "${CHECKPOINT:?}" "${TOKENS:?}" "${REGISTRY:?}" "${OUTPUT_ROOT:?}"
    cd "$repo_root"
    python3 experiments/confirm/comprehensive_live_producers.py normal \\
      --checkpoint "$(absolute_path "$CHECKPOINT")" \\
      --tokens "$(absolute_path "$TOKENS")" \\
      --registry "$(absolute_path "$REGISTRY")" \\
      --device "${DEVICE:-cuda:0}" --registry-role construction \\
      --output-root "$(absolute_path "$OUTPUT_ROOT")"
    ;;
  normal-validation)
    : "${CHECKPOINT:?}" "${TOKENS:?}" "${REGISTRY:?}" "${OUTPUT_ROOT:?}"
    : "${PAIR_PRODUCT_LOWER:?}"
    cd "$repo_root"
    python3 experiments/confirm/comprehensive_live_producers.py normal \\
      --checkpoint "$(absolute_path "$CHECKPOINT")" \\
      --tokens "$(absolute_path "$TOKENS")" \\
      --registry "$(absolute_path "$REGISTRY")" \\
      --device "${DEVICE:-cuda:0}" --registry-role validation \\
      --pair-product-lower "$PAIR_PRODUCT_LOWER" \\
      --output-root "$(absolute_path "$OUTPUT_ROOT")"
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
  snapshot-set)
    : "${ARTIFACT_ROOT:?}" "${SNAPSHOTS:?}" "${OUTPUT:?}"
    cd "$repo_root"
    set --
    for snapshot in $SNAPSHOTS; do
      set -- "$@" --file "$snapshot"
    done
    python3 experiments/confirm/capture_confirmation_stage.py file-set \
      --artifact-root "$(absolute_path "$ARTIFACT_ROOT")" "$@" \
      --output "$(absolute_path "$OUTPUT")"
    ;;
  trajectory)
    : "${CHECKPOINT_MANIFEST:?}" "${ARTIFACT_ROOT:?}" "${TOKENS:?}"
    : "${REGISTRY:?}" "${OUTPUT:?}"
    cd "$repo_root"
    python3 experiments/confirm/comprehensive_live_producers.py trajectory \\
      --checkpoint-manifest "$(absolute_path "$CHECKPOINT_MANIFEST")" \\
      --artifact-root "$(absolute_path "$ARTIFACT_ROOT")" \\
      --tokens "$(absolute_path "$TOKENS")" \\
      --registry "$(absolute_path "$REGISTRY")" \\
      --device "${DEVICE:-cuda:0}" --output "$(absolute_path "$OUTPUT")"
    ;;
  dynamics)
    : "${SNAPSHOTS:?}" "${NORMAL_METADATA:?}" "${NORMAL_ACTIONS:?}"
    : "${NORMAL_LEFT_BASIS:?}" "${SINGULAR_VALUES:?}"
    : "${DIRECT_TRAJECTORY:?}" "${OUTPUT:?}"
    cd "$repo_root"
    set --
    for snapshot in $SNAPSHOTS; do
      set -- "$@" --snapshot "$(absolute_path "$snapshot")"
    done
    if [ -n "${PREDICTION_LOCK:-}" ]; then
      set -- "$@" --prediction-lock "$(absolute_path "$PREDICTION_LOCK")"
    fi
    python3 experiments/confirm/comprehensive_live_producers.py dynamics \\
      "$@" --normal-metadata "$(absolute_path "$NORMAL_METADATA")" \\
      --normal-actions "$(absolute_path "$NORMAL_ACTIONS")" \\
      --normal-left-basis "$(absolute_path "$NORMAL_LEFT_BASIS")" \\
      --singular-values "$(absolute_path "$SINGULAR_VALUES")" \\
      --direct-trajectory "$(absolute_path "$DIRECT_TRAJECTORY")" \\
      --tensor-group "${TENSOR_GROUP:-phi}" \\
      --output "$(absolute_path "$OUTPUT")"
    ;;
  intervention)
    : "${SOURCE_MANIFEST:?}" "${LOW_MANIFEST:?}" "${HIGH_MANIFEST:?}"
    : "${ARTIFACT_ROOT:?}" "${TOKENS:?}" "${REGISTRY:?}" "${OUTPUT:?}"
    : "${NORMAL_METADATA:?}" "${NORMAL_LEFT_BASIS:?}" "${SINGULAR_VALUES:?}"
    cd "$repo_root"
    python3 experiments/confirm/comprehensive_live_producers.py intervention \\
      --source-manifest "$(absolute_path "$SOURCE_MANIFEST")" \\
      --low-manifest "$(absolute_path "$LOW_MANIFEST")" \\
      --high-manifest "$(absolute_path "$HIGH_MANIFEST")" \\
      --artifact-root "$(absolute_path "$ARTIFACT_ROOT")" \\
      --tokens "$(absolute_path "$TOKENS")" \\
      --registry "$(absolute_path "$REGISTRY")" \\
      --normal-metadata "$(absolute_path "$NORMAL_METADATA")" \\
      --normal-left-basis "$(absolute_path "$NORMAL_LEFT_BASIS")" \\
      --singular-values "$(absolute_path "$SINGULAR_VALUES")" \\
      --device "${DEVICE:-cuda:0}" --output "$(absolute_path "$OUTPUT")"
    ;;
  application)
    : "${CHECKPOINT_MANIFEST:?}" "${ARTIFACT_ROOT:?}" "${TOKENS:?}"
    : "${REGISTRY:?}" "${OUTPUT:?}"
    cd "$repo_root"
    set --
    if [ -n "${PREDICTION_LOCK:-}" ]; then
      set -- "$@" --prediction-lock "$(absolute_path "$PREDICTION_LOCK")"
    fi
    python3 experiments/confirm/comprehensive_live_producers.py application "$@" \\
      --checkpoint-manifest "$(absolute_path "$CHECKPOINT_MANIFEST")" \\
      --artifact-root "$(absolute_path "$ARTIFACT_ROOT")" \\
      --tokens "$(absolute_path "$TOKENS")" \\
      --registry "$(absolute_path "$REGISTRY")" \\
      --device "${DEVICE:-cuda:0}" \\
      --registry-role "${REGISTRY_ROLE:-validation}" \\
      --subdivisions "${SUBDIVISIONS:-8}" \\
      --output "$(absolute_path "$OUTPUT")"
    ;;
  construction-summary)
    : "${REGISTRY:?}" "${NORMAL_METADATA:?}" "${NORMAL_ACTIONS:?}"
    : "${DYNAMICS:?}" "${INTERVENTION:?}" "${APPLICATION:?}" "${OUTPUT:?}"
    cd "$repo_root"
    python3 experiments/confirm/comprehensive_live_producers.py \\
      construction-summary --registry "$(absolute_path "$REGISTRY")" \\
      --normal-metadata "$(absolute_path "$NORMAL_METADATA")" \\
      --normal-actions "$(absolute_path "$NORMAL_ACTIONS")" \\
      --dynamics "$(absolute_path "$DYNAMICS")" \\
      --intervention "$(absolute_path "$INTERVENTION")" \\
      --application "$(absolute_path "$APPLICATION")" \\
      --safety-factor "${SAFETY_FACTOR:-0.9}" \\
      --application-subdivisions "${SUBDIVISIONS:-8}" \\
      --output "$(absolute_path "$OUTPUT")"
    ;;
  training-commands)
    : "${TOKENS:?}" "${TOKENIZER:?}" "${REGISTRY:?}" "${CAMPAIGN_ID:?}"
    cd "$repo_root"
    python3 experiments/confirm/run_comprehensive_confirmation.py training-commands \\
      --tokens "$(absolute_path "$TOKENS")" \\
      --tokenizer "$(absolute_path "$TOKENIZER")" \\
      --registry "$(absolute_path "$REGISTRY")" \\
      --output-root "$data_root/campaigns/$CAMPAIGN_ID/T" \\
      --devices "${DEVICES:-cuda:0,cuda:1}" \\
      --output "$(absolute_path "${OUTPUT:-$data_root/campaigns/$CAMPAIGN_ID/training_commands.json}")"
    ;;
  intervention-commands)
    : "${SOURCE_CHECKPOINT:?}" "${TOKENS:?}" "${TOKENIZER:?}"
    : "${REGISTRY:?}" "${CAMPAIGN_ID:?}"
    cd "$repo_root"
    python3 experiments/confirm/run_comprehensive_confirmation.py \\
      intervention-commands \\
      --source-checkpoint "$(absolute_path "$SOURCE_CHECKPOINT")" \\
      --tokens "$(absolute_path "$TOKENS")" \\
      --tokenizer "$(absolute_path "$TOKENIZER")" \\
      --registry "$(absolute_path "$REGISTRY")" \\
      --output-root "$data_root/campaigns/$CAMPAIGN_ID/I" \\
      --device "${DEVICE:-cuda:0}" \\
      --rate-multipliers "${RATE_MULTIPLIERS:-0.75,1.25}" \\
      --output "$(absolute_path "${OUTPUT:-$data_root/campaigns/$CAMPAIGN_ID/intervention_commands.json}")"
    ;;
  seal-construction)
    : "${CAMPAIGN_ID:?}" "${ARTIFACT_ROOT:?}" "${SUMMARY:?}" "${OUTPUT:?}"
    cd "$repo_root"
    python3 experiments/confirm/run_comprehensive_confirmation.py seal-construction \\
      --campaign-id "$CAMPAIGN_ID" \\
      --artifact-root "$(absolute_path "$ARTIFACT_ROOT")" \\
      --summary "$SUMMARY" --output "$(absolute_path "$OUTPUT")"
    ;;
  seal-request)
    : "${STAGE:?}" "${CAMPAIGN_ID:?}" "${ARTIFACT_ROOT:?}" "${ARTIFACTS:?}" "${OUTPUT:?}"
    cd "$repo_root"
    set --
    for artifact in $ARTIFACTS; do set -- "$@" --artifact "$artifact"; done
    python3 experiments/confirm/run_comprehensive_confirmation.py seal-request \\
      --stage "$STAGE" --campaign-id "$CAMPAIGN_ID" \\
      --artifact-root "$(absolute_path "$ARTIFACT_ROOT")" "$@" \\
      --output "$(absolute_path "$OUTPUT")"
    ;;
  lock-predictions)
    : "${CAMPAIGN_ID:?}" "${ARTIFACT_ROOT:?}" "${CONSTRUCTION_REQUESTS:?}" "${OUTPUT:?}"
    cd "$repo_root"
    set --
    for request in $CONSTRUCTION_REQUESTS; do set -- "$@" --request "$request"; done
    python3 experiments/confirm/run_comprehensive_confirmation.py lock-predictions \\
      --campaign-id "$CAMPAIGN_ID" \\
      --artifact-root "$(absolute_path "$ARTIFACT_ROOT")" "$@" \\
      --output "$(absolute_path "$OUTPUT")"
    ;;
  stage-analyze)
    : "${REQUEST:?}" "${OUTPUT:?}"
    cd "$repo_root"
    python3 experiments/analysis/comprehensive_confirmation_analysis.py stage \\
      --request "$(absolute_path "$REQUEST")" \\
      --output "$(absolute_path "$OUTPUT")"
    ;;
  campaign-analyze)
    : "${REQUESTS:?}" "${OUTPUT:?}"
    cd "$repo_root"
    set --
    for request in $REQUESTS; do set -- "$@" --request "$(absolute_path "$request")"; done
    python3 experiments/analysis/comprehensive_confirmation_analysis.py campaign \\
      "$@" --output "$(absolute_path "$OUTPUT")"
    ;;
  help|*)
    echo "commands: plan registry qualify normal-engineering" >&2
    echo "          normal-construction normal-validation checkpoint-set" >&2
    echo "          snapshot-set trajectory dynamics intervention" >&2
    echo "          application" >&2
    echo "          construction-summary training-commands" >&2
    echo "          intervention-commands seal-construction" >&2
    echo "          seal-request lock-predictions" >&2
    echo "          stage-analyze campaign-analyze" >&2
    [ "$command" = help ] || exit 2
    ;;
esac
"""


def expected_files():
    protocols = _protocol_payloads()
    files = dict(protocols)
    files[Path("README.md")] = README.encode("utf-8")
    files[Path("execution_plan.json")] = _json({
        "schema_version": "pldr-comprehensive-execution-plan-v4",
        "architecture": ARCHITECTURE,
        "resources": RESOURCE_CAPS,
        "stage_order": [stage["id"] for stage in STAGES],
        "stages": list(STAGES),
        "scientific_launch_gate": [
            "sealed Q record for cuda:0", "sealed Q record for cuda:1",
            "context-256 parameter-normal engineering record below 20 GiB",
            "numeric prediction lock derived from construction artifacts",
        ],
    })
    files[Path("READINESS.json")] = _json({
        "schema_version": "pldr-comprehensive-readiness-v4",
        "status": "QUALIFICATION_REQUIRED",
        "scientific_execution_authorized": False,
        "reason": (
            "Q on both devices, one complete context-256 parameter-normal "
            "engineering run, and the numeric prediction lock do not yet exist."
        ),
    })
    files[Path("run_next_confirmation.sh")] = WRAPPER.encode("utf-8")
    manifest = "".join(
        f"{_digest(files[path])}  {path.as_posix()}\n"
        for path in sorted(files, key=lambda item: item.as_posix()))
    files[Path("STAGING_MANIFEST.sha256")] = manifest.encode("ascii")
    return files


def _existing(output):
    if not output.exists():
        return set()
    return {
        path.relative_to(output)
        for path in output.rglob("*") if path.is_file()
    }


def _mutable(path):
    return bool(path.parts) and Path(path.parts[0]) in MUTABLE


def stage(output, check=False):
    output = Path(output).resolve()
    expected = expected_files()
    actual = _existing(output)
    unknown = sorted(
        (path for path in actual - set(expected) if not _mutable(path)),
        key=lambda item: item.as_posix())
    stale = [
        path.as_posix() for path, payload in expected.items()
        if not (output / path).is_file() or (output / path).read_bytes() != payload
    ]
    if check:
        if stale or unknown:
            raise SystemExit(
                "comprehensive staging is stale: "
                + ", ".join(stale + [f"unknown:{path}" for path in unknown]))
        print(f"comprehensive staging: verified ({len(expected)} files)")
        return
    if unknown:
        raise SystemExit(
            "refusing to overwrite staging with unknown files: "
            + ", ".join(path.as_posix() for path in unknown))
    output.mkdir(parents=True, exist_ok=True)
    for relative, payload in expected.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    (output / "run_next_confirmation.sh").chmod(0o755)
    print(f"comprehensive staging: wrote {len(expected)} files to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    stage(arguments.output, arguments.check)


if __name__ == "__main__":
    main()
