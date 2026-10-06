#!/usr/bin/env python3
"""Stage the deterministic gate-shape confirmation bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIRM = ROOT / "experiments" / "confirm"
sys.path.insert(0, str(CONFIRM))

from gate_shape_protocol_specs import (  # noqa: E402
    ARCHITECTURE,
    RESOURCE_CAPS,
    STAGES,
    STATUS,
)


PROTOCOL_ROOT = ROOT / "experiments/protocols/gate_shape_confirmation"
DEFAULT_OUTPUT = (
    ROOT.parent / "experiment-data/campaigns/gate-shape/confirmation-program")
MUTABLE = {Path("qualification"), Path("construction"), Path("campaigns")}


def _json(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _digest(payload):
    return hashlib.sha256(payload).hexdigest()


def _protocols():
    if not PROTOCOL_ROOT.is_dir():
        raise FileNotFoundError("run scripts/gen_gate_shape_protocols.py first")
    files = {}
    for source in sorted(PROTOCOL_ROOT.iterdir()):
        if source.is_file():
            files[Path("protocols") / source.name] = source.read_bytes()
    if not files:
        raise FileNotFoundError("gate-shape protocol directory is empty")
    return files


README = f"""# PLDR-LLM gate-shape confirmation

This is the prospective Q, T, TR, G, C, I, A, and R confirmation bundle for
the exact final-LayerNorm gate and normalized row-shape theory. The scientific
objects are 16 by 16 gate-sector loss matrices, 48 by 48 live AdamW blocks,
exact gate/shape contrast work, direct chronological products, and exact PLGA
divided differences. No parameter-normal SVD or rank-length Lanczos basis is
part of the program.

Status: `{STATUS}`. Scientific training is not authorized until the same
CPU-realized context-256 fixture passes Q on both registered devices and the
measured unit costs are entered before protocol freeze. Construction artifacts
must be sealed into a source-owned lock before validation evidence is opened.

The planned scientific subtotal is {RESOURCE_CAPS['scientific_subtotal_gpu_hours']:.2f}
GPU-hours, with {RESOURCE_CAPS['engineering_reserve_gpu_hours']:.2f} GPU-hours
of engineering reserve. The hard stops are {RESOURCE_CAPS['hard_gpu_hours']:.2f}
GPU-hours and {RESOURCE_CAPS['hard_wall_clock_hours']:.1f} wall-clock hours.

Run `./run_next_confirmation.sh help` for the single launch interface.
Mutable evidence belongs only below `qualification/`, `construction/`, and
`campaigns/CAMPAIGN_ID/`. The protocols, execution plan, readiness record, and
staging manifest are deterministic.
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
    python3 experiments/confirm/run_gate_shape_confirmation.py plan \\
      --output "$(absolute_path "${OUTPUT:-$data_root/execution_plan.runtime.json}")"
    ;;
  fixture)
    cd "$repo_root"
    python3 experiments/confirm/gate_shape_live.py fixture \\
      --output "$(absolute_path "${OUTPUT:-$data_root/qualification/fixture.npz}")"
    ;;
  qualify)
    : "${FIXTURE:?}" "${DEVICE:?}" "${OUTPUT:?}"
    cd "$repo_root"
    python3 experiments/confirm/gate_shape_live.py qualify \\
      --fixture "$(absolute_path "$FIXTURE")" --device "$DEVICE" \\
      --output "$(absolute_path "$OUTPUT")"
    ;;
  registry)
    : "${TOKENS:?}" "${TOKENIZER:?}"
    cd "$repo_root"
    python3 experiments/confirm/gate_shape_live.py registry \\
      --tokens "$(absolute_path "$TOKENS")" \\
      --tokenizer "$(absolute_path "$TOKENIZER")" \\
      --output "$(absolute_path "${OUTPUT:-$data_root/construction/registry.json}")"
    ;;
  training-commands)
    : "${TOKENS:?}" "${TOKENIZER:?}" "${REGISTRY:?}" "${CAMPAIGN_ID:?}"
    cd "$repo_root"
    python3 experiments/confirm/run_gate_shape_confirmation.py training-commands \\
      --tokens "$(absolute_path "$TOKENS")" \\
      --tokenizer "$(absolute_path "$TOKENIZER")" \\
      --registry "$(absolute_path "$REGISTRY")" \\
      --output-root "$data_root/campaigns/$CAMPAIGN_ID/T" \\
      --devices "${DEVICES:-cuda:0,cuda:1}" \\
      --output "$(absolute_path "${OUTPUT:-$data_root/campaigns/$CAMPAIGN_ID/training_commands.json}")"
    ;;
  checkpoint-set)
    : "${ARTIFACT_ROOT:?}" "${CHECKPOINTS:?}" "${OUTPUT:?}"
    cd "$repo_root"
    set --
    for item in $CHECKPOINTS; do set -- "$@" --checkpoint "$item"; done
    python3 experiments/confirm/capture_confirmation_stage.py checkpoint-set \\
      --artifact-root "$(absolute_path "$ARTIFACT_ROOT")" "$@" \\
      --output "$(absolute_path "$OUTPUT")"
    ;;
  file-set)
    : "${ARTIFACT_ROOT:?}" "${FILES:?}" "${OUTPUT:?}"
    cd "$repo_root"
    set --
    for item in $FILES; do set -- "$@" --file "$item"; done
    python3 experiments/confirm/capture_confirmation_stage.py file-set \\
      --artifact-root "$(absolute_path "$ARTIFACT_ROOT")" "$@" \\
      --output "$(absolute_path "$OUTPUT")"
    ;;
  geometry)
    : "${CHECKPOINT:?}" "${TOKENS:?}" "${REGISTRY:?}" "${ROLE:?}"
    : "${LAYER:?}" "${OUTPUT:?}"
    cd "$repo_root"
    set --
    if [ -n "${TARGET_CHECKPOINT:-}" ]; then
      set -- "$@" --target-checkpoint "$(absolute_path "$TARGET_CHECKPOINT")"
    fi
    python3 experiments/confirm/gate_shape_live.py geometry "$@" \\
      --checkpoint "$(absolute_path "$CHECKPOINT")" \\
      --tokens "$(absolute_path "$TOKENS")" \\
      --registry "$(absolute_path "$REGISTRY")" \\
      --registry-role "$ROLE" --layer "$LAYER" \\
      --device "${DEVICE:-cuda:0}" --output "$(absolute_path "$OUTPUT")"
    ;;
  intervention-geometry)
    : "${CHECKPOINT:?}" "${TOKENS:?}" "${REGISTRY:?}"
    : "${LAYER:?}" "${OUTPUT:?}"
    cd "$repo_root"
    python3 experiments/confirm/gate_shape_live.py intervention-geometry \\
      --checkpoint "$(absolute_path "$CHECKPOINT")" \\
      --tokens "$(absolute_path "$TOKENS")" \\
      --registry "$(absolute_path "$REGISTRY")" --layer "$LAYER" \\
      --device "${DEVICE:-cuda:0}" --output "$(absolute_path "$OUTPUT")"
    ;;
  capture-block)
    : "${GEOMETRY:?}" "${SNAPSHOT:?}" "${OUTPUT:?}"
    cd "$repo_root"
    python3 experiments/confirm/gate_shape_live.py capture-block \\
      --geometry "$(absolute_path "$GEOMETRY")" \\
      --snapshot "$(absolute_path "$SNAPSHOT")" \\
      --output "$(absolute_path "$OUTPUT")"
    ;;
  energy-step)
    : "${SOURCE:?}" "${TARGET:?}" "${OUTPUT:?}"
    cd "$repo_root"
    python3 experiments/confirm/gate_shape_live.py energy-step \\
      --source "$(absolute_path "$SOURCE")" \\
      --target "$(absolute_path "$TARGET")" \\
      --output "$(absolute_path "$OUTPUT")"
    ;;
  application)
    : "${CHECKPOINT:?}" "${TOKENS:?}" "${REGISTRY:?}" "${ROLE:?}"
    : "${PAIR_ID:?}" "${LAYER:?}" "${HEAD:?}" "${OUTPUT:?}"
    cd "$repo_root"
    python3 experiments/confirm/gate_shape_application.py \\
      --checkpoint "$(absolute_path "$CHECKPOINT")" \\
      --tokens "$(absolute_path "$TOKENS")" \\
      --registry "$(absolute_path "$REGISTRY")" \\
      --registry-role "$ROLE" --pair-id "$PAIR_ID" \\
      --layer "$LAYER" --head "$HEAD" --device "${DEVICE:-cuda:0}" \\
      --output "$(absolute_path "$OUTPUT")"
    ;;
  intervention-commands)
    : "${SOURCE_CHECKPOINT:?}" "${TOKENS:?}" "${TOKENIZER:?}"
    : "${REGISTRY:?}" "${CAMPAIGN_ID:?}"
    cd "$repo_root"
    python3 experiments/confirm/run_gate_shape_confirmation.py intervention-commands \\
      --source-checkpoint "$(absolute_path "$SOURCE_CHECKPOINT")" \\
      --tokens "$(absolute_path "$TOKENS")" \\
      --tokenizer "$(absolute_path "$TOKENIZER")" \\
      --registry "$(absolute_path "$REGISTRY")" \\
      --device "${DEVICE:-cuda:0}" \\
      --output-root "$data_root/campaigns/$CAMPAIGN_ID/I" \\
      --output "$(absolute_path "${OUTPUT:-$data_root/campaigns/$CAMPAIGN_ID/intervention_commands.json}")"
    ;;
  lock-construction)
    : "${CAMPAIGN_ID:?}" "${ARTIFACT_ROOT:?}" "${CAPTURE_SET:?}"
    : "${REGISTRY:?}" "${OUTPUT:?}"
    cd "$repo_root"
    python3 experiments/confirm/run_gate_shape_confirmation.py lock-construction \\
      --campaign-id "$CAMPAIGN_ID" \\
      --artifact-root "$(absolute_path "$ARTIFACT_ROOT")" \\
      --capture-set "$(absolute_path "$CAPTURE_SET")" \\
      --registry "$(absolute_path "$REGISTRY")" \\
      --safety-factor "${SAFETY_FACTOR:-1.10}" \\
      --output "$(absolute_path "$OUTPUT")"
    ;;
  seal-stage)
    : "${STAGE:?}" "${CAMPAIGN_ID:?}" "${ARTIFACT_ROOT:?}"
    : "${ARTIFACTS:?}" "${OUTPUT:?}"
    cd "$repo_root"
    set --
    for item in $ARTIFACTS; do set -- "$@" --artifact "$item"; done
    python3 experiments/confirm/run_gate_shape_confirmation.py seal-stage \\
      --stage "$STAGE" --campaign-id "$CAMPAIGN_ID" \\
      --artifact-root "$(absolute_path "$ARTIFACT_ROOT")" "$@" \\
      --output "$(absolute_path "$OUTPUT")"
    ;;
  analyze-stage)
    : "${REQUEST:?}" "${OUTPUT:?}"
    cd "$repo_root"
    python3 experiments/analysis/analyze_gate_shape_confirmation.py stage \\
      --request "$(absolute_path "$REQUEST")" \\
      --output "$(absolute_path "$OUTPUT")"
    ;;
  analyze-campaign)
    : "${REQUESTS:?}" "${OUTPUT:?}"
    cd "$repo_root"
    set --
    for item in $REQUESTS; do
      set -- "$@" --request "$(absolute_path "$item")"
    done
    python3 experiments/analysis/analyze_gate_shape_confirmation.py campaign \\
      "$@" --output "$(absolute_path "$OUTPUT")"
    ;;
  help|*)
    echo "commands: plan fixture qualify registry training-commands" >&2
    echo "          checkpoint-set file-set geometry intervention-geometry" >&2
    echo "          capture-block energy-step" >&2
    echo "          application intervention-commands lock-construction" >&2
    echo "          seal-stage analyze-stage analyze-campaign" >&2
    [ "$command" = help ] || exit 2
    ;;
esac
"""


def expected_files():
    files = _protocols()
    files[Path("README.md")] = README.encode("utf-8")
    files[Path("execution_plan.json")] = _json({
        "schema_version": "pldr-gate-shape-execution-plan-v1",
        "status": STATUS,
        "architecture": ARCHITECTURE,
        "resources": RESOURCE_CAPS,
        "stage_order": [stage["id"] for stage in STAGES],
        "stages": list(STAGES),
    })
    files[Path("READINESS.json")] = _json({
        "schema_version": "pldr-gate-shape-readiness-v1",
        "status": "QUALIFICATION_REQUIRED",
        "scientific_execution_authorized": False,
        "reason": (
            "The same CPU-realized context-256 fixture has not yet passed "
            "sealed Q analysis on both registered devices."),
    })
    files[Path("run_next_confirmation.sh")] = WRAPPER.encode("utf-8")
    manifest = "".join(
        f"{_digest(files[path])}  {path.as_posix()}\n"
        for path in sorted(files, key=lambda item: item.as_posix()))
    files[Path("STAGING_MANIFEST.sha256")] = manifest.encode("ascii")
    return files


def _immutable_existing(output):
    if not output.exists():
        return set()
    return {
        path.relative_to(output)
        for path in output.rglob("*") if path.is_file()
        and not any(path.relative_to(output).is_relative_to(root) for root in MUTABLE)
    }


def stage(output, check=False):
    output = Path(output).resolve()
    expected = expected_files()
    actual = _immutable_existing(output)
    stale = [
        path for path, payload in expected.items()
        if not (output / path).is_file() or (output / path).read_bytes() != payload]
    unknown = actual - set(expected)
    if check:
        if stale or unknown:
            raise SystemExit(
                "gate-shape staging is stale: "
                + ", ".join(sorted(
                    [path.as_posix() for path in stale]
                    + [f"unknown:{path.as_posix()}" for path in unknown])))
        print(f"gate-shape staging: verified ({len(expected)} immutable files)")
        return
    if unknown:
        raise SystemExit(
            "refusing to overwrite unknown immutable staging files: "
            + ", ".join(sorted(path.as_posix() for path in unknown)))
    output.mkdir(parents=True, exist_ok=True)
    for relative, payload in expected.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    (output / "run_next_confirmation.sh").chmod(0o755)
    for relative in MUTABLE:
        (output / relative).mkdir(parents=True, exist_ok=True)
    print(f"gate-shape staging: wrote {len(expected)} immutable files to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    stage(arguments.output, arguments.check)


if __name__ == "__main__":
    main()
