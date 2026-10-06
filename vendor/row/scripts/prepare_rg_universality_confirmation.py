#!/usr/bin/env python3
"""Stage the separate RG-universality campaign in experiment-data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIRM = ROOT / "experiments" / "confirm"
sys.path.insert(0, str(CONFIRM))

from rg_universality_protocol_specs import (  # noqa: E402
    ARCHITECTURE_RUNS,
    RESOURCE_CAPS,
    STAGES,
)


SOURCE = (
    ROOT / "experiments" / "protocols" / "rg_universality_confirmation"
)
DEFAULT_OUTPUT = (
    ROOT.parent / "experiment-data" / "campaigns" / "rg-universality"
    / "rg-universality-program"
)


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _payloads():
    names = (
        ["README.md", "campaign_design.json", "trajectory_artifact_schema.json",
         "analysis_record_schema.json"]
        + [f"{stage['id'].lower()}_protocol.json" for stage in STAGES]
    )
    paths = [SOURCE / name for name in names]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "generate RG-universality protocols first: "
            + ", ".join(missing))
    return {
        Path("protocols") / path.name: path.read_bytes()
        for path in sorted(paths)
    }


README = """# PLDR transition-kernel RG campaign staging

This is the separate, longer two-GPU program for the physical transition-
kernel RG, continuous-flow qualification, and gapped Gaussian normal-
relaxation class. It does not delay the normal-stability confirmation
program.

Commands:

    ./run_rg_confirmation.sh plan
    ./run_rg_confirmation.sh qualify
    MODE=kernel-flow INPUTS="system-a.npz system-b.npz" OUTPUT=analysis.json \
      ./run_rg_confirmation.sh analyze

Use MODE=universality only after R2 and R3 pass and all six preregistered
architecture-seed trajectory artifacts are present.
"""


WRAPPER = """#!/bin/sh
set -eu

repo_root="${PLDR_ROW_CODE_ROOT:?Set PLDR_ROW_CODE_ROOT to the public vendor/row directory}"
data_root="${PLDR_ROW_RUN_ROOT:?Set PLDR_ROW_RUN_ROOT to the campaign directory}"
launch_dir=__(pwd)
command=__{1-}

absolute_path() {
  case "__1" in
    /*) printf '%s\n' "__1" ;;
    *) printf '%s/%s\n' "__launch_dir" "__1" ;;
  esac
}

case "__command" in
  plan)
    cd "__repo_root"
    python3 experiments/confirm/run_rg_universality_campaign.py plan \
      --output "__(absolute_path "__{OUTPUT:-__data_root/execution_plan.json}")"
    ;;
  qualify)
    cd "__repo_root"
    PYTHONPATH=experiments/confirm python3 \
      experiments/confirm/run_rg_universality_campaign.py qualify \
      --output "__(absolute_path "__{OUTPUT:-__data_root/r0_qualification.json}")"
    ;;
  analyze)
    : "__{MODE:?}" "__{INPUTS:?}" "__{OUTPUT:?}"
    cd "__repo_root"
    set --
    for artifact in __INPUTS; do
      set -- "__@" "__(absolute_path "__artifact")"
    done
    python3 experiments/confirm/run_rg_universality_campaign.py analyze \
      --mode "__MODE" --inputs "__@" --output "__(absolute_path "__OUTPUT")"
    ;;
  *)
    echo "usage: __0 {plan|qualify|analyze}" >&2
    exit 2
    ;;
esac
""".replace("__", "$")


def expected_payloads():
    protocols = _payloads()
    files = dict(protocols)
    files[Path("README.md")] = README.encode("utf-8")
    files[Path("rg_universality_design.json")] = (
        json.dumps({
            "schema_version": "pldr-rg-universality-staging-v1",
            "stage_order": [stage["id"] for stage in STAGES],
            "architecture_runs": list(ARCHITECTURE_RUNS),
            "resource_caps": RESOURCE_CAPS,
            "campaign_directory": "campaigns/CAMPAIGN_ID",
            "separate_from": "../confirmation-program",
            "analyzer":
                "experiments/analysis/analyze_rg_universality.py",
        }, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    files[Path("run_rg_confirmation.sh")] = WRAPPER.encode("utf-8")
    manifest = "".join(
        f"{_sha256(files[path])}  {path.as_posix()}\n"
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
                "RG-universality staging is stale: " + ", ".join(stale))
        print(f"RG-universality staging: {len(expected)} files verified")
        return
    for relative, payload in expected.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    (output / "run_rg_confirmation.sh").chmod(0o755)
    print(
        f"RG-universality staging: wrote {len(expected)} files to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    stage(arguments.output, arguments.check)


if __name__ == "__main__":
    main()
