#!/usr/bin/env python3
"""Stage an immutable chronological-confirmation bundle in experiment-data."""

from __future__ import annotations
from companion_paths import legacy_path

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments" / "protocols" / "chronological_confirmation"
DEFAULT_OUTPUT = Path(
    legacy_path('/pldr-data/row/rev38/chronological-collapse-confirmation')
)
SOURCES = (
    "requirements.txt",
    "experiments/train_run.py",
    "experiments/instrument.py",
    "experiments/optimizer_ledger.py",
    "experiments/pldr_model_v510.py",
    "experiments/power_law_attention_layer_v510.py",
    "experiments/confirm/confirmation_artifacts.py",
    "experiments/confirm/gate_shape.py",
    "experiments/confirm/gate_shape_evidence.py",
    "experiments/confirm/row_map_dynamics.py",
    "experiments/confirm/row_map_live.py",
    "experiments/confirm/observable_margin.py",
    "experiments/confirm/observable_margin_specs.py",
    "experiments/confirm/observable_margin_live.py",
    "experiments/confirm/chronological_history.py",
    "experiments/confirm/chronological_collapse.py",
    "experiments/confirm/chronological_confirmation_specs.py",
    "experiments/confirm/capture_chronological_confirmation.py",
    "experiments/confirm/capture_chronological_plga.py",
    "experiments/confirm/run_chronological_intervention.py",
    "experiments/confirm/run_chronological_confirmation.py",
    "experiments/analysis/analyze_chronological_confirmation.py",
    "experiments/analysis/analyze_chronological_plga.py",
    "experiments/analysis/analyze_chronological_intervention.py",
    "scripts/run_chronological_qualification.py",
    "scripts/gen_chronological_protocols.py",
    "scripts/stage_chronological_confirmation.py",
    "experiments/tests/test_chronological_capture_rev38.py",
    "experiments/tests/test_chronological_campaign_rev38.py",
    "experiments/tests/test_chronological_collapse_rev38.py",
    "experiments/tests/test_chronological_history_rev38.py",
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def expected_files() -> dict[Path, bytes]:
    if not PROTOCOL.is_dir():
        raise ValueError("generate chronological protocols before staging")
    files = {
        Path("protocol") / path.relative_to(PROTOCOL): path.read_bytes()
        for path in PROTOCOL.rglob("*")
        if path.is_file()
    }
    for relative in SOURCES:
        source = ROOT / relative
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"staged source is missing or invalid: {relative}")
        files[Path("source") / relative] = source.read_bytes()
    protocol_manifest = files[Path("protocol/CHECKSUMS.sha256")]
    files[Path("STAGING.json")] = _json({
        "schema_version": "pldr-chronological-staging-v1",
        "campaign_id": "pldr-chronological-collapse-confirmation-v1",
        "manuscript_series": "rev38",
        "protocol_manifest_sha256": _sha256(protocol_manifest),
        "source_files": list(SOURCES),
        "qualification_command": [
            "python3", "source/scripts/run_chronological_qualification.py",
            "--output-dir", "qualification", "--require-pass",
        ],
        "policy": (
            "Successor tensors are retained only for coverage checks. "
            "Construction constants are immutable before held-out records open."
        ),
    })
    manifest = "".join(
        f"{_sha256(files[path])}  {path.as_posix()}\n"
        for path in sorted(files, key=lambda item: item.as_posix())
    )
    files[Path("MANIFEST.sha256")] = manifest.encode("ascii")
    return files


def _actual_files(output: Path) -> dict[Path, bytes]:
    if not output.exists():
        return {}
    if output.is_symlink() or not output.is_dir():
        raise ValueError("staging output must be a regular directory")
    actual: dict[Path, bytes] = {}
    for path in output.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"staging output contains a symlink: {path}")
        if path.is_file():
            actual[path.relative_to(output)] = path.read_bytes()
    return actual


def stage(output: Path, *, check: bool) -> None:
    expected = expected_files()
    actual = _actual_files(output)
    if check:
        if actual != expected:
            raise SystemExit("chronological confirmation staging is stale")
        print(f"chronological staging: verified ({len(expected)} files)")
        return
    unknown = set(actual) - set(expected)
    if unknown:
        names = ", ".join(sorted(path.as_posix() for path in unknown))
        raise SystemExit(f"refusing to overwrite unknown staged files: {names}")
    output.mkdir(parents=True, exist_ok=True)
    for relative, payload in expected.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    if _actual_files(output) != expected:
        raise SystemExit("chronological staging verification failed")
    print(f"chronological staging: wrote {len(expected)} files to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    stage(arguments.output.resolve(), check=arguments.check)


if __name__ == "__main__":
    main()
