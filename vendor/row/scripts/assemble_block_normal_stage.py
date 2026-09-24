#!/usr/bin/env python3
"""Assemble every sealed source/native pair for one campaign stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_block_normal_confirmation import (  # type: ignore  # noqa: E402
    analyze_records,
    write_json_atomic,
)
from assemble_block_normal_record import assemble  # noqa: E402
from confirm.native_determinism import sha256_path  # noqa: E402


def assemble_stage(
    source_dir: str | Path,
    native_dir: str | Path,
    output_dir: str | Path,
    *,
    expected_count: int,
) -> list[dict]:
    """Require an exact source membership and assemble every matched record."""

    source_root = Path(source_dir).resolve()
    native_root = Path(native_dir).resolve()
    output_root = Path(output_dir).resolve()
    if expected_count < 1:
        raise ValueError("stage expected count must be positive")
    source_paths = sorted(source_root.glob("*.source.json"))
    native_paths = sorted(native_root.glob("*.native.json"))
    if len(source_paths) != expected_count or len(native_paths) != expected_count:
        raise ValueError("stage source/native membership is incomplete")
    source_stems = {
        path.name.removesuffix(".source.json") for path in source_paths
    }
    native_stems = {
        path.name.removesuffix(".native.json") for path in native_paths
    }
    if source_stems != native_stems:
        raise ValueError("stage source and native identifiers disagree")
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for stem in sorted(source_stems):
        source_path = source_root / f"{stem}.source.json"
        native_path = native_root / f"{stem}.native.json"
        source = json.loads(source_path.read_text(encoding="utf-8"))
        native = json.loads(native_path.read_text(encoding="utf-8"))
        if native.get("source_prediction_sha256") != sha256_path(source_path):
            raise ValueError("native record is not bound to its source file")
        record = assemble(source, native)
        write_json_atomic(output_root / f"{stem}.record.json", record)
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--analysis-output", type=Path, required=True)
    arguments = parser.parse_args()
    records = assemble_stage(
        arguments.source_dir,
        arguments.native_dir,
        arguments.output_dir,
        expected_count=arguments.expected_count,
    )
    report = analyze_records(records)
    write_json_atomic(arguments.analysis_output, report)
    print(json.dumps({
        "record_count": len(records),
        "outcome_counts": report["outcome_counts"],
        "analysis_output": str(arguments.analysis_output.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
