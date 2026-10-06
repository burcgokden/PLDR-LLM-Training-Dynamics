#!/usr/bin/env python3
"""Relocate and independently replay a staged predictive-closure record."""

from __future__ import annotations
from companion_paths import configured_path

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from row_rgmap.analysis import seal_record  # noqa: E402
from row_rgmap.staged_validation import (  # noqa: E402
    RELOCATION_SCHEMA,
    declared_relative_files,
    lexical_file_census,
    validate_staged_relocation,
    validate_staged_result,
)
from scripts.analyze_staged_predictive_closure import analyze  # noqa: E402
from scripts.run_energy_closure_probe_v3 import write_new_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--run-path", required=True)
    parser.add_argument("--refinedweb-root", default=configured_path('assets:refinedweb'))
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    data_root = Path(arguments.data_root).resolve()
    relative = Path(arguments.run_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("staged run path must be safe and data-root relative")
    run_root = (data_root / relative).resolve()
    output = Path(arguments.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if output.parent != run_root:
        raise ValueError("relocation record must be written in the staged run root")
    manifest = json.loads((run_root / "run-manifest.json").read_text(encoding="utf-8"))
    result = json.loads((run_root / "result.json").read_text(encoding="utf-8"))
    config = json.loads((run_root / "inputs" / "config.json").read_text(encoding="utf-8"))
    validate_staged_result(
        result, manifest, config, run_root=run_root,
        data_root=data_root, repository_root=ROOT,
        refinedweb_root=Path(arguments.refinedweb_root),
    )
    source_census = lexical_file_census(
        run_root, declared_relative_files(manifest)
    )
    with tempfile.TemporaryDirectory(
        prefix=".staged-closure-relocation-", dir=data_root
    ) as temporary:
        copy_root = Path(temporary) / "relocated"
        shutil.copytree(run_root, copy_root)
        copied_manifest = json.loads(
            (copy_root / "run-manifest.json").read_text(encoding="utf-8")
        )
        copied_result = analyze(
            copy_root, data_root, manifest=copied_manifest,
            refinedweb_root=Path(arguments.refinedweb_root),
        )
        copied_census = lexical_file_census(
            copy_root, declared_relative_files(copied_manifest)
        )
    fields = (
        "code_commit",
        "analysis_sources",
        "input_run_manifest",
        "stage_id",
        "gpu_time_accounting",
        "gpu_seconds",
        "gpu_hours",
        "validity",
        "scientific_payload",
    )
    replay_equal = all(copied_result[key] == result[key] for key in fields)
    if not replay_equal or copied_census != source_census:
        raise ValueError("relocated staged analysis differs from the source analysis")
    payload = seal_record(
        {
            "schema_version": RELOCATION_SCHEMA,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "code_commit": manifest["code_commit"],
            "input_run_manifest_record_sha256": manifest["record_sha256"],
            "input_analysis_record_sha256": result["record_sha256"],
            "source_census": source_census,
            "relocated_census": copied_census,
            "scientific_payload_equal": replay_equal,
            "temporary_copy_removed": True,
        }
    )
    validate_staged_relocation(payload, result, manifest, run_root)
    write_new_json(output, payload)
    print(
        json.dumps(
            {
                "status": "complete",
                "record_sha256": payload["record_sha256"],
                "file_count": source_census["file_count"],
                "total_size_bytes": source_census["total_size_bytes"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
