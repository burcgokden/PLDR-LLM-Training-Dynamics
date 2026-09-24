#!/usr/bin/env python3
"""Preserve, remove, and reseal manifest-bound replication bytecode."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import tempfile

from bundle_hygiene import is_excluded_path
import seal_direct_work_replication as sealer


INCIDENT_ID = "direct-work-replication-bytecode-hygiene-20260901T0020Z"
ARCHIVE_SCHEMA = "pldr-bytecode-hygiene-archive-v1"
EXPECTED_BYTECODE = frozenset({
    Path(
        "source/experiments/analysis/__pycache__/"
        "analyze_direct_work_precision.cpython-314.pyc"
    ),
    Path(
        "source/experiments/analysis/__pycache__/"
        "analyze_direct_work_replication.cpython-314.pyc"
    ),
    Path(
        "source/experiments/confirm/__pycache__/"
        "direct_work_replication_specs.cpython-314.pyc"
    ),
    Path(
        "source/experiments/confirm/__pycache__/"
        "resource_executor.cpython-314.pyc"
    ),
})


def regular_files(root: Path, *, omit: frozenset[Path] = frozenset()) -> set[Path]:
    files: set[Path] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"unexpected symlink: {path}")
        if path.is_file():
            relative = path.relative_to(root)
            if relative not in omit:
                files.add(relative)
    return files


def verified_prior_rows(campaign_root: Path) -> dict[Path, str]:
    manifest = campaign_root / "MANIFEST.sha256"
    expected = regular_files(
        campaign_root, omit=frozenset({Path("MANIFEST.sha256")})
    )
    sealer.verify_manifest(manifest, campaign_root, expected)
    rows = sealer.parse_manifest(manifest)
    bytecode = {
        relative for relative in rows
        if is_excluded_path(relative)
    }
    if bytecode != EXPECTED_BYTECODE:
        raise ValueError(
            "unexpected manifest-bound bytecode set: "
            + ", ".join(str(path) for path in sorted(bytecode))
        )
    return rows


def excluded_rows(
    campaign_root: Path, prior_rows: dict[Path, str]
) -> list[dict[str, object]]:
    rows = []
    for relative in sorted(EXPECTED_BYTECODE):
        source = campaign_root / relative
        if (
            not source.is_file()
            or source.is_symlink()
            or sealer.sha256_path(source) != prior_rows[relative]
        ):
            raise ValueError(f"manifest-bound bytecode changed: {relative}")
        rows.append({
            "path": relative.as_posix(),
            "sha256": prior_rows[relative],
            "size_bytes": source.stat().st_size,
        })
    return rows


def build_external_incident(
    binding_root: Path,
    campaign_root: Path,
    incident_root: Path,
    rows: list[dict[str, object]],
) -> Path:
    if incident_root.exists():
        raise FileExistsError(f"incident already exists: {incident_root}")
    incident_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{INCIDENT_ID}.", dir=incident_root.parent
    ))
    try:
        prior_seal = temporary / "prior-seal"
        prior_seal.mkdir(parents=True)
        prior_digests: dict[str, str] = {}
        for name in sealer.ROOT_SEAL_FILES:
            source = campaign_root / name
            if not source.is_file() or source.is_symlink():
                raise ValueError(f"prior seal file is absent: {name}")
            target = prior_seal / name
            shutil.copy2(source, target)
            prior_digests[name] = sealer.sha256_path(target)

        for row in rows:
            relative = Path(str(row["path"]))
            target = temporary / "excluded-bytecode" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(campaign_root / relative, target)

        archive = {
            "schema_version": ARCHIVE_SCHEMA,
            "campaign_id": sealer.CAMPAIGN_ID,
            "incident_id": INCIDENT_ID,
            "status": "preserved-before-bytecode-free-reseal",
            "campaign_root": campaign_root.relative_to(binding_root).as_posix(),
            "excluded_files": rows,
            "prior_seal_sha256": prior_digests,
            "preservation_rule": (
                "all-prior-nonseal-nonbytecode-files-remain-byte-identical"
            ),
            "scientific_disposition": (
                "no-record-runtime-analysis-or-decision-value-changed"
            ),
        }
        sealer.write_json(temporary / "incident.json", archive)
        sealer.write_bytes(
            temporary / "README.md",
            (
                "# Direct-work replication bytecode hygiene incident\n\n"
                "Four interpreter cache files were manifest-bound by the "
                "packaging seal. This incident preserves those files and "
                "the complete prior seal before a bytecode-free reseal. "
                "Every nonseal, nonbytecode file remains byte-identical.\n"
            ).encode("utf-8"),
        )
        selected = regular_files(temporary)
        sealer.write_bytes(
            temporary / "MANIFEST.sha256",
            sealer.manifest_bytes(temporary, list(selected)),
        )
        sealer.verify_manifest(
            temporary / "MANIFEST.sha256", temporary, selected
        )
        os.replace(temporary, incident_root)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return incident_root / "MANIFEST.sha256"


def campaign_incident(
    binding_root: Path,
    external_manifest: Path,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    prior_manifest = external_manifest.parent / "prior-seal/MANIFEST.sha256"
    return {
        "schema_version": sealer.PACKAGING_INCIDENT_SCHEMA,
        "campaign_id": sealer.CAMPAIGN_ID,
        "incident_id": INCIDENT_ID,
        "status": "resolved-and-resealed",
        "external_incident_manifest": external_manifest.relative_to(
            binding_root
        ).as_posix(),
        "external_incident_manifest_sha256": sealer.sha256_path(
            external_manifest
        ),
        "prior_manifest_sha256": sealer.sha256_path(prior_manifest),
        "excluded_files": rows,
        "preservation_rule": (
            "all-prior-nonseal-nonbytecode-files-are-byte-identical"
        ),
    }


def apply_remediation(
    binding_root: Path, campaign_root: Path, incident_root: Path
) -> None:
    if not campaign_root.is_relative_to(binding_root):
        raise ValueError("campaign root escapes binding root")
    if (campaign_root / sealer.PACKAGING_INCIDENT_FILE).exists():
        raise FileExistsError("campaign packaging incident already exists")
    prior_rows = verified_prior_rows(campaign_root)
    rows = excluded_rows(campaign_root, prior_rows)
    external_manifest = build_external_incident(
        binding_root, campaign_root, incident_root, rows
    )
    sealer.write_json(
        campaign_root / sealer.PACKAGING_INCIDENT_FILE,
        campaign_incident(binding_root, external_manifest, rows),
    )

    for relative in sorted(EXPECTED_BYTECODE):
        (campaign_root / relative).unlink()
    cache_directories = sorted(
        {
            (campaign_root / relative).parent
            for relative in EXPECTED_BYTECODE
        },
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in cache_directories:
        directory.rmdir()
    for name in sealer.ROOT_SEAL_FILES:
        (campaign_root / name).unlink()

    sealer.seal(binding_root, campaign_root)
    print(
        "bytecode remediation: preserved prior seal at "
        f"{incident_root} and wrote a bytecode-free superseding seal"
    )


def verify_remediation(binding_root: Path, campaign_root: Path) -> None:
    sealer.verify(binding_root, campaign_root)
    incident = sealer.load_json(
        campaign_root / sealer.PACKAGING_INCIDENT_FILE
    )
    print(
        "bytecode remediation: verified "
        f"{len(incident['excluded_files'])} archived files"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("apply", "verify"))
    parser.add_argument("--binding-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--incident-root", type=Path)
    arguments = parser.parse_args()
    binding_root = arguments.binding_root.resolve(strict=True)
    campaign_root = arguments.campaign_root.resolve(strict=True)
    incident_root = (
        arguments.incident_root.resolve(strict=False)
        if arguments.incident_root is not None
        else binding_root
        / "experiment-data/manuscript-revisions/rev52/incidents"
        / INCIDENT_ID
    )
    if not incident_root.is_relative_to(binding_root):
        raise ValueError("incident root escapes binding root")
    if arguments.command == "apply":
        apply_remediation(binding_root, campaign_root, incident_root)
    else:
        verify_remediation(binding_root, campaign_root)


if __name__ == "__main__":
    main()
