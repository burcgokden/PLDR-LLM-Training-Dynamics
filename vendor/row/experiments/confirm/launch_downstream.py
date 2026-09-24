#!/usr/bin/env python3
"""Single authorization boundary for E1 through E8 launch commands."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import assemble_e0
from require_e0_pass import require_e0_pass


EXPERIMENTS = tuple(f"E{i}" for i in range(1, 9))
DEFAULT_RECORD = assemble_e0.OUT / "E0_RECORD.json"


def authorize(record):
    """Require PASS and bind it to the current protocol and source bundle."""
    return require_e0_pass(
        record,
        expected_source_hash=assemble_e0.source_bundle_hash(),
        expected_protocol_hash=assemble_e0.sha256_file(
            assemble_e0.PROTOCOL_PATH
        ),
    )


def launch(experiment, command, record=DEFAULT_RECORD, dry_run=False):
    if experiment not in EXPERIMENTS:
        raise ValueError(f"experiment must be one of {', '.join(EXPERIMENTS)}")
    if not command:
        raise ValueError("a downstream command is required")
    authorize(Path(record))
    if dry_run:
        return 0
    return subprocess.run(list(command), check=False).returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    parser.add_argument("--record", default=str(DEFAULT_RECORD))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    raise SystemExit(
        launch(args.experiment, command, args.record, args.dry_run)
    )


if __name__ == "__main__":
    main()
