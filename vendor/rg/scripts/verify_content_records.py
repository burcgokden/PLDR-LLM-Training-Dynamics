#!/usr/bin/env python3
"""Verify canonical SHA-256 digests on one or more JSON records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from row_rgmap.analysis import verify_record_seal as verify_content_record  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", nargs="+")
    arguments = parser.parse_args()
    for name in arguments.records:
        path = Path(name)
        record = json.loads(path.read_text(encoding="utf-8"))
        verify_content_record(record)
        print(f"{path}: {record['record_sha256']}")


if __name__ == "__main__":
    main()
