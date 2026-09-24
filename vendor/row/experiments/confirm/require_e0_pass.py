#!/usr/bin/env python3
"""Authorize a downstream confirmation launch from the final E0 record."""

import argparse
import json
from pathlib import Path

from e0_decision import E0Outcome, decide_e0


def require_e0_pass(path, expected_source_hash=None, expected_protocol_hash=None):
    path = Path(path)
    with path.open() as handle:
        record = json.load(handle)
    decision = decide_e0(record)
    if decision.outcome is not E0Outcome.PASS:
        raise RuntimeError(
            f"downstream launch denied: E0 outcome {decision.outcome.value}")
    if expected_source_hash is not None and record["source_hash"] != expected_source_hash:
        raise RuntimeError("downstream launch denied: E0 source hash is stale")
    if (expected_protocol_hash is not None
            and record["protocol_hash"] != expected_protocol_hash):
        raise RuntimeError("downstream launch denied: E0 protocol hash is stale")
    return decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("record")
    parser.add_argument("--source-hash")
    parser.add_argument("--protocol-hash")
    args = parser.parse_args()
    require_e0_pass(args.record, args.source_hash, args.protocol_hash)
    print("E0 PASS: downstream launch authorized")


if __name__ == "__main__":
    main()
