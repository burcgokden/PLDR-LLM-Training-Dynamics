#!/usr/bin/env python3
"""Live producer for the temporal-context direct-work replication."""

from __future__ import annotations

from pathlib import Path
import sys

EXPERIMENTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENTS))

from confirm.direct_work_live import argument_parser, produce
from confirm.direct_work_replication_specs import (
    CAMPAIGN_ID,
    RECORD_SCHEMA,
    RELEASE_ID,
)


def main() -> None:
    arguments = argument_parser().parse_args()
    if not arguments.allow_disjoint_registry:
        raise ValueError("replication requires its frozen disjoint registry")
    produce(
        arguments,
        record_schema=RECORD_SCHEMA,
        campaign_id=CAMPAIGN_ID,
        release_id=RELEASE_ID,
    )


if __name__ == "__main__":
    main()
