#!/usr/bin/env python3
"""Execute the orbitwise producer graph with a deadline and observed resource admission."""

from __future__ import annotations

import execute_source_resolved_plan as executor

from confirm.orbitwise_specs import CAMPAIGN_ID, LAUNCH_PLAN_SCHEMA


executor.CAMPAIGN_ID = CAMPAIGN_ID
executor.LAUNCH_PLAN_SCHEMA = LAUNCH_PLAN_SCHEMA
executor.RESOURCE_SCHEMA = "pldr-orbitwise-resource-record-v2"
executor.PREFLIGHT_SCHEMA = "pldr-orbitwise-preflight-failure-v2"
executor.EXECUTION_SCHEMA = "pldr-orbitwise-execution-report-v2"
executor.CAMPAIGN_LABEL = "orbitwise"


if __name__ == "__main__":
    executor.main()
