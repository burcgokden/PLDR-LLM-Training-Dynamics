#!/usr/bin/env python3
"""Preflight the registered v3 decision rules under their exact Gaussian null."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from row_rgmap.analysis import file_sha256, seal_record  # noqa: E402
from row_rgmap.confirmation_v3 import (  # noqa: E402
    _cross_segment_stability,
    _design_stability,
    _holm,
    _normal_calibration,
    _variance_scaling,
)
from row_rgmap.statistics import normal_ks_distance  # noqa: E402
from row_rgmap.provenance_v3 import (
    derive_seed,
    git_blob_descriptor,
    validate_seed_registry,
)  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _committed_identity() -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if status:
        raise RuntimeError("qualification requires a clean committed repository")
    return commit


def _nested_normal_law_rows(
    base_blocks: np.ndarray,
    block_sizes: list[int],
    calibration: dict[int, dict[str, np.ndarray]],
) -> list[dict]:
    smallest = min(block_sizes)
    rows = []
    for block_size in block_sizes:
        factor = block_size // smallest
        count = len(base_blocks) // factor
        values = base_blocks[: count * factor].reshape(count, factor).sum(axis=1)
        values = (values - values.mean()) / values.std()
        distance = normal_ks_distance(values)
        null = calibration[count]["ks"]
        rows.append(
            {
                "normal_ks_bootstrap_p_value": float(
                    (1 + np.sum(null >= distance)) / (len(null) + 1)
                ),
                "normal_ks_holm_adjusted_p_value": None,
                "normal_ks_rejected": None,
            }
        )
    return rows


def _nested_variance_series(base_blocks: np.ndarray, smallest: int) -> np.ndarray:
    return np.repeat(base_blocks / smallest, smallest)

def _alternative_detection(
    rules: dict,
    qualification: dict,
    campaigns: int,
    cells: int,
    design_batches: int,
    holdout_batches: int,
    critical: float,
    simulation_seed: int,
) -> dict:
    target = float(rules["target_mde_per_update"])
    multiple = float(qualification["alternative_effect_multiple_of_target_mde"])
    resource_fraction = float(qualification["alternative_resource_mde_fraction"])
    effect = multiple * target
    variance_degrees = design_batches - 1
    variance_lower_quantile = float(stats.chi2.ppf(
        float(rules["familywise_alpha"]) / cells, variance_degrees
    ))
    variance_upper_multiplier = variance_degrees / variance_lower_quantile
    batch_standard_deviation = (
        resource_fraction * target * math.sqrt(holdout_batches)
        / (critical * math.sqrt(variance_upper_multiplier))
    )
    generator = np.random.default_rng(simulation_seed)
    detections = 0
    trials = campaigns * cells * 2
    for _ in range(campaigns * cells):
        for direction in (-1.0, 1.0):
            mean = direction * effect
            design = generator.normal(
                loc=mean, scale=batch_standard_deviation, size=design_batches
            )
            holdout = generator.normal(
                loc=mean, scale=batch_standard_deviation, size=holdout_batches
            )
            selected = -1.0 if float(design.mean()) < 0.0 else 1.0
            design_variance_upper = (
                variance_degrees * float(design.var(ddof=1))
                / variance_lower_quantile
            )
            powered = critical * math.sqrt(
                design_variance_upper / holdout_batches
            ) <= target
            standard_error = float(holdout.std(ddof=1) / math.sqrt(holdout_batches))
            lower = float(holdout.mean()) - critical * standard_error
            upper = float(holdout.mean()) + critical * standard_error
            detected = (
                powered and selected == direction
                and ((direction < 0.0 and upper < 0.0)
                     or (direction > 0.0 and lower > 0.0))
            )
            detections += int(detected)
    probability = detections / trials
    return {
        "simulation_seed": simulation_seed,
        "effect_log_gain_per_update": effect,
        "effect_multiple_of_target_mde": multiple,
        "resource_mde_fraction": resource_fraction,
        "variance_upper_multiplier": variance_upper_multiplier,
        "batch_standard_deviation_at_target_mde": batch_standard_deviation,
        "trial_count": trials,
        "directional_detection_probability": probability,
        "monte_carlo_standard_error": math.sqrt(
            probability * (1.0 - probability) / trials
        ),
    }


def qualify(protocol: dict) -> dict:
    rules = protocol["analysis"]
    qualification = protocol["qualification"]
    campaigns = int(qualification["null_campaigns"])
    cells = int(rules["familywise_cell_count"])
    design_batches = int(protocol["segments"]["design_updates"]) // int(
        rules["time_batch_length"]
    )
    holdout_batches = int(protocol["segments"]["holdout_updates"]) // int(
        rules["time_batch_length"]
    )
    alpha = float(rules["familywise_alpha"])
    critical = float(stats.t.ppf(1.0 - alpha / cells, holdout_batches - 1))
    randomness = protocol.get("randomness")
    if randomness is None:
        master_seed = int(rules["gaussian_fixed_law"]["seed"])
        seeds = {
            domain: derive_seed(master_seed, domain)
            for domain in ("qualification:null", "qualification:alternative")
        }
    else:
        seeds = validate_seed_registry(protocol)
    null_seed = seeds["qualification:null"]
    alternative_seed = seeds["qualification:alternative"]
    generator = np.random.default_rng(null_seed)

    false_flow_campaigns = 0
    stable_campaigns = 0
    law_rejection_campaigns = 0
    scaling_pass_cells = 0
    total_cells = campaigns * cells

    law_sizes = [int(value) for value in rules["law_block_sizes"]]
    smallest_law = min(law_sizes)
    base_law_count = int(protocol["segments"]["holdout_updates"]) // smallest_law
    gaussian_rules = rules["gaussian_fixed_law"]
    calibration = {}
    for block_size in law_sizes:
        count = int(protocol["segments"]["holdout_updates"]) // block_size
        if count not in calibration:
            calibration[count] = _normal_calibration(
                count,
                int(gaussian_rules["bootstrap_replicates"]),
                int(gaussian_rules["seed"]),
            )

    variance_sizes = [int(value) for value in rules["variance_block_sizes"]]
    smallest_variance = min(variance_sizes)
    design_base_count = int(protocol["segments"]["design_updates"]) // smallest_variance
    holdout_base_count = int(protocol["segments"]["holdout_updates"]) // smallest_variance
    scaling_rules = rules["variance_scaling"]

    for _ in range(campaigns):
        campaign_false_flow = False
        campaign_stable = True
        campaign_law_rows = []
        for _cell in range(cells):
            design = generator.normal(size=design_batches)
            holdout = generator.normal(size=holdout_batches)
            design_mean = float(design.mean())
            holdout_mean = float(holdout.mean())
            standard_error = float(holdout.std(ddof=1) / math.sqrt(len(holdout)))
            if (
                design_mean < 0.0 and holdout_mean + critical * standard_error < 0.0
            ) or (
                design_mean > 0.0 and holdout_mean - critical * standard_error > 0.0
            ):
                campaign_false_flow = True
            design_stable = _design_stability(
                design, 1, rules["stationarity"]
            )["passed"]
            cross_stable = _cross_segment_stability(
                design, holdout, 1, rules["stationarity"]
            )["passed"]
            campaign_stable = campaign_stable and design_stable and cross_stable

            base_law = generator.normal(size=base_law_count)
            campaign_law_rows.extend(
                _nested_normal_law_rows(base_law, law_sizes, calibration)
            )

            design_base = generator.normal(
                scale=math.sqrt(smallest_variance), size=design_base_count
            )
            holdout_base = generator.normal(
                scale=math.sqrt(smallest_variance), size=holdout_base_count
            )
            design_series = _nested_variance_series(design_base, smallest_variance)
            holdout_series = _nested_variance_series(holdout_base, smallest_variance)
            design_scaling = _variance_scaling(design_series, variance_sizes)
            holdout_scaling = _variance_scaling(holdout_series, variance_sizes)
            if (
                float(scaling_rules["minimum_variance_exponent"])
                <= holdout_scaling["variance_exponent"]
                <= float(scaling_rules["maximum_variance_exponent"])
                and abs(
                    holdout_scaling["variance_exponent"]
                    - design_scaling["variance_exponent"]
                ) <= float(scaling_rules["maximum_design_holdout_exponent_shift"])
            ):
                scaling_pass_cells += 1
        _holm(campaign_law_rows, alpha)
        false_flow_campaigns += int(campaign_false_flow)
        stable_campaigns += int(campaign_stable)
        law_rejection_campaigns += int(
            any(row["normal_ks_rejected"] for row in campaign_law_rows)
        )

    flow_fwer = false_flow_campaigns / campaigns
    stable_probability = stable_campaigns / campaigns
    law_nonrejection = 1.0 - law_rejection_campaigns / campaigns
    scaling_cell_pass = scaling_pass_cells / total_cells
    alternative = _alternative_detection(
        rules,
        qualification,
        campaigns,
        cells,
        design_batches,
        holdout_batches,
        critical,
        alternative_seed,
    )
    gates = {
        "flow_familywise_error_passed": flow_fwer
        <= float(qualification["maximum_acceptable_flow_familywise_error"]),
        "gaussian_familywise_nonrejection_passed": law_nonrejection
        >= float(qualification["minimum_acceptable_gaussian_familywise_nonrejection"]),
        "stationary_domain_pass_probability_passed": stable_probability
        >= float(qualification["minimum_stationary_domain_pass_probability"]),
        "alternative_directional_power_passed": alternative[
            "directional_detection_probability"
        ] >= float(qualification["minimum_acceptable_directional_detection_probability"]),
    }
    return seal_record(
        {
            "schema_version": "pldr-row-rg-confirmation-v3-qualification-v1",
            "completed_at_utc": _utc_now(),
            "protocol_id": protocol["protocol_id"],
            "simulation_seed": null_seed,
            "null_campaigns": campaigns,
            "cells_per_campaign": cells,
            "student_t_critical_value": critical,
            "flow_null_familywise_error": flow_fwer,
            "stationary_domain_all_cell_pass_probability": stable_probability,
            "gaussian_law_familywise_nonrejection_probability": law_nonrejection,
            "finite_variance_scaling_cell_pass_probability": scaling_cell_pass,
            "alternative_power": alternative,
            "null_monte_carlo_standard_errors": {
                "flow_familywise_error": math.sqrt(
                    flow_fwer * (1.0 - flow_fwer) / campaigns
                ),
                "stationary_domain_all_cell_pass": math.sqrt(
                    stable_probability * (1.0 - stable_probability) / campaigns
                ),
                "gaussian_familywise_nonrejection": math.sqrt(
                    law_nonrejection * (1.0 - law_nonrejection) / campaigns
                ),
                "finite_variance_scaling_cell_pass": math.sqrt(
                    scaling_cell_pass * (1.0 - scaling_cell_pass) / total_cells
                ),
            },
            "gates": gates,
            "all_launch_gates_passed": all(gates.values()),
            "scope": {
                "flow_simulation_uses_independent_normal_batch_means": True,
                "law_simulation_uses_nested_exact_normal_block_sums": True,
                "variance_simulation_uses_nested_exact_normal_block_sums": True,
                "qualification_does_not_use_confirmation_holdout_data": True,
            },
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    protocol_path = Path(arguments.protocol)
    output = Path(arguments.output)
    if output.exists():
        raise FileExistsError(output)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    record = qualify(protocol)
    record["protocol_file_sha256"] = file_sha256(protocol_path)
    commit = _committed_identity()
    record["code_commit"] = commit
    source_paths = [
        "scripts/qualify_confirmation_v3.py",
        "src/row_rgmap/analysis.py",
        "src/row_rgmap/confirmation_v3.py",
        "src/row_rgmap/diagnostics_v3.py",
        "src/row_rgmap/statistics.py",
        "src/row_rgmap/provenance_v3.py",
    ]
    record["qualification_sources"] = []
    for repository_path in source_paths:
        digest, size = git_blob_descriptor(ROOT, commit, repository_path)
        record["qualification_sources"].append(
            {
                "repository_path": repository_path,
                "code_commit": commit,
                "sha256": digest,
                "size_bytes": size,
            }
        )
    unsigned = dict(record)
    unsigned.pop("record_sha256")
    record = seal_record(unsigned)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "all_launch_gates_passed": record["all_launch_gates_passed"],
        "record_sha256": record["record_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
