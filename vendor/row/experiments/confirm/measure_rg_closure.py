#!/usr/bin/env python3
"""Measure deductive-sector and conditional full-layer RG closure for E8."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from campaign_record import strict_dumps  # noqa: E402
from renormalization_group import (  # noqa: E402
    block_average_semigroup_residual,
    block_softmax_aggregation,
    deductive_rg_closure_bound,
    metric_layernorm_scale_identity,
    telescoping_closure_bound,
)


STAGES = (
    "input_projection_rope",
    "deductive_map",
    "scores",
    "mask_softmax",
    "values_output",
    "residual_ffn",
)
OTHER_DEFECT_STAGES = tuple(
    stage for stage in STAGES if stage != "deductive_map")
REQUIRED = {
    "rotated_queries",
    "block_size",
    "repeat_block_size",
    "metric_layernorm_epsilon",
    "metric_layernorm_gamma",
    "metric_layernorm_beta",
    "layernorm_lipschitz",
    "direct_row_map_upper",
    "deductive_downstream_lipschitz",
    "observed_deductive_difference",
    "attention_logits",
    "full_layer_other_defect_bounds",
    "full_layer_stage_lipschitz",
    "observed_full_layer_difference",
}


def _duplicates_rejected(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON key: {name}")
        result[name] = value
    return result


def _load(path):
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON token: {value}")

    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(
            stream,
            object_pairs_hook=_duplicates_rejected,
            parse_constant=reject_constant,
        )
    if not isinstance(value, dict) or set(value) != REQUIRED:
        keys = set(value) if isinstance(value, dict) else set()
        raise ValueError(
            "RG input schema mismatch: "
            f"missing={sorted(REQUIRED - keys)}, "
            f"extra={sorted(keys - REQUIRED)}")
    return value


def _closed_numeric_mapping(value, names, label):
    if not isinstance(value, dict) or set(value) != set(names):
        keys = set(value) if isinstance(value, dict) else set()
        raise ValueError(
            f"{label} schema mismatch: "
            f"missing={sorted(set(names) - keys)}, "
            f"extra={sorted(keys - set(names))}")
    result = {}
    for name in names:
        number = float(value[name])
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(
                f"{label}.{name} must be finite and nonnegative")
        result[name] = number
    return result


def _nonnegative(value, name):
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _ratio(observed, upper):
    observed = _nonnegative(observed, "observed")
    upper = _nonnegative(upper, "upper")
    if upper == 0.0:
        return 0.0 if observed == 0.0 else float(np.finfo(float).max)
    return observed / upper


def _measurement(name, value, unit, method):
    return {
        "name": name,
        "value": float(value),
        "unit": unit,
        "method": method,
        "status": "OBSERVED",
        "reason_code": None,
    }


def measure(value):
    if not isinstance(value, dict) or set(value) != REQUIRED:
        keys = set(value) if isinstance(value, dict) else set()
        raise ValueError(
            "RG input schema mismatch: "
            f"missing={sorted(REQUIRED - keys)}, "
            f"extra={sorted(keys - REQUIRED)}")
    queries = np.asarray(value["rotated_queries"], dtype=float)
    block_size = value["block_size"]
    repeat_block_size = value["repeat_block_size"]
    closure = deductive_rg_closure_bound(
        rotated_queries=queries,
        block_size=block_size,
        layernorm_lipschitz=value["layernorm_lipschitz"],
        direct_row_map_upper=value["direct_row_map_upper"],
        downstream_lipschitz=value["deductive_downstream_lipschitz"],
        observed_deductive_difference=
            value["observed_deductive_difference"],
    )
    layernorm = metric_layernorm_scale_identity(
        closure["coarse_normalized_density"],
        block_size=block_size,
        epsilon=value["metric_layernorm_epsilon"],
        gamma=value["metric_layernorm_gamma"],
        beta=value["metric_layernorm_beta"],
    )
    repeated_residual = block_average_semigroup_residual(
        queries, block_size, repeat_block_size)
    softmax = block_softmax_aggregation(
        value["attention_logits"], block_size)

    other_defects = _closed_numeric_mapping(
        value["full_layer_other_defect_bounds"],
        OTHER_DEFECT_STAGES,
        "full_layer_other_defect_bounds",
    )
    stage_lipschitz = _closed_numeric_mapping(
        value["full_layer_stage_lipschitz"],
        STAGES,
        "full_layer_stage_lipschitz",
    )
    defects = [
        closure["deductive_closure_upper"]
        if stage == "deductive_map" else other_defects[stage]
        for stage in STAGES
    ]
    lipschitz = [stage_lipschitz[stage] for stage in STAGES]
    full = telescoping_closure_bound(defects, lipschitz)
    observed_full = _nonnegative(
        value["observed_full_layer_difference"],
        "observed_full_layer_difference",
    )
    full_ratio = _ratio(observed_full, full["closure_upper"])

    measurements = [
        _measurement(
            "rg_block_size", block_size,
            "fine_tokens_per_coarse_token", "aligned_query_block_average"),
        _measurement(
            "rg_density_identity_residual",
            closure["density_identity_residual"],
            "relative_frobenius", "fine_equals_coarse_plus_covariance"),
        _measurement(
            "rg_layernorm_scale_residual", layernorm["relative_residual"],
            "relative_frobenius", "epsilon_rescaling_identity"),
        _measurement(
            "rg_within_covariance_trace",
            closure["within_covariance_trace"],
            "query_squared", "mean_within_block_query_variance"),
        _measurement(
            "rg_deductive_closure_upper",
            closure["deductive_closure_upper"],
            "deductive_tensor_frobenius",
            "direct_seminorm_times_density_defect"),
        _measurement(
            "rg_deductive_closure_ratio",
            closure["deductive_closure_ratio"],
            "ratio", "observed_over_deductive_closure_upper"),
        _measurement(
            "rg_softmax_aggregation_residual",
            softmax["aggregation_residual"],
            "absolute_probability",
            "fine_block_mass_vs_block_logsumexp_softmax"),
        _measurement(
            "rg_repeated_block_semigroup_residual", repeated_residual,
            "relative_l2", "C_c_composed_C_b_vs_C_bc"),
        _measurement(
            "rg_full_layer_closure_upper", full["closure_upper"],
            "layer_output_frobenius", "transported_stage_defect_sum"),
        _measurement(
            "rg_full_layer_closure_ratio", full_ratio,
            "ratio", "observed_over_full_layer_closure_upper"),
    ]
    return {
        "schema_version": "pldr-rg-closure-measurement-v1",
        "protocol_id": "E8",
        "stages": list(STAGES),
        "renormalized_layernorm_epsilon":
            layernorm["renormalized_epsilon"],
        "mean_logit_approximation_error":
            softmax["mean_logit_approximation_error"],
        "maximum_within_block_logit_oscillation":
            softmax["maximum_within_block_logit_oscillation"],
        "full_layer_transported_contributions":
            full["transported_contributions"],
        "observed_full_layer_difference": observed_full,
        "measurements": measurements,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = measure(_load(args.input))
    Path(args.output).write_text(strict_dumps(result), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
