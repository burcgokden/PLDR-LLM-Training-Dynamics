"""Stage-complete same-source tensor, attention, and logit bridge."""

from __future__ import annotations

import hashlib
import json
import math


SCHEMA_VERSION = "pldr-same-source-bridge-v2"
REQUIRED_STAGES = (
    "A_to_A_LM",
    "A_LM_to_A_P",
    "A_P_to_G_LM",
    "scores",
    "softmax",
    "attention_output",
    "residual",
    "layer_norm",
    "logits",
)


def _digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _finite(value, label, *, nonnegative=False):
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0):
        qualifier = "finite and nonnegative" if nonnegative else "finite"
        raise ValueError(f"{label} must be {qualifier}")
    return value


def _sha256(value, label):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return value


def construct_same_source_bridge(payload):
    """Transport one immutable source pair with the row map applied once."""

    required = {
        "schema_version", "pair_id", "source_a_sha256", "source_b_sha256",
        "source_row_distance", "direct_row_map_upper",
        "row_map_defect_upper", "observed_row_map_distance",
        "absolute_budget", "stages", "observed_final_error",
        "logit_margin_lower", "reference_mean",
        "reference_mean_abs_lower", "normalized_budget",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("same-source payload has missing or unknown fields")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unknown same-source bridge schema")
    pair_id = payload["pair_id"]
    if not isinstance(pair_id, str) or not pair_id:
        raise ValueError("pair_id must be a nonempty string")
    source_a = _sha256(payload["source_a_sha256"], "source_a_sha256")
    source_b = _sha256(payload["source_b_sha256"], "source_b_sha256")
    if source_a == source_b:
        raise ValueError("the paired perturbation must use two distinct sources")

    source_distance = _finite(
        payload["source_row_distance"], "source row distance",
        nonnegative=True)
    direct = _finite(
        payload["direct_row_map_upper"], "direct row-map upper",
        nonnegative=True)
    row_defect = _finite(
        payload["row_map_defect_upper"], "row-map defect upper",
        nonnegative=True)
    row_observed = _finite(
        payload["observed_row_map_distance"], "observed row-map distance",
        nonnegative=True)
    distance = math.nextafter(
        direct * source_distance + row_defect, math.inf)
    transported = [{
        "name": "row_map",
        "input_upper": source_distance,
        "gain_upper": direct,
        "defect_upper": row_defect,
        "output_upper": distance,
        "observed_output_distance": row_observed,
        "observed_enclosed": row_observed <= distance,
    }]

    rows = payload["stages"]
    if not isinstance(rows, list) or len(rows) != len(REQUIRED_STAGES):
        raise ValueError("same-source bridge must contain every downstream stage")
    for expected_name, row in zip(REQUIRED_STAGES, rows):
        required_row = {
            "name", "pair_id", "input_source_a_sha256",
            "input_source_b_sha256", "gain_upper", "defect_upper",
            "observed_output_distance",
        }
        if not isinstance(row, dict) or set(row) != required_row:
            raise ValueError("same-source stage has missing or unknown fields")
        if row["name"] != expected_name:
            raise ValueError("same-source stages are absent, duplicated, or reordered")
        if (
            row["pair_id"] != pair_id
            or row["input_source_a_sha256"] != source_a
            or row["input_source_b_sha256"] != source_b
        ):
            raise ValueError("a downstream stage changed the source pair")
        gain = _finite(row["gain_upper"], "stage gain", nonnegative=True)
        defect = _finite(row["defect_upper"], "stage defect", nonnegative=True)
        observed = _finite(
            row["observed_output_distance"], "observed stage distance",
            nonnegative=True)
        upper = math.nextafter(gain * distance + defect, math.inf)
        transported.append({
            "name": expected_name,
            "input_upper": distance,
            "gain_upper": gain,
            "defect_upper": defect,
            "output_upper": upper,
            "observed_output_distance": observed,
            "observed_enclosed": observed <= upper,
        })
        distance = upper

    absolute_budget = _finite(
        payload["absolute_budget"], "absolute cache budget", nonnegative=True)
    observed_final = _finite(
        payload["observed_final_error"], "observed final error",
        nonnegative=True)
    margin = payload["logit_margin_lower"]
    if margin is not None:
        margin = _finite(margin, "logit margin lower", nonnegative=True)

    mean = payload["reference_mean"]
    mean_floor = payload["reference_mean_abs_lower"]
    normalized_budget = payload["normalized_budget"]
    normalization_claimed = any(
        value is not None for value in (mean, mean_floor, normalized_budget))
    if normalization_claimed and any(
        value is None for value in (mean, mean_floor, normalized_budget)
    ):
        raise ValueError(
            "mean, absolute mean floor, and normalized budget are one claim")
    if normalization_claimed:
        mean = _finite(mean, "reference mean")
        mean_floor = _finite(
            mean_floor, "absolute mean floor", nonnegative=True)
        normalized_budget = _finite(
            normalized_budget, "normalized budget", nonnegative=True)
        if mean_floor <= 0:
            raise ValueError("the absolute mean floor must be strictly positive")
        mean_floor_enclosed = abs(mean) >= mean_floor
        normalized_upper = distance / mean_floor
        normalized_budget_met = normalized_upper <= normalized_budget
    else:
        mean_floor_enclosed = True
        normalized_upper = None
        normalized_budget_met = True

    conditions = {
        "stage_graph_complete": True,
        "same_source_pair_preserved": True,
        "row_map_applied_exactly_once": True,
        "every_stage_observed_enclosed": all(
            row["observed_enclosed"] for row in transported),
        "absolute_cache_budget_met": distance <= absolute_budget,
        "observed_final_error_enclosed": observed_final <= distance,
        "absolute_mean_floor_enclosed": mean_floor_enclosed,
        "normalized_budget_met_when_claimed": normalized_budget_met,
        "argmax_preserved_when_claimed": (
            True if margin is None else distance < margin / 2.0
        ),
    }
    result = {
        "schema_version": "pldr-same-source-bridge-record-v2",
        "pair_id": pair_id,
        "source_a_sha256": source_a,
        "source_b_sha256": source_b,
        "transported_stages": transported,
        "final_absolute_upper": distance,
        "absolute_budget": absolute_budget,
        "observed_final_error": observed_final,
        "logit_margin_lower": margin,
        "reference_mean": mean,
        "reference_mean_abs_lower": mean_floor,
        "normalized_upper": normalized_upper,
        "normalized_budget": normalized_budget,
        "conditions": conditions,
        "decision": "CONFIRMED" if all(conditions.values()) else "NOT_CONFIRMED",
    }
    result["record_sha256"] = _digest(result)
    return result
