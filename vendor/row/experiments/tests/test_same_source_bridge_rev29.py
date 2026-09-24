from __future__ import annotations

from pathlib import Path
import sys

import pytest


CONFIRM = Path(__file__).resolve().parents[1] / "confirm"
sys.path.insert(0, str(CONFIRM))

from same_source_bridge import (  # noqa: E402
    REQUIRED_STAGES,
    SCHEMA_VERSION,
    construct_same_source_bridge,
)


def fixture():
    pair = "same-pair"
    source_a = "a" * 64
    source_b = "b" * 64
    return {
        "schema_version": SCHEMA_VERSION,
        "pair_id": pair,
        "source_a_sha256": source_a,
        "source_b_sha256": source_b,
        "source_row_distance": 0.1,
        "direct_row_map_upper": 0.01,
        "row_map_defect_upper": 1e-6,
        "observed_row_map_distance": 0.0,
        "absolute_budget": 0.01,
        "stages": [{
            "name": name,
            "pair_id": pair,
            "input_source_a_sha256": source_a,
            "input_source_b_sha256": source_b,
            "gain_upper": 0.8,
            "defect_upper": 1e-6,
            "observed_output_distance": 0.0,
        } for name in REQUIRED_STAGES],
        "observed_final_error": 0.0,
        "logit_margin_lower": 0.1,
        "reference_mean": -0.2,
        "reference_mean_abs_lower": 0.19,
        "normalized_budget": 0.1,
    }


def test_complete_same_source_chain_confirms_absolute_error():
    result = construct_same_source_bridge(fixture())
    assert result["decision"] == "CONFIRMED"
    assert [row["name"] for row in result["transported_stages"]] == list(
        ("row_map", *REQUIRED_STAGES))
    assert result["transported_stages"][1]["input_upper"] == (
        result["transported_stages"][0]["output_upper"])
    assert result["conditions"]["row_map_applied_exactly_once"]
    assert result["conditions"]["absolute_mean_floor_enclosed"]


def test_source_swap_and_omitted_stage_are_semantic_kills():
    payload = fixture()
    payload["stages"][4]["input_source_b_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="changed the source pair"):
        construct_same_source_bridge(payload)
    payload = fixture()
    payload["stages"].pop(3)
    with pytest.raises(ValueError, match="every downstream stage"):
        construct_same_source_bridge(payload)


def test_absolute_budget_and_margin_are_distinct_thresholds():
    payload = fixture()
    payload["absolute_budget"] = 1e-8
    result = construct_same_source_bridge(payload)
    assert not result["conditions"]["absolute_cache_budget_met"]
    assert result["conditions"]["argmax_preserved_when_claimed"]
    assert result["decision"] == "NOT_CONFIRMED"


def test_signed_mean_is_normalized_only_through_its_absolute_floor():
    payload = fixture()
    payload["reference_mean"] = -0.01
    result = construct_same_source_bridge(payload)
    assert not result["conditions"]["absolute_mean_floor_enclosed"]
    assert result["decision"] == "NOT_CONFIRMED"


def test_partial_normalization_claim_is_rejected():
    payload = fixture()
    payload["reference_mean"] = None
    with pytest.raises(ValueError, match="one claim"):
        construct_same_source_bridge(payload)
