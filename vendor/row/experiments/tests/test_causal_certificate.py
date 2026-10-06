import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.build_causal_remainder_certificate import build
from scripts.stage_causal_row_map_confirmation import build_plan, campaign_spec


def test_certificate_adds_outward_jvp_error_to_taylor_radius(tmp_path):
    source = tmp_path / "validated.json"
    source.write_text(json.dumps({
        "schema_version": "pldr-causal-validated-jet-bounds-v1",
        "successor_values_present": False,
        "bounds_valid_on_complete_parameter_segment": True,
        "source_state_digest": "a" * 64,
        "rows": [{
            "point_second_directional_uppers": ["2", "4", "3"],
            "third_directional_modulus": "4",
            "subdivisions": 2,
            "jvp_error_radius": "1/10",
        }],
    }), encoding="utf-8")
    result = build(source)
    assert result["row_taylor_remainder_radius_rational"] == [
        {"numerator": 5, "denominator": 2}]
    assert result["row_jvp_error_radius_rational"] == [
        {"numerator": 1, "denominator": 10}]
    assert result["row_remainder_radius_rational"] == [
        {"numerator": 13, "denominator": 5}]
    assert result["row_remainder_radius_float64_outward"][0] >= 2.6
    assert result["successor_values_present"] is False
    assert result["frozen_before_successor"] is True


def test_certificate_rejects_unbounded_jvp_error(tmp_path):
    source = tmp_path / "invalid.json"
    source.write_text(json.dumps({
        "schema_version": "pldr-causal-validated-jet-bounds-v1",
        "successor_values_present": False,
        "bounds_valid_on_complete_parameter_segment": True,
        "source_state_digest": "b" * 64,
        "rows": [{
            "point_second_directional_uppers": ["1", "1"],
            "third_directional_modulus": "0",
            "subdivisions": 1,
        }],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid schema"):
        build(source)


def test_staged_plan_closes_lock_and_final_analysis_dependencies(tmp_path):
    plan = build_plan(
        tmp_path,
        tmp_path / "tokens.npy",
        tmp_path / "tokenizer.model",
        tmp_path / "registry.json",
    )
    nodes = {node["id"]: node for node in plan["nodes"]}
    assert len(nodes) == len(plan["nodes"])
    assert "analyze-construction-plga" in nodes["construction-lock"][
        "depends_on"]
    assert "--require-complete-grid" in nodes["construction-lock"]["argv"]
    assert nodes["train-causal-heldout-s9444-o24576"]["depends_on"] == [
        "qualification", "construction-lock"]
    final = nodes["finalize-causal-campaign"]
    assert final["stage"] == "campaign-analysis"
    assert "analyze-heldout-plga" in final["depends_on"]
    assert "analyze-heldout-intervention" in final["depends_on"]
    assert "qualification" in final["depends_on"]
    assert "--qualification" in final["argv"]
    assert campaign_spec() == json.loads(json.dumps(campaign_spec()))
