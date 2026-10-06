"""Regression tests for the sign-complete direct-collapse contract."""

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "analysis"))
sys.path.insert(0, str(EXP / "confirm"))
sys.path.insert(0, str(EXP.parent / "scripts"))

import direct_collapse as C  # noqa: E402
import e0_prime_adapters as A  # noqa: E402
import e0_prime_decision as D  # noqa: E402
import gen_e0_prime_protocol as G  # noqa: E402


def pass_record():
    return {
        "schema_version": D.SCHEMA_VERSION,
        "campaign_id": "e0-prime-fresh-001",
        "bindings": {
            name: character * 64
            for name, character in zip(D.BINDINGS, "abcdef")
        },
        "availability": {
            name: "PASS" for name in D.AVAILABILITY_CHECKS
        },
        "core_checks": {name: "PASS" for name in D.CORE_CHECKS},
        "mechanism_checks": {
            name: "UNRESOLVED" for name in D.MECHANISM_CHECKS
        },
        "eligibility": {
            name: {
                "status": "ELIGIBLE",
                "n": D.REGISTERED_MINIMUMS[name],
                "minimum": D.REGISTERED_MINIMUMS[name],
            }
            for name in D.FAMILIES
        },
        "gate": {"estimate": 0.2, "interval": [0.1, 0.3]},
        "branch_pilot": {"status": "POSITIVE_CAPTURE_RESOLVED"},
        "families": {
            name: {
                "status": "CONCLUSIVE",
                "p_value": 1e-6,
                "direction_pass": True,
            }
            for name in D.FAMILIES
        },
    }


def test_signed_gate_upper_envelope_handles_all_signs():
    positive = C.signed_gate_step(0.5, 0.9, 0.1, 2.0, 1.0)
    zero = C.signed_gate_step(0.5, 0.9, 0.1, 0.0, 1.0)
    negative = C.signed_gate_step(0.5, 0.9, 0.1, -2.0, 1.0)
    assert positive["realized"] == pytest.approx(0.65)
    assert zero["realized"] == pytest.approx(0.45)
    assert negative["realized"] == pytest.approx(0.25)
    assert negative["upper"] == pytest.approx(0.45)
    assert C.gate_sign_branch([-0.3, -0.1]) == "NONPOSITIVE"
    assert C.gate_sign_branch([0.1, 0.3]) == "POSITIVE"
    assert C.gate_sign_branch([-0.1, 0.1]) == "SIGN_UNRESOLVED"


def test_trajectory_local_tube_and_entry_horizon():
    result = C.trajectory_guard_tube(
        score_initial=np.array([2.0, 3.0]),
        threshold_initial=np.array([3.0, 4.0]),
        score_increment_upper=np.array([[0.2, 0.1], [0.1, 0.2]]),
        threshold_increment_lower=np.array([[0.3, 0.2], [0.2, 0.3]]),
    )
    assert result["closed"]
    assert result["minimum_margin"] == pytest.approx(1.0)
    assert C.common_entry_horizon(1.0, 0.1, 0.5) == 4
    assert np.allclose(
        C.geometric_tail_bound(1.0, 0.5, np.arange(5)),
        [1.0, 0.5, 0.25, 0.125, 0.0625],
    )


def test_global_checkpoint_clock_and_strict_json():
    row = A.global_checkpoint_step(
        "ckpt_24000.pt", {"step": 2000},
        {"global_step_offset": 22000},
    )
    assert row["global_step"] == 24000
    assert row["segment_local_step"] == 2000
    with pytest.raises(ValueError, match="disagree"):
        A.global_checkpoint_step(
            "ckpt_24000.pt", {"step": 2000},
            {"global_step_offset": 20000},
        )
    with pytest.raises(ValueError):
        A.nullable_measurement(float("nan"))
    text = A.strict_json_text(A.nullable_measurement(
        None, "CAP_EXHAUSTED"
    ))
    assert json.loads(text)["value"] is None


def test_preconditioner_metadata_and_power_are_qualified():
    cell = {"averaging_preconditioner": A.PRECONDITIONER_METADATA}
    assert A.assemble_preconditioner_metadata(cell, cell)["status"] == "PASS"
    missing = A.assemble_preconditioner_metadata(cell, {})
    assert missing == {
        "status": "MISSING",
        "value": None,
        "reason_code": "MEASUREMENT_ERROR",
    }
    assert G.PLANNING["joint_power_raw"] == pytest.approx(0.846665)
    assert G.PLANNING["joint_power"] == pytest.approx(0.84)
    assert G.REGISTERED_MINIMUMS == D.REGISTERED_MINIMUMS
    schema = G.schema_object()
    assert set(schema["properties"]["availability"]["required"]) \
        == set(D.AVAILABILITY_CHECKS)
    assert set(schema["properties"]["mechanism_checks"]["required"]) \
        == set(D.MECHANISM_CHECKS)


def test_gate_router_is_total_on_finite_sign_intervals():
    assert A.gate_pilot_route(-0.2, [-0.3, -0.1])["status"] \
        == "RUN_NONPOSITIVE_COLLAPSE"
    assert A.gate_pilot_route(0.2, [0.1, 0.3])["status"] \
        == "RUN_POSITIVE_CAPTURE"
    assert A.gate_pilot_route(0.0, [-0.1, 0.1])["status"] \
        == "NOT_RUN_SIGN_UNRESOLVED"
    assert A.gate_pilot_route(float("nan"), [0.0, 1.0])["status"] \
        == "ERROR"


def test_negative_gate_can_pass_the_direct_collapse_theory():
    record = pass_record()
    record["gate"] = {"estimate": -0.2, "interval": [-0.3, -0.1]}
    record["branch_pilot"] = {
        "status": "NONPOSITIVE_COLLAPSE_RESOLVED"
    }
    assert D.decide(record)["outcome"] == "PASS"


def test_total_decision_precedence_and_mechanism_separation():
    record = pass_record()
    record["mechanism_checks"] = {
        name: "FAIL" for name in D.MECHANISM_CHECKS
    }
    assert D.decide(record)["outcome"] == "PASS"
    record["core_checks"]["trajectory_guard_tube"] = "FAIL"
    assert D.decide(record)["outcome"] == "MODEL_REJECTED"
    record = pass_record()
    record["availability"]["strict_json"] = "ERROR"
    record["core_checks"]["trajectory_guard_tube"] = "FAIL"
    assert D.decide(record)["outcome"] == "INCOMPLETE"


def test_unresolved_sign_and_capped_eligibility_are_indeterminate():
    record = pass_record()
    record["gate"] = {"estimate": 0.0, "interval": [-0.1, 0.1]}
    record["branch_pilot"] = {"status": "SIGN_UNRESOLVED"}
    assert D.decide(record)["outcome"] == "INDETERMINATE"
    record = pass_record()
    record["eligibility"]["direct_tail_contraction"] = {
        "status": "MAX_CAP_UNRESOLVED", "n": 20, "minimum": 80
    }
    assert D.decide(record)["outcome"] == "INDETERMINATE"


def test_failed_family_is_not_confirmed_and_bindings_are_checked():
    record = pass_record()
    record["families"]["guard_tube_closure"]["direction_pass"] = False
    assert D.decide(record)["outcome"] == "NOT_CONFIRMED"
    record = pass_record()
    record["eligibility"]["capture_quench_contrast"]["minimum"] = 1
    assert D.decide(record)["outcome"] == "INCOMPLETE"
    record = pass_record()
    D.authorize_downstream(record, record["bindings"])
    stale = copy.deepcopy(record["bindings"])
    stale["source_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="stale"):
        D.authorize_downstream(record, stale)


def test_malformed_records_are_total():
    for value in (None, [], 1, {}, {"schema_version": D.SCHEMA_VERSION}):
        decision = D.decide(value)
        assert decision["outcome"] == "INCOMPLETE"
        assert decision["reasons"]
