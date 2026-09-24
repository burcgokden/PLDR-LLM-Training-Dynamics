from __future__ import annotations

from pathlib import Path
import sys


ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
CONFIRM = Path(__file__).resolve().parents[1] / "confirm"
sys.path.insert(0, str(ANALYSIS))
sys.path.insert(0, str(CONFIRM))

from composite_confirmation_analysis import RAW_ARRAYS  # noqa: E402
from program_energy_protocol_specs import STAGES  # noqa: E402


def test_n_record_contains_sufficient_tensors_not_derived_verdict_inputs():
    forbidden = {
        "capacity_dimensions",
        "fisher_ritz_intervals",
        "full_hessian_ritz_intervals",
        "hessian_residual_bounds",
        "sector_operators_construction",
        "sector_operators_validation",
        "common_metrics",
        "forcing_construction",
        "forcing_validation",
        "quadratic_coefficients",
        "direct_operator_gains",
        "direct_additive_bounds",
        "clip_mask_fractions",
        "decay_mask_fractions",
    }
    assert RAW_ARRAYS["N"].isdisjoint(forbidden)
    assert {
        "construction_observation_factors",
        "validation_observation_factors",
        "validation_full_hessian_matrices",
        "validation_hessian_fisher_matrices",
        "normal_states_construction",
        "normal_states_validation",
        "direct_initial_defects_construction",
        "direct_observed_defects_construction",
        "direct_initial_defects_validation",
        "direct_observed_defects_validation",
    } <= RAW_ARRAYS["N"]


def test_a_record_forces_analyzer_to_construct_stage_envelopes():
    assert RAW_ARRAYS["A"].isdisjoint({
        "operator_norms", "additive_defects", "argmax_equal",
        "logit_margins", "caller_pass",
    })
    assert {
        "construction_row_defects",
        "construction_stage_differences",
        "observed_stage_differences",
        "reference_logits",
        "comparison_logits",
    } <= RAW_ARRAYS["A"]


def test_t_contract_is_primary_only_before_replication_gate():
    stage = next(row for row in STAGES if row["id"] == "T")
    assert "primary_seed_bound" in stage["required_checks"]
    assert "four_primary_landmarks" in stage["required_checks"]
    assert "two_distinct_seeds" not in stage["required_checks"]
