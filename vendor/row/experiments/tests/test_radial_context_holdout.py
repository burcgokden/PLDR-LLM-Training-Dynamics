from analysis.analyze_radial_context_holdout import decide_outcome
from confirm.radial_context_holdout_specs import (
    DECISION_POLICY,
    RESOURCE_BUDGET,
    campaign_design,
    validate_design,
)
from scripts.stage_radial_context_holdout import (
    PREVIOUS_MEASUREMENT_CHUNKS,
    REGISTRY_CHUNKS,
    TERMINAL_BATCH_CHUNKS,
)


def test_holdout_design_freezes_two_gpu_budget_and_disjoint_contexts():
    design = campaign_design()
    validate_design(design)
    assert DECISION_POLICY["minimum_radial_eligible_maps_per_heldout_cell"] == 64
    assert DECISION_POLICY["minimum_overall_raw_sign_agreement_fraction"] == 0.90
    assert DECISION_POLICY["minimum_powered_cell_raw_sign_agreement_fraction"] == 0.70
    assert RESOURCE_BUDGET["working_aggregate_reserved_device_hours"] == 0.20
    assert RESOURCE_BUDGET["hard_aggregate_reserved_device_hours"] == 0.50
    assert not set(REGISTRY_CHUNKS) & set(TERMINAL_BATCH_CHUNKS)
    assert not PREVIOUS_MEASUREMENT_CHUNKS & (
        set(REGISTRY_CHUNKS) | set(TERMINAL_BATCH_CHUNKS)
    )


def test_holdout_decision_support_refutation_and_insufficiency():
    counts = [96] * 18
    outcome, inadequate, low = decide_outcome(
        technical=True,
        eligible_counts=counts,
        cell_agreement_fractions=[0.95] * 18,
        overall_agreement_fraction=0.95,
    )
    assert (outcome, inadequate, low) == ("supported", [], [])

    outcome, inadequate, low = decide_outcome(
        technical=True,
        eligible_counts=counts,
        cell_agreement_fractions=[0.95] * 17 + [0.69],
        overall_agreement_fraction=0.94,
    )
    assert outcome == "refuted"
    assert inadequate == []
    assert low == [17]

    outcome, inadequate, low = decide_outcome(
        technical=True,
        eligible_counts=[96] * 17 + [0],
        cell_agreement_fractions=[0.95] * 17 + [None],
        overall_agreement_fraction=0.95,
    )
    assert outcome == "insufficient"
    assert inadequate == [17]
    assert low == []

    outcome, _, _ = decide_outcome(
        technical=False,
        eligible_counts=counts,
        cell_agreement_fractions=[0.95] * 18,
        overall_agreement_fraction=0.95,
    )
    assert outcome == "insufficient"
