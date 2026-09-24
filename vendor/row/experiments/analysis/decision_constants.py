"""Versioned decision designs for the E5 and E6 confirmation studies.

Every attempt is a complete configuration.  A censoring repeat uses a
new attempt identifier, a fresh seed block, its own sample size and,
for E6, its own exact null calibration and alpha allocation.  No
decision function mutates a first-attempt configuration.
"""

DECISION_API_VERSION = "3.1.0"

RAW_ENDPOINT_FIELDS = (
    "word", "act", "probe_steps", "rowmap_samples", "normalization",
)
FORBIDDEN_ENDPOINT_SUMMARIES = (
    "entry", "collapsed", "collapse_step", "J",
)

E5_SPEC = dict(
    version=DECISION_API_VERSION,
    arms=("captured", "separated", "gateheld"),
    predictions={
        "captured": "quiet-tail",
        "separated": "sep-window",
        "gateheld": "unsep-window",
    },
    attempts=(
        dict(attempt_id="e5-primary", seed_block="e5-primary-fresh",
             seeds_per_arm=3),
        dict(attempt_id="e5-censoring-repeat",
             seed_block="e5-repeat-fresh", seeds_per_arm=6),
    ),
    cycle_fields=("r_min", "q_step", "m_cyc", "t_cyc", "n_act"),
    burn_frac=0.1,
    min_persistence_probes=3,
)

E6_SPEC = dict(
    version=DECISION_API_VERSION,
    # Two simultaneously certified cells on each side in every attempt.
    attempts=(
        dict(attempt_id="e6-primary", seed_block="e6-primary-fresh",
             n_cells=4, seeds_per_cell=10, cell_majority=7,
             cells_required=4, alpha=0.025),
        dict(attempt_id="e6-censoring-repeat",
             seed_block="e6-repeat-fresh", n_cells=4,
             seeds_per_cell=12, cell_majority=8,
             cells_required=4, alpha=0.025),
    ),
    study_alpha=0.05,
    design_search=dict(
        power_floor=0.80,
        candidates={
            "e6-primary": (
                dict(n_cells=4, seeds_per_cell=10,
                     cell_majority=7, cells_required=4),
                dict(n_cells=4, seeds_per_cell=11,
                     cell_majority=8, cells_required=4),
            ),
            "e6-censoring-repeat": (
                dict(n_cells=4, seeds_per_cell=12,
                     cell_majority=8, cells_required=4),
                dict(n_cells=4, seeds_per_cell=13,
                     cell_majority=9, cells_required=4),
            ),
        },
    ),
    null_family=dict(
        p_match=0.5,
        delta_max=0.25,
        delta_step=0.0125,
        w_max=1.0,
        w_step=0.02,
    ),
    alt_planning=dict(p_match=0.9, delta=0.05, w=0.25),
    alt_report_grid=dict(
        p_list=(0.7, 0.8, 0.9),
        w_list=(0.0, 0.25, 0.5, 0.75, 1.0),
        delta_list=(0.0, 0.1, 0.2),
    ),
    pilot=dict(
        # More than one cell on each predicted side.
        n_cells_per_side=2,
        seeds_per_cell=8,
        p_grid=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        delta_grid=(0.0, 0.1, 0.2, 0.25),
        w_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
        lr_confidence=0.95,
        mc_calibration_reps=2000,
        mc_error_alpha=0.01,
        precision_target_width=0.20,
        max_cells_per_side=8,
        max_seeds_per_cell=24,
    ),
    # Compatibility names consumed by generated explanatory prose;
    # decisions use the joint pilot region and cap above.
    pilot_bands=dict(p_match_min=0.8, w_seed_max=0.5,
                     delta_cell_max=0.15),
    burn_frac=0.1,
    min_persistence_probes=3,
)


def _e5_attempt(template, J_crit, entry_bound, q_step, m_quantum,
                thm_tier):
    return dict(
        rule="e5",
        version=DECISION_API_VERSION,
        attempt_id=template["attempt_id"],
        seed_block=template["seed_block"],
        arms=list(E5_SPEC["arms"]),
        seeds_per_arm=int(template["seeds_per_arm"]),
        predictions=dict(E5_SPEC["predictions"]),
        raw_fields=list(RAW_ENDPOINT_FIELDS),
        forbidden_summaries=list(FORBIDDEN_ENDPOINT_SUMMARIES),
        cycle_fields=list(E5_SPEC["cycle_fields"]),
        burn_frac=float(E5_SPEC["burn_frac"]),
        min_persistence_probes=int(E5_SPEC["min_persistence_probes"]),
        J_crit=float(J_crit),
        entry_bound=int(entry_bound),
        q_step=float(q_step),
        m_quantum=float(m_quantum),
        thm_tier=(dict(thm_tier) if thm_tier else None),
    )


def e5_config(J_crit, entry_bound, q_step, m_quantum, thm_tier=None):
    """Build the complete E5 study object with two typed attempts."""
    attempts = [
        _e5_attempt(t, J_crit, entry_bound, q_step, m_quantum,
                    thm_tier)
        for t in E5_SPEC["attempts"]
    ]
    return dict(
        rule="e5-study",
        version=DECISION_API_VERSION,
        attempts=attempts,
        repeat_only_after="INDETERMINATE",
        second_indeterminate="INDETERMINATE",
    )


def _e6_attempt(template, total_threshold, J_crit, calibration=None):
    return dict(
        rule="e6",
        version=DECISION_API_VERSION,
        attempt_id=template["attempt_id"],
        seed_block=template["seed_block"],
        n_cells=int(template["n_cells"]),
        seeds_per_cell=int(template["seeds_per_cell"]),
        cell_majority=int(template["cell_majority"]),
        cells_required=int(template["cells_required"]),
        alpha=float(template["alpha"]),
        total_threshold=int(total_threshold),
        J_crit=float(J_crit),
        burn_frac=float(E6_SPEC["burn_frac"]),
        min_persistence_probes=int(E6_SPEC["min_persistence_probes"]),
        raw_fields=list(RAW_ENDPOINT_FIELDS),
        forbidden_summaries=list(FORBIDDEN_ENDPOINT_SUMMARIES),
        calibration=(dict(calibration) if calibration else None),
    )


def e6_config(calibrations, J_crit, attempt_templates=None):
    """Build the complete E6 study object.

    ``calibrations`` must contain one exact-enumeration record per
    declared attempt, in order.  Each record supplies its own
    threshold and is retained verbatim for provenance.
    """
    templates = (E6_SPEC["attempts"] if attempt_templates is None
                 else tuple(attempt_templates))
    if len(calibrations) != len(templates):
        raise ValueError("one E6 calibration is required per attempt")
    attempts = []
    for template, cal in zip(templates, calibrations):
        if "threshold" not in cal:
            raise ValueError("each E6 calibration needs a threshold")
        attempts.append(_e6_attempt(
            template, cal["threshold"], J_crit, calibration=cal))
    alpha_sum = sum(a["alpha"] for a in attempts)
    if alpha_sum > E6_SPEC["study_alpha"] + 1e-12:
        raise ValueError("attempt alpha allocations exceed study alpha")
    return dict(
        rule="e6-study",
        version=DECISION_API_VERSION,
        study_alpha=float(E6_SPEC["study_alpha"]),
        alpha_sum=float(alpha_sum),
        attempts=attempts,
        repeat_only_after="INDETERMINATE",
        second_indeterminate="INDETERMINATE",
    )
