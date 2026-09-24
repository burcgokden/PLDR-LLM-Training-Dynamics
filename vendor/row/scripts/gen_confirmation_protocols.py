#!/usr/bin/env python3
"""Generate the structured-path direct-certificate E0-E8 specifications."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
OUT = EXPERIMENTS / "protocols" / "row_map_confirmation"
TEX_OUT = ROOT / "docs" / "figures" / "confirmation_protocol_table.tex"
sys.path.insert(0, str(EXPERIMENTS / "confirm"))

import campaign_design  # noqa: E402
import full_stack_protocol_specs  # noqa: E402
import campaign_record  # noqa: E402
import estimator_manifest  # noqa: E402
import protocol_units  # noqa: E402
import schedule_clock  # noqa: E402


ALPHA_FAMILY = 0.05

COMMON = {
    "clock": (
        "Every checkpoint and observation uses the global optimizer step. "
        "The segment offset plus segment-local step must equal that clock."
    ),
    "binding": (
        "The raw record binds the protocol, closed schema, source manifest, "
        "model checkpoint, data order, and tokenizer by recomputed SHA-256."
    ),
    "missingness": (
        "A missing or failed primary measurement is recorded as null with a "
        "reason code and yields INCOMPLETE, never a numerical substitute."
    ),
    "multiplicity": (
        "E0 through E6 are exact-grid deterministic decisions. E7 and E8 use "
        "exact paired sign randomization with Bonferroni control at familywise "
        "alpha 0.05. No large-sample approximation enters a primary decision."
    ),
}


SPECS = {
    "E0": {
        "slug": "joint_regularity_cover",
        "title": "Joint regularity and physical chordal row-cover enclosure",
        "depends_on": ["Q0"],
        "target": (
            "The exact row-map calculus, joint row-parameter derivative "
            "bounds, and deterministic transfer from a pairwise-chord "
            "Jacobian cover to the physical segment tube."
        ),
        "estimand": (
            "Maximum held-out row-to-cover radius ratio, held-out "
            "Jacobian-to-certificate ratio, exact cover decomposition "
            "residual, and mixed- and third-derivative enclosure ratios."
        ),
        "population": (
            "Registered layers, seeds, checkpoints, cover points, and "
            "held-out rows from the declared compact tubes."
        ),
        "unit": "seed by layer by checkpoint tube",
        "sample_size": 96,
        "tests": 5,
        "effect_sd": 0.40,
        "allocation": (
            "Eight seeds crossed with all criterion layers and twelve fixed "
            "global checkpoints; row-cover and validation points are frozen."
        ),
        "procedure": [
            "Declare the registered rows and every pairwise physical chord.",
            "Subdivide each chord without filling its ambient coordinate box.",
            "Reconstruct each complete d_k by d_k Jacobian with d_k basis JVPs.",
            "Enclose the full Jacobian matrix with exact rational interval jets.",
            "Bound its operator norm by a validated induced-norm inequality.",
            "Verify that every held-out row lies in the declared row cover.",
            "Evaluate mixed and third derivatives on the optimizer segments.",
            "Test dense held-out rows against grid plus numerical plus L2 h.",
        ],
        "decision": (
            "CONFIRMED requires cover identity residual at most 1e-10 and "
            "every held-out cover-membership, held-out Jacobian, mixed-"
            "derivative, and third-derivative enclosure ratio at most one."
        ),
        "thresholds": {
            "cover_identity_error_max": 1e-10,
            "heldout_jacobian_ratio_max": 1.0,
            "heldout_cover_radius_ratio_max": 1.0,
            "mixed_derivative_enclosure_ratio_max": 1.0,
            "third_derivative_enclosure_ratio_max": 1.0,
        },
        "exclusions": (
            "Only unreadable checkpoints or absent declared tubes are "
            "unavailable. A large derivative or loose cover is retained."
        ),
        "contingency": (
            "Refine the frozen chord subdivision or use outward-rounded "
            "Taylor models without changing the direct endpoint."
        ),
        "prediction": "The finite cover encloses the direct row Jacobian tube.",
    },
    "E1": {
        "slug": "exact_optimizer_transport",
        "title": "Exact operation-ordered AdamW transport",
        "depends_on": ["E0"],
        "target": (
            "The matrix-valued lifted recurrence with the implemented "
            "clipping, moments, bias correction, adaptive denominator, "
            "decay, Jacobian motion, cover motion, and interventions."
        ),
        "estimand": (
            "Full AdamW replay residual, lifted recurrence residual, defect "
            "recomposition residual, primitive enclosure ratios, applied "
            "learning rate, bias-corrected step, schedule increment, decay "
            "coefficient, and optimizer-state clock."
        ),
        "population": "Registered training steps and named tensor groups.",
        "unit": "seed by tensor group by global optimizer step",
        "sample_size": 64,
        "tests": 6,
        "effect_sd": 0.50,
        "allocation": (
            "Eight seeds, four named tensor groups, and two fixed steps per "
            "group; all tensor names are registered before launch."
        ),
        "procedure": [
            "Retain pre-step parameters, raw and clipped gradients, and moments.",
            "Replay bias correction, adaptive denominator, loss step, and decay.",
            "Tie lifted alpha and beta1 to the snapshot step and optimizer ledger.",
            "Retain the applied rate, schedule increment, decay coefficient, and clock.",
            "Recompose normal, decay, row, Taylor, coordinate, and intervention terms.",
            "Compare the lifted identity before applying any spectral summary.",
        ],
        "decision": (
            "CONFIRMED requires all three exact identity residuals and the "
            "post-step boundary residual at most 1e-10, with coordinate-motion "
            "and Taylor enclosure ratios at most one."
        ),
        "thresholds": {
            "adamw_ledger_residual_max": 1e-10,
            "lifted_recurrence_residual_max": 1e-10,
            "defect_recomposition_residual_max": 1e-10,
            "intervention_boundary_residual_max": 1e-10,
            "coordinate_motion_enclosure_ratio_max": 1.0,
            "taylor_remainder_enclosure_ratio_max": 1.0,
        },
        "exclusions": (
            "Optimizer modes outside the registered AdamW implementation are "
            "outside the population and cannot be silently recoded."
        ),
        "contingency": (
            "Increase retained tensor precision or interval width while "
            "preserving the exact operation order."
        ),
        "prediction": "The complete lifted identity closes term by term.",
    },
    "E2": {
        "slug": "normal_operator_spectrum",
        "title": "Normal operator and nominal two-sided spectrum",
        "depends_on": ["E1"],
        "target": (
            "The self-adjoint reference normal operator, its positive lower "
            "edge, strict Jury margins, and the explicit normal residual."
        ),
        "estimand": (
            "Lower and upper normal edges, minimum spectrum-derived Jury "
            "margin, separate lower and upper schedule margins, spectral "
            "loading, self-adjoint and normal-operator identity residuals, "
            "and primitive-linearization and nonlinear enclosure ratios."
        ),
        "population": "Registered modes at the frozen nominal checkpoints.",
        "unit": "seed by layer by nominal checkpoint",
        "sample_size": 80,
        "tests": 9,
        "effect_sd": 0.45,
        "allocation": "Ten seeds crossed with eight fixed layer-checkpoint units.",
        "procedure": [
            "Construct the primitive derivatives of Q and Psi at the registered center.",
            "Construct the registered self-adjoint operator on the ambient stack.",
            "Measure self-adjointness and both normal spectral edges.",
            "Derive every scalar Jury matrix from those edges and the registered optimizer scalars.",
            "Compute lower and upper schedule margins and spectral loading.",
            "Enclose the full primitive linearization mismatch and nonlinear remainder.",
        ],
        "decision": (
            "CONFIRMED requires positive lower and upper schedule margins, "
            "spectral loading below one, a positive lower normal edge and "
            "minimum Jury margin, "
            "self-adjoint and normal identity residuals at most 1e-10, and "
            "both primitive residual-enclosure ratios at most one."
        ),
        "thresholds": {
            "lower_normal_edge_min": 0.0,
            "jury_margin_min": 0.0,
            "jury_lower_margin_min": 0.0,
            "jury_upper_margin_min": 0.0,
            "spectral_loading_max": 1.0,
            "normal_operator_self_adjoint_residual_max": 1e-10,
            "normal_operator_identity_residual_max": 1e-10,
            "normal_linearization_residual_ratio_max": 1.0,
            "nonlinear_residual_enclosure_ratio_max": 1.0,
        },
        "exclusions": (
            "A neutral or unstable mode is retained and fails the registered "
            "condition; it is not removed as an outlier."
        ),
        "contingency": (
            "Shrink the registered auxiliary box only by a frozen rule and "
            "repeat the full E0-E2 dependency chain."
        ),
        "prediction": "Nominal normal modes satisfy the strict two-sided band.",
    },
    "E3": {
        "slug": "structured_path_lyapunov",
        "title": "Structured primitive-family and pathwise Lyapunov certificate",
        "depends_on": ["E2"],
        "target": (
            "The discrete Lyapunov construction at a nominal matrix, its "
            "exact extension to the correlated multiaffine primitive box, "
            "and contraction on every registered graph window."
        ),
        "estimand": (
            "Lyapunov equation residual, metric lower edge, nominal gain, "
            "exact structured-family gain and slack, corner gain, "
            "multiaffine reconstruction, primitive-box membership, path-"
            "window gain, and observed H-gain."
        ),
        "population": (
            "Registered primitive boxes, their correlated corner matrices, "
            "and admissible path-graph edges."
        ),
        "unit": "seed by layer by primitive path cell",
        "sample_size": 64,
        "tests": 5,
        "effect_sd": 0.50,
        "allocation": "Eight seeds crossed with eight fixed layer-box certificates.",
        "procedure": [
            "Construct the nominal Lyapunov witness with exact rational algebra.",
            "Charge the complete equation residual against its dissipation margin.",
            "Form the alpha, decay, and normal-eigenvalue primitive box.",
            "Enumerate its eight correlated multiaffine corner matrices.",
            "Prove every corner LMI by exact LDL and use convexity for the box.",
            "Enumerate every admissible graph path over the registered window.",
            "Evaluate observed H-gains and barycentric reconstruction residuals.",
        ],
        "decision": (
            "CONFIRMED requires a positive metric, exact structured-family "
            "gain and every registered path-window gain below one, positive "
            "structured slack, corner and observed gains no greater than the "
            "exact bound, multiaffine residual at most 1e-10, and primitive-"
            "box enclosure ratio at most one."
        ),
        "thresholds": {
            "nominal_lyapunov_residual_max": 1e-10,
            "metric_eigenvalue_min": 0.0,
            "structured_family_q_max": 1.0,
            "structured_family_slack_min": 0.0,
            "structured_slack_identity_error_max": 1e-10,
            "corner_gain_excess_max": 0.0,
            "multiaffine_reconstruction_residual_max": 1e-10,
            "primitive_box_enclosure_ratio_max": 1.0,
            "path_window_gain_max": 1.0,
            "observed_gain_excess_max": 0.0,
        },
        "exclusions": (
            "A primitive state outside its registered path cell is reported "
            "as a graph-membership failure and is not omitted."
        ),
        "contingency": (
            "Subdivide the primitive cell or introduce cell-indexed metrics "
            "and repeat all edge and window checks."
        ),
        "prediction": (
            "Correlated primitive families contract on every admissible path "
            "window."
        ),
    },
    "E4": {
        "slug": "complete_defect_decay",
        "title": "Complete forcing and defect envelope",
        "depends_on": ["E1", "E2"],
        "target": (
            "The complete disturbance W equals the reference forcing plus "
            "every named optimizer and Jacobian-preconditioner defect. A "
            "nonnegative matrix comparison P v at most kappa v closes the "
            "registered affine tail recurrence for every later update."
        ),
        "estimand": (
            "Forcing, complete defect, complete disturbance, persistent "
            "source floor, d-infinity plus K r^n envelope ratio, transient "
            "rate, positive-comparison witness ratio, and recomposition residual."
        ),
        "population": "Registered tail steps and criterion-layer lifted states.",
        "unit": "seed by layer by global optimizer step",
        "sample_size": 80,
        "tests": 4,
        "effect_sd": 0.45,
        "allocation": "Ten seeds crossed with eight fixed future steps.",
        "procedure": [
            "Record every primitive term and its source class before forming a norm.",
            "Recompose F plus E and compare with the realized lifted disturbance.",
            "Evaluate each primitive deterministic bound and the aggregate H-bound.",
            "Derive d-infinity plus K r^n from frozen source bounds and tail offsets.",
            "Construct the positive comparison matrix and both forcing vectors.",
            "Verify P v at most kappa v and the affine recurrence exactly.",
        ],
        "decision": (
            "CONFIRMED requires recomposition residual at most 1e-10, "
            "the derived persistent-plus-geometric envelope ratio at most one, "
            "a transient rate below one, and P v over kappa v at most one."
        ),
        "thresholds": {
            "defect_recomposition_residual_max": 1e-10,
            "envelope_ratio_max": 1.0,
            "geometric_rate_max": 1.0,
            "positive_comparison_witness_ratio_max": 1.0,
        },
        "exclusions": (
            "A small primitive defect is still recorded. Missing any term "
            "makes the complete-ledger endpoint unavailable."
        ),
        "contingency": (
            "Widen primitive interval bounds before changing the frozen "
            "geometric rate or certificate checkpoint."
        ),
        "prediction": "The complete disturbance lies below its geometric envelope.",
    },
    "E5": {
        "slug": "direct_tail_self_map",
        "title": "Componentwise forward-invariant direct tail",
        "depends_on": ["E0", "E3", "E4"],
        "target": (
            "The lifted and componentwise radius budgets that prove the "
            "complete direct certificate tube maps into its successor."
        ),
        "estimand": (
            "Initial membership, one-edge and closed-window lifted budgets, "
            "minimum path-prefix and auxiliary budgets, held-out image margin, "
            "escape count, and family-box enclosure ratio."
        ),
        "population": "Registered certificate tubes and held-out one-step images.",
        "unit": "seed by layer by direct certificate tube",
        "sample_size": 64,
        "tests": 4,
        "effect_sd": 0.50,
        "allocation": "Eight seeds crossed with eight fixed tube transitions.",
        "procedure": [
            "Freeze centers, radii, derivative bounds, and outward rounding.",
            "Evaluate every registered path prefix and the block budget "
            "gamma B plus Omega.",
            "Retain q B plus w as the one-edge transition diagnostic.",
            "Evaluate every auxiliary component budget a plus ell B plus L b.",
            "Check initial membership and held-out images without changing the box.",
        ],
        "decision": (
            "CONFIRMED requires a strictly contracting registered window, all "
            "membership, prefix, block, and auxiliary budget margins nonnegative, "
            "zero one-step escapes, and family-box enclosure ratio at most one."
        ),
        "thresholds": {
            "initial_membership_margin_min": 0.0,
            "lifted_budget_slack_min": 0.0,
            "path_window_gain_max": 1.0,
            "block_lifted_budget_slack_min": 0.0,
            "prefix_budget_slack_min": 0.0,
            "auxiliary_budget_slack_min": 0.0,
            "heldout_image_margin_min": 0.0,
            "one_step_escape_count_max": 0.0,
            "family_box_enclosure_ratio_max": 1.0,
        },
        "exclusions": (
            "A held-out escape remains a failed self-map check even if a later "
            "state returns to the tube."
        ),
        "contingency": (
            "Recompute a larger tube from primitive bounds and repeat E3-E5 "
            "without reusing the failed certificate."
        ),
        "prediction": "Every certified direct tube maps into its successor.",
    },
    "E6": {
        "slug": "finite_entry_horizon",
        "title": "Computable all-layer finite-entry horizon",
        "depends_on": ["E0", "E3", "E4", "E5"],
        "target": (
            "The path-block product-convolution envelope, strict complete "
            "persistent-floor margin, and sustained finite all-layer entry."
        ),
        "estimand": (
            "Predicted and observed entry step, identity error, complete "
            "persistent floor, registered path length and gain, block and schedule "
            "product-convolution bounds, criterion, direct upper bound, and "
            "sustained-entry indicator."
        ),
        "population": "Held-forward trajectories from frozen certificate checkpoints.",
        "unit": "seed trajectory",
        "sample_size": 64,
        "tests": 3,
        "effect_sd": 0.50,
        "allocation": "Eight starting checkpoints for each of eight seeds.",
        "procedure": [
            "Freeze the graph, path length, window gain, metric bounds, vanishing "
            "and persistent block disturbances, cover floor, and criterion.",
            "Compute the block monotonicity start and first certified horizon integer.",
            "Compute the time-indexed schedule product-convolution and its horizon.",
            "Seal the prediction before opening future checkpoints.",
            "Measure sustained all-layer entry and the complete-floor condition.",
        ],
        "decision": (
            "CONFIRMED requires every registered path window to contract, every "
            "predicted horizon to cover observed sustained entry, a positive "
            "criterion-to-complete-floor margin, and direct entry below the "
            "registered physical criterion. Horizon "
            "sharpness is reported descriptively."
        ),
        "thresholds": {
            "coverage_min": 1.0,
            "floor_margin_min": 0.0,
            "path_window_gain_max": 1.0,
            "sustained_entry_fraction_min": 1.0,
        },
        "exclusions": (
            "A trajectory ending before the predicted horizon is censored and "
            "makes the endpoint incomplete unless every seed is equally extended."
        ),
        "contingency": "Apply one fixed equal extension to every censored trajectory.",
        "prediction": "The computed horizon covers sustained all-layer entry.",
    },
    "E7": {
        "slug": "certificate_interventions",
        "title": "Paired schedule and certificate interventions",
        "depends_on": ["E3", "E4", "E5"],
        "target": (
            "Causal sensitivity of direct collapse to robust contraction and "
            "complete disturbance under registered anneal, near-edge pulse, "
            "and sham branches while the remaining certificate clauses are held."
        ),
        "estimand": (
            "Within-seed contrasts in robust q, complete disturbance envelope, "
            "predicted horizon and direct upper bound, with lower and upper "
            "Jury margins, spectral loading, schedule product-convolution, "
            "and avalanche rate, size, and support retained as mechanism "
            "diagnostics. Held-out loss is a separate descriptive outcome."
        ),
        "population": "Matched checkpoint branches from registered seeds.",
        "unit": "seed-matched three-arm branch set",
        "sample_size": 72,
        "tests": 6,
        "effect_sd": 0.45,
        "allocation": (
            "Twenty-four seeds crossed with three checkpoints form seventy-two "
            "matched sets; all three branch labels are fixed before outcomes."
        ),
        "required_arms": [
            "contraction_improving", "defect_increasing", "sham",
        ],
        "procedure": [
            "Fork every arm from identical model, optimizer, token, and clock state.",
            "Apply a stable loading reduction, a nominal-band-preserving near-edge pulse, or sham.",
            "Recompute every affected certificate constant in each arm.",
            "Measure Jury margins, loading, product-convolution, and Jacobian avalanches.",
            "Compare paired robust gain, disturbance, horizon, and direct endpoint.",
        ],
        "decision": (
            "CONFIRMED requires the contraction arm to lower robust q, horizon, "
            "and direct upper bound, while the defect arm raises disturbance, "
            "horizon, and direct upper bound, under one-sided paired bounds."
        ),
        "thresholds": {
            "contraction_q_contrast_max": 0.0,
            "contraction_horizon_contrast_max": 0.0,
            "contraction_direct_contrast_max": 0.0,
            "defect_envelope_contrast_min": 0.0,
            "defect_horizon_contrast_min": 0.0,
            "defect_direct_contrast_min": 0.0,
        },
        "exclusions": (
            "If any arm is unavailable, the entire matched set is unavailable "
            "for paired inference."
        ),
        "contingency": (
            "Use the registered stronger dose tier only when its manipulation "
            "check fails and the other certificate clauses remain valid."
        ),
        "prediction": (
            "Schedule-controlled certificate changes move the horizon and "
            "Jacobian-burst observables in the predicted direction."
        ),
    },
    "E8": {
        "slug": "heldout_scale_discrimination",
        "title": "Held-out scale transfer, observable bridge, and RG closure",
        "depends_on": ["E6", "E7"],
        "target": (
            "Out-of-sample specificity of the complete direct bound and finite "
            "horizon, plus the direct-to-inference order-parameter bridge, "
            "and deductive-sector renormalization closure across unseen "
            "schedule, width, depth, and token-block cells."
        ),
        "estimand": (
            "Theorem-bound ratio, sufficient-horizon coverage, direct entry "
            "relative to the physical criterion, order-parameter bridge "
            "ratio, paired model-score error against frozen baselines, and "
            "the exact density, LayerNorm, softmax, and blocking-semigroup "
            "residuals, deductive and conditional full-layer closure ratios, "
            "and descriptive avalanche scaling observables."
        ),
        "population": "Held-out seeds, schedules, and architecture cells.",
        "unit": "held-out architecture trajectory",
        "block_sizes": [2, 4, 8],
        "sample_size": 72,
        "tests": 12,
        "effect_sd": 0.45,
        "allocation": (
            "Eight held-out seeds cross three architecture and three schedule "
            "cells. Every unit is measured at block sizes 2, 4, and 8, and "
            "the held-out partition hash is frozen before fitting."
        ),
        "required_arms": [
            "theory", "constant_rate", "loss_only",
            "unconstrained_trend",
        ],
        "procedure": [
            "Recompute every derivative, cover, optimizer, metric, and tube constant.",
            "Cross the frozen width-depth cells with warm-up, peak-rate, and floor cells.",
            "Fit the analytical and baseline predictors on the training partition.",
            "Freeze code, constants, and held-out predictions by digest.",
            "Collect independent and cached deductive tensors on held-out prompts.",
            "Compute RMSE, tensor mean, row spread, downstream gain, and bridge ratio.",
            "Block rotated queries at b=2, 4, and 8 and repeat aligned blocking.",
            "Measure exact density, LayerNorm, and block-logsumexp identities.",
            "Measure deductive closure and the transported full-layer defect sum.",
            "Score avalanche tails and finite-size scaling separately from collapse.",
            "Score bound validity, entry horizon, direct endpoint, and loss separately.",
        ],
        "decision": (
            "CONFIRMED requires theorem-bound ratio at most one, direct entry "
            "in every held-out cell, every predicted horizon to cover the "
            "observed sustained entry, order-parameter bridge ratio at most "
            "one, all four exact RG identity residuals at most 1e-10, "
            "deductive and conditional full-layer RG closure ratios at most "
            "one, and smaller paired analytical model score than every "
            "baseline. A nontrivial RG fixed point, avalanche scaling, "
            "horizon sharpness, and loss are reported separately."
        ),
        "thresholds": {
            "theorem_bound_ratio_max": 1.0,
            "heldout_entry_fraction_min": 1.0,
            "horizon_coverage_min": 1.0,
            "order_parameter_bridge_ratio_max": 1.0,
            "order_parameter_bridge_identity_error_max": 1e-10,
            "rg_density_identity_residual_max": 1e-10,
            "rg_layernorm_scale_residual_max": 1e-10,
            "rg_softmax_aggregation_residual_max": 1e-10,
            "rg_repeated_block_semigroup_residual_max": 1e-10,
            "rg_deductive_closure_ratio_max": 1.0,
            "rg_full_layer_closure_ratio_max": 1.0,
            "paired_model_score_difference_max": 0.0,
        },
        "exclusions": (
            "No held-out cell may move to training after any outcome is read."
        ),
        "contingency": (
            "Reduce microbatch size without changing batch order or optimizer "
            "steps; otherwise mark the architecture cell unavailable."
        ),
        "prediction": (
            "The direct certificate transfers, bounds the inference order "
            "parameter, closes under deductive-sector blocking, and "
            "outpredicts frozen baselines."
        ),
    },
}

full_stack_protocol_specs.apply(SPECS)


def qualification_text():
    return """# Q0: Instrument qualification

Q0 runs before campaign initialization and uses no training outcome.

1. Exact linear fixture: dense autograd and basis JVP reconstruction reproduce
   the planted matrix in float64.
2. Planted quadratic and multilinear fixtures: row derivatives, mixed
   row-parameter derivatives, and third-order remainders match analytic values.
3. Implemented PLDR fixture: dense autograd and exactly d_k basis JVPs agree;
   two central-difference steps remain diagnostic.
4. Optimizer fixture: operation-ordered replay matches AdamW after clipping,
   bias correction, adaptive preconditioning, decay, and intervention.
5. Schedule fixture: applied warm-up and cosine endpoints use the registered
   clock, and direct recurrence matches the nonautonomous product-convolution.
6. Observable fixture: an exact linear row map satisfies the RMSE
   order-parameter bridge with its known operator norm.
7. Renormalization fixture: optimizer-time cocycle blocking, aligned token
   blocking, query-density decomposition, LayerNorm rescaling, and block
   log-sum-exp aggregation satisfy their exact identities.
9. Complete-stack fixture: every coordinate of every full Jacobian is
   enumerated for every architecture layer; sparse or invalid layer lists and
   scalar primary charts are rejected.
10. Orthogonal-mode fixture: a stable registered projection paired with an
    expanding orthogonal mode fails the complete block comparison.
11. Normal-closure fixture: B, DQ, R, K, the closure complement, and the
    once-frozen forcing replay on disjoint construction and validation cells.
12. Program-state fixture: branch-resolved clipped AdamW derives its own
    positive radius map and rejects clipping or square-root branch crossings.
13. Runtime and time fixture: outward primitive enclosures charge every
    ordered block, and only finite implemented schedule, frozen checkpoint,
    or controlled infinite tail semantics are accepted.
8. Serialization fixture: duplicate keys, nonfinite numbers, unknown fields,
   stale path bindings, and inconsistent clocks are rejected.

The launcher writes the Q0 result into a fresh namespace. Any failed fixture
stops launch.
"""


def render_protocol(pid, spec):
    dependencies = ", ".join(spec["depends_on"])
    procedure = "\n".join(
        f"{index}. {value}"
        for index, value in enumerate(spec["procedure"], start=1)
    )
    thresholds = "\n".join(
        f"- {name}: {value}" for name, value in spec["thresholds"].items()
    )
    return f"""# {pid}: {spec['title']}

Status: prospective and generated.
Dependencies: {dependencies}.

## Analytical target

{spec['target']}

## Estimand

{spec['estimand']}

## Population and replication

Population: {spec['population']}

Unit of replication: {spec['unit']}

Registered sample size: {spec['sample_size']} units.

Allocation: {spec['allocation']}

## Measurement procedure

{procedure}

{COMMON['clock']}

{COMMON['binding']}

## Primary decision

{spec['decision']}

Registered numerical boundaries:

{thresholds}

{COMMON['multiplicity']}

## Missingness and exclusions

{spec['exclusions']}

{COMMON['missingness']}

## Registered grid and decision procedure

The registration enumerates all
{len(protocol_units.expected_unit_keys(pid))} exact record keys for this
experiment. Completeness is equality of the realized and registered key sets,
including arms and block sizes. Deterministic certificate endpoints use their
printed inequalities. Paired endpoints use the exact finite-sample procedure
stated above.

## Contingency

{spec['contingency']}

## Predicted consequence

{spec['prediction']}
"""


def readme_text(registration):
    rows = "\n".join(
        f"- [{pid}]({pid.lower()}_{SPECS[pid]['slug']}.md): "
        f"{SPECS[pid]['title']}"
        for pid in campaign_record.PROTOCOLS
    )
    return f"""# Direct row-map certificate confirmation program

This directory is generated by scripts/gen_confirmation_protocols.py. It
contains one qualification stage and nine prospective experiments for the
complete direct row-map collapse certificate.

[Q0 instrument qualification](q0_instrument_qualification.md)

{rows}

All raw observations must validate against record_schema.json and the runtime
checks in experiments/confirm/campaign_record.py. The program uses one global
optimizer clock, physical row units, complete d_k by d_k Jacobians, and
on-disk digest verification. Every record key and estimator owner is
enumerated before launch; completeness is exact set equality.

The immutable registration is [registration.json](registration.json).
"""






def _managed_existing():
    if not OUT.exists():
        return set()
    return {
        path
        for path in OUT.iterdir()
        if path.is_file() and path.suffix in {".md", ".json"}
    }




