#!/usr/bin/env python3
"""Generate the prospective observable-closure protocol and record schema."""

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "protocols"
PROTOCOL = OUT / "e0_prime_observable_closure.md"
SCHEMA = OUT / "e0_prime_record_schema.json"
DESIGN = OUT / "e0_prime_design.json"

FAMILIES = (
    "direct_tail_contraction",
    "guard_tube_closure",
    "capture_quench_contrast",
)
REGISTERED_MINIMUMS = {
    "direct_tail_contraction": 80,
    "guard_tube_closure": 80,
    "capture_quench_contrast": 32,
}
SAMPLE_SIZES = {
    "minimum_effective_tail_and_guard_windows": 80,
    "matched_seeds_per_quench_arm": 32,
    "checkpoint_replays_per_cell": 24,
    "extension_block_steps": 8000,
    "maximum_extension_blocks": 4,
}
PLANNING = {
    "quiet_log_ratio_mean": -0.08,
    "quiet_log_ratio_sd": 0.15,
    "guard_surplus_mean": 0.08,
    "guard_surplus_sd": 0.25,
    "capture_probability": 0.75,
    "release_probability": 0.35,
    "family_wise_alpha": 0.05,
    "monte_carlo_trials": 200000,
    "seed": 2301,
}


def simulated_joint_power():
    """Deterministic planning power for the exact three-family Holm rule."""
    p = PLANNING
    trials = p["monte_carlo_trials"]
    n_window = SAMPLE_SIZES["minimum_effective_tail_and_guard_windows"]
    n_quench = SAMPLE_SIZES["matched_seeds_per_quench_arm"]
    rng = np.random.default_rng(p["seed"])
    tail_scale = p["quiet_log_ratio_sd"] / math.sqrt(n_window)
    tail_z = -rng.normal(
        p["quiet_log_ratio_mean"], tail_scale, trials
    ) / tail_scale
    guard_scale = p["guard_surplus_sd"] / math.sqrt(n_window)
    guard_z = rng.normal(
        p["guard_surplus_mean"], guard_scale, trials
    ) / guard_scale
    capture = (
        rng.random((trials, n_quench)) < p["capture_probability"]
    ).mean(axis=1)
    release = (
        rng.random((trials, n_quench)) < p["release_probability"]
    ).mean(axis=1)
    pooled = (capture + release) / 2.0
    quench_se = np.sqrt(np.maximum(
        2.0 * pooled * (1.0 - pooled) / n_quench,
        np.finfo(float).tiny,
    ))
    quench_z = (capture - release) / quench_se
    ordered_z = np.sort(
        np.stack((tail_z, guard_z, quench_z), axis=1), axis=1
    )[:, ::-1]
    alpha = p["family_wise_alpha"]
    critical = np.array([
        NormalDist().inv_cdf(1.0 - alpha / 3.0),
        NormalDist().inv_cdf(1.0 - alpha / 2.0),
        NormalDist().inv_cdf(1.0 - alpha),
    ])
    return float(np.mean(np.all(ordered_z >= critical, axis=1)))


PLANNING["joint_power_raw"] = simulated_joint_power()
PLANNING["joint_power"] = (
    math.floor(100.0 * PLANNING["joint_power_raw"]) / 100.0
)


def protocol_text():
    n = SAMPLE_SIZES
    p = PLANNING
    return f"""# E0-prime: direct observable row-map closure

Status: prospective design; no outcome has been observed under this protocol.

## Objective

E0-prime tests the premises of the direct row-map collapse theorem. Its
primary decision does not require a low-rank anchor chart, Gauss-Newton
dominance, positive fiber curvature, or a positive reload coefficient.
Those quantities remain in a separately labelled mechanism-discrimination
battery.

## Separation of stages

- **D, design development.** Disjoint seeds and data blocks locate schedules,
  the physical normalized-stress path, trajectory-local state tubes, event
  density, and eligible quiet-tail windows. D rows cannot enter an E0-prime
  estimator.
- **Q, instrument qualification.** Synthetic or planted cases qualify the
  row-map probe, optimizer-operation ledger, global checkpoint clock,
  preconditioner metadata path, perturbation replay, strict JSON writer, and
  cap-exhaustion route. Q rows cannot enter a model estimate.
- **C, fresh confirmation.** Fresh seeds, data order, checkpoints, optimizer
  states, and bindings are used only after D and Q are frozen.

## Confirmatory cells and estimands

1. **C-tail.** Gate-closed and reload-free windows estimate a simultaneous
   upper bound q on direct all-layer row-map ratios J_i(t+1)/J_i(t). The
   directional criterion is q < 1.
2. **C-tube.** Matched checkpoint replays estimate an upper one-step score
   increment and a lower threshold increment on a trajectory-local tube.
   The directional criterion is nonnegative simultaneous surplus
   Delta c lower minus Delta x upper, including numerical error.
3. **C-quench.** Two capture-side and two release-side schedules use shared
   seed labels. The primary contrast is all-layer finite-horizon entry
   probability, derived only from raw row-map probes.
4. **C-sign.** A simultaneous interval for the signed gate coefficient
   selects one branch before its pilot is read. A nonpositive interval runs
   the collapse-aiding branch; a positive interval runs the rebuild-versus-
   capture branch; an interval containing zero is indeterminate under a
   value-blind extension rule.

## Replication and extension

- eligible effective windows: {n['minimum_effective_tail_and_guard_windows']}
  each for C-tail and C-tube;
- matched seed pairs per quench arm: {n['matched_seeds_per_quench_arm']};
- checkpoint replays per C-tube cell: {n['checkpoint_replays_per_cell']};
- extension blocks: {n['extension_block_steps']} global steps, controlled
  only by eligible-unit counts, up to
  {n['maximum_extension_blocks']} blocks.

No effect estimate, sign, interval endpoint, or p-value can trigger
extension. Maximum-cap exhaustion is INDETERMINATE. A missing or corrupt
artifact is INCOMPLETE.

## Primary families and uncertainty

Holm is applied once at family-wise level 0.05 across exactly:

- direct_tail_contraction;
- guard_tube_closure;
- capture_quench_contrast.

Moving-block bootstrap intervals are simultaneous across layers and declared
steps. Quench cells use shared-seed paired intervals. Planning assumes a
quiet-tail mean log ratio {p['quiet_log_ratio_mean']}, standard deviation
{p['quiet_log_ratio_sd']}, guard surplus mean {p['guard_surplus_mean']},
standard deviation {p['guard_surplus_sd']}, and capture/release probabilities
{p['capture_probability']}/{p['release_probability']}. A deterministic
{p['monte_carlo_trials']}-trial calculation at seed {p['seed']} gives raw
joint power {p['joint_power_raw']:.6f}; the registered conservative value is
{p['joint_power']:.2f}.

## Sign-complete branch rule

For a simultaneous gate interval [L,U]:

- U <= 0 selects NONPOSITIVE_COLLAPSE_RESOLVED;
- L > 0 selects POSITIVE_CAPTURE_RESOLVED;
- L <= 0 < U selects SIGN_UNRESOLVED and yields INDETERMINATE after the
  registered count-only extension is exhausted.

A nonpositive coefficient is not a failure of the collapse theory. It
removes adverse rebuild from the one-sided envelope. The positive branch is
needed only for regenerative maintenance claims.

## Mechanism-discrimination battery

Anchor-chart fidelity, the physical normalized-stress path, a fresh
confirmatory box/band test, Gauss-Newton/full-Hessian agreement,
curvature-factory identity, fiber convexity, and crossing signature are all
reported with PASS, FAIL, UNRESOLVED, or NOT_APPLICABLE. Their outcomes select
or reject particular curvature-sandpile realizations. They do not change a
primary direct-collapse decision.

## Total decision

The executable precedence is INCOMPLETE, MODEL_REJECTED, INDETERMINATE,
NOT_CONFIRMED, then PASS. MODEL_REJECTED is reserved for a failed direct
row-map probe, quiet-tail contraction construct, trajectory-local guard tube,
or operation-ordered optimizer ledger. Inferential misses with complete
construct-valid evidence are NOT_CONFIRMED. Only an exact hash-bound PASS
authorizes downstream confirmation.

## Downstream graph

E7 requires terminal E1, E2, and E3 records because its branch selector
consumes E3. E8 consumes all 9 E5 primary runs and all 40 E6 primary runs.
Triggered E5 and E6 repeat strata remain separate sensitivity analyses and
are never pooled with or substituted for the 49 primary records.
"""


def closed_object(required, properties):
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(required),
        "properties": properties,
    }


def enum(*values):
    return {"type": "string", "enum": list(values)}


def schema_object():
    availability_names = (
        "bindings", "fresh_namespace", "strict_json",
        "global_checkpoint_clock", "averaging_preconditioner_metadata",
        "eligible_unit_ledger",
    )
    core_names = (
        "direct_row_map_probe", "quiet_tail_contraction",
        "trajectory_guard_tube", "ordered_adamw_ledger",
    )
    mechanism_names = (
        "anchor_chart", "normalized_stress_path", "fresh_box_band",
        "gauss_newton_bridge", "curvature_factory", "fiber_convexity",
        "crossing_signature",
    )
    binding_names = (
        "protocol_sha256", "schema_sha256", "source_sha256",
        "calibration_sha256", "data_order_sha256", "tokenizer_sha256",
    )
    family_row = closed_object(
        ("status", "p_value", "direction_pass"),
        {
            "status": enum("CONCLUSIVE", "UNRESOLVED", "MISSING", "ERROR"),
            "p_value": {"type": ["number", "null"], "minimum": 0,
                        "maximum": 1},
            "direction_pass": {"type": ["boolean", "null"]},
        },
    )
    eligibility_rows = {
        name: closed_object(
            ("status", "n", "minimum"),
            {
                "status": enum(
                    "ELIGIBLE", "MAX_CAP_UNRESOLVED", "MISSING", "ERROR"
                ),
                "n": {"type": "integer", "minimum": 0},
                "minimum": {"const": REGISTERED_MINIMUMS[name]},
            },
        )
        for name in FAMILIES
    }
    properties = {
        "schema_version": {"const": "pldr-e0-prime-observable-v2"},
        "campaign_id": {"type": "string", "minLength": 1},
        "bindings": closed_object(
            binding_names,
            {
                name: {"type": "string", "pattern": "^[0-9a-f]{64}$"}
                for name in binding_names
            },
        ),
        "availability": closed_object(
            availability_names,
            {
                name: enum("PASS", "MISSING", "ERROR")
                for name in availability_names
            },
        ),
        "core_checks": closed_object(
            core_names,
            {
                name: enum("PASS", "FAIL", "UNRESOLVED", "MISSING", "ERROR")
                for name in core_names
            },
        ),
        "mechanism_checks": closed_object(
            mechanism_names,
            {
                name: enum("PASS", "FAIL", "UNRESOLVED", "NOT_APPLICABLE")
                for name in mechanism_names
            },
        ),
        "eligibility": closed_object(FAMILIES, eligibility_rows),
        "gate": closed_object(
            ("estimate", "interval"),
            {
                "estimate": {"type": "number"},
                "interval": {
                    "type": "array", "minItems": 2, "maxItems": 2,
                    "items": {"type": "number"},
                },
            },
        ),
        "branch_pilot": closed_object(
            ("status",),
            {
                "status": enum(
                    "NONPOSITIVE_COLLAPSE_RESOLVED",
                    "POSITIVE_CAPTURE_RESOLVED", "SIGN_UNRESOLVED",
                    "UNRESOLVED", "MISSING", "ERROR",
                )
            },
        ),
        "families": closed_object(
            FAMILIES, {name: family_row for name in FAMILIES}
        ),
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "pldr-e0-prime-observable-v2",
        **closed_object(properties, properties),
    }


def design_object():
    return {
        "schema_version": "pldr-e0-prime-design-v1",
        "families": list(FAMILIES),
        "registered_minimums": REGISTERED_MINIMUMS,
        "sample_sizes": SAMPLE_SIZES,
        "planning": PLANNING,
        "gate_branches": {
            "upper_nonpositive": "NONPOSITIVE_COLLAPSE_RESOLVED",
            "lower_positive": "POSITIVE_CAPTURE_RESOLVED",
            "interval_contains_zero": "SIGN_UNRESOLVED",
        },
        "e7_dependencies": ["E0-prime", "E1", "E2", "E3"],
        "e8_primary_inputs": {"E5": 9, "E6": 40},
        "e8_triggered_sensitivity_inputs": {"E5": 18, "E6": 48},
    }


def rendered():
    return {
        PROTOCOL: protocol_text(),
        SCHEMA: json.dumps(schema_object(), indent=2, sort_keys=True) + "\n",
        DESIGN: json.dumps(design_object(), indent=2, sort_keys=True) + "\n",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    failures = []
    for path, expected in rendered().items():
        if args.check:
            actual = path.read_text(encoding="utf-8") if path.exists() else None
            if actual != expected:
                failures.append(str(path.relative_to(ROOT)))
        else:
            path.write_text(expected, encoding="utf-8")
            print(f"wrote {path.relative_to(ROOT)}")
    if failures:
        print("E0-prime generation mismatch: " + ", ".join(failures))
        return 1
    if args.check:
        print("E0-prime generated artifacts: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
