#!/usr/bin/env python3
"""Joint E6 pilot inference and design mapping.

The pilot uses two cells on each predicted side with shared seed
labels.  It evaluates the exact latent-sign/shared-seed likelihood on
the declared finite grid and inverts a pointwise Monte Carlo test at
every grid member.  Binomial order statistics and a union bound control
calibration uncertainty simultaneously over that grid.  The resulting
joint region, not fitted-member marginal intervals, is mapped into both
complete E6 attempt configurations.  If the desired precision
exceeds the declared cap, or seed sharing is not identified well
enough, the output switches to the executable independent-block
randomization fallback and labels the pilot exploratory.
"""

import copy
import itertools
import json
import math
import os
from enum import Enum
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EXP, "analysis"))
sys.path.insert(0, HERE)

import bridge  # noqa: E402
import decision_constants as dc  # noqa: E402
import theory  # noqa: E402

J_CRIT = 0.1
BURN_FRAC = 0.1


class PilotRegionStatus(str, Enum):
    NONEMPTY = "nonempty"
    UNRESOLVED = "unresolved"
    EMPTY = "empty"


def score_raw_record(rec, predicted, where="pilot record"):
    """Apply E6 raw endpoint, ordering, routing, and gate semantics."""
    endpoint = theory.derive_capture_endpoint(
        rec, J_CRIT, dc.E6_SPEC["min_persistence_probes"], where=where,
        require_step_one=True)
    status = endpoint["status"]
    st = theory.classify_event_word(rec["word"], rec["act"])
    if status == "censored" or st["phen"] == "censored":
        return None, False, endpoint, st
    captured = status == "finite_horizon_entry"
    if predicted == "capture":
        if not captured:
            return False, False, endpoint, st
        entry = int(endpoint["entry"])
        last_event_step = st["last_event"] + 1 if st["last_event"] >= 0 else 0
        kill = bool(last_event_step >= entry)
        gate_closed = not np.any(np.asarray(rec["act"])[entry - 1:] > 0)
        return bool(gate_closed), kill, endpoint, st
    jet = endpoint["jet_step"]
    if st["phen"] == "unsep-window":
        burn = int(BURN_FRAC * len(jet))
        match = bool((not captured) and np.min(jet[burn:]) > J_CRIT)
    else:
        match = bool(not captured)
    return match, False, endpoint, st


def run_match(name, predicted, delta_J0, runs=None):
    """Generate one raw record and score it by the E6 semantics."""
    kwargs = {} if runs is None else {"runs": runs}
    rec, aux = bridge.run_record(name, delta_J0=delta_J0, **kwargs)
    match, kill, endpoint, st = score_raw_record(
        rec, predicted, where=f"pilot run {name}")
    return match, kill, rec, endpoint, st


def loglik(matrix, p, delta, w, p_seed=None):
    """Exact likelihood after summing latent seed coins and signs."""
    matrix = np.asarray(matrix, dtype=int)
    n_cells, n_seeds = matrix.shape
    p_seed = p if p_seed is None else p_seed
    total = 0.0
    for latent in itertools.product((0, 1), repeat=n_seeds):
        prob_latent = np.prod([
            p_seed if bit else 1.0 - p_seed for bit in latent])
        if prob_latent == 0:
            continue
        cell_product = 1.0
        for c in range(n_cells):
            sign_mix = 0.0
            for sign in (1.0, -1.0):
                pc = np.clip(p + sign * delta, 0.0, 1.0)
                prob = 1.0
                for j, bit in enumerate(latent):
                    pi = np.clip((1.0 - w) * pc + w * bit,
                                 1e-12, 1.0 - 1e-12)
                    prob *= pi if matrix[c, j] else 1.0 - pi
                sign_mix += 0.5 * prob
            cell_product *= sign_mix
        total += prob_latent * cell_product
    return float(math.log(max(total, 1e-300)))


def declared_grid(spec=None):
    spec = dc.E6_SPEC if spec is None else spec
    pilot = spec["pilot"]
    return list(itertools.product(
        pilot["p_grid"], pilot["delta_grid"], pilot["w_grid"]))


def likelihood_surface(matrix, spec=None):
    """Exact log likelihood at every member of the declared grid."""
    points = declared_grid(spec)
    values = np.array([loglik(matrix, *point) for point in points])
    return points, values


def fit(matrix, spec=None):
    points, values = likelihood_surface(matrix, spec)
    index = int(np.argmax(values))
    return float(values[index]), tuple(float(x) for x in points[index])


def sample_member(rng, p, delta, w, n_cells, n_seeds):
    """Sample one match matrix from a declared family member."""
    latent = rng.random(n_seeds) < p
    out = np.zeros((n_cells, n_seeds), dtype=int)
    for c in range(n_cells):
        sign = 1.0 if rng.random() < 0.5 else -1.0
        pc = np.clip(p + sign * delta, 0.0, 1.0)
        pi = np.clip((1.0 - w) * pc + w * latent, 0.0, 1.0)
        out[c] = rng.random(n_seeds) < pi
    return out


def _binomial_cdf(k, n, p):
    """Stable Binomial(n,p) CDF for integer ``k``."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0:
        return 1.0
    if p >= 1:
        return 0.0
    logs = [
        math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
        + j * math.log(p) + (n - j) * math.log1p(-p)
        for j in range(k + 1)
    ]
    peak = max(logs)
    return float(math.exp(peak) * math.fsum(math.exp(x - peak) for x in logs))


def _simultaneous_upper_order(n_cal, probability, family_size, mc_alpha):
    """Order statistic whose coverage holds over the entire finite grid."""
    if not (0 < probability < 1 and 0 < mc_alpha < 1):
        raise ValueError("probability and mc_alpha must lie in (0,1)")
    per_member = mc_alpha / float(family_size)
    target = 1.0 - per_member
    for order in range(1, n_cal + 1):
        if _binomial_cdf(order - 1, n_cal, probability) >= target:
            return order
    return None


def finite_grid_neyman_region(matrix, n_cal=None, seed=0, spec=None):
    """Invert pointwise tests at every member of the declared grid.

    Each candidate has its own sampling distribution and upper
    acceptance threshold.  A binomial order statistic plus a union bound
    makes the Monte Carlo quantile guarantee simultaneous across the
    entire finite grid.  If the requested simultaneous guarantee cannot
    be resolved with ``n_cal`` simulations, the affected thresholds are
    infinite, which is valid and transparently uninformative.
    """
    spec = dc.E6_SPEC if spec is None else spec
    pilot = spec["pilot"]
    n_cal = int(pilot["mc_calibration_reps"] if n_cal is None else n_cal)
    if n_cal < 20:
        raise ValueError("at least 20 calibration simulations are required")
    matrix = np.asarray(matrix, dtype=int)
    if matrix.ndim != 2 or not np.all(np.isin(matrix, (0, 1))):
        raise ValueError("matrix must be a two-dimensional 0/1 array")
    confidence = float(pilot["lr_confidence"])
    mc_alpha = float(pilot["mc_error_alpha"])
    points, values = likelihood_surface(matrix, spec)
    family_size = len(points)
    order = _simultaneous_upper_order(
        n_cal, confidence, family_size, mc_alpha)
    rng = np.random.default_rng(seed)
    surface = []
    empirical_acceptance = []
    for point, observed_ll in zip(points, values):
        simulated_stats = np.empty(n_cal)
        for draw in range(n_cal):
            simulated = sample_member(rng, *point, *matrix.shape)
            simulated_stats[draw] = -2.0 * loglik(simulated, *point)
        threshold = (math.inf if order is None else
                     float(np.partition(simulated_stats, order - 1)[order - 1]))
        observed_stat = -2.0 * float(observed_ll)
        accepted = bool(observed_stat <= threshold + 1e-12)
        empirical_acceptance.append(float(np.mean(simulated_stats <= threshold)))
        p, delta, w = point
        surface.append(dict(
            p=float(p), delta=float(delta), w=float(w),
            loglik=float(observed_ll), member_statistic=observed_stat,
            member_threshold=threshold, in_region=accepted,
        ))
    region = [
        {key: row[key] for key in
         ("p", "delta", "w", "loglik", "member_statistic",
          "member_threshold")}
        for row in surface if row["in_region"]
    ]
    best_index = int(np.argmax(values))
    best = tuple(float(x) for x in points[best_index])
    region_status = (PilotRegionStatus.EMPTY if not region else
                     PilotRegionStatus.UNRESOLVED if order is None else
                     PilotRegionStatus.NONEMPTY)
    return dict(
        descriptive_likelihood_maximizer=dict(
            p=best[0], delta=best[1], w=best[2],
            loglik=float(values[best_index])),
        confidence=confidence,
        calibration_reps=n_cal,
        calibration_reps_per_member=n_cal,
        mc_error_alpha=mc_alpha,
        per_member_mc_alpha=mc_alpha / family_size,
        simultaneous_calibration_confidence=1.0 - mc_alpha,
        order_statistic=order,
        unresolved_quantile=bool(order is None),
        region_status=region_status.value,
        model_consistent=bool(region),
        uncertainty_resolved=bool(region and order is not None),
        coverage_hat=float(min(empirical_acceptance)),
        coverage_lower=confidence,
        calibration_scope="finite_grid_neyman_inversion",
        likelihood_surface=surface,
        region=region,
    )


def joint_lr_region(matrix, n_cal=None, seed=0, spec=None):
    """Compatibility entry point for finite-grid Neyman inversion."""
    return finite_grid_neyman_region(
        matrix, n_cal=n_cal, seed=seed, spec=spec)


def precision_and_fallback(region_result, matrix, spec=None):
    """Evaluate width, coverage, expansion, cap, and fallback routing."""
    spec = dc.E6_SPEC if spec is None else spec
    pilot = spec["pilot"]
    region = region_result["region"]
    if not region:
        return dict(
            region_status=PilotRegionStatus.EMPTY.value,
            widths=None, max_width=None,
            target_width=float(pilot["precision_target_width"]),
            width_met=False, cap_hit=False,
            coverage_ok=True, sharing_unresolved=False,
            status="MODEL_REJECTED",
            fallback={"active": False,
                      "allocation": None,
                      "procedure": "no design is defined for an empty region"},
        )
    widths = {
        key: max(row[key] for row in region) - min(row[key] for row in region)
        for key in ("p", "delta", "w")
    }
    target = float(pilot["precision_target_width"])
    max_width = max(widths.values())
    width_met = max_width <= target + 1e-12
    ratio = max_width / target if target > 0 else math.inf
    n_cells, n_seeds = np.asarray(matrix).shape
    multiplier = max(1.0, ratio ** 2)
    proposed_cells_side_raw = int(math.ceil((n_cells / 2) * multiplier))
    proposed_seeds_raw = int(math.ceil(n_seeds * multiplier))
    max_cells_side = int(pilot["max_cells_per_side"])
    max_seeds = int(pilot["max_seeds_per_cell"])
    cap_hit = (proposed_cells_side_raw > max_cells_side
               or proposed_seeds_raw > max_seeds)
    sharing_unresolved = max(row["w"] for row in region) > 0.5
    coverage_ok = (
        region_result["coverage_lower"] >= pilot["lr_confidence"]
        and not region_result.get("unresolved_quantile", False))
    if cap_hit or not coverage_ok:
        status = "exploratory"
    elif not width_met:
        status = "expand_pilot"
    elif sharing_unresolved:
        status = "independent_block_fallback"
    else:
        status = "confirmatory"
    fallback = dict(
        active=bool(status == "independent_block_fallback"),
        allocation="independent_seed_blocks",
        procedure=("assign disjoint fresh seed blocks to cells, randomize "
                   "registered block labels, set w=0, and exactly "
                   "recalibrate both attempts"),
    )
    return dict(
        widths=widths,
        max_width=float(max_width),
        target_width=target,
        width_met=bool(width_met),
        proposed_cells_per_side_raw=proposed_cells_side_raw,
        proposed_seeds_per_cell_raw=proposed_seeds_raw,
        proposed_cells_per_side=min(proposed_cells_side_raw,
                                    max_cells_side),
        proposed_seeds_per_cell=min(proposed_seeds_raw, max_seeds),
        cap_hit=bool(cap_hit),
        coverage_ok=bool(coverage_ok),
        sharing_unresolved=bool(sharing_unresolved),
        status=status,
        fallback=fallback,
    )


def independent_block_fallback(region_result, J_crit=J_CRIT, spec=None,
                               power_floor=0.80):
    """Execute the independent-block reallocation and exact recalibration.

    The fresh allocation removes the shared-seed component by design.
    The null family retains its full cell-effect range but fixes w=0;
    the fully shared corner is therefore reported for comparison and
    excluded from the fallback calibration.  Power is checked over the
    entire pilot region after projection to the registered w=0 design.
    """
    spec = dc.E6_SPEC if spec is None else spec
    if not region_result.get("region"):
        return dict(active=False, status="MODEL_REJECTED",
                    reason="empty finite-grid inversion")
    fallback_spec = copy.deepcopy(spec)
    fallback_spec["null_family"]["w_max"] = 0.0
    fallback_spec["null_family"]["w_step"] = 1.0
    fallback_spec["null_family"]["include_shared_corner"] = False
    calibrations = []
    powers = []
    projected = sorted({
        (float(point["p"]), float(point["delta"]), 0.0)
        for point in region_result["region"]
    })
    for template in fallback_spec["attempts"]:
        shape = {key: template[key] for key in
                 ("seeds_per_cell", "n_cells", "cell_majority",
                  "cells_required")}
        calibration = theory.e6_calibrate_threshold(
            shape, template["alpha"], fallback_spec)
        calibrations.append(calibration)
        region_power = [
            theory.e6_pass_prob_exact(
                calibration["threshold"], shape, p, delta, w)
            for p, delta, w in projected
        ]
        powers.append(dict(
            attempt_id=template["attempt_id"],
            threshold=calibration["threshold"],
            min_region_power=float(min(region_power)),
            max_region_power=float(max(region_power)),
        ))
    study = dc.e6_config(calibrations, J_crit)
    floor_met = all(
        row["min_region_power"] >= float(power_floor) for row in powers)
    return dict(
        active=True,
        allocation="independent_seed_blocks",
        randomized_labels_within_registered_blocks=True,
        null_family=fallback_spec["null_family"],
        study=study,
        region_power=powers,
        power_floor=float(power_floor),
        power_floor_met=bool(floor_met),
        status="confirmatory" if floor_met else "exploratory",
    )


def matched_block_randomization(match_if_capture, match_if_release,
                                attempt_config):
    """Exact conditional label-randomization distribution.

    Rows are prospective schedule cells and columns are matched seed
    blocks.  The two Boolean matrices give each raw record's potential
    match under capture-side and release-side labels.  Every assignment
    with the registered number of capture labels is enumerated, and the
    executable count-plus-coverage statistic is rescored.
    """
    cap = np.asarray(match_if_capture, dtype=bool)
    rel = np.asarray(match_if_release, dtype=bool)
    if cap.shape != rel.shape or cap.ndim != 2:
        raise ValueError("potential-match matrices need one common 2-D shape")
    n_cells, n_seeds = cap.shape
    if n_cells != int(attempt_config["n_cells"]) \
            or n_seeds != int(attempt_config["seeds_per_cell"]):
        raise ValueError("potential-match matrix shape differs from attempt")
    n_capture = n_cells // 2
    majority = int(attempt_config["cell_majority"])
    threshold = int(attempt_config["total_threshold"])
    required = int(attempt_config["cells_required"])
    assignments = []
    passes = 0
    for capture_rows in itertools.combinations(range(n_cells), n_capture):
        capture_rows = set(capture_rows)
        matrix = np.stack([
            cap[c] if c in capture_rows else rel[c]
            for c in range(n_cells)
        ])
        cell_counts = matrix.sum(axis=1).astype(int)
        total = int(cell_counts.sum())
        coverage = int(np.sum(cell_counts >= majority))
        passed = total >= threshold and coverage >= required
        passes += int(passed)
        assignments.append(dict(
            capture_rows=sorted(capture_rows),
            cell_counts=cell_counts.tolist(),
            total=total,
            coverage=coverage,
            passed=bool(passed),
        ))
    return dict(
        n_assignments=len(assignments),
        pass_count=passes,
        conditional_pass_probability=passes / float(len(assignments)),
        assignments=assignments,
    )


def map_region_to_study(region_result, J_crit=J_CRIT, spec=None,
                        power_floor=0.80):
    """Calibrate both attempts and report worst power over the region."""
    spec = dc.E6_SPEC if spec is None else spec
    if not region_result.get("region"):
        return dict(status="MODEL_REJECTED", study=None, region_power=[],
                    power_floor=float(power_floor),
                    power_floor_met=False)
    calibrations = []
    powers = []
    for template in spec["attempts"]:
        shape = {key: template[key] for key in
                 ("seeds_per_cell", "n_cells", "cell_majority",
                  "cells_required")}
        calibration = theory.e6_calibrate_threshold(
            shape, template["alpha"], spec)
        calibrations.append(calibration)
        region_power = [theory.e6_pass_prob_exact(
            calibration["threshold"], shape,
            point["p"], point["delta"], point["w"])
            for point in region_result["region"]]
        powers.append(dict(
            attempt_id=template["attempt_id"],
            threshold=calibration["threshold"],
            min_region_power=float(min(region_power)),
            max_region_power=float(max(region_power)),
        ))
    study = dc.e6_config(calibrations, J_crit)
    floor_met = all(
        row["min_region_power"] >= float(power_floor) for row in powers)
    return dict(study=study, region_power=powers,
                power_floor=float(power_floor),
                power_floor_met=bool(floor_met))


def select_pilot_action(region_result, matrix, J_crit=J_CRIT, spec=None,
                        power_floor=0.80):
    """Map the joint pilot result to a frozen confirmatory action."""
    spec = dc.E6_SPEC if spec is None else spec
    if not region_result.get("region"):
        precision = precision_and_fallback(region_result, matrix, spec)
        return dict(
            status="MODEL_REJECTED",
            reasons=["finite-grid Neyman inversion is empty"],
            precision=precision, shared_design=None, fallback_design=None)
    precision = precision_and_fallback(region_result, matrix, spec)
    shared = map_region_to_study(
        region_result, J_crit=J_crit, spec=spec,
        power_floor=power_floor)
    fallback = None
    reasons = []
    if precision["status"] == "exploratory":
        status = "exploratory"
        if precision["cap_hit"]:
            reasons.append("pilot precision requirement exceeds the cap")
        if not precision["coverage_ok"]:
            reasons.append("simultaneous per-member quantile guarantee is unresolved")
    elif precision["status"] == "expand_pilot":
        status = "expand_pilot"
        reasons.append("collect the registered expanded pilot before E6")
    elif precision["status"] == "independent_block_fallback":
        fallback = independent_block_fallback(
            region_result, J_crit=J_crit, spec=spec,
            power_floor=power_floor)
        status = ("confirmatory_independent_blocks"
                  if fallback["power_floor_met"] else "exploratory")
        if not fallback["power_floor_met"]:
            reasons.append("independent-block worst-region power misses floor")
    elif shared["power_floor_met"]:
        status = "confirmatory_shared_blocks"
    else:
        status = "exploratory"
        reasons.append("shared-block worst-region power misses floor")
    return dict(status=status, reasons=reasons, precision=precision,
                shared_design=shared, fallback_design=fallback)


def analyze(matrix, n_cal=None, seed=0, potential_matches=None):
    matrix = np.asarray(matrix, dtype=int)
    if matrix.ndim != 2 or matrix.shape[0] < 4:
        raise ValueError("pilot requires at least four cells")
    result = finite_grid_neyman_region(matrix, n_cal=n_cal, seed=seed)
    result["match_matrix"] = matrix.tolist()
    selection = select_pilot_action(result, matrix)
    result["precision"] = selection["precision"]
    result["design_map"] = selection["shared_design"]
    result["selection"] = selection
    if (potential_matches is not None
            and selection["shared_design"] is not None):
        cap, rel = potential_matches
        result["block_randomization"] = matched_block_randomization(
            cap, rel, selection["shared_design"]["study"]["attempts"][0])
    return result


def main():
    delta_J0 = float(sys.argv[1])
    pilot = dc.E6_SPEC["pilot"]
    seeds = tuple(range(1001, 1001 + int(pilot["seeds_per_cell"])))
    cells = (("cap1", "capture"), ("cap2", "capture"),
             ("rel1", "release"), ("rel2", "release"))
    matrix = np.zeros((len(cells), len(seeds)), dtype=int)
    detail = {}
    for c, (cell_name, predicted) in enumerate(cells):
        detail[cell_name] = {}
        for j, seed_value in enumerate(seeds):
            run_name = f"c-e0p-{cell_name}-s{seed_value}"
            match, kill, rec, endpoint, st = run_match(
                run_name, predicted, delta_J0)
            if match is None or kill:
                raise RuntimeError(
                    f"pilot run {run_name} is censored or violates causal order")
            matrix[c, j] = int(match)
            detail[cell_name][str(seed_value)] = dict(
                match=bool(match), endpoint=endpoint["status"],
                entry=endpoint["entry"], phenotype=st["phen"],
                n_events=rec["n_events"])
    output = analyze(matrix)
    output["per_run"] = detail
    outdir = os.path.join(HERE, "out", "e0")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "pilot_fit.json"), "w") as handle:
        json.dump(output, handle, indent=1)
    print(json.dumps({
        "descriptive_likelihood_maximizer": output[
            "descriptive_likelihood_maximizer"],
        "coverage_lower": output["coverage_lower"],
        "region_status": output["region_status"],
        "region_size": len(output["region"]),
        "precision": output["precision"],
        "region_power": (output["design_map"]["region_power"]
                         if output["design_map"] is not None else []),
    }, indent=1))


if __name__ == "__main__":
    main()
