#!/usr/bin/env python3
"""Write calibration/calibration.json: the calibrated constants the
E-series protocol generators consume.

Sources, disclosed per field:
  "stack"      -- parsed MECHANICALLY from the training stack's
                  source (train_run.py / measure_fidelity.py), so the
                  calibration cannot disagree with the code that runs
                  (the generator gate re-checks the equality);
  "reference"  -- the training stack's fixed schedule constants
                  (verbatim from the archived launch scripts);
  "derived"    -- computed here from reference constants by the
                  paper's formulas (theory.py);
  "archived"   -- exploratory numbers read from the archived record
                  (calibration only, no confirmatory weight);
  "E0"         -- measured by the E0 protocol before any dependent
                  dose is evaluated; dependent protocols print the
                  binding formula, and the generator refuses to emit
                  a numeric field that has neither a value nor a
                  binding formula.

Conventions frozen here (and gate-checked by gen_protocols.py):
  * normalized sharpness sigma = (1 - beta1) * eta * lambda; the
    damped flip boundary reads (2 - d)(1 + beta1);
  * the carrier unit identity z = w^2: the derived event factor
    rho_w = z_floor/z_min is a POWER ratio and the amplitude factor
    is sqrt(rho_w);
  * rates are per-step log multipliers -log(1 - eta*theta);
  * the event end is the power-floor convention (first step with
    z < z_floor) everywhere;
  * three distinct collapse endpoints (layer-flat, all-flat,
    operator-invariant); theorem claims bind to all-flat.
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
import theory  # noqa: E402


def _stack_optimizer():
    """Parse the optimizer constants from the training stack source,
    so the calibration file cannot drift from the code."""
    src = open(os.path.join(HERE, "..", "train_run.py")).read()
    m_eps = re.search(r"eps=([0-9.e-]+)", src)
    m_betas = re.search(r"betas=\(([0-9.]+),\s*([0-9.]+)\)", src)
    fid = open(os.path.join(HERE, "..", "measure_fidelity.py")).read()
    m_eps2 = re.search(r"eps=([0-9.e-]+)", fid)
    eps = float(m_eps.group(1))
    if float(m_eps2.group(1)) != eps:
        raise SystemExit("make_calibration: train_run.py and "
                         "measure_fidelity.py disagree on eps")
    return dict(
        source="stack (parsed from train_run.py; equality with "
               "measure_fidelity.py asserted here and re-checked by "
               "the generator gate)",
        beta1=float(m_betas.group(1)) if m_betas else 0.9,
        beta2=float(m_betas.group(2)) if m_betas else 0.95,
        lam_wd=0.1,
        clip=1.0,
        eps=eps,
    )


CAL = {
    "e0_acceptance": {
        "source": "reference (prospectively fixed construct-validity limits)",
        "gn_full_residual_relative_max": 0.10,
        "factory_residual_relative_max": 0.10,
        "ledger_residual_relative_max": 0.10,
        "optimizer_ledger_relative_max": 1.0e-5,
        "fidelity_eps_T_max": 0.15,
        "fidelity_kappa_T_max": 1.0e6,
        "fidelity_eps_off_max": 0.50,
        "regular_window_fraction_min": 0.80,
    },
    "optimizer": _stack_optimizer(),
    "conventions": {
        "source": "derived",
        "sigma_convention": "sigma = (1 - beta1) * eta * lambda "
                            "(normalized); flip at (2 - d)(1 + "
                            "beta1); divide by (1 - beta1) * eta "
                            "for eta*lambda units",
        "carrier_unit_identity": "z = w^2 (power = squared "
                                 "amplitude); event amplitude "
                                 "factor = sqrt(rho_w)",
        "rate_unit": "per-step log multiplier -log(1 - eta*theta)",
        "event_end": "generalized exit (E-end): the event ends at "
                     "the first step with (eps <= 0 OR y <= "
                     "y_floor) AND z below the power gate (z_floor "
                     "at nonpositive margin, eps/b_sat + tau_z at "
                     "positive margin); on the entry-gap domain "
                     "(B-gap) this reduces bit-exactly to the "
                     "two-condition convention (first step with "
                     "z < z_floor AND eps <= 0, exit margin in "
                     "[b - c, 0]); the resolver is fuel-indexed "
                     "and total, its Exhausted branch rejected by "
                     "the map (one convention across map, "
                     "theorems, Lean, generator, and measurement "
                     "code)",
        "event_clock": "one clock: the multiplicative jet event "
                       "of theory.run_event; the stored carrier "
                       "enters through the seeded entry operator "
                       "theory.seeded_entry (kick, truncation, "
                       "seed floor, sign law; z_0 = w_entry^2 "
                       "exactly); the drain law is exact in "
                       "logarithmic form, log(y_N/y_0) = 2*sum "
                       "log(1 - eta*(th_J + lam_wd + kappa*z_n)); "
                       "no linear margin recurrence and no drain "
                       "coefficient a_b exist",
        "activity_unit": "the activity counter is the RELOAD "
                         "DURATION ceil((J_pre - J_post)/"
                         "(eta*delta_J0)) in GLOBAL steps; the "
                         "fast-clock iteration count is a "
                         "diagnostic only",
    },
    "endpoints": {
        "source": "derived",
        "layer_flat": "min-layer median row-map Jacobian singular "
                      "value, last-quarter mean < 0.1 (existential: "
                      "at least one layer flat; the archived phase "
                      "criterion)",
        "all_flat": "max-layer median row-map Jacobian singular "
                    "value, last-quarter mean < 0.1 (universal: "
                    "every criterion layer flat; the closure "
                    "theorem's collapsed set binds here)",
        "operator_invariant": "m_gen(G) < 0.01 (generation "
                              "criterion; model-level claims are "
                              "scoped to this endpoint only through "
                              "its own registered link)",
    },
    "decay_ledger": {
        "source": "reference",
        "p_i": [2, 3],
        "note": "each decayed path factor carries one decoupled "
                "weight-decay share inside its own update; the "
                "manufactured curvature kappa*u^2 drains at "
                "2*p_i*lam_wd and the base at mu_b, each counted "
                "once; there is no independent reversion parameter",
    },
    "cells": {
        "source": "reference",
        "boundary_rate_cell": {"lr": 7.5e-4, "schedule": "const",
                               "warmup": 250, "steps": 24000},
        "reference_swept_cell": {"lr": 1e-3, "schedule": "cosine",
                                 "warmup": 1000, "steps": 24000},
        "subcritical_control": {"lr": 3e-4, "schedule": "const",
                                "warmup": 250, "steps": 24000},
        "batch_grid": [8, 16, 32, 64],
    },
    "boundaries": {
        "source": "derived",
        "frozen_flip_sigma": theory.flip_boundary(0.9, 0.0),
        "frozen_flip_eta_lambda":
            theory.flip_boundary(0.9, 0.0) / (1 - 0.9),
        "note": "the certificate boundary is evaluated per logged "
                "step by theory.c_grnt_box from the directional "
                "drifts theory.drift_dir and the CERTIFIED "
                "constants theory.cp_dir_bounds on the declared "
                "box; the search never leaves the box and reports "
                "box-edge outcomes",
    },
    "mean_field_null": {
        "source": "reference",
        "tau": 1.5, "gamma": 2.0, "alpha_D": 2.0,
        "peak_power_map_b": 2.0,
        "integrated_power_map_b": 1.0,
        "derived_sets": {
            "integrated": {"tau": 1.5, "gamma": 2.0},
            "peak": {"tau": 1.25, "gamma": 4.0},
        },
    },
    "event_conventions": {
        "source": "archived",
        "note": "detection windows and exclusion lag carried "
                "unchanged from the archived record's declared "
                "constants (run-record appendix) so future "
                "construct-valid series remain comparable; the "
                "event END is the power-floor convention "
                "(conventions block)",
    },
    "gated_model_planning": {
        "source": "derived",
        "note": "the planning instance of the regime clauses: one "
                "update law, one event clock, three schedule arms "
                "(capture; two-phase separated maintenance at "
                "sep_period with the high threshold c_sep_hi; "
                "constant gate-held maintenance at c_recurrent); "
                "E2's planning event library and E5's operating "
                "characteristic run through theory.run_event / "
                "theory.gated_model_step with the decision taken "
                "by theory.decide_e5 under its generated "
                "configuration, and the generator gate "
                "verifies the exact log-drain identity, the "
                "invariant power cap with (S+), COMPLETED reloads "
                "on the two-phase arm (sep-window, zero "
                "unseparated pairs, certified by the classifier), "
                "overhanging reloads on the constant gate-held "
                "arm (unsep-window), the capture envelope "
                "balance, and entry within the printed bound",
        "eta": 0.05,
        "d0": 0.01, "mu_b": 0.1,
        "delta_a": 0.2, "delta_J0": 0.2,
        "th_a_loss": 0.2, "th_g_loss": 0.0, "th_J_loss": 0.01,
        "kappa": 1.0, "lam_wd_model": 0.0, "s_shed": 0.5,
        "burst": {"h": 0.1, "b_sat": 2.0, "z_seed": 0.01,
                  "z_floor": 0.002, "w_kick_max": 0.1,
                  "y_floor": 0.01, "tau_z": 0.02},
        "J_crit": 0.1, "J0": 0.6,
        "c_recurrent": 0.4, "c_captured": 1.0,
        "sep_period": 12, "c_sep_hi": 1.0,
        "T_maint": 2400,
    },
    "measured_in_E0": {
        "source": "E0",
        "fields": [
            "kappa_i", "h_max", "f_max", "C_phi", "kappa_phi",
            "z_min_seed", "sigma_g2", "SC_constants",
            "loading_slope_se", "decay_shares_per_factor",
            "defect_window_stats", "gate_regression_deltaJ0",
            "path_box_containment", "K_s_K_q_on_path_box",
            "threshold_quotient_edge_state",
            "edge_state_increment_simultaneous_interval",
            "complete_source_increment_simultaneous_interval",
            "stationary_threshold_regression",
            "e6_pilot_match_mean", "e6_pilot_cell_effect",
            "e6_pilot_seed_weight"
        ],
        "binding": "every E1-E7 dose that consumes one of these "
                   "fields prints its formula in the protocol and "
                   "is evaluated mechanically when E0 lands",
    },
    "power_sim": {
        "source": "derived",
        "e1_placements": 10,
        "e1_effect_sigma": 3.0,
        "e5_reps": 200,
        "e6_verdict_reps": 120,
        "e5_primary_censor_rate": 0.10,
        "e6_primary_censor_rate": 0.10,
        "repeat_censor_rate": 0.02,
        "rng_seed": 20260817,
    },
    "planning": {
        "source": "archived-record planning values; every design "
                  "is powered at these effect sizes and "
                  "re-evaluated at E0 before launch; all event "
                  "quantities come from the ONE event clock "
                  "(theory.run_event at the gated planning "
                  "instance), gate-checked at generation",
        "per_step_loading": 1e-4,
        "pulse_dose_stress": 0.02,
        "paired_effect_over_sd": 1.2,
        "event_induction_prob": 0.9,
        "e1_face_seeds": 6,
        "e1_eta_ratios": [1.0, 0.5, 0.25, 0.125],
        "e1_reactivation_log_sd": 0.15,
        "e1_reactivation_slope_tolerance": 0.5,
        "loading_slope_snr_per_window": 0.8,
        "n_probe_windows": 60,
        "n_events_e0": 40,
        "signature_frac_model": 0.6,
        "signature_frac_null": 0.15,
        "n_events_e2": 60,
        "e2_power_rel_sd": 0.05,
        "e2_reset_meas_sd": 0.05,
        "exit_frac_model": 0.8,
        "exit_frac_null": 0.2,
        "n_arrivals_e3": 200,
        "rc_planning": 0.7,
        "n_receivers": 6,
        "n_events_e4": 40,
        "flux_ratio_model": 2.5,
        "flux_lognorm_sd": 0.8,
        "e5_d0_jitter_rel": 0.1,
        "e5_obs_noise_J": 0.01,
        "n_events_e7": 500,
        "e7_s_min": 5,
        "e7_s_cutoff": 200,
        "e7_fit_hi": 100,
        "e7_tau_tol": 0.15,
        "e7_gamma_tol": 0.5,
        "e7_gamma_se": 0.15,
        "route_gap_over_se": 3.0,
        "block_bootstrap_len": 50,
        "cp_cert_box": {
            "sigma": [0.05, 3.4],
            "beta1": [0.85, 0.95],
            "d": [0.0, 2e-4],
            "grid_check_n": 21,
        },
        "eps_max_event": 0.05,
        "w_max2_event": 0.01,
        "e2_entry_margin_lo": 0.002,
        "e2_entry_margin_hi": 0.05,
        "e2_library_size": 16,
        "e2_logdrain_meas_sd": 0.05,
        "tsep_min_gap": 2,
        "tsep_max_topple_frac": 0.5,
    },
}


def main():
    out = os.path.join(HERE, "calibration.json")
    with open(out, "w") as fh:
        json.dump(CAL, fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
