"""Permanent witnesses for the corrected composition interfaces
(the repaired claims each carry the witness that made the repair
necessary):

  1. ZERO-RELAXATION NONTERMINATION: a legal gap-domain event with
     th_J = lam_wd = 0 reaches a fixed state with positive margin
     and exhausts EVERY finite fuel; the corrected well-posedness
     package carries the strict relaxation floor theta_- > 0, and
     the region bounds reject theta' = 0 by hypothesis.
  2. ENDPOINT/AVERAGE: a legal cycle whose endpoint floor exceeds
     the criterion while its cycle average falls below it; the
     endpoint floor (M-end) licenses stroboscopic maintenance
     only, and the average clause requires the independent
     computable certificate (M-avg-c), which refuses here.
  3. CARRIER SIGN: the smooth carrier multiplier reads the
     post-arrival stress x' = b' + kappa*u^2; the former condition
     eta*(B_max + kappa*u_max^2) <= 1 admits a negative
     multiplier, the strengthened eta*(B_max + A_max +
     kappa*u_max^2) <= 1 excludes it.
  4. TWO-PHASE RETENTION FLOOR: the theorem-tier per-event jet
     retention bound is uniform over gap-domain entries and is
     verified against run_event on an entry grid.
  5. UNIFORM FUEL: the whole-box termination bound covers the gap
     domain, the middle region including b = c, and the
     overloaded region, verified against run_event.
  6. SATURATED REBUILD: the jet rebuild clips at the normalized
     ceiling J <= 1, so jet box invariance needs no rate condition
     on delta_J0.
"""

import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "analysis"))
import theory  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)


def _planning():
    cal = json.load(open(os.path.join(
        EXP, "calibration", "calibration.json")))
    return cal["gated_model_planning"], cal["planning"]


# 1 ------------------------------------------------------ relaxation


def test_zero_relaxation_event_exhausts_every_fuel():
    burst = dict(h=1.0, b_sat=1.0, z_seed=1.0, z_floor=0.001,
                 y_floor=0.01, tau_z=0.02, w_kick_max=1.0)
    for fuel in (100, 1000, 5000):
        ev = theory.run_event(
            b=0.0, a=1.0, g=1.0, J=1.0, w=0.0, c=0.5, eta=0.1,
            kappa=1.0, th_J_loss=0.0, lam_wd=0.0, burst=burst,
            fuel=fuel)
        assert not ev["ended"]
        assert ev["status"] == "exhausted"
        assert ev["J"] == pytest.approx(0.9)
        assert ev["eps_end"] == pytest.approx(0.31)
    # the corrected hypotheses exclude the witness: every region
    # bound REQUIRES strict relaxation
    with pytest.raises(ValueError):
        theory.event_time_bound_middle(1.0, 0.1, 0.0, 0.0, 1.0,
                                       1.0, 1.0, 0.001, 0.01)
    with pytest.raises(ValueError):
        theory.event_fuel_required_uniform(
            1.0, 0.0, 0.5, 0.1, 0.0, 0.0, 1.0, 1.0, 1.0, 0.001,
            0.01, 0.02)


# 2 ------------------------------------------------ endpoint/average


def test_endpoint_floor_does_not_license_cycle_average():
    q, r, t_cyc, m, j_crit = 0.99, 0.1, 20, 0.01, 0.15
    q_cyc = r * q ** t_cyc
    m_cyc = t_cyc * m * q ** t_cyc
    floor = theory.maintenance_floor(q_cyc, m_cyc)
    assert floor == pytest.approx(0.178152613361, abs=1e-12)
    assert floor > j_crit            # (M-end) holds...
    j = [r * floor]
    for _ in range(t_cyc):
        j.append(q * j[-1] + m)
    assert j[-1] >= floor            # ...and propagates...
    assert sum(j) / len(j) == pytest.approx(0.110083885590,
                                            abs=1e-12)
    assert sum(j) / len(j) < j_crit  # ...but the average FAILS
    # and the computable certificate (M-avg-c) correctly refuses:
    # the envelope average is below the criterion
    avg_cert = theory.maintenance_cycle_average(
        floor, r_min=r, q_min=q, m_seq=[m] * t_cyc)
    assert avg_cert == pytest.approx(0.103410980415, abs=1e-12)
    assert avg_cert < j_crit


def test_m_avg_c_passes_at_the_planning_theorem_tier():
    # the theorem-tier instance in the generated gate report must
    # pass both (M-end) and (M-avg-c): recurrent non-vacuity at
    # theorem constants, in the average form the measured phase
    # criterion reads
    rep = json.load(open(os.path.join(
        EXP, "protocols", "gate_report.json")))
    m_checks = [c for c in rep["checks"]
                if c["name"].startswith("[thm] maintenance")]
    assert len(m_checks) == 1 and m_checks[0]["ok"]


# 3 --------------------------------------------------- carrier sign


def test_carrier_multiplier_needs_post_arrival_ceiling():
    eta, b_max, arr, ku2 = 1.0, 0.5, 0.5, 0.5
    a_max = arr
    x_max_old = b_max + ku2            # post-reversion ceiling
    x_max_new = b_max + a_max + ku2    # post-arrival ceiling
    assert eta * x_max_old <= 1.0      # the OLD condition passes
    x_read = (b_max + arr) + ku2       # what the update reads
    assert 1.0 - eta * x_read < 0.0    # ...multiplier NEGATIVE
    assert eta * x_max_new > 1.0       # the NEW condition excludes
    # and whenever the new condition holds the multiplier is
    # nonnegative for every legal b' = b + arr
    eta2 = 1.0 / x_max_new
    for b in np.linspace(0.0, b_max, 5):
        for a in np.linspace(0.0, a_max, 5):
            assert 1.0 - eta2 * ((b + a) + ku2) >= -1e-12


# 4 ---------------------------------------------- retention floor


def test_retention_floor_two_phase_bounds_run_event():
    gp, pl = _planning()
    bu = gp["burst"]
    b_star = gp["d0"] / gp["mu_b"]
    zcap = theory.z_cap_invariant(
        bu["h"], pl["eps_max_event"], bu["b_sat"],
        w_max2=pl["w_max2_event"], z_seed=bu["z_seed"])
    delta_b = 0.5 * (gp["c_recurrent"] - b_star)
    r_floor = theory.retention_floor_two_phase(
        pl["eps_max_event"], delta_b, gp["eta"], gp["th_J_loss"],
        gp["lam_wd_model"], gp["kappa"], bu["h"], zcap,
        bu["z_floor"])
    assert 0.0 < r_floor < 1.0
    worst = 1.0
    for b in np.linspace(0.0, gp["c_recurrent"] - delta_b, 5):
        for eps0 in np.linspace(1e-4, pl["eps_max_event"], 5):
            y0 = eps0 + (gp["c_recurrent"] - b)
            j0 = math.sqrt(y0 / gp["kappa"])
            if j0 > 1.0:
                continue
            for w in (0.0, 0.1):
                ev = theory.run_event(
                    b, 1.0, 1.0, j0, w, gp["c_recurrent"],
                    gp["eta"], gp["kappa"], gp["th_J_loss"],
                    gp["lam_wd_model"], bu, fuel=200000)
                assert ev["ended"]
                worst = min(worst, ev["J"] / j0)
    assert worst >= r_floor          # uniform lower bound


# 5 ------------------------------------------------- uniform fuel


def test_uniform_fuel_covers_whole_box():
    gp, pl = _planning()
    bu = gp["burst"]
    b_star = gp["d0"] / gp["mu_b"]
    c = gp["c_recurrent"]
    zcap = theory.z_cap_invariant(
        bu["h"], pl["eps_max_event"], bu["b_sat"],
        w_max2=pl["w_max2_event"], z_seed=bu["z_seed"])
    y0_max = c + pl["eps_max_event"] - b_star
    fuel = theory.event_fuel_required_uniform(
        y0_max, 1.2 * c, c, gp["eta"], gp["th_J_loss"],
        gp["lam_wd_model"], bu["h"], bu["b_sat"], zcap,
        bu["z_floor"], bu["y_floor"], bu["tau_z"])
    # entries on the gap domain, in the middle region, at the
    # equality boundary b = c, and overloaded all end within fuel
    for b in list(np.linspace(0.0, c, 7)) + [c, 1.2 * c]:
        y0 = (0.02 + max(c - b, 0.0)) if b <= c else 0.15
        j0 = min(math.sqrt(y0 / gp["kappa"]), 1.0)
        ev = theory.run_event(
            b, 1.0, 1.0, j0, 0.05, c, gp["eta"], gp["kappa"],
            gp["th_J_loss"], gp["lam_wd_model"], bu,
            fuel=2 * fuel)
        assert ev["ended"], f"entry b = {b} must end"
        assert ev["n"] <= fuel


# 6 ---------------------------------------- direct-certificate source boundary




def test_gate_report_is_tiered():
    # per-class blocks with separate counts, a deferred block, and
    # no collapsed single pass count field
    rep = json.load(open(os.path.join(
        EXP, "protocols", "gate_report.json")))
    assert set(rep["counts"]) >= {"[thm]", "[orbit]", "[unit]",
                                  "[box]", "[stack]"}
    for v in rep["counts"].values():
        assert set(v) == {"passed", "failed"}
    assert isinstance(rep["deferred"], list) and rep["deferred"]
    assert {
        "threshold_quotient_edge_state",
        "edge_state_increment_simultaneous_interval",
        "complete_source_increment_simultaneous_interval",
        "stationary_threshold_regression",
    } <= set(rep["deferred"])
    assert "total" not in rep and "n_checks" not in rep


# 7 --------------------------------------------- saturated rebuild


def test_jet_rebuild_saturates_at_one():
    gp, _ = _planning()
    n = 3
    state = dict(b=np.full(n, 0.1), a=np.ones(n), g=np.ones(n),
                 J=np.array([0.99, 1.0, 0.5]), w=np.zeros(n),
                 pend=np.zeros(n), act=np.array([5.0, 5.0, 0.0]))
    params = dict(beta=np.zeros((n, n)), d0=gp["d0"],
                  delta_a=gp["delta_a"], delta_J0=5.0,
                  kappa=gp["kappa"], th_a_loss=gp["th_a_loss"],
                  th_g_loss=gp["th_g_loss"],
                  th_J_loss=gp["th_J_loss"], lam_wd=0.0,
                  mu_b=gp["mu_b"], s_shed=gp["s_shed"],
                  burst=gp["burst"])
    c_t = np.full(n, 10.0)           # no topples: smooth branch
    new, top = theory.gated_model_step(state, params, c_t,
                                       gp["eta"], fuel=200000)
    assert not top.any()
    assert np.all(new["J"] <= 1.0)   # the clip binds at sites 0,1
    assert new["J"][2] <= 0.5        # gate closed: no rebuild
