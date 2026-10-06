"""Permanent regression tests of the factorized-stress chain: the
drive-equality structure, the closure step conditions (Q+) with the
period-two necessity witness, the strict burst package, the honest
certificate boundary, the derived carrier contraction, and the
conditional-domination cascade bound.

Each counterexample here is a NECESSITY witness: a future weakening
of the corresponding hypothesis breaks the matching test.
"""

import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", "analysis"))

import theory  # noqa: E402
import retired  # noqa: E402


# ------------------------------------------- frozen source and base


def test_frozen_source_and_base_manufacture_nothing_end_to_end():
    # Lean: FactoredState.driveRate at v = 0, d0 = 0.  The end-to-end
    # regression the drive mismatch would have failed: a frozen
    # source (all factor velocities zero) and a frozen base (d0 = 0)
    # load exactly nothing through the closed map.
    assert theory.drive_law(0.0, 2.0, 0.7, 0.0) == 0.0
    n = 1
    params = dict(
        beta=np.zeros((n, n)),
        d0=np.zeros(n),
        delta=np.zeros(n),
        kappa=np.array([0.5]),
        th_a=np.zeros(n),
        th_g=np.zeros(n),
        th_J=np.zeros(n),
        mu_b=0.0,
        mu_x=0.0,
        r=0.5,
        rho_w=0.5,
    )
    state = dict(
        x=np.array([0.3]), b=np.array([0.5]), a=np.array([1.0]),
        g=np.array([1.0]), J=np.array([1.0]), w=np.array([0.0]),
        pend=np.zeros(n),
    )
    for _ in range(50):
        state, topple = retired.closed_model_step(
            state, params, np.array([10.0]), 1.0)
        assert not topple.any()
    # frozen everything: the stress ledger has not moved at all
    assert state["x"][0] == pytest.approx(0.3, abs=1e-15)
    # and with reversion on (mu_x > 0) the ONLY motion is reversion
    params["mu_x"] = 0.4
    state = dict(
        x=np.array([0.3]), b=np.array([0.5]), a=np.array([1.0]),
        g=np.array([1.0]), J=np.array([1.0]), w=np.array([0.0]),
        pend=np.zeros(n),
    )
    eta = 0.1
    state, _ = retired.closed_model_step(state, params,
                                        np.array([10.0]), eta)
    assert state["x"][0] == pytest.approx(0.3 * (1 - eta * 0.4))


def test_drive_equality_loading_step_vs_drive_law():
    # Structural drive equality: the loading-law increment minus
    # eta * drive_law is exactly the reversion term, for random data.
    rng = np.random.default_rng(7)
    for _ in range(50):
        x, eta, d0, kappa, u, v, lam_wd, h = rng.uniform(
            0.01, 1.0, size=8)
        p_i = int(rng.integers(2, 4))
        x1 = retired.loading_step(x, eta, d0, kappa, u, v, p_i,
                                 lam_wd, h_euler=h)
        drive = theory.drive_law(d0, kappa, u, v)
        assert x1 - x - eta * drive == pytest.approx(
            -eta * (2.0 * p_i * lam_wd * x + lam_wd * h), rel=1e-12)


# ------------------------------------------- (Q+) and the period two


def _period_two_setup():
    n = 1
    params = dict(
        beta=np.zeros((n, n)),
        d0=np.array([0.5]),      # the old lam0-as-rate abuse
        delta=np.zeros(n),
        kappa=np.array([0.5]),
        th_a=np.zeros(n),
        th_g=np.zeros(n),
        th_J=np.zeros(n),
        mu_b=0.0,
        mu_x=np.array([3.0]),    # eta*mu_x = 3 > 1: violates (Q+)
        r=0.5,
        rho_w=0.5,
    )
    state = dict(
        x=np.zeros(n), b=np.zeros(n), a=np.zeros(n), g=np.zeros(n),
        J=np.zeros(n), w=np.zeros(n), pend=np.zeros(n),
    )
    return state, params


def test_period_two_witness_reproduces_without_validation():
    # The closure counterexample as a permanent regression: with the
    # (Q+) validator off, the admitted data produce the period-two
    # orbit 0.5, 0, 0.5, 0, ... with no topples, not the balance
    # point d0/mu_x = 1/6.  Lean: ClosureCaptured period-two witness.
    state, params = _period_two_setup()
    xs, topples = [], []
    for t in range(8):
        c_t = np.array([1.0 if t == 0 else 1.5])
        state, topple = retired.closed_model_step(
            state, params, c_t, 1.0, validate=False)
        xs.append(float(state["x"][0]))
        topples.append(bool(topple.any()))
    assert xs == [0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0]
    assert not any(topples)
    assert abs(xs[-1] - 0.5 / 3.0) > 0.1  # not the claimed limit


def test_period_two_data_rejected_by_Qplus_validator():
    state, params = _period_two_setup()
    with pytest.raises(ValueError, match="mu_x"):
        retired.closed_model_step(state, params, np.array([1.0]), 1.0)


def test_compliant_run_converges_to_balance():
    # Under (Q+) (eta*mu_x <= 1) the same one-site model converges to
    # d0/mu_x, as the repaired closure theorem states.
    state, params = _period_two_setup()
    eta = 0.25                    # eta*mu_x = 0.75 <= 1
    c_t = np.array([10.0])        # no topples in range
    for _ in range(4000):
        state, _ = retired.closed_model_step(state, params, c_t, eta)
    assert state["x"][0] == pytest.approx(0.5 / 3.0, abs=1e-6)


# ------------------------------------------- certificate boundary


def test_lyap2_closed_form_matches_solver():
    for sigma in (0.3, 0.8, 1.5, 2.5):
        for beta1 in (0.85, 0.9, 0.95):
            for d in (0.02, 0.05):
                if not theory.jury_stable(sigma, beta1, d):
                    continue
                s = 1.0 + beta1 - sigma - d
                q = beta1 * (1.0 - d)
                assert theory.lyap2_residual(s, q) < 1e-12
                p11, p12, p22 = theory.lyap2_closed_form(s, q)
                P = theory.lyap_solve_2x2(
                    theory.companion_matrix(sigma, beta1, d))
                assert p11 == pytest.approx(P[0, 0], rel=1e-9)
                assert p12 == pytest.approx(P[0, 1], rel=1e-9)
                assert p22 == pytest.approx(P[1, 1], rel=1e-9)


def test_sharpness_ramp_moves_certificate_at_zero_eps_nu():
    # The frozen-boundary witness: a pure sharpness change (zero
    # second-moment drift) has positive full companion drift, and the
    # certificate boundary strictly retreats from the frozen flip.
    beta1, d = 0.9, 0.05
    eps_A = theory.drift_full((1.0, beta1, d), (1.004, beta1, d))
    assert eps_A == pytest.approx(0.004)
    C_P = 5.0
    boundary = theory.c_grnt(beta1, d, eps_A=eps_A, C_P=C_P)
    assert boundary < theory.flip_boundary(beta1, d)
    assert boundary > 0.0


def test_cp_dir_bounds_dominate_grid_check():
    # The certified constants bound the secondary finite-difference
    # grid statistic on the same box (the grid is a sanity check of
    # the certificate, never the evaluation).
    box = (0.5, 1.5, 0.88, 0.92, 0.0, 2e-4)
    bounds = theory.cp_dir_bounds(*box)
    grid = theory.cp_grid_check(*box, n=9)
    assert bounds["C_P"] > 0.0
    assert grid <= bounds["C_P"]


def test_cp_dir_bounds_certify_matrix_lipschitz():
    # Property check of the certificate: for random pairs in the
    # box, ||P(nu) - P(nu')||_op <= K_s |ds| + K_q |dq|.
    box = (0.3, 1.6, 0.85, 0.95, 0.0, 2e-4)
    bounds = theory.cp_dir_bounds(*box)
    rng = np.random.default_rng(2)
    for _ in range(200):
        nu = (rng.uniform(box[0], box[1]),
              rng.uniform(box[2], box[3]),
              rng.uniform(box[4], box[5]))
        nu2 = (rng.uniform(box[0], box[1]),
               rng.uniform(box[2], box[3]),
               rng.uniform(box[4], box[5]))
        P1 = theory.lyap_solve_2x2(theory.companion_matrix(*nu))
        P2 = theory.lyap_solve_2x2(theory.companion_matrix(*nu2))
        gap = float(np.linalg.norm(P1 - P2, 2))
        ds, dq = theory.drift_dir(nu, nu2)
        assert gap <= bounds["K_s"] * ds + bounds["K_q"] * dq + 1e-9


def test_cp_dir_bounds_reject_boundary_box():
    # A box reaching the flip boundary has no positive margin.
    with pytest.raises(ValueError, match="gamma2"):
        theory.cp_dir_bounds(0.1, 3.8, 0.85, 0.95, 0.0, 2e-4)


def test_c_grnt_box_confined_and_flagged():
    bounds = theory.cp_dir_bounds(0.3, 3.0, 0.85, 0.95, 0.0, 2e-4)
    # tiny budget: certified through the whole box, reported at edge
    res = theory.c_grnt_box(0.9, 1e-4, 1e-12, 0.0, bounds)
    assert res["at_edge"] and res["sigma_star"] == 3.0
    # large budget: certified nowhere on the box
    res = theory.c_grnt_box(0.9, 1e-4, 1.0, 1.0, bounds)
    assert res["empty"]
    # intermediate: interior boundary strictly inside the box (pmax
    # turns back up toward the flip, so the certified set is an
    # interior band and the search reports its upper edge)
    res = theory.c_grnt_box(0.9, 1e-4, 8e-6, 0.0, bounds)
    assert not res["empty"] and not res["at_edge"]
    assert 0.3 < res["sigma_star"] < 3.0


# ------------------------------------------- strict burst package


def test_reset_interval_strict_inside_unit_interval():
    # The planning constants of the calibration satisfy (E-seed), the
    # margin cap, (M-drain), and (M-sep) AS PRINTED (event-end
    # convention, sqrt(Q0)): the interval is strictly inside (0, 1).
    r_lo, r_hi = retired.reset_interval_strict(
        z0=0.005, eps0=0.0004, h=0.15, a=1.0, c=0.1, z_min=0.005,
        eps_plus=0.0005, r_minus=0.05)
    assert 0.0 < r_lo <= r_hi < 1.0
    assert r_lo == pytest.approx(
        1.0 - math.sqrt(0.005 + 0.0004 ** 2) / 0.1)
    assert r_hi == pytest.approx(
        1.0 - (0.15 * 0.005 - 0.0005) / 0.1)


def test_event_end_convention_witness():
    # The convention witness that retired the first-crossing bound:
    # at the OLD planning constants the first-crossing variant
    # returns (0.95, 0.955) although the theorem's (M-sep) fails
    # (sqrt(Q0) = 0.2236 > 0.095); the event-end form raises.
    kw = dict(z0=0.05, eps0=0.0005, h=0.1, a=1.0, c=0.1, z_min=0.05,
              eps_plus=0.0005, r_minus=0.05)
    lo, hi = retired.reset_interval_strict_firstcrossing(**kw)
    assert (round(lo, 3), round(hi, 3)) == (0.95, 0.955)
    with pytest.raises(ValueError, match="M-sep"):
        retired.reset_interval_strict(**kw)


def test_reset_interval_review_numbers_raise():
    # The review's reset example fails the strict package for every
    # admissible z_min: it is the necessity witness for (M-sep).
    for z_min in (0.01, 0.1, 0.5, 1.0):
        with pytest.raises(ValueError):
            retired.reset_interval_strict(
                z0=1.0, eps0=0.1, h=0.1, a=1.0, c=0.1, z_min=z_min,
                eps_plus=0.1, r_minus=0.05)
    # and the unstrengthened interval indeed reaches below zero
    lo, hi = retired.reset_interval(1.0, 0.1, 0.1, 1.0)
    assert 1.0 + lo / 0.1 < 0.0  # retained fraction below zero
    assert hi == 0.0             # and touches one


def test_rho_w_derived_and_event_end_bound():
    assert theory.rho_w_derived(0.01, 0.05) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        theory.rho_w_derived(0.05, 0.05)
    n = retired.event_end_bound(eps0=0.1, estar=0.05, h=0.1, a=1.0,
                               b=0.05, z_floor=0.01, z_cap=0.06)
    expected = (math.ceil((0.1 + 0.05) / (0.1 * 1.0 * 0.01))
                + math.ceil(math.log(0.06 / 0.01)
                            / (2 * 0.1 * 0.05)))
    assert n == expected
    assert 330 <= n <= 332


def test_carrier_contraction_on_simulated_burst():
    # (E-end) + (E-seed) give the derived rho_w on the real normal
    # form: run to the first step with z < z_floor and compare.
    z_min, z_floor = 0.05, 0.01
    z0, eps0, h, a, b = 0.05, 0.2, 0.05, 1.0, 0.6
    assert z0 >= z_min
    z, eps = z0, eps0
    for _ in range(100000):
        z, eps = retired.burst_step(z, eps, h, a, b)
        if z < z_floor:
            break
    assert z < z_floor
    assert z / z0 <= theory.rho_w_derived(z_floor, z_min)


# ------------------------------------------- entry vs trigger


def test_entry_prob_below_trigger_band_upper():
    # The stay-below (entry) probability at a margin is far below
    # the upward trigger band's upper bound at the same margin:
    # the two events must never be conflated.
    eta, mu, s2B, T = 0.05, 0.5, 0.25, 200
    delta = 0.05
    p_entry = theory.entry_prob_mc(delta, eta, mu, s2B, T,
                                   n_mc=4000, seed=3)
    _, upper = theory.passage_band(delta, eta, mu, s2B, T)
    assert 0.0 <= p_entry <= 1.0
    assert p_entry < upper + 1e-12


# ------------------------------------------- domination MC


def test_subcritical_domination_bounds_total_progeny():
    # (K-dom) reference instantiation: each topple's children are
    # independent Bernoulli(p_j), sum p_j = Rc = 0.7.  The empirical
    # mean total progeny obeys the (1 - Rc)^-1 bound and the tails
    # are dominated geometrically (generation-k survival <= Rc^k).
    rng = np.random.default_rng(11)
    Rc = 0.7
    p = np.array([0.3, 0.25, 0.15])
    n_casc = 2000
    sizes = np.empty(n_casc)
    deep = 0
    k_test = 5
    for i in range(n_casc):
        gen, total, depth = 1, 1, 0
        while gen > 0 and total < 10000:
            children = int(
                (rng.uniform(size=(gen, 3)) < p).sum())
            total += children
            gen = children
            depth += 1 if children > 0 else 0
        sizes[i] = total
        if depth >= k_test:
            deep += 1
    mean, se = sizes.mean(), sizes.std() / math.sqrt(n_casc)
    assert mean <= 1.0 / (1.0 - Rc) + 3.0 * se
    # generation-k survival is dominated by Rc^k (3x MC slack)
    assert deep / n_casc <= (Rc ** k_test) * 3.0
