"""Regression tests of the Revision 11 chain: the single curvature
state (no independent stress ledger; x = b + kappa u^2 definitional),
the decay ledger (every decay term counted once), the derived event
map with its exact drain identity and amplitude factor, the
activity-gated rebuild and its dichotomy (capture collapses the jet;
a constant schedule under the maintenance inequality keeps it), the
certified directional certificate constants, and the retired-form
witnesses (duplicate curvature, decay double count, naive clipping
defect, first-crossing reset) pinned as counterexample regressions.
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


# --------------------------------------- the single curvature state


def _gated_params(n=1, **over):
    params = dict(
        beta=np.zeros((n, n)),
        d0=np.full(n, 0.02),
        delta_a=np.full(n, 0.2),
        delta_J0=np.full(n, 2.0),
        kappa=np.full(n, 1.0),
        th_a_loss=np.full(n, 0.2),
        th_g_loss=np.zeros(n),
        th_J_loss=np.full(n, 0.01),
        lam_wd=0.0,
        mu_b=0.1,
        s_shed=0.5,
        burst=dict(h=0.1, b_sat=0.6, z_seed=0.01, z_floor=0.002),
    )
    params.update(over)
    return params


def _gated_state(n=1, J=0.5):
    return dict(
        b=np.full(n, 0.1), a=np.full(n, 1.0), g=np.full(n, 1.0),
        J=np.full(n, J), w=np.zeros(n), pend=np.zeros(n),
        act=np.zeros(n),
    )


def test_no_stress_ledger_in_state():
    # The constraint x = b + kappa u^2 is definitional: the map's
    # state carries no stress entry, and the curvature is always
    # computed through theory.curvature.
    state = _gated_state()
    params = _gated_params()
    new, _ = theory.gated_model_step(state, params,
                                     np.array([1.0]), 0.05, fuel=200000)
    assert set(new) == {"b", "a", "g", "J", "w", "pend", "act"}
    x = theory.curvature(new["b"], params["kappa"], new["a"],
                         new["g"], new["J"])
    assert x[0] == pytest.approx(
        new["b"][0] + params["kappa"][0]
        * (new["a"][0] * new["g"][0] * new["J"][0]) ** 2)


def test_retired_map_violates_the_constraint_smooth_and_topple():
    # The registered witnesses that retired the independent ledger:
    # from x = b + kappa u^2 = 1 (b = 0, u = 1, kappa = 1, source
    # contraction 0.2, mu_x = 0.4, eta = 0.01) the retired smooth
    # branch returns 0.992 while the factor and base laws return
    # 0.996004; the topple branch (c = 0.9, r = 1/2) returns 0.45.
    ones = np.ones(1)
    state = dict(x=ones.copy(), b=np.zeros(1), a=ones.copy(),
                 g=ones.copy(), J=ones.copy(), w=np.zeros(1),
                 pend=np.zeros(1))
    params = dict(beta=np.zeros((1, 1)), d0=np.zeros(1),
                  delta=np.zeros(1), kappa=1.0,
                  th_a=np.full(1, 0.2), th_g=np.zeros(1),
                  th_J=np.zeros(1), mu_b=0.0, mu_x=0.4,
                  r=0.5, rho_w=0.2)
    new, top = retired.closed_model_step(state, params,
                                         np.full(1, 2.0), 0.01)
    fac = new["b"] + 1.0 * (new["a"] * new["g"] * new["J"]) ** 2
    assert not top[0]
    assert new["x"][0] == pytest.approx(0.992)
    assert fac[0] == pytest.approx(0.996004)
    new2, top2 = retired.closed_model_step(state, params,
                                           np.full(1, 0.9), 0.01)
    fac2 = new2["b"] + 1.0 * (new2["a"] * new2["g"] * new2["J"]) ** 2
    assert top2[0]
    assert new2["x"][0] == pytest.approx(0.45)
    assert fac2[0] == pytest.approx(0.996004)


# --------------------------------------------------- decay ledger


def test_pure_decay_rate_counted_once():
    # Loss rates, base, loading, rebuild, carrier all zero; three
    # decayed path factors at lam_wd: the exact curvature increment
    # matches -2 * 3 * lam_wd * kappa u^2 to first order.
    kappa, lam_wd = 1.0, 0.1
    a, g, J = 0.8, 0.9, 0.7
    u = a * g * J
    target = theory.pure_decay_rate(3, lam_wd, kappa, u)
    for eta in (1e-3, 1e-4, 1e-5):
        first, exact = theory.curvature_increment_exact(
            0.0, kappa, a, g, J, 0.0, eta, d0=0.0, delta=0.0,
            th_a_loss=0.0, th_g_loss=0.0, th_J_loss=0.0,
            lam_wd=lam_wd, mu_b=0.0)
        assert first / eta == pytest.approx(target, rel=1e-12)
        assert abs(exact - first) <= 10.0 * eta ** 2


def test_retired_loading_law_double_counts_decay():
    # The registered decay double-count witness: p = 2 decayed path
    # factors, lam_wd = 0.1, x = kappa u^2 = 1, factor velocity
    # carrying exactly the decay shares: the retired law drains at
    # first-order rate -0.8 where the true factory rate is -0.4.
    p_i, lam_wd, kappa, u = 2, 0.1, 1.0, 1.0
    x = kappa * u * u
    v = theory.source_velocity(1, 1, 1, 0.0, 0.0, lam_wd, lam_wd,
                               0.0, kappa)
    assert v == pytest.approx(-p_i * lam_wd * u)
    eta = 1e-4
    xp = retired.loading_step(x, eta, 0.0, kappa, u, v, p_i, lam_wd)
    assert (xp - x) / eta == pytest.approx(-0.8)
    assert theory.pure_decay_rate(p_i, lam_wd, kappa, u) == \
        pytest.approx(-0.4)


# ------------------------------------------------ averaged defect


def test_clipping_defect_witness_and_repaired_form():
    # The registered optimizer witness: two-step scalar window
    # g = (-2, 2), clip level 1, inverse preconditioners (1, 2).
    # The retired form (average before preconditioning) returns 0;
    # the true signed preconditioned defect is -1/2; the repaired
    # form (preconditioning inside the average) bounds it.
    g_list = [-2.0, 2.0]
    pinv_list = [1.0, 2.0]
    assert theory.clip_defect_naive(g_list, 1.0) == 0.0
    signed = sum(p * (max(-1.0, min(1.0, g)) - g)
                 for p, g in zip(pinv_list, g_list)) / 2.0
    assert signed == pytest.approx(-0.5)
    eps_c = theory.clip_defect_preconditioned(pinv_list, g_list, 1.0)
    assert eps_c == pytest.approx(1.5)
    assert eps_c >= abs(signed)


# ---------------------------------------------------- event map


def test_drain_identity_exact_on_normal_form():
    # Lean: EventMap.drain_identity.  eps at the event end equals
    # eps0 - h a * (integrated power), exactly, and the retained
    # fraction identity follows.
    z0, eps0, h, a, b, z_floor, c = 0.005, 0.0004, 0.15, 1.0, 0.6, \
        0.001, 0.1
    n, ztr, etr = retired.simulate_burst_to_floor(z0, eps0, h, a, b,
                                                 z_floor)
    assert n is not None
    total = sum(ztr[:n])
    assert etr[n] == pytest.approx(eps0 - h * a * total, abs=1e-12)
    r_sim = 1.0 + etr[n] / c
    assert r_sim == pytest.approx(
        retired.retained_fraction_from_drain(eps0, h, a, total, c))


def test_strict_interval_brackets_simulated_event():
    # The event-end strict interval brackets the simulated retained
    # fraction at the planning constants (b_sat >= h a / 2 so the
    # energy bound |eps_end| <= sqrt(Q0) applies).
    z0, eps0, h, a, b_sat = 0.005, 0.0004, 0.15, 1.0, 0.6
    z_min, eps_plus, c, r_minus, z_floor = 0.005, 0.0005, 0.1, \
        0.05, 0.001
    assert b_sat >= h * a / 2.0
    r_lo, r_hi = retired.reset_interval_strict(
        z0, eps0, h, a, c, z_min, eps_plus, r_minus)
    n, ztr, etr = retired.simulate_burst_to_floor(z0, eps0, h, a,
                                                 b_sat, z_floor)
    r_sim = 1.0 + etr[n] / c
    assert r_lo - 1e-12 <= r_sim <= r_hi + 1e-12


def test_amp_factor_licensed_on_event():
    # z = w^2 and z_end < z_floor give |w_end|/|w_0| below the
    # amplitude factor sqrt(z_floor / z0); the sign is preserved.
    burst = dict(h=0.1, b_sat=0.6, z_seed=0.01, z_floor=0.002)
    w = -0.3
    ev = theory.run_event(b=0.0, a=1.0, g=1.0, J=1.0, w=w, c=0.08,
                          eta=0.01, kappa=0.1, th_J_loss=0.1,
                          lam_wd=0.1, burst=burst, fuel=200000)
    assert ev["ended"]
    z0 = max(burst["z_seed"], w * w)
    assert abs(ev["w_end"]) / abs(w) < math.sqrt(
        burst["z_floor"] / z0) + 1e-12
    assert ev["w_end"] < 0.0
    # rho_w bookkeeping: the derived factor is a power ratio and the
    # licensed amplitude factor is its square root
    rho = theory.rho_w_derived(0.01, 0.05)
    assert theory.amp_factor(rho) == pytest.approx(math.sqrt(0.2))


def test_event_ledger_and_shed():
    # A toppling site resolves its event: curvature drop realized
    # through the jet, shed = s_shed * drop, activity set to the
    # event length, event ends at the power floor.
    params = _gated_params()
    state = _gated_state(J=0.7)     # x = 0.1 + 0.49 = 0.59 > c
    c_t = np.array([0.3])
    new, topple = theory.gated_model_step(state, params, c_t, 0.05, fuel=200000)
    assert topple[0]
    x_end = theory.curvature(new["b"], params["kappa"], new["a"],
                             new["g"], new["J"])[0]
    assert x_end <= c_t[0] + 1e-9        # the event ended below c
    assert new["J"][0] < state["J"][0]   # drop realized via the jet
    assert new["act"][0] > 0
    assert new["pend"][0] > 0.0
    # the shed is s_shed times the realized curvature drop
    x_pre_smooth = None  # recompute the pre-event (post-smooth) x
    A = 0.0
    a1 = (1 - 0.05 * (0.2 + 0.0)) * 1.0 + 0.05 * 0.2
    g1 = 1.0
    J1 = (1 - 0.05 * (0.01 + 0.0)) * 0.7 + 0.05 * 2.0 * A
    b1 = 0.1 + 0.05 * 0.02 - 0.05 * 0.1 * 0.1
    x_pre_smooth = b1 + (a1 * g1 * J1) ** 2
    drop = x_pre_smooth - x_end
    assert new["pend"][0] == pytest.approx(params["s_shed"] * drop,
                                           rel=1e-9)


# --------------------------------------------- the gated dichotomy


def test_captured_phase_gate_closes_and_jet_collapses():
    # Capture (thresholds above every loaded value): no topples, so
    # the gate never opens, the rebuild is off as a CONSEQUENCE, and
    # the jet enters the collapsed set within the printed bound.
    params = _gated_params()
    state = _gated_state(J=0.5)
    c_t = np.array([1.0])
    eta = 0.05
    J_crit = 0.1
    q = 1.0 - eta * params["th_J_loss"][0]
    bound = theory.entry_time_bound(0.5, J_crit, q)
    entry = None
    for t in range(4000):
        state, topple = theory.gated_model_step(state, params, c_t,
                                                eta, fuel=200000)
        assert not topple.any()
        if entry is None and state["J"][0] <= J_crit:
            entry = t
        if entry is not None:
            assert state["J"][0] <= J_crit + 1e-12  # absorption
    assert entry is not None and entry <= bound
    # one curvature, one limit: x -> b -> d0/mu_b as u dies
    assert state["b"][0] == pytest.approx(0.2, abs=1e-3)
    x = theory.curvature(state["b"], params["kappa"], state["a"],
                         state["g"], state["J"])[0]
    assert x < 0.25


def test_recurrent_phase_maintains_jet_above_criterion():
    # Constant schedule with sustained drive: events recur, the gate
    # keeps opening, and the jet holds above the criterion (the
    # maintenance side of the dichotomy) under the SAME update law.
    params = _gated_params()
    state = _gated_state(J=0.5)
    c_t = np.array([0.3])
    eta = 0.05
    J_crit = 0.1
    topples = 0
    J_min_late = 1.0
    for t in range(5000):
        state, topple = theory.gated_model_step(state, params, c_t,
                                                eta, fuel=200000)
        topples += int(topple.any())
        if t >= 500:
            J_min_late = min(J_min_late, float(state["J"][0]))
    assert topples >= 5                # recurrent events
    assert J_min_late > J_crit         # the jet never collapses


def test_entry_time_bound_cases():
    assert theory.entry_time_bound(1.0, 0.1, 0.5) == 4
    assert 0.5 ** 4 <= 0.1
    assert theory.entry_time_bound(1.0, 0.1, 0.0) == 1   # q = 0
    assert theory.entry_time_bound(0.05, 0.1, 0.5) == 0  # already in
    with pytest.raises(ValueError):
        theory.entry_time_bound(1.0, 0.1, 1.0)           # q < 1 only


def test_maintenance_floor_invariant():
    q, m = 0.9, 0.02
    floor = theory.maintenance_floor(q, m)
    assert floor == pytest.approx(0.2)
    J = floor
    for _ in range(200):
        J = q * J + m
        assert J >= floor - 1e-12
    J = 0.5
    for _ in range(200):
        J = q * J + m
        assert J >= floor - 1e-12


# ------------------------------------------- recurrent stay-below


def test_hold_below_bound_dominates_mc():
    # The recurrent stay-below envelope: MC hold probability of the
    # floored AR(1) with balance above the threshold is below
    # (1 - p_min)^T.
    c, L, a, sig = 0.1, 0.15, 0.2, 0.05
    gap = L - c
    p_min = theory.p_min_crossing(c, gap, a, sig)
    T = 50
    bound = theory.hold_below_bound(p_min, T)
    rng = np.random.default_rng(9)
    n_mc = 4000
    x = np.zeros(n_mc)
    alive = np.ones(n_mc, dtype=bool)
    for _ in range(T):
        x = np.maximum((1 - a) * x + a * L
                       + rng.normal(0.0, sig, size=n_mc), 0.0)
        alive &= x < c
    assert alive.mean() <= bound + 3.0 / math.sqrt(n_mc)


def test_p_min_crossing_validation():
    with pytest.raises(ValueError):
        theory.p_min_crossing(0.1, -0.1, 0.2, 0.05)
    with pytest.raises(ValueError):
        theory.p_min_crossing(0.1, 0.05, 1.5, 0.05)
    with pytest.raises(ValueError):
        theory.hold_below_bound(1.5, 10)
