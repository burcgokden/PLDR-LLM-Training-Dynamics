"""The overload classification: driven persistence, headroom
resolution, strict least counts, and fuel-as-data.

The persistent-overload orbit is the PERMANENT NEGATIVE CONTROL of
the classification theorem (paper Thm. overloadclass): a legal
one-site orbit with one-gap reload saturation topples at every
global step for all time while its base relaxes far below
threshold.  No proof that tracks the base alone can close the
guard; the test iterates the FULL global map, never one inner
resolver.  The companion positive controls certify (O-head)
resolution, the (O-load) all-hot sampled-jet ceiling, the strict
least counts at their equality boundaries, and starvation/success
at adjacent fuel values through the global interface (the shared
strict-indexing convention: an exit at inner index N needs granted
fuel N + 1)."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "analysis"))
import theory  # noqa: E402


WITNESS_BURST = dict(h=0.1, b_sat=1.0, z_seed=0.01, z_floor=0.001,
                     y_floor=0.01, tau_z=0.02, w_kick_max=0.1)


def _witness_params(**over):
    p = dict(beta=np.zeros((1, 1)), d0=0.0, delta_a=0.0,
             delta_J0=20.0, kappa=1.0, th_a_loss=0.0,
             th_g_loss=0.0, th_J_loss=0.1, lam_wd=0.0, mu_b=0.1,
             s_shed=0.0, burst=dict(WITNESS_BURST))
    p.update(over)
    return p


def _one_site_state(b=1.0, a=1.0, g=1.0, J=1.0, w=0.0, act=1.0):
    return {k: np.array([v]) for k, v in
            dict(b=b, a=a, g=g, J=J, w=w, pend=0.0, act=act).items()}


def test_persistent_overload_witness_pinned():
    # The driven witness: (O-reb) holds (eta*delta_J0 = 2 >= 1,
    # kappa*(a g)^2 = 1 > c = 0.5), so the site topples at EVERY
    # global step for all time even as b -> d0/mu_b = 0 < c.  The
    # orbit is pinned to printed precision; the reload counter is
    # renewed to one by every event, and the exits shift from hot
    # to cold as the base relaxes through the threshold, exactly
    # the two-clock signature E2 reports.
    state = _one_site_state()
    params = _witness_params()
    c_t, eta = np.array([0.5]), 0.1
    pinned = {
        1: (0.99, 0.0751679124330011, "hot"),
        51: (0.5989560064661613, 0.036904511051694164, "hot"),
        101: (0.36237201786049716, 0.04408646932896339, "cold"),
        201: (0.13263987810938235, 0.22965563415019818, "cold"),
        300: (0.04904089407128583, 0.3017050170948095, "cold"),
    }
    for step in range(1, 301):
        events = []
        state, top = theory.gated_model_step(
            state, params, c_t, eta, fuel=200000, events_out=events)
        assert bool(top[0]), f"step {step} did not topple"
        assert len(events) == 1 and events[0]["ended"]
        assert state["act"][0] == 1.0
        if step in pinned:
            b_ref, j_ref, ex_ref = pinned[step]
            assert float(state["b"][0]) == pytest.approx(
                b_ref, abs=1e-15)
            assert float(state["J"][0]) == pytest.approx(
                j_ref, abs=1e-15)
            assert events[0]["exit_type"] == ex_ref
    # base far below threshold, toppling never stopped: the
    # refuted base-relaxation resolution can never return
    assert float(state["b"][0]) < 0.05 < 0.5


def test_driven_rebuild_certificate_on_witness():
    # (O-reb) margin positive on the witness constants; the
    # one-gap rebuilt jet saturates at 1 for every post-event jet.
    margin = theory.driven_rebuild_margin(
        eta=0.1, delta_J0=20.0, kappa=1.0, a_inf=1.0, g_inf=1.0,
        c_plus=0.5, th_J_loss=0.1, lam_wd=0.0)
    assert margin == pytest.approx(0.5)
    for j_out in (0.0, 0.3, 0.99):
        assert theory.rebuilt_jet(
            j_out, w=0.03, eta=0.1, th_J_loss=0.1, lam_wd=0.0,
            kappa=1.0, delta_J0=20.0) == 1.0
    # without saturation the rebuild is the clipped affine step
    assert theory.rebuilt_jet(
        0.5, w=0.0, eta=0.1, th_J_loss=0.1, lam_wd=0.0,
        kappa=1.0, delta_J0=1.0) == pytest.approx(
            (1.0 - 0.01) * 0.5 + 0.1)
    with pytest.raises(ValueError):
        theory.driven_rebuild_margin(
            eta=0.1, delta_J0=5.0, kappa=1.0, a_inf=1.0, g_inf=1.0,
            c_plus=0.5, th_J_loss=0.1, lam_wd=0.0)


def test_driven_load_all_hot_and_sampled_jet_ceiling():
    # (O-load): base equilibrium d0/mu_b = 0.6 above the band, so
    # the guard fires from the base alone at every step, every
    # exit is hot, and every LOGGED post-event jet sits at or
    # below the exhaustion ceiling J_drv = sqrt(y_floor/kappa)/
    # (a_inf g_inf): collapse by drive when J_drv < J_crit.
    assert theory.driven_load_margin(
        d0=0.06, mu_b=0.1, c_plus=0.5) == pytest.approx(0.1)
    j_drv = theory.driven_sampled_jet_max(
        y_floor=WITNESS_BURST["y_floor"], kappa=1.0,
        a_inf=1.0, g_inf=1.0)
    assert j_drv == pytest.approx(0.1)
    state = _one_site_state()
    params = _witness_params(d0=0.06)
    c_t, eta = np.array([0.5]), 0.1
    for step in range(1, 101):
        events = []
        state, top = theory.gated_model_step(
            state, params, c_t, eta, fuel=200000, events_out=events)
        assert bool(top[0])
        assert events[0]["exit_type"] == "hot"
        assert float(state["J"][0]) <= j_drv + 1e-12
        assert float(state["b"][0]) > 0.5


def test_overload_resolution_under_headroom():
    # (O-head): the saturated-jet drive kappa*(a g)^2 = 0.0625
    # fits under c_- = 0.5 with gap delta_b = 0.4, so the run is
    # certified to resolve within N_res global steps of the full
    # map, and the site stays quiet forever after.
    margin = theory.overload_headroom_margin(
        kappa=1.0, a_sup=0.5, g_sup=0.5, d0_sup=0.0, mu_b=0.1,
        c_minus=0.5)
    assert margin == pytest.approx(0.4375)
    n_res = theory.overload_resolution_bound(
        b0=1.0, d0_sup=0.0, mu_b=0.1, eta=0.1, kappa=1.0,
        a_sup=0.5, g_sup=0.5, c_minus=0.5, delta_b=0.4)
    state = _one_site_state(a=0.5, g=0.5)
    params = _witness_params()
    c_t, eta = np.array([0.5]), 0.1
    last_topple = None
    for step in range(1, n_res + 101):
        state, top = theory.gated_model_step(
            state, params, c_t, eta, fuel=200000)
        if bool(top[0]):
            last_topple = step
    assert last_topple is not None and last_topple <= n_res
    # the certificate refuses configurations without headroom
    with pytest.raises(ValueError):
        theory.overload_resolution_bound(
            b0=1.0, d0_sup=0.0, mu_b=0.1, eta=0.1, kappa=1.0,
            a_sup=1.0, g_sup=1.0, c_minus=0.5, delta_b=0.4)


def test_zero_entry_event_class():
    # A legal overloaded entry at zero jet: y0 = 0, the event ends
    # at inner index zero, the product drain form holds trivially,
    # and the logarithmic endpoint is out of domain (E2 reports
    # the class separately; nothing raises).
    ev = theory.run_event(
        b=0.75, a=1.0, g=1.0, J=0.0, w=0.0, c=0.5, eta=0.1,
        kappa=1.0, th_J_loss=0.1, lam_wd=0.0,
        burst=dict(WITNESS_BURST), fuel=1000)
    assert ev["ended"] and ev["n"] == 0
    assert ev["exit_type"] == "hot"
    assert ev["sum_log_phi"] == 0.0
    assert 1.0 * (1.0 * 1.0 * ev["J"]) ** 2 == 0.0


def test_strict_exit_equality_regression():
    # The equality boundary Z_cap = Z_tgt: the exit is strict, so
    # the per-entry count is one, never zero, and it dominates the
    # actual first exit index.
    bound = theory.event_time_bound_overloaded(
        y0=0.0, b=0.75, c=0.5, eta=0.1, th_J_loss=0.1, lam_wd=0.0,
        h=0.1, b_sat=1.0, z_cap=0.5, y_floor=0.125, tau_z=0.25)
    burst = dict(h=0.1, b_sat=1.0, z_seed=0.5, z_floor=0.01,
                 y_floor=0.125, tau_z=0.25, w_kick_max=0.8)
    ev = theory.run_event(
        b=0.75, a=1.0, g=1.0, J=0.0, w=0.0, c=0.5, eta=0.1,
        kappa=1.0, th_J_loss=0.1, lam_wd=0.0, burst=burst,
        fuel=1000)
    assert bound == 1
    assert ev["ended"] and ev["n"] == 1
    assert bound >= ev["n"]


def test_strict_geometric_count_boundaries():
    # least strict count on a boundary grid: at, just above, and
    # just below exact geometric equality, against the powers
    # actually compared
    # exact equality (representable exactly at rho = 1/2): the
    # count advances by one past the equality boundary
    for k_true in (0, 1, 2, 7, 40):
        z0 = float(2 ** k_true)                  # (1/2)^k * z0 == 1
        assert theory.strict_geometric_count(z0, 1.0, 0.5) \
            == k_true + 1
    # least-count invariants on a floating boundary grid: strict
    # passage at k, no strict passage at k - 1, against the powers
    # actually compared
    for rho in (0.5, 0.9, 0.975, 0.99):
        for k_true in (0, 1, 2, 7, 40):
            target = 1.0
            for wobble in (1.0 - 1e-9, 1.0, 1.0 + 1e-9):
                z0 = target / rho ** k_true * wobble
                k = theory.strict_geometric_count(z0, target, rho)
                assert k_true <= k <= k_true + 1
                assert rho ** k * z0 < target
                assert k == 0 or rho ** (k - 1) * z0 >= target
    assert theory.strict_geometric_count(0.5, 1.0, 0.9) == 0
    with pytest.raises(ValueError):
        theory.strict_geometric_count(1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        theory.strict_geometric_count(1.0, 0.0, 0.5)


def test_overloaded_bound_dominates_first_hit_on_grid():
    # property check: the analytic per-entry count dominates the
    # empirical first-hit index over a boundary grid that includes
    # the shallow-overload limit and the cap-at-target boundary
    for b in (0.5 + 1e-6, 0.55, 0.75, 0.9):
        for z_seed in (0.02, 0.26, 0.5):
            burst = dict(h=0.1, b_sat=1.0, z_seed=z_seed,
                         z_floor=0.01, y_floor=0.125, tau_z=0.25,
                         w_kick_max=1.0)
            ev = theory.run_event(
                b=b, a=1.0, g=1.0, J=0.0, w=0.0, c=0.5, eta=0.1,
                kappa=1.0, th_J_loss=0.1, lam_wd=0.0, burst=burst,
                fuel=100000)
            bound = theory.event_time_bound_overloaded(
                y0=0.0, b=b, c=0.5, eta=0.1, th_J_loss=0.1,
                lam_wd=0.0, h=0.1, b_sat=1.0, z_cap=max(z_seed, 0.5),
                y_floor=0.125, tau_z=0.25)
            assert ev["ended"] and bound >= ev["n"]


def test_fuel_starvation_and_success_through_global_map():
    # fuel is data: through the GLOBAL interface, granted fuel
    # N + 1 succeeds and N starves (the strict-indexing convention
    # shared by paper, Lean, and Python)
    def entry():
        return _one_site_state(b=0.1, act=0.0)

    params = _witness_params(delta_J0=0.0)
    c_t, eta = np.array([0.5]), 0.1
    events = []
    state, top = theory.gated_model_step(
        entry(), params, c_t, eta, fuel=200000, events_out=events)
    assert bool(top[0]) and events[0]["ended"]
    n_exit = events[0]["n"]
    assert n_exit >= 1
    state, top = theory.gated_model_step(
        entry(), params, c_t, eta, fuel=n_exit + 1)
    assert bool(top[0])
    with pytest.raises(theory.EventNotEnded):
        theory.gated_model_step(entry(), params, c_t, eta,
                                fuel=n_exit)


def test_gated_model_step_requires_fuel():
    # the hidden default is gone: fuel is a required argument
    with pytest.raises(TypeError):
        theory.gated_model_step(_one_site_state(),
                                _witness_params(),
                                np.array([0.5]), 0.1)
    with pytest.raises(TypeError):
        theory.run_event(
            b=0.75, a=1.0, g=1.0, J=0.0, w=0.0, c=0.5, eta=0.1,
            kappa=1.0, th_J_loss=0.1, lam_wd=0.0,
            burst=dict(WITNESS_BURST))
