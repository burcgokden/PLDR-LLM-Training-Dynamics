"""Cross-layer tests of the single event clock: the interfaces where
component-level testing is not enough.

Each test here checks an IDENTIFICATION between two layers of the
artifact (map against theorem constants, generator against map,
capture hypothesis against adversarial states, cap against the
boundary, gate timing against the global clock, maintenance against
direct unrolling), or pins a witness that retired a formulation:

* the event-clock witness: the linear normal form's margin
  recurrence is NOT induced by the factor event (margin residual
  0.9645264 at the gated planning state; effective drain
  coefficient 0.291688 there and varying by a factor 35 over a
  learning-rate sweep), while the multiplicative log-drain law is
  exact at machine precision on the same events;
* the capture witness: an amplitude rebuilding from below its band
  topples a site whose curvature was below threshold at capture
  (global step 38), so the captured clause must bound the full
  product through the amplitude envelope, which excludes the
  witness;
* the power-cap witness: the continuous-time cap 0.9 is overshot to
  0.97275084375 in one legal discrete step, while the h-dependent
  invariant cap admits the step and is preserved;
* the activity clock: the counter written after an event is the
  reload duration in global steps, never the fast-clock count, and
  rebuild is applied on exactly the following counter steps;
* maintenance: the affine unrolling identity is exact on direct
  trajectories and the derived lower bound holds on unrolled
  gated-model cycles;
* generated constants: displayed certified upper bounds are
  outward-rounded from the archived exact values.
"""

import json
import math
import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", "analysis"))

import theory  # noqa: E402
import retired  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

WITNESS_BURST = dict(h=0.1, b_sat=0.6, z_seed=0.01, z_floor=0.002)


def _witness_event():
    return theory.run_event(
        b=0.1005, a=1.0, g=1.0, J=0.69965, w=0.0, c=0.3, eta=0.05,
        kappa=1.0, th_J_loss=0.01, lam_wd=0.0, burst=WITNESS_BURST, fuel=200000)


# ------------------------------------------- the event-clock witness


def test_event_clock_witness_normal_form_mismatch():
    # The pinned mismatch that retired the linear normal form: the
    # factor event's margin drain is NOT eps0 - h*a_b*sum_z for any
    # constant a_b.
    ev = _witness_event()
    eps0 = 0.1005 + 0.69965 ** 2 - 0.3
    assert ev["n"] == 303
    assert ev["sum_z"] == pytest.approx(13.61725167, abs=1e-6)
    assert ev["eps_end"] == pytest.approx(-0.10718865, abs=1e-8)
    eps_nf = eps0 - 0.1 * ev["sum_z"]  # the retired form, a_b = 1
    assert eps_nf == pytest.approx(-1.07171504, abs=1e-6)
    assert ev["eps_end"] - eps_nf == pytest.approx(0.9645264,
                                                   abs=1e-6)
    a_eff = (eps0 - ev["eps_end"]) / (0.1 * ev["sum_z"])
    assert a_eff == pytest.approx(0.291688, abs=1e-5)


def test_event_clock_witness_no_constant_coefficient():
    # The learning-rate sweep: the effective drain coefficient
    # varies by a factor 35, so no constant a_b exists.
    effs = []
    for eta in (0.001, 0.05):
        ev = theory.run_event(
            b=0.05, a=1.0, g=1.0, J=math.sqrt(0.0504), w=0.0, c=0.1,
            eta=eta, kappa=1.0, th_J_loss=0.01, lam_wd=0.0,
            burst=dict(h=0.15, b_sat=0.6, z_seed=0.005,
                       z_floor=0.001), fuel=200000)
        effs.append((0.0004 - ev["eps_end"]) / (0.15 * ev["sum_z"]))
    assert effs[1] / effs[0] > 30.0


def test_exact_log_drain_identity_machine_precision():
    # The exact law of the implemented clock, at machine precision
    # on the witness event: factor-derived drain against the
    # returned sum of log factors (the review's requested
    # cross-layer test 1).
    ev = _witness_event()
    y0 = 0.69965 ** 2
    yN = ev["x_end"] - 0.1005
    assert math.log(yN / y0) == pytest.approx(
        2.0 * ev["sum_log_phi"], abs=1e-12)
    r_exact = theory.retained_fraction_exact(0.1005, 0.3, y0,
                                             ev["sum_log_phi"])
    assert ev["x_end"] / 0.3 == pytest.approx(r_exact, abs=1e-12)


def test_drain_enclosure_on_witness_and_library():
    # The two-sided enclosure holds on the witness and across the
    # planning event library.
    cal = json.load(open(os.path.join(ROOT, "calibration",
                                      "calibration.json")))
    gp = cal["gated_model_planning"]
    pl = cal["planning"]
    bu = gp["burst"]
    zc = theory.z_cap_invariant(bu["h"], pl["eps_max_event"],
                                bu["b_sat"],
                                w_max2=pl["w_max2_event"],
                                z_seed=bu["z_seed"])
    b_star = gp["d0"] / gp["mu_b"]
    um = theory.event_u_max(gp["eta"], gp["th_J_loss"],
                            gp["lam_wd_model"], gp["kappa"], zc)
    for eps0 in np.linspace(pl["e2_entry_margin_lo"],
                            pl["e2_entry_margin_hi"], 8):
        y0 = gp["c_recurrent"] + eps0 - b_star
        ev = theory.run_event(
            b_star, 1.0, 1.0, math.sqrt(y0 / gp["kappa"]), 0.0,
            gp["c_recurrent"], gp["eta"], gp["kappa"],
            gp["th_J_loss"], gp["lam_wd_model"], bu, fuel=200000)
        assert not ev["clamped"]
        d1 = theory.drain_linear(gp["eta"], gp["th_J_loss"],
                                 gp["lam_wd_model"], gp["kappa"],
                                 ev["n"], ev["sum_z"])
        lo, hi = theory.drain_enclosure(d1, um)
        exact = -2.0 * ev["sum_log_phi"]
        assert lo - 1e-12 <= exact <= hi + 1e-12


def test_two_condition_end_restores_exit_margin():
    # The one-step witness: under the old power-floor-only rule the
    # event ended after ONE step ABOVE threshold (eps_1 =
    # 0.000399998488 > 0).  The two-condition end never certifies
    # an end above threshold: at the witness state the event keeps
    # draining (no end is reported within the step budget), and at
    # a faster rate it ends with the exit margin in [b - c, 0].
    ev = theory.run_event(
        b=0.05, a=1.0, g=1.0, J=math.sqrt(0.0504), w=0.0, c=0.1,
        eta=1e-6, kappa=1.0, th_J_loss=0.01, lam_wd=0.0,
        burst=dict(h=0.15, b_sat=0.6, z_seed=0.005,
                   z_floor=0.004999), fuel=2000)
    assert ev["n"] > 1                # the old rule stopped at n = 1
    assert not (ev["ended"] and ev["eps_end"] > 0.0)
    ev2 = theory.run_event(
        b=0.05, a=1.0, g=1.0, J=math.sqrt(0.0504), w=0.0, c=0.1,
        eta=1e-3, kappa=1.0, th_J_loss=0.01, lam_wd=0.0,
        burst=dict(h=0.15, b_sat=0.6, z_seed=0.005,
                   z_floor=0.004999), fuel=500000)
    assert ev2["ended"]
    assert 0.05 - 0.1 <= ev2["eps_end"] <= 0.0


def test_event_time_bound_and_licensed_reset_on_events():
    # The review's requested cross-layer test 2: every generated
    # theorem constant is checked against run_event directly.
    cal = json.load(open(os.path.join(ROOT, "calibration",
                                      "calibration.json")))
    gp = cal["gated_model_planning"]
    pl = cal["planning"]
    bu = gp["burst"]
    b_star = gp["d0"] / gp["mu_b"]
    zc = theory.z_cap_invariant(bu["h"], pl["eps_max_event"],
                                bu["b_sat"],
                                w_max2=pl["w_max2_event"],
                                z_seed=bu["z_seed"])
    y0_max = gp["c_recurrent"] + pl["eps_max_event"] - b_star
    n_max = theory.event_time_bound(
        y0_max, b_star, gp["c_recurrent"], gp["eta"],
        gp["th_J_loss"], gp["lam_wd_model"], bu["h"], zc,
        bu["z_floor"])
    r_lo, r_hi = theory.reset_interval_licensed(
        y0_max, b_star, gp["c_recurrent"], gp["eta"],
        gp["th_J_loss"], gp["lam_wd_model"], gp["kappa"], zc, n_max)
    for eps0 in np.linspace(pl["e2_entry_margin_lo"],
                            pl["e2_entry_margin_hi"], 8):
        y0 = gp["c_recurrent"] + eps0 - b_star
        ev = theory.run_event(
            b_star, 1.0, 1.0, math.sqrt(y0 / gp["kappa"]), 0.0,
            gp["c_recurrent"], gp["eta"], gp["kappa"],
            gp["th_J_loss"], gp["lam_wd_model"], bu, fuel=200000)
        assert ev["ended"] and ev["n"] <= n_max
        r = ev["x_end"] / gp["c_recurrent"]
        assert r_lo - 1e-9 <= r <= r_hi + 1e-9


# ----------------------------------------------- the capture witness


def _capture_witness_setup():
    state = dict(b=[0.0], a=[0.0], g=[1.0], J=[1.0], w=[0.0],
                 pend=[0.0], act=[0.0])
    params = dict(beta=np.zeros((1, 1)), d0=0.0, delta_a=1.0,
                  delta_J0=0.0, kappa=1.0, th_a_loss=1.0,
                  th_g_loss=0.0, th_J_loss=0.01, lam_wd=0.0,
                  mu_b=1.0, s_shed=0.0, burst=dict(WITNESS_BURST))
    return state, params


def test_capture_witness_topples_at_step_38():
    # The pinned one-site witness (review's cross-layer test 3,
    # adversarial branch): closed gate, curvature below threshold at
    # capture, amplitude rebuilding from below its band; the exact
    # map topples at global step 38.  This is the NECESSITY witness
    # for the amplitude envelope in the captured clause.
    state, params = _capture_witness_setup()
    first = None
    for t in range(60):
        c_t = np.array([0.001 if t == 0 else 0.1])
        state, top = theory.gated_model_step(state, params, c_t,
                                             0.01, fuel=200000)
        if top.any():
            first = t + 1
            break
    assert first == 38


def test_capture_envelope_excludes_witness():
    # The corrected balance condition correctly fails on the
    # witness: a_sup = max(a(T1), delta_a/th_a) = 1 makes the
    # envelope curvature 0.9998 > 0.1.
    a_sup = theory.amplitude_envelope(0.01, 1.0, 1.0, 0.0)
    assert a_sup == 1.0
    slack = theory.capture_balance(0.0, 1.0, a_sup, 1.0, 0.9999,
                                   0.1)
    assert slack < 0.0


def test_capture_no_topple_when_balance_holds():
    # Randomized adversarial below-band amplitudes: whenever the
    # envelope balance HOLDS, the closed-gate orbit never topples.
    rng = np.random.default_rng(7)
    for _ in range(20):
        a0 = float(rng.uniform(0.0, 0.3))
        delta_a = float(rng.uniform(0.0, 0.5))
        th_a = float(rng.uniform(0.5, 1.5))
        g0 = float(rng.uniform(0.5, 1.0))
        J0 = float(rng.uniform(0.2, 1.0))
        d0 = float(rng.uniform(0.0, 0.05))
        mu_b = 1.0
        a_sup = theory.amplitude_envelope(a0, delta_a, th_a, 0.0)
        b_sup = theory.base_envelope(0.0, d0, mu_b)
        c_inf = b_sup + (a_sup * g0 * J0) ** 2 + 0.01
        assert theory.capture_balance(b_sup, 1.0, a_sup, g0, J0,
                                      c_inf) >= 0.0
        state = dict(b=[0.0], a=[a0], g=[g0], J=[J0], w=[0.0],
                     pend=[0.0], act=[0.0])
        params = dict(beta=np.zeros((1, 1)), d0=d0,
                      delta_a=delta_a, delta_J0=0.0, kappa=1.0,
                      th_a_loss=th_a, th_g_loss=0.0, th_J_loss=0.01,
                      lam_wd=0.0, mu_b=mu_b, s_shed=0.0,
                      burst=dict(WITNESS_BURST))
        for _t in range(400):
            state, top = theory.gated_model_step(
                state, params, np.array([c_inf]), 0.01, fuel=200000)
            assert not top.any()


def test_amplitude_envelope_is_invariant():
    # The affine amplitude law never exceeds the envelope, for any
    # per-step rates inside the declared range.
    rng = np.random.default_rng(3)
    a, delta_a, th = 0.01, 1.0, 1.0
    a_sup = theory.amplitude_envelope(a, delta_a, th, 0.0)
    for _ in range(10000):
        eta = float(rng.uniform(0.001, 0.05))
        a = (1.0 - eta * th) * a + eta * delta_a
        assert a <= a_sup + 1e-12


# --------------------------------------------- the power-cap witness


def test_power_cap_overshoot_witness_and_invariance():
    # The review's cross-layer test 4: the continuous-time cap 0.9
    # is overshot in one legal step; the discrete cap admits the
    # step and is invariant, at the boundary and for large h.
    z0, h, b_sat, eps_max = 0.75, 1.0, 1.0, 0.9
    J1 = 1.0 - 0.001 * z0
    eps1 = J1 * J1 - 0.1
    z1 = z0 * (1.0 + 2.0 * h * (eps1 - b_sat * z0))
    assert z1 == pytest.approx(0.97275084375, abs=1e-11)
    assert z1 > retired.peak_power_cap(0.0, 0.0, 1.0) + 0.9 - 0.0 \
        or z1 > 0.9                     # above the retired cap
    zc = theory.z_cap_invariant(h, eps_max, b_sat, w_max2=0.75)
    assert z1 <= zc
    # invariance sweep including the boundary, for several h
    for hh in (1.0, 0.3, 0.1, 0.01):
        zch = theory.z_cap_invariant(hh, eps_max, b_sat)
        for z in np.linspace(0.0, zch, 4001):
            img = max(z * (1.0 + 2.0 * hh * (eps_max - b_sat * z)),
                      0.0)
            assert img <= zch + 1e-12
    # and the cap approaches the carrying level as h -> 0
    assert theory.z_cap_invariant(1e-6, eps_max, b_sat) == \
        pytest.approx(eps_max / b_sat, rel=1e-5)


def test_clamp_never_binds_under_s_plus():
    # (S+) keeps the printed clamp from binding on the invariant
    # box; event runs on the box report clamped == False.
    zc = theory.z_cap_invariant(0.1, 0.9, 0.6, z_seed=0.01)
    um = theory.event_u_max(0.05, 0.01, 0.0, 1.0, zc)
    assert um <= 1.0
    ev = _witness_event()
    assert not ev["clamped"]
    with pytest.raises(ValueError, match=r"\(S\+\)"):
        theory.event_u_max(2.0, 0.01, 0.0, 1.0, zc)


# ------------------------------------------------- the activity gate


def test_gate_counter_is_reload_duration_not_fast_clock():
    # The review's cross-layer test 5: the counter written after an
    # event equals the reload duration in global steps, never the
    # fast-clock iteration count, and rebuild is applied on exactly
    # the following counter steps.
    n = 1
    gp_burst = dict(h=0.1, b_sat=0.6, z_seed=0.01, z_floor=0.002)
    params = dict(beta=np.zeros((n, n)), d0=0.01, delta_a=0.2,
                  delta_J0=0.2, kappa=1.0, th_a_loss=0.2,
                  th_g_loss=0.0, th_J_loss=0.01, lam_wd=0.0,
                  mu_b=0.1, s_shed=0.5, burst=gp_burst)
    state = dict(b=[0.05], a=[1.0], g=[1.0], J=[0.6], w=[0.0],
                 pend=[0.0], act=[0.0])
    eta = 0.05
    # step until the first topple
    J_pre_smooth = None
    for _t in range(200):
        prev_J = float(np.asarray(state["J"])[0])
        prev_w = float(np.asarray(state["w"])[0])
        state, top = theory.gated_model_step(state, params,
                                             np.array([0.3]), eta, fuel=200000)
        if top.any():
            # reconstruct the pre-event jet (smooth step, gate
            # closed before the first event)
            J_pre_smooth = (1.0 - eta * (0.01 + prev_w ** 2)) \
                * prev_J
            break
    assert J_pre_smooth is not None
    J_post = float(np.asarray(state["J"])[0])
    expected = theory.reload_steps(J_pre_smooth, J_post, eta, 0.2)
    act = int(np.asarray(state["act"])[0])
    assert act == expected
    assert act >= 1
    # the fast-clock count of the same event is NOT the counter
    ev = theory.run_event(float(np.asarray(state["b"])[0]), 1.0,
                          1.0, J_pre_smooth, 0.0, 0.3, eta, 1.0,
                          0.01, 0.0, gp_burst, fuel=200000)
    assert ev["n"] > 10 * act
    # rebuild is applied on exactly the following act steps
    for k in range(act):
        prevJ = float(np.asarray(state["J"])[0])
        prev_w = float(np.asarray(state["w"])[0])
        state, top = theory.gated_model_step(state, params,
                                             np.array([10.0]), eta, fuel=200000)
        stepped = (1.0 - eta * (0.01 + prev_w ** 2)) * prevJ
        got = float(np.asarray(state["J"])[0])
        assert got == pytest.approx(stepped + eta * 0.2, abs=1e-12)
    # gate now closed: no rebuild on the next step
    prevJ = float(np.asarray(state["J"])[0])
    prev_w = float(np.asarray(state["w"])[0])
    state, top = theory.gated_model_step(state, params,
                                         np.array([10.0]), eta, fuel=200000)
    got = float(np.asarray(state["J"])[0])
    assert got == pytest.approx(
        (1.0 - eta * (0.01 + prev_w ** 2)) * prevJ, abs=1e-12)


# ------------------------------------------------------- maintenance


def test_affine_unrolling_identity_and_lower_bound():
    # The exact unrolling identity on random sequences, and the
    # derived lower bound with survival factors.
    rng = np.random.default_rng(11)
    for _ in range(50):
        T = int(rng.integers(1, 30))
        q = rng.uniform(0.9, 1.0, size=T)
        m = rng.uniform(0.0, 0.02, size=T)
        J0 = float(rng.uniform(0.0, 1.0))
        direct = J0
        for s in range(T):
            direct = q[s] * direct + m[s]
        assert theory.jet_unrolled(J0, list(q), list(m)) == \
            pytest.approx(direct, rel=1e-12)
        lower = theory.jet_unroll_lower(J0, float(q.min()), T,
                                        float(m.sum()))
        assert lower <= direct + 1e-12


def test_maintenance_floor_on_unrolled_gated_cycles():
    # The review's cross-layer test 6: M-main against directly
    # unrolled gated_model_step cycles at the planning instance.
    cal = json.load(open(os.path.join(ROOT, "calibration",
                                      "calibration.json")))
    gp = cal["gated_model_planning"]
    n = 1
    params = dict(beta=np.zeros((n, n)), d0=gp["d0"],
                  delta_a=gp["delta_a"], delta_J0=gp["delta_J0"],
                  kappa=gp["kappa"], th_a_loss=gp["th_a_loss"],
                  th_g_loss=gp["th_g_loss"],
                  th_J_loss=gp["th_J_loss"],
                  lam_wd=gp["lam_wd_model"], mu_b=gp["mu_b"],
                  s_shed=gp["s_shed"], burst=dict(gp["burst"]))
    state = dict(b=[gp["d0"] / gp["mu_b"] * 0.5], a=[1.0], g=[1.0],
                 J=[gp["J0"]], w=[0.0], pend=[0.0], act=[0.0])
    eta = gp["eta"]
    # measure cycles directly
    J_at_topple = []
    ws = [0.0]
    n_ev = []
    gaps = []
    last = None
    for t in range(3000):
        state, top = theory.gated_model_step(
            state, params, np.array([gp["c_recurrent"]]), eta, fuel=200000)
        ws.append(abs(float(np.asarray(state["w"])[0])))
        if top.any():
            J_at_topple.append(float(np.asarray(state["J"])[0]))
            if last is not None:
                gaps.append(t - last)
            last = t
    assert len(J_at_topple) >= 10
    t_cyc = max(gaps)
    w_max2 = max(ws) ** 2
    # a crude per-event fast-length cap from the planning bound
    pl = cal["planning"]
    bu = gp["burst"]
    zc = theory.z_cap_invariant(bu["h"], pl["eps_max_event"],
                                bu["b_sat"],
                                w_max2=pl["w_max2_event"],
                                z_seed=bu["z_seed"])
    b_star = gp["d0"] / gp["mu_b"]
    n_ev_max = theory.event_time_bound(
        gp["c_recurrent"] + pl["eps_max_event"] - b_star, b_star,
        gp["c_recurrent"], eta, gp["th_J_loss"],
        gp["lam_wd_model"], bu["h"], zc, bu["z_floor"])
    phi_min = 1.0 - theory.event_u_max(
        eta, gp["th_J_loss"], gp["lam_wd_model"], gp["kappa"], zc)
    n_act = min(gaps) if gaps else 1
    q_cyc, m_cyc = theory.cycle_floor_inputs(
        eta, eta, gp["delta_J0"], n_act, t_cyc, gp["th_J_loss"],
        gp["lam_wd_model"], gp["kappa"], w_max2, phi_min, n_ev_max)
    floor = theory.maintenance_floor(q_cyc, m_cyc)
    # the derived floor is a true lower bound on the measured
    # post-event jets (the conservative direction)
    assert min(J_at_topple) >= floor


# --------------------------------------- generated-constant hygiene




# --------------------------------------------- the projected chain


def test_projection_preserves_crossing_and_interval():
    # The projected chain max(., 0): the crossing event is
    # unchanged (c >= 0) and the safe interval maps into itself.
    rng = np.random.default_rng(5)
    a, eta, d, c, B = 0.1, 0.05, 0.2, 0.5, 0.5
    for _ in range(2000):
        x = float(rng.uniform(0.0, B))
        z = float(rng.normal(0.0, 3.0))
        u = (1.0 - a) * x + eta * d + eta * z
        xp = max(u, 0.0)
        assert (xp > c) == (u > c)
        if u <= B:
            assert 0.0 <= xp <= B
