"""Settlement and saturation regressions: the terminal carrier
settlement (E-settle), the clipped curvature law, the clipped
recurrent envelopes, the uniform fuel, and the pure-decay product.

Each pinned witness in this file began life as a refutation of a
prior theorem interface and is kept as a permanent negative
control guarding the repaired one:

  1. the HOT-EXIT witness: a legal one-site entry whose event ends
     through the generalized (jet-exhaustion) exit with a residual
     margin, whose PRE-settlement carrier leaves the declared box
     by a factor of 7.25, and whose SETTLED successor lies inside
     the box with the settled power ledgered;
  2. the BINDING-CLIP witness: a legal saturated jet rebuild on
     which the affine curvature expansion is wrong and the exact
     clipped law with its defect term is right;
  3. the CLIPPED-ENVELOPE witness: an early oversized injection,
     clipped away and then decayed, on which the affine unrolling
     overstates the orbit and the (N-sat) certificate refuses;
  4. the UNEQUAL-SHARE pure-decay witness: the share-sum premise
     controls the first-order coefficient only, never the exact
     finite-step common-power factor;
  5. the OVERLOAD-TARGET monotonicity: the hardest overloaded
     entry is b -> c+ (smallest exit target), never b_max.
"""

import math

import numpy as np
import pytest

from analysis import theory


# ---------------------------------------------------- the witnesses

# the hot-exit witness constants (one site)
HOT_BURST = dict(h=0.1, b_sat=1.0, z_seed=0.01, z_floor=0.001,
                 y_floor=0.01, tau_z=0.02, w_kick_max=0.1)
HOT_W_MAX = 0.1


def _hot_witness_params():
    return dict(
        beta=np.zeros((1, 1)), d0=0.1, delta_a=0.0, delta_J0=0.0,
        kappa=1.0, th_a_loss=0.0, th_g_loss=0.0, th_J_loss=0.1,
        lam_wd=0.0, mu_b=0.1, s_shed=0.5, burst=HOT_BURST)


def _hot_witness_state():
    return dict(b=[1.0], a=[1.0], g=[1.0], J=[1.0], w=[0.0],
                pend=[0.0], act=[0.0])


# 1 ------------------------------------------- the hot-exit witness


def test_hot_exit_witness_presettlement_orbit_pinned():
    # layer one: the pinned pre-settlement orbit.  The event enters
    # at the post-smooth-step state (b = 1, J = 0.99), ends after
    # 50 internal steps through the JET-EXHAUSTION conjunct with a
    # residual positive margin (a generalized hot exit), and its
    # pre-settlement carrier is 7.25 times the declared box
    events = []
    new, top = theory.gated_model_step(
        _hot_witness_state(), _hot_witness_params(),
        np.array([0.5]), 0.1, events_out=events, fuel=200000)
    assert bool(top[0])
    (ev,) = events
    assert ev["ended"] and ev["status"] == "ended"
    assert ev["exit_type"] == "hot"
    assert ev["n"] == 50
    assert ev["J"] == pytest.approx(0.07830369961002033, abs=1e-15)
    assert ev["eps_end"] == pytest.approx(0.5061314693726162,
                                          abs=1e-12)
    # the exit fired through jet exhaustion, not the cold conjunct
    assert ev["eps_end"] > 0.0
    y_end = 1.0 * (1.0 * 1.0 * ev["J"]) ** 2
    assert y_end <= HOT_BURST["y_floor"]
    # the pre-settlement exit carrier leaves the declared box
    assert ev["w_exit"] == pytest.approx(0.7250148061672582,
                                         abs=1e-12)
    assert abs(ev["w_exit"]) > 7.0 * HOT_W_MAX
    z_exit = ev["w_exit"] ** 2
    assert z_exit < ev["eps_end"] / HOT_BURST["b_sat"] \
        + HOT_BURST["tau_z"]          # the generalized gate held
    assert z_exit > 500.0 * HOT_BURST["z_floor"]


def test_hot_exit_witness_settled_successor_in_box():
    # layer two: the SETTLED successor lies inside the declared box
    events = []
    new, top = theory.gated_model_step(
        _hot_witness_state(), _hot_witness_params(),
        np.array([0.5]), 0.1, events_out=events, fuel=200000)
    (ev,) = events
    assert ev["w_end"] == pytest.approx(
        math.sqrt(HOT_BURST["z_floor"]), abs=1e-15)
    assert abs(ev["w_end"]) <= HOT_W_MAX
    box = dict(b_max=1.0, a_max=1.0, g_max=1.0, j_max=1.0,
               w_max=HOT_W_MAX, pend_max=1.0)
    assert theory.state_in_box(new, box) == []
    # the shed ledger is unchanged by settlement (stress-based)
    assert float(new["pend"][0]) == pytest.approx(
        0.4869842653136919, abs=1e-12)


def test_hot_exit_witness_settled_power_pinned():
    # layer three: the settled power is ledgered, never discarded
    events = []
    theory.gated_model_step(
        _hot_witness_state(), _hot_witness_params(),
        np.array([0.5]), 0.1, events_out=events, fuel=200000)
    (ev,) = events
    z_exit = ev["w_exit"] ** 2
    assert ev["d_set"] == pytest.approx(z_exit - HOT_BURST["z_floor"],
                                       abs=1e-15)
    assert ev["d_set"] == pytest.approx(0.524646469161747, abs=1e-12)
    # exact power accounting: exit power = settled power + emission
    assert ev["d_set"] + ev["w_end"] ** 2 == pytest.approx(z_exit,
                                                          abs=1e-15)


def test_settlement_is_identity_on_cold_exits():
    # (E-settle) is the identity on every cold exit: d_set = 0 and
    # the emitted carrier IS the exit carrier (bit-equal), so the
    # gap-domain reduction stays bit-exact
    burst = dict(h=0.1, b_sat=1.0, z_seed=0.01, z_floor=0.001,
                 y_floor=0.01, tau_z=0.02, w_kick_max=0.1)
    for b, c, j0 in [(0.1, 0.5, 0.6), (0.0, 0.5, 1.0),
                     (0.3, 0.5, 0.4)]:
        ev = theory.run_event(
            b=b, a=1.0, g=1.0, J=j0, w=0.05, c=c, eta=0.05,
            kappa=1.0, th_J_loss=0.01, lam_wd=0.0, burst=burst, fuel=200000)
        assert ev["ended"]
        assert ev["exit_type"] == "cold"
        assert ev["d_set"] == 0.0
        assert ev["w_end"] == ev["w_exit"]
        assert ev["w_end"] ** 2 < burst["z_floor"]


def test_settled_successor_in_box_randomized():
    # property test: after ANY step from a box-sampled state under
    # legal constants, the successor satisfies the carrier clause
    # |w| <= max(w_start_max, sqrt(z_floor)) and the J/act/pend
    # clauses of the box self-map (every ended event settles)
    rng = np.random.default_rng(20260818)
    burst = dict(h=0.1, b_sat=1.0, z_seed=0.01, z_floor=0.001,
                 y_floor=0.01, tau_z=0.05, w_kick_max=0.1)
    n_sites = 5
    params = dict(
        beta=0.05 * np.ones((n_sites, n_sites)) / n_sites,
        d0=0.02, delta_a=0.0, delta_J0=0.2, kappa=1.0,
        th_a_loss=0.0, th_g_loss=0.0, th_J_loss=0.1, lam_wd=0.0,
        mu_b=0.1, s_shed=0.5, burst=burst)
    c_t = np.full(n_sites, 0.5)
    w_cap = 0.1
    for _ in range(20):
        state = dict(
            b=rng.uniform(0.0, 0.9, n_sites),
            a=rng.uniform(0.2, 1.0, n_sites),
            g=rng.uniform(0.2, 1.0, n_sites),
            J=rng.uniform(0.0, 1.0, n_sites),
            w=rng.uniform(-w_cap, w_cap, n_sites),
            pend=rng.uniform(0.0, 0.2, n_sites),
            act=rng.integers(0, 3, n_sites).astype(float))
        for _ in range(3):
            state, _ = theory.gated_model_step(
                state, params, c_t, 0.05, rng=rng, fuel=200000)
            w_bound = max(w_cap, math.sqrt(burst["z_floor"]))
            assert float(np.max(np.abs(state["w"]))) <= w_bound
            assert float(np.max(state["J"])) <= 1.0 + 1e-12
            assert float(np.min(state["J"])) >= -1e-12
            assert float(np.min(state["pend"])) >= 0.0
            assert float(np.min(state["act"])) >= 0.0


# 2 --------------------------------------- the binding-clip witness


def test_exact_curvature_law_on_a_binding_clip():
    # the saturated rebuild: j1_unclipped = 0.99 + 0.02 = 1.01,
    # chi = 0.01, j1 = 1; the actual curvature increment is ZERO
    # while the affine expansion claims a positive move; the exact
    # clipped law closes the accounting through the defect
    first, exact, chi, defect = theory.curvature_increment_exact_gated(
        b=0.0, kappa=1.0, a=1.0, g=1.0, J=1.0, w=0.0, eta=0.1,
        d0=0.0, delta=0.0, delta_J0=0.2, A=1.0, th_a_loss=0.0,
        th_g_loss=0.0, th_J_loss=0.1, lam_wd=0.0, mu_b=0.0)
    assert chi == pytest.approx(0.01, abs=1e-15)
    assert exact == pytest.approx(0.0, abs=1e-15)
    # the unclipped affine increment of the curvature is 0.0201
    # (u^2 moves from 1 to 1.01^2); the defect removes exactly it
    assert defect == pytest.approx(0.0201, abs=1e-15)
    j1_unclipped = 0.99 + 0.02
    exact_unclipped = (theory.curvature(0.0, 1.0, 1.0, 1.0,
                                        j1_unclipped)
                       - theory.curvature(0.0, 1.0, 1.0, 1.0, 1.0))
    assert exact == pytest.approx(exact_unclipped - defect,
                                  abs=1e-15)


def test_gated_law_reduces_to_affine_when_clip_inactive():
    # chi = 0 off the clip and the exact increment matches the
    # closed-gate law extended by the rebuild term
    first, exact, chi, defect = theory.curvature_increment_exact_gated(
        b=0.3, kappa=1.0, a=0.8, g=0.9, J=0.5, w=0.05, eta=0.01,
        d0=0.02, delta=0.01, delta_J0=0.2, A=1.0, th_a_loss=0.02,
        th_g_loss=0.01, th_J_loss=0.1, lam_wd=0.01, mu_b=0.1)
    assert chi == 0.0
    assert defect == 0.0
    # first order approximates the exact increment at small eta
    assert exact == pytest.approx(first, abs=5e-4)


def test_gated_law_matches_the_transition_jet_step():
    # the jet step inside the gated law IS the transition's step
    # (same clip), checked against gated_model_step on the
    # saturating configuration of the jet-rebuild test
    state = dict(b=[0.0], a=[1.0], g=[1.0], J=[0.99], w=[0.0],
                 pend=[0.0], act=[5.0])
    params = dict(
        beta=np.zeros((1, 1)), d0=0.0, delta_a=0.0, delta_J0=5.0,
        kappa=1.0, th_a_loss=0.0, th_g_loss=0.0, th_J_loss=0.1,
        lam_wd=0.0, mu_b=0.0, burst=HOT_BURST, s_shed=0.5)
    new, top = theory.gated_model_step(
        state, params, np.array([10.0]), 0.1, fuel=200000)
    assert not bool(top[0])
    j1_unclipped = (1.0 - 0.1 * 0.1) * 0.99 + 0.1 * 5.0
    chi = max(j1_unclipped - 1.0, 0.0)
    assert chi > 0.0
    assert float(new["J"][0]) == pytest.approx(j1_unclipped - chi,
                                               abs=1e-15)
    assert float(new["J"][0]) == 1.0


# 3 ----------------------------------- the clipped-envelope witness


def test_nsat_refuses_the_clipped_envelope_witness():
    # q = 0.1, J0 = 0, injections (50, 0): the clipped orbit is
    # (1, 0.1) while the affine unrolling claims 0.5; the (N-sat)
    # certificate refuses at the first prefix
    j = 0.0
    orbit = []
    for m in (50.0, 0.0):
        j = min(0.1 * j + m, 1.0)
        orbit.append(j)
    assert orbit == [1.0, pytest.approx(0.1)]
    affine = theory.jet_unrolled(0.0, [0.1, 0.1], [50.0, 0.0])
    assert affine == pytest.approx(5.0)      # injection-first: 0.5
    # in the certificate normalization (injection after decay) the
    # claimed envelope is 0.5; both overstate the clipped endpoint
    assert 0.1 < 0.5 < affine
    with pytest.raises(ValueError, match="N-sat"):
        theory.maintenance_prefix_min(
            J_start=0.0, r_min=1.0, q_min=0.1, m_seq=[50.0, 0.0])
    with pytest.raises(ValueError, match="N-sat"):
        theory.maintenance_cycle_average(
            J_start=0.0, r_min=1.0, q_min=0.1, m_seq=[50.0, 0.0])


def test_clipped_injection_then_decay_against_composed_map():
    # the composed map: one oversized rebuild step (clip binds at
    # J = 1), then pure decay with the gate closed; the affine
    # formula overstates every later value and (N-sat) refuses,
    # while the clipped orbit is exactly the transition's
    state = dict(b=[0.0], a=[1.0], g=[1.0], J=[0.0], w=[0.0],
                 pend=[0.0], act=[1.0])
    params = dict(
        beta=np.zeros((1, 1)), d0=0.0, delta_a=0.0, delta_J0=50.0,
        kappa=1.0, th_a_loss=0.0, th_g_loss=0.0, th_J_loss=1.0,
        lam_wd=0.0, mu_b=0.0, burst=HOT_BURST, s_shed=0.5)
    c_t = np.array([10.0])                   # no topples
    js = []
    for _ in range(3):
        state, _ = theory.gated_model_step(state, params, c_t, 0.1, fuel=200000)
        js.append(float(state["J"][0]))
    assert js == [1.0, pytest.approx(0.9), pytest.approx(0.81)]
    affine = theory.jet_unrolled(0.0, [0.9, 0.9, 0.9],
                                 [5.0, 0.0, 0.0])
    assert affine > js[-1]                   # the overstatement
    with pytest.raises(ValueError, match="N-sat"):
        theory.maintenance_prefix_min(
            J_start=0.0, r_min=1.0, q_min=0.9,
            m_seq=[5.0, 0.0, 0.0])


def test_nsat_passes_at_the_planning_scale():
    # the theorem-tier planning constants satisfy (N-sat): the
    # certificates keep certifying there (regression against
    # over-tight guards)
    val = theory.maintenance_prefix_min(
        J_start=0.6, r_min=0.5, q_min=0.99, m_seq=[0.01] * 20)
    assert val > 0.0
    avg = theory.maintenance_cycle_average(
        J_start=0.6, r_min=0.5, q_min=0.99, m_seq=[0.01] * 20)
    assert avg >= val


def test_clipped_floor_comparison_certificate():
    # the (M-end) clipped floor: valid whenever J_floor <= 1, with
    # no envelope hypothesis; the affine minorant never exceeds
    # the clip, so the clipped orbit stays above it
    floor = theory.maintenance_floor_clipped(0.9, 0.05)
    assert floor == pytest.approx(0.5)
    rng = np.random.default_rng(7)
    j = 0.6
    jhat = 0.6
    for _ in range(200):
        m = float(rng.uniform(0.05, 60.0))   # oversized allowed
        j = min(0.9 * j + m, 1.0)
        jhat = 0.9 * jhat + 0.05
        assert j >= jhat - 1e-12
        assert jhat <= 1.0 + 1e-12
    assert j >= floor - 1e-9
    # the gate refuses when the floor exceeds the jet ceiling
    with pytest.raises(ValueError, match="J_floor"):
        theory.maintenance_floor_clipped(0.99, 0.02)


# 4 ------------------------------- the unequal-share pure decay


def test_pure_decay_exact_product_vs_common_power():
    # shares (lam_a, lam_g, lam_J) = (1, 0, 0), eta = 1/2, p = 3,
    # lam_wd = 1/3: the share sum holds, the exact squared factor
    # is the per-factor product 1/4, and the common-power claim
    # (5/6)^6 is a DIFFERENT number; only the first-order
    # coefficient is shared
    eta = 0.5
    lam_a, lam_g, lam_J = 1.0, 0.0, 0.0
    p, lam_wd = 3, 1.0 / 3.0
    assert lam_a + lam_g + lam_J == pytest.approx(p * lam_wd)
    exact_sq = ((1.0 - eta * lam_a) * (1.0 - eta * lam_g)
                * (1.0 - eta * lam_J)) ** 2
    common_sq = (1.0 - eta * lam_wd) ** (2 * p)
    assert exact_sq == pytest.approx(0.25)
    assert common_sq == pytest.approx((5.0 / 6.0) ** 6)
    assert abs(exact_sq - common_sq) > 0.08
    # the first-order coefficient IS controlled by the share sum
    # (pure_decay_rate): -2 p lam_wd = -(2 sum lam_c) = -2
    eta_small = 1e-6
    exact_small = ((1.0 - eta_small * lam_a)
                   * (1.0 - eta_small * lam_g)
                   * (1.0 - eta_small * lam_J)) ** 2
    rate = theory.pure_decay_rate(p, lam_wd, 1.0, 1.0)
    assert (exact_small - 1.0) / eta_small == pytest.approx(
        rate, abs=1e-4)
    # equal shares recover the common power exactly
    lam_eq = lam_wd
    eq_sq = ((1.0 - eta * lam_eq) ** 3) ** 2
    assert eq_sq == pytest.approx(common_sq)


# 5 ---------------------------- overload target and granted fuel


def test_overload_worst_case_is_the_shallow_limit():
    # the exit target grows with the overload, so the per-entry
    # bound DECREASES in b; the uniform bound (target tau_z, the
    # b -> c+ limit) dominates every entry, including b = c + 1e-6
    kw = dict(eta=0.01, th_J_loss=0.1, lam_wd=0.0, h=0.1,
              b_sat=1.0, y_floor=0.01, tau_z=0.05)
    y0_max = 1.0
    z_cap = theory.z_cap_invariant(0.1, 0.7, 1.0)
    unif = theory.event_time_bound_overloaded_uniform(
        y0_max, z_cap=z_cap, **kw)
    c = 0.5
    bounds = []
    for b in (c + 1e-6, c + 0.004, c + 0.5, 10.0):
        per = theory.event_time_bound_overloaded(
            y0_max, b, c, z_cap=z_cap, **kw)
        assert per <= unif
        bounds.append(per)
    assert bounds[0] >= bounds[-1]           # shallow is hardest
    assert bounds[0] > bounds[2]
    # a shallow overloaded entry ends within the uniform bound
    burst = dict(h=0.1, b_sat=1.0, z_seed=0.01, z_floor=0.001,
                 y_floor=0.01, tau_z=0.05)
    ev = theory.run_event(
        b=c + 1e-6, a=1.0, g=1.0, J=0.1, w=0.0, c=c, eta=0.01,
        kappa=1.0, th_J_loss=0.1, lam_wd=0.0, burst=burst, fuel=200000)
    assert ev["ended"]
    assert ev["n"] <= unif


def test_granted_fuel_strict_indexing_convention():
    # the resolver checks the exit at the top of each iteration:
    # an event ending at step index N needs granted fuel N + 1;
    # ``event_fuel_required*`` returns the granted value
    burst = dict(h=0.1, b_sat=1.0, z_seed=0.01, z_floor=0.001,
                 y_floor=0.01, tau_z=0.05)
    kw = dict(b=0.1, a=1.0, g=1.0, J=0.6, w=0.05, c=0.5, eta=0.05,
              kappa=1.0, th_J_loss=0.1, lam_wd=0.0, burst=burst)
    ev = theory.run_event(**kw, fuel=200000)
    assert ev["ended"]
    n = ev["n"]
    assert n > 0
    starved = theory.run_event(fuel=n, **kw)
    assert not starved["ended"]              # fuel N is not enough
    granted = theory.run_event(fuel=n + 1, **kw)
    assert granted["ended"] and granted["n"] == n
    # the whole-box granted fuel covers the witness with margin
    z_cap = theory.z_cap_invariant(0.1, 0.7, 1.0, z_seed=0.01)
    fuel = theory.event_fuel_required_uniform(
        1.0, 1.2, 0.5, 0.05, 0.1, 0.0, 0.1, 1.0, z_cap,
        0.001, 0.01, 0.05)
    assert fuel > n + 1
    assert theory.run_event(fuel=fuel, **kw)["ended"]
