"""Executable twins of the corrected-theory cores.

Each test mirrors a printed statement of the paper and, where one
exists, a machine-checked Lean declaration, so the Python
implementation of the theory cannot drift from either.
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


# ------------------------------------------------------ D2: drift


def test_sqrt_drift_exact_beats_naive_half_factor():
    # Lean: SqrtDrift.naive_half_factor_lt.  v=1, g=0, beta2=19/20:
    # exact drift 1 - sqrt(0.95) = 0.0253206... > 0.025 naive.
    exact = theory.sqrt_drift(1.0, 0.0, 0.95)
    naive = 0.5 * (1.0 - 0.95) * abs(0.0 - 1.0)
    assert exact == pytest.approx(1.0 - math.sqrt(0.95), rel=1e-12)
    assert exact > naive


def test_flip_boundary_limits():
    # d = 0: undamped 2(1+beta1); beta1 = 0: sigma + d = 2.
    assert theory.flip_boundary(0.9, 0.0) == pytest.approx(3.8)
    assert theory.flip_boundary(0.0, 0.3) == pytest.approx(1.7)
    # eta*lambda units at beta1 = 0.9, d = 0: 3.8/(1-0.9) = 38.
    assert theory.flip_boundary(0.9, 0.0) / 0.1 == pytest.approx(38.0)


def test_lyapunov_drain_identity():
    # Lean: ProductStability.lyap2_step.  Q(Ax) = Q(x) - |x|^2.
    A = theory.companion_matrix(sigma=1.0, beta1=0.9, d=1e-4)
    P = theory.lyap_solve_2x2(A)
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.normal(size=2)
        q = x @ P @ x
        qA = (A @ x) @ P @ (A @ x)
        assert qA == pytest.approx(q - x @ x, rel=1e-9, abs=1e-9)
    # P >= I
    evals = np.linalg.eigvalsh(P)
    assert evals.min() >= 1.0 - 1e-9


def test_c_grnt_below_frozen_and_monotone_in_drift():
    hi = theory.flip_boundary(0.9, 1e-4)
    g1 = theory.c_grnt(0.9, 1e-4, eps_A=0.01, C_P=1.0)
    g2 = theory.c_grnt(0.9, 1e-4, eps_A=0.05, C_P=1.0)
    assert 0.0 < g2 < g1 < hi
    assert theory.c_grnt(0.9, 1e-4, eps_A=0.0, C_P=1.0) == hi


# ------------------------------------------------------ D1: loading


def test_frozen_source_manufactures_nothing():
    # Lean: LoadingChainRule.frozen_source_no_loading /
    # FactoredState.driveRate.  With v = 0 and no base rate, the
    # stress does not load: the drive is exactly zero.
    x = 1.0
    a, g, J, w = 0.5, 1.0, 1.0, 0.0
    th_a, th_g, th_J, kappa = 1.0, 0.0, 0.0, 2.0
    # choose delta so v = 0 exactly at this state:
    delta = th_a * a
    v = theory.source_velocity(a, g, J, w, delta, th_a, th_g, th_J,
                               kappa)
    assert v == pytest.approx(0.0)
    assert theory.drive_law(0.0, kappa, a * g * J, v) == 0.0
    x1 = retired.loading_step(x, eta=0.1, d0=0.0, kappa=kappa,
                             u=a * g * J, v=v, p_i=2, lam_wd=0.0)
    assert x1 == pytest.approx(x)


def test_loading_step_matches_polynomial_identity():
    # Lean: LoadingChainRule.loading_step_eta at first order.
    lam0, kappa, u, v, eta = 0.3, 2.0, 0.4, 0.7, 1e-4
    before = lam0 + kappa * u ** 2
    after = lam0 + kappa * (u + eta * v) ** 2
    predicted = eta * (2.0 * kappa * u * v)
    assert after - before == pytest.approx(
        predicted + eta ** 2 * kappa * v ** 2, rel=1e-12)


# ------------------------------------------------------ D3: burst


def test_zero_seed_orbit_is_stationary():
    # The (E-seed) witness: z0 = 0, eps0 > 0 never exits.
    n, ztr, etr = retired.simulate_burst(0.0, 0.3, h=0.1, a=1.0,
                                        b=0.5, max_steps=200)
    assert n is None
    assert etr[-1] == pytest.approx(0.3)


def test_seeded_burst_exits_inside_reset_interval():
    # Exit dichotomy: signed shallow exit within the two-sided bound.
    z0, eps0, h, a, b = 1e-4, 0.3, 0.05, 1.0, 0.6
    q0 = a * z0 + eps0 ** 2
    assert h * math.sqrt(q0) <= 0.125 and \
        h * b * (q0 / a) <= 0.125  # step conditions
    n, ztr, etr = retired.simulate_burst(z0, eps0, h, a, b)
    assert n is not None
    lo, hi_ = retired.reset_interval(z0, eps0, h, a)
    assert lo <= etr[n] <= hi_
    # positivity and cap along the orbit
    zcap = q0 / a
    assert all(z > 0 for z in ztr[: n + 1])
    assert max(ztr[: n + 1]) <= zcap + 1e-12


def test_integrated_power_identity():
    # Lean: BurstNormalForm.eps_telescoping / integrated_power_eq.
    z0, eps0, h, a, b = 1e-3, 0.2, 0.05, 1.0, 0.6
    n, ztr, etr = retired.simulate_burst(z0, eps0, h, a, b)
    total = sum(ztr[:n])
    assert total == pytest.approx(
        retired.integrated_power(eps0, etr[n], h, a), rel=1e-9)


# ------------------------------------------------------ D4: children


def test_child_count_separation_witness():
    # Lean: ChildCount.critical_childMean_eq: ten receivers within
    # one tenth of a unit shed each: child mean 10, mass 1.
    F = [lambda y: 1.0 if y >= 0.1 else 0.0] * 10
    beta_row = [0.1] * 10
    assert theory.child_mean(F, beta_row, 1.0) == 10.0
    assert theory.transported_mass(F, beta_row, 1.0) == \
        pytest.approx(1.0)


def test_dose_response_bound_dominates_child_mean():
    rng = np.random.default_rng(1)
    fmax = 2.0
    for _ in range(20):
        beta_row = rng.uniform(0, 0.2, size=8)
        sigma = rng.uniform(0, 1)
        F = [
            (lambda y, s=s: min(fmax * y, 1.0) * s)
            for s in rng.uniform(0, 1, size=8)
        ]
        cm = theory.child_mean(F, beta_row, sigma)
        assert cm <= theory.dose_response_bound(
            fmax, sigma, beta_row.sum()) + 1e-12


# ------------------------------------------------------ D6: noise


def test_ar1_variance_conventions_and_ratio():
    # Lean: FirstPassageEnvelope.ar1_variance_fixed_point_{scaled,
    # unscaled} and the exact 1/eta^2 ratio.
    eta, mu, s2 = 0.1, 0.5, 1.0
    a = eta * mu
    v_scaled = theory.ar1_stationary_var(eta, mu, s2, scaled=True)
    v_unscaled = theory.ar1_stationary_var(eta, mu, s2, scaled=False)
    # fixed-point property
    assert (1 - a) ** 2 * v_scaled + eta ** 2 * s2 == \
        pytest.approx(v_scaled, rel=1e-12)
    assert (1 - a) ** 2 * v_unscaled + s2 == \
        pytest.approx(v_unscaled, rel=1e-12)
    assert v_unscaled == pytest.approx(v_scaled / eta ** 2, rel=1e-12)


def test_passage_band_orders_and_brackets_simulation():
    eta, mu, s2, B = 0.05, 0.5, 4.0, 16.0
    delta = 0.15
    T = 400
    lower, upper = theory.passage_band(delta, eta, mu, s2 / B, T)
    assert 0.0 <= lower <= upper <= 1.0
    # Monte Carlo of the AR(1) chain from the balance point
    rng = np.random.default_rng(2)
    a = eta * mu
    d = 0.3
    L = eta * d / a
    hits = 0
    trials = 400
    for _ in range(trials):
        x = L
        crossed = False
        for _t in range(T):
            x = (1 - a) * x + eta * d + eta * rng.normal(
                0.0, math.sqrt(s2 / B))
            if x > L + delta:
                crossed = True
                break
        hits += crossed
    p_hat = hits / trials
    assert lower - 0.05 <= p_hat <= upper + 0.05


# ------------------------------------------------------ D8: maps


def test_monomial_transform_crackling_invariance():
    tau, gamma, alphaD = 1.5, 2.0, 2.0
    for b in (1.0, 2.0, 1.5):
        tau_o, gamma_o = theory.monomial_transform(tau, gamma, b)
        assert gamma_o * (tau_o - 1.0) == pytest.approx(
            gamma * (tau - 1.0))
    assert theory.monomial_transform(tau, gamma, 2.0) == (
        pytest.approx(1.25), pytest.approx(4.0))
    with pytest.raises(ValueError):
        theory.monomial_transform(tau, gamma, -1.0)


# ------------------------------------------- closed model invariance


def _params(n, rng):
    beta = rng.uniform(0, 1, size=(n, n))
    np.fill_diagonal(beta, 0.0)
    beta = 0.8 * beta / beta.sum(axis=1, keepdims=True)  # row sums .8
    return dict(
        beta=beta,
        d0=rng.uniform(0.01, 0.1, size=n),
        kappa=rng.uniform(0.5, 2.0, size=n),
        delta=rng.uniform(0.0, 0.05, size=n),
        th_a=np.full(n, 0.3),
        th_g=np.full(n, 0.1),
        th_J=np.full(n, 0.1),
        mu_b=0.4,
        mu_x=0.4,
        r=0.7,
        rho_w=0.3,
    )


def _state(n, rng, wmax=0.5):
    return dict(
        x=rng.uniform(0, 1, size=n),
        b=rng.uniform(0, 0.3, size=n),
        a=rng.uniform(0, 0.1, size=n),
        g=rng.uniform(0.5, 1.0, size=n),
        J=rng.uniform(0.5, 1.0, size=n),
        w=rng.uniform(-wmax, wmax, size=n),
        pend=rng.uniform(0, 0.5, size=n),
    )


def test_closed_model_stress_confinement_and_ledger():
    # Theorem wellposed (ii)-(iv): x in [0, c] after every step;
    # sheds nonnegative; ledger exact at topples.
    rng = np.random.default_rng(3)
    n = 6
    params = _params(n, rng)
    c_t = np.full(n, 1.0)
    eta = 0.05
    state = _state(n, rng)
    for _ in range(200):
        new, topple = retired.closed_model_step(state, params, c_t,
                                               eta)
        assert (new["x"] >= -1e-12).all()
        assert (new["x"] <= c_t + 1e-12).all()
        assert (new["pend"] >= -1e-12).all()
        if topple.any():
            # ledger: loaded = r*c + pend, and pend splits by
            # row-sum + dissipation = 1 identically
            loaded = np.where(topple,
                              new["pend"] + params["r"] * c_t, 0.0)
            assert (loaded[topple] > c_t[topple]).all()
            # shed exceeds (1 - r) * c at every topple
            assert (new["pend"][topple] >
                    (1 - params["r"]) * c_t[topple] - 1e-12).all()
        state = new


def test_closed_model_factor_and_carrier_invariance():
    # Theorem wellposed (v)-(vii) on the factored state: amplitude
    # confined to [0, amax] under delta <= th_a * amax; chain factor
    # and jet monotone nonincreasing in [0, 1]; carrier confined.
    rng = np.random.default_rng(4)
    n = 5
    params = _params(n, rng)
    amax = params["delta"].max() / params["th_a"].min() + 1e-9
    amax = max(amax, 0.2)
    wmax = 0.6
    eta = 0.05
    # step conditions (S)/(Q+)
    assert (params["delta"] <= params["th_a"] * amax + 1e-12).all()
    assert eta * (params["th_J"].max()
                  + params["kappa"].max() * wmax ** 2) <= 1.0
    bmax = max(0.3, params["d0"].max() / params["mu_b"])
    assert eta * (bmax + params["kappa"].max() * amax ** 2) <= 2.0
    c_t = np.full(n, 1.0)
    state = _state(n, rng, wmax=wmax)
    state["a"] = rng.uniform(0, amax, size=n)
    g_prev, J_prev = state["g"].copy(), state["J"].copy()
    for _ in range(300):
        state, _ = retired.closed_model_step(state, params, c_t, eta)
        assert (state["a"] >= -1e-12).all()
        assert (state["a"] <= amax + 1e-9).all()
        assert (state["g"] >= -1e-12).all()
        assert (state["g"] <= g_prev + 1e-12).all()
        assert (state["J"] >= -1e-12).all()
        assert (state["J"] <= J_prev + 1e-12).all()
        assert (np.abs(state["w"]) <= wmax + 1e-9).all()
        g_prev, J_prev = state["g"].copy(), state["J"].copy()


def test_closed_model_capture_absorbs():
    # Closure theorem clause (i), simulated: rising thresholds at the
    # capture rate then a floor above the balance point: no topples
    # after T0, stress converges to d0 / mu_x, the jet enters and
    # stays below the criterion (finite-time collapse entry), and the
    # base parks at d0 / mu_b.
    rng = np.random.default_rng(5)
    n = 4
    params = _params(n, rng)
    eta = 0.05
    amax = 0.2
    Dmax = params["d0"] + 2 * params["kappa"] * amax \
        * params["delta"]
    state = dict(
        x=rng.uniform(0, 0.5, size=n),
        b=rng.uniform(0, 0.3, size=n),
        a=rng.uniform(0, 0.1, size=n),
        g=rng.uniform(0.5, 1.0, size=n),
        J=np.full(n, 1.0),
        w=rng.uniform(-0.3, 0.3, size=n),
        pend=np.zeros(n),
    )
    c = np.full(n, 1.0)
    # balance condition at the floor
    assert (Dmax <= params["mu_x"] * 3.0).all()
    toppled_after = 0
    J_crit = 0.1
    entry_step = None
    for t in range(20000):
        if t < 200:
            c = c + eta * Dmax  # capture inequality, sitewise
        state, topple = retired.closed_model_step(state, params, c,
                                                 eta)
        if t >= 1 and topple.any():
            toppled_after += 1
        if entry_step is None and (state["J"] <= J_crit).all():
            entry_step = t
        if entry_step is not None:
            assert (state["J"] <= J_crit + 1e-12).all()  # absorption
    assert toppled_after == 0
    assert entry_step is not None  # finite-time collapse entry
    # printed entry bound: J decays at least geometrically at rate
    # (1 - eta * th_J)
    t_bound = math.ceil(math.log(J_crit)
                        / math.log(1 - eta * params["th_J"].min()))
    assert entry_step <= t_bound
    balance = params["d0"] / params["mu_x"]
    assert np.allclose(state["x"], balance, atol=1e-3)
    assert np.allclose(state["b"], params["d0"] / params["mu_b"],
                       atol=1e-3)
    assert np.abs(state["w"]).max() < 1e-3
    # the source dies through the jet factor: u -> 0 although the
    # amplitude converges to its band value delta / th_a
    u = state["a"] * state["g"] * state["J"]
    assert np.abs(u).max() < 1e-3
    assert np.allclose(state["a"],
                       params["delta"] / params["th_a"], atol=1e-3)


def test_reload_gap():
    # Lemma timescales: from a reset, at least (1-r) c / g steps to
    # the next crossing when each step gains at most g.
    c, r, g = 1.0, 0.6, 0.01
    x = r * c
    steps = 0
    while x <= c:
        x += g
        steps += 1
    assert steps >= (1 - r) * c / g
