"""Executable counterexamples for the formerly printed statements of
the reduced-model theorems (each corrected in the manuscript; the
corrected hypotheses are exactly what these tests show to be
necessary).  Every test simulates the relevant caricature directly in
numpy, with the concrete numbers the corrections cite.

Guarded corrections:
  1. fixed-threshold edge-riding needs the initial interval x0 in [0, c];
  2. warm-up onset ordering needs a shared initial state and threshold;
  3. the two-site strictly-positive-shed clause needs f > 0;
  4. band fixed-point uniqueness needs eta != 0;
  5. flux monotonicity under freezing is nonstrict without a positive
     removed throughput (zero coupling);
  6. the branching fixed-point ratio (1-p)/p is not a probability for
     p < 1/2; extinction is the least fixed point in [0, 1];
  7. the factory two-sided display needs the remainder in absolute
     value (a signed remainder breaks the former lower bound).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "analysis"))


def drive_reset(c, d, r, x):
    return r * x if c < x + d else x + d


def orbit(c, d, r, x0, n):
    xs = [x0]
    for _ in range(n):
        xs.append(drive_reset(c, d, r, xs[-1]))
    return np.array(xs)


# 1 -------------------------------------------------------------------------


def test_edge_riding_needs_initial_interval():
    """c=1, d=0.1, r=1, x0=2: every step topples back to 2; the orbit
    never meets the band (c-d, c] and limsup = 2 > c.  With x0 in
    [0, c] the corrected statement holds (band entry and sandwich)."""
    xs = orbit(1.0, 0.1, 1.0, 2.0, 200)
    assert np.all(xs == 2.0)
    assert not np.any((xs > 0.9) & (xs <= 1.0))
    # corrected hypothesis: from inside [0, c] the orbit enters the
    # band and its running maximum is sandwiched
    ys = orbit(1.0, 0.1, 1.0, 0.0, 200)
    assert np.all((ys >= 0.0) & (ys <= 1.0))
    assert np.any((ys > 0.9) & (ys <= 1.0))
    tail = ys[100:]
    assert 0.9 <= tail.max() <= 1.0


# 2 -------------------------------------------------------------------------


def test_onset_ordering_needs_shared_initial_state():
    """A site with smaller drive but initialized nearer threshold
    crosses first; with shared x0 (and shared threshold) the
    larger-drive site crosses no later."""
    c = 1.0

    def first_cross(d, x0):
        x = x0
        for t in range(1000):
            if x + d > c:
                return t
            x = x + d
        return None

    assert first_cross(0.1, 0.95) < first_cross(0.3, 0.0)
    for x0 in (0.0, 0.4, 0.85):
        assert first_cross(0.3, x0) <= first_cross(0.1, x0)


# 3 -------------------------------------------------------------------------


def test_two_site_shed_zero_at_f_zero():
    """With f = 0 site 1 still topples recurrently but every shed is
    exactly zero: strictly positive shed needs f > 0."""
    c1, d1, r, f = 1.0, 0.3, 0.5, 0.0
    x, sheds = 0.0, []
    for _ in range(200):
        if c1 < x + d1:
            sheds.append(f * x)
            x = r * x
        else:
            x = x + d1
    assert len(sheds) > 10
    assert all(s == 0.0 for s in sheds)


# 4 -------------------------------------------------------------------------


def test_band_uniqueness_needs_eta_nonzero():
    """At eta = 0 the averaged band step is the identity: every point
    is fixed although mu + kap*W != 0.  With eta != 0 the fixed point
    is unique."""
    mu, kap, W, delta = 1.0, 1.0, 1.0, 1.0

    def step(eta, u):
        return u - eta * ((mu + kap * W) * u - delta)

    for u in (-3.0, 0.0, 0.7, 42.0):
        assert step(0.0, u) == u
    eta = 0.5
    fixed = [u for u in np.linspace(-5, 5, 10001)
             if abs(step(eta, u) - u) < 1e-12]
    ub = delta / (mu + kap * W)
    assert all(abs(u - ub) < 2e-3 for u in fixed)


# 5 -------------------------------------------------------------------------


def test_flux_freeze_nonstrict_at_zero_coupling():
    """Two sites, beta = 0 (no shedding into site 1): freezing site 2
    leaves site 1's stationary topple rate exactly unchanged, so the
    strict-decrease clause fails without a positive removed
    throughput; with beta_{21} > 0 the decrease is strict."""
    eta, c = 1.0, 1.0
    d = np.array([0.4, 0.7])

    def rates(beta, live):
        b = beta[np.ix_(live, live)]
        rhs = eta * d[live] / c
        return np.linalg.solve(np.eye(len(live)) - b.T, rhs)

    beta0 = np.zeros((2, 2))
    r_all = rates(beta0, [0, 1])
    r_frozen = rates(beta0, [0])
    assert np.isclose(r_all[0], r_frozen[0])
    beta = np.array([[0.0, 0.0], [0.3, 0.0]])   # site 2 sheds into 1
    assert rates(beta, [0])[0] < rates(beta, [0, 1])[0]


# 6 -------------------------------------------------------------------------


def test_extinction_is_least_fixed_point():
    """f(s) = 1 - p + p s^2: at p = 1/4 the ratio root is 3 (not a
    probability) and the iterates from 0 converge to 1; at p = 3/4
    they converge to (1-p)/p = 1/3, the least fixed point."""
    def f(p, s):
        return 1 - p + p * s * s

    p = 0.25
    assert np.isclose((1 - p) / p, 3.0)
    s = 0.0
    for _ in range(10000):
        s = f(p, s)
    assert abs(s - 1.0) < 1e-3
    p = 0.75
    s = 0.0
    for _ in range(10000):
        s = f(p, s)
    assert abs(s - (1 - p) / p) < 1e-9
    assert (1 - p) / p < 1.0


# 7 -------------------------------------------------------------------------


def test_factory_display_needs_absolute_remainder():
    """lam = base + rho^2 q + r with q in [qlo, qhi].  The former
    display's lower bound qlo rho^2 - r fails for r < 0; the corrected
    base + qlo rho^2 - |r| holds."""
    base, qlo, qhi, q, rho, r = 0.0, 1.0, 2.0, 1.0, 0.1, -0.02
    lam = base + rho ** 2 * q + r
    old_lower = qlo * rho ** 2 - r
    assert old_lower > lam            # the former bound is violated
    new_lower = base + qlo * rho ** 2 - abs(r)
    new_upper = base + qhi * rho ** 2 + abs(r)
    assert new_lower <= lam <= new_upper


# 8 -------------------------------------------------------------------------
# (cases 8-12: the corrected-statement suite of the revision that
# followed the wave-8 record; each is also machine-checked in
# PldrLlmCurvatureSandpile/Counterexamples.lean)


def test_carrier_convergence_needs_eta_positive():
    """Sub-threshold two-dimensional contraction: at eta = 0 the
    hypotheses 0 <= eta, eta*mu <= 1 hold, q = max(1 - eta*mu, q_w^2)
    evaluates to 1, and the orbit is constant at u0 != u*: the
    convergence clause needs eta > 0 (and mu > 0)."""
    eta, mu, lam0, kap = 0.0, 1.0, 1.0, 1.0
    ustar, R = 1.0, 1.0
    q_w = max(abs(1 - eta * lam0),
              abs(1 - eta * (lam0 + kap * (ustar + R) ** 2)))
    q = max(1 - eta * mu, q_w ** 2)
    assert q == 1.0
    u, w = 0.5, 0.5
    for _ in range(100):
        u = u - eta * (-mu * ustar + mu * u + kap * u * w * w)
        w = (1 - eta * (lam0 + kap * u * u)) * w
    assert u == 0.5 and w == 0.5          # constant orbit, no decay


# 9 -------------------------------------------------------------------------


def test_window_drift_needs_mu_nonneg():
    """Finite-window drift: at eta = 1, mu = -1, kap = delta = 0,
    u0 = 1 the next iterate is 2 while the displayed window bound
    (with m = W = 0) is u0 + N*eta*delta = 1."""
    eta, mu, kap, delta = 1.0, -1.0, 0.0, 0.0
    u0, w0 = 1.0, 0.0
    u1 = u0 - eta * (-delta + mu * u0 + kap * u0 * w0 ** 2)
    bound = u0 + 1 * eta * (delta - kap * 0.0 * 0.0)
    assert u1 == 2.0 and bound == 1.0 and u1 > bound


# 10 ------------------------------------------------------------------------


def test_band_no_crossing_needs_etamu_le_one():
    """Band dynamics, oscillation off: at mu = delta = 1, eta = 3/2,
    W = 0, u0 = 0 the next iterate is 3/2, crossing the cap
    delta/mu = 1 from below."""
    eta, mu, delta, W = 1.5, 1.0, 1.0, 0.0
    u0 = 0.0
    cap = delta / mu
    u1 = (1 - eta * (mu + W)) * u0 + eta * delta
    assert u0 < cap < u1


# 11 ------------------------------------------------------------------------


def test_moving_band_window_admits_negative_qstar():
    """Moving band: the window 0 < eta*(mu + kap*Wmax) <= 1 admits
    eta = -1, mu = -2, kap = 1, [Wmin, Wmax] = [0, 1], for which
    q* = 1 - eta*(mu + kap*Wmin) = -1 and the one-step tracking
    display fails as a bound."""
    eta, mu, kap = -1.0, -2.0, 1.0
    wmin, wmax = 0.0, 1.0
    assert 0 < eta * (mu + kap * wmax) <= 1
    qstar = 1 - eta * (mu + kap * wmin)
    assert qstar == -1.0
    assert not 0 <= qstar < 1     # not a contraction factor
    # the tracking display |u_t - b| <= q*^t |u0 - b| (constant band,
    # zero variation sum) is false at t = 1: the left side is positive
    # while q*^1 < 0; and the orbit never converges (|u_t - b| = 1
    # forever, alternating sign)
    b = 1.0    # band point of mu + kap*Wmin = -2 with delta = -2
    delta = b * (mu + kap * wmin)
    u = 2.0
    for t in range(1, 8):
        u = u - eta * ((mu + kap * wmin) * u - delta)
        assert abs(u - b) == 1.0
        if t % 2 == 1:
            assert not abs(u - b) <= qstar ** t * 1.0


# 12 ------------------------------------------------------------------------


def test_two_site_orbit_reversal():
    """Rerouting is a one-step pointwise comparison only: with c = 1,
    d = 0.2, r = 0, f = 0.5, x0 = 0.9 the frozen (reset r) and
    rerouted (reset r + f) orbits reach 0.6 and 0.425 after four
    steps, reversing the pointwise order, because the reset map is
    discontinuous and not monotone."""
    c, d = 1.0, 0.2

    def orbit4(reset, x0):
        x = x0
        for _ in range(4):
            xp = x + d
            x = reset * x if xp > c else xp
        return x

    frozen = orbit4(0.0, 0.9)
    rerouted = orbit4(0.5, 0.9)
    assert abs(frozen - 0.6) < 1e-12
    assert abs(rerouted - 0.425) < 1e-12
    assert rerouted < frozen               # orbit order reversed

    def step(reset, x):
        xp = x + d
        return reset * x if xp > c else xp

    # while the one-step comparison at a common state does hold
    # (larger reset factor keeps at least as much stress)
    for x in (0.0, 0.5, 0.9, 1.5):
        assert step(0.0, x) <= step(0.5, x)
