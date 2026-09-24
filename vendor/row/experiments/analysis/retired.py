"""RETIRED Revision 10 forms, kept ONLY as counterexample
regressions.  Nothing in the live analysis, protocol generation, or
power simulation imports this module; the tests do, to pin the
witnesses that retired each form:

* ``closed_model_step``: the factored map with an INDEPENDENT stress
  ledger x updated beside the factor states.  Retired because it
  carries two incompatible versions of the same curvature: from
  x = b + kappa u^2 = 1 (b = 0, u = 1, kappa = 1, source contraction
  0.2, mu_x = 0.4, eta = 0.01) the smooth branch returns x+ = 0.992
  while the factor and base laws return 0.996004, and the topple
  branch (c = 0.9, r = 1/2) returns x+ = 0.45 against the same
  0.996004 (the registered constraint-violation witnesses).
* ``loading_step``: the loading law that subtracts the full decay
  reversion mu_x x after a factor velocity that already carries the
  decay shares.  Retired because it counts weight decay twice: on
  the pure-decay witness (p = 2, lam_wd = 0.1, x = kappa u^2 = 1)
  it drains at first-order rate -0.8 where the true factory rate is
  -0.4 (theory.pure_decay_rate).
* ``reset_interval_strict_firstcrossing``: the strict reset interval
  evaluated with the FIRST-CROSSING drain bound
  min(h a z_cap, sqrt(Q0)) instead of the event-end bound sqrt(Q0).
  Retired because the strict burst theorem reads the reset at the
  power-floor event end: at the old planning constants
  (z0 = z_min = 0.05, eps0 = 0.0005, h = 0.1, a = 1, c = 0.1,
  r_minus = 0.05) it returns (0.95, 0.955) although the theorem's
  (M-sep) fails, sqrt(Q0) = 0.2236 > 0.095 (the registered
  convention witness; the corrected chain licenses its reset
  interval from the event clock instead,
  theory.reset_interval_licensed).

Revision 11 forms, retired because the LINEAR BURST NORMAL FORM is
not the recurrence induced by the factor-derived event (the
event-clock witness: at the gated planning state b = 0.1005,
a = g = kappa = 1, J = 0.69965, c = 0.3, eta = 0.05,
th_J = 0.01, h = 0.1, b_sat = 0.6, z_seed = 0.01, z_floor = 0.002,
theory.run_event drains to eps_N = -0.10718865 (retained fraction
0.6427045) while the normal form with a_b = 1 predicts
-1.07171504 (retained -2.5723835): a margin residual of 0.9645264,
and the event-average effective a_b = 0.291688 varies by a factor
35 over a learning-rate sweep, so no constant drain coefficient
exists; pinned by tests):

* ``burst_step`` / ``simulate_burst_to_floor``: the independent
  margin recurrence eps' = eps - h*a*z beside the logistic power,
  iterated to the power floor.  The exact drain law of the
  implemented event is logarithmic (theory.run_event returns
  sum_log_phi; theory.drain_enclosure bounds the linear reading).
* ``retained_fraction_from_drain``: the linear per-event reset
  relation r = 1 + (eps0 - h*a*sum_z)/c with its constant a_b.
  The exact relation is theory.retained_fraction_exact.
* ``reset_interval_strict`` / ``reset_interval``: reset intervals
  whose hypotheses ((M-drain), (M-sep), sqrt(Q0)) are about the
  normal form; the event clock licenses
  theory.reset_interval_licensed with no separation condition.
* ``event_end_bound``: the normal-form event-length bound
  N1 + N2 = ceil((eps0+estar)/(h*a*z_floor)) + ...; the event
  clock's bound is theory.event_time_bound (unconditional
  geometric jet drain).
* ``integrated_power`` / ``peak_power_cap``: (eps0 - epsN)/(h*a)
  and z0 + eps0^2/a, both a_b-based; the event's drained curvature
  is exact from its factors (run_event's drop) and the power cap is
  the h-dependent discrete invariant theory.z_cap_invariant (the
  continuous-time cap is overshot in one legal step: z = 0.75,
  h = 1, b_sat = 1, eps_max = 0.9 gives z' = 0.97275084375 > 0.9;
  pinned by a test).

The live forms are in ``theory.py``: the single curvature state
(x = b + kappa u^2 derived, never stored), the decay ledger, the
single event clock with its exact logarithmic drain law, and the
licensed reset interval.
"""

import math

import numpy as np

from theory import drive_law, source_velocity


def loading_step(x, eta, d0, kappa, u, v, p_i, lam_wd, h_euler=0.0):
    """RETIRED Revision 10 loading law (double-counted decay): the
    factor velocity v carries the decay shares AND the reversion
    2 p lam_wd x is subtracted again.  Kept as the decay-ledger
    necessity witness."""
    return (
        x
        + eta * drive_law(d0, kappa, u, v)
        - eta * (2.0 * p_i * lam_wd * x + lam_wd * h_euler)
    )


def reset_interval_strict_firstcrossing(z0, eps0, h, a, c, z_min,
                                        eps_plus, r_minus):
    """RETIRED first-crossing variant of the strict reset interval:
    tests (M-sep) with min(h a z_cap, sqrt(Q0)), the shallow bound at
    the first sign crossing, where the theorem reads the event end
    and uses sqrt(Q0).  Kept as the convention necessity witness."""
    if not (0.0 < z_min <= z0):
        raise ValueError("seed floor violated: need 0 < z_min <= z0")
    if not (0.0 <= eps0 <= eps_plus):
        raise ValueError("entry-margin cap violated: need "
                         "0 <= eps0 <= eps_plus")
    if c <= 0.0 or h <= 0.0 or a <= 0.0:
        raise ValueError("need positive c, h, a")
    if not h * a * z_min > eps_plus:
        raise ValueError("M-drain violated: need h*a*z_min > eps_plus")
    q0 = a * z0 + eps0 * eps0
    zcap = z0 + eps0 * eps0 / a
    drain_hi = min(h * a * zcap, math.sqrt(q0))
    r_lo = 1.0 - drain_hi / c
    r_hi = 1.0 - (h * a * z_min - eps_plus) / c
    if not (r_minus > 0.0 and r_lo >= r_minus):
        raise ValueError("M-sep (first-crossing form) violated")
    if r_lo > r_hi:
        raise ValueError("inconsistent interval")
    return r_lo, r_hi


def closed_model_step(state, params, c_t, eta_t, rng=None,
                      z_seed=0.0, validate=True):
    """RETIRED Revision 10 closed map on the factored state WITH an
    independent stress ledger.  Kept verbatim as the target of the
    constraint-violation, period-two, and convergence regressions.

    state:  dict of 1-d arrays x, b, a, g, J, w, pend (per site):
            stress ledger, base curvature, coupling amplitude, chain
            factor, jet magnitude, carrier, pending shed.
    params: dict with beta (matrix), d0, delta, kappa, th_a, th_g,
            th_J (arrays), mu_b, mu_x (array or scalar), r, rho_w
            (scalars).
    Returns (new_state, toppled_mask)."""
    x, b, a, g, J, w, pend = (
        state[k].copy() for k in ("x", "b", "a", "g", "J", "w", "pend")
    )
    beta = params["beta"]
    d0, delta, kappa = params["d0"], params["delta"], params["kappa"]
    th_a, th_g, th_J = params["th_a"], params["th_g"], params["th_J"]
    mu_b, mu_x = params["mu_b"], params["mu_x"]
    r, rho_w = params["r"], params["rho_w"]

    if validate:
        checks = [
            ("eta*mu_x", np.max(eta_t * np.asarray(mu_x))),
            ("eta*mu_b", np.max(eta_t * np.asarray(mu_b))),
            ("eta*th_a", np.max(eta_t * np.asarray(th_a))),
            ("eta*th_g", np.max(eta_t * np.asarray(th_g))),
            ("eta*(th_J + kappa*w^2)",
             np.max(eta_t * (np.asarray(th_J) + kappa * w * w))),
        ]
        for name, val in checks:
            if val > 1.0 + 1e-12:
                raise ValueError(
                    "closure step condition (Q+) violated: "
                    f"{name} = {val:.6g} > 1"
                )

    arrivals = beta.T @ pend
    u = a * g * J
    v = source_velocity(a, g, J, w, delta, th_a, th_g, th_J, kappa)
    drive = drive_law(d0, kappa, u, v)
    loaded = np.maximum(
        0.0, x + arrivals + eta_t * drive - eta_t * mu_x * x
    )
    topple = loaded > c_t

    new = {}
    new["x"] = np.where(topple, r * c_t, loaded)
    new["pend"] = np.where(topple, loaded - r * c_t, 0.0)
    q = b + kappa * u * u          # carrier damping reads the base STATE
    new["w"] = np.where(topple, rho_w * w, (1.0 - eta_t * q) * w)
    new["b"] = b + eta_t * d0 - eta_t * mu_b * b
    new["a"] = (1.0 - eta_t * th_a) * a + eta_t * delta
    new["g"] = (1.0 - eta_t * th_g) * g
    new["J"] = (1.0 - eta_t * (th_J + kappa * w * w)) * J
    if rng is not None and z_seed > 0.0:
        kick = rng.normal(0.0, math.sqrt(z_seed), size=w.shape)
        new["w"] = np.where(topple, new["w"] + kick, new["w"])
    return new, topple


# ------------------------------------------- Revision 11 normal form


def burst_step(z, eps, h, a, b):
    """RETIRED one step of the linear burst normal form: the
    independent margin recurrence eps' = eps - h*a*z beside the
    logistic power.  Not the recurrence induced by the factor
    event (see the module docstring's event-clock witness)."""
    return z * (1.0 + 2.0 * h * (eps - b * z)), eps - h * a * z


def simulate_burst_to_floor(z0, eps0, h, a, b, z_floor,
                            max_steps=200000):
    """RETIRED normal-form iteration to the power floor.  Kept as
    the second simulator whose planning outputs (event 135 steps,
    retained fraction 0.38523, sum_z 0.41251 at the archived T-sep
    constants) belong to the normal form and not to the map's
    event; the live planners consume theory.run_event only."""
    if not 0.0 < z_floor <= z0:
        raise ValueError("need 0 < z_floor <= z0")
    z, eps = float(z0), float(eps0)
    ztr, etr = [z], [eps]
    for n in range(1, max_steps + 1):
        z, eps = burst_step(z, eps, h, a, b)
        ztr.append(z)
        etr.append(eps)
        if z < z_floor:
            return n, ztr, etr
    return None, ztr, etr


def retained_fraction_from_drain(eps0, h, a, sum_z, c):
    """RETIRED linear per-event reset relation
    r = 1 + (eps0 - h*a*sum_z)/c.  On the event-clock witness it
    predicts an impossible negative retained curvature; the exact
    relation is theory.retained_fraction_exact."""
    return 1.0 + (eps0 - h * a * sum_z) / c


def reset_interval(z0, eps0, h, a):
    """RETIRED unstrengthened normal-form exit-margin interval."""
    q0 = a * z0 + eps0 * eps0
    zcap = q0 / a
    return (-min(h * a * zcap, math.sqrt(q0)), 0.0)


def reset_interval_strict(z0, eps0, h, a, c, z_min, eps_plus,
                          r_minus):
    """RETIRED strict-burst-package reset interval ((M-drain),
    (M-sep), sqrt(Q0)): hypotheses and conclusion are about the
    normal form's margin recurrence, which the factor event does
    not realize.  The event clock licenses
    theory.reset_interval_licensed instead."""
    if not (0.0 < z_min <= z0):
        raise ValueError("seed floor violated: need 0 < z_min <= z0")
    if not (0.0 <= eps0 <= eps_plus):
        raise ValueError("entry-margin cap violated: need "
                         "0 <= eps0 <= eps_plus")
    if c <= 0.0 or h <= 0.0 or a <= 0.0:
        raise ValueError("need positive c, h, a")
    if not h * a * z_min > eps_plus:
        raise ValueError("M-drain violated: need h*a*z_min > eps_plus")
    q0 = a * z0 + eps0 * eps0
    drain_hi = math.sqrt(q0)
    r_lo = 1.0 - drain_hi / c
    r_hi = 1.0 - (h * a * z_min - eps_plus) / c
    if not (r_minus > 0.0 and r_lo >= r_minus):
        raise ValueError("M-sep violated: need "
                         "sqrt(Q0) <= (1 - r_minus)*c "
                         "with r_minus > 0")
    if r_lo > r_hi:
        raise ValueError("inconsistent interval: hypotheses admit no "
                         "retained fraction")
    return r_lo, r_hi


def event_end_bound(eps0, estar, h, a, b, z_floor, z_cap):
    """RETIRED normal-form event-length bound; the event clock's
    bound is theory.event_time_bound."""
    if min(eps0, estar, h, a, z_floor) <= 0.0 or b < 0.0:
        raise ValueError("need positive eps0, estar, h, a, z_floor "
                         "and b >= 0")
    if not z_floor < z_cap:
        raise ValueError("need z_floor < z_cap")
    n1 = math.ceil((eps0 + estar) / (h * a * z_floor))
    n2 = math.ceil(math.log(z_cap / z_floor) / (2.0 * h * estar))
    return n1 + n2


def integrated_power(eps0, epsN, h, a):
    """RETIRED a_b-based integrated power (eps0 - epsN)/(h*a)."""
    return (eps0 - epsN) / (h * a)


def peak_power_cap(z0, eps0, a):
    """RETIRED continuous-time power cap z0 + eps0^2/a: overshot in
    one legal discrete step (module docstring witness); the discrete
    invariant cap is theory.z_cap_invariant."""
    return z0 + eps0 * eps0 / a


def simulate_burst(z0, eps0, h, a, b, max_steps=100000):
    """RETIRED normal-form iteration to the signed exit (eps <= 0)
    or the step cap.  Kept only for the normal-form regression pins
    (seed witness, exit dichotomy, telescoping)."""
    z, eps = float(z0), float(eps0)
    ztr, etr = [z], [eps]
    for n in range(1, max_steps + 1):
        z, eps = burst_step(z, eps, h, a, b)
        ztr.append(z)
        etr.append(eps)
        if eps <= 0.0:
            return n, ztr, etr
    return None, ztr, etr
