"""The theory, executable: estimators and evaluators for the gated
curvature-sandpile model on the single curvature state and the
single event clock.

Every function here mirrors a printed statement of the paper, so the
analysis pipeline, the protocol generators, and the power simulations
all consume one implementation of the theory's formulas.  There is
one implementation of the event: ``run_event``; every planner,
theorem constant, and endpoint that mentions an event consumes it.

The structural conventions of the single-clock chain:

* there is ONE curvature: the stress is the DERIVED quantity
  x = b + kappa * u^2 with u = a * g * J (``curvature``); no function
  stores or updates an independent stress ledger, so the constraint
  x = b + kappa u^2 holds on every branch by construction;
* there is ONE event clock: the multiplicative jet event of
  ``run_event``.  Its drain law is EXACT in logarithmic form: with
  jet curvature y = kappa*(a g J)^2 and per-step factor
  phi_n = 1 - eta*(th_J + lam_wd + kappa*z_n),

      log(y_N / y_0) = 2 * sum_{n<N} log(phi_n),

  and the linear functional D1 = 2*eta*((th_J+lam_wd)*N +
  kappa*sum_z) encloses it two-sidedly (``drain_enclosure``).  No
  linear margin recurrence and no drain coefficient a_b exist
  anywhere in this module;
* every weight-decay term is counted once (the decay ledger): each
  factor rate decomposes as theta_f = theta_f_loss + lam_wd, the
  loss-only source velocity ``source_velocity_loss`` carries no
  decay, and the curvature increment inherits exactly
  -(mu_b * b + 2 p lam_wd kappa u^2); there is no independent
  reversion parameter mu_x;
* the jet rebuild is ACTIVITY-GATED (hypothesis H-gate) with the
  gate in GLOBAL-STEP units: after an event the activity counter is
  the RELOAD DURATION ceil((J_pre - J_post)/(eta*delta_J0)) global
  steps (``reload_steps``), the number of rebuild steps needed to
  restore the drained jet at the scheduled rate; the fast-clock
  iteration count is exported only as a diagnostic;
* the event is a FUEL-INDEXED TOTAL RESOLVER with a tagged result
  (``status`` = "ended"/"exhausted"); the global map REJECTS the
  exhausted branch (``gated_model_step`` raises ``EventNotEnded``),
  and on the entry-gap domain (B-gap, gated by ``b_gap_slack``) the
  branch is unreachable for fuel at or above the termination bound;
* the stored carrier enters every event through the SEEDED ENTRY
  OPERATOR ``seeded_entry`` (kick, truncation, seed floor, sign
  law), which restores z_0 = w_entry^2 exactly; amplitude
  statements compare against the seeded amplitude |w_entry|;
* the event end is the GENERALIZED exit rule (E-end): first step
  with (eps <= 0 OR y <= y_floor) AND
  z < max(z_floor, (eps)_+ / b_sat + tau_z).  On the entry-gap
  domain with y_floor <= delta_b it reduces EXACTLY to the
  two-condition convention (first step with z < z_floor AND
  eps <= 0, exit margin in [b - c, 0]); with y_floor > 0 it is
  additionally total on the overloaded-base branch b >= c;
* the event power is confined to the DISCRETE invariant cap
  ``z_cap_invariant`` (h-dependent; proved by maximizing the exact
  quadratic update, no continuous-time argument), and the step
  condition (S+) eta*(th_J + lam_wd + kappa*Z_cap) <= 1 keeps the
  printed clamp in the jet factor from ever binding on the box;
* the stochastic curvature chain is PROJECTED at zero,
  x' = max((1-a)x + eta*d + eta*zeta, 0); the stay-below bounds are
  stated for the projected chain (projection maps the safe interval
  into itself and preserves the crossing event);
* the primary stress is the physical preconditioned Gauss--Newton
  Rayleigh quotient x. The optimizer recurrence derives the
  dimensionless coordinate sigma = (1 - beta1) * eta * x from that
  physical state and the current applied learning rate. Full-Hessian
  and raw-metric probes are separately named robustness observables
  and never replace x.

Retired forms live in ``retired.py`` and are consumed only by
counterexample regression tests: the Revision 10 independent stress
ledger, double-counted decay, and first-crossing reset bound, and
the Revision 11 linear burst normal form with its drain coefficient
a_b, retired because the factor-derived event does not realize its
margin recurrence (the event-clock witness pins the mismatch).

Contents:
  optimizer / boundary (D2):
    sqrt_drift            exact preconditioner-scale drift statistic
    flip_boundary         damped flip boundary of the frozen recurrence
    jury_stable           full Jury window of the damped recurrence
    companion_matrix      2x2 companion matrix A(nu)
    lyap_solve_2x2        numeric discrete Lyapunov solution P
    lyap2_closed_form     the printed closed-form P entries
    lyap2_residual        self-check of the closed form
    pmax_2x2              top eigenvalue of P
    drift_full            FULL companion drift ||A(nu') - A(nu)||
    drift_dir             directional drifts (|ds|, |dq|)
    cp_dir_bounds         CERTIFIED directional Lipschitz constants
    cp_grid_check         secondary finite-difference sanity check
    c_grnt                scalar-budget boundary (superseded by the
                          band for stability language; kept for
                          calibrated single-number budgets)
    c_grnt_box            box-confined upper component edge (a
                          component edge, NOT a downward threshold)
    cert_band             THE certified sharpness band [lo, hi]
  curvature law (D1):
    source_velocity, source_velocity_loss, drive_law, curvature,
    curvature_increment_exact (closed gate; clip provably inert),
    curvature_increment_exact_gated (clipped transition, with the
    clipping defect chi and the exact defect identity),
    pure_decay_rate
  averaged defect (D1):
    clip_defect_preconditioned, clip_defect_naive (retired form)
  the event clock (D3):
    z_cap_invariant       h-dependent discrete invariant power cap
    event_u_max           clamp margin u_max on the invariant box
    seeded_entry          THE seeded entry operator (kick, floor,
                          sign law; z0 = w_entry^2 exact)
    run_event             THE event (single implementation; a
                          fuel-indexed total resolver, tagged;
                          every ended event emits through the
                          terminal settlement operator E-settle)
    EventNotEnded         the rejected Exhausted branch
    drain_linear          the linear drain functional D1
    drain_enclosure       two-sided enclosure of the exact log drain
    retained_fraction_exact  exact retained fraction from the event
    event_time_bound      finite event-length bound N1 + N2
    event_fuel_required   granted fuel (bound + 1, strict
                          indexing) on the (B-gap) domain
    event_time_bound_overloaded  per-entry bound for the
                          generalized exit at b > c
    event_time_bound_overloaded_uniform  uniform overloaded bound
                          at the worst target tau_z (b -> c+)
    b_gap_slack           the (B-gap) parameter inequality (gate)
    reset_interval_licensed  licensed reset interval [r_lo, 1]
    rho_w_derived, amp_factor  carrier power/amplitude factors
  offspring (D4):
    strict_empirical_margin_cdf, child_mean, transported_mass,
    dose_response_bound, fixed_receiver_envelope,
    multitype_weight_certificate
  capture (D5):
    physical_threshold, threshold_increment_exact,
    threshold_increment_interval, certified_capture_margin,
    classify_schedule_certificate, capture_rhs, capture_side,
    amplitude_envelope, base_envelope, drive_ceiling (D_max),
    capture_balance
  noise (D6, projected chain):
    ar1_stationary_var, passage_band, entry_prob_mc,
    p_min_crossing, hold_below_bound
  gated closure (D7):
    entry_time_bound, reload_steps, jet_unrolled,
    jet_unroll_lower, cycle_floor_inputs, maintenance_floor,
    maintenance_floor_clipped clipped floor comparison: (M-end)
                          for the clipped recursion, gated by
                          J_floor <= 1, no envelope hypothesis
    maintenance_prefix_min    all-prefix (within-cycle) envelope;
                          enforces the (N-sat) certificate
    maintenance_prefix_floor  crash-adjusted closed form
    maintenance_cycle_average cycle-average lower bound (M-avg,
                          independent of the endpoint floor; the
                          statement matched to the measured,
                          time-averaged phase criterion; enforces
                          (N-sat))
    active_reload_count   gate-indicator count (never gap length)
    state_in_box          declared-box membership (self-map tests)
  observables (D8):
    monomial_transform
  gated model:
    gated_model_step      one global step on the single state
"""

import math

import numpy as np

# ----------------------------------------------------------------- D2


def sqrt_drift(v_prev, g, beta2):
    """Exact per-step relative drift of sqrt(v) under the second
    moment EMA (Lean: SqrtDrift.ema_sqrt_drift).  No linearization.
    This is ONE contribution to the full companion drift consumed by
    the certificate boundary; see ``drift_full``."""
    v_new = beta2 * v_prev + (1.0 - beta2) * g * g
    if v_prev <= 0 or v_new <= 0:
        raise ValueError("second moments must be positive")
    return abs(math.sqrt(v_new) - math.sqrt(v_prev)) / math.sqrt(v_prev)


def whitening_transient(t, beta1, beta2):
    """Adam bias-correction transient ``sqrt(1-beta2^t)/(1-beta1^t)``."""
    t = int(t)
    beta1 = float(beta1)
    beta2 = float(beta2)
    if t < 1 or not (0 <= beta1 < 1) or not (0 <= beta2 < 1):
        raise ValueError("require t >= 1 and beta1,beta2 in [0,1)")
    return math.sqrt(1.0 - beta2 ** t) / (1.0 - beta1 ** t)


def whitening_transient_safe_bound(beta1):
    """Uniform bound valid for every ``t >= 1`` and every ``beta2``."""
    beta1 = float(beta1)
    if not 0 <= beta1 < 1:
        raise ValueError("beta1 must lie in [0,1)")
    return 1.0 / (1.0 - beta1)

def flip_boundary(beta1, d):
    """Normalized sharpness at the damped flip root: p_d(-1) = 0 at
    sigma = (2 - d)(1 + beta1) (Lean/paper Thm on the damped
    boundary).  The frozen unit convention:
    sigma = (1 - beta1) * eta * lam; divide by (1 - beta1) * eta for
    eta*lam units."""
    return (2.0 - d) * (1.0 + beta1)


def jury_stable(sigma, beta1, d):
    """All roots of z^2 - (1 + beta1 - sigma - d) z + beta1 (1 - d)
    strictly inside the unit circle (the printed Jury window)."""
    a1 = -(1.0 + beta1 - sigma - d)
    a0 = beta1 * (1.0 - d)
    p1 = 1.0 + a1 + a0        # p(1) > 0
    pm1 = 1.0 - a1 + a0       # p(-1) > 0
    return abs(a0) < 1.0 and p1 > 0.0 and pm1 > 0.0


def companion_matrix(sigma, beta1, d):
    """Companion matrix of the damped EMA recurrence
    theta' = (1 + beta1 - sigma - d) theta - beta1 (1 - d) theta_-."""
    return np.array(
        [[1.0 + beta1 - sigma - d, -beta1 * (1.0 - d)], [1.0, 0.0]]
    )


def lyap_solve_2x2(A):
    """Numeric solution P of P = I + A^T P A for a Schur-stable 2x2
    matrix, by the 3x3 linear system in (p11, p12, p22).  The printed
    closed form is ``lyap2_closed_form``; the two agree to rounding
    (tested)."""
    a11, a12 = A[0, 0], A[0, 1]
    a21, a22 = A[1, 0], A[1, 1]
    M = np.array(
        [
            [1.0 - a11 * a11, -2.0 * a11 * a21, -a21 * a21],
            [-a11 * a12, 1.0 - (a11 * a22 + a12 * a21), -a21 * a22],
            [-a12 * a12, -2.0 * a12 * a22, 1.0 - a22 * a22],
        ]
    )
    rhs = np.array([1.0, 0.0, 1.0])
    p11, p12, p22 = np.linalg.solve(M, rhs)
    return np.array([[p11, p12], [p12, p22]])


def lyap2_closed_form(s, q):
    """The PRINTED closed-form discrete Lyapunov solution for the
    companion matrix A = [[s, -q], [1, 0]] (trace coefficient s,
    determinant q), solving P = I + A^T P A exactly (Lean:
    CompanionDrift.lyap2_closed_form):

        p11 = 2 (1 + q) / ((1 - q) (1 + q - s) (1 + q + s))
        p12 = -q s p11 / (1 + q)
        p22 = 1 + q^2 p11

    The Jury window |q| < 1, 1 + q - s > 0, 1 + q + s > 0 makes every
    denominator positive.  Returns (p11, p12, p22)."""
    den = (1.0 - q) * (1.0 + q - s) * (1.0 + q + s)
    if den <= 0.0 or (1.0 + q) <= 0.0:
        raise ValueError("closed-form P requires the Jury window")
    p11 = 2.0 * (1.0 + q) / den
    p12 = -q * s * p11 / (1.0 + q)
    p22 = 1.0 + q * q * p11
    return p11, p12, p22


def lyap2_residual(s, q):
    """Max abs entry of P - (I + A^T P A) for the closed-form P: the
    self-check that the printed entries solve the Lyapunov equation."""
    p11, p12, p22 = lyap2_closed_form(s, q)
    P = np.array([[p11, p12], [p12, p22]])
    A = np.array([[s, -q], [1.0, 0.0]])
    return float(np.abs(P - (np.eye(2) + A.T @ P @ A)).max())


def pmax_2x2(P):
    """Top eigenvalue of the symmetric 2x2 Lyapunov weight matrix."""
    tr = P[0, 0] + P[1, 1]
    det = P[0, 0] * P[1, 1] - P[0, 1] * P[1, 0]
    disc = max(tr * tr / 4.0 - det, 0.0)
    return tr / 2.0 + math.sqrt(disc)


def drift_full(nu_prev, nu_cur):
    """FULL companion drift: the induced infinity norm of
    A(nu_cur) - A(nu_prev), with nu = (sigma, beta1, d).  Every
    parameter change moves it: tracked sharpness, learning rate (in
    normalized sigma units), momentum coefficient, and decay.  A
    sharpness ramp at zero second-moment drift therefore has a
    strictly positive budget."""
    dA = companion_matrix(*nu_cur) - companion_matrix(*nu_prev)
    return float(np.abs(dA).sum(axis=1).max())


def _sq(nu):
    sigma, beta1, d = nu
    return 1.0 + beta1 - sigma - d, beta1 * (1.0 - d)


def drift_dir(nu_prev, nu_cur):
    """Directional companion drifts (|ds|, |dq|) between two
    parameter points nu = (sigma, beta1, d), in the trace/determinant
    coordinates s = 1 + beta1 - sigma - d, q = beta1 (1 - d).  These
    are the per-step inputs of the directional certificate budget
    K_s |ds| + K_q |dq| (see ``cp_dir_bounds``); note
    |ds| + |dq| = drift_full for the companion form."""
    s0, q0 = _sq(nu_prev)
    s1, q1 = _sq(nu_cur)
    return abs(s1 - s0), abs(q1 - q0)


def cp_dir_bounds(sigma_lo, sigma_hi, beta1_lo, beta1_hi, d_lo, d_hi):
    """CERTIFIED directional Lipschitz constants of the Lyapunov
    matrix P on the declared box (paper Thm cpbound; Lean:
    CertBox).  Closed-form worst-case bounds, evaluated at the box
    corners; no grids, no sampling.  For any nu, nu' in the box,

        ||P(nu) - P(nu')||_op <= K_s |s - s'| + K_q |q - q'|
                              <= C_P ||nu - nu'||_1 ,

    with s, q the trace/determinant coordinates and C_P = K_s + K_q
    (the chart bounds |ds| <= ||dnu||_1 and |dq| <= ||dnu||_1 hold on
    the box).  The bounds derive from exact difference identities of
    the closed-form entries; the Jury margins

        gamma1 = min(1 + q - s) = sigma_lo + d_lo (1 - beta1_hi)
        gamma2 = min(1 + q + s) = 2 + beta1_lo (2 - d_hi)
                                    - sigma_hi - d_hi

    must be positive and q_hi = beta1_hi (1 - d_lo) < 1, else the
    box touches the stability boundary and a ValueError names the
    violated margin.  Returns a dict with K_s, K_q, C_P and the
    intermediate constants (printed wherever C_P is used)."""
    if sigma_lo <= 0.0:
        raise ValueError("box margin violated: need sigma_lo > 0")
    q_lo = beta1_lo * (1.0 - d_hi)
    q_hi = beta1_hi * (1.0 - d_lo)
    if not q_hi < 1.0:
        raise ValueError("box margin violated: need "
                         "q_hi = beta1_hi*(1 - d_lo) < 1")
    gamma1 = sigma_lo + d_lo * (1.0 - beta1_hi)
    gamma2 = (2.0 + beta1_lo * (2.0 - d_hi) - sigma_hi - d_hi)
    if gamma1 <= 0.0:
        raise ValueError("box margin violated: gamma1 <= 0")
    if gamma2 <= 0.0:
        raise ValueError("box margin violated: gamma2 <= 0 "
                         "(box reaches the flip boundary)")
    s_lo = 1.0 + beta1_lo - sigma_hi - d_hi
    s_hi = 1.0 + beta1_hi - sigma_lo - d_lo
    s_max = max(abs(s_lo), abs(s_hi))
    e_min = gamma1 * gamma2
    d_min = (1.0 - q_hi) * e_min
    p11_max = 2.0 * (1.0 + q_hi) / d_min
    l11_s = (4.0 * (1.0 + q_hi) * s_max
             / ((1.0 - q_hi) * e_min * e_min))
    l_dd = 1.0 + 2.0 * q_hi + 3.0 * q_hi * q_hi + s_max * s_max
    l11_q = 2.0 / d_min + 2.0 * (1.0 + q_hi) * l_dd / (d_min * d_min)
    f_max = q_hi * s_max / (1.0 + q_lo)
    l12_s = f_max * l11_s + p11_max * q_hi / (1.0 + q_lo)
    l12_q = (f_max * l11_q
             + p11_max * s_max / ((1.0 + q_lo) * (1.0 + q_lo)))
    l22_s = q_hi * q_hi * l11_s
    l22_q = q_hi * q_hi * l11_q + 2.0 * q_hi * p11_max
    k_s = max(l11_s, l22_s) + l12_s
    k_q = max(l11_q, l22_q) + l12_q
    return dict(
        K_s=float(k_s), K_q=float(k_q), C_P=float(k_s + k_q),
        gamma1=float(gamma1), gamma2=float(gamma2),
        q_lo=float(q_lo), q_hi=float(q_hi), s_max=float(s_max),
        D_min=float(d_min), p11_max=float(p11_max),
        box=dict(sigma=(sigma_lo, sigma_hi),
                 beta1=(beta1_lo, beta1_hi), d=(d_lo, d_hi)),
    )


def cp_grid_check(sigma_lo, sigma_hi, beta1_lo, beta1_hi, d_lo, d_hi,
                  n=41):
    """SECONDARY sanity check of the certified constants: the maximum
    over a finite-difference grid of the l1 gradient norm of
    pmax(P).  A grid statistic of the largest eigenvalue only; it is
    NOT the certified matrix Lipschitz constant (that is
    ``cp_dir_bounds``) and must lie below C_P (tested)."""
    sig = np.linspace(sigma_lo, sigma_hi, n)
    be = np.linspace(beta1_lo, beta1_hi, n)
    dd = np.linspace(d_lo, d_hi, n)
    pm = np.full((n, n, n), np.nan)
    for i, s_ in enumerate(sig):
        for j, b_ in enumerate(be):
            for k, d_ in enumerate(dd):
                if jury_stable(s_, b_, d_):
                    s_c = 1.0 + b_ - s_ - d_
                    q_c = b_ * (1.0 - d_)
                    p11, p12, p22 = lyap2_closed_form(s_c, q_c)
                    pm[i, j, k] = pmax_2x2(
                        np.array([[p11, p12], [p12, p22]]))
    gs, gb, gd = np.gradient(pm, sig, be, dd)
    l1 = np.abs(gs) + np.abs(gb) + np.abs(gd)
    val = np.nanmax(l1)
    if not np.isfinite(val):
        raise ValueError("no Jury-stable interior on the box")
    return float(val)


def c_grnt(beta1, d, eps_A, C_P, tol=1e-10):
    """Certificate boundary of the reduced damped-EMA recurrence, in
    normalized-sharpness units, under the scalar budget
    C_P * eps_A * (pmax(sigma) - 1) < 1.  ``eps_A`` is the full
    companion drift ``drift_full``; ``C_P`` a Lipschitz constant
    valid at the evaluated sigma (use ``c_grnt_box`` for the
    box-confined directional form; this scalar form is kept for
    calibrated single-number budgets).  The identification of this
    reduced-model certificate with the live instability onset is
    hypothesis I2, tested by E0/E2; the function itself guarantees
    only the reduced recurrence.  Monotone bisection from the frozen
    flip; pmax diverges at the frozen boundary, so the root is
    interior at any positive budget."""
    hi = flip_boundary(beta1, d)
    lo = 0.0
    if eps_A <= 0 or C_P <= 0:
        return hi

    def ok(sigma):
        if not jury_stable(sigma, beta1, d):
            return False
        P = lyap_solve_2x2(companion_matrix(sigma, beta1, d))
        return C_P * eps_A * (pmax_2x2(P) - 1.0) < 1.0

    if ok(hi * (1.0 - 1e-12)):
        return hi
    while hi - lo > tol * max(hi, 1.0):
        mid = (lo + hi) / 2.0
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def c_grnt_box(beta1, d, eps_s, eps_q, bounds, n_scan=400,
               tol=1e-10):
    """Certificate boundary CONFINED to the certified box: the
    largest sigma in [sigma_lo, sigma_hi] of ``bounds['box']`` at
    which the directional budget

        (K_s * eps_s + K_q * eps_q) * (pmax(sigma) - 1) < 1

    holds, with (K_s, K_q) the certified constants of
    ``cp_dir_bounds`` and (eps_s, eps_q) the per-step directional
    drifts of the logged path (``drift_dir``).  The search never
    leaves the box: if the budget still holds at the upper edge the
    result is the edge with ``at_edge = True`` (reported, never
    extrapolated); if it holds nowhere on the box the result is
    ``empty = True``.  Returns a dict with sigma_star, at_edge,
    empty, and the budget."""
    box = bounds["box"]
    sigma_lo, sigma_hi = box["sigma"]
    budget = bounds["K_s"] * eps_s + bounds["K_q"] * eps_q

    def ok(sigma):
        if not jury_stable(sigma, beta1, d):
            return False
        P = lyap_solve_2x2(companion_matrix(sigma, beta1, d))
        return budget * (pmax_2x2(P) - 1.0) < 1.0

    if budget <= 0.0:
        return dict(sigma_star=sigma_hi, at_edge=True, empty=False,
                    budget=budget)
    if ok(sigma_hi):
        return dict(sigma_star=sigma_hi, at_edge=True, empty=False,
                    budget=budget)
    grid = np.linspace(sigma_lo, sigma_hi, n_scan)
    ok_idx = [i for i, s_ in enumerate(grid) if ok(s_)]
    if not ok_idx:
        return dict(sigma_star=None, at_edge=False, empty=True,
                    budget=budget)
    lo = grid[ok_idx[-1]]
    hi = grid[min(ok_idx[-1] + 1, n_scan - 1)]
    while hi - lo > tol * max(hi, 1.0):
        mid = (lo + hi) / 2.0
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return dict(sigma_star=float(lo), at_edge=False, empty=False,
                budget=budget)


def cert_band(beta1, d, eps_s, eps_q, bounds, sigma_ref=None,
              n_scan=400, tol=1e-10):
    """The CERTIFIED SHARPNESS BAND (paper Thm certband): the
    connected component of the certified set

        { sigma in [sigma_lo, sigma_hi] :
          jury_stable AND
          (K_s*eps_s + K_q*eps_q) * (pmax(sigma) - 1) < 1 }

    containing the reference operating point sigma_ref (or, when
    sigma_ref is None, the component whose interior maximizes the
    certificate margin).  The Lyapunov gain pmax is U-SHAPED on the
    box (it diverges at both Jury boundaries), so the certified set
    is an interior band and is NOT downward closed in general: a
    single upper supremum described as a threshold ``below which''
    the certificate holds is wrong (the non-monotone gain witness
    is pinned by a test), and every stability statement must be of
    the form ``inside the certified band''.  The logged path must
    remain inside the reported component.  Returns a dict with the
    outward-refined component edges sigma_cert_lo/sigma_cert_hi,
    at_lo_edge/at_hi_edge (True when the component touches the box
    edge: reported, never extrapolated), downward_closed (True only
    when the component reaches the lower box edge), empty, and the
    budget.  Raises when sigma_ref is given but not certified."""
    box = bounds["box"]
    sigma_lo, sigma_hi = box["sigma"]
    budget = bounds["K_s"] * eps_s + bounds["K_q"] * eps_q

    def ok(sigma):
        if not jury_stable(sigma, beta1, d):
            return False
        P = lyap_solve_2x2(companion_matrix(sigma, beta1, d))
        return budget * (pmax_2x2(P) - 1.0) < 1.0

    def margin(sigma):
        if not jury_stable(sigma, beta1, d):
            return -math.inf
        P = lyap_solve_2x2(companion_matrix(sigma, beta1, d))
        return 1.0 - budget * (pmax_2x2(P) - 1.0)

    if budget <= 0.0:
        return dict(sigma_cert_lo=sigma_lo, sigma_cert_hi=sigma_hi,
                    at_lo_edge=True, at_hi_edge=True,
                    downward_closed=True, empty=False,
                    budget=budget)
    grid = np.linspace(sigma_lo, sigma_hi, n_scan)
    good = [ok(s_) for s_ in grid]
    # connected components as index runs on the grid
    comps = []
    start = None
    for i, g_ in enumerate(good):
        if g_ and start is None:
            start = i
        if not g_ and start is not None:
            comps.append((start, i - 1))
            start = None
    if start is not None:
        comps.append((start, n_scan - 1))
    if not comps:
        return dict(sigma_cert_lo=None, sigma_cert_hi=None,
                    at_lo_edge=False, at_hi_edge=False,
                    downward_closed=False, empty=True,
                    budget=budget)
    if sigma_ref is not None:
        if not ok(sigma_ref):
            raise ValueError("the reference operating point is not "
                             "certified at this budget")
        sel = None
        for i0, i1 in comps:
            if grid[i0] <= sigma_ref <= grid[i1]:
                sel = (i0, i1)
        if sel is None:
            # sigma_ref certified between grid points: take the
            # nearest component
            sel = min(comps, key=lambda c: min(
                abs(grid[c[0]] - sigma_ref),
                abs(grid[c[1]] - sigma_ref)))
    else:
        sel = max(comps, key=lambda c: max(
            margin(grid[i]) for i in range(c[0], c[1] + 1)))
    i0, i1 = sel

    def refine(lo_in, hi_out):
        # inward-certified lo_in, outward hi_out: bisect INWARD so
        # the reported edge stays certified
        while abs(hi_out - lo_in) > tol * max(abs(hi_out), 1.0):
            mid = (lo_in + hi_out) / 2.0
            if ok(mid):
                lo_in = mid
            else:
                hi_out = mid
        return lo_in

    at_lo = i0 == 0
    at_hi = i1 == n_scan - 1
    lo_edge = grid[i0] if at_lo else refine(grid[i0], grid[i0 - 1])
    hi_edge = grid[i1] if at_hi else refine(grid[i1], grid[i1 + 1])
    return dict(sigma_cert_lo=float(lo_edge),
                sigma_cert_hi=float(hi_edge),
                at_lo_edge=at_lo, at_hi_edge=at_hi,
                downward_closed=bool(at_lo), empty=False,
                budget=budget)


# ----------------------------------------------------------------- D1


def source_velocity(a, g, J, w, delta, th_a, th_g, th_J, kappa):
    """Source velocity induced by the TOTAL factor laws (rates
    include their weight-decay shares): with u = a*g*J,

        v = delta*g*J - (th_a + th_g + th_J + kappa*w^2) * u.

    The rebuild ``delta`` feeds the coupling amplitude, so it enters
    the product through g*J: when the jet or the chain factor is
    gone, rebuilding the amplitude restores nothing.  For the
    decay-ledger form of the curvature law use
    ``source_velocity_loss`` (Lean: CurvatureState.srcVel)."""
    u = a * g * J
    return delta * g * J - (th_a + th_g + th_J + kappa * w * w) * u


def source_velocity_loss(a, g, J, w, delta, th_a_loss, th_g_loss,
                         th_J_loss, kappa):
    """LOSS-ONLY source velocity of the decay ledger: the factor
    rates with every weight-decay share removed,

        v_loss = delta*g*J
                 - (th_a_loss + th_g_loss + th_J_loss + kappa*w^2) u.

    The curvature increment then reads
    eta*(d0 + 2 kappa u v_loss) - eta*(mu_b b + 2 p lam_wd kappa u^2)
    + O(eta^2): decay appears only in the explicit reversion term,
    counted once (Lean: DecayLedger.counted_once)."""
    u = a * g * J
    return (delta * g * J
            - (th_a_loss + th_g_loss + th_J_loss + kappa * w * w) * u)


def drive_law(d0, kappa, u, v):
    """The derived drive D = d0 + 2*kappa*u*v (Lean:
    CurvatureState.driveRate).  ``d0`` is the BASE-LOADING RATE, a
    separately measured process; the base curvature b is a state
    variable and never appears here.  A frozen source and frozen base
    (v = 0, d0 = 0) manufacture exactly zero."""
    return d0 + 2.0 * kappa * u * v


def curvature(b, kappa, a, g, J):
    """THE curvature: x = b + kappa * (a g J)^2.  A derived quantity
    of the state (b, a, g, J), never stored; every consumer computes
    it through this function, so the constraint x = b + kappa u^2 is
    definitional (Lean: CurvatureState.curv)."""
    u = a * g * J
    return b + kappa * u * u


def curvature_increment_exact(b, kappa, a, g, J, w, eta, d0, delta,
                              th_a_loss, th_g_loss, th_J_loss,
                              lam_wd, mu_b):
    """EXACT one-step increment of the derived curvature under the
    factor and base laws (each factor decayed at lam_wd, gated
    rebuild off), returned as (first_order, exact_increment):

      first_order = eta*(d0 + 2 kappa u v_loss)
                    - eta*(mu_b b + 2 p lam_wd kappa u^2),  p = 3

    with the O(eta^2) remainder carried inside exact_increment
    (Lean: CurvatureState.increment_exact).  The curvature law is a
    COROLLARY of the factor laws: this function steps the factors
    and differences the curvature; nothing else is added.

    DOMAIN (closed gate): the jet rebuild is off here, so with
    J <= 1 and a nonnegative decay factor the stepped jet satisfies
    j1 <= J <= 1 and the saturation clip of the transition is
    provably inert: on this domain the unclipped step IS the
    transition's step.  The law of the OPEN-gate clipped transition
    is ``curvature_increment_exact_gated``, which carries the
    clipping defect explicitly."""
    u = a * g * J
    v_loss = source_velocity_loss(a, g, J, w, delta, th_a_loss,
                                  th_g_loss, th_J_loss, kappa)
    a1 = (1.0 - eta * (th_a_loss + lam_wd)) * a + eta * delta
    g1 = (1.0 - eta * (th_g_loss + lam_wd)) * g
    j1 = (1.0 - eta * (th_J_loss + lam_wd + kappa * w * w)) * J
    b1 = b + eta * d0 - eta * mu_b * b
    exact = curvature(b1, kappa, a1, g1, j1) - curvature(
        b, kappa, a, g, J)
    first = (eta * (d0 + 2.0 * kappa * u * v_loss)
             - eta * (mu_b * b + 2.0 * 3.0 * lam_wd * kappa * u * u))
    return first, exact


def curvature_increment_exact_gated(b, kappa, a, g, J, w, eta, d0,
                                    delta, delta_J0, A, th_a_loss,
                                    th_g_loss, th_J_loss, lam_wd,
                                    mu_b):
    """EXACT one-step increment of the derived curvature under the
    CLIPPED open-gate transition (paper Thm curvlaw, clipped form):
    the jet step is the transition's own saturated rebuild,

        j1_unclipped = (1 - eta*(th_J + lam_wd + kappa*w^2))*J
                       + eta*delta_J0*A,
        chi          = max(j1_unclipped - 1, 0),
        j1           = j1_unclipped - chi = min(j1_unclipped, 1),

    with chi >= 0 the CLIPPING DEFECT, zero exactly when the clip
    is inactive.  Returns

        (first_order, exact_increment, chi, defect)

    where exact_increment is the increment of the clipped
    transition, defect = kappa*(a1*g1)^2 * (j1_unclipped^2 - j1^2)
    = kappa*(a1*g1)^2 * chi * (j1_unclipped + j1) >= 0 is the exact
    curvature contribution removed by the clip, and the identity

        exact_increment = exact_unclipped - defect

    holds to machine precision (tested on a binding clip).  The
    affine (unclipped) law is recovered exactly when chi = 0; the
    closure discharges that premise by certificate (drive ceiling
    on the captured branch, (N-sat) on the recurrent branch), never
    by assumption.  first_order is the same displayed coefficient
    as ``curvature_increment_exact`` plus the gated rebuild term
    eta*2*kappa*u*(a*g*delta_J0*A)."""
    u = a * g * J
    v_loss = source_velocity_loss(a, g, J, w, delta, th_a_loss,
                                  th_g_loss, th_J_loss, kappa)
    a1 = (1.0 - eta * (th_a_loss + lam_wd)) * a + eta * delta
    g1 = (1.0 - eta * (th_g_loss + lam_wd)) * g
    j1_unclipped = ((1.0 - eta * (th_J_loss + lam_wd
                                  + kappa * w * w)) * J
                    + eta * delta_J0 * A)
    chi = max(j1_unclipped - 1.0, 0.0)
    j1 = j1_unclipped - chi
    b1 = b + eta * d0 - eta * mu_b * b
    exact = curvature(b1, kappa, a1, g1, j1) - curvature(
        b, kappa, a, g, J)
    defect = kappa * (a1 * g1) ** 2 * chi * (j1_unclipped + j1)
    first = (eta * (d0 + 2.0 * kappa * u
                    * (v_loss + a * g * delta_J0 * A))
             - eta * (mu_b * b + 2.0 * 3.0 * lam_wd * kappa * u * u))
    return first, exact, chi, defect


def pure_decay_rate(p_i, lam_wd, kappa, u):
    """The pure-decay first-order curvature rate: with the loss
    rates, base, loading, rebuild, and carrier all zero and p_i
    decayed path factors, the manufactured curvature kappa u^2
    drains at exactly

        -2 p_i lam_wd kappa u^2

    per unit step (Lean: DecayLedger.pure_decay_rate).  The retired
    loading law returned twice this (the registered double-count
    witness; see retired.loading_step)."""
    return -2.0 * p_i * lam_wd * kappa * u * u


def clip_defect_preconditioned(pinv_list, g_list, clip_level):
    """The averaged clipping defect with the preconditioner INSIDE
    the average (paper Prop. windowdefect):

        eps_c = mean_s || P_s^{-1} (clip(g_s) - g_s) ||.

    Bounded when needed by (1/eps) * mean ||clip(g) - g|| since
    ||P^{-1}|| <= 1/eps.  The retired form (norm of the average
    before preconditioning) is ``clip_defect_naive``, kept only as
    the necessity witness for the ordering."""
    total = 0.0
    for pinv, g in zip(pinv_list, g_list):
        g = np.asarray(g, dtype=float)
        d = np.clip(g, -clip_level, clip_level) - g
        total += float(np.linalg.norm(np.asarray(pinv) * d))
    return total / len(g_list)


def clip_defect_naive(g_list, clip_level):
    """RETIRED defect form: the norm of the AVERAGE clipping defect,
    taken before preconditioning.  Averaging and variable
    preconditioning do not commute: on the two-step scalar window
    g = (-2, 2), clip level 1, inverse preconditioners (1, 2), this
    returns 0 while the true preconditioned defect is 1/2 (the
    registered counterexample; tested)."""
    acc = None
    for g in g_list:
        g = np.asarray(g, dtype=float)
        d = np.clip(g, -clip_level, clip_level) - g
        acc = d if acc is None else acc + d
    return float(np.linalg.norm(acc / len(g_list)))


# ----------------------------------------------------------------- D3


def z_cap_invariant(h, eps_max, b_sat, w_max2=0.0, z_seed=0.0):
    """The h-dependent DISCRETE invariant event power cap (paper Thm
    wellposed): with eps_max = X_max - c_- the largest admissible
    margin, K = eps_max/b_sat the carrying level, and the exact
    quadratic update z' = max(z*(1 + 2h*(eps - b_sat*z)), 0),

        Z_cap(h) = max(w_max^2, z_seed,
                       min((1 + 2h*eps_max)^2 / (8h*b_sat),
                           K * (1 + h*eps_max/2))),

    and [0, Z_cap] is invariant: from z <= K the one-step image is
    at most K*(1 + h*eps_max/2) (vertex bound of the increment on
    [0, K]); from z in (K, Z_cap] the update contracts; and the
    global vertex bound (1 + 2h*eps_max)^2/(8h*b_sat) caps the image
    everywhere.  Proved by maximizing the exact discrete update; no
    continuous-time monotonicity argument (Z_cap(h) -> K as h -> 0).
    The overshoot witness (z = 0.75, h = 1, b_sat = 1,
    eps_max = 0.9 -> z' = 0.97275084375, above the retired
    continuous-time cap 0.9 but below Z_cap(1) = 0.98) is pinned by
    a test."""
    if h <= 0.0 or eps_max <= 0.0 or b_sat <= 0.0:
        raise ValueError("need positive h, eps_max, b_sat")
    k_lvl = eps_max / b_sat
    cap_small = k_lvl * (1.0 + h * eps_max / 2.0)
    cap_vertex = (1.0 + 2.0 * h * eps_max) ** 2 / (8.0 * h * b_sat)
    return max(w_max2, z_seed, min(cap_small, cap_vertex))


def event_u_max(eta, th_J_loss, lam_wd, kappa, z_cap):
    """The clamp margin on the invariant box:
    u_max = eta*(th_J + lam_wd + kappa*Z_cap).  The step condition
    (S+) is u_max <= 1: the printed clamp in the event's jet factor
    never binds while the power stays inside the invariant cap, so
    every smooth identity holds on the box without case analysis.
    Raises when (S+) fails (the clamp branch must then be analyzed
    explicitly, and no enclosure below is licensed)."""
    if min(eta, kappa, z_cap) < 0.0 or th_J_loss + lam_wd < 0.0:
        raise ValueError("need nonnegative eta, kappa, z_cap and "
                         "th_J_loss + lam_wd")
    u = eta * (th_J_loss + lam_wd + kappa * z_cap)
    if u > 1.0:
        raise ValueError("(S+) violated: eta*(th_J + lam_wd + "
                         f"kappa*Z_cap) = {u:.6g} > 1")
    return u


def drain_linear(eta, th_J_loss, lam_wd, kappa, n_steps, sum_z):
    """The linear drain functional of a recorded event:

        D1 = 2*eta*((th_J + lam_wd)*N + kappa*sum_z),

    the first-order reading of the exact logarithmic drain law.  D1
    is an observable (N and sum_z are recorded); the exact drain
    -log(y_N/y_0) lies in ``drain_enclosure(D1, u_max)``."""
    return 2.0 * eta * ((th_J_loss + lam_wd) * n_steps
                        + kappa * sum_z)


def drain_enclosure(d_lin, u_max):
    """Two-sided enclosure of the exact log drain by its linear
    functional (paper Thm eventmap): with per-step
    u_n = eta*(th_J + lam_wd + kappa*z_n) <= u_max < 1 on the
    invariant box,

        D1 <= -log(y_N / y_0) <= D1 / (1 - u_max),

    from u <= -log(1 - u) <= u/(1 - u_max) summed over the event.
    The enclosure is a theorem, not a registered approximation: any
    recorded event outside the band falsifies the event map (E2's
    primary endpoint uses the exact identity; the enclosure is the
    preregistered residual band for the linear reading)."""
    if not 0.0 <= u_max < 1.0:
        raise ValueError("need 0 <= u_max < 1 ((S+) with margin)")
    if d_lin < 0.0:
        raise ValueError("need D1 >= 0")
    return d_lin, d_lin / (1.0 - u_max)


def retained_fraction_exact(b, c, y0, sum_log_phi):
    """The EXACT per-event retained fraction of the event clock
    (paper Thm eventmap): with entry jet curvature y_0 and the
    recorded per-step factors phi_n,

        x_N / c = (b + y_0 * exp(2 * sum_n log phi_n)) / c .

    An identity of the implemented recurrence with zero free
    parameters; ``run_event`` returns sum_log_phi, and the identity
    against its directly returned x_end holds at machine precision
    (tested)."""
    if c <= 0.0:
        raise ValueError("need c > 0")
    return (b + y0 * math.exp(2.0 * sum_log_phi)) / c


def strict_geometric_count(z0, target, rho):
    """The STRICT LEAST COUNT of geometric passage: the least
    K >= 0 with rho^K * z0 < target (paper Lem. strictcount).
    Every event exit gate in the model is STRICT (z < gate), so
    every closed-form count that certifies an exit must certify
    strict passage: the logarithmic ceiling alone accepts exact
    equality rho^K * z0 = target, which does NOT pass a strict
    gate, and the count must advance by one there (the equality
    bump).  The implementation computes the ceiling estimate and
    then settles the boundary by direct comparison in both
    directions, so exact equality and floating rounding at the
    boundary are both handled by the powers actually compared
    (Lean: TerminationUniform.power_below_of_decay_bounded
    consumes the strict hypothesis rho^K * z0 < target that this
    count guarantees)."""
    if not 0.0 < rho < 1.0:
        raise ValueError("need 0 < rho < 1 (a contraction factor)")
    if target <= 0.0:
        raise ValueError("need target > 0 (a strict gate level)")
    if z0 < 0.0:
        raise ValueError("need z0 >= 0")
    if z0 < target:
        return 0
    k = max(int(math.ceil(math.log(z0 / target)
                          / abs(math.log(rho)))), 0)
    while rho ** k * z0 >= target:
        k += 1
    while k > 0 and rho ** (k - 1) * z0 < target:
        k -= 1
    return k


def event_time_bound(y0, b, c, eta, th_J_loss, lam_wd, h, z_cap,
                     z_floor):
    """Finite event-length bound N1 + N2 of the event clock (paper
    Thm eventmap, termination clause): with theta' = th_J + lam_wd
    > 0, eta*theta' < 1, c > b, and eps_star = (c - b)/2, the jet
    curvature drains unconditionally at factor (1 - eta*theta')^2
    per step, so the margin is at or below -eps_star within

        N1 = ceil( log(y0/eps_star) / (2*|log(1 - eta*theta')|) )

    steps (N1 = 0 if y0 <= eps_star); if additionally
    2*h*eps_star < 1, the power then decays geometrically and the
    two-condition event end (z < z_floor AND eps <= 0) is reached
    within

        N2 = the least K with (1 - 2*h*eps_star)^K * Z_cap < z_floor

    further steps (``strict_geometric_count``: the exit gate is
    strict, so the count certifies strict passage, advancing by one
    at exact logarithmic equality).  Every event satisfies
    N <= N1 + N2 (tested against ``run_event``)."""
    thp = th_J_loss + lam_wd
    if thp <= 0.0 or not 0.0 < eta * thp < 1.0:
        raise ValueError("need th_J + lam_wd > 0 and "
                         "0 < eta*(th_J + lam_wd) < 1")
    if not c > b:
        raise ValueError("need c > b (the event's exit room)")
    eps_star = (c - b) / 2.0
    if not 2.0 * h * eps_star < 1.0:
        raise ValueError("need 2*h*eps_star < 1 (power decay step)")
    if not 0.0 < z_floor <= z_cap:
        raise ValueError("need 0 < z_floor <= z_cap")
    if y0 <= eps_star:
        n1 = 0
    else:
        n1 = math.ceil(math.log(y0 / eps_star)
                       / (2.0 * abs(math.log(1.0 - eta * thp))))
    n2 = strict_geometric_count(z_cap, z_floor,
                                1.0 - 2.0 * h * eps_star)
    return n1 + n2


def reset_interval_licensed(y0, b, c, eta, th_J_loss, lam_wd, kappa,
                            z_cap, n_max):
    """The LICENSED reset interval of the event clock (paper Thm
    eventmap, reset clause): with phi_min = 1 - eta*(th_J + lam_wd +
    kappa*Z_cap) >= 0 under (S+) and N <= n_max (the event-length
    bound ``event_time_bound``), the retained fraction at the event
    end lies in

        [ (b + y0 * phi_min^(2*n_max)) / c ,  1 ] :

    the upper endpoint is the two-condition end convention
    (eps_end <= 0), the lower endpoint is the per-step factor floor
    run for at most n_max steps.  No separation condition is
    needed."""
    if c <= 0.0 or y0 < 0.0 or b < 0.0:
        raise ValueError("need c > 0, y0 >= 0, b >= 0")
    u = event_u_max(eta, th_J_loss, lam_wd, kappa, z_cap)
    phi_min = 1.0 - u
    r_lo = (b + y0 * phi_min ** (2 * n_max)) / c
    if r_lo > 1.0:
        raise ValueError("inconsistent entry: b + y0*phi_min^(2N) "
                         "> c means the state could not have "
                         "toppled to this event")
    return r_lo, 1.0


def rho_w_derived(z_floor, z_min):
    """The derived carrier event POWER factor: with the event-end
    convention (E-end) ending every burst at the first step with
    z < z_floor and the seed floor (E-seed) z0 >= z_min, the
    post-event carrier power satisfies z_N/z0 <= z_floor/z_min < 1
    (Lean: StrictBurst.carrier_contraction).  A POWER ratio; the
    factor the theorems license on the signed carrier AMPLITUDE is
    its square root, ``amp_factor`` (unit identity z = w^2,
    registered)."""
    if not 0.0 < z_floor < z_min:
        raise ValueError("need 0 < z_floor < z_min")
    return z_floor / z_min


def amp_factor(rho_w):
    """The carrier AMPLITUDE event factor sqrt(rho_w): z = w^2 and
    z_N/z0 < rho_w give |w_N|/|w_entry| < sqrt(rho_w) (Lean:
    EventMap.amp_contraction).  The reference value is the SEEDED
    entry amplitude |w_entry| = |w_tilde| of ``seeded_entry`` (for
    which z_0 = w_entry^2 holds exactly), never the stored
    pre-event amplitude when the seed floor binds: with
    w_pre = 0 and z_seed > 0 the event intentionally creates
    amplitude (bounded by the seed and the truncation), and no
    contraction relative to |w_pre| is claimed.  The map applies
    this factor, never rho_w itself, to the signed amplitude."""
    if not 0.0 < rho_w < 1.0:
        raise ValueError("need 0 < rho_w < 1")
    return math.sqrt(rho_w)


B_GAP_LATEX = (r"c_- \ge \frac{d_0^{\sup}}{\mu_b}"
               r" + \frac{A_{\max}}{\eta_-\,\mu_b} + \delta_b")


def b_gap_slack(c_min, d0_sup, mu_b, a_max, eta_min, delta_b):
    """The NORMATIVE entry-gap parameter inequality (B-gap) of the
    totality theorem (paper Lem. bgap), the PER-STEP arrival
    envelope: the entry gap b^+ <= c - delta_b holds at every site
    and time whenever

        c_min >= d0_sup/mu_b + A_max/(eta_min*mu_b) + delta_b ,

    with c_min the threshold floor, d0_sup/mu_b the arrival-free
    base ceiling of the affine base reversion, and
    A_max/(eta_min*mu_b) the amplification of the PER-STEP arrival
    ceiling A_max through the base ceiling B_max (an arrival may
    land at every step; no timing hypothesis).  This is the
    box-tier inequality: it is what the well-posedness base
    invariant consumes, and it is the single source of the printed
    display (``B_GAP_LATEX``; the generator serializes exactly this
    function and this string).  Returns the signed slack (the
    condition holds iff the slack is nonnegative).  On the gap
    domain the sharper event termination bound ``event_time_bound``
    applies with c - b >= delta_b, the generalized exit reduces
    bit-exactly to the two-condition convention when
    y_floor <= delta_b, and the resolver's Exhausted branch is
    unreachable for sufficient fuel.  The per-cycle form under
    separated cycles is the COROLLARY-tier ``b_gap_slack_tsep``
    and is never this gate."""
    if mu_b <= 0.0:
        raise ValueError("need mu_b > 0 (the base must revert for "
                         "a finite base ceiling)")
    if delta_b <= 0.0:
        raise ValueError("need delta_b > 0")
    if a_max < 0.0:
        raise ValueError("need A_max >= 0")
    if a_max > 0.0 and eta_min <= 0.0:
        raise ValueError("need eta_min > 0 with nonzero arrivals "
                         "(the ceiling absorbs an arrival per step "
                         "only at a positive rate floor)")
    amp = a_max / (eta_min * mu_b) if a_max > 0.0 else 0.0
    return c_min - (d0_sup / mu_b + amp + delta_b)


def b_gap_slack_tsep(c_min, d0_sup, mu_b, arrivals_cyc_max,
                     eta_star, delta_b):
    """The PER-CYCLE entry-gap slack under SEPARATED CYCLES
    (T-sep): a corollary-tier sharpening of ``b_gap_slack``, valid
    only when at most one arrival batch lands per burst-reload
    cycle and the base reverts between batches,

        c_min >= d0_sup/mu_b + A_cyc_max + eta_star*d0_sup
                 + delta_b .

    NEVER the theorem gate: the box-tier inequality is
    ``b_gap_slack`` (the per-step envelope).  Reported separately
    by the generator under its own (T-sep) label."""
    if mu_b <= 0.0:
        raise ValueError("need mu_b > 0 (the base must revert for "
                         "a finite base ceiling)")
    if delta_b <= 0.0:
        raise ValueError("need delta_b > 0")
    return c_min - (d0_sup / mu_b + arrivals_cyc_max
                    + eta_star * d0_sup + delta_b)


def event_fuel_required(y0_max, delta_b, eta, th_J_loss, lam_wd, h,
                        z_cap, z_floor):
    """The GRANTED FUEL sufficient for the resolver on the
    entry-gap domain (paper Thm eventterm): under (B-gap) every
    entry has c - b >= delta_b, so the two-stage bound
    ``event_time_bound`` applies with the gap constant,
    eps_star = delta_b/2, uniformly over entries, and the exit
    fires at some step index n <= N.  STRICT-INDEXING CONVENTION:
    the resolver checks the exit rule at the top of each iteration
    and its loop runs while n < fuel, so reaching the check at
    index N requires fuel >= N + 1; this function returns the
    GRANTED value N + 1 (the fuel the totality theorem cites),
    while the ``event_time_bound*`` family returns the step-count
    bound N itself.  Fuel at or above this value makes the
    Exhausted branch unreachable on the domain.  The WHOLE-BOX
    granted fuel, covering entries off the gap domain as well, is
    ``event_fuel_required_uniform``."""
    if delta_b <= 0.0:
        raise ValueError("need delta_b > 0")
    return event_time_bound(y0_max, 0.0, delta_b, eta, th_J_loss,
                            lam_wd, h, z_cap, z_floor) + 1


def event_time_bound_overloaded(y0, b, c, eta, th_J_loss, lam_wd,
                                h, b_sat, z_cap, y_floor, tau_z):
    """Finite event-length bound for the generalized exit on the
    OVERLOADED-BASE branch b > c (paper Lem. overload, inner
    clauses; a bound on ONE inner event, never on the global
    consecutive-topple run, which the overload classification
    theorem covers): with theta' = th_J + lam_wd > 0,
    eta*theta' < 1, jet-exhaustion floor y_floor > 0, and power
    tolerance tau_z > y_floor/b_sat, the jet curvature reaches
    y <= y_floor within

        N1' = ceil( log(y0/y_floor) / (2*|log(1 - eta*theta')|) )

    steps (0 if y0 <= y_floor); thereafter the margin lies in
    [b - c, b - c + y_floor], so while the power sits at or above
    the exit target Z_tgt = (b - c)/b_sat + tau_z it exceeds the
    current equilibrium eps/b_sat by at least
    gamma = tau_z - y_floor/b_sat > 0 and the exact update
    contracts it by the factor 1 - 2*h*b_sat*gamma
    = 1 - 2*h*(b_sat*tau_z - y_floor) per step; the exit clause
    z < eps/b_sat + tau_z is STRICT, so the stage-two count is the
    strict least count

        N3 = the least K with rho^K * Z_cap < Z_tgt,
        rho = 1 - 2*h*(b_sat*tau_z - y_floor)

    (``strict_geometric_count``: zero only when Z_cap < Z_tgt
    strictly; exact equality Z_cap = Z_tgt needs one step and the
    count advances by one at exact logarithmic equality).
    Requires 2*h*(b_sat*tau_z - y_floor) < 1.  Together with
    ``event_time_bound`` on the gap domain this makes the
    generalized exit total on the whole box."""
    thp = th_J_loss + lam_wd
    if thp <= 0.0 or not 0.0 < eta * thp < 1.0:
        raise ValueError("need th_J + lam_wd > 0 and "
                         "0 < eta*(th_J + lam_wd) < 1")
    if not b > c:
        raise ValueError("overloaded branch requires b > c (the "
                         "gap-domain bound is event_time_bound)")
    if y_floor <= 0.0 or b_sat <= 0.0:
        raise ValueError("need y_floor > 0 and b_sat > 0")
    if not tau_z > y_floor / b_sat:
        raise ValueError("need tau_z > y_floor/b_sat (the "
                         "tolerance must absorb the equilibrium "
                         "drift)")
    gamma_z = b_sat * tau_z - y_floor
    if not 2.0 * h * gamma_z < 1.0:
        raise ValueError("need 2*h*(b_sat*tau_z - y_floor) < 1 "
                         "(power step on the overloaded branch)")
    if y0 <= y_floor:
        n1 = 0
    else:
        n1 = math.ceil(math.log(y0 / y_floor)
                       / (2.0 * abs(math.log(1.0 - eta * thp))))
    z_tgt = (b - c) / b_sat + tau_z
    n3 = strict_geometric_count(z_cap, z_tgt,
                                1.0 - 2.0 * h * gamma_z)
    return n1 + n3


def event_time_bound_overloaded_uniform(y0_max, eta, th_J_loss,
                                        lam_wd, h, b_sat, z_cap,
                                        y_floor, tau_z):
    """The UNIFORM overloaded-branch event-length bound over the
    whole overloaded region b in (c, b_max] (paper Thm totality,
    overloaded constituent): the per-entry bound
    ``event_time_bound_overloaded`` uses the exit target
    Z_tgt(b) = (b - c)/b_sat + tau_z, which INCREASES with the
    overload b - c, and a larger target is easier to fall below;
    the hardest overloaded entry is therefore the limit b -> c+,
    where the target attains its infimum tau_z.  The uniform bound
    substitutes that worst target with the strict least count,

        N3 = the least K with rho^K * Z_cap < tau_z,
        rho = 1 - 2*h*(b_sat*tau_z - y_floor),

    together with the box supremum y0_max of the initial jet
    curvature in the stage-one constituent N1'.  Monotonicity is
    displayed, not assumed: for every b > c,
    event_time_bound_overloaded(y0, b, c, ...) <= this value when
    y0 <= y0_max (tested, including at b = c + 1e-6)."""
    thp = th_J_loss + lam_wd
    if thp <= 0.0 or not 0.0 < eta * thp < 1.0:
        raise ValueError("need th_J + lam_wd > 0 and "
                         "0 < eta*(th_J + lam_wd) < 1")
    if y_floor <= 0.0 or b_sat <= 0.0:
        raise ValueError("need y_floor > 0 and b_sat > 0")
    if not tau_z > y_floor / b_sat:
        raise ValueError("need tau_z > y_floor/b_sat (the "
                         "tolerance must absorb the equilibrium "
                         "drift)")
    gamma_z = b_sat * tau_z - y_floor
    if not 2.0 * h * gamma_z < 1.0:
        raise ValueError("need 2*h*(b_sat*tau_z - y_floor) < 1 "
                         "(power step on the overloaded branch)")
    if y0_max <= y_floor:
        n1 = 0
    else:
        n1 = math.ceil(math.log(y0_max / y_floor)
                       / (2.0 * abs(math.log(1.0 - eta * thp))))
    n3 = strict_geometric_count(z_cap, tau_z,
                                1.0 - 2.0 * h * gamma_z)
    return n1 + n3


def event_time_bound_middle(y0, eta, th_J_loss, lam_wd, h, b_sat,
                            z_cap, z_floor, y_floor):
    """Finite event-length bound for the generalized exit on the
    WHOLE region b <= c, including the middle region
    c - delta_b < b <= c and the equality boundary b = c (paper
    Lem. midterm), with NO entry-gap hypothesis: with
    theta' = th_J + lam_wd > 0, eta*theta' < 1, jet-exhaustion
    floor y_floor > 0, and h*b_sat*z_floor < 1.

    Stage one (region-free jet exhaustion): every clock factor
    obeys phi <= 1 - eta*theta' with no condition on the power, so
    the jet curvature reaches the target level
    y_star = min(y_floor, b_sat*z_floor/2) within

        N_y = ceil( log(y0/y_star) / (2*|log(1 - eta*theta')|) )

    steps (0 if y0 <= y_star).  From then on the first exit
    conjunct holds at every step (y <= y_floor), and the margin
    obeys eps = (b - c) + y <= y_star <= b_sat*z_floor/2 on
    b <= c.

    Stage two (power passage at the worst-case gate): while
    z >= z_floor the exact power update contracts by the factor
    1 - h*b_sat*z_floor per step (eps - b_sat*z <=
    b_sat*z_floor/2 - b_sat*z_floor), so z crosses the worst-case
    gate z_floor within

        N_z = the least K with
              (1 - h*b_sat*z_floor)^K * z_cap < z_floor

    further steps (``strict_geometric_count``: the exit gate is
    strict); the actual gate is z_floor when eps <= 0 and at
    least tau_z >= z_floor otherwise, so the exit fires no later.
    Every event entered at b <= c ends within N_y + N_z (tested
    against ``run_event``)."""
    thp = th_J_loss + lam_wd
    if thp <= 0.0 or not 0.0 < eta * thp < 1.0:
        raise ValueError("need th_J + lam_wd > 0 and "
                         "0 < eta*(th_J + lam_wd) < 1")
    if y_floor <= 0.0 or b_sat <= 0.0:
        raise ValueError("need y_floor > 0 and b_sat > 0 (the "
                         "middle region terminates only under the "
                         "generalized exit)")
    if not 0.0 < z_floor <= z_cap:
        raise ValueError("need 0 < z_floor <= z_cap")
    if not h * b_sat * z_floor < 1.0:
        raise ValueError("need h*b_sat*z_floor < 1 (power step on "
                         "the middle region)")
    y_star = min(y_floor, b_sat * z_floor / 2.0)
    if y0 <= y_star:
        n_y = 0
    else:
        n_y = math.ceil(math.log(y0 / y_star)
                        / (2.0 * abs(math.log(1.0 - eta * thp))))
    n_z = strict_geometric_count(z_cap, z_floor,
                                 1.0 - h * b_sat * z_floor)
    return n_y + n_z


def event_fuel_required_uniform(y0_max, b_max, c_min, eta,
                                th_J_loss, lam_wd, h, b_sat, z_cap,
                                z_floor, y_floor, tau_z):
    """The uniform GRANTED FUEL of the totality theorem (paper Thm
    totality): N* + 1 with N* the maximum of the region bounds,
    every state argument at its box supremum,

        b <= c:  event_time_bound_middle at y0_max (no gap
                 hypothesis; covers the entry-gap domain, the
                 middle region, and the equality boundary b = c),
        b >  c:  event_time_bound_overloaded_uniform  (the worst
                 exit target is the INFIMUM tau_z, attained in the
                 limit b -> c+; the target grows with the overload
                 and a larger target is easier to fall below, so
                 the hardest overloaded entry is the shallowest),

    so every event entered from the state box ends within N* clock
    steps and the resolver's Exhausted branch is unreachable at
    any fuel at or above the granted value: the composed settled
    step at such fuel is a self-map of the box (the no-rejection
    theorem).  The +1 is the strict-indexing convention
    (``event_fuel_required``); the generator emits and gates this
    granted value.  When the box has b_max <= c_min the overloaded
    region of the box is EMPTY and its constituent is zero; the
    generator prints that fact rather than leaving the branch
    silent.  Requires strict relaxation th_J + lam_wd > 0 and the
    generalized-exit constants y_floor > 0,
    tau_z > y_floor/b_sat."""
    n_mid = event_time_bound_middle(y0_max, eta, th_J_loss, lam_wd,
                                    h, b_sat, z_cap, z_floor,
                                    y_floor)
    if b_max > c_min:
        n_over = event_time_bound_overloaded_uniform(
            y0_max, eta, th_J_loss, lam_wd, h, b_sat,
            z_cap, y_floor, tau_z)
    else:
        n_over = 0
    return max(n_mid, n_over) + 1


def rebuilt_jet(J_out, w, eta, th_J_loss, lam_wd, kappa, delta_J0):
    """The ONE-GAP REBUILT JET (paper Def. rebuiltjet): the jet the
    guard sees one global step after an ended event whose reload
    gate is open,

        J_reb = min( phi * J_out + eta*delta_J0, 1 ),
        phi   = max(1 - eta*(th_J + lam_wd + kappa*w^2), 0),

    with J_out the post-event jet and w the settled carrier
    (|w| <= sqrt(z_floor) on every ended event).  Under one-gap
    reload saturation, eta*delta_J0 >= 1, this equals 1 for every
    J_out >= 0: the clipped rebuild restores the full jet
    contribution before every guard, which is why no proof that
    tracks the base alone can close the guard (the driven
    persistence theorem)."""
    phi = max(1.0 - eta * (th_J_loss + lam_wd + kappa * w * w), 0.0)
    return min(phi * J_out + eta * delta_J0, 1.0)


def driven_load_margin(d0, mu_b, c_plus):
    """The (O-load) SUSTAINED BASE DRIVE margin (paper Thm.
    overloadclass, driven clause, base-driven certificate):
    returns

        d0/mu_b - c_plus,

    with c_plus an upper band edge of the threshold.  When the
    margin is positive and the entry base is overloaded, the base
    equilibrium sits above the threshold band, so b stays above c
    at every global step and the guard x = b + kappa*(a g J)^2 > c
    fires at EVERY global step for all time regardless of the jet:
    the site is in the driven regime with all-hot exits, and every
    logged post-event jet obeys the exhaustion bound
    J_out <= ``driven_sampled_jet_max``.  When that bound is below
    J_crit the driven site satisfies the measured phase criterion
    at every sample: collapse by drive."""
    if mu_b <= 0.0:
        raise ValueError("need mu_b > 0")
    return d0 / mu_b - c_plus


def driven_rebuild_margin(eta, delta_J0, kappa, a_inf, g_inf,
                          c_plus, th_J_loss, lam_wd):
    """The (O-reb) REBUILD DRIVE margin (paper Thm. overloadclass,
    driven clause, rebuild-driven certificate): requires one-gap
    reload saturation eta*delta_J0 >= 1 and strict jet relaxation
    th_J + lam_wd > 0 (so every event drains at least one clock
    step and writes reload duration >= 1, keeping the gate open),
    and returns

        kappa*(a_inf * g_inf)^2 - c_plus,

    with a_inf, g_inf retention floors of the amplitude and chain
    factors along the run (hypothesis G-ret; sufficient: the
    amplitude fixed point delta_a/(th_a + lam_wd) at or above
    a_inf, and th_g = lam_wd = 0 for the chain).  When the margin
    is positive, the rebuilt guard exceeds the threshold band
    REGARDLESS of the base: J_reb = 1 at every guard
    (``rebuilt_jet`` under saturation), so
    x = b + kappa*(a g)^2 > c_plus >= c fires at every global step
    for all time even as b -> d0/mu_b below the band.  The
    persistent-overload witness orbit (eta*delta_J0 = 2,
    kappa*(a g)^2 = 1 > c = 0.5) is the canonical instance and a
    pinned regression."""
    if not eta * delta_J0 >= 1.0:
        raise ValueError("(O-reb) requires one-gap reload "
                         "saturation eta*delta_J0 >= 1")
    if not th_J_loss + lam_wd > 0.0:
        raise ValueError("(O-reb) requires strict jet relaxation "
                         "th_J + lam_wd > 0")
    return kappa * (a_inf * g_inf) ** 2 - c_plus


def driven_sampled_jet_max(y_floor, kappa, a_inf, g_inf):
    """The DRIVEN-REGIME SAMPLED JET CEILING J_drv (paper Thm.
    overloadclass, collapse-by-drive verdict): on an all-hot
    driven run every exit fires through jet exhaustion
    y <= y_floor, so every logged post-event jet obeys

        J_out <= sqrt(y_floor / kappa) / (a_inf * g_inf) = J_drv.

    When J_drv < J_crit, every logged sample satisfies the phase
    criterion, hence so does every window average: the driven
    regime is a collapse regime of the measured criterion
    (collapse by drive).  The bound applies to HOT exits (the
    (O-load) branch, where the margin stays positive); on the
    (O-reb) branch with a subthreshold base the late exits are
    cold and the logged jet is bounded by the cold-exit envelope
    instead, the bimodal-sampling prediction of E5."""
    if kappa <= 0.0 or a_inf <= 0.0 or g_inf <= 0.0:
        raise ValueError("need kappa, a_inf, g_inf > 0")
    if y_floor < 0.0:
        raise ValueError("need y_floor >= 0")
    return math.sqrt(y_floor / kappa) / (a_inf * g_inf)


def overload_headroom_margin(kappa, a_sup, g_sup, d0_sup, mu_b,
                             c_minus):
    """The (O-head) REBUILD HEADROOM margin (paper Thm.
    overloadres): returns

        c_minus - d0_sup/mu_b - kappa*(a_sup * g_sup)^2,

    the room under the lower threshold band edge left by the
    relaxed base equilibrium b_inf = d0_sup/mu_b PLUS the
    saturated-jet drive ceiling kappa*(a_sup*g_sup)^2 (the jet is
    clipped at 1, so this ceiling covers every rebuilt jet,
    however many reload steps accumulate).  (O-head) holds with
    gap delta_b when the margin is at least delta_b > 0; then the
    guard is closed at every state with b <= b_inf + delta_b/2 and
    stays closed forever (``overload_resolution_bound``)."""
    if mu_b <= 0.0:
        raise ValueError("need mu_b > 0")
    return c_minus - d0_sup / mu_b - kappa * (a_sup * g_sup) ** 2


def overload_resolution_bound(b0, d0_sup, mu_b, eta, kappa, a_sup,
                              g_sup, c_minus, delta_b):
    """The (O-head) RESOLUTION COUNT N_res (paper Thm.
    overloadres): under the headroom certificate
    ``overload_headroom_margin`` >= delta_b > 0, a maximal
    consecutive-topple run from an overloaded entry at base b0
    ends within

        N_res = ceil( log((b0 - b_inf)/(delta_b/2))
                      / |log(1 - eta*mu_b)| )

    global steps, b_inf = d0_sup/mu_b (0 if b0 is already at or
    below b_inf + delta_b/2): the arrival-free base contracts
    affinely toward b_inf at factor 1 - eta*mu_b per global step
    (events freeze the base under T-sep), and once
    b <= b_inf + delta_b/2 the full rebuilt guard obeys

        x = b + kappa*(a g J)^2
          <= b_inf + delta_b/2 + kappa*(a_sup g_sup)^2
          <= c_minus - delta_b/2 < c ,

    so the guard is closed, the sublevel set {b <= b_inf +
    delta_b/2} is forward invariant, no event ever fires again,
    the remaining reload completes, the gate closes, and the jet
    decays monotonically: the orbit enters the entry-gap domain
    and the conditional regime theorems apply.  The proof tracks
    the coupled (b, J, act) state: base by the affine contraction,
    jet by the saturated ceiling, gate by the reload-counter
    induction; a proof that follows the base alone cannot close
    the guard (the driven witness)."""
    if delta_b <= 0.0:
        raise ValueError("need delta_b > 0")
    if not 0.0 < eta * mu_b < 1.0:
        raise ValueError("need 0 < eta*mu_b < 1")
    margin = overload_headroom_margin(kappa, a_sup, g_sup, d0_sup,
                                      mu_b, c_minus)
    if not margin >= delta_b:
        raise ValueError(
            "(O-head) fails: headroom margin "
            f"{margin:.6g} < delta_b = {delta_b:.6g} (the "
            "saturated-jet drive does not fit under the relaxed "
            "base; the run is not certified to resolve)")
    b_inf = d0_sup / mu_b
    if b0 <= b_inf + delta_b / 2.0:
        return 0
    return int(math.ceil(math.log((b0 - b_inf) / (delta_b / 2.0))
                         / abs(math.log(1.0 - eta * mu_b))))


def retention_floor_two_phase(eps_max, delta_b, eta, th_J_loss,
                              lam_wd, kappa, h, z_cap, z_floor):
    """The TWO-PHASE per-event jet retention floor on the entry-gap
    domain under the two-condition exit (paper Lem. retention): a
    uniform lower bound on J_end/J_entry, independent of the event
    length, used by the THEOREM tier of the maintenance
    certificates.  Write g = c - b^+ >= delta_b for the entry gap,
    phi_min = 1 - eta*(th_J + lam_wd + kappa*z_cap) for the worst
    clock factor, and theta' = th_J + lam_wd.

    Phase A (margin positive).  At the first step n1 with
    eps <= 0, the previous step had y > g, so one clock factor
    gives y_{n1} >= phi_min^2 * g; more generally at the first
    step n2 with y <= g/2 the same one-step overshoot gives
    y_{n2} >= phi_min^2 * (g/2).

    Phase B (tail at negative margin).  For n >= n2 the margin
    obeys -eps_n = g - y_n >= g/2, so while z >= z_floor the power
    contracts by at least 1 - h*g per step, the tail is at most

        N_2 = the least K with (1 - h*g)^K * z_cap < z_floor

    steps, and the telescoped power update bounds the tail power
    mass: sum z_n <= z_cap/(h*g).  The exact log-drain law with
    log(1-u) >= -u/(1-u_max) then bounds the tail drain, giving

        J_end/J_entry >= phi_min * sqrt( (g/(2*y0)) *
            exp( -(2*eta/phi_min) *
                 (theta'*N_2 + kappa*z_cap/(h*g)) ) ).

    The entry jet at gap g obeys the EXACT identity
    y0 = eps_0 + g <= eps_max + g (the entry margin is capped by
    the invariant margin ceiling), so the paired ratio
    g/(2*y0) >= g/(2*(eps_max + g)); every factor is then monotone
    increasing in g, and the bound is evaluated at the worst gap
    g = delta_b:

        r >= phi_min * sqrt( (delta_b * D)
                             / (2*(eps_max + delta_b)) ),
        D = exp( -(2*eta/phi_min) *
                 (theta'*N_2 + kappa*z_cap/(h*delta_b)) ),

    uniformly over the domain (tested against ``run_event`` over
    an entry grid).  If the event ends earlier on any branch the
    bound holds a fortiori."""
    thp = th_J_loss + lam_wd
    if thp <= 0.0:
        raise ValueError("need th_J + lam_wd > 0")
    if delta_b <= 0.0 or eps_max <= 0.0:
        raise ValueError("need delta_b > 0 and eps_max > 0")
    if not 0.0 < z_floor <= z_cap:
        raise ValueError("need 0 < z_floor <= z_cap")
    if not h * delta_b < 1.0:
        raise ValueError("need h*delta_b < 1 (power step at the "
                         "gap constant)")
    u_max = eta * (thp + kappa * z_cap)
    if not u_max < 1.0:
        raise ValueError("need eta*(th_J + lam_wd + kappa*Z_cap) "
                         "< 1 (S+)")
    phi_min = 1.0 - u_max
    n2 = strict_geometric_count(z_cap, z_floor,
                                1.0 - h * delta_b)
    drain = math.exp(-(2.0 * eta / phi_min)
                     * (thp * n2 + kappa * z_cap / (h * delta_b)))
    return phi_min * math.sqrt(
        delta_b * drain / (2.0 * (eps_max + delta_b)))


class EventNotEnded(RuntimeError):
    """A fuel-indexed event resolution returned without reaching its
    exit rule (the Exhausted branch of the tagged resolver).  The
    global map REJECTS this branch: ``gated_model_step`` raises it
    rather than consuming a fuel-truncated state as an ordinary
    step.  On the entry-gap domain (B-gap) the branch is unreachable
    for fuel at or above ``event_time_bound``/``event_fuel_required``
    (a theorem); reaching it therefore witnesses an entry outside
    the declared domain or insufficient fuel, never a valid step."""


def seeded_entry(w_pre, z_seed, xi=0.0, w_kick_max=None):
    """The SEEDED ENTRY OPERATOR applied to the stored carrier at
    event entry (paper Def. eventmap, entry clause): with kick
    realization xi (drawn by the caller from the registered law,
    Normal(0, z_seed); xi = 0 in deterministic evaluation),

        w_kick  = clip(w_pre + xi, [-w_kick_max, w_kick_max]),
        z_0     = max(z_seed, w_kick^2),
        w_tilde = s * sqrt(z_0),

    where the sign law is s = sgn(w_kick), with the measure-zero
    tie sgn(0) := +1 (immaterial under the continuous kick law;
    stated for definiteness).  The event clock is initialized with
    z_0 = w_tilde^2 EXACTLY, restoring the identity z = w^2 at
    entry; every amplitude statement of the event is made relative
    to |w_tilde| (the seeded amplitude), never to |w_pre| when the
    seed floor binds.  Returns (w_tilde, z0)."""
    w_kick = w_pre + xi
    if w_kick_max is not None:
        w_kick = min(max(w_kick, -w_kick_max), w_kick_max)
    z0 = max(z_seed, w_kick * w_kick)
    if w_kick > 0.0:
        s = 1.0
    elif w_kick < 0.0:
        s = -1.0
    else:
        s = 1.0
    return s * math.sqrt(z0), z0


def run_event(b, a, g, J, w, c, eta, kappa, th_J_loss, lam_wd,
              burst, fuel, xi=0.0):
    """Resolve ONE burst event at a toppling site: THE event clock,
    the single implementation consumed by the map, the theorem
    constants, the protocol generator, and the endpoints (paper
    Def. eventmap).  This is the FUEL-INDEXED RESOLVER: the fuel
    F is DECLARED MODEL DATA, a required argument with no hidden
    default (the generator instantiates the printed granted value;
    ``gated_model_step`` threads its own ``fuel`` argument through
    unchanged).  The resolver is total by construction and returns
    a TAGGED result, ``status`` equal to "ended" (the exit rule
    fired at inner time n <= fuel) or "exhausted" (the fuel bound
    was reached first).  The global map rejects the exhausted
    branch (``gated_model_step`` raises ``EventNotEnded``); on the
    entry-gap domain (B-gap), b <= c - delta_b, the branch is
    unreachable for fuel at or above the termination bound
    (``event_time_bound``), and on the whole box for fuel at or
    above the granted value ``event_fuel_required_uniform``.

    ENTRY: the stored carrier passes through the seeded entry
    operator ``seeded_entry`` (kick realization ``xi``, truncation
    ``burst["w_kick_max"]`` if present), which sets the seeded
    carrier w_tilde and z_0 = w_tilde^2 exactly.  The returned
    ``w_entry`` is w_tilde; amplitude statements compare |w_end|
    with |w_entry|.

    The base, amplitude, and chain factor are frozen for the event
    (timescale hypothesis T-sep); per fast-clock step the jet
    contracts multiplicatively under the oscillation work term,

        phi_n = max(1 - eta*(th_J + lam_wd + kappa*z_n), 0),
        J_{n+1} = phi_n * J_n,

    the margin is DERIVED from the factors, eps_n = x_n - c with
    x_n = b + kappa*(a g J_n)^2, and the power follows the logistic
    update z_{n+1} = max(z_n*(1 + 2h*(eps_{n+1} - b_sat*z_n)), 0).
    With jet curvature y_n = kappa*(a g J_n)^2 the drain law is
    EXACT in unconditional product form,

        y_N = y_0 * prod_{n<N} phi_n^2,

    valid for EVERY legal event including zero-entry events
    (y_0 = 0, a legal overloaded entry at J = 0); on the positive
    domain y_0 > 0 with positive factors it has the logarithmic
    form

        log(y_N / y_0) = 2 * sum_{n<N} log(phi_n),

    (returned as sum_log_phi; identity tested at machine
    precision), and there the linear functional
    D1 = ``drain_linear`` encloses it (``drain_enclosure``).
    Zero-entry events are OUTSIDE the logarithmic endpoint and are
    reported as their own event class (E2).  There is no linear margin recurrence and
    no drain coefficient a_b: the exact per-step increment is
    eps_{n+1} - eps_n = -(1 - phi_n^2) * y_n.

    EXIT (E-end, generalized): the event ends at the first step
    with

        (eps <= 0  OR  y <= y_floor)  AND  z < z_gate,
        z_gate = z_floor                            if eps <= 0,
                 max(z_floor, eps/b_sat + tau_z)    if eps > 0,

    with y_floor = burst.get("y_floor", 0) and
    tau_z = burst.get("tau_z", 0).  On the entry-gap domain
    (b <= c - delta_b with y_floor <= delta_b) this REDUCES
    BIT-EXACTLY to the two-condition convention (first step with
    z < z_floor AND eps <= 0, exit margin in [b - c, 0]): there
    y <= y_floor implies eps <= 0, so the first conjunct collapses
    to eps <= 0 and the power gate reads z < z_floor (tested).
    With y_floor > 0 the exit is additionally total on the
    OVERLOADED-BASE branch b >= c (where eps >= b - c > 0 for
    every finite time and the power equilibrates near eps/b_sat,
    so the two-condition exit cannot fire): the site exits hot
    with the residual excess base-carried.  Whether the guard
    STOPS re-firing on the global clock is a property of the
    coupled (b, J, act) feedback, NEVER of the base alone: under
    the headroom certificate (O-head) the consecutive-topple run
    resolves within ``overload_resolution_bound`` global steps,
    while under either sustained-drive certificate, (O-load) or
    (O-reb), the site topples at EVERY global step for all time,
    the DRIVEN regime of the classification theorem (paper Thm.
    overloadclass; the base relaxes exactly as before in that
    regime and the relaxation is irrelevant to the full guard,
    which reads x = b + kappa*(a g J)^2 after the gated rebuild).
    Termination of ONE inner event within ``event_time_bound``
    steps on the gap domain (and ``event_time_bound_overloaded``
    off it) is a theorem; it never implies the global run ends.

    SETTLEMENT (E-settle, terminal carrier settlement): every ENDED
    event emits its carrier through the terminal settlement
    operator, the exit-side counterpart of the entry truncation,

        z_settled = min(z_exit, z_floor),
        w_end     = sign * sqrt(z_settled),
        d_set     = z_exit - z_settled >= 0,

    so |w_end| <= sqrt(z_floor) on EVERY ended event (the
    ended-event carrier floor of the self-map theorem), and the
    SETTLED POWER d_set is recorded in the event record as
    terminal dissipation (a ledgered observable with units of
    power; the E2 protocol reads its per-event fraction).  The
    field name is d_set, NEVER zeta: zeta is the transfer kernel's
    DIMENSIONLESS dissipation fraction of the row-sum identity,
    a different quantity with different units (the de-collided
    symbol ledger; the units sweep gates the distinction).  On every COLD exit z_exit < z_floor
    already, so the settlement is the IDENTITY and the gap-domain
    reduction above stays bit-exact (tested).  The pre-settlement
    exit carrier is returned as w_exit; the exit type is returned
    as exit_type ("cold" when the margin is nonpositive at exit,
    "hot" on the generalized branch with residual margin; None on
    the exhausted branch, which the global map rejects and which
    is NOT settled).

    Returns a dict with the post-event jet J, the SETTLED signed
    carrier amplitude w_end, the pre-settlement exit carrier
    w_exit, the settled power d_set, the exit type exit_type, the
    seeded entry carrier w_entry, fast-clock length n (a
    DIAGNOSTIC, not a global-step duration; the global gate
    duration is ``reload_steps``), integrated power sum_z, exact
    log-drain sum sum_log_phi, end curvature/margin, curvature
    drop, whether the clamp ever bound (clamped; impossible under
    (S+) on the invariant box), ended, and status."""
    h, b_sat = burst["h"], burst["b_sat"]
    z_seed, z_floor = burst["z_seed"], burst["z_floor"]
    y_floor = burst.get("y_floor", 0.0)
    tau_z = burst.get("tau_z", 0.0)
    w_entry, z = seeded_entry(w, z_seed, xi=xi,
                              w_kick_max=burst.get("w_kick_max"))
    sign = 1.0 if w_entry >= 0.0 else -1.0
    x0 = curvature(b, kappa, a, g, J)
    eps = x0 - c
    n = 0
    sum_z = 0.0
    sum_log_phi = 0.0
    clamped = False
    ended = False
    while n < fuel:
        y = kappa * (a * g * J) ** 2
        z_gate = (z_floor if eps <= 0.0
                  else max(z_floor, eps / b_sat + tau_z))
        if (eps <= 0.0 or y <= y_floor) and z < z_gate:
            ended = True
            break
        sum_z += z
        fac = 1.0 - eta * (th_J_loss + lam_wd + kappa * z)
        if fac < 0.0:
            clamped = True
            fac = 0.0
        sum_log_phi += math.log(fac) if fac > 0.0 else -math.inf
        J = fac * J
        eps = curvature(b, kappa, a, g, J) - c
        z = max(z * (1.0 + 2.0 * h * (eps - b_sat * z)), 0.0)
        n += 1
    x_end = curvature(b, kappa, a, g, J)
    z_exit = z
    w_exit = sign * math.sqrt(z_exit)
    if ended:
        # (E-settle): the terminal settlement operator, identity on
        # every cold exit (z_exit < z_floor there), projection to
        # the cold floor on a generalized hot exit; the settled
        # power is ledgered, never silently discarded
        z_settled = min(z_exit, z_floor)
        d_set = z_exit - z_settled
        exit_type = "cold" if eps <= 0.0 else "hot"
    else:
        z_settled = z_exit
        d_set = 0.0
        exit_type = None
    return dict(J=J, w_end=sign * math.sqrt(z_settled),
                w_exit=w_exit, d_set=d_set, exit_type=exit_type,
                w_entry=w_entry,
                n=n, sum_z=sum_z,
                sum_log_phi=sum_log_phi, x_end=x_end,
                eps_end=x_end - c, drop=x0 - x_end,
                clamped=clamped, ended=ended,
                status="ended" if ended else "exhausted")


# ----------------------------------------------------------------- D4



def returned_shed(M, c, r, f):
    """Exact two-site split of mass after a threshold event.

    The local reset retains ``r*c``.  A fraction ``f`` of the remaining
    shed returns locally, while ``1-f`` is exported.  Thus the new local
    state is ``r*c + f*(M-r*c)`` and local state plus export equals ``M``.
    """
    M, c, r, f = map(float, (M, c, r, f))
    if not all(map(math.isfinite, (M, c, r, f))):
        raise ValueError("returned-shed arguments must be finite")
    if c < 0 or not (0 <= r <= 1) or not (0 <= f <= 1):
        raise ValueError("require c >= 0 and r,f in [0,1]")
    retained = r * c
    if M < retained:
        raise ValueError("event mass M must be at least r*c")
    shed = M - retained
    local = retained + f * shed
    exported = (1.0 - f) * shed
    return dict(local=local, exported=exported, shed=shed,
                conserved=local + exported)
def child_mean(F_list, beta_row, sigma):
    """Expected child count for one specified parent row.

    This rowwise diagnostic is not by itself a common offspring law.
    ``fixed_receiver_envelope`` constructs the parent-independent law
    needed by a single-type Galton--Watson comparison.
    """
    return sum(F(b * sigma) for F, b in zip(F_list, beta_row))


def transported_mass(F_list, beta_row, sigma):
    """Shed-weighted transported response mass:
    sum_j beta_ij * F_j(sigma).  A mass functional; NEVER an
    offspring count (Lean: TransferKernel.transported_mass_le)."""
    return sum(b * F(sigma) for F, b in zip(F_list, beta_row))


def dose_response_bound(fmax, sigma, rowsum):
    """R_c = fmax * sigma * rowsum, the subcriticality criterion of
    the dose-response theorem.  A bound on the CONDITIONAL offspring
    mean; the distributional conclusions require the dominance
    hypothesis (K-dom), tested by E3."""
    return fmax * sigma * rowsum


def strict_empirical_margin_cdf(margins, dose):
    """Empirical left-limit CDF ``P(margin < dose)``."""
    arr = np.asarray(margins, dtype=float).ravel()
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("margins must be a nonempty finite sample")
    if not math.isfinite(float(dose)):
        raise ValueError("dose must be finite")
    return float(np.mean(arr < float(dose)))


def fixed_receiver_envelope(parent_probabilities):
    """Construct the parent-independent product-law envelope.

    Taking the receiverwise supremum of conditional bounds gives the
    fixed law ``sum_j Bernoulli(q_j)``.  Its mean, not the maximum row
    mean, is the single-type subcriticality quantity.
    """
    probs = np.asarray(parent_probabilities, dtype=float)
    if probs.ndim != 2 or probs.shape[0] < 1 or probs.shape[1] < 1:
        raise ValueError("parent_probabilities must be a nonempty matrix")
    if not np.all(np.isfinite(probs)) or np.any((probs < 0) | (probs > 1)):
        raise ValueError("conditional probabilities must lie in [0,1]")
    q = np.max(probs, axis=0)
    return dict(q=q, mean=float(np.sum(q)),
                subcritical=bool(np.sum(q) < 1.0))


def multitype_weight_certificate(mean_matrix, weights, rho):
    """Check the weighted multitype certificate ``M w <= rho w``."""
    matrix = np.asarray(mean_matrix, dtype=float)
    weight = np.asarray(weights, dtype=float).ravel()
    rho = float(rho)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] \
            or matrix.shape[0] != weight.size:
        raise ValueError("mean_matrix must be square and match weights")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("mean_matrix must be finite and nonnegative")
    if not np.all(np.isfinite(weight)) or np.any(weight <= 0):
        raise ValueError("weights must be finite and strictly positive")
    if not math.isfinite(rho) or rho < 0:
        raise ValueError("rho must be finite and nonnegative")
    image = matrix @ weight
    return dict(image=image, bound=rho * weight,
                certified=bool(np.all(image <= rho * weight)), rho=rho)


# ----------------------------------------------------------------- D5


def physical_threshold(edge_state, one_minus_beta1, eta):
    """Physical row threshold ``c=s/((1-beta1)*eta)``."""
    edge_state = float(edge_state)
    one_minus_beta1 = float(one_minus_beta1)
    eta = float(eta)
    if not all(map(math.isfinite, (edge_state, one_minus_beta1, eta))):
        raise ValueError("threshold arguments must be finite")
    if one_minus_beta1 <= 0 or eta <= 0:
        raise ValueError("one_minus_beta1 and eta must be positive")
    return edge_state / (one_minus_beta1 * eta)


def threshold_increment_exact(edge_t, edge_next, one_minus_beta1,
                              eta_t, eta_next):
    """Return the exact quotient increment and its two summands."""
    vals = tuple(map(float, (edge_t, edge_next, one_minus_beta1,
                            eta_t, eta_next)))
    if not all(map(math.isfinite, vals)):
        raise ValueError("increment arguments must be finite")
    edge_t, edge_next, one_minus_beta1, eta_t, eta_next = vals
    if one_minus_beta1 <= 0 or eta_t <= 0 or eta_next <= 0:
        raise ValueError("one_minus_beta1 and learning rates must be positive")
    schedule = (edge_t * (eta_t - eta_next)
                / (one_minus_beta1 * eta_t * eta_next))
    edge = ((edge_next - edge_t) / (one_minus_beta1 * eta_next))
    total = schedule + edge
    direct = (physical_threshold(edge_next, one_minus_beta1, eta_next)
              - physical_threshold(edge_t, one_minus_beta1, eta_t))
    return dict(schedule=schedule, edge=edge, total=total, direct=direct)


def threshold_increment_interval(edge_t, edge_increment_interval,
                                 one_minus_beta1, eta_t, eta_next):
    """Propagate a simultaneous interval for ``edge_next-edge_t``."""
    lo, hi = map(float, edge_increment_interval)
    if lo > hi:
        raise ValueError("edge increment interval must be ordered")
    lower = threshold_increment_exact(
        edge_t, float(edge_t) + lo, one_minus_beta1,
        eta_t, eta_next)["total"]
    upper = threshold_increment_exact(
        edge_t, float(edge_t) + hi, one_minus_beta1,
        eta_t, eta_next)["total"]
    return dict(lower=lower, upper=upper)


def certified_capture_margin(threshold_increment_lower,
                             complete_source_increment_upper):
    """Positive exactly when threshold speed beats every source."""
    return (float(threshold_increment_lower)
            - float(complete_source_increment_upper))


def classify_schedule_certificate(threshold_interval, source_interval,
                                  atol=0.0):
    """Classify separated intervals, retaining overlap as boundary."""
    tlo, thi = map(float, threshold_interval)
    slo, shi = map(float, source_interval)
    if not all(map(math.isfinite, (tlo, thi, slo, shi, float(atol)))):
        raise ValueError("certificate intervals and tolerance must be finite")
    if tlo > thi or slo > shi or atol < 0:
        raise ValueError("intervals must be ordered and tolerance nonnegative")
    if tlo > shi + atol:
        return "capture"
    if thi + atol < slo:
        return "release"
    return "boundary"


def capture_lhs(eta_t, eta_next, c_t, edge_increment_lower=None,
                one_minus_beta1=1.0):
    """Exact lower threshold increment expressed from ``c_t``.

    A simultaneous lower bound for the edge-state increment is
    mandatory, so the nominal schedule term cannot be mistaken for a
    sufficient capture condition.
    """
    if edge_increment_lower is None:
        raise ValueError("a simultaneous edge_increment_lower is required")
    eta_t = float(eta_t)
    eta_next = float(eta_next)
    c_t = float(c_t)
    one_minus_beta1 = float(one_minus_beta1)
    if eta_t <= 0 or eta_next <= 0 or one_minus_beta1 <= 0:
        raise ValueError("learning rates and one_minus_beta1 must be positive")
    return (c_t * (eta_t - eta_next) / eta_next
            + float(edge_increment_lower) / (one_minus_beta1 * eta_next))


def capture_rhs(eta_t, b, kappa, u, v_loss, d0, mu_b, p_i, lam_wd,
                defect_ceiling=0.0):
    """Net curvature loading rate at the capture question, positive
    part, on the decay ledger: eta * (d0 + 2 kappa u v_loss + defect
    - mu_b b - 2 p lam_wd kappa u^2), clipped at zero.  Calls
    ``drive_law`` (one drive) and carries the explicit
    ``defect_ceiling``: the budget for the averaged-model defect plus
    the Euler residual, so the comparison bounds the actual averaged
    law, not its idealization."""
    rate = eta_t * (drive_law(d0, kappa, u, v_loss)
                    + defect_ceiling
                    - mu_b * b
                    - 2.0 * p_i * lam_wd * kappa * u * u)
    return max(rate, 0.0)


def capture_side(eta_t, eta_next, c_t, b, kappa, u, v_loss, d0,
                 mu_b, p_i, lam_wd, defect_ceiling=0.0,
                 edge_increment_lower=None, one_minus_beta1=1.0):
    """Test the sufficient lower certificate against all sources."""
    lower_speed = capture_lhs(
        eta_t, eta_next, c_t,
        edge_increment_lower=edge_increment_lower,
        one_minus_beta1=one_minus_beta1,
    )
    source_upper = capture_rhs(
        eta_t, b, kappa, u, v_loss, d0, mu_b, p_i, lam_wd,
        defect_ceiling=defect_ceiling)
    return certified_capture_margin(lower_speed, source_upper) >= 0.0


def amplitude_envelope(a_t1, delta_a, th_a_loss, lam_wd):
    """The amplitude envelope of the captured phase (paper Thm
    gatedclosure, captured clause): the affine amplitude law
    a' = (1 - eta*(th_a + lam_wd))*a + eta*delta_a with any per-step
    rates eta in [eta_-, eta^*], eta^*(th_a + lam_wd) <= 1, keeps

        a_t <= a_sup = max(a(T1), delta_a/(th_a + lam_wd))

    for all t >= T1: the fixed point delta_a/(th_a + lam_wd) is
    eta-independent and the map is monotone toward it.  This is the
    invariant that controls the FULL product u = a*g*J after
    capture; the amplitude-below-band capture witness (a rebuilding
    from 0.01 toward band value 1 topples a site whose curvature was
    below threshold at capture) is pinned by a test as the necessity
    witness for bounding u by the envelope rather than by u(T1)."""
    th = th_a_loss + lam_wd
    if th <= 0.0:
        if delta_a > 0.0:
            raise ValueError("unbounded amplitude: delta_a > 0 with "
                             "th_a + lam_wd <= 0 has no envelope")
        return a_t1
    return max(a_t1, delta_a / th)


def base_envelope(b_t1, d0, mu_b):
    """The base envelope of the captured phase: the affine base law
    b' = (1 - eta*mu_b)*b + eta*d0 (no arrivals after events cease)
    keeps b_t <= b_sup = max(b(T1), d0/mu_b) for all t >= T1, by the
    same monotone-affine invariant as ``amplitude_envelope``."""
    if mu_b <= 0.0:
        if d0 > 0.0:
            raise ValueError("unbounded base: d0 > 0 with mu_b <= 0 "
                             "has no envelope")
        return b_t1
    return max(b_t1, d0 / mu_b)


def drive_ceiling(d0_sup, kappa, a_sup, g_sup, J_sup, delta_a,
                  defect_ceiling=0.0):
    """The DRIVE CEILING D_max on the invariant box (paper Thm
    wellposed, drive clause): a closed-form box supremum of the
    drive D = d0 + 2*kappa*u*v_loss + defect,

        D_max = d0_sup + 2*kappa*(a_sup*g_sup*J_sup)
                          *(delta_a*g_sup*J_sup)
                + defect_ceiling ,

    with every sign-indefinite term bounded in the direction the
    capture induction uses: u <= a_sup*g_sup*J_sup (the amplitude
    envelope and the box ceilings g_sup, J_sup <= 1), and the
    positive part of the loss-only source velocity bounded by its
    rebuild share, (v_loss)_+ <= delta_a*g_sup*J_sup (every
    reversion term is nonpositive).  The averaged-model defect
    enters once through ``defect_ceiling``; the Euler remainder is
    carried separately in C_D and never double counted.  The
    captured-regime anneal condition consumes this value, so it is
    computable from planning constants (a GENERATOR GATE); the
    capture inequality reads

        c(t+1) - c(t) >= eta_t * (D_max + C_D * eta_t)."""
    if min(d0_sup, kappa, a_sup, g_sup, J_sup, delta_a) < 0.0:
        raise ValueError("need nonnegative box ceilings")
    if defect_ceiling < 0.0:
        raise ValueError("need defect_ceiling >= 0")
    u_sup = a_sup * g_sup * J_sup
    v_plus = delta_a * g_sup * J_sup
    return d0_sup + 2.0 * kappa * u_sup * v_plus + defect_ceiling


def capture_balance(b_sup, kappa, a_sup, g_t1, J_t1, c_inf):
    """The envelope balance condition of the corrected captured
    clause (paper Thm gatedclosure): with the gate closed from T1,

        b_sup + kappa*(a_sup * g(T1) * J(T1))^2 <= inf_{t>=T1} c(t)

    implies no site topples at any t >= T1 (g has no rebuild and is
    nonincreasing; J is nonincreasing while the gate is closed; a is
    bounded by the envelope; b by its envelope).  Returns the signed
    slack c_inf - (b_sup + kappa*(a_sup*g_t1*J_t1)^2); the condition
    holds iff the slack is nonnegative."""
    return c_inf - (b_sup + kappa * (a_sup * g_t1 * J_t1) ** 2)


# ----------------------------------------------------------------- D6


def ar1_stationary_var(eta, mu_x, sig2_over_B, scaled=True):
    """Stationary variance of the curvature AR(1) under the PRINTED
    convention xi = eta * zeta (scaled=True), or the literal
    convention (scaled=False), exactly (Lean:
    FirstPassageEnvelope.ar1_variance_fixed_point_*)."""
    a = eta * mu_x
    if not 0.0 < a < 2.0:
        raise ValueError("need 0 < eta*mu_x < 2")
    if scaled:
        return eta * sig2_over_B / (mu_x * (2.0 - a))
    return sig2_over_B / (eta * mu_x * (2.0 - a))


def _phi_bar(x):
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def passage_band(delta_margin, eta, mu_x, sig2_over_B, T):
    """Two-sided band for the UPWARD passage probability of the
    Gaussian model (paper Prop. passageband): (lower, upper).  An
    upward crossing is a TOPPLING TRIGGER; it is not entry into
    collapse.  Valid in the MARGINAL regime (balance point at or
    below the threshold, L < c); the recurrent stay-below bound is
    ``hold_below_bound``."""
    s_inf = math.sqrt(ar1_stationary_var(eta, mu_x, sig2_over_B))
    upper = min(1.0, T * _phi_bar(delta_margin / s_inf))
    # lower bound: single-time tail at the horizon with the exact
    # finite-time variance (v_T <= v_inf; mean at the balance point)
    a = eta * mu_x
    vT = (eta ** 2) * sig2_over_B * (
        (1.0 - (1.0 - a) ** (2 * T)) / (1.0 - (1.0 - a) ** 2)
    )
    lower = _phi_bar(delta_margin / math.sqrt(vT)) if vT > 0 else 0.0
    return lower, upper


def entry_prob_mc(delta_margin, eta, mu_x, sig2_over_B, T,
                  n_mc=20000, seed=0):
    """Monte Carlo estimate of the ENTRY (stay-below) probability
    under the pinned noise convention: the margin process starts
    ``delta_margin`` below the reload level, the deterministic drive
    pulls it back up to the level (sustained reload), and entry into
    collapse requires the noise to hold it below the level for T
    consecutive steps.  Chain: x' = (1 - eta*mu_x) x + eta*mu_x*L
    + eta*zeta with Var zeta = sig2_over_B, x0 = L - delta_margin,
    event {x_t < L for all t <= T}.  This SIGNED margin process is
    deliberately unprojected (no nonnegativity is claimed for it;
    the projected chain and its stay-below bound are
    ``p_min_crossing``/``hold_below_bound``).  This is the
    collapse-entry event of the closure theorem's recurrent clause;
    it is distinct from (and typically far smaller than) the upward
    trigger band."""
    a = eta * mu_x
    if not 0.0 < a < 2.0:
        raise ValueError("need 0 < eta*mu_x < 2")
    rng = np.random.default_rng(seed)
    L = 0.0
    x = np.full(n_mc, L - delta_margin)
    alive = np.ones(n_mc, dtype=bool)
    sd = math.sqrt(sig2_over_B)
    for _t in range(T):
        x = (1.0 - a) * x + a * L + eta * rng.normal(0.0, sd,
                                                     size=n_mc)
        alive &= x < L
        if not alive.any():
            break
    return float(alive.mean())


def p_min_crossing(c, gap, a, eta_sig):
    """Uniform per-step crossing probability of the RECURRENT
    stay-below model (paper Prop. holdbelow), stated for the
    PROJECTED chain x' = max((1-a) x + a L + eta*zeta, 0): the
    printed process, matching the nonnegative-state premise (a
    nondegenerate Gaussian has unbounded negative support, so the
    unprojected chain leaves x >= 0 with positive probability; the
    projection at zero maps [0, c] into itself and PRESERVES the
    crossing event, {x' > c} = {unprojected > c} for c >= 0, so the
    per-step bound is unchanged).  For the margin y = c - x with
    balance gap = L - c > 0 and y in [0, c] on the hold event, the
    per-step probability of crossing the threshold is at least

        p_min = Phi_bar( ((1-a) c - a gap) / eta_sig ),

    uniformly on the hold event (worst case y = c).  ``eta_sig`` is
    the per-step noise scale eta * sd(zeta)."""
    if not (0.0 < a < 1.0 and gap > 0.0 and c > 0.0
            and eta_sig > 0.0):
        raise ValueError("need 0 < a < 1, gap > 0, c > 0, "
                         "eta_sig > 0")
    return _phi_bar(((1.0 - a) * c - a * gap) / eta_sig)


def hold_below_bound(p_min, T):
    """The recurrent stay-below (entry) envelope for the PROJECTED
    chain: with per-step crossing probability at least p_min
    uniformly on the hold event (``p_min_crossing``; unchanged by
    the projection, which preserves the crossing event), the
    probability of holding below the threshold through T consecutive
    steps is at most (1 - p_min)^T."""
    if not 0.0 <= p_min <= 1.0:
        raise ValueError("need p_min in [0, 1]")
    return (1.0 - p_min) ** T


# ----------------------------------------------------------------- D7


def entry_time_bound(J0, J_crit, q):
    """Printed captured-phase entry step (paper Thm gatedclosure;
    Lean: GatedClosure.entry_time): with per-step jet factor at most
    q = 1 - eta_- theta_- and zero rebuild (gate closed), the jet
    enters J <= J_crit no later than

        ceil( log(J_crit / J0) / log q )        for 0 < q < 1,
        1                                        for q = 0

    (one-step entry at the boundary case eta_- theta_- = 1).
    Returns 0 when J0 <= J_crit already."""
    if not (0.0 < J_crit and J0 >= 0.0 and 0.0 <= q < 1.0):
        raise ValueError("need J_crit > 0, J0 >= 0, 0 <= q < 1")
    if J0 <= J_crit:
        return 0
    if q == 0.0:
        return 1
    return math.ceil(math.log(J_crit / J0) / math.log(q))


def reload_steps(J_pre, J_post, eta, delta_J0):
    """The RELOAD DURATION written into the activity counter after
    an event (paper Def. gatedmodel): the number of global rebuild
    steps needed to restore the drained jet at the scheduled rate,

        R = ceil( (J_pre - J_post) / (eta * delta_J0) )

    global steps (zero when the rebuild is off or nothing drained).
    This gives the activity variable physical units: the fast-clock
    iteration count of the inner event is a diagnostic only and
    never enters the global dynamics (the two clocks have different
    units under T-sep)."""
    if delta_J0 <= 0.0 or eta <= 0.0:
        return 0
    drained = J_pre - J_post
    if drained <= 0.0:
        return 0
    return int(math.ceil(drained / (eta * delta_J0)))


def jet_unrolled(J0, q_list, m_list):
    """The EXACT affine unrolling identity of the jet law (paper Thm
    gatedclosure, recurrent clause): for J_{s+1} = q_s*J_s + m_s,

        J_T = (prod_s q_s) * J_0
              + sum_r m_r * prod_{s=r+1}^{T-1} q_s :

    every rebuild injection is multiplied by every LATER contraction
    (its survival factor).  Identity function for tests; the lower
    bound the closure theorem consumes is ``jet_unroll_lower``."""
    if len(q_list) != len(m_list):
        raise ValueError("need matching q and m sequences")
    total = float(J0)
    for q, m in zip(q_list, m_list):
        total = q * total + m
    return total


def jet_unroll_lower(J0, q_min, T, m_total):
    """The derived cycle lower bound (paper Thm gatedclosure,
    recurrent clause): if every per-step contraction obeys
    0 <= q_min <= q_s <= 1 and the injections are nonnegative with
    total m_total, then

        J_T >= q_min^T * (J_0 + m_total) :

    each injection survives to cycle end with factor at least
    q_min^T.  All inputs are LOWER bounds computed at the extremal
    eta, z, w of the box (bounds enter in the correct direction: an
    upper bound on a multiplier never lower-bounds the state).  The
    raw sum eta*delta_J0*N_act alone is an upper value before the
    later contractions, not a lower bound (retired form; the
    maintenance witness is pinned by a test)."""
    if not (0.0 <= q_min <= 1.0 and T >= 0 and m_total >= 0.0
            and J0 >= 0.0):
        raise ValueError("need 0 <= q_min <= 1, T >= 0, "
                         "m_total >= 0, J0 >= 0")
    return q_min ** T * (J0 + m_total)


def cycle_floor_inputs(eta_min, eta_max, delta_J0, n_act, t_cyc,
                       th_J_loss, lam_wd, kappa, w_max2, phi_min,
                       n_ev_max):
    """The DERIVED inputs (q_cyc, m_cyc) of the maintenance floor
    (paper Thm gatedclosure, recurrent clause), from the exact
    unrolling: over one burst-reload cycle of t_cyc global steps
    containing one event (jet crash factor at least
    phi_min^n_ev_max) and n_act active-rebuild steps,

        q_step = 1 - eta_max*(th_J + lam_wd + kappa*w_max^2),
        q_cyc  = phi_min^n_ev_max * q_step^t_cyc,
        m_cyc  = eta_min * delta_J0 * n_act * q_step^t_cyc,

    so J_next >= q_cyc*J_cur + m_cyc is a DERIVED inequality of the
    map (``jet_unroll_lower`` applied to the cycle), and
    ``maintenance_floor(q_cyc, m_cyc)`` is its invariant floor.
    Requires q_step >= 0 (the step condition at the box corner)."""
    q_step = 1.0 - eta_max * (th_J_loss + lam_wd + kappa * w_max2)
    if q_step < 0.0:
        raise ValueError("step condition violated at the box corner: "
                         "eta_max*(th_J + lam_wd + kappa*w_max^2) > 1")
    if not (0.0 <= phi_min <= 1.0 and n_ev_max >= 0):
        raise ValueError("need phi_min in [0, 1] and n_ev_max >= 0")
    q_cyc = phi_min ** n_ev_max * q_step ** t_cyc
    m_cyc = eta_min * delta_J0 * n_act * q_step ** t_cyc
    return q_cyc, m_cyc


def maintenance_floor(q_cyc, m_cyc):
    """The recurrent-phase jet floor (paper Thm gatedclosure,
    recurrent clause): if over every event cycle the jet obeys
    J' >= q_cyc * J + m_cyc with cycle contraction q_cyc in [0, 1)
    and delivered rebuild m_cyc > 0, then
    J_floor = m_cyc / (1 - q_cyc) is invariant from below:
    J_0 >= J_floor implies J_k >= J_floor for all k.  The cycle
    inequality is DERIVED from the map by ``cycle_floor_inputs`` /
    ``jet_unroll_lower``, never assumed.  The MAINTENANCE INEQUALITY
    of the closure theorem is J_floor > J_crit; the recurrent clause
    is horizon-explicit in the chain factor (g has no rebuild, so
    with th_g + lam_wd > 0 the curvature floor carries the factor
    g(t) >= g_0*(1 - eta^*(th_g + lam_wd))^t and no schedule
    maintains curvature forever; the asymptotic clause holds in the
    th_g = lam_wd = 0 subclass)."""
    if not (0.0 <= q_cyc < 1.0 and m_cyc > 0.0):
        raise ValueError("need 0 <= q_cyc < 1 and m_cyc > 0")
    return m_cyc / (1.0 - q_cyc)


def maintenance_floor_clipped(q_cyc, m_cyc):
    """The endpoint maintenance floor of the CLIPPED cycle
    recursion (paper Thm clippedfloor, the clipped floor
    comparison): the composed cycle map is
    J' = min(q_cyc*J + m, 1) with per-cycle delivered rebuild
    m >= m_cyc; whenever the floor respects the jet ceiling,

        J_floor = m_cyc/(1 - q_cyc) <= 1
        (equivalently m_cyc <= 1 - q_cyc),

    the affine comparison orbit Jhat' = q_cyc*Jhat + m_cyc opened
    at Jhat_0 = J_0 <= 1 never exceeds max(J_0, J_floor) <= 1, so
    it never exceeds the clip level and minorizes the clipped
    orbit by induction: J_k >= Jhat_k for all k, hence
    liminf J_k >= J_floor WITH NO nonsaturation hypothesis.  This
    is the certificate the (M-end) clause of the closure consumes
    for the clipped transition; the prefix and average
    certificates additionally carry the all-prefix nonsaturation
    certificate (N-sat) (``maintenance_prefix_min``).  Raises
    unless the floor ceiling J_floor <= 1 holds (a generated
    gate)."""
    floor = maintenance_floor(q_cyc, m_cyc)
    if floor > 1.0:
        raise ValueError(
            "clipped floor comparison needs J_floor = "
            f"m_cyc/(1 - q_cyc) = {floor:.6g} <= 1 (the jet "
            "ceiling); the affine minorant would exceed the clip")
    return floor


def maintenance_prefix_min(J_start, r_min, q_min, m_seq):
    """The ALL-PREFIX maintenance lower envelope over one recurrent
    cycle (paper Thm maintenanceprefix): the cycle opens at the
    endpoint value J_start, the event crash retains at least
    r_min*J_start, and the reload/decay steps obey
    J_{l+1} = q_l*J_l + m_l with q_l in [q_min, 1] and the
    injections m_l >= 0 at their ACTUAL indices (m_seq[l] must be a
    LOWER bound on the injection delivered at inner index l; zero
    off the gate).  Every prefix then satisfies

        J_l >= L_l = q_min^l * (r_min*J_start + sum_{r<l} m_r),

    and the returned value is min_{0<=l<=T} L_l: a lower bound on
    the jet at EVERY global time of the cycle, including the
    immediate post-crash state (l = 0).  The all-prefix maintenance
    condition of the recurrent regime is

        maintenance_prefix_min(...) > J_crit ,

    strictly stronger than the endpoint floor ``maintenance_floor``
    (the within-cycle witness r*J_floor < J_crit < J_floor is
    pinned by a test as the necessity control: the endpoint floor
    alone does NOT certify the interior).  Every bound enters in
    the direction that lowers L_l.

    (N-sat), the ALL-PREFIX NONSATURATION CERTIFICATE: the
    composed transition clips the jet at 1, and the affine
    envelope minorizes the CLIPPED orbit only when the envelope
    itself never demands more than the ceiling,

        q_min^l * (r_min*J_start + sum_{r<l} m_r) <= 1
        for every prefix l <= T

    (the hypothesis of the formal recurrent iterate theorem,
    promoted to this certificate and to a generated gate).  This
    function REFUSES certification (raises) when (N-sat) fails:
    an early oversized injection would be clipped away and then
    decayed, and the affine envelope would overstate the orbit
    (the clipped-envelope witness q = 0.1, injections (50, 0),
    actual endpoint 0.1 against affine 0.5, is pinned by a
    test)."""
    if not (0.0 <= r_min <= 1.0 and 0.0 <= q_min <= 1.0
            and J_start >= 0.0):
        raise ValueError("need r_min, q_min in [0, 1] and "
                         "J_start >= 0")
    if any(m < 0.0 for m in m_seq):
        raise ValueError("need nonnegative injection lower bounds")
    level = r_min * J_start
    lo = level
    acc = level
    pw = 1.0
    if level > 1.0:
        raise ValueError(
            "all-prefix nonsaturation certificate (N-sat) fails "
            f"at prefix 0: envelope {level:.6g} > 1")
    for k, m in enumerate(m_seq):
        acc += m
        pw *= q_min
        env = pw * acc
        if env > 1.0:
            raise ValueError(
                "all-prefix nonsaturation certificate (N-sat) "
                f"fails at prefix {k + 1}: envelope {env:.6g} > 1 "
                "(the affine formula is not a lower bound of the "
                "clipped orbit)")
        lo = min(lo, env)
    return lo


def nsat_envelope_max(J_start, r_min, q_min, m_seq):
    """The (N-sat) certificate VALUE: the maximum prefix envelope

        max_{0<=l<=T} q_min^l * (r_min*J_start + sum_{r<l} m_r)

    of the cycle.  The all-prefix nonsaturation certificate (N-sat)
    holds iff this value is at most 1 (the jet ceiling); it is the
    quantity the generator gates at both tiers and the hypothesis
    under which the affine prefix envelope minorizes the CLIPPED
    orbit (``maintenance_prefix_min`` enforces it by refusal; this
    function reports the value without raising)."""
    if not (0.0 <= r_min <= 1.0 and 0.0 <= q_min <= 1.0
            and J_start >= 0.0):
        raise ValueError("need r_min, q_min in [0, 1] and "
                         "J_start >= 0")
    if any(m < 0.0 for m in m_seq):
        raise ValueError("need nonnegative injection lower bounds")
    level = r_min * J_start
    hi = level
    acc = level
    pw = 1.0
    for m in m_seq:
        acc += m
        pw *= q_min
        hi = max(hi, pw * acc)
    return hi


def maintenance_prefix_floor(J_floor, r_min, q_min, t_cyc):
    """The COARSE closed-form sufficient bound of the all-prefix
    maintenance theorem: every prefix envelope of a cycle of length
    t_cyc opening at J_floor satisfies

        L_l >= q_min^l * r_min * J_floor
            >= r_min * q_min^t_cyc * J_floor ,

    so the pointwise (all-time) maintenance conclusion holds
    whenever this value exceeds J_crit.  A rigorous LOWER bound on
    ``maintenance_prefix_min`` (which evaluates the exact prefix
    minimum with the injections at their actual indices and is
    always at least this coarse value); the certificate the
    generator and E5 evaluate is the exact minimum."""
    if not (0.0 <= r_min <= 1.0 and 0.0 <= q_min <= 1.0
            and J_floor >= 0.0 and t_cyc >= 0):
        raise ValueError("need r_min, q_min in [0, 1], "
                         "J_floor >= 0, t_cyc >= 0")
    return r_min * q_min ** t_cyc * J_floor


def maintenance_cycle_average(J_start, r_min, q_min, m_seq):
    """The CYCLE-AVERAGE maintenance lower bound (paper Thm
    maintenanceavg), the INDEPENDENT certificate (M-avg): the mean
    of the prefix lower envelope L_l of ``maintenance_prefix_min``
    over the cycle,

        (1/(T+1)) * sum_{0<=l<=T} L_l ,

    a lower bound on the time average of the jet over the cycle.
    This is the statement matched to the MEASURED phase criterion,
    which is a time average (the last-quarter mean of the
    minimum-layer median row-map Jacobian singular value): if the
    bound exceeds J_crit, every running mean over a window of at
    least one cycle length exceeds J_crit up to the explicit
    boundary term of windows not aligned with cycle endpoints.

    PREMISES: the prefix-envelope hypotheses of
    ``maintenance_prefix_min`` (per-index injection lower bounds
    m_seq, retention r_min, decay floor q_min) AND the all-prefix
    nonsaturation certificate (N-sat), enforced here identically.
    The endpoint floor ``maintenance_floor`` does NOT imply this
    bound: the certificate is independent, and the adverse cycle
    with endpoint floor above the criterion and cycle average
    below it is pinned by a test as the permanent negative
    control.  The pointwise conclusion needs the stronger
    ``maintenance_prefix_min`` > J_crit."""
    if not (0.0 <= r_min <= 1.0 and 0.0 <= q_min <= 1.0
            and J_start >= 0.0):
        raise ValueError("need r_min, q_min in [0, 1] and "
                         "J_start >= 0")
    if any(m < 0.0 for m in m_seq):
        raise ValueError("need nonnegative injection lower bounds")
    level = r_min * J_start
    total = level
    acc = level
    pw = 1.0
    if level > 1.0:
        raise ValueError(
            "all-prefix nonsaturation certificate (N-sat) fails "
            f"at prefix 0: envelope {level:.6g} > 1")
    for k, m in enumerate(m_seq):
        acc += m
        pw *= q_min
        env = pw * acc
        if env > 1.0:
            raise ValueError(
                "all-prefix nonsaturation certificate (N-sat) "
                f"fails at prefix {k + 1}: envelope {env:.6g} > 1 "
                "(the affine formula is not a lower bound of the "
                "clipped orbit)")
        total += env
    return total / (len(m_seq) + 1)


def active_reload_count(act_series):
    """The ACTUAL active-reload count of a recorded orbit: the
    number of global steps at which the activity gate is open,
    N_act = #{t : act_t > 0}.  The rebuild total m_cyc of the
    maintenance constants must use this count (or a proved lower
    bound on it), NEVER an inter-event gap length: the reload
    counter may expire before the next event, and substituting the
    whole gap overstates the delivered rebuild (pinned by a
    test)."""
    return int(sum(1 for a in act_series if a > 0))


# ----------------------------------------------------------------- D8


def monomial_transform(tau, gamma, b):
    """Exponent transform under S_obs = A S^b, b > 0 (paper Thm
    monomial): returns (tau_obs, gamma_obs)."""
    if b <= 0:
        raise ValueError("the tail transform requires b > 0")
    return 1.0 + (tau - 1.0) / b, b * gamma


# ---------------------------------------------------------- gated map


def gated_model_step(state, params, c_t, eta_t, fuel, rng=None,
                     validate=True, events_out=None):
    """One global step of the gated curvature sandpile on the SINGLE
    curvature state and the SINGLE event clock (paper Def.
    gatedmodel).

    state:  dict of 1-d arrays b, a, g, J, w, pend, act (per site):
            base curvature, coupling amplitude, chain factor, jet
            magnitude, carrier amplitude, pending shed, remaining
            RELOAD steps (global-step units).  There is NO stress
            entry: the curvature is derived, x = b + kappa (a g J)^2.
    params: dict with beta (matrix), d0, delta_a, delta_J0, kappa,
            th_a_loss, th_g_loss, th_J_loss (arrays or scalars),
            lam_wd, mu_b, s_shed (scalars), burst (dict with h,
            b_sat, z_seed, z_floor).  Rates are LOSS-ONLY; each
            factor additionally decays at lam_wd (decay ledger).
    c_t:    per-site thresholds (array)
    fuel:   the resolver fuel F, DECLARED MODEL DATA (paper Def.
            gatedmodel): a required argument threaded unchanged to
            ``run_event`` for every event this step resolves.
            There is no hidden default; the generator instantiates
            the printed granted value
            (``event_fuel_required_uniform`` on the whole box).
            Fuel below the granted bound can starve a legal event
            (EventNotEnded); tests pin starvation at F - 1 and
            success at F on the boundary entry.

    Step semantics: arrivals land on the base (hypothesis K-land);
    the jet rebuild is delta_J0 * A with A = (act > 0) the activity
    gate (hypothesis H-gate), SATURATED at the normalized ceiling
    J <= 1 (the box invariance of the jet needs no rate condition
    on delta_J0); the guard is strict (x > c after the
    smooth step); a toppling site resolves its burst through
    ``run_event`` (the inner event is instantaneous on the global
    clock under T-sep; reset, carrier, and shed all derived).  The
    activity counter is then set to the RELOAD DURATION
    ``reload_steps`` in GLOBAL steps (the jet deficit over the
    per-step rebuild), never to the fast-clock iteration count: the
    gate is a post-event recovery window with physical units, and
    the recurrent regime is defined by SEPARATED burst-reload
    cycles.  An orbit that topples every global step is outside
    the recurrent regime and inside the DRIVEN regime of the
    classification theorem (paper Thm. overloadclass): under
    (O-load) (sustained base drive d0/mu_b above the threshold
    band) or (O-reb) (one-gap reload saturation eta*delta_J0 >= 1
    with retained amplitude/chain drive kappa*(a g)^2 above the
    band) the guard fires at every global step for all time, and
    no bound on the consecutive-topple run exists; under the
    headroom certificate (O-head) the run resolves within
    ``overload_resolution_bound`` global steps.  The planners
    fail generation on driven cells for the recurrent protocols
    and route them to the driven discriminator (E5).  With
    validate=True the step conditions (S+)/(Q+) of the closure
    theorem are enforced (the carrier condition is eta*x <= 1, the
    strengthened form the geometric carrier bound needs).

    Every ended event emits its carrier through the terminal
    settlement operator (E-settle) inside ``run_event``, so the
    successor carrier satisfies |w| <= sqrt(z_floor) <= w_max at
    every toppled site (the box self-map clause); the settled
    power and the exit type are per-event observables.  Pass a
    list as ``events_out`` to collect the full per-site event
    records of this step (site index added under "site"); the
    return signature is unchanged.
    Returns (new_state, toppled_mask)."""
    b, a, g, J, w, pend, act = (
        np.asarray(state[k], dtype=float).copy()
        for k in ("b", "a", "g", "J", "w", "pend", "act")
    )
    beta = params["beta"]
    d0 = np.asarray(params["d0"], dtype=float)
    delta_a = np.asarray(params["delta_a"], dtype=float)
    delta_J0 = np.asarray(params["delta_J0"], dtype=float)
    kappa = np.asarray(params["kappa"], dtype=float)
    th_a = np.asarray(params["th_a_loss"], dtype=float)
    th_g = np.asarray(params["th_g_loss"], dtype=float)
    th_J = np.asarray(params["th_J_loss"], dtype=float)
    lam_wd = params["lam_wd"]
    mu_b = params["mu_b"]
    s_shed = params["s_shed"]
    burst = params["burst"]

    arrivals = beta.T @ pend
    b_in = b + arrivals               # (K-land): sheds load the base
    u = a * g * J
    q_curv = b_in + kappa * u * u

    if validate:
        checks = [
            ("eta*mu_b", eta_t * mu_b),
            ("eta*(th_a_loss + lam_wd)",
             float(np.max(eta_t * (th_a + lam_wd)))),
            ("eta*(th_g_loss + lam_wd)",
             float(np.max(eta_t * (th_g + lam_wd)))),
            ("eta*(th_J_loss + lam_wd + kappa*w^2)",
             float(np.max(eta_t * (th_J + lam_wd + kappa * w * w)))),
            ("eta*(b + kappa*u^2)",
             float(np.max(eta_t * q_curv))),
        ]
        for name, val in checks:
            if val > 1.0 + 1e-12:
                raise ValueError(
                    "closure step condition (S)/(Q+) violated: "
                    f"{name} = {val:.6g} > 1"
                )

    A = (act > 0).astype(float)       # the activity gate (H-gate)
    new_a = (1.0 - eta_t * (th_a + lam_wd)) * a + eta_t * delta_a
    new_g = (1.0 - eta_t * (th_g + lam_wd)) * g
    new_J = np.minimum(
        (1.0 - eta_t * (th_J + lam_wd + kappa * w * w)) * J
        + eta_t * delta_J0 * A, 1.0)   # saturated rebuild: the jet
    # magnitude is normalized, J in [0, 1] is invariant with NO
    # rate condition on delta_J0 (paper Def. gatedmodel; the clip
    # only binds when the unsaturated rebuild would leave the box)
    new_b = b_in + eta_t * d0 - eta_t * mu_b * b_in
    new_w = (1.0 - eta_t * q_curv) * w
    new_act = np.maximum(act - 1.0, 0.0)

    x_new = curvature(new_b, kappa, new_a, new_g, new_J)
    topple = x_new > c_t              # the strict guard

    new_pend = np.zeros_like(pend)
    kap = np.broadcast_to(kappa, x_new.shape)
    thj = np.broadcast_to(th_J, x_new.shape)
    dj0 = np.broadcast_to(delta_J0, x_new.shape)
    for i in np.flatnonzero(topple):
        # the kick realization of the seeded entry operator: the
        # registered TRUNCATED seed-kick law (truncation
        # burst["w_kick_max"] applied inside ``seeded_entry``)
        # keeps the seeded carrier inside the invariant box premise
        # |w| <= w_max of the closure theorem (the stochastic
        # curvature chain is separately projected at zero; see the
        # noise section)
        xi_i = 0.0
        if rng is not None and burst["z_seed"] > 0.0:
            xi_i = float(rng.normal(
                0.0, math.sqrt(burst["z_seed"])))
        J_pre = float(new_J[i])
        ev = run_event(float(new_b[i]), float(new_a[i]),
                       float(new_g[i]), J_pre, float(new_w[i]),
                       float(c_t[i]), eta_t, float(kap[i]),
                       float(thj[i]), lam_wd, burst, fuel,
                       xi=xi_i)
        if not ev["ended"]:
            raise EventNotEnded(
                "event resolution exhausted its fuel at site "
                f"{i}: b = {float(new_b[i]):.6g}, "
                f"c = {float(c_t[i]):.6g}, "
                f"eps_end = {ev['eps_end']:.6g}. The global map "
                "rejects the Exhausted branch; this entry lies "
                "outside the (B-gap) domain (or the fuel is below "
                "the termination bound)."
            )
        new_J[i] = ev["J"]
        new_w[i] = ev["w_end"]
        new_pend[i] = s_shed * max(ev["drop"], 0.0)
        new_act[i] = reload_steps(J_pre, ev["J"], eta_t,
                                  float(dj0[i]))
        if events_out is not None:
            events_out.append(dict(site=int(i), **ev))

    new = dict(b=new_b, a=new_a, g=new_g, J=new_J, w=new_w,
               pend=new_pend, act=new_act)
    return new, topple


def state_in_box(state, box):
    """The DECLARED STATE BOX membership check (paper Thm totality,
    self-map clause): returns the list of violations (empty iff the
    state lies in the box).  ``box`` is a dict of the box ceilings;
    only the supplied keys are checked:

        b_max     0 <= b <= b_max        (base ceiling B_max)
        a_max     0 <= a <= a_max
        g_max     0 <= g <= g_max
        j_max     0 <= J <= j_max        (normalized jet, j_max = 1)
        w_max     |w| <= w_max           (carrier box)
        pend_max  0 <= pend <= pend_max
                  act >= 0 and integer-valued (always checked)

    Used by the post-step box tests (every settled successor of an
    ended event must pass) and available as a debug assertion; the
    theorem-tier constants are the generator's."""
    tol = 1e-12
    out = []

    def _arr(k):
        return np.asarray(state[k], dtype=float)

    for key, name in (("b_max", "b"), ("a_max", "a"),
                      ("g_max", "g"), ("j_max", "J"),
                      ("pend_max", "pend")):
        if key in box:
            v = _arr(name)
            if float(np.min(v)) < -tol:
                out.append(f"{name} < 0")
            if float(np.max(v)) > box[key] + tol:
                out.append(f"{name} > {key} = {box[key]:.6g}")
    if "w_max" in box:
        w = _arr("w")
        if float(np.max(np.abs(w))) > box["w_max"] + tol:
            out.append(f"|w| > w_max = {box['w_max']:.6g}")
    act = _arr("act")
    if float(np.min(act)) < -tol:
        out.append("act < 0")
    if float(np.max(np.abs(act - np.round(act)))) > tol:
        out.append("act not integer-valued")
    return out


# ------------------------------------------- event-word partition


def classify_event_word(topple_word, act_series, run_bound=None,
                        quiet_tail_frac=0.25, min_window=20):
    """The OBSERVATION-WINDOW PHENOTYPE of one logged orbit
    segment: finite-window statistics and exactly one phenotype
    label.  A finite window does NOT decide an asymptotic orbit
    class (every finite prefix of the event/reload records has
    continuations in all four classes); class membership is
    concluded ONLY through the placement theorems, whose bridge
    statements predict the phenotype of every sufficiently long
    window under each certificate.  Measured phenotypes therefore
    TEST certificates; a phenotype that matches no certificate
    prediction is an unaccounted observation and is treated as
    adverse evidence by the decision rules, never assigned a
    class.

    Phenotypes (exactly one):

      "censored"     the window is too short for the declared
                     bounds (fewer than ``min_window`` steps), or
                     events continue with fewer than two events
                     observed (pair structure undecidable);
      "quiet-tail"   no event in the last ``quiet_tail_frac`` of
                     the window (the capture certificate predicts
                     this phenotype after the entry time);
      "long-run"     the maximum consecutive-topple run reaches
                     ``run_bound`` (default max(5, T//10); the
                     sustained-drive certificate predicts this
                     phenotype);
      "sep-window"   events continue, runs stay under the bound,
                     and ZERO successive event pairs are
                     unseparated (the completed-reload recurrent
                     certificate (R-sep) predicts this phenotype
                     in every window with at least two events);
      "unsep-window" events continue, runs stay under the bound,
                     and at least one successive pair has NO
                     completed reload (the strict alternating
                     (O-int) and gate-held (O-hold) certificates
                     predict unseparated fraction ONE).

    Reload separation between successive events at steps t1 < t2
    reads the states strictly after the earlier event and at or
    before the later one: ``act_series[t1 .. t2-1]`` (the state
    after step m is ``act_series[m]``), matching the formal
    definition (Lean ClosureFourRegime.ReloadSeparated): the ONE
    meaning of separation, used identically by the certificates,
    the gates, and the decision rules; no gap or
    reload-positivity proxy stands in for it anywhere.

    The reported ``unsep_frac`` and ``gate_closure_frac`` are
    logged statistics with a generated sensitivity table
    (``phenotype_sensitivity``); no density threshold assigns the
    phenotype.

    Returns a dict: phen, n_events, density, max_run, n_pairs,
    n_unsep, unsep_frac, gate_closure_frac, last_event.
    """
    word = np.asarray(topple_word, dtype=int).ravel()
    act = np.asarray(act_series, dtype=float).ravel()
    if word.shape != act.shape:
        raise ValueError("topple_word and act_series must have the "
                         "same length")
    T = len(word)
    if T == 0:
        raise ValueError("empty event word")
    if run_bound is None:
        run_bound = max(5, T // 10)
    events = np.flatnonzero(word)
    n_events = int(len(events))
    density = n_events / float(T)
    # maximum consecutive-topple run
    max_run = 0
    run = 0
    for b in word:
        run = run + 1 if b else 0
        max_run = max(max_run, run)
    # successive pairs and their reload separation
    n_pairs = 0
    n_unsep = 0
    n_sep = 0
    for t1, t2 in zip(events[:-1], events[1:]):
        n_pairs += 1
        if np.any(act[t1:t2] == 0):
            n_sep += 1
        else:
            n_unsep += 1
    unsep_frac = (n_unsep / n_pairs) if n_pairs else 0.0
    gate_closure_frac = (n_sep / n_pairs) if n_pairs else 0.0
    last_event = int(events[-1]) if n_events else -1
    quiet = (n_events == 0
             or last_event < (1.0 - quiet_tail_frac) * (T - 1))
    if T < min_window:
        phen = "censored"
    elif quiet:
        phen = "quiet-tail"
    elif max_run >= run_bound:
        phen = "long-run"
    elif n_pairs == 0:
        phen = "censored"
    elif n_unsep == 0:
        phen = "sep-window"
    else:
        phen = "unsep-window"
    return dict(phen=phen, n_events=n_events, density=density,
                max_run=max_run, n_pairs=n_pairs, n_unsep=n_unsep,
                unsep_frac=unsep_frac,
                gate_closure_frac=gate_closure_frac,
                last_event=last_event)


def phenotype_sensitivity(topple_word, act_series,
                          window_fracs=(0.25, 0.5, 0.75, 1.0),
                          run_bounds=(None, 5, 10),
                          quiet_tail_fracs=(0.125, 0.25, 0.5)):
    """The GENERATED SENSITIVITY TABLE of the window phenotype:
    the phenotype recomputed over trailing sub-windows and over
    the declared threshold grid, so every reported phenotype
    carries its stability under the observation choices.  Returns
    a list of dicts (window_frac, T, run_bound, quiet_tail_frac,
    phen)."""
    word = np.asarray(topple_word, dtype=int).ravel()
    act = np.asarray(act_series, dtype=float).ravel()
    T = len(word)
    rows = []
    for wf in window_fracs:
        n = max(1, int(round(wf * T)))
        w = word[T - n:]
        a = act[T - n:]
        for rb in run_bounds:
            for qf in quiet_tail_fracs:
                st = classify_event_word(w, a, run_bound=rb,
                                         quiet_tail_frac=qf)
                rows.append(dict(window_frac=wf, T=n,
                                 run_bound=(rb if rb is not None
                                            else max(5, n // 10)),
                                 quiet_tail_frac=qf,
                                 phen=st["phen"]))
    return rows


def intermittent_witness_config():
    """The PINNED INTERMITTENT WITNESS (paper Ex. intwitness; Lean
    IntermittentRegime.witnessPhi/witnessPlo/witnessBounds/
    witnessEntry): the exact rational constants of the alternating
    two-phase schedule whose orbit inhabits the intermittent class.

    One site, no arrivals and no shed.  The base is an EXACT fixed
    point at both rates (d0 = mu_b * b = 0.075); even steps run at
    (c, eta) = (2, 0.1), where the clip ceiling closes the guard
    (x <= 0.75 + 1 = 1.75 < 2); odd steps run at (0.5, 0.001),
    where the base alone forces it (0.75 > 0.5); every exit is hot
    with J_out <= sqrt(y_floor) = 0.1; every event writes a reload
    counter of about 101 against a one-step gap, so the gate NEVER
    closes and the reload never completes.  The topple word is
    0101..., the maximum run is one, and the logged jet oscillates
    between about 0.09999 (post-event) and 0.29900 (post-rebuild):
    the intermittent band.

    Returns (state, params, schedule) with schedule(t) -> (c_t,
    eta_t); the three pinned forms (this config, the manuscript
    example, and the Lean witness) share these constants and must
    not drift apart (test_intermittent_witness pins the orbit).
    """
    state = dict(b=np.array([0.75]), a=np.array([1.0]),
                 g=np.array([1.0]), J=np.array([0.1]),
                 w=np.array([0.0]), pend=np.array([0.0]),
                 act=np.array([100.0]))
    params = dict(beta=np.zeros((1, 1)), d0=np.array([0.075]),
                  delta_a=np.array([0.0]),
                  delta_J0=np.array([2.0]), kappa=np.array([1.0]),
                  th_a_loss=np.array([0.0]),
                  th_g_loss=np.array([0.0]),
                  th_J_loss=np.array([0.1]), lam_wd=0.0, mu_b=0.1,
                  s_shed=0.0,
                  burst=dict(h=0.1, b_sat=1.0, z_seed=0.01,
                             z_floor=0.001, y_floor=0.01,
                             tau_z=0.02, w_kick_max=0.1))

    def schedule(t):
        if t % 2 == 0:
            return np.array([2.0]), 0.1
        return np.array([0.5]), 0.001

    return state, params, schedule


def run_intermittent_witness(n_steps=40, fuel=200_000):
    """Iterate the pinned intermittent witness through the FULL
    composed transition (``gated_model_step`` with validation) and
    return the orbit record the regression and the planning gate
    consume: the topple word, the per-step reload counter and jet,
    the post-event jets, and the derived event-word statistics
    (``classify_event_word``)."""
    state, params, schedule = intermittent_witness_config()
    word = []
    act_series = []
    J_series = []
    J_out = []
    for t in range(n_steps):
        c_t, eta_t = schedule(t)
        state, toppled = gated_model_step(state, params, c_t,
                                          eta_t, fuel=fuel,
                                          validate=True)
        bit = int(bool(toppled[0]))
        word.append(bit)
        act_series.append(float(state["act"][0]))
        J_series.append(float(state["J"][0]))
        if bit:
            J_out.append(float(state["J"][0]))
    stats = classify_event_word(word, act_series)
    return dict(word=word, act_series=act_series,
                J_series=J_series, J_out=J_out, state=state,
                stats=stats)


def separated_witness_config():
    """The PINNED SEPARATED WITNESS (paper Ex. sepwitness; Lean
    RecurrentSeparated.sepPhi/sepPlo/sepBounds/sepJHold/sepEntry):
    the exact rational constants of the period-four schedule whose
    orbit inhabits the SEPARATED class with every reload
    completed.

    One site, no arrivals and no shed.  The base is an EXACT fixed
    point (d0 = mu_b * b = 0.075); steps with t % 4 != 3 run at
    (c, eta, delta_J0) = (2, 0.1, 0.5), where the clip ceiling
    closes the guard (x <= 0.75 + 1 = 1.75 < 2) and the rebuild
    quantum is eta*delta_J0 = 0.05; steps with t % 4 == 3 run at
    (0.5, 0.1, 2), where the base alone forces the guard
    (0.75 > 0.5) and the write quantum is eta*delta_J0 = 0.2.
    Every exit is hot with J_out <= sqrt(y_floor) = 0.1; the
    pre-event jet cap 0.15 fits under ONE write quantum, so every
    written reload duration is at most one, the counter reaches
    zero at the first quiet step of every gap, and the gate
    CLOSES between bursts: the exact opposite side of the write
    law from the intermittent overhang.

    Returns (state, params, schedule) with
    schedule(t) -> (c_t, eta_t, delta_J0_t); the three pinned
    forms (this config, the manuscript example, and the Lean
    witness) share these constants and must not drift apart
    (test_separated_witness pins the orbit)."""
    state = dict(b=np.array([0.75]), a=np.array([1.0]),
                 g=np.array([1.0]), J=np.array([0.1]),
                 w=np.array([0.0]), pend=np.array([0.0]),
                 act=np.array([1.0]))
    params = dict(beta=np.zeros((1, 1)), d0=np.array([0.075]),
                  delta_a=np.array([0.0]),
                  delta_J0=np.array([0.5]), kappa=np.array([1.0]),
                  th_a_loss=np.array([0.0]),
                  th_g_loss=np.array([0.0]),
                  th_J_loss=np.array([0.1]), lam_wd=0.0, mu_b=0.1,
                  s_shed=0.0,
                  burst=dict(h=0.1, b_sat=1.0, z_seed=0.01,
                             z_floor=0.001, y_floor=0.01,
                             tau_z=0.02, w_kick_max=0.1))

    def schedule(t):
        if t % 4 == 3:
            return np.array([0.5]), 0.1, np.array([2.0])
        return np.array([2.0]), 0.1, np.array([0.5])

    return state, params, schedule


def run_separated_witness(n_steps=40, fuel=200_000):
    """Iterate the pinned separated witness through the FULL
    composed transition (``gated_model_step`` with validation) and
    return the orbit record the regression and the planning gate
    consume: the topple word, the per-step reload counter and jet,
    the post-event jets, and the derived window phenotype
    (``classify_event_word``): sep-window with zero unseparated
    pairs and gate-closure fraction one."""
    state, params, schedule = separated_witness_config()
    word = []
    act_series = []
    J_series = []
    J_out = []
    for t in range(n_steps):
        c_t, eta_t, dj0_t = schedule(t)
        params["delta_J0"] = dj0_t
        state, toppled = gated_model_step(state, params, c_t,
                                          eta_t, fuel=fuel,
                                          validate=True)
        bit = int(bool(toppled[0]))
        word.append(bit)
        act_series.append(float(state["act"][0]))
        J_series.append(float(state["J"][0]))
        if bit:
            J_out.append(float(state["J"][0]))
    stats = classify_event_word(word, act_series)
    return dict(word=word, act_series=act_series,
                J_series=J_series, J_out=J_out, state=state,
                stats=stats)


def precond_replacement_ceiling(eps_nu, T_av):
    """The CORRECTED averaged-preconditioner replacement budget
    (paper Prop. averaged, corrected orientation; Lean
    ``replacement_bound`` / ``delta_le_eps_div`` /
    ``drift_chain``): the coefficient charged against the realized
    preconditioned magnitude is
    delta_P = max_s ||I - P_s Pbar_t^{-1}||, NOT the displayed-
    window discrepancy eps_P = max_s ||Pbar_t P_s^{-1} - I||
    (scalar counterexample: P_s = 1, Pbar_t = 0.8 gives eps_P =
    0.2 against true replacement error 0.25).  Under the
    strengthened window condition T_av * eps_nu <= 1/4 the
    two-sided drift chain gives eps_P <= 2*T_av*eps_nu <= 1/2 and

        delta_P <= eps_P / (1 - eps_P) <= 2*eps_P
                <= 4 * T_av * eps_nu,

    the displayed small-drift ceiling with the corrected
    orientation.  Raises when the window condition fails (the
    coefficient is uncontrolled as eps_P approaches one)."""
    if eps_nu < 0 or T_av < 0:
        raise ValueError("eps_nu and T_av must be nonnegative")
    if T_av * eps_nu > 0.25 + 1e-12:
        raise ValueError(
            "window condition T_av * eps_nu <= 1/4 violated: "
            f"{T_av * eps_nu:.6g} (the replacement coefficient "
            "is uncontrolled as eps_P approaches one)")
    return 4.0 * T_av * eps_nu


# ------------------------------------------ protocol decision rules

import itertools as _itertools
import json as _json
import os as _os

import decision_constants as _dc

DECISION_API_VERSION = _dc.DECISION_API_VERSION

# canonical location of the generated configuration objects
# (written by scripts/gen_protocols.py next to the protocols)
_DECISION_CONFIG_PATH = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir,
    "protocols", "decision_configs.json")


class DecisionSchemaError(ValueError):
    """A decision input violates its declared typed schema: a
    missing or extra arm or cell, a wrong seed count, a missing
    field, an empty or non-finite trace, or a configuration whose
    version does not match the decision API.  No partial design can
    reach a verdict: schema violations raise, never PASS."""


def load_decision_config(rule, path=None):
    """Load the generated configuration object for ``rule`` ("e5"
    or "e6") from ``experiments/protocols/decision_configs.json``
    (written at generation time by scripts/gen_protocols.py).
    Future data are scored by loading THIS object, so the
    confirmatory rule and the generated protocol cannot drift."""
    with open(path or _DECISION_CONFIG_PATH) as fh:
        configs = _json.load(fh)
    if rule not in configs:
        raise DecisionSchemaError(
            f"no generated configuration for rule {rule!r}")
    cfg = configs[rule]
    _check_study_config(cfg, rule)
    return cfg


def _check_config_version(config, rule):
    if not isinstance(config, dict):
        raise DecisionSchemaError(
            f"{rule}: configuration must be a dict, got "
            f"{type(config).__name__}")
    if config.get("rule") != rule:
        raise DecisionSchemaError(
            f"configuration rule tag {config.get('rule')!r} does "
            f"not match the invoked rule {rule!r}")
    if config.get("version") != DECISION_API_VERSION:
        raise DecisionSchemaError(
            f"{rule}: configuration version "
            f"{config.get('version')!r} does not match the "
            f"decision API version {DECISION_API_VERSION!r}")


def _check_study_config(config, rule):
    """Validate a complete study object and all attempt objects."""
    if not isinstance(config, dict):
        raise DecisionSchemaError(f"{rule} study: configuration must be a dict")
    if config.get("rule") != f"{rule}-study":
        raise DecisionSchemaError(
            f"{rule} study: rule tag must be {rule + '-study'!r}")
    if config.get("version") != DECISION_API_VERSION:
        raise DecisionSchemaError(
            f"{rule} study: stale version {config.get('version')!r}")
    attempts = config.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != 2:
        raise DecisionSchemaError(
            f"{rule} study: exactly two typed attempts are required")
    ids = []
    blocks = []
    for attempt in attempts:
        _check_config_version(attempt, rule)
        ids.append(attempt.get("attempt_id"))
        blocks.append(attempt.get("seed_block"))
    if any(not isinstance(x, str) or not x for x in ids + blocks):
        raise DecisionSchemaError(
            f"{rule} study: every attempt needs an id and seed block")
    if len(set(ids)) != len(ids) or len(set(blocks)) != len(blocks):
        raise DecisionSchemaError(
            f"{rule} study: attempt ids and seed blocks must be distinct")
    if rule == "e6":
        alpha_sum = sum(float(a["alpha"]) for a in attempts)
        if alpha_sum > float(config["study_alpha"]) + 1e-12:
            raise DecisionSchemaError(
                "e6 study: per-attempt alpha allocations exceed study alpha")


def _trace(rec, key, where, length=None, integral=False):
    """Extract and validate one 1-D finite trace from a record."""
    if key not in rec:
        raise DecisionSchemaError(f"{where}: missing field {key!r}")
    arr = np.asarray(rec[key], dtype=float).ravel()
    if arr.size == 0:
        raise DecisionSchemaError(f"{where}: empty trace {key!r}")
    if not np.all(np.isfinite(arr)):
        raise DecisionSchemaError(
            f"{where}: non-finite value in trace {key!r}")
    if length is not None and arr.size != length:
        raise DecisionSchemaError(
            f"{where}: trace {key!r} has length {arr.size} against "
            f"{length}")
    if integral and not np.all(np.isin(arr, (0.0, 1.0))):
        raise DecisionSchemaError(
            f"{where}: trace {key!r} must be 0/1")
    return arr


def derive_capture_endpoint(rec, J_crit, min_persistence_probes=3,
                            where="record", require_step_one=False):
    """Derive an observed finite-horizon endpoint from raw probes.

    The row-map tensor has axes ``(probe, layer, visited_row)``.  The
    normalizer may be scalar, per layer, or probe by layer.  A terminal
    all-layer-flat run establishes finite-horizon entry only.  Future
    invariance is a separate analytical certificate.  The step hold is
    causal: global steps preceding the first probe remain missing.  A
    confirmation record can require its first probe at global step one.

    Supplied endpoint summaries are rejected so contradictory metadata
    cannot affect the result.
    """
    forbidden = set(_dc.FORBIDDEN_ENDPOINT_SUMMARIES) & set(rec)
    if forbidden:
        raise DecisionSchemaError(
            f"{where}: forbidden endpoint summaries {sorted(forbidden)}")
    if "word" not in rec:
        raise DecisionSchemaError(f"{where}: missing field 'word'")
    horizon = np.asarray(rec["word"]).size
    if horizon == 0:
        raise DecisionSchemaError(f"{where}: empty global horizon")
    steps = _trace(rec, "probe_steps", where, integral=False)
    if not np.all(steps == np.floor(steps)):
        raise DecisionSchemaError(f"{where}: probe_steps must be integers")
    steps = steps.astype(int)
    if np.any(np.diff(steps) <= 0) or steps[0] < 1 or steps[-1] > horizon:
        raise DecisionSchemaError(
            f"{where}: probe_steps must increase strictly inside 1..{horizon}")
    if require_step_one and steps[0] != 1:
        raise DecisionSchemaError(
            f"{where}: confirmation records must probe at global step 1")
    if "rowmap_samples" not in rec:
        raise DecisionSchemaError(f"{where}: missing field 'rowmap_samples'")
    samples = np.asarray(rec["rowmap_samples"], dtype=float)
    if samples.ndim != 3 or samples.shape[0] != len(steps) \
            or samples.shape[1] < 1 or samples.shape[2] < 1:
        raise DecisionSchemaError(
            f"{where}: rowmap_samples must have shape "
            "(n_probe,n_layer,n_visited_row)")
    if not np.all(np.isfinite(samples)) or np.any(samples < 0):
        raise DecisionSchemaError(
            f"{where}: rowmap_samples must be finite and nonnegative")
    if "normalization" not in rec:
        raise DecisionSchemaError(f"{where}: missing field 'normalization'")
    norm = np.asarray(rec["normalization"], dtype=float)
    target = samples.shape[:2]
    if norm.ndim == 0:
        norm = np.full(target, float(norm))
    elif norm.ndim == 1 and norm.size == samples.shape[1]:
        norm = np.broadcast_to(norm[None, :], target).astype(float)
    elif norm.ndim == 2 and norm.shape == target:
        norm = norm.astype(float)
    else:
        raise DecisionSchemaError(
            f"{where}: normalization must be scalar, per layer, or "
            "probe by layer")
    if not np.all(np.isfinite(norm)) or np.any(norm <= 0):
        raise DecisionSchemaError(
            f"{where}: normalization must be finite and positive")
    layer_probe = np.median(samples, axis=2) / norm
    all_layer_probe = np.max(layer_probe, axis=1)
    below = all_layer_probe <= float(J_crit)
    trailing = 0
    for flag in below[::-1]:
        if not flag:
            break
        trailing += 1
    minimum = int(min_persistence_probes)
    if minimum < 1:
        raise DecisionSchemaError(
            f"{where}: min_persistence_probes must be positive")
    if trailing >= minimum:
        start = len(steps) - trailing
        status = "finite_horizon_entry"
        entry = int(steps[start])
    elif trailing > 0 or len(steps) < minimum:
        status = "censored"
        entry = None
    else:
        status = "no_finite_horizon_entry"
        entry = None
    clock = np.arange(1, horizon + 1)
    idx = np.searchsorted(steps, clock, side="right") - 1
    observed = idx >= 0
    jet_step = np.full(horizon, np.nan, dtype=float)
    jet_step[observed] = all_layer_probe[idx[observed]]
    return dict(
        finite_horizon_entry=entry,
        future_invariance_certified=False,
        observation_mask=observed,
        status=status,
        entry=entry,
        trailing_probes=int(trailing),
        historical_terminal_all_flat=bool(below[-1]),
        probe_steps=steps,
        layer_probe=layer_probe,
        all_layer_probe=all_layer_probe,
        jet_step=jet_step,
    )


# ------------------------------- shared measurement functions


def measure_gate_slope(J, act, word):
    """THE GATE REGRESSION on reload windows, in global-step units
    (hypothesis H-gate): the mean one-step jet increment over the
    steps at which the reload gate is open and no event fires,

        slope = mean{ J[t] - J[t-1] : act[t-1] > 0, word[t] = 0 } ,

    the executable form of the printed regression of jet rebuild
    on the reload indicator (the event step itself is the crash,
    not the rebuild, and is excluded).  Applied identically to
    future data and inside the operating characteristic.  Returns
    nan when the record contains no such step (no reload
    observed)."""
    J = np.asarray(J, dtype=float).ravel()
    act = np.asarray(act, dtype=float).ravel()
    word = np.asarray(word, dtype=int).ravel()
    if J.shape != act.shape or J.shape != word.shape:
        raise ValueError("J, act, and word must have the same "
                         "length")
    sel = (act[:-1] > 0) & (word[1:] == 0)
    if not np.any(sel):
        return float("nan")
    return float(np.mean(np.diff(J)[sel]))


def measure_cycle_constants(word, act, J, q_step, m_quantum):
    """THE MEASURED CYCLE CONSTANTS of a logged maintenance orbit,
    computed from the full traces exactly as the generation gate
    computes them for the planning orbit: cycle length t_cyc = the
    minimum inter-event gap; realized per-event retention r_min =
    min over events of J_post/J_pre (clamped to [0, 1]); the
    gate-indicator active-reload count n_act of a representative
    cycle (``active_reload_count``, never an inter-event gap
    length); and the injection sequence m_seq = m_quantum on the
    n_act active steps and zero off the gate.  ``q_step`` (the
    per-step decay floor) and ``m_quantum`` (the per-active-step
    rebuild quantum eta * delta_J0) are measurement constants from
    the configuration object, re-measured by E0.  Returns None when
    the record carries fewer than two events (no cycle exists);
    otherwise a dict with the constants and the derived certificate
    inputs (q_cyc, m_cyc)."""
    word = np.asarray(word, dtype=int).ravel()
    act = np.asarray(act, dtype=float).ravel()
    J = np.asarray(J, dtype=float).ravel()
    events = np.flatnonzero(word)
    if len(events) < 2:
        return None
    gaps = np.diff(events)
    t_cyc = int(gaps.min())
    rs = []
    for t in events:
        if t >= 1 and J[t - 1] > 0:
            rs.append(min(max(J[t] / J[t - 1], 0.0), 1.0))
    if not rs:
        return None
    r_min = float(min(rs))
    i0 = int(events[len(events) // 2])
    n_act = active_reload_count(act[i0:i0 + t_cyc])
    m_seq = [m_quantum] * n_act + [0.0] * max(t_cyc - n_act, 0)
    q_cyc = r_min * q_step ** t_cyc
    m_cyc = sum(m_seq) * q_step ** t_cyc
    return dict(r_min=r_min, q_step=float(q_step), t_cyc=t_cyc,
                n_act=n_act, m_seq=m_seq, q_cyc=q_cyc, m_cyc=m_cyc)


def cycle_certificates(cyc, J_crit):
    """Evaluate the maintenance certificates at measured cycle
    constants, through the SAME functions the theorem constants
    use: the clipped endpoint floor (M-end), the (N-sat) envelope
    maximum, the all-prefix minimum (M-prefix), and the cycle
    average (M-avg-c), with the licensure label.  Certificates that
    are undefined at the constants (contraction at or above one,
    no delivered rebuild, or (N-sat) failing) are reported as
    failed, never silently passed."""
    out = dict(floor=0.0, nsat=float("inf"), prefix_min=0.0,
               cycle_avg=0.0, licensure="none", defined=False)
    if cyc is None:
        return out
    if not (0.0 <= cyc["q_cyc"] < 1.0 and cyc["m_cyc"] > 0.0):
        return out
    try:
        floor = maintenance_floor_clipped(cyc["q_cyc"],
                                          cyc["m_cyc"])
    except ValueError:
        return out
    nsat = nsat_envelope_max(floor, cyc["r_min"], cyc["q_step"],
                             cyc["m_seq"])
    out.update(floor=floor, nsat=nsat, defined=True)
    if nsat <= 1.0:
        pmin = maintenance_prefix_min(floor, cyc["r_min"],
                                      cyc["q_step"], cyc["m_seq"])
        cavg = maintenance_cycle_average(floor, cyc["r_min"],
                                         cyc["q_step"],
                                         cyc["m_seq"])
        out.update(prefix_min=pmin, cycle_avg=cavg)
    lic = ("pointwise" if out["prefix_min"] > J_crit
           else "time-average" if out["cycle_avg"] > J_crit
           else "stroboscopic-only" if out["floor"] > J_crit
           else "none")
    out["licensure"] = lic
    return out


# --------------------------------------------------------- E5


def validate_e5_design(arms, config):
    """STRICT SCHEMA VALIDATION of an E5 design against the
    configuration object: exactly the three declared arms, exactly
    the declared seed count per arm, every per-seed record a dict
    carrying the full global traces and raw row-map probes.  Entry is
    derived inside the rule; trusted endpoint summaries are forbidden.
    Raises DecisionSchemaError on ANY violation; an empty design,
    a missing arm, a wrong seed count, or a missing field can
    never reach a verdict."""
    _check_config_version(config, "e5")
    if not isinstance(arms, dict):
        raise DecisionSchemaError(
            f"e5: design must be a dict, got "
            f"{type(arms).__name__}")
    want = list(config["arms"])
    if sorted(arms.keys()) != sorted(want):
        raise DecisionSchemaError(
            f"e5: design arms {sorted(arms.keys())} against the "
            f"declared {sorted(want)}")
    n = int(config["seeds_per_arm"])
    for name in want:
        seeds = arms[name]
        if not isinstance(seeds, (list, tuple)) or len(seeds) != n:
            raise DecisionSchemaError(
                f"e5: arm {name!r} has "
                f"{len(seeds) if isinstance(seeds, (list, tuple)) else 'non-list'} "
                f"seed records against the declared {n}")
        for i, rec in enumerate(seeds):
            where = f"e5 {name}[{i}]"
            if not isinstance(rec, dict):
                raise DecisionSchemaError(
                    f"{where}: record must be a dict")
            word = _trace(rec, "word", where, integral=True)
            _trace(rec, "act", where, length=word.size)
            derive_capture_endpoint(
                rec, config["J_crit"],
                config["min_persistence_probes"], where=where, require_step_one=True)


def decide_e5(arms, config):
    """THE E5 DECISION FUNCTION: one executable object, applied
    identically to future data and inside the operating
    characteristic (``gen_protocols._power_e5`` is a Monte Carlo of
    THIS function, wrapped in the study orchestrator
    ``decide_e5_study``, on fully generated observables; the
    protocol's decision section is rendered from this object).  No
    simplified twin exists.

    ``config`` is the MANDATORY generated configuration object
    (``load_decision_config("e5")``); there are no optional
    decision constants.  The input schema is validated first
    (``validate_e5_design``): exactly three arms (captured,
    separated, gateheld), exactly the declared seed count, full
    traces per seed.  Every derived observable the printed rule
    consumes is computed INSIDE the rule from the traces by the
    declared measurement functions and reported per seed: the
    window phenotype (``classify_event_word``), the gate
    regression on reload windows (``measure_gate_slope``), the
    measured cycle constants (``measure_cycle_constants``), and
    the maintenance certificates at those constants
    (``cycle_certificates``, the same functions the theorem
    constants use).

    Clauses, predeclared:

      ROUTING.  Every seed's event word is routed by its window
      phenotype.  A non-censored phenotype different from the
      arm's certificate prediction is an UNACCOUNTED ORBIT and
      KILLS the construct.

      CAPTURED.  Every seed: quiet-tail; raw-probe-derived persistent
      all-layer entry within the
      configured bound; events cease BEFORE the derived entry
      (the last event's one-based global step is strictly less than
      entry; otherwise the causal-order KILL); reload activity is
      zero from entry through the observed horizon.

      SEPARATED MAINTENANCE.  Every seed: sep-window with ZERO
      unseparated pairs and strictly positive gate-closure
      fraction; post-burn-in jet minimum ABOVE J_crit; the
      MEASURED TIME-AVERAGED criterion (post-burn-in mean jet)
      above J_crit; the CYCLE-AVERAGE certificate (M-avg-c) at
      the measured cycle constants above J_crit; a POSITIVE gate
      regression on reload windows.

      GATE-HELD MAINTENANCE.  Every seed: unsep-window with
      unseparated fraction ONE and maximum run one; the same
      jet-minimum, time-average, cycle-average, and
      gate-regression clauses as the separated arm.

      REPORTED WITH LICENSURE.  The all-prefix minimum (M-prefix)
      and the endpoint floor (M-end) at the measured constants are
      computed and reported per seed with the licensure label; the
      pointwise conclusion is claimed only at seeds where the
      all-prefix certificate passes.  The theorem-tier certificate
      values from the configuration are echoed alongside.  Neither
      report gates the verdict (the decision gates on the measured
      tier: the cycle-average and time-average clauses above).

      GATE-REGRESSION KILL.  A null gate regression EVERYWHERE
      (no maintenance seed with a positive measured slope) KILLS
      the construct (H-gate).

      CENSORING.  Censored seeds make the attempt INDETERMINATE;
      the predeclared repeat is the study orchestrator
      ``decide_e5_study`` (one separately configured fresh-seed
      repeat; a second indeterminate outcome remains INDETERMINATE).

    Returns dict(verdict = "PASS" | "FAIL" | "KILL" |
    "INDETERMINATE", reasons, arms = per-seed reports including
    every derived observable, version)."""
    validate_e5_design(arms, config)
    J_crit = float(config["J_crit"])
    entry_bound = int(config["entry_bound"])
    burn_frac = float(config["burn_frac"])
    q_step = float(config["q_step"])
    m_quantum = float(config["m_quantum"])
    predictions = config["predictions"]
    reasons = []
    arm_out = {}
    verdict = "PASS"
    slopes = []
    for name in config["arms"]:
        seeds = arms[name]
        pred = predictions[name]
        seed_out = []
        for i, rec in enumerate(seeds):
            word = np.asarray(rec["word"], dtype=int).ravel()
            act = np.asarray(rec["act"], dtype=float).ravel()
            endpoint = derive_capture_endpoint(
                rec, J_crit, config["min_persistence_probes"],
                where=f"e5 {name}[{i}]", require_step_one=True)
            J = endpoint["jet_step"]
            st = classify_event_word(word, act)
            entry = endpoint["entry"]
            ok = True
            why = []
            slope = float("nan")
            cyc = None
            cert = None
            gate_closed = None
            if st["phen"] == "censored" or endpoint["status"] == "censored":
                ok = None
                why.append("censored window or endpoint horizon")
            elif st["phen"] != pred:
                verdict = "KILL"
                ok = False
                why.append(
                    f"unaccounted orbit: phenotype {st['phen']} "
                    f"against predicted {pred}")
            elif name == "captured":
                gate_closed = False
                if endpoint["status"] != "finite_horizon_entry" \
                        or entry is None or entry > entry_bound:
                    ok = False
                    why.append("no persistent all-layer entry within the bound")
                if entry is not None:
                    last_event_step = (st["last_event"] + 1
                                       if st["last_event"] >= 0 else 0)
                    if last_event_step >= entry:
                        verdict = "KILL"
                        ok = False
                        why.append("capture causal order: event cessation "
                                   "does not strictly precede entry")
                    gate_closed = not np.any(act[entry - 1:] > 0)
                    if not gate_closed:
                        ok = False
                        why.append("reload gate is not closed from entry "
                                   "through the observed horizon")
            else:
                burn = int(burn_frac * len(J))
                J_min = float(np.min(J[burn:]))
                J_avg = float(np.mean(J[burn:]))
                if J_min <= J_crit:
                    ok = False
                    why.append("post-burn-in jet at or below the "
                               "criterion")
                if J_avg <= J_crit:
                    ok = False
                    why.append("measured time-averaged criterion "
                               "at or below J_crit")
                if name == "separated":
                    if st["n_unsep"] != 0:
                        ok = False
                        why.append("an unseparated pair in the "
                                   "separated arm")
                    if st["gate_closure_frac"] <= 0:
                        ok = False
                        why.append("no completed reload observed")
                else:
                    if st["unsep_frac"] < 1.0:
                        ok = False
                        why.append("a separated pair in the "
                                   "gate-held arm")
                    if st["max_run"] != 1:
                        ok = False
                        why.append("a run of length two")
                slope = measure_gate_slope(J, act, word)
                slopes.append(slope)
                if not (slope > 0.0):
                    ok = False
                    why.append("gate regression not positive on "
                               "the reload windows")
                cyc = measure_cycle_constants(word, act, J,
                                              q_step, m_quantum)
                cert = cycle_certificates(cyc, J_crit)
                if not (cert["defined"]
                        and cert["cycle_avg"] > J_crit):
                    ok = False
                    why.append("cycle-average certificate "
                               "(M-avg-c) at or below J_crit at "
                               "the measured cycle constants")
            seed_out.append(dict(
                seed=i, phen=st["phen"], ok=ok, why=why, stats=st,
                gate_slope=slope, cycle=cyc, certificates=cert,
                gate_closed_after_entry=(
                    gate_closed if name == "captured" else None),
                endpoint={k: endpoint[k] for k in
                          ("status", "entry", "trailing_probes",
                           "historical_terminal_all_flat")}))
            reasons.extend(f"{name}[{i}]: {w}" for w in why)
        arm_out[name] = seed_out
    if slopes and not any(s > 0.0 for s in slopes):
        verdict = "KILL"
        reasons.append("gate regression null everywhere (H-gate)")
    flat = [s for seeds in arm_out.values() for s in seeds]
    if verdict != "KILL":
        if any(s["ok"] is None for s in flat):
            verdict = "INDETERMINATE"
        elif any(s["ok"] is False for s in flat):
            verdict = "FAIL"
    return dict(verdict=verdict, reasons=reasons, arms=arm_out,
                thm_tier=config.get("thm_tier"),
                version=config["version"])


def decide_e5_study(attempts, config):
    """THE E5 STUDY ORCHESTRATOR: the explicit, typed repeat
    state of the printed censoring policy, executed and simulated
    (never left to prose).  ``attempts`` is the list of scored
    designs in order: the first attempt at the declared seed
    count, and, ONLY when the first is INDETERMINATE, one separately
    configured repeat with a fresh seed block.  Rules: a non-indeterminate first
    attempt IS the study verdict (a supplied second attempt then
    violates the schema); a first INDETERMINATE with no repeat
    yet returns INDETERMINATE with the repeat instruction; a
    second INDETERMINATE outcome has the configured terminal status.
    Returns dict(verdict, attempts=[per-attempt outputs],
    repeat_pending)."""
    _check_study_config(config, "e5")
    if not isinstance(attempts, (list, tuple)) \
            or not 1 <= len(attempts) <= len(config["attempts"]):
        raise DecisionSchemaError(
            "e5 study: attempts must be a list of one or "
            f"{len(config['attempts'])} designs")
    out1 = decide_e5(attempts[0], config["attempts"][0])
    if out1["verdict"] != "INDETERMINATE":
        if len(attempts) > 1:
            raise DecisionSchemaError(
                "e5 study: a second attempt is only allowed "
                "after a first INDETERMINATE outcome")
        return dict(verdict=out1["verdict"], attempts=[out1],
                    repeat_pending=False)
    if len(attempts) == 1:
        return dict(verdict="INDETERMINATE", attempts=[out1],
                    repeat_pending=True)
    out2 = decide_e5(attempts[1], config["attempts"][1])
    if out2["verdict"] == "INDETERMINATE":
        return dict(verdict=config["second_indeterminate"],
                    attempts=[out1, out2],
                    repeat_pending=False)
    return dict(verdict=out2["verdict"], attempts=[out1, out2],
                repeat_pending=False)


# --------------------------------------------------------- E6


def validate_e6_design(cells, config):
    """Strict schema validation of an E6 design.

    Every cell must carry simultaneous threshold- and source-increment
    intervals whose strict classification agrees with its declared side.
    Boundary cells are ineligible.  Every run then carries global
    event/reload traces and raw within-layer row-map probes; endpoint
    summaries and Boolean-only inputs cannot reach a verdict.
    """
    _check_config_version(config, "e6")
    if not isinstance(cells, dict):
        raise DecisionSchemaError(
            f"e6: design must be a dict, got "
            f"{type(cells).__name__}")
    if len(cells) != int(config["n_cells"]):
        raise DecisionSchemaError(
            f"e6: {len(cells)} cells against the declared "
            f"{config['n_cells']}")
    sides = [cell.get("predicted") for cell in cells.values()
             if isinstance(cell, dict)]
    if sides.count("capture") < 2 or sides.count("release") < 2:
        raise DecisionSchemaError(
            "e6: at least two cells are required on each predicted side")
    n = int(config["seeds_per_cell"])
    for cname, cell in cells.items():
        where = f"e6 cell {cname!r}"
        if not isinstance(cell, dict):
            raise DecisionSchemaError(f"{where}: must be a dict")
        if cell.get("predicted") not in ("capture", "release"):
            raise DecisionSchemaError(
                f"{where}: 'predicted' must be 'capture' or "
                f"'release', got {cell.get('predicted')!r}")
        seeds = cell.get("seeds")
        cert = cell.get("schedule_certificate")
        if not isinstance(cert, dict):
            raise DecisionSchemaError(
                f"{where}: missing simultaneous schedule_certificate")
        try:
            threshold_interval = cert["threshold_increment_interval"]
            source_interval = cert["source_increment_interval"]
            side = classify_schedule_certificate(
                threshold_interval, source_interval)
        except (KeyError, TypeError, ValueError) as exc:
            raise DecisionSchemaError(
                f"{where}: invalid schedule_certificate: {exc}") from exc
        if side == "boundary":
            raise DecisionSchemaError(
                f"{where}: touching or overlapping certificate intervals "
                "define a boundary cell, which is not E6-eligible")
        if side != cell["predicted"]:
            raise DecisionSchemaError(
                f"{where}: certificate class {side!r} disagrees with "
                f"predicted side {cell['predicted']!r}")
        if not isinstance(seeds, (list, tuple)) or len(seeds) != n:
            raise DecisionSchemaError(
                f"{where}: needs exactly {n} per-seed run "
                "records under 'seeds'")
        for i, rec in enumerate(seeds):
            rwhere = f"{where} seed[{i}]"
            if not isinstance(rec, dict):
                raise DecisionSchemaError(
                    f"{rwhere}: record must be a dict")
            word = _trace(rec, "word", rwhere, integral=True)
            _trace(rec, "act", rwhere, length=word.size)
            derive_capture_endpoint(
                rec, config["J_crit"],
                config["min_persistence_probes"], where=rwhere,
                require_step_one=True)


def decide_e6(cells, config):
    """THE E6 DECISION FUNCTION: one executable object, applied
    identically to future data and inside the operating
    characteristic; the protocol's decision section is rendered
    from this object and no simplified twin exists.  ``config`` is
    the MANDATORY generated configuration object
    (``load_decision_config("e6")``): the count threshold is
    chosen at generation by exact enumeration over the declared
    crossed cell-by-seed null family and lives ONLY there; the
    function has no default threshold.

    The input is the full per-run record set
    (``validate_e6_design``): per cell, simultaneous threshold- and
    source-increment intervals must strictly certify its declared
    capture or release side.  Touching intervals are boundary cells
    and are rejected before outcome scoring.  Each cell also carries
    the declared number of seeded runs with logged traces.  The match
    indicator of every run is computed INSIDE the rule:

      CAPTURE-PREDICTED cell.  A run matches iff the raw probes yield
      an observed finite-horizon all-layer entry, event cessation precedes
      that entry, and reload activity is zero from entry through the
      observed horizon.  Failure of strict event ordering is the
      causal-order violation and kills the study whatever the counts
      say; a still-open reload gate is a nonmatch.

      RELEASE-PREDICTED cell.  A run whose window phenotype
      (``classify_event_word``) is unsep-window (bounded runs,
      overhanging reloads) is ROUTED to the gate-held certificate
      side: it matches iff it did not collapse AND its
      post-burn-in jet minimum stays above J_crit (the gate-held
      maintenance prediction on its logged trace).  Any other run
      matches iff it did not collapse.

      CENSORING.  A run with a censored phenotype censors its
      seed; any censored seed makes the attempt INDETERMINATE
      (the predeclared repeat is ``decide_e6_study``).

    PASS requires BOTH clauses: the total match count across all
    cells reaches the generated threshold (the error-controlling
    COUNT clause, calibrated at the least favorable member of the
    declared crossed family), AND every cell reaches its declared
    seed majority (the COVERAGE clause, which never raises the
    null pass rate and has independent content at the generated
    threshold: a design with one unpredicted schedule and the
    count carried by the other three fails coverage; for the primary
    design, (10, 10, 10, 3) reaches the count threshold and fails
    one cell's coverage requirement).

    Returns dict(verdict, total, cells=per-cell reports with
    per-run matches and routing, version)."""
    validate_e6_design(cells, config)
    K = int(config["total_threshold"])
    J_crit = float(config["J_crit"])
    burn_frac = float(config["burn_frac"])
    majority = int(config["cell_majority"])
    cells_required = int(config["cells_required"])
    total = 0
    n_cells_ok = 0
    cell_out = {}
    censored = False
    killed = False
    kill_reasons = []
    for cname, cell in cells.items():
        pred = cell["predicted"]
        run_out = []
        n_match = 0
        for i, rec in enumerate(cell["seeds"]):
            word = np.asarray(rec["word"], dtype=int).ravel()
            act = np.asarray(rec["act"], dtype=float).ravel()
            endpoint = derive_capture_endpoint(
                rec, J_crit, config["min_persistence_probes"],
                where=f"e6 {cname}[{i}]", require_step_one=True)
            J = endpoint["jet_step"]
            st = classify_event_word(word, act)
            captured = endpoint["status"] == "finite_horizon_entry"
            routed = False
            match = None
            gate_closed = None
            if st["phen"] == "censored" or endpoint["status"] == "censored":
                censored = True
            elif pred == "capture":
                gate_closed = False
                if captured:
                    cstep = int(endpoint["entry"])
                    last_event_step = (st["last_event"] + 1
                                       if st["last_event"] >= 0 else 0)
                    if last_event_step >= cstep:
                        killed = True
                        kill_reasons.append(
                            f"{cname}[{i}]: capture causal "
                            "order (event cessation does not "
                            "strictly precede derived entry)")
                    gate_closed = not np.any(act[cstep - 1:] > 0)
                match = captured and gate_closed
            else:
                if st["phen"] == "unsep-window":
                    routed = True
                    burn = int(burn_frac * len(J))
                    held = float(np.min(J[burn:])) > J_crit
                    match = (not captured) and held
                else:
                    match = not captured
            if match:
                n_match += 1
            run_out.append(dict(seed=i, phen=st["phen"],
                                routed=routed, match=match,
                                gate_closed_after_entry=(
                                    gate_closed
                                    if pred == "capture" else None),
                                endpoint={k: endpoint[k] for k in
                                          ("status", "entry",
                                           "trailing_probes",
                                           "historical_terminal_all_flat")}))
        ok = n_match >= majority
        n_cells_ok += int(ok)
        total += n_match
        cell_out[cname] = dict(
            predicted=pred, schedule_certificate=cell["schedule_certificate"],
            matches=n_match, majority=ok, runs=run_out)
    if killed:
        verdict = "KILL"
    elif censored:
        verdict = "INDETERMINATE"
    elif total >= K and n_cells_ok >= cells_required:
        verdict = "PASS"
    else:
        verdict = "FAIL"
    return dict(verdict=verdict, total=total, cells=cell_out,
                reasons=kill_reasons, version=config["version"])


def decide_e6_study(attempts, config):
    """THE E6 STUDY ORCHESTRATOR: the explicit repeat state of the
    censoring policy (one separately configured fresh-seed repeat
    after a first INDETERMINATE; a second indeterminate outcome has
    the configured terminal status), executed and simulated exactly
    like ``decide_e5_study``."""
    _check_study_config(config, "e6")
    if not isinstance(attempts, (list, tuple)) \
            or not 1 <= len(attempts) <= len(config["attempts"]):
        raise DecisionSchemaError(
            "e6 study: attempts must be a list of one or "
            f"{len(config['attempts'])} designs")
    out1 = decide_e6(attempts[0], config["attempts"][0])
    if out1["verdict"] != "INDETERMINATE":
        if len(attempts) > 1:
            raise DecisionSchemaError(
                "e6 study: a second attempt is only allowed "
                "after a first INDETERMINATE outcome")
        return dict(verdict=out1["verdict"], attempts=[out1],
                    repeat_pending=False)
    if len(attempts) == 1:
        return dict(verdict="INDETERMINATE", attempts=[out1],
                    repeat_pending=True)
    out2 = decide_e6(attempts[1], config["attempts"][1])
    if out2["verdict"] == "INDETERMINATE":
        return dict(verdict=config["second_indeterminate"],
                    attempts=[out1, out2],
                    repeat_pending=False)
    return dict(verdict=out2["verdict"], attempts=[out1, out2],
                repeat_pending=False)


# --------------------- the E6 crossed null family, enumerated


def _bernoulli_sum_dist(pis):
    """Exact distribution of a sum of independent Bernoulli(p_i)
    variables (Poisson-binomial), by convolution."""
    d = np.zeros(len(pis) + 1)
    d[0] = 1.0
    for p in pis:
        d[1:] = d[1:] * (1.0 - p) + d[:-1] * p
        d[0] *= (1.0 - p)
    return d


def e6_pass_prob_exact(K, config_shape, p, delta, w, p_seed=None):
    """EXACT pass probability of the E6 count-and-coverage clause
    pair under one member of the crossed cell-by-seed family.

    The family (the declared dependence model of the paired
    allocation): per seed j a shared latent u_j ~ Bernoulli(p_u)
    (drawn once, shared across cells: the pairing the allocation
    creates); per cell c an independent two-point cell effect
    p_c = clip(p +/- delta) with probability one half each; the
    match indicator M_{c,j} ~ Bernoulli(pi_{c,j}) conditionally
    independent given the latents, with

        pi_{c,j} = clip((1 - w) * p_c + w * u_j, 0, 1) .

    Marginal mean p (null: one half); achieved intra-cell
    correlation 4 (1-w)^2 delta^2; achieved cross-cell same-seed
    correlation w^2.  w = 1 is the fully shared-seed member, where
    every cell copies the seed coin; w = 0, delta = 0 is the
    independent-cell corner.

    Computation is exact.  Exchangeability reduces the shared-seed
    vectors to their `seeds + 1` possible success counts, weighted by
    the exact binomial multiplicity; the two-point cell effect is
    marginalized inside each cell's Poisson-binomial count
    distribution, followed by a joint
    (total, majority-count) recursion across cells.
    ``config_shape`` carries (seeds_per_cell, n_cells,
    cell_majority, cells_required)."""
    seeds = int(config_shape["seeds_per_cell"])
    n_cells = int(config_shape["n_cells"])
    majority = int(config_shape["cell_majority"])
    cells_required = int(config_shape["cells_required"])
    if p_seed is None:
        p_seed = p
    total_pass = 0.0
    for n_shared in range(seeds + 1):
        pu = (math.comb(seeds, n_shared)
              * p_seed ** n_shared
              * (1.0 - p_seed) ** (seeds - n_shared))
        if pu == 0.0:
            continue
        u = [1] * n_shared + [0] * (seeds - n_shared)
        dsum = None
        for sgn in (1.0, -1.0):
            pc = min(max(p + sgn * delta, 0.0), 1.0)
            pis = [min(max((1.0 - w) * pc + w * x, 0.0), 1.0)
                   for x in u]
            d = _bernoulli_sum_dist(pis)
            dsum = d if dsum is None else dsum + d
        dcell = 0.5 * dsum
        # joint recursion over the (conditionally iid) cells
        state = {(0, 0): 1.0}
        for _ in range(n_cells):
            new = {}
            for (tot, nm), pr in state.items():
                for k in range(seeds + 1):
                    if dcell[k] == 0.0:
                        continue
                    key = (tot + k,
                           nm + (1 if k >= majority else 0))
                    new[key] = new.get(key, 0.0) + pr * dcell[k]
            state = new
        pp = sum(pr for (tot, nm), pr in state.items()
                 if tot >= K and nm >= cells_required)
        total_pass += pu * pp
    return float(total_pass)


def e6_family_grid(spec=None):
    """The declared (delta, w) grid of the crossed null family
    (from the versioned decision-constant module), always
    including the two closed-form corners."""
    if spec is None:
        spec = _dc.E6_SPEC
    fam = spec["null_family"]
    deltas = np.round(np.arange(
        0.0, fam["delta_max"] + 1e-12, fam["delta_step"]), 10)
    ws = np.round(np.arange(
        0.0, fam["w_max"] + 1e-12, fam["w_step"]), 10)
    return [float(d) for d in deltas], [float(w) for w in ws]


def e6_worst_null_pass(K, config_shape, spec=None):
    """The LEAST FAVORABLE member of the declared crossed null
    family for the count-and-coverage pair at threshold K: exact
    enumeration at EVERY member of the declared grid (the two
    closed-form corners included).  The declared null family is
    this dense grid together with its corners; alongside the
    worst member, the largest pass-probability step between
    adjacent grid members is reported (the mesh-sensitivity
    witness printed with the calibration, so the resolution of
    the declared grid is auditable).  Returns dict(worst, delta,
    w, max_adjacent_step, n_members)."""
    if spec is None:
        spec = _dc.E6_SPEC
    deltas, ws = e6_family_grid(spec)
    p = spec["null_family"]["p_match"]
    vals = np.zeros((len(deltas), len(ws)))
    for a, d in enumerate(deltas):
        for b, w in enumerate(ws):
            vals[a, b] = e6_pass_prob_exact(K, config_shape, p, d,
                                            w)
    worst = float(vals.max())
    ai, bi = np.unravel_index(int(vals.argmax()), vals.shape)
    step = 0.0
    if vals.shape[0] > 1:
        step = max(step, float(np.abs(np.diff(vals,
                                              axis=0)).max()))
    if vals.shape[1] > 1:
        step = max(step, float(np.abs(np.diff(vals,
                                              axis=1)).max()))
    return dict(worst=worst, delta=float(deltas[ai]),
                w=float(ws[bi]), max_adjacent_step=step,
                n_members=int(vals.size))


def e6_calibrate_threshold(config_shape, alpha, spec=None):
    """Choose the E6 count threshold AT GENERATION: the smallest
    total K whose worst-case exact false-pass probability over the
    declared crossed null family is at or below alpha.  The two
    closed-form corner members (independent cells; fully shared
    seeds) are evaluated first as a cheap lower bound on the worst
    case, so the full grid enumeration runs only at candidate
    thresholds.  Returns dict(threshold, worst=the worst-case
    record of ``e6_worst_null_pass`` at the chosen K,
    corners=the exact corner values at the chosen K and at K-1
    for the permanent regressions)."""
    if spec is None:
        spec = _dc.E6_SPEC
    n_tot = (int(config_shape["seeds_per_cell"])
             * int(config_shape["n_cells"]))
    include_shared = bool(
        spec["null_family"].get("include_shared_corner", True))
    for K in range(n_tot + 1):
        shared = e6_pass_prob_exact(K, config_shape, 0.5, 0.0,
                                    1.0)
        indep = e6_pass_prob_exact(K, config_shape, 0.5, 0.0, 0.0)
        cheap_worst = max(shared, indep) if include_shared else indep
        if cheap_worst > alpha:
            continue
        rec = e6_worst_null_pass(K, config_shape, spec)
        if rec["worst"] <= alpha:
            corners = dict(
                shared_at_K=shared,
                shared_at_Kminus1=e6_pass_prob_exact(
                    K - 1, config_shape, 0.5, 0.0, 1.0),
                independent_at_K=indep,
            )
            return dict(threshold=K, worst=rec, corners=corners)
    raise RuntimeError(
        "no count threshold controls the declared alpha at the "
        "least favorable member of the declared crossed family")


def e6_search_attempt_designs(spec=None):
    """Select both immutable E6 attempts from preregistered candidates.

    Candidates are evaluated in declared order.  A candidate is selected
    only when exact crossed-family calibration controls its allocated
    alpha and exact planning power reaches the registered floor.  The
    returned ledger records every evaluated candidate and its rejection
    reason, so the printed constants are outputs of executable design
    selection rather than hand-edited thresholds.
    """
    if spec is None:
        spec = _dc.E6_SPEC
    search = spec["design_search"]
    floor = float(search["power_floor"])
    alt = spec["alt_planning"]
    selected = []
    calibrations = []
    ledger = []
    for base in spec["attempts"]:
        attempt_rows = []
        chosen = None
        chosen_cal = None
        candidates = search["candidates"][base["attempt_id"]]
        for candidate in candidates:
            shape = {key: int(candidate[key]) for key in
                     ("seeds_per_cell", "n_cells", "cell_majority",
                      "cells_required")}
            cal = e6_calibrate_threshold(shape, base["alpha"], spec)
            power = e6_pass_prob_exact(
                cal["threshold"], shape, alt["p_match"], alt["delta"],
                alt["w"])
            accepted = cal["worst"]["worst"] <= base["alpha"] \
                and power >= floor
            attempt_rows.append(dict(
                shape=shape, threshold=cal["threshold"],
                worst_null=cal["worst"]["worst"],
                planning_power=float(power), accepted=bool(accepted)))
            if accepted:
                chosen = dict(base)
                chosen.update(shape)
                chosen_cal = cal
                break
        if chosen is None:
            raise RuntimeError(
                f"no E6 candidate meets alpha and power for "
                f"{base['attempt_id']}")
        selected.append(chosen)
        calibrations.append(chosen_cal)
        ledger.append(dict(attempt_id=base["attempt_id"],
                           evaluated=attempt_rows,
                           selected_shape={key: chosen[key] for key in
                                           ("n_cells", "seeds_per_cell",
                                            "cell_majority",
                                            "cells_required")}))
    return dict(power_floor=floor, selected_templates=selected,
                calibrations=calibrations, ledger=ledger)
