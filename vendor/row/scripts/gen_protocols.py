#!/usr/bin/env python3
"""Generate the E-series protocol documents from the calibrated
constants, under three mechanically enforced gates, IN ORDER: theory
first, then completeness, then power.

GATE 0 (theory, units, box, constants): generation ABORTS before any
power computation if any of the following fails at the planning
constants, and the full check list is archived as
experiments/protocols/gate_report.json:
  * theorem hypotheses: the invariant power cap with the clamp
    margin (S+); the event termination bound and licensed reset
    interval; the EXACT log-drain identity, enclosure, exit margin,
    and bounds verified on a planning event run through
    theory.run_event (the ONE event clock; the cross-layer identity
    check); the capture envelope balance; the step conditions
    (S+)/(Q+) of the gated model; the regime clauses themselves
    (the planning instance collapses under capture, maintains the
    jet with COMPLETED reloads on the two-phase separated arm, and
    maintains it with OVERHANGING reloads on the constant
    gate-held arm, all through theory.gated_model_step); and the
    COMPOSITIONAL gate: every planning orbit's label must equal
    its window phenotype under theory.classify_event_word (the
    separated label is certified by the classifier and the
    zero-counter evidence, never by a gap or reload-positivity
    proxy);
  * unit conventions: the frozen sharpness unit, the carrier unit
    identity z = w^2 with amplitude factor sqrt(rho_w), the
    declared rate unit, the two-condition event end, and the
    reload-duration activity unit;
  * certificate box: the declared box has positive Jury margins
    (theory.cp_dir_bounds raises otherwise), the schedule-implied
    decay parameters of every cell lie inside the box, and the grid
    sanity statistic lies below the certified constant;
  * optimizer constants: the calibration equals the training stack
    (eps, betas parsed from train_run.py / measure_fidelity.py).

GATE 1 (completeness): every protocol is a structured object with
required design fields (estimand, population, unit of replication,
sample size, allocation, intervention, exclusion, estimator,
uncertainty model, alpha, multiplicity, effect model, decision
partition, power, contingency).  A recursive walk over every rendered
field fails generation on a missing value, an unresolved template, or
a deferred value outside its declared scope.  Deferred values (E0
outputs consumed by E1-E7) are typed objects carrying their producer
and binding formula; E0 itself may contain none.

GATE 2 (power): every protocol carries a simulated power from a
direct Monte Carlo of its actual sampling law and decision rule at
the calibrated (planning) effect size; generation fails below the
pre-specified minimum POWER_MIN = 0.80.  Planning effect sizes come
from the calibration file's ``planning`` block and are re-evaluated
at E0 before launch; the designs are powered at the planning values.

Writes experiments/protocols/e0..e7 + README + gate_report.json and
the manuscript's docs/figures/protocol_macros.tex.
"""

import dataclasses
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
EXP = os.path.join(ROOT, "experiments")
sys.path.insert(0, os.path.join(EXP, "analysis"))
import numpy as np  # noqa: E402
import theory  # noqa: E402
import decision_constants  # noqa: E402

CALPATH = os.path.join(EXP, "calibration", "calibration.json")
OUTDIR = os.path.join(EXP, "protocols")
MACROPATH = os.path.join(ROOT, "docs", "figures",
                         "protocol_macros.tex")

POWER_MIN = 0.80
DELTA_U = 0.05     # global strictness slack: u_max <= 1 - delta_u
FORBIDDEN = ("TBD", "to be reported", "{")


# ----------------------------------------------------------- schema


@dataclasses.dataclass(frozen=True)
class DeferredValue:
    """A typed placeholder for an E0-measured quantity consumed by a
    later protocol.  Allowed only in protocols that depend on E0;
    always paired in the text with the planning value the design was
    powered at."""
    producer: str
    name: str
    formula: str

    def render(self):
        return (f"[{self.producer} output {self.name}; binding "
                f"formula: {self.formula}]")


@dataclasses.dataclass
class Protocol:
    id: str
    title: str
    depends_on: list
    kill_condition: str
    estimand: str
    population: str
    unit_of_replication: str
    sample_size: str
    allocation: str
    intervention: str
    exclusion: str
    estimator: str
    uncertainty_model: str
    alpha: str
    multiplicity: str
    effect_model: str
    decision_partition: str
    power: str
    contingency: str
    body_extra: str = ""


def _assert_complete(obj, path, allow_deferred):
    if obj is None:
        raise SystemExit(f"gen_protocols: {path} is unresolved (None)")
    if isinstance(obj, DeferredValue):
        if not allow_deferred:
            raise SystemExit(
                f"gen_protocols: {path} defers to {obj.producer} but "
                "the protocol does not (or may not) depend on it")
        _assert_complete(obj.render(), path + ".render", True)
        return
    if isinstance(obj, str):
        for bad in FORBIDDEN:
            if bad in obj:
                raise SystemExit(
                    f"gen_protocols: {path} contains forbidden "
                    f"fragment {bad!r}: {obj[:80]!r}")
        return
    if isinstance(obj, (int, float)):
        if isinstance(obj, float) and not math.isfinite(obj):
            raise SystemExit(f"gen_protocols: {path} is not finite")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            _assert_complete(v, f"{path}.{k}", allow_deferred)
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _assert_complete(v, f"{path}[{i}]", allow_deferred)
        return
    if dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            _assert_complete(getattr(obj, f.name),
                             f"{path}.{f.name}", allow_deferred)
        return
    raise SystemExit(f"gen_protocols: {path} has unrenderable type "
                     f"{type(obj).__name__}")


def _render(p: Protocol):
    allow_deferred = "E0" in p.depends_on
    _assert_complete(p, p.id, allow_deferred)
    deps = ", ".join(p.depends_on) if p.depends_on else "none"
    rows = [
        ("Estimand", p.estimand),
        ("Population", p.population),
        ("Unit of replication", p.unit_of_replication),
        ("Sample size", p.sample_size),
        ("Allocation", p.allocation),
        ("Intervention", p.intervention),
        ("Exclusion", p.exclusion),
        ("Estimator", p.estimator),
        ("Uncertainty model", p.uncertainty_model),
        ("Alpha", p.alpha),
        ("Multiplicity", p.multiplicity),
        ("Effect model", p.effect_model),
    ]
    table = "\n".join(f"- **{k}.** {v}" for k, v in rows)
    return f"""# {p.id.upper()}: {p.title}

Gated by: {deps}.  Pre-specified design; generated by
`scripts/gen_protocols.py` under the theory/units/box/constants gate
(gate_report.json), the completeness gate, and the power gate
(minimum simulated power {POWER_MIN:.2f}).

## Design schema
{table}

{p.body_extra}
## Decision rule (full partition)
{p.decision_partition}

## Power (simulated at generation time)
{p.power}

## Contingency
{p.contingency}

## Kill target
{p.kill_condition}
"""


def _gate_power(pid, value):
    if not value >= POWER_MIN:
        raise SystemExit(
            f"gen_protocols: {pid} simulated power {value:.2f} is "
            f"below the pre-specified minimum {POWER_MIN:.2f}; "
            "redesign the protocol (the design must grow) instead of "
            "printing an underpowered plan")
    return value


# --------------------------------------------------- gate 0 (theory)


def granted_fuel(gp, pl):
    """The whole-box GRANTED FUEL at the planning constants: the
    printed value the totality theorem cites, and the ONE fuel the
    generator instantiates for every executable step and event it
    runs.  Fuel is model data with no hidden default; the [thm]
    uniform-fuel gate asserts the identity between this value and
    the per-constituent computation, so the instantiated and the
    printed value cannot drift apart."""
    bu = gp["burst"]
    b_star = gp["d0"] / gp["mu_b"]
    y0_max = gp["c_recurrent"] + pl["eps_max_event"] - b_star
    zcap = theory.z_cap_invariant(
        bu["h"], pl["eps_max_event"], bu["b_sat"],
        w_max2=pl["w_max2_event"], z_seed=bu["z_seed"])
    return theory.event_fuel_required_uniform(
        y0_max, gp["d0"] / gp["mu_b"], gp["c_recurrent"],
        gp["eta"], gp["th_J_loss"], gp["lam_wd_model"], bu["h"],
        bu["b_sat"], zcap, bu["z_floor"], bu["y_floor"],
        bu["tau_z"])


def _gated_params_from(gp, n):
    """Planning-instance parameter dict for theory.gated_model_step
    with n uncoupled replica sites."""
    return dict(
        beta=np.zeros((n, n)),
        d0=np.full(n, gp["d0"]),
        delta_a=np.full(n, gp["delta_a"]),
        delta_J0=np.full(n, gp["delta_J0"]),
        kappa=np.full(n, gp["kappa"]),
        th_a_loss=np.full(n, gp["th_a_loss"]),
        th_g_loss=np.full(n, gp["th_g_loss"]),
        th_J_loss=np.full(n, gp["th_J_loss"]),
        lam_wd=gp["lam_wd_model"],
        mu_b=gp["mu_b"],
        s_shed=gp["s_shed"],
        burst=dict(gp["burst"]),
    )


def _gated_state_from(gp, n):
    return dict(
        b=np.full(n, gp["d0"] / gp["mu_b"] * 0.5),
        a=np.full(n, 1.0), g=np.full(n, 1.0),
        J=np.full(n, gp["J0"]), w=np.zeros(n),
        pend=np.zeros(n), act=np.zeros(n),
    )


def _gated_dichotomy(gp, fuel, n=1, rng=None, d0_jitter_rel=0.0,
                     T_cap=4000, T_rec=5000):
    """Run the planning instance of the gated model through the
    capture and constant (gate-held) arms (one update law; the
    schedule decides) and return the per-site outcomes, including
    the cycle statistics of the constant arm (inter-event gap
    floor, toppling-step fraction, written reload durations) and
    the full per-site event words, counters, and jets the decision
    function consumes.  The constant arm's cycles are GATE-HELD:
    the written reloads overhang the gaps, no pair carries a
    completed reload, and the classifier certifies the
    unsep-window phenotype (the separated arm is
    ``_gated_separated``, whose reloads complete).  Used by gate 0
    (deterministic, n = 1) and by the E5 operating characteristic
    (n replicas with jitter and seed noise)."""
    eta = gp["eta"]
    q = 1.0 - eta * gp["th_J_loss"]
    bound = theory.entry_time_bound(gp["J0"], gp["J_crit"], q)
    base_d0 = gp["d0"]
    jit = np.ones(n)
    if rng is not None and d0_jitter_rel > 0.0:
        jit = 1.0 + d0_jitter_rel * rng.standard_normal(n)

    # captured arm: thresholds above every loaded value
    params = _gated_params_from(gp, n)
    params["d0"] = base_d0 * jit
    state = _gated_state_from(gp, n)
    c_cap = np.full(n, gp["c_captured"])
    cap_topples = np.zeros(n)
    entry = np.full(n, -1, dtype=int)
    cap_word = np.zeros((T_cap, n), dtype=np.int8)
    cap_act = np.zeros((T_cap, n), dtype=np.float32)
    cap_J = np.zeros((T_cap, n), dtype=np.float32)
    for t in range(T_cap):
        state, topple = theory.gated_model_step(state, params, c_cap,
                                                eta, fuel, rng=rng)
        cap_topples += topple
        newly = (entry < 0) & (state["J"] <= gp["J_crit"])
        entry[newly] = t
        cap_word[t] = topple
        cap_act[t] = state["act"]
        cap_J[t] = state["J"]
    cap_J_end = state["J"].copy()

    # recurrent arm: constant schedule with sustained drive
    params = _gated_params_from(gp, n)
    params["d0"] = base_d0 * jit
    state = _gated_state_from(gp, n)
    c_rec = np.full(n, gp["c_recurrent"])
    rec_topples = np.zeros(n)
    rec_J_min = np.full(n, np.inf)
    last_topple = np.full(n, -1, dtype=int)
    rec_min_gap = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    rec_reload_min = np.full(n, np.iinfo(np.int64).max,
                             dtype=np.int64)
    rec_word = np.zeros((T_rec, n), dtype=np.int8)
    rec_act = np.zeros((T_rec, n), dtype=np.float32)
    rec_J = np.zeros((T_rec, n), dtype=np.float32)
    site0_act = []
    site0_topple = []
    site0_r = []
    for t in range(T_rec):
        j_before0 = float(state["J"][0])
        state, topple = theory.gated_model_step(state, params, c_rec,
                                                eta, fuel, rng=rng)
        rec_topples += topple
        for i in np.flatnonzero(topple):
            if last_topple[i] >= 0:
                rec_min_gap[i] = min(rec_min_gap[i],
                                     t - last_topple[i])
            last_topple[i] = t
            rec_reload_min[i] = min(rec_reload_min[i],
                                    int(state["act"][i]))
        if t >= 500:
            rec_J_min = np.minimum(rec_J_min, state["J"])
        rec_word[t] = topple
        rec_act[t] = state["act"]
        rec_J[t] = state["J"]
        # site-0 series for the maintenance certificates: the gate
        # indicator (never a gap length), topple marks, and the
        # per-event jet retention realized by the orbit
        site0_act.append(float(state["act"][0]))
        site0_topple.append(bool(topple[0]))
        if topple[0]:
            site0_r.append(float(state["J"][0])
                           / max(j_before0, 1e-300))
    rec_frac = rec_topples / float(T_rec)
    return dict(cap_topples=cap_topples, entry=entry, bound=bound,
                cap_J_end=cap_J_end, rec_topples=rec_topples,
                rec_J_min=rec_J_min, rec_min_gap=rec_min_gap,
                rec_topple_frac=rec_frac,
                rec_reload_min=rec_reload_min,
                cap_word=cap_word, cap_act=cap_act, cap_J=cap_J,
                rec_word=rec_word, rec_act=rec_act, rec_J=rec_J,
                site0_act=site0_act, site0_topple=site0_topple,
                site0_r=site0_r)


def _gated_separated(gp, fuel, n=1, rng=None, d0_jitter_rel=0.0,
                     T_sep=None):
    """Run the TWO-PHASE SEPARATED maintenance arm of the planning
    instance: threshold ``c_sep_hi`` on all steps except every
    ``sep_period``-th, which drops to ``c_recurrent`` and fires
    the event.  The written reload (about 8 counts at the planning
    constants) expires strictly inside every gap, the gate CLOSES
    between bursts, and the classifier certifies the sep-window
    phenotype with ZERO unseparated pairs: separated burst-reload
    cycles by COMPLETED RELOAD, the one meaning of separation.
    Returns per-site words, counters, and jets, plus summary
    statistics."""
    eta = gp["eta"]
    P = int(gp["sep_period"])
    if T_sep is None:
        T_sep = int(gp["T_maint"])
    base_d0 = gp["d0"]
    jit = np.ones(n)
    if rng is not None and d0_jitter_rel > 0.0:
        jit = 1.0 + d0_jitter_rel * rng.standard_normal(n)
    params = _gated_params_from(gp, n)
    params["d0"] = base_d0 * jit
    state = _gated_state_from(gp, n)
    c_hi = np.full(n, gp["c_sep_hi"])
    c_lo = np.full(n, gp["c_recurrent"])
    sep_word = np.zeros((T_sep, n), dtype=np.int8)
    sep_act = np.zeros((T_sep, n), dtype=np.float32)
    sep_J = np.zeros((T_sep, n), dtype=np.float32)
    sep_J_min = np.full(n, np.inf)
    burn = T_sep // 10
    for t in range(T_sep):
        c_t = c_lo if t % P == 0 else c_hi
        state, topple = theory.gated_model_step(state, params, c_t,
                                                eta, fuel, rng=rng)
        sep_word[t] = topple
        sep_act[t] = state["act"]
        sep_J[t] = state["J"]
        if t >= burn:
            sep_J_min = np.minimum(sep_J_min, state["J"])
    return dict(sep_word=sep_word, sep_act=sep_act, sep_J=sep_J,
                sep_J_min=sep_J_min)


def _run_gates(cal, pl, write=True):
    """GATE 0: theory hypotheses, unit conventions, certificate box,
    and optimizer constants, checked BEFORE any power computation.
    Aborts generation on any failure; archives the full check list
    as gate_report.json (unless write=False).  Returns the derived
    reference quantities the protocols print.

    Every check name carries its CLASS tag, so the report never
    overstates what was checked:
      [thm]   a theorem hypothesis or theorem constant evaluated at
              the planning constants (available planning-instance
              gates; they do not verify E0-measured hypotheses);
      [orbit] a property of one selected planning trajectory
              through the gated model (a realized-path check, not a
              box-quantified certificate);
      [unit]  a declared unit or convention consistency check;
      [box]   a certificate-box or certified-band check;
      [stack] agreement between calibration and the training stack.
    E0-measured hypotheses (the registry items deferred to E0) are
    NOT checked here; the protocols carry them as typed deferred
    values."""
    checks = []
    fails = []

    def check(name, ok, detail=""):
        checks.append({"name": name, "ok": bool(ok),
                       "detail": str(detail)})
        if not ok:
            fails.append(name)

    # the ONE event clock at the planning instance: invariant cap,
    # (S+), termination bound, licensed reset, and the exact
    # log-drain identity checked at generation (the cross-layer
    # check: the theorem constants and the map share run_event)
    gp = cal["gated_model_planning"]
    bu = gp["burst"]
    b_star = gp["d0"] / gp["mu_b"]
    y0_max = gp["c_recurrent"] + pl["eps_max_event"] - b_star
    fuel_pl = granted_fuel(gp, pl)   # the ONE instantiated fuel
    zcap = None
    n_max = None
    r_lic = None
    try:
        zcap = theory.z_cap_invariant(
            bu["h"], pl["eps_max_event"], bu["b_sat"],
            w_max2=pl["w_max2_event"], z_seed=bu["z_seed"])
        u_max = theory.event_u_max(
            gp["eta"], gp["th_J_loss"], gp["lam_wd_model"],
            gp["kappa"], zcap)
        check("[thm] invariant power cap and clamp margin (S+)",
              True,
              f"Z_cap = {zcap:.4f}, u_max = {u_max:.4f} <= 1 "
              "(the clamp never binds on the box)")
        check("[thm] strict global margin: u_max <= 1 - delta_u",
              u_max <= 1.0 - DELTA_U,
              f"u_max = {u_max:.4f} <= {1.0 - DELTA_U} (slack "
              f"delta_u = {DELTA_U}: the logarithmic drain law and "
              "the positive lower reset factor hold with explicit "
              "constants on the whole box, never at the boundary "
              "u_max = 1)")
    except ValueError as e:
        check("[thm] invariant power cap and clamp margin (S+)",
              False, e)
        u_max = None
    try:
        n_max = theory.event_time_bound(
            y0_max, b_star, gp["c_recurrent"], gp["eta"],
            gp["th_J_loss"], gp["lam_wd_model"], bu["h"], zcap,
            bu["z_floor"])
        r_lic = theory.reset_interval_licensed(
            y0_max, b_star, gp["c_recurrent"], gp["eta"],
            gp["th_J_loss"], gp["lam_wd_model"], gp["kappa"], zcap,
            n_max)
        check("[thm] event termination bound and licensed reset "
              "interval", True,
              f"N <= {n_max} steps; retained fraction in "
              f"[{r_lic[0]:.4f}, {r_lic[1]:.4f}]")
    except ValueError as e:
        check("[thm] event termination bound and licensed reset "
              "interval", False, e)
    # the entry gap (B-gap): the totality theorem's domain
    # inequality, in its NORMATIVE per-step arrival-envelope form
    # (theory.b_gap_slack, the single source of the printed
    # display; the per-cycle form under separated cycles is the
    # corollary tier b_gap_slack_tsep and is never this gate).
    # The planning lattice is uncoupled (beta = 0, A_max = 0); a
    # weakly coupled instance exercises the amplified arrival term.
    delta_b = None
    fuel_req = None
    fuel_uniform = None
    try:
        delta_b = 0.5 * (gp["c_recurrent"] - b_star)
        a_max_pl = 0.0                      # beta = 0 lattice
        gap_slack = theory.b_gap_slack(
            gp["c_recurrent"], gp["d0"], gp["mu_b"], a_max_pl,
            gp["eta"], delta_b)
        gap_slack_tsep = theory.b_gap_slack_tsep(
            gp["c_recurrent"], gp["d0"], gp["mu_b"], a_max_pl,
            gp["eta"], delta_b)
        fuel_req = theory.event_fuel_required(
            y0_max, delta_b, gp["eta"], gp["th_J_loss"],
            gp["lam_wd_model"], bu["h"], zcap, bu["z_floor"])
        check("[thm] entry gap (B-gap, per-step arrival envelope) "
              "and gap-domain resolver fuel",
              gap_slack >= 0.0 and fuel_req <= fuel_pl,
              f"delta_b = {delta_b:.3f}, slack = {gap_slack:.4f} "
              ">= 0 (the normative inequality c_- >= d0_sup/mu_b "
              "+ A_max/(eta_-*mu_b) + delta_b of Lem. bgap, "
              "serialized from theory.b_gap_slack / B_GAP_LATEX; "
              f"A_max = {a_max_pl} at beta = 0); (T-sep) "
              f"corollary-tier slack {gap_slack_tsep:.4f}, "
              "reported, never gated; granted gap-domain fuel "
              f"(bound + 1, strict indexing) {fuel_req} <= the "
              f"instantiated whole-box fuel {fuel_pl}.  This "
              "inequality is also the (R-gap) "
              "hypothesis of the recurrent clause of the closure: "
              "every recurrent-cycle event enters on the gap "
              "domain and exits cold")
    except ValueError as e:
        check("[thm] entry gap (B-gap, per-step arrival envelope) "
              "and gap-domain resolver fuel", False, e)
    # the UNIFORM fuel of the totality theorem: the whole-box
    # bound (b <= c via the middle-region lemma, b > c via the
    # overloaded lemma), so the composed step at this fuel is a
    # self-map of the box (the no-rejection theorem)
    try:
        b_max_box = gp["d0"] / gp["mu_b"]     # beta = 0: no
        # arrival amplification through the base ceiling
        n_mid = theory.event_time_bound_middle(
            y0_max, gp["eta"], gp["th_J_loss"], gp["lam_wd_model"],
            bu["h"], bu["b_sat"], zcap, bu["z_floor"],
            bu["y_floor"])
        overloaded_empty = b_max_box <= gp["c_recurrent"]
        if overloaded_empty:
            n_over = 0
            over_txt = (f"overloaded region EMPTY at this box "
                        f"(b_max = {b_max_box:.3g} <= c_min = "
                        f"{gp['c_recurrent']:.3g}; its constituent "
                        "is zero by region emptiness, never by "
                        "silence)")
        else:
            n_over = theory.event_time_bound_overloaded_uniform(
                y0_max, gp["eta"], gp["th_J_loss"],
                gp["lam_wd_model"], bu["h"], bu["b_sat"], zcap,
                bu["y_floor"], bu["tau_z"])
            over_txt = (f"overloaded constituent {n_over} at the "
                        "worst exit target tau_z (the b -> c+ "
                        "limit; the target grows with the "
                        "overload, so the shallowest entry is the "
                        "hardest)")
        fuel_uniform = theory.event_fuel_required_uniform(
            y0_max, b_max_box, gp["c_recurrent"], gp["eta"],
            gp["th_J_loss"], gp["lam_wd_model"], bu["h"],
            bu["b_sat"], zcap, bu["z_floor"], bu["y_floor"],
            bu["tau_z"])
        assert fuel_uniform == max(n_mid, n_over) + 1
        check("[thm] uniform whole-box fuel (no-rejection "
              "self-map of the settled step; fuel is data, "
              "single-sourced)",
              fuel_uniform == fuel_pl,
              f"granted fuel N* + 1 = {fuel_uniform} = the "
              "instantiated fuel of every executable step and "
              "event the generator runs (no hidden default) "
              f"(N* = max of the b <= c middle-region bound "
              f"{n_mid} and the b > c overloaded bound; "
              f"{over_txt}; +1 is the strict-indexing convention: "
              "the resolver checks the exit at the top of each "
              "iteration; the resolver's Exhausted branch is "
              "unreachable from the box at or above this granted "
              "fuel, with th_J >= theta_- > 0, y_floor > 0, and "
              "tau_z > y_floor/b_sat, and every ended event "
              "emits through the terminal settlement E-settle)")
    except ValueError as e:
        check("[thm] uniform whole-box fuel (no-rejection "
              "self-map of the settled step)", False, e)
    # the two-phase retention floor, verified against run_event
    # over an entry grid at every generation (the theorem-tier
    # maintenance certificates consume this bound)
    try:
        r_floor = theory.retention_floor_two_phase(
            pl["eps_max_event"], delta_b, gp["eta"],
            gp["th_J_loss"], gp["lam_wd_model"], gp["kappa"],
            bu["h"], zcap, bu["z_floor"])
        worst_r = 1.0
        n_grid = 0
        for bfrac in (0.0, 0.5, 1.0):
            b_e = bfrac * (gp["c_recurrent"] - delta_b)
            for efrac in (0.02, 0.5, 1.0):
                eps0 = efrac * pl["eps_max_event"]
                y0 = eps0 + (gp["c_recurrent"] - b_e)
                j0 = math.sqrt(y0 / gp["kappa"])
                if j0 > 1.0:
                    continue
                for w0 in (0.0, 0.1):
                    ev_g = theory.run_event(
                        b_e, 1.0, 1.0, j0, w0, gp["c_recurrent"],
                        gp["eta"], gp["kappa"], gp["th_J_loss"],
                        gp["lam_wd_model"], bu, fuel=fuel_pl)
                    if not ev_g["ended"]:
                        raise ValueError(
                            f"grid event at b = {b_e} exhausted")
                    worst_r = min(worst_r, ev_g["J"] / j0)
                    n_grid += 1
        check("[thm] two-phase retention floor verified against "
              "run_event on an entry grid",
              0.0 < r_floor <= worst_r,
              f"floor {r_floor:.4f} <= worst realized retention "
              f"{worst_r:.4f} over {n_grid} grid events (uniform "
              "over gap-domain entries, independent of event "
              "length; Lem. retention)")
    except ValueError as e:
        check("[thm] two-phase retention floor verified against "
              "run_event on an entry grid", False, e)
    # pinned divergence control: the normative per-step gate and
    # the (T-sep) per-cycle formula are DIFFERENT inequalities and
    # can disagree at nonzero coupling; the gate must be the
    # normative one (regression for the formula-mismatch defect)
    _c, _d0, _mu, _A, _eta, _db = 0.5, 0.02, 0.1, 0.05, 0.05, 0.1
    _norm = theory.b_gap_slack(_c, _d0, _mu, _A, _eta, _db)
    _tsep = theory.b_gap_slack_tsep(_c, _d0, _mu, _A, _eta, _db)
    check("[thm] B-gap formula divergence control (normative gate "
          "fails where the per-cycle form passes)",
          _norm < 0.0 and _tsep > 0.0,
          f"at (c, d0, mu_b, A, eta, delta_b) = (0.5, 0.02, 0.1, "
          f"0.05, 0.05, 0.1): normative slack {_norm:.1f} < 0, "
          f"per-cycle slack {_tsep:.3f} > 0; the generator gates "
          "on the normative inequality only")
    # a WEAKLY COUPLED instance passing the normative envelope:
    # nonzero arrivals exercise the amplified term (the beta = 0
    # planning lattice cannot)
    pi_max_c = gp["s_shed"] * gp["kappa"] * 1.0 ** 2
    bc_max = None
    try:
        amp_ceiling = (gp["c_recurrent"]
                       - gp["d0"] / gp["mu_b"] - delta_b) / 2.0
        a_max_c = amp_ceiling * gp["eta"] * gp["mu_b"]
        bc_max = a_max_c / pi_max_c
        slack_c = theory.b_gap_slack(
            gp["c_recurrent"], gp["d0"], gp["mu_b"], a_max_c,
            gp["eta"], delta_b)
        check("[thm] coupled B-gap instance (nonzero arrivals "
              "through the per-step envelope)",
              slack_c >= 0.0 and bc_max > 0.0,
              f"A_max = {a_max_c:.2e} (column-sum ceiling b_c <= "
              f"{bc_max:.2e} at pi_max = {pi_max_c:.2f}), "
              f"amplified term A_max/(eta_-*mu_b) = "
              f"{a_max_c / (gp['eta'] * gp['mu_b']):.3f}, slack = "
              f"{slack_c:.4f} >= 0: the coupled envelope is "
              "exercised at generation, not only at beta = 0")
    except ValueError as e:
        check("[thm] coupled B-gap instance (nonzero arrivals "
              "through the per-step envelope)", False, e)
    # the carrier step condition reads the POST-ARRIVAL stress:
    # eta* X'_max <= 1 with X'_max = B_max + A_max + kappa*u_max^2
    # (the strengthened (S+) of the corrected well-posedness)
    x_prime_max = (gp["d0"] / gp["mu_b"]) + a_max_pl \
        + gp["kappa"] * 1.0 ** 2
    check("[thm] carrier post-arrival stress ceiling "
          "(eta* X'_max <= 1)",
          gp["eta"] * x_prime_max <= 1.0,
          f"X'_max = B_max + A_max + kappa*u_max^2 = "
          f"{x_prime_max:.3f}, eta* X'_max = "
          f"{gp['eta'] * x_prime_max:.4f} <= 1 (the multiplier "
          "the smooth carrier update actually reads is bounded "
          "at the post-arrival, pre-reversion base)")
    ev = theory.run_event(
        b_star, 1.0, 1.0, math.sqrt(y0_max / gp["kappa"]), 0.0,
        gp["c_recurrent"], gp["eta"], gp["kappa"], gp["th_J_loss"],
        gp["lam_wd_model"], bu, fuel=fuel_pl)
    tsep = dict(event_fast_steps=int(ev["n"]),
                sum_z=float(ev["sum_z"]),
                retained_fraction_sim=float(
                    ev["x_end"] / gp["c_recurrent"]))
    r_direct = ev["x_end"] / gp["c_recurrent"]
    r_exact = theory.retained_fraction_exact(
        b_star, gp["c_recurrent"], y0_max, ev["sum_log_phi"])
    d1 = theory.drain_linear(gp["eta"], gp["th_J_loss"],
                             gp["lam_wd_model"], gp["kappa"],
                             ev["n"], ev["sum_z"])
    exact_drain = -2.0 * ev["sum_log_phi"]
    if u_max is not None:
        d_lo, d_hi = theory.drain_enclosure(d1, u_max)
        enc_ok = (d_lo - 1e-12 <= exact_drain <= d_hi + 1e-12)
    else:
        d_lo = d_hi = None
        enc_ok = False
    ident_ok = (abs(r_direct - r_exact) < 1e-12
                and not ev["clamped"] and ev["ended"]
                and ev["eps_end"] <= 0.0
                and (n_max is None or ev["n"] <= n_max)
                and (r_lic is None
                     or r_lic[0] - 1e-9 <= r_direct <= r_lic[1]))
    check("[thm] single-clock identity (exact log-drain law, "
          "enclosure, exit margin, and bounds, on a planning event "
          "through run_event)", ident_ok and enc_ok,
          f"N = {ev['n']}, identity residual "
          f"{abs(r_direct - r_exact):.2e}, exact drain "
          f"{exact_drain:.4f} in [{d_lo:.4f}, {d_hi:.4f}]"
          if d_lo is not None else "enclosure unavailable")

    # the terminal settlement (E-settle): identity on the gap
    # domain (cold exits are untouched, so the gap-domain
    # reduction stays bit-exact), and the settled ended-event
    # carrier floor on a hot-exit entry (the box self-map's
    # carrier clause holds on BOTH exit branches)
    check("[unit] terminal settlement identity on the gap domain "
          "(E-settle; cold exits untouched)",
          ev["ended"] and ev["exit_type"] == "cold"
          and ev["d_set"] == 0.0 and ev["w_end"] == ev["w_exit"],
          f"planning event: exit_type = {ev['exit_type']}, "
          f"d_set = {ev['d_set']}, w_end == w_exit bit-equal "
          "(settlement is the identity on every cold exit)")
    try:
        ev_hot = theory.run_event(
            gp["c_recurrent"] + 0.2, 1.0, 1.0, 0.3, 0.05,
            gp["c_recurrent"], gp["eta"], gp["kappa"],
            gp["th_J_loss"], gp["lam_wd_model"], bu,
            fuel=fuel_pl)
        hot_ok = (ev_hot["ended"]
                  and ev_hot["exit_type"] == "hot"
                  and ev_hot["eps_end"] > 0.0
                  and abs(ev_hot["w_end"])
                  <= math.sqrt(bu["z_floor"]) + 1e-15
                  and ev_hot["d_set"] >= 0.0)
        check("[thm] settled ended-event carrier floor on a "
              "hot-exit entry (|w+| <= sqrt(z_floor))",
              hot_ok,
              f"overloaded entry b = c + 0.2: ended in "
              f"{ev_hot['n']} steps, exit_type = "
              f"{ev_hot['exit_type']}, pre-settlement |w_exit| = "
              f"{abs(ev_hot['w_exit']):.4f}, settled |w_end| = "
              f"{abs(ev_hot['w_end']):.4f} <= sqrt(z_floor) = "
              f"{math.sqrt(bu['z_floor']):.4f}, settled power "
              f"d_set = {ev_hot['d_set']:.4f} ledgered (E-settle: "
              "the exit applies the same truncation discipline "
              "as the seeded entry)")
    except ValueError as e:
        check("[thm] settled ended-event carrier floor on a "
              "hot-exit entry (|w+| <= sqrt(z_floor))", False, e)

    # the two-regime planning instance (also validates the step
    # conditions (S+)/(Q+): gated_model_step raises on any
    # violation, and REJECTS any fuel-exhausted event); the
    # recurrent arm must show SEPARATED cycles: a
    # one-topple-per-step orbit FAILS generation
    maint = None
    try:
        dic = _gated_dichotomy(gp, granted_fuel(gp, pl), n=1)
        cap_ok = (dic["cap_topples"][0] == 0 and dic["entry"][0] >= 0
                  and dic["entry"][0] <= dic["bound"]
                  and dic["cap_J_end"][0] <= gp["J_crit"])
        rec_ok = (dic["rec_topples"][0] >= 5
                  and dic["rec_J_min"][0] > gp["J_crit"])
        check("[orbit] captured/gate-held planning instances "
              "(capture collapses the jet within the printed "
              "bound; the constant schedule maintains it)",
              cap_ok and rec_ok,
              f"captured: entry {dic['entry'][0]} <= bound "
              f"{dic['bound']}, topples {dic['cap_topples'][0]:.0f}; "
              f"gate-held: topples {dic['rec_topples'][0]:.0f}, "
              f"min J {dic['rec_J_min'][0]:.3f} > J_crit "
              f"{gp['J_crit']} (a realized-path check; the "
              "certificate check is the maintenance gate below)")
        # the SEPARATED maintenance arm: the two-phase (R-sep)
        # schedule whose reloads COMPLETE; the gate is the
        # classifier itself plus the zero-counter evidence, never
        # a gap or reload-positivity proxy (one meaning of
        # separated: the completed reload)
        sep = _gated_separated(gp, granted_fuel(gp, pl), n=1)
        sep_stats = theory.classify_event_word(
            sep["sep_word"][:, 0], sep["sep_act"][:, 0])
        sep_ok = (sep_stats["phen"] == "sep-window"
                  and sep_stats["n_unsep"] == 0
                  and sep_stats["gate_closure_frac"] == 1.0
                  and sep_stats["max_run"] == 1
                  and float(sep["sep_J_min"][0]) > gp["J_crit"])
        check("[orbit] separated burst-reload cycles (the "
              "two-phase (R-sep) maintenance arm: COMPLETED "
              "reloads, certified by the classifier)", sep_ok,
              f"phenotype {sep_stats['phen']} with "
              f"{sep_stats['n_unsep']} unseparated of "
              f"{sep_stats['n_pairs']} pairs, gate-closure "
              f"fraction {sep_stats['gate_closure_frac']:.2f}, "
              f"max run {sep_stats['max_run']}, post-burn-in "
              f"min J {float(sep['sep_J_min'][0]):.3f} > J_crit "
              f"{gp['J_crit']}: every written reload expires "
              "strictly inside its gap and the gate closes "
              "between bursts")
        # the GATE-HELD arm: the constant schedule whose reloads
        # OVERHANG every gap (the terminal-reload theorem forbids
        # separated recurrence at constant schedule), gated as the
        # intermittent-class maintenance instance it is
        rec_stats = theory.classify_event_word(
            dic["rec_word"][:, 0], dic["rec_act"][:, 0])
        rec_events = np.flatnonzero(dic["rec_word"][:, 0])
        over_ok = all(
            dic["rec_act"][t1, 0] >= (t2 - t1)
            for t1, t2 in zip(rec_events[1:-1], rec_events[2:]))
        gh_ok = (rec_stats["phen"] == "unsep-window"
                 and rec_stats["unsep_frac"] == 1.0
                 and rec_stats["max_run"] == 1
                 and over_ok)
        check("[orbit] gate-held intermittent maintenance (the "
              "constant arm: overhanging reloads, certified by "
              "the classifier and the written-duration evidence)",
              gh_ok,
              f"phenotype {rec_stats['phen']} with unseparated "
              f"fraction {rec_stats['unsep_frac']:.2f}, max run "
              f"{rec_stats['max_run']}, min inter-event gap "
              f"{dic['rec_min_gap'][0]}, every written reload at "
              "least its following gap: the constant schedule "
              "maintains the jet with the gate held open "
              "(event fast-clock length is a diagnostic, not a "
              "duration)")
        # the COMPOSITIONAL gate: every planning orbit's label
        # equals its classifier phenotype (the cross-layer
        # assertion whose absence let a mislabeled orbit through)
        cap_stats = theory.classify_event_word(
            dic["cap_word"][:, 0], dic["cap_act"][:, 0])
        check("[orbit] gate labels equal classifier phenotypes "
              "(captured, separated, gate-held planning orbits)",
              (cap_stats["phen"] == "quiet-tail"
               and sep_stats["phen"] == "sep-window"
               and rec_stats["phen"] == "unsep-window"),
              f"captured: {cap_stats['phen']}; separated: "
              f"{sep_stats['phen']}; gate-held: "
              f"{rec_stats['phen']}: each planning orbit is "
              "located by the same classifier the decision rules "
              "consume")
        tsep["rec_min_gap"] = int(dic["rec_min_gap"][0])
        tsep["rec_topple_frac"] = float(dic["rec_topple_frac"][0])
        tsep["sep_n_pairs"] = int(sep_stats["n_pairs"])
        tsep["sep_gate_closure"] = float(
            sep_stats["gate_closure_frac"])
        tsep["sep_J_min"] = float(sep["sep_J_min"][0])

        # the DRIVEN planning cell: the classification's third
        # regime as an executable instance (the witness orbit of
        # the driven persistence theorem): under (O-reb) at the
        # witness constants, one-gap reload saturation
        # eta*deltaJ0 = 2 with drive kappa*(a g)^2 = 1 above
        # c = 0.5, the site must topple at EVERY bounded global
        # step with the reload counter renewed, even as the base
        # relaxes far below threshold (base relaxation is
        # irrelevant to the full rebuilt guard)
        drv_burst = dict(h=0.1, b_sat=1.0, z_seed=0.01,
                         z_floor=0.001, y_floor=0.01, tau_z=0.02,
                         w_kick_max=0.1)
        drv_params = dict(beta=np.zeros((1, 1)), d0=0.0,
                          delta_a=0.0, delta_J0=20.0, kappa=1.0,
                          th_a_loss=0.0, th_g_loss=0.0,
                          th_J_loss=0.1, lam_wd=0.0, mu_b=0.1,
                          s_shed=0.0, burst=drv_burst)
        drv_state = dict(b=np.array([1.0]), a=np.array([1.0]),
                         g=np.array([1.0]), J=np.array([1.0]),
                         w=np.array([0.0]), pend=np.array([0.0]),
                         act=np.array([1.0]))
        drv_margin = theory.driven_rebuild_margin(
            0.1, 20.0, 1.0, 1.0, 1.0, 0.5, 0.1, 0.0)
        drv_all = True
        drv_b_end = None
        for _t in range(300):
            drv_state, drv_top = theory.gated_model_step(
                drv_state, drv_params, np.array([0.5]), 0.1,
                fuel_pl)
            if not bool(drv_top[0]) or drv_state["act"][0] < 1.0:
                drv_all = False
                break
            drv_b_end = float(drv_state["b"][0])
        check("[orbit] driven planning cell (the classification's "
              "third regime, executable)",
              drv_margin > 0.0 and drv_all
              and drv_b_end is not None and drv_b_end < 0.5,
              f"(O-reb) margin kappa*(a g)^2 - c = "
              f"{drv_margin:.3f} > 0 at saturation "
              "eta*deltaJ0 = 2; the witness orbit topples at all "
              "300 bounded global steps with the reload renewed "
              f"while the base relaxes to {drv_b_end:.3f} below "
              "c = 0.5: driven persistence, with base relaxation "
              "irrelevant to the guard")

        # the INTERMITTENT planning cell: the classification's
        # fourth regime as an executable instance (the witness
        # orbit of the intermittent persistence theorem): under
        # (O-int) at the witness constants, the alternating
        # schedule (c, eta) = (2, 0.1)/(0.5, 0.001) holds the base
        # at its exact fixed point 0.75, closes the guard at every
        # even step by the clip ceiling, forces it at every odd
        # step from the base alone, exits hot at the exhaustion
        # ceiling 0.1, and writes a reload of about 101 against a
        # one-step gap: infinite events, run length one, NO
        # completed reload; neither the separated nor the
        # unbounded-run reading applies
        int_rec = theory.run_intermittent_witness(n_steps=40,
                                                  fuel=fuel_pl)
        int_stats = int_rec["stats"]
        int_ok = ("".join(map(str, int_rec["word"])) == "01" * 20
                  and int_stats["max_run"] == 1
                  and int_stats["unsep_frac"] == 1.0
                  and min(int_rec["act_series"]) >= 1.0
                  and max(int_rec["J_out"]) <= 0.1
                  and abs(float(int_rec["state"]["b"][0]) - 0.75)
                  < 1e-12)
        check("[orbit] intermittent planning cell (the "
              "classification's fourth regime, executable)",
              int_ok and int_stats["phen"] == "unsep-window",
              f"topple word (01)^20 at run length "
              f"{int_stats['max_run']}, unseparated pair fraction "
              f"{int_stats['unsep_frac']:.2f} with counter floor "
              f"{min(int_rec['act_series']):.0f}, hot-exit jet at "
              "or under 0.1 and rebuild band top "
              f"{max(int_rec['J_series']):.3f} at or under 0.3: "
              "bounded bursts with incomplete reloads, the "
              "intermittent class inhabited")

        # the SEPARATED planning cell: the completed-reload
        # witness (the classification's second regime as an
        # executable instance, the twin of the Lean witness
        # RecurrentSeparated.sepEntry): period-four schedule,
        # base-forced hot-exit events, written reload of ONE
        # against a four-step gap, the counter at zero through the
        # middle of every gap
        sw_rec = theory.run_separated_witness(n_steps=48,
                                              fuel=fuel_pl)
        sw_stats = sw_rec["stats"]
        sw_ok = ("".join(map(str, sw_rec["word"]))
                 == "0001" * 12
                 and sw_stats["max_run"] == 1
                 and sw_stats["n_unsep"] == 0
                 and sw_stats["gate_closure_frac"] == 1.0
                 and max(sw_rec["J_out"]) <= 0.1
                 and abs(float(sw_rec["state"]["b"][0]) - 0.75)
                 < 1e-12)
        check("[orbit] separated planning cell (the "
              "classification's second regime, executable)",
              sw_ok and sw_stats["phen"] == "sep-window",
              f"topple word (0001)^12 at run length "
              f"{sw_stats['max_run']}, "
              f"{sw_stats['n_unsep']} unseparated of "
              f"{sw_stats['n_pairs']} pairs with gate-closure "
              f"fraction {sw_stats['gate_closure_frac']:.2f}, "
              "hot-exit jet at or under 0.1, base at its exact "
              "fixed point: every written reload completes "
              "strictly inside its gap, the separated class "
              "inhabited")

        # the WINDOW-PHENOTYPE classifier on one representative
        # word of EACH phenotype (including the censored outcome):
        # the decision rules consume exactly this classifier, so
        # its assignments are gated at generation
        ph_quiet = theory.classify_event_word(
            [1, 0, 1, 0] + [0] * 36,
            [3, 2, 3, 2, 1, 0] + [0] * 34)["phen"]
        ph_run = theory.classify_event_word(
            [1] * 40, [1] * 40)["phen"]
        w_sep, a_sep = [], []
        for _ in range(8):
            w_sep += [1, 0, 0, 0, 0]
            a_sep += [2, 1, 0, 0, 0]
        ph_sep = theory.classify_event_word(w_sep, a_sep)["phen"]
        ph_int = int_stats["phen"]
        ph_cen = theory.classify_event_word(
            [1, 0, 0, 1], [2, 1, 0, 2])["phen"]
        check("[orbit] window-phenotype classifier (five "
              "phenotypes, executable; finite windows never "
              "assign an asymptotic class)",
              (ph_quiet, ph_run, ph_sep, ph_int, ph_cen)
              == ("quiet-tail", "long-run", "sep-window",
                  "unsep-window", "censored"),
              f"assigned ({ph_quiet}, {ph_run}, {ph_sep}, "
              f"{ph_int}, {ph_cen}) on the five representative "
              "words: cessation, unit density, completed-reload "
              "cycles, the intermittent witness, and a "
              "short-window censoring")

        # the (O-head) resolution certificate on the headroom
        # variant: the saturated-jet drive 0.0625 fits under
        # c = 0.5 with gap 0.4, so the run must end within the
        # printed count and stay quiet through the margin
        res_n = theory.overload_resolution_bound(
            b0=1.0, d0_sup=0.0, mu_b=0.1, eta=0.1, kappa=1.0,
            a_sup=0.5, g_sup=0.5, c_minus=0.5, delta_b=0.4)
        res_state = dict(b=np.array([1.0]), a=np.array([0.5]),
                         g=np.array([0.5]), J=np.array([1.0]),
                         w=np.array([0.0]), pend=np.array([0.0]),
                         act=np.array([1.0]))
        res_last = None
        for _t in range(1, res_n + 51):
            res_state, res_top = theory.gated_model_step(
                res_state, drv_params, np.array([0.5]), 0.1,
                fuel_pl)
            if bool(res_top[0]):
                res_last = _t
        check("[thm] overload resolution under headroom (O-head)",
              res_last is not None and res_last <= res_n,
              f"last topple at global step {res_last} <= printed "
              f"count N_res = {res_n}; quiet through the checked "
              "margin (the guard is closed at the saturated-jet "
              "ceiling under the relaxed base)")

        # the maintenance certificates: the SAME functions the
        # theorem constants use (endpoint floor M-end, all-prefix
        # envelope M-prefix, cycle average M-avg-c), invoked as
        # gates at TWO tiers.  [thm]: the box worst case, with the
        # per-event retention floor from the two-phase retention
        # lemma (uniform over gap-domain entries, independent of
        # the event length); GATED on M-end and M-avg-c, the
        # certificates the recurrent regime's stroboscopic and
        # time-average clauses consume (the measured phase
        # criterion is a time average); the pointwise (M-prefix)
        # licensure is recorded at whichever tier it holds.
        # [orbit]: the measured cycle constants of the planning
        # orbit (realized per-event retention, gate-indicator
        # reload count, realized cycle length); GATED as before.
        # E5 reports all three quantities at both tiers.
        t_cyc = int(dic["rec_min_gap"][0])
        topple_idx = [i for i, x in enumerate(dic["site0_topple"])
                      if x]
        i0 = topple_idx[len(topple_idx) // 2]
        n_act = theory.active_reload_count(
            dic["site0_act"][i0:i0 + t_cyc])
        m_seq = ([gp["eta"] * gp["delta_J0"]] * n_act
                 + [0.0] * (t_cyc - n_act))
        q_step = 1.0 - gp["eta"] * (gp["th_J_loss"]
                                    + gp["lam_wd_model"]
                                    + gp["kappa"]
                                    * pl["w_max2_event"])
        r_thm = theory.retention_floor_two_phase(
            pl["eps_max_event"], delta_b, gp["eta"],
            gp["th_J_loss"], gp["lam_wd_model"], gp["kappa"],
            bu["h"], zcap, bu["z_floor"])
        r_orb = min(dic["site0_r"])
        maint = {}
        for tag, r_min in (("thm", r_thm), ("orbit", r_orb)):
            q_cyc = r_min * q_step ** t_cyc
            m_cyc = sum(m_seq) * q_step ** t_cyc
            # (M-end) consumes the CLIPPED floor comparison: valid
            # for the saturated transition whenever the floor
            # respects the jet ceiling (J_floor <= 1, gated below)
            floor = (theory.maintenance_floor_clipped(q_cyc, m_cyc)
                     if q_cyc < 1.0 and m_cyc > 0.0 else 0.0)
            # (N-sat): the all-prefix nonsaturation certificate
            # under which the affine envelope minorizes the
            # clipped orbit; the prefix and average certificates
            # refuse without it
            nsat = theory.nsat_envelope_max(floor, r_min, q_step,
                                            m_seq)
            if nsat <= 1.0:
                pmin = theory.maintenance_prefix_min(
                    floor, r_min, q_step, m_seq)
                cavg = theory.maintenance_cycle_average(
                    floor, r_min, q_step, m_seq)
            else:
                pmin = 0.0
                cavg = 0.0
            lic = ("pointwise" if pmin > gp["J_crit"]
                   else "time-average" if cavg > gp["J_crit"]
                   else "stroboscopic-only" if floor > gp["J_crit"]
                   else "none")
            # the running-window corollary: the smallest window
            # length L (in cycles of T+1 steps) whose mean stays
            # above J_crit given the aligned-cycle average and the
            # envelope minimum (the two-boundary deficit)
            if cavg > gp["J_crit"]:
                win_min = math.ceil(2.0 * (t_cyc + 1)
                                    * (cavg - pmin)
                                    / (cavg - gp["J_crit"]))
                win_min = max(win_min, t_cyc + 1)
            else:
                win_min = None
            maint[tag] = dict(r_min=r_min, q_step=q_step,
                              t_cyc=t_cyc, n_act=n_act,
                              floor=floor, prefix_min=pmin,
                              cycle_avg=cavg, nsat=nsat,
                              licensure=lic,
                              window_min=win_min)
        m_o = maint["orbit"]
        m_t = maint["thm"]
        check("[thm] clipped floor ceiling and all-prefix "
              "nonsaturation certificate (J_floor <= 1 and "
              "(N-sat) <= 1, both tiers)",
              m_t["floor"] <= 1.0 and m_t["nsat"] <= 1.0
              and m_o["floor"] <= 1.0 and m_o["nsat"] <= 1.0,
              f"theorem tier: J_floor = {m_t['floor']:.3f} <= 1 "
              "(the clipped floor comparison licenses (M-end) for "
              "the saturated transition with no envelope "
              f"hypothesis), (N-sat) envelope max {m_t['nsat']:.3f}"
              " <= 1 (the affine prefix envelope minorizes the "
              "clipped orbit; hypothesis of (M-prefix)/(M-avg-c)); "
              f"orbit tier: J_floor = {m_o['floor']:.3f}, (N-sat) "
              f"{m_o['nsat']:.3f}.  These are the saturation "
              "certificates of the recurrent clause; an early "
              "oversized injection fails (N-sat) and refuses "
              "certification (the clipped-envelope witness is "
              "pinned by a test)")
        check("[thm] maintenance certificates at the box worst "
              "case (M-end endpoint floor and M-avg-c computable "
              "cycle average vs J_crit; two-phase retention "
              "floor)",
              m_t["floor"] > gp["J_crit"]
              and m_t["cycle_avg"] > gp["J_crit"],
              f"r_min = {m_t['r_min']:.3f} (two-phase retention "
              "floor, uniform over gap-domain entries), floor = "
              f"{m_t['floor']:.3f} (M-end), cycle avg = "
              f"{m_t['cycle_avg']:.3f} (M-avg-c) vs J_crit "
              f"{gp['J_crit']}; prefix min = "
              f"{m_t['prefix_min']:.3f} (M-prefix), licensure: "
              f"{m_t['licensure']}; minimal running window "
              f"{m_t['window_min']} steps.  The theorem-tier "
              "instance passes the stroboscopic and time-average "
              "certificates (the measured phase criterion reads "
              "the time average); the pointwise clause holds at "
              "the orbit tier only and is stated at that "
              "strength")
        check("[orbit] maintenance certificates at the measured "
              "cycle constants (endpoint floor, all-prefix "
              "minimum, cycle average vs J_crit)",
              m_o["floor"] > gp["J_crit"]
              and m_o["cycle_avg"] > gp["J_crit"],
              f"r_min = {m_o['r_min']:.3f} (realized), n_act = "
              f"{m_o['n_act']} (gate indicator, never gap length), "
              f"floor = {m_o['floor']:.3f}, prefix min = "
              f"{m_o['prefix_min']:.3f}, cycle avg = "
              f"{m_o['cycle_avg']:.3f} vs J_crit {gp['J_crit']}; "
              f"licensure: {m_o['licensure']}; minimal running "
              f"window {m_o['window_min']} steps (the measured "
              "criterion is a time average; pointwise additionally "
              "needs the all-prefix inequality); E5 reports all "
              "three quantities at both tiers and gates on the "
              "E0-measured tier")
    except ValueError as e:
        check("[orbit] two-regime planning instance (step "
              "conditions (S+)/(Q+))", False, e)
        dic = None

    # the capture envelope balance at the planning instance: the
    # captured arm's no-topple clause is theorem-backed (the full
    # product a*g*J is bounded through the amplitude and base
    # envelopes, never by u(T1) alone)
    a_sup = theory.amplitude_envelope(
        1.0, gp["delta_a"], gp["th_a_loss"], gp["lam_wd_model"])
    b_sup = theory.base_envelope(b_star * 0.5, gp["d0"], gp["mu_b"])
    slack = theory.capture_balance(b_sup, gp["kappa"], a_sup, 1.0,
                                   gp["J0"], gp["c_captured"])
    check("[thm] capture envelope balance (b_sup + "
          "kappa*(a_sup*g*J)^2 <= c)", slack >= 0.0,
          f"a_sup = {a_sup:.3f}, b_sup = {b_sup:.3f}, slack = "
          f"{slack:.3f}")

    # the drive ceiling D_max: defined in closed form and computed
    # here, so the captured-regime anneal condition is computable
    # in fact (the capture inequality consumes this value)
    dmax = None
    try:
        dmax = theory.drive_ceiling(gp["d0"], gp["kappa"], a_sup,
                                    1.0, 1.0, gp["delta_a"])
        v0 = theory.source_velocity_loss(
            1.0, 1.0, gp["J0"], 0.0, gp["delta_a"],
            gp["th_a_loss"], gp["th_g_loss"], gp["th_J_loss"],
            gp["kappa"])
        rhs0 = theory.capture_rhs(gp["eta"], b_star * 0.5,
                                  gp["kappa"], gp["J0"], v0,
                                  gp["d0"], gp["mu_b"], 3.0,
                                  gp["lam_wd_model"])
        check("[thm] drive ceiling D_max dominates the loading "
              "rate", rhs0 <= gp["eta"] * dmax + 1e-12,
              f"D_max = {dmax:.4f} (closed-form box supremum: "
              "d0_sup + 2*kappa*u_sup*(v_loss)_+ + defect, every "
              "sign-indefinite term bounded in the capture "
              f"direction); planning loading rate {rhs0:.4f} <= "
              f"eta*D_max = {gp['eta'] * dmax:.4f}; the anneal "
              "condition c(t+1) - c(t) >= eta_t*(D_max + C_D*"
              "eta_t) is computable from this value")
    except ValueError as e:
        check("[thm] drive ceiling D_max dominates the loading "
              "rate", False, e)

    # unit conventions
    rho_w = theory.rho_w_derived(bu["z_floor"], bu["z_seed"])
    check("[unit] carrier unit identity (amplitude factor = "
          "sqrt(rho_w), relative to the SEEDED entry amplitude)",
          abs(theory.amp_factor(rho_w) - math.sqrt(rho_w)) < 1e-15,
          f"rho_w = {rho_w:.3f} (power), amplitude factor "
          f"{math.sqrt(rho_w):.4f}; the reference value is "
          "|w_entry| of theory.seeded_entry (z_0 = w_entry^2 "
          "exactly), never the stored pre-event amplitude when the "
          "seed floor binds")
    beta1 = cal["optimizer"]["beta1"]
    flip = theory.flip_boundary(beta1, 0.0)
    check("[unit] sharpness unit (flip (2-d)(1+beta1); eta*lambda "
          "= flip/(1-beta1))",
          abs(flip / (1 - beta1)
              - cal["boundaries"]["frozen_flip_eta_lambda"]) < 1e-9,
          f"flip sigma {flip}, eta*lambda {flip / (1 - beta1):.1f}")
    check("[unit] rate unit declared",
          "log multiplier" in cal["conventions"]["rate_unit"],
          cal["conventions"]["rate_unit"])
    check("[unit] event-end convention declared (generalized exit; "
          "two-condition on the entry-gap domain)",
          "generalized" in cal["conventions"]["event_end"]
          and "two-condition" in cal["conventions"]["event_end"],
          cal["conventions"]["event_end"])
    check("[unit] activity unit declared (reload duration, global "
          "steps)",
          "RELOAD DURATION" in cal["conventions"]["activity_unit"],
          cal["conventions"]["activity_unit"])

    # certificate box and certified band
    box = pl["cp_cert_box"]
    bounds = None
    band = None
    try:
        bounds = theory.cp_dir_bounds(
            box["sigma"][0], box["sigma"][1], box["beta1"][0],
            box["beta1"][1], box["d"][0], box["d"][1])
        check("[box] certificate box Jury margins", True,
              f"gamma1 = {bounds['gamma1']:.3g}, gamma2 = "
              f"{bounds['gamma2']:.3g}, K_s = {bounds['K_s']:.4g}, "
              f"K_q = {bounds['K_q']:.4g}")
    except ValueError as e:
        check("[box] certificate box Jury margins", False, e)
    if bounds is not None:
        lam_wd = cal["optimizer"]["lam_wd"]
        cellnames = ("boundary_rate_cell", "reference_swept_cell",
                     "subcritical_control")
        ds = [cal["cells"][k]["lr"] * lam_wd for k in cellnames]
        check("[box] schedule-implied decay parameters inside the "
              "box d-range",
              all(box["d"][0] <= d <= box["d"][1] for d in ds),
              f"d values {ds} in {box['d']}")
        grid_stat = theory.cp_grid_check(
            box["sigma"][0], box["sigma"][1], box["beta1"][0],
            box["beta1"][1], box["d"][0], box["d"][1],
            n=box["grid_check_n"])
        check("[box] finite-difference sanity statistic below the "
              "certified constant",
              grid_stat <= bounds["C_P"],
              f"grid {grid_stat:.4g} <= C_P {bounds['C_P']:.4g}")
        # the certified BAND at the planning drift budget: the
        # certified set is an interior band in general (the
        # Lyapunov gain is U-shaped), so both edges are reported
        # and downward closure is checked, never assumed
        try:
            budget_pl = 1.5 * cal["cells"]["boundary_rate_cell"][
                "lr"]
            band = theory.cert_band(
                cal["optimizer"]["beta1"], ds[0],
                eps_s=budget_pl / bounds["K_s"], eps_q=0.0,
                bounds=bounds)
            check("[box] certified band nonempty at the planning "
                  "drift budget (edges reported; downward closure "
                  "checked, never assumed)",
                  not band["empty"],
                  f"band [{band['sigma_cert_lo']:.4g}, "
                  f"{band['sigma_cert_hi']:.4g}] at budget "
                  f"{budget_pl:.3g}; at_lo_edge = "
                  f"{band['at_lo_edge']}, at_hi_edge = "
                  f"{band['at_hi_edge']}, downward_closed = "
                  f"{band['downward_closed']}; all stability "
                  "language reads INSIDE the band (a scalar upper "
                  "edge is a component edge, not a threshold)")
        except ValueError as e:
            check("[box] certified band nonempty at the planning "
                  "drift budget", False, e)

    # optimizer constants against the stack
    import re as _re
    src = open(os.path.join(EXP, "train_run.py")).read()
    eps_stack = float(_re.search(r"eps=([0-9.e-]+)", src).group(1))
    check("[stack] calibration optimizer eps equals train_run.py",
          eps_stack == cal["optimizer"]["eps"],
          f"stack {eps_stack}, calibration {cal['optimizer']['eps']}")
    fid = open(os.path.join(EXP, "measure_fidelity.py")).read()
    eps_fid = float(_re.search(r"eps=([0-9.e-]+)", fid).group(1))
    check("[stack] measure_fidelity.py eps equals the calibration",
          eps_fid == cal["optimizer"]["eps"],
          f"fidelity {eps_fid}")
    m_betas = _re.search(r"betas=\(([0-9.]+),\s*([0-9.]+)\)", fid)
    check("[stack] betas equal the calibration",
          (float(m_betas.group(1)), float(m_betas.group(2)))
          == (cal["optimizer"]["beta1"], cal["optimizer"]["beta2"]),
          f"stack betas ({m_betas.group(1)}, {m_betas.group(2)})")

    # TIERED result blocks: theorem, orbit, and consistency
    # (unit/box/stack) checks are counted per class, and the
    # deferred future-measurement items are listed as a block of
    # their own; no single collapsed pass count is reported
    def _cls(name):
        return name.split("]")[0] + "]" if name.startswith("[") \
            else "[other]"
    counts = {}
    for c in checks:
        k = _cls(c["name"])
        counts.setdefault(k, {"passed": 0, "failed": 0})
        counts[k]["passed" if c["ok"] else "failed"] += 1
    deferred_block = sorted(cal["measured_in_E0"]["fields"])
    report = {
        "order": "gate 0 (theory, units, box, constants) runs and "
                 "must pass BEFORE any power computation",
        "b_gap_display": theory.B_GAP_LATEX,
        "classes": {
            "[thm]": "theorem hypothesis or theorem constant at "
                     "the planning constants (available "
                     "planning-instance gates)",
            "[orbit]": "property of one selected planning "
                       "trajectory (realized path, not a "
                       "box-quantified certificate)",
            "[unit]": "declared unit or convention consistency",
            "[box]": "certificate box or certified band",
            "[stack]": "calibration vs training stack",
            "deferred": "E0-measured registry hypotheses are NOT "
                        "checked here; protocols carry them as "
                        "typed deferred values (listed in the "
                        "deferred block below)",
        },
        "counts": counts,
        "checks": checks,
        "deferred": deferred_block,
        "tsep": tsep,
    }
    count_line = "; ".join(
        f"{k} {v['passed']} passed, {v['failed']} failed"
        for k, v in sorted(counts.items())) \
        + f"; deferred {len(deferred_block)} E0-measured items"
    if write:
        os.makedirs(OUTDIR, exist_ok=True)
        with open(os.path.join(OUTDIR, "gate_report.json"),
                  "w") as fh:
            json.dump(report, fh, indent=1, default=float)
        print("  wrote gate_report.json (" + count_line + ")")
    else:
        print("  gate report NOT written (--check/--dry-run): "
              + count_line)
    if fails:
        raise SystemExit("gen_protocols: GATE 0 failed BEFORE any "
                         "power computation: " + "; ".join(fails))
    return dict(zcap=zcap, u_max=u_max, n_max=n_max, r_lic=r_lic,
                rho_w=rho_w, bounds=bounds, tsep=tsep, dmax=dmax,
                delta_b=delta_b, fuel_req=fuel_req,
                fuel_uniform=fuel_uniform, band=band,
                maint=maint)


# ------------------------------------------------- power simulations


def _power_e0(pl, rng):
    """Joint MC of the two stochastic E0 decisions: the loading-slope
    factor-two check and the crossing-signature fraction."""
    trials = 4000
    se_rel = 1.0 / (pl["loading_slope_snr_per_window"]
                    * math.sqrt(pl["n_probe_windows"]))
    slope = rng.normal(1.0, se_rel, size=trials)
    sig = rng.binomial(pl["n_events_e0"], pl["signature_frac_model"],
                       size=trials)
    ok = (slope > 0.5) & (slope < 2.0) & (
        sig >= pl["n_events_e0"] / 2.0)
    return float(ok.mean()), se_rel


def _power_e1(pl, rng):
    """Joint E1 operating characteristic.

    The primary pulse comparison uses alpha 0.025.  Three independent
    face-intervention blocks estimate the two eta-squared and one
    eta-fourth reactivation slopes; their 95 percent intervals must lie
    inside the registered equivalence bands around (2, 2, 4).
    """
    trials = 4000
    n = 10
    d = rng.normal(pl["paired_effect_over_sd"], 1.0, size=(trials, n))
    tstat = d.mean(axis=1) / (d.std(axis=1, ddof=1) / math.sqrt(n))
    stage1 = rng.binomial(n, pl["event_induction_prob"], size=trials)
    pulse_ok = (stage1 >= 7) & (tstat > 2.262)  # t_{0.975, 9}
    ratios = np.asarray(pl["e1_eta_ratios"], dtype=float)
    n_face = int(pl["e1_face_seeds"])
    x = np.repeat(np.log(ratios), n_face)
    xc = x - x.mean()
    denom = float(np.sum(xc ** 2))
    targets = np.asarray([2.0, 2.0, 4.0])
    noise = rng.normal(
        0.0, pl["e1_reactivation_log_sd"],
        size=(trials, len(targets), len(x)))
    y = targets[None, :, None] * x[None, None, :] + noise
    slopes = np.sum(y * xc[None, None, :], axis=2) / denom
    fitted = slopes[:, :, None] * x[None, None, :]
    resid = y - fitted - np.mean(y - fitted, axis=2, keepdims=True)
    se = np.sqrt(np.sum(resid ** 2, axis=2)
                 / (len(x) - 2) / denom)
    tol = float(pl["e1_reactivation_slope_tolerance"])
    face_ok = np.all(
        (slopes - 1.96 * se >= targets[None, :] - tol)
        & (slopes + 1.96 * se <= targets[None, :] + tol), axis=1)
    return float(np.mean(pulse_ok & face_ok))


def _power_e2(pl, cal, rng):
    """PRIMARY: the event-level reset law at the GLOBAL clock.  The
    observable per detected event is the measured log drain (the
    jet curvature immediately before and after the event step, read
    at high global-step cadence); the PREDICTION is the surrogate
    drain evaluated at the ESTIMATED entry state, with the latent
    fast path integrated out (the training loop performs one
    optimizer update per global step and never executes an inner
    event clock, so no endpoint consumes inner-clock states).
    Build the planning event library through theory.run_event over
    the registered entry-margin range; per simulated event the
    measured drain is the true drain plus measurement noise, and
    the predicted drain is the library functional evaluated at the
    entry margin estimated with the planning relative error (the
    propagated state-estimation uncertainty).  Regress measured on
    predicted with the errors-in-variables attenuation correction
    for the propagated prediction variance and score: slope
    significantly positive AND inside the noise-widened band
    [0.8, 1.2].  Returns (power, u_max, library drain spread)."""
    trials = 4000
    n = pl["n_events_e2"]
    gp = cal["gated_model_planning"]
    bu = gp["burst"]
    b_star = gp["d0"] / gp["mu_b"]
    c = gp["c_recurrent"]
    zcap = theory.z_cap_invariant(
        bu["h"], pl["eps_max_event"], bu["b_sat"],
        w_max2=pl["w_max2_event"], z_seed=bu["z_seed"])
    u_max = theory.event_u_max(
        gp["eta"], gp["th_J_loss"], gp["lam_wd_model"], gp["kappa"],
        zcap)
    fuel_pl = granted_fuel(gp, pl)
    margins = np.linspace(pl["e2_entry_margin_lo"],
                          pl["e2_entry_margin_hi"],
                          pl["e2_library_size"])
    drains = []
    for eps0 in margins:
        y0 = c + eps0 - b_star
        ev = theory.run_event(
            b_star, 1.0, 1.0, math.sqrt(y0 / gp["kappa"]), 0.0, c,
            gp["eta"], gp["kappa"], gp["th_J_loss"],
            gp["lam_wd_model"], bu, fuel=fuel_pl)
        drains.append(-2.0 * ev["sum_log_phi"])
    drains = np.array(drains)
    idx = rng.integers(0, len(margins), size=(trials, n))
    true_m = margins[idx]
    true_L = drains[idx]
    # measured log drain: observable at the global clock, with the
    # planning measurement noise
    L = true_L + pl["e2_logdrain_meas_sd"] \
        * rng.standard_normal((trials, n))
    # predicted log drain: the surrogate functional at the
    # ESTIMATED entry state (relative state-estimation error on the
    # entry margin, propagated through the library)
    est_m = true_m * (1.0 + pl["e2_power_rel_sd"]
                      * rng.standard_normal((trials, n)))
    P = np.interp(est_m, margins, drains)
    Pc = P - P.mean(axis=1, keepdims=True)
    Lc = L - L.mean(axis=1, keepdims=True)
    var_obs = (Pc * Pc).sum(axis=1) / (n - 1)
    # attenuation correction for the propagated state-estimation
    # variance in the regressor (errors-in-variables; the
    # correction factor is re-measured at E0): var(noise) ~
    # (rel_sd * mean margin * local slope)^2
    dslope = np.gradient(drains, margins).mean()
    var_noise = (pl["e2_power_rel_sd"] * true_m.mean(axis=1)
                 * dslope) ** 2
    atten = np.clip(1.0 - var_noise / np.maximum(var_obs, 1e-12),
                    0.2, 1.0)
    slope = (Pc * Lc).sum(axis=1) / (Pc * Pc).sum(axis=1) / atten
    resid = Lc - slope[:, None] * Pc
    se = np.sqrt((resid ** 2).sum(axis=1) / (n - 2)
                 / (Pc * Pc).sum(axis=1))
    t = slope / (se + 1e-15)
    ok = (t > 1.671) & (slope >= 0.8) & (slope <= 1.2)
    spread = float(drains.max() - drains.min())
    return float(ok.mean()), u_max, spread


def _power_e3(pl, rng):
    """R_c confidence bound below one, jointly with the (K-dom)
    stratified dominance check passing in all three strata."""
    trials = 4000
    n = pl["n_arrivals_e3"]
    rc = pl["rc_planning"]
    mean = rng.normal(rc, math.sqrt(rc / n), size=trials)
    ub = mean + 1.645 * math.sqrt(rc / n)
    # dominance specificity: three strata each tested at alpha/3
    strata_ok = rng.uniform(size=(trials, 3)) > 0.05 / 3.0
    ok = (ub < 1.0) & strata_ok.all(axis=1)
    return float(ok.mean())


def _power_e4(pl, rng):
    """Paired log-flux comparison over events and matched nulls, with
    the factor-two magnitude check."""
    trials = 4000
    n = pl["n_events_e4"]
    mu = math.log(pl["flux_ratio_model"])
    d = rng.normal(mu, pl["flux_lognorm_sd"], size=(trials, n))
    m = d.mean(axis=1)
    se = d.std(axis=1, ddof=1) / math.sqrt(n)
    ok = (m / se > 1.685) & (np.abs(m - mu) <= math.log(2.0))
    return float(ok.mean())


def _raw_decision_record(word, act, jet):
    """Synthetic raw-probe record for operating-characteristic MC."""
    jet = np.maximum(np.asarray(jet, dtype=float).ravel(), 0.0)
    total = len(jet)
    # Four visited-row samples in each of two layers retain the raw
    # endpoint axes while making the planning jet their exact median.
    samples = np.repeat(jet[:, None, None], 2, axis=1)
    samples = np.repeat(samples, 4, axis=2)
    return dict(word=np.asarray(word, dtype=int).ravel(),
                act=np.asarray(act, dtype=float).ravel(),
                probe_steps=np.arange(1, total + 1),
                rowmap_samples=samples, normalization=1.0)


def _censored_raw_record(rec, J_crit):
    """Return a raw record whose final dip is too short to license entry.

    This is used only by the registered operating-characteristic
    censoring model.  It preserves the event and reload traces and changes
    the raw probes, never a derived endpoint label.
    """
    out = {
        key: (value.copy() if isinstance(value, np.ndarray) else value)
        for key, value in rec.items()
    }
    samples = np.asarray(out["rowmap_samples"], dtype=float).copy()
    norm = np.asarray(out["normalization"], dtype=float)
    if norm.ndim == 0:
        norm = np.full(samples.shape[1], float(norm))
    samples[:] = (2.0 * float(J_crit) * norm[None, :, None])
    samples[-1] = 0.5 * float(J_crit) * norm[:, None]
    out["rowmap_samples"] = samples
    return out


def _simulate_e5_designs(gp, pl, rng, fuel, n_rep, attempt_cfg):
    """Generate complete raw E5 designs for one immutable attempt."""
    nseeds = int(attempt_cfg["seeds_per_arm"])
    nsites = nseeds * int(n_rep)
    if nsites == 0:
        return []
    dic = _gated_dichotomy(
        gp, fuel, n=nsites, rng=rng,
        d0_jitter_rel=pl["e5_d0_jitter_rel"],
        T_rec=int(gp["T_maint"]))
    sep = _gated_separated(
        gp, fuel, n=nsites, rng=rng,
        d0_jitter_rel=pl["e5_d0_jitter_rel"])
    raw_noise = pl["e5_obs_noise_J"] / math.sqrt(8.0)
    capJ = dic["cap_J"] + raw_noise * rng.standard_normal(
        dic["cap_J"].shape).astype(np.float32)
    recJ = dic["rec_J"] + raw_noise * rng.standard_normal(
        dic["rec_J"].shape).astype(np.float32)
    sepJ = sep["sep_J"] + raw_noise * rng.standard_normal(
        sep["sep_J"].shape).astype(np.float32)
    designs = []
    for r in range(int(n_rep)):
        arms = {"captured": [], "separated": [], "gateheld": []}
        for j in range(nseeds):
            i = nseeds * r + j
            arms["captured"].append(_raw_decision_record(
                dic["cap_word"][:, i], dic["cap_act"][:, i],
                capJ[:, i]))
            arms["separated"].append(_raw_decision_record(
                sep["sep_word"][:, i], sep["sep_act"][:, i],
                sepJ[:, i]))
            arms["gateheld"].append(_raw_decision_record(
                dic["rec_word"][:, i], dic["rec_act"][:, i],
                recJ[:, i]))
        designs.append(arms)
    return designs


def _power_e5(pl, cal, rng, e5_cfg):
    """E5: the operating characteristic of the WRITTEN decision
    rule, computed as a Monte Carlo of ``theory.decide_e5_study``
    (the study orchestrator wrapping ``theory.decide_e5``) under
    the MANDATORY generated configuration object, never of a
    simplified twin.  Each replication runs the planning instance
    through all THREE arms (matched capture, two-phase separated
    maintenance, and constant gate-held maintenance schedules; one
    update law) at the declared seeds per arm, with the planning
    d0 jitter, stochastic event seeds, and observation noise on
    the jet readout, and hands the decision function the FULL
    per-seed traces the schema declares (event word, reload
    counter and raw row-map probes).  Every endpoint and derived observable of the
    printed rule is computed INSIDE the rule from those traces by
    the declared measurement functions and scored on every
    replication: the window-phenotype routing, every captured-arm
    clause and both kills, the jet-minimum and measured
    time-average clauses, the cycle-average certificate at the
    measured cycle constants, the gate regression on reload
    windows with its null-everywhere kill, the censoring
    indeterminate, and the orchestrated repeat policy.  The registered
    censoring model opens a genuine second attempt with its own seed count
    and newly generated raw records.  Returns
    (power, reps, halfwidth, verdicts, branch_counts)."""
    gp = cal["gated_model_planning"]
    reps = cal["power_sim"]["e5_reps"]
    fuel = granted_fuel(gp, pl)
    first_cfg, second_cfg = e5_cfg["attempts"]
    primary = _simulate_e5_designs(
        gp, pl, rng, fuel, reps, first_cfg)
    censor_rate = float(cal["power_sim"].get(
        "e5_primary_censor_rate", 0.0))
    repeat_rate = float(cal["power_sim"].get(
        "repeat_censor_rate", 0.0))
    open_repeat = rng.random(reps) < censor_rate
    repeat_indices = np.flatnonzero(open_repeat)
    repeats = _simulate_e5_designs(
        gp, pl, rng, fuel, len(repeat_indices), second_cfg)
    repeat_censored = rng.random(len(repeat_indices)) < repeat_rate
    verdicts = []
    branches = {
        "first_PASS": 0, "first_FAIL": 0, "first_KILL": 0,
        "first_INDETERMINATE": 0, "repeat_triggered": 0,
        "second_PASS": 0, "second_FAIL": 0, "second_KILL": 0,
        "second_INDETERMINATE": 0,
        "final_PASS": 0, "final_FAIL": 0, "final_KILL": 0,
        "final_INDETERMINATE": 0,
    }
    repeat_pos = 0
    for r in range(reps):
        arms = primary[r]
        attempt_designs = [arms]
        if open_repeat[r]:
            arms["captured"][0] = _censored_raw_record(
                arms["captured"][0], first_cfg["J_crit"])
            second = repeats[repeat_pos]
            if repeat_censored[repeat_pos]:
                second["captured"][0] = _censored_raw_record(
                    second["captured"][0], second_cfg["J_crit"])
            attempt_designs.append(second)
            repeat_pos += 1
        out = theory.decide_e5_study(attempt_designs, e5_cfg)
        first_verdict = out["attempts"][0]["verdict"]
        branches[f"first_{first_verdict}"] += 1
        if len(out["attempts"]) == 2:
            branches["repeat_triggered"] += 1
            second_verdict = out["attempts"][1]["verdict"]
            branches[f"second_{second_verdict}"] += 1
        verdicts.append(out["verdict"])
        branches[f"final_{out['verdict']}"] += 1
    power = float(np.mean([v == "PASS" for v in verdicts]))
    halfwidth = 1.96 * math.sqrt(
        max(power * (1.0 - power), 1e-12) / reps)
    return power, reps, halfwidth, verdicts, branches


def _simulate_e6_designs(gp, pl, rng, fuel, n_rep, attempt_cfg, alt):
    """Generate complete raw E6 cell dictionaries for one attempt."""
    n_rep = int(n_rep)
    if n_rep == 0:
        return []
    n_cells = int(attempt_cfg["n_cells"])
    seeds = int(attempt_cfg["seeds_per_cell"])
    shared = (rng.uniform(size=(n_rep, seeds))
              < alt["p_match"]).astype(float)
    sign = np.where(rng.uniform(size=(n_rep, n_cells)) < 0.5,
                    1.0, -1.0)
    pc = np.clip(alt["p_match"] + sign * alt["delta"], 0.0, 1.0)
    pi = np.clip((1.0 - alt["w"]) * pc[:, :, None]
                 + alt["w"] * shared[:, None, :], 0.0, 1.0)
    conform = rng.uniform(size=(n_rep, n_cells, seeds)) < pi
    predicted_capture = np.zeros((n_cells,), dtype=bool)
    predicted_capture[: n_cells // 2] = True
    need_cap = conform == predicted_capture[None, :, None]
    n_cap = int(need_cap.sum())
    n_rec = int(n_rep * n_cells * seeds - n_cap)
    pool = _gated_dichotomy(
        gp, fuel, n=max(n_cap, n_rec, 1), rng=rng,
        d0_jitter_rel=pl["e5_d0_jitter_rel"],
        T_rec=int(gp["T_maint"]))
    raw_noise = pl["e5_obs_noise_J"] / math.sqrt(8.0)
    capJ = pool["cap_J"] + raw_noise * rng.standard_normal(
        pool["cap_J"].shape).astype(np.float32)
    recJ = pool["rec_J"] + raw_noise * rng.standard_normal(
        pool["rec_J"].shape).astype(np.float32)
    designs = []
    icap = irec = 0
    for r in range(n_rep):
        cells = {}
        for c in range(n_cells):
            pred = "capture" if predicted_capture[c] else "release"
            certificate = dict(
                threshold_increment_interval=(
                    (2.0, 2.0) if pred == "capture" else (1.0, 1.0)),
                source_increment_interval=(
                    (1.0, 1.0) if pred == "capture" else (2.0, 2.0)),
            )
            recs = []
            for j in range(seeds):
                if need_cap[r, c, j]:
                    i = icap
                    icap += 1
                    recs.append(_raw_decision_record(
                        pool["cap_word"][:, i],
                        pool["cap_act"][:, i], capJ[:, i]))
                else:
                    i = irec
                    irec += 1
                    recs.append(_raw_decision_record(
                        pool["rec_word"][:, i],
                        pool["rec_act"][:, i], recJ[:, i]))
            cells[f"cell{c + 1}"] = dict(
                predicted=pred, schedule_certificate=certificate, seeds=recs)
        designs.append(cells)
    return designs


def _power_e6(pl, cal, rng, e5_cfg):
    """E6, three layers, all of one executable object:

    (i) CALIBRATION.  The count threshold is chosen HERE, at
    generation, by EXACT ENUMERATION over the declared crossed
    cell-by-seed null family (``theory.e6_calibrate_threshold``):
    match mean one half, a two-point cell effect, and a shared
    seed effect covering the pairing the allocation creates, on
    the declared dense parameter grid with the two closed-form
    corner members (independent cells; fully shared seeds).  The
    chosen threshold is the smallest total whose worst-case exact
    false pass over the family is at or below the declared level;
    the worst member, the exact corner values (the fully
    shared-seed pass probabilities at the chosen threshold and one
    below it, the permanent regressions), and the grid's largest
    adjacent-member step are all archived.

    (ii) EXACT CHARACTERISTICS.  Planning power and the
    false-pass/false-fail table are exact enumerations of the
    count-and-coverage pair over the declared (match mean,
    cell-effect, seed-sharing) grid, not Monte Carlo.

    (iii) VERDICT-LEVEL MC.  The FULL rule
    (``theory.decide_e6_study`` wrapping ``theory.decide_e6``) is
    Monte Carlo scored on fully generated per-run records through
    the gated planning model (capture and release run types with
    d0 jitter, stochastic event seeds, and jet readout noise),
    with per-run conformity drawn from the crossed model at the
    planning member, so the phenotype routing, the early-collapse
    kill, the censoring, and the repeat policy execute on
    generated records.

    Both complete attempts receive their own exact calibration and
    exact operating-characteristic table.  Registered censoring opens
    a genuinely new second-attempt record set, so the verdict-level
    Monte Carlo executes both schemas and archives every branch.

    Returns a dict containing compatibility fields for the primary
    attempt plus ``attempt_characteristics`` for both attempts."""
    gp = cal["gated_model_planning"]
    spec = decision_constants.E6_SPEC
    design_search = theory.e6_search_attempt_designs(spec)
    calibrations = design_search["calibrations"]
    e6_cfg = decision_constants.e6_config(
        calibrations=calibrations, J_crit=gp["J_crit"],
        attempt_templates=design_search["selected_templates"])
    alt = spec["alt_planning"]
    grid = spec["alt_report_grid"]
    attempt_characteristics = []
    for attempt, calib in zip(e6_cfg["attempts"], calibrations):
        attempt_shape = {key: attempt[key] for key in
                         ("seeds_per_cell", "n_cells",
                          "cell_majority", "cells_required")}
        attempt_K = attempt["total_threshold"]
        attempt_power = theory.e6_pass_prob_exact(
            attempt_K, attempt_shape, alt["p_match"], alt["delta"],
            alt["w"])
        attempt_table = []
        for p in grid["p_list"]:
            for w in grid["w_list"]:
                for d in grid["delta_list"]:
                    pr = theory.e6_pass_prob_exact(
                        attempt_K, attempt_shape, p, d, w)
                    attempt_table.append(dict(
                        p=p, w=w, delta=d, pass_rate=pr,
                        false_fail=1.0 - pr))
        attempt_characteristics.append(dict(
            attempt_id=attempt["attempt_id"],
            seed_block=attempt["seed_block"],
            threshold=attempt_K,
            alpha_declared=attempt["alpha"],
            worst=calib["worst"], corners=calib["corners"],
            power_planning=float(attempt_power),
            power_dep_min=float(min(
                row["pass_rate"] for row in attempt_table
                if row["p"] == alt["p_match"])),
            table=attempt_table,
            seeds_per_cell=attempt_shape["seeds_per_cell"],
            cells_required=attempt_shape["cells_required"],
            cell_majority=attempt_shape["cell_majority"],
            n_cells=attempt_shape["n_cells"]))

    primary = e6_cfg["attempts"][0]
    primary_chars = attempt_characteristics[0]
    shape = {key: primary[key] for key in
             ("seeds_per_cell", "n_cells", "cell_majority",
              "cells_required")}
    K = primary["total_threshold"]

    # Verdict-level MC of the full study rule on generated records.
    reps = cal["power_sim"]["e6_verdict_reps"]
    fuel = granted_fuel(gp, pl)
    first_cfg, second_cfg = e6_cfg["attempts"]
    primary_designs = _simulate_e6_designs(
        gp, pl, rng, fuel, reps, first_cfg, alt)
    censor_rate = float(cal["power_sim"].get(
        "e6_primary_censor_rate", 0.0))
    repeat_rate = float(cal["power_sim"].get(
        "repeat_censor_rate", 0.0))
    open_repeat = rng.random(reps) < censor_rate
    repeat_indices = np.flatnonzero(open_repeat)
    repeat_designs = _simulate_e6_designs(
        gp, pl, rng, fuel, len(repeat_indices), second_cfg, alt)
    repeat_censored = rng.random(len(repeat_indices)) < repeat_rate
    verdicts = []
    branches = {
        "first_PASS": 0, "first_FAIL": 0, "first_KILL": 0,
        "first_INDETERMINATE": 0, "repeat_triggered": 0,
        "second_PASS": 0, "second_FAIL": 0, "second_KILL": 0,
        "second_INDETERMINATE": 0,
        "final_PASS": 0, "final_FAIL": 0, "final_KILL": 0,
        "final_INDETERMINATE": 0,
    }
    repeat_pos = 0
    for r in range(reps):
        cells = primary_designs[r]
        attempts = [cells]
        if open_repeat[r]:
            cells["cell1"]["seeds"][0] = _censored_raw_record(
                cells["cell1"]["seeds"][0], first_cfg["J_crit"])
            second = repeat_designs[repeat_pos]
            if repeat_censored[repeat_pos]:
                second["cell1"]["seeds"][0] = _censored_raw_record(
                    second["cell1"]["seeds"][0],
                    second_cfg["J_crit"])
            attempts.append(second)
            repeat_pos += 1
        out = theory.decide_e6_study(attempts, e6_cfg)
        first_verdict = out["attempts"][0]["verdict"]
        branches[f"first_{first_verdict}"] += 1
        if len(out["attempts"]) == 2:
            branches["repeat_triggered"] += 1
            second_verdict = out["attempts"][1]["verdict"]
            branches[f"second_{second_verdict}"] += 1
        verdicts.append(out["verdict"])
        branches[f"final_{out['verdict']}"] += 1
    counts = {
        verdict: verdicts.count(verdict)
        for verdict in ("PASS", "FAIL", "KILL", "INDETERMINATE")
    }
    verdict_mc = dict(
        reps=reps,
        pass_rate=float(np.mean([v == "PASS" for v in verdicts])),
        counts=counts,
        branch_counts=branches,
        censoring_model=dict(
            primary_rate=censor_rate, repeat_rate=repeat_rate))

    return dict(threshold=K, config=e6_cfg,
                alpha_declared=primary["alpha"],
                worst=primary_chars["worst"],
                corners=primary_chars["corners"],
                power_planning=primary_chars["power_planning"],
                power_dep_min=primary_chars["power_dep_min"],
                table=primary_chars["table"],
                verdict_mc=verdict_mc,
                seeds_per_cell=shape["seeds_per_cell"],
                cells_required=shape["cells_required"],
                cell_majority=shape["cell_majority"],
                alt_planning=dict(alt),
                null_family=dict(spec["null_family"]),
                pilot_bands=dict(spec["pilot_bands"]),
                attempt_calibrations=calibrations,
                attempt_characteristics=attempt_characteristics,
                design_search=design_search,
                attempts=e6_cfg["attempts"],
                study_alpha=e6_cfg["study_alpha"],
                alpha_sum=e6_cfg["alpha_sum"])


def _power_e7(pl, mf, rng):
    """Joint MC: bias-corrected tau MLE inside tolerance, gamma
    inside tolerance, crackling identity, and route-rate ordering."""
    smin, scut = pl["e7_s_min"], pl["e7_s_cutoff"]
    fit_hi = pl["e7_fit_hi"]
    tau_true = mf["tau"]
    svals = np.arange(smin, 2001)
    w = svals ** (-tau_true) * np.exp(-svals / scut)
    w = w / w.sum()
    taugrid = np.arange(1.1, 2.0, 0.0025)
    fit_s = np.arange(smin, fit_hi + 1)
    logZ = np.log(np.power.outer(
        fit_s.astype(float), -taugrid).sum(axis=0))

    def tau_hat(sample):
        s = sample[(sample >= smin) & (sample <= fit_hi)]
        if len(s) < 30:
            return None
        ll = -taugrid * np.log(s).sum() - len(s) * logZ
        return taugrid[np.argmax(ll)]

    trials = 400
    n = pl["n_events_e7"]
    raw = []
    for _ in range(trials):
        sample = rng.choice(svals, size=n, p=w)
        th = tau_hat(sample)
        raw.append(th if th is not None else np.nan)
    raw = np.array(raw, float)
    bias = float(np.nanmedian(raw)) - tau_true
    ok_tau = np.abs(raw - bias - tau_true) <= pl["e7_tau_tol"]
    gam = rng.normal(mf["gamma"], pl["e7_gamma_se"], size=trials)
    ok_gam = np.abs(gam - mf["gamma"]) <= pl["e7_gamma_tol"]
    ident = np.abs(gam * (raw - bias - 1.0)
                   - (mf["alpha_D"] - 1.0)) <= 0.3
    gapz = pl["route_gap_over_se"]
    r = np.array([2.0, 1.0, 0.0]) * gapz
    rh = r[None, :] + rng.normal(0.0, 1.0, size=(trials, 3))
    ok_route = (rh[:, 0] > rh[:, 1]) & (rh[:, 1] > rh[:, 2])
    ok = ok_tau & ok_gam & ident & ok_route \
        & np.isfinite(raw)
    return float(ok.mean()), bias


def _power_e8(pl, cal, rng):
    """E8: the baseline-comparison operating characteristic, by
    direct MC through theory.gated_model_step.  Data are generated
    by the recurrent planning instance with the E5 planning d0
    jitter, stochastic event seeds, and jet readout noise; the
    surrogate's one-step predictions (the deterministic planning
    instance at the replica's own logged d0) compete against the
    preregistered baselines: a memoryless Bernoulli event-word
    model at the in-sample rate (the density-matched null) and an
    in-sample AR(1) jet model (the local-quadratic stand-in).  A
    replica succeeds when the surrogate wins BOTH comparisons
    (event-word log-loss and jet one-step error).  Returns (power,
    reps, T).  This is an ASSUMED-MODEL characteristic: the data
    come from the surrogate itself, so the number validates the
    decision rule and the scoring code, not the mechanism; the
    confirmatory comparison runs on E0-prime-authorized training data."""
    gp = cal["gated_model_planning"]
    fuel = granted_fuel(gp, pl)
    reps, T = 200, 1500
    eps = 0.05
    d0_jit = gp["d0"] * (1.0 + pl["e5_d0_jitter_rel"]
                         * rng.standard_normal(reps))
    c_arr = np.full(reps, gp["c_recurrent"])
    # the surrogate forecast: deterministic run at the logged d0
    params = _gated_params_from(gp, reps)
    params["d0"] = d0_jit
    state = _gated_state_from(gp, reps)
    ref_word = np.zeros((T, reps))
    ref_J = np.zeros((T, reps))
    for t in range(T):
        state, top = theory.gated_model_step(state, params, c_arr,
                                             gp["eta"], fuel)
        ref_word[t] = top.astype(float)
        ref_J[t] = state["J"]
    # the data runs: same logged d0, stochastic seeds and readout
    params = _gated_params_from(gp, reps)
    params["d0"] = d0_jit
    state = _gated_state_from(gp, reps)
    word = np.zeros((T, reps))
    Jobs = np.zeros((T, reps))
    for t in range(T):
        state, top = theory.gated_model_step(state, params, c_arr,
                                             gp["eta"], fuel,
                                             rng=rng)
        word[t] = top.astype(float)
        Jobs[t] = state["J"] + pl["e5_obs_noise_J"] \
            * rng.standard_normal(reps)
    # event-word log-loss: surrogate forecast against the
    # density-matched Bernoulli baseline (in-sample rate, a
    # deliberate handicap for the surrogate)
    p_sur = eps + (1.0 - 2.0 * eps) * ref_word
    ll_sur = -(word * np.log(p_sur)
               + (1.0 - word) * np.log(1.0 - p_sur)).mean(axis=0)
    rate = word.mean(axis=0).clip(eps, 1.0 - eps)
    ll_ber = -(word * np.log(rate[None, :])
               + (1.0 - word) * np.log(1.0 - rate[None, :])
               ).mean(axis=0)
    # jet error: surrogate trajectory against the in-sample AR(1)
    # (local-quadratic stand-in; in-sample fit, same handicap)
    mse_sur = ((Jobs - ref_J) ** 2).mean(axis=0)
    x, y = Jobs[:-1], Jobs[1:]
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    cov = ((x - xm) * (y - ym)).mean(axis=0)
    var = ((x - xm) ** 2).mean(axis=0)
    slope = cov / np.maximum(var, 1e-12)
    icpt = ym - slope * xm
    mse_ar = ((y - (icpt[None, :] + slope[None, :] * x)) ** 2
              ).mean(axis=0)
    ok = (ll_sur < ll_ber) & (mse_sur < mse_ar)
    return float(ok.mean()), reps, T


# ------------------------------------------------------------- main




