"""The separated witness: the pinned second-regime regression.

The separated witness orbit is the PERMANENT INHABITATION CONTROL
of the completed-reload recurrent regime (paper Ex. sepwitness;
Lean RecurrentSeparated.separated_witness_class): a legal one-site
orbit under the period-four schedule (c, eta, delta_J0) =
(2, 0.1, 0.5) on the three quiet steps and (0.5, 0.1, 2) on the
event step, with the base pinned at its exact fixed point 0.75.
Every event exits hot at the exhaustion ceiling 0.1, every written
reload duration is ONE against a four-step gap (the completion
clause J_hold = 0.15 <= 0.2 = the write quantum, the exact
opposite side of the write law from the intermittent overhang),
the counter reaches zero at the FIRST quiet step of every gap, and
the gate CLOSES between bursts: every successive pair is
reload-separated, the sep-window phenotype with zero unseparated
pairs.  The test iterates the FULL global map with validation
enabled and pins the topple word, the written durations, the
zero-counter states, the base fixed point, the hot-exit jets, and
the jet band at their exact values.  The companion test pins the
TERMINAL-RELOAD fact behind the gate-held structure (Lean
GateHeldIntermittent.reload_terminal_class_fin): at a constant
schedule, a state with a completed reload and sub-threshold
stress never topples again, so separated recurrence at bounded
stress REQUIRES schedule modulation."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "analysis"))
import theory  # noqa: E402


FUEL = 200_000

# the pinned first/last post-event jets of the 48-step prefix
PINNED_J_OUT_FIRST = 0.098610137811
PINNED_J_OUT_LAST = 0.099975466407


def test_witness_word_counter_and_separation():
    """The topple word is (0001)^12, every written reload duration
    is one, the counter is zero at every non-event post-step
    state, and every successive pair is reload-separated: the
    sep-window phenotype with gate-closure fraction one."""
    rec = theory.run_separated_witness(n_steps=48, fuel=FUEL)
    assert "".join(map(str, rec["word"])) == "0001" * 12
    stats = rec["stats"]
    assert stats["phen"] == "sep-window"
    assert stats["max_run"] == 1
    assert stats["n_events"] == 12
    assert stats["n_pairs"] == 11
    assert stats["n_unsep"] == 0
    assert stats["gate_closure_frac"] == 1.0
    acts = [int(a) for a in rec["act_series"]]
    words = rec["word"]
    for a, b in zip(acts, words):
        assert a == b, "written duration one at events, zero " \
            "elsewhere"


def test_witness_base_fixed_point_and_jet_band():
    """The base is an exact fixed point through both phases; every
    post-event jet sits at or under the exhaustion ceiling 0.1 and
    every rebuilt jet at or under the hold cap 0.15."""
    rec = theory.run_separated_witness(n_steps=48, fuel=FUEL)
    assert abs(float(rec["state"]["b"][0]) - 0.75) < 1e-12
    assert max(rec["J_out"]) <= 0.1
    assert max(rec["J_series"]) <= 0.15
    assert abs(rec["J_out"][0] - PINNED_J_OUT_FIRST) < 1e-11
    assert abs(rec["J_out"][-1] - PINNED_J_OUT_LAST) < 1e-11


def test_witness_matches_lean_constants():
    """The executable constants equal the Lean witness constants
    (RecurrentSeparated.sepPhi/sepPlo/sepBounds/sepJHold/sepEntry):
    the three pinned forms share one constant block."""
    state, params, schedule = theory.separated_witness_config()
    assert float(state["b"][0]) == 0.75
    assert float(state["J"][0]) == 0.1
    assert int(state["act"][0]) == 1
    c_hi, eta_hi, dj_hi = schedule(0)
    c_lo, eta_lo, dj_lo = schedule(3)
    assert (float(c_hi[0]), eta_hi, float(dj_hi[0])) \
        == (2.0, 0.1, 0.5)
    assert (float(c_lo[0]), eta_lo, float(dj_lo[0])) \
        == (0.5, 0.1, 2.0)
    assert float(params["d0"][0]) == 0.075
    assert params["mu_b"] == 0.1
    assert float(params["th_J_loss"][0]) == 0.1
    assert params["burst"]["z_floor"] == 0.001
    assert params["burst"]["y_floor"] == 0.01
    # the completion margin: J_hold <= the event-step write quantum
    assert 0.1 + eta_hi * float(dj_hi[0]) <= 0.15 + 1e-12
    assert 0.15 <= eta_lo * float(dj_lo[0])


def test_completed_reload_terminal_at_constant_schedule():
    """The terminal-reload fact (Lean GateHeldIntermittent.
    reload_terminal_class_fin), executable: at the constant
    schedule of the gate-held instance, a state with counter ZERO
    and sub-threshold stress never topples again; the jet decays
    in place and the guard stays closed.  Separated recurrence at
    bounded stress therefore requires schedule modulation: at a
    constant schedule a completed reload ENDS the event set."""
    state = dict(b=np.array([0.75]), a=np.array([1.0]),
                 g=np.array([1.0]), J=np.array([0.15]),
                 w=np.array([0.0]), pend=np.array([0.0]),
                 act=np.array([0.0]))
    params = dict(beta=np.zeros((1, 1)), d0=np.array([0.075]),
                  delta_a=np.array([0.0]),
                  delta_J0=np.array([0.5]), kappa=np.array([1.0]),
                  th_a_loss=np.array([0.0]),
                  th_g_loss=np.array([0.0]),
                  th_J_loss=np.array([0.05]), lam_wd=0.0,
                  mu_b=0.1, s_shed=0.0,
                  burst=dict(h=0.1, b_sat=1.0, z_seed=0.01,
                             z_floor=0.001, y_floor=0.01,
                             tau_z=0.02, w_kick_max=0.1))
    J0 = float(state["J"][0])
    for _t in range(500):
        state, top = theory.gated_model_step(
            state, params, np.array([0.8]), 0.05, fuel=FUEL,
            validate=True)
        assert not bool(top[0]), "an event after a completed " \
            "reload at constant schedule"
        assert float(state["act"][0]) == 0.0
    assert float(state["J"][0]) < J0
