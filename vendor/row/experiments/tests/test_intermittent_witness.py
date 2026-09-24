"""The intermittent witness: the pinned fourth-regime regression.

The intermittent witness orbit is the PERMANENT INHABITATION
CONTROL of the corrected event-count partition (paper Prop.
topplepartition, Ex. intwitness; Lean
ClosureFourRegime.intermittent_witness_class): a legal one-site
orbit under the alternating two-phase schedule
(c, eta) = (2, 0.1) / (0.5, 0.001) with the base pinned at its
exact fixed point 0.75.  The orbit has INFINITELY many events at
bounded run length one with NO completed reload between events, so
it lies in neither the separated nor the unbounded-run class: it
inhabits the intermittent class.  The test iterates the FULL
global map with validation enabled and pins the topple word, the
run length, the counter floor, the base fixed point, the hot-exit
jets, and the two-sided jet band at their exact values; the
companion checks pin the window-phenotype classifier on one
representative word of EACH phenotype, so a classifier change
cannot silently reassign the observations the decision rules
consume (a finite window never assigns an asymptotic class)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "analysis"))
import theory  # noqa: E402


FUEL = 200_000

# the pinned first/last post-event jets of the 40-step prefix (the
# review-witness trace, reproduced bit-exactly)
PINNED_J_OUT_FIRST = 0.099993485577
PINNED_J_OUT_LAST = 0.099987341559


def test_witness_word_run_and_counter():
    """The topple word is 0101..., the maximum run is one, and the
    counter never reaches zero: the orbit is in the intermittent
    class (infinite events, bounded runs, no completed reload)."""
    rec = theory.run_intermittent_witness(n_steps=40, fuel=FUEL)
    assert "".join(map(str, rec["word"])) == "01" * 20
    stats = rec["stats"]
    assert stats["phen"] == "unsep-window"
    assert stats["max_run"] == 1
    assert stats["n_events"] == 20
    assert stats["density"] == 0.5
    assert stats["n_pairs"] == 19
    assert stats["n_unsep"] == 19
    assert stats["unsep_frac"] == 1.0
    assert stats["gate_closure_frac"] == 0.0
    assert min(rec["act_series"]) >= 99.0


def test_witness_base_fixed_point_and_jet_band():
    """The base is an exact fixed point at both rates; every
    post-event jet sits at the exhaustion ceiling and every
    post-rebuild jet at the intermittent band top."""
    rec = theory.run_intermittent_witness(n_steps=40, fuel=FUEL)
    assert abs(float(rec["state"]["b"][0]) - 0.75) < 1e-12
    # hot exits: J_out <= sqrt(y_floor) = 0.1, pinned to the trace
    assert abs(rec["J_out"][0] - PINNED_J_OUT_FIRST) < 1e-12
    assert abs(rec["J_out"][-1] - PINNED_J_OUT_LAST) < 1e-12
    assert max(rec["J_out"]) <= 0.1
    # the two-sided band: post-rebuild jets near 0.299, inside the
    # certificate ceiling Jout + eta_hi*deltaJ0 = 0.3
    post_rebuild = [j for j, b in zip(rec["J_series"], rec["word"])
                    if b == 0]
    assert min(post_rebuild) > 0.29
    assert max(post_rebuild) <= 0.3


def test_witness_reload_overhang():
    """Every event writes a reload duration far above the one-step
    gap (the (I-over) overhang: about 101 against a gap of one), so
    no reload can complete between events."""
    state, params, schedule = theory.intermittent_witness_config()
    writes = []
    prev_act = float(state["act"][0])
    for t in range(40):
        c_t, eta_t = schedule(t)
        state, toppled = theory.gated_model_step(
            state, params, c_t, eta_t, fuel=FUEL, validate=True)
        if bool(toppled[0]):
            writes.append(float(state["act"][0]))
            assert prev_act >= 1.0
        prev_act = float(state["act"][0])
    assert writes
    assert min(writes) >= 51.0


def test_classifier_quiet_tail():
    """A word whose events cease before the last quarter receives
    the quiet-tail phenotype (the capture certificate's bridge
    prediction)."""
    word = [1, 0, 1, 0] + [0] * 36
    act = [3, 2, 3, 2] + [1, 0] + [0] * 34
    stats = theory.classify_event_word(word, act)
    assert stats["phen"] == "quiet-tail"


def test_classifier_long_run():
    """A unit-density word receives the long-run phenotype (the
    sustained-drive certificate's bridge prediction)."""
    word = [1] * 40
    act = [1] * 40
    stats = theory.classify_event_word(word, act)
    assert stats["phen"] == "long-run"
    assert stats["max_run"] == 40


def test_classifier_sep_window():
    """Isolated events with the counter reaching zero in every gap
    receive the sep-window phenotype (the completed-reload
    certificate's bridge prediction)."""
    word, act = [], []
    for _ in range(8):
        word += [1, 0, 0, 0, 0]
        act += [2, 1, 0, 0, 0]
    stats = theory.classify_event_word(word, act)
    assert stats["phen"] == "sep-window"
    assert stats["gate_closure_frac"] == 1.0


def test_classifier_unsep_window():
    """Isolated events whose reloads never complete receive the
    unsep-window phenotype, and the driven witness does NOT: the
    two signatures are separated by the run length, not the
    density."""
    word, act = [], []
    for _ in range(10):
        word += [1, 0]
        act += [5, 4]
    stats = theory.classify_event_word(word, act)
    assert stats["phen"] == "unsep-window"
    assert stats["unsep_frac"] == 1.0


def test_classifier_rejects_mismatched_lengths():
    """The classifier refuses a word/counter length mismatch rather
    than silently truncating."""
    import pytest
    with pytest.raises(ValueError):
        theory.classify_event_word([1, 0], [1.0])


def test_constant_schedule_intermittent_instance():
    """The intermittent class is not an artifact of schedule
    alternation (paper Rem. intconstant): at the constant schedule
    (c, eta) = (0.8, 0.05) with the base pinned at its fixed point
    0.75, delta_J0 = 0.5 and th_J = 0.05, the executable transition
    runs an exactly periodic orbit of period six, one cold event
    per period with a written reload of six against a six-step gap
    and counter floor one: infinitely many events, bounded runs,
    no completed reload.  Pinned at executable strength; the
    alternating witness carries the theorem."""
    state = dict(b=np.array([0.75]), a=np.array([1.0]),
                 g=np.array([1.0]), J=np.array([0.1]),
                 w=np.array([0.0]), pend=np.array([0.0]),
                 act=np.array([50.0]))
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
    word, act, writes = [], [], []
    for _t in range(1200):
        state, top = theory.gated_model_step(
            state, params, np.array([0.8]), 0.05, fuel=FUEL,
            validate=True)
        word.append(int(bool(top[0])))
        act.append(float(state["act"][0]))
        if word[-1]:
            writes.append(float(state["act"][0]))
    stats = theory.classify_event_word(word, act)
    assert stats["phen"] == "unsep-window"
    assert stats["max_run"] == 1
    assert stats["unsep_frac"] == 1.0
    assert min(act) >= 1.0
    assert writes and min(writes) == 6.0
    events = [i for i, b in enumerate(word) if b]
    gaps = {b - a for a, b in zip(events[:-1], events[1:])}
    assert gaps == {6}
    assert abs(float(state["b"][0]) - 0.75) < 1e-9
