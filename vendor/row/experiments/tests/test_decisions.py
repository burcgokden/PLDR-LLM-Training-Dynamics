"""Decision-layer tests for raw endpoints and typed attempts."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
CONFIRM = Path(__file__).resolve().parents[1] / "confirm"
sys.path.insert(0, str(ANALYSIS))
sys.path.insert(0, str(CONFIRM))

import decision_constants as dc  # noqa: E402
import pilot_fit  # noqa: E402
import theory  # noqa: E402

T = 200
E5_STUDY = dc.e5_config(
    J_crit=0.1, entry_bound=50, q_step=0.999,
    m_quantum=0.01)
E5_CFG = E5_STUDY["attempts"][0]
E6_CALS = [
    dict(threshold=33, worst=dict(worst=0.014689708983855389)),
    dict(threshold=38, worst=dict(worst=0.01941241507368547)),
]
E6_STUDY = dc.e6_config(E6_CALS, J_crit=0.1)
E6_CFG = E6_STUDY["attempts"][0]


def _raw(word, act, jet, layer2=None, probes=None):
    word = np.asarray(word, dtype=int)
    act = np.asarray(act, dtype=float)
    jet = np.asarray(jet, dtype=float)
    if probes is None:
        probes = np.arange(1, len(word) + 1)
    probes = np.asarray(probes, dtype=int)
    first = jet[probes - 1]
    second = first if layer2 is None else np.asarray(layer2)[probes - 1]
    layers = np.stack([first, second], axis=1)
    samples = np.repeat(layers[:, :, None], 4, axis=2)
    return dict(word=word, act=act, probe_steps=probes,
                rowmap_samples=samples, normalization=1.0)


def _captured_seed(entry=20, last_event=5):
    word = np.zeros(T, dtype=int)
    word[last_event] = 1
    act = np.zeros(T)
    act[last_event:last_event + 3] = [2, 1, 0]
    jet = np.full(T, 0.05)
    jet[:entry - 1] = 0.5
    return _raw(word, act, jet)


def _separated_seed():
    word = np.zeros(T, dtype=int)
    act = np.zeros(T)
    jet = np.full(T, 0.4)
    for t in range(10, T - 3, 10):
        word[t] = 1
        jet[t] = 0.32
        act[t:t + 3] = [3, 2, 1]
        jet[t + 1:t + 4] = [0.36, 0.38, 0.40]
    return _raw(word, act, jet)


def _gateheld_seed():
    word = np.zeros(T, dtype=int)
    act = np.array([9 - (t % 8) for t in range(T)], dtype=float)
    jet = np.full(T, 0.45)
    for t in range(8, T, 8):
        word[t] = 1
        jet[t] = 0.36
        for k in range(1, min(8, T - t)):
            jet[t + k] = min(0.36 + 0.013 * k, 0.45)
    return _raw(word, act, jet)


def _arms(n=3):
    return dict(
        captured=[_captured_seed() for _ in range(n)],
        separated=[_separated_seed() for _ in range(n)],
        gateheld=[_gateheld_seed() for _ in range(n)],
    )


def _cap_run(captured=True):
    return _captured_seed() if captured else _gateheld_seed()


def _rel_run(captured=False, gateheld=True):
    if captured:
        return _captured_seed(entry=100, last_event=5)
    return _gateheld_seed() if gateheld else _separated_seed()

def _schedule_certificate(side):
    if side == "capture":
        threshold, source = (2.0, 2.0), (1.0, 1.0)
    else:
        threshold, source = (1.0, 1.0), (2.0, 2.0)
    return dict(
        threshold_increment_interval=threshold,
        source_increment_interval=source,
    )



def _cells(n=10):
    return {
        "c1": dict(predicted="capture",
                   schedule_certificate=_schedule_certificate("capture"),
                   seeds=[_cap_run() for _ in range(n)]),
        "c2": dict(predicted="capture",
                   schedule_certificate=_schedule_certificate("capture"),
                   seeds=[_cap_run() for _ in range(n)]),
        "c3": dict(predicted="release",
                   schedule_certificate=_schedule_certificate("release"),
                   seeds=[_rel_run() for _ in range(n)]),
        "c4": dict(predicted="release",
                   schedule_certificate=_schedule_certificate("release"),
                   seeds=[_rel_run(gateheld=False) for _ in range(n)]),
    }


def test_raw_endpoint_first_persistent_all_layer_entry():
    rec = _captured_seed(entry=20)
    endpoint = theory.derive_capture_endpoint(rec, 0.1, 3)
    assert endpoint["status"] == "finite_horizon_entry"
    assert endpoint["entry"] == 20
    assert endpoint["trailing_probes"] == T - 19


def test_raw_endpoint_rejects_trusted_summaries():
    for key, value in (("entry", 20), ("collapsed", True),
                       ("collapse_step", 20), ("J", [0.0] * T)):
        rec = _captured_seed()
        rec[key] = value
        with pytest.raises(theory.DecisionSchemaError):
            theory.derive_capture_endpoint(rec, 0.1, 3)


def test_raw_endpoint_brief_terminal_dip_is_censored():
    word = np.zeros(T)
    act = np.zeros(T)
    jet = np.full(T, 0.5)
    jet[-2:] = 0.05
    endpoint = theory.derive_capture_endpoint(_raw(word, act, jet),
                                               0.1, 3)
    assert endpoint["status"] == "censored"
    assert endpoint["entry"] is None


def test_raw_endpoint_never_hides_a_high_layer():
    word = np.zeros(T)
    act = np.zeros(T)
    low = np.full(T, 0.05)
    high = np.full(T, 0.4)
    endpoint = theory.derive_capture_endpoint(
        _raw(word, act, low, layer2=high), 0.1, 3)
    assert endpoint["status"] == "no_finite_horizon_entry"
    assert np.allclose(endpoint["all_layer_probe"], high)


def test_raw_endpoint_rejects_out_of_range_probe_index():
    rec = _captured_seed()
    rec["probe_steps"][-1] = T + 1
    with pytest.raises(theory.DecisionSchemaError):
        theory.derive_capture_endpoint(rec, 0.1, 3)


def test_decide_e5_pass_and_reports_derived_endpoint():
    out = theory.decide_e5(_arms(), E5_CFG)
    assert out["verdict"] == "PASS", out["reasons"]
    assert out["arms"]["captured"][0]["endpoint"]["entry"] == 20
    assert out["arms"]["separated"][0]["gate_slope"] > 0


def test_decide_e5_event_after_derived_entry_is_causal_kill():
    arms = _arms()
    bad = _captured_seed()
    bad["word"][50] = 1
    bad["act"][50:53] = [2, 1, 0]
    arms["captured"][0] = bad
    out = theory.decide_e5(arms, E5_CFG)
    assert out["verdict"] == "KILL"
    assert any("does not strictly precede" in reason
               for reason in out["reasons"])


def test_decide_e5_requires_reload_gate_closed_after_entry():
    arms = _arms()
    arms["captured"][0]["act"][30] = 1
    out = theory.decide_e5(arms, E5_CFG)
    assert out["verdict"] == "FAIL"
    assert not out["arms"]["captured"][0]["gate_closed_after_entry"]


def test_decide_e5_requires_event_cessation_strictly_before_entry():
    arms = _arms()
    # Entry is global step 20, represented by array index 19.
    arms["captured"][0]["word"][19] = 1
    out = theory.decide_e5(arms, E5_CFG)
    assert out["verdict"] == "KILL"


def test_e5_study_uses_distinct_complete_attempts_without_mutation():
    before = copy.deepcopy(E5_STUDY)
    first = _arms(3)
    # A two-probe horizon is endpoint-censored.
    short = _raw(np.zeros(2), np.zeros(2), np.full(2, 0.4))
    first["separated"][0] = short
    pending = theory.decide_e5_study([first], E5_STUDY)
    assert pending["verdict"] == "INDETERMINATE"
    second = _arms(6)
    resolved = theory.decide_e5_study([first, second], E5_STUDY)
    assert resolved["verdict"] == "PASS"
    assert E5_STUDY == before
    assert E5_STUDY["attempts"][0]["seed_block"] \
        != E5_STUDY["attempts"][1]["seed_block"]


@pytest.mark.parametrize("terminal", ["FAIL", "KILL", "INDETERMINATE"])
def test_e5_second_attempt_terminal_branches(terminal):
    first = _arms(3)
    first["captured"][0] = _raw(
        np.zeros(2), np.zeros(2), np.full(2, 0.4))
    second = _arms(6)
    if terminal == "FAIL":
        quiet_high = _captured_seed()
        quiet_high["rowmap_samples"][:] = 0.5
        second["captured"][0] = quiet_high
    elif terminal == "KILL":
        second["captured"][0]["word"][50] = 1
    else:
        second["captured"][0] = _raw(
            np.zeros(2), np.zeros(2), np.full(2, 0.4))
    out = theory.decide_e5_study([first, second], E5_STUDY)
    assert out["attempts"][1]["verdict"] == terminal
    assert out["verdict"] == terminal


def test_decide_e5_schema_is_strict():
    with pytest.raises(theory.DecisionSchemaError):
        theory.decide_e5({}, E5_CFG)
    bad = _arms()
    del bad["captured"][0]["rowmap_samples"]
    with pytest.raises(theory.DecisionSchemaError):
        theory.decide_e5(bad, E5_CFG)


def test_decide_e6_pass_and_routing():
    out = theory.decide_e6(_cells(), E6_CFG)
    assert out["verdict"] == "PASS"
    assert out["total"] == 40
    assert out["cells"]["c3"]["runs"][0]["routed"]


def test_decide_e6_count_and_coverage_clauses():
    cells = _cells()
    # Seven matches in the fourth cell: total 37, coverage passes.
    cells["c4"]["seeds"][-3:] = [
        _rel_run(captured=True, gateheld=False) for _ in range(3)]
    out = theory.decide_e6(cells, E6_CFG)
    assert out["verdict"] == "PASS" and out["total"] == 37
    # Three matches in the fourth cell: total 33 meets count but not
    # the seven-of-ten coverage clause.
    cells["c4"]["seeds"] = (
        [_rel_run(gateheld=False) for _ in range(3)]
        + [_rel_run(captured=True, gateheld=False) for _ in range(7)])
    out = theory.decide_e6(cells, E6_CFG)
    assert out["total"] == 33
    assert out["verdict"] == "FAIL"


def test_decide_e6_event_after_entry_is_causal_kill():
    cells = _cells()
    bad = _captured_seed()
    bad["word"][50] = 1
    bad["act"][50:53] = [2, 1, 0]
    cells["c1"]["seeds"][0] = bad
    out = theory.decide_e6(cells, E6_CFG)
    assert out["verdict"] == "KILL"


def test_decide_e6_capture_match_requires_closed_reload_gate():
    cells = _cells()
    cells["c1"]["seeds"][0]["act"][30] = 1
    out = theory.decide_e6(cells, E6_CFG)
    run = out["cells"]["c1"]["runs"][0]
    assert run["match"] is False
    assert run["gate_closed_after_entry"] is False


def test_e6_study_has_fresh_distinct_calibrated_attempts():
    assert E6_STUDY["alpha_sum"] <= E6_STUDY["study_alpha"]
    first, second = E6_STUDY["attempts"]
    assert (first["seeds_per_cell"], first["total_threshold"]) == (10, 33)
    assert (second["seeds_per_cell"], second["total_threshold"]) == (12, 38)
    assert first["seed_block"] != second["seed_block"]
    cells = _cells(10)
    short = _raw(np.zeros(2), np.zeros(2), np.full(2, 0.4))
    cells["c3"]["seeds"][0] = short
    pending = theory.decide_e6_study([cells], E6_STUDY)
    assert pending["repeat_pending"]
    resolved = theory.decide_e6_study([cells, _cells(12)], E6_STUDY)
    assert resolved["verdict"] == "PASS"


@pytest.mark.parametrize("terminal", ["FAIL", "KILL", "INDETERMINATE"])
def test_e6_second_attempt_terminal_branches(terminal):
    first = _cells(10)
    first["c1"]["seeds"][0] = _raw(
        np.zeros(2), np.zeros(2), np.full(2, 0.4))
    second = _cells(12)
    if terminal == "FAIL":
        second["c4"]["seeds"] = (
            [_rel_run(gateheld=False) for _ in range(2)]
            + [_rel_run(captured=True, gateheld=False) for _ in range(10)])
    elif terminal == "KILL":
        second["c1"]["seeds"][0]["word"][50] = 1
    else:
        second["c1"]["seeds"][0] = _raw(
            np.zeros(2), np.zeros(2), np.full(2, 0.4))
    out = theory.decide_e6_study([first, second], E6_STUDY)
    assert out["attempts"][1]["verdict"] == terminal
    assert out["verdict"] == terminal


def test_e6_schema_requires_two_cells_per_side():
    cells = _cells()
    cells["c2"]["predicted"] = "release"
    with pytest.raises(theory.DecisionSchemaError):
        theory.decide_e6(cells, E6_CFG)


def test_e6_schema_rejects_boundary_schedule_cell():
    cells = _cells()
    cells["c1"]["schedule_certificate"] = dict(
        threshold_increment_interval=(1.0, 1.3),
        source_increment_interval=(1.3, 2.0),
    )
    with pytest.raises(theory.DecisionSchemaError, match="boundary cell"):
        theory.decide_e6(cells, E6_CFG)

def test_e6_exact_attempt_calibrations_control_union_alpha():
    expected = ((10, 7, 0.025, 33), (12, 8, 0.025, 38))
    search = theory.e6_search_attempt_designs()
    assert search["power_floor"] == 0.80
    worsts = []
    for template, calibration, (_, _, alpha, threshold) in zip(
            search["selected_templates"], search["calibrations"], expected):
        assert calibration["threshold"] == threshold
        assert calibration["worst"]["worst"] <= alpha
        assert template["seeds_per_cell"] in (10, 12)
        worsts.append(calibration["worst"]["worst"])
    assert sum(a["alpha"] for a in dc.E6_SPEC["attempts"]) <= 0.05
    assert worsts[0] < 0.015 and worsts[1] < 0.020


def test_e6_exchangeable_enumeration_retains_exact_shared_corners():
    first = dc.E6_SPEC["attempts"][0]
    shape = {key: first[key] for key in
             ("seeds_per_cell", "n_cells", "cell_majority",
              "cells_required")}
    assert abs(theory.e6_pass_prob_exact(33, shape, 0.5, 0, 1)
               - 11 / 1024) < 1e-12
    second = dc.E6_SPEC["attempts"][1]
    second_shape = {key: second[key] for key in
                    ("seeds_per_cell", "n_cells", "cell_majority",
                     "cells_required")}
    assert abs(theory.e6_pass_prob_exact(
        38, second_shape, 0.5, 0, 1) - 79 / 4096) < 1e-12


def test_pilot_inverts_every_grid_member_with_simultaneous_mc_bound():
    matrix = np.array([[1, 1], [1, 0], [1, 1], [0, 1]])
    result = pilot_fit.finite_grid_neyman_region(matrix, n_cal=20, seed=7)
    assert result["region"]
    assert {"p", "delta", "w", "member_statistic"} <= result["region"][0].keys()
    assert result["coverage_lower"] <= result["coverage_hat"]
    assert len(result["likelihood_surface"]) == 180
    assert result["calibration_scope"] == "finite_grid_neyman_inversion"
    assert result["simultaneous_calibration_confidence"] == pytest.approx(
        1.0 - dc.E6_SPEC["pilot"]["mc_error_alpha"])
    assert result["unresolved_quantile"]


def test_pilot_precision_state_machine():
    base = dict(coverage_lower=0.96)
    matrix = np.zeros((4, 8), dtype=int)
    confirm = dict(base, region=[
        dict(p=0.8, delta=0.1, w=0.1),
        dict(p=0.9, delta=0.2, w=0.2)])
    assert pilot_fit.precision_and_fallback(confirm, matrix)["status"] \
        == "confirmatory"
    expand = dict(base, region=[
        dict(p=0.6, delta=0.0, w=0.0),
        dict(p=0.9, delta=0.2, w=0.2)])
    assert pilot_fit.precision_and_fallback(expand, matrix)["status"] \
        == "expand_pilot"
    sharing = dict(base, region=[
        dict(p=0.8, delta=0.1, w=0.6),
        dict(p=0.9, delta=0.2, w=0.7)])
    assert pilot_fit.precision_and_fallback(sharing, matrix)["status"] \
        == "independent_block_fallback"
    capped = dict(base, region=[
        dict(p=0.1, delta=0.0, w=0.0),
        dict(p=0.9, delta=0.25, w=1.0)])
    assert pilot_fit.precision_and_fallback(capped, matrix)["status"] \
        == "exploratory"


def test_independent_block_fallback_recalibrates_both_attempts():
    region = dict(region=[dict(p=0.9, delta=0.05, w=0.75)])
    out = pilot_fit.independent_block_fallback(region)
    assert out["null_family"]["w_max"] == 0.0
    assert out["study"]["alpha_sum"] <= out["study"]["study_alpha"]
    assert len(out["study"]["attempts"]) == 2
    assert len(out["region_power"]) == 2


def test_matched_block_randomization_enumerates_registered_labels():
    cap = np.zeros((4, 10), dtype=bool)
    rel = np.zeros((4, 10), dtype=bool)
    cap[:2] = True
    rel[2:] = True
    out = pilot_fit.matched_block_randomization(cap, rel, E6_CFG)
    assert out["n_assignments"] == 6
    assert out["pass_count"] == 1
    assert out["conditional_pass_probability"] == pytest.approx(1 / 6)


def test_preconditioner_replacement_ceiling():
    assert theory.precond_replacement_ceiling(0.001, 100) == 0.4
    with pytest.raises(ValueError):
        theory.precond_replacement_ceiling(0.01, 100)


def test_phenotype_sensitivity_table():
    rec = theory.run_separated_witness(n_steps=48)
    rows = theory.phenotype_sensitivity(rec["word"], rec["act_series"])
    assert len(rows) == 36
    assert all(row["phen"] == "sep-window" for row in rows
               if row["window_frac"] == 1.0)
