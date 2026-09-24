"""Regression tests for the Rev22 construct-validity contract."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "analysis"))
sys.path.insert(0, str(EXP / "confirm"))

import e0_decision as D  # noqa: E402
import factory_residual as FR  # noqa: E402
import instrument  # noqa: E402
import optimizer_ledger as OL  # noqa: E402
import oriented_headroom as OH  # noqa: E402
import pilot_fit  # noqa: E402
import source_edge_certificate as SEC  # noqa: E402
import stress_units as SU  # noqa: E402
import theory  # noqa: E402


def _record(**changes):
    values = dict(
        schema_version=D.SCHEMA_VERSION,
        protocol_hash="a" * 64,
        schema_hash="b" * 64,
        source_hash="c" * 64,
        calibration_hash="d" * 64,
        curvature_kind="gauss_newton",
        metric_kind="preconditioned",
        unit_kind="physical",
        residual_certificate_pass=True,
        n_upper_crossings=40,
        pilot_region_status="nonempty",
        model_consistent=True,
        uncertainty_resolved=True,
        prerequisites={name: True for name in D.REQUIRED_PREREQUISITES},
        family_p_values={name: 1e-4 for name in D.INFERENTIAL_FAMILIES},
        family_directional_pass={
            name: True for name in D.INFERENTIAL_FAMILIES},
    )
    values.update(changes)
    return D.E0Record(**values)


def test_primary_stress_units_and_moving_schedule_identity():
    SU.require_primary(SU.PRIMARY_STRESS)
    with pytest.raises(ValueError, match="primary stress"):
        SU.require_primary(SU.StressMetadata(
            SU.CurvatureKind.FULL_HESSIAN,
            SU.MetricKind.PRECONDITIONED,
            SU.UnitKind.PHYSICAL))
    out = SU.normalized_stress_increment(3.0, 4.0, 0.1, 0.02, 0.01)
    assert out["total"] == pytest.approx(
        out["curvature"] + out["schedule"])
    assert out["total"] == pytest.approx(0.1 * (0.01 * 4 - 0.02 * 3))


def test_oriented_upper_headroom_is_not_nearest_edge_margin():
    state = np.array([0.2, 0.8])
    nearest = OH.certificate_margin_nearest(state, 0.0, 1.0)
    upper = OH.topple_headroom_upper(state, 1.0)
    assert np.allclose(nearest, [0.2, 0.2])
    assert np.allclose(upper, [0.8, 0.2])
    assert not OH.strict_upper_child(0.8, 0.2, 1.0)
    assert OH.strict_upper_child(0.8, 0.2000001, 1.0)
    assert OH.strict_empirical_upper_cdf([0.1, 0.2, 0.2], 0.2) \
        == pytest.approx(1 / 3)


def test_total_e0_decision_partition_and_exact_holm_family_count():
    assert D.decide_e0(_record()).outcome is D.E0Outcome.PASS
    assert D.decide_e0(_record(n_upper_crossings=39)).outcome \
        is D.E0Outcome.NOT_CONFIRMED
    assert D.decide_e0(_record(
        pilot_region_status="empty", model_consistent=False)).outcome \
        is D.E0Outcome.MODEL_REJECTED
    assert D.decide_e0(_record(
        pilot_region_status="unresolved",
        uncertainty_resolved=False)).outcome is D.E0Outcome.INDETERMINATE
    missing = {name: True for name in D.REQUIRED_PREREQUISITES}
    missing["all_values_present"] = False
    assert D.decide_e0(_record(prerequisites=missing)).outcome \
        is D.E0Outcome.INCOMPLETE
    with pytest.raises(ValueError, match="exactly three"):
        D.holm_all_pass({**_record().family_p_values, "extra": 0.1})


def test_gauss_newton_matvec_excludes_model_curvature():
    weight = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    output = weight.square()
    loss = 0.5 * output.square().sum()
    mv = instrument.gauss_newton_matvec(loss, output, [weight])
    assert float(mv([torch.ones_like(weight)])[0]) == pytest.approx(16.0)
    mv_pre = instrument.gauss_newton_matvec(
        loss, output, [weight], [torch.full_like(weight, 0.5)])
    assert float(mv_pre([torch.ones_like(weight)])[0]) \
        == pytest.approx(4.0)


def test_direct_adamw_decay_ledger_is_roundoff_small():
    parameter = torch.nn.Parameter(torch.tensor([1.5], dtype=torch.float64))
    optimizer = torch.optim.AdamW(
        [parameter], lr=0.01, betas=(0.9, 0.95), eps=1e-5,
        weight_decay=0.1)
    before = [parameter.detach().clone()]
    (parameter.square().sum()).backward()
    optimizer.step()
    residual = OL.adamw_step_ledger_residual(
        optimizer, [parameter], before, 0.01)
    assert residual < 1e-11


def test_empty_pilot_region_is_typed_model_rejection():
    region = {"region": [], "coverage_lower": 0.95,
              "unresolved_quantile": False}
    matrix = np.zeros((4, 3), dtype=int)
    precision = pilot_fit.precision_and_fallback(region, matrix)
    action = pilot_fit.select_pilot_action(region, matrix)
    assert precision["region_status"] == "empty"
    assert precision["status"] == "MODEL_REJECTED"
    assert action["status"] == "MODEL_REJECTED"
    assert action["shared_design"] is None


def test_source_edge_certificate_uses_uniform_finite_horizon_envelopes():
    windows = []
    for index, physical_threshold in enumerate((1, 3, 5, 7, 9)):
        windows.append({
            "w": index, "lo": 10 * index + 1, "hi": 10 * index + 10,
            "eta": 0.1, "band_hi": 0.01 * physical_threshold,
            "band_empty": False, "excluded": False,
        })
    loading = {"closed": {"windows": [
        [index, 0.1, 0.1, 0.1] for index in range(5)]},
        "open": {"windows": []}}
    result = SEC.finite_horizon_certificate(
        windows, loading, 0.1, theory.classify_schedule_certificate)
    assert result["certificate_ready"]
    assert result["classification"] == "capture"
    assert result["scope"] == "finite_observed_horizon"
    assert result["n_pairs"] == 4


def test_factory_residual_uses_physical_curvature_decomposition():
    boundary = {
        "kappa": {"kappa": 2.0},
        "primary_series": {
            "t": [1, 2], "x": [5.0, 5.0],
            "factor_t": [1, 2], "a": [1.0, 1.0], "J": [1.0, 1.0],
        },
    }
    # b=3 and kappa*u^2=2, so 2*p*kappa*u^2 + 2*b is 14 or 18.
    battery = {"ckpts": [{
        "step": 1, "primary_defined": True,
        "radial_deriv_p2": 14.0, "radial_deriv_p3": 18.0,
    }]}
    result = FR.certificate(boundary, battery, relative_limit=1e-12)
    assert result["certificate_ready"]
    assert result["pass"]
    assert result["relative_residual_max"] == pytest.approx(0.0)
