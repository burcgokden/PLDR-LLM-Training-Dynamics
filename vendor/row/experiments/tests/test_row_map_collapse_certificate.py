"""Regression tests for the complete row-map collapse certificate."""

import sys
from pathlib import Path

import numpy as np
import pytest

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(ANALYSIS))

import theory  # noqa: E402


def test_threshold_quotient_increment_is_exact():
    out = theory.threshold_increment_exact(
        edge_t=0.8, edge_next=0.77, one_minus_beta1=0.1,
        eta_t=0.02, eta_next=0.015)
    assert out["total"] == pytest.approx(out["direct"])
    assert out["total"] == pytest.approx(out["schedule"] + out["edge"])


def test_stationary_threshold_can_cancel_nominal_schedule_motion():
    b, eta_t, eta_next, c0 = 0.1, 0.02, 0.01, 3.0
    edge_t = b * eta_t * c0
    edge_next = b * eta_next * c0
    out = theory.threshold_increment_exact(
        edge_t, edge_next, b, eta_t, eta_next)
    assert out["schedule"] > 0
    assert out["edge"] < 0
    assert out["total"] == pytest.approx(0.0)


def test_capture_requires_simultaneous_edge_lower_bound():
    with pytest.raises(ValueError, match="edge_increment_lower"):
        theory.capture_lhs(0.02, 0.01, 3.0)
    value = theory.capture_lhs(
        0.02, 0.01, 3.0, edge_increment_lower=-0.003,
        one_minus_beta1=0.1)
    assert value == pytest.approx(0.0)


def test_interval_classifier_keeps_touching_case_on_boundary():
    assert theory.classify_schedule_certificate((2.0, 2.4), (1.0, 1.9)) \
        == "capture"
    assert theory.classify_schedule_certificate((1.0, 1.2), (1.3, 2.0)) \
        == "release"
    assert theory.classify_schedule_certificate((1.0, 1.3), (1.3, 2.0)) \
        == "boundary"


def test_fixed_receiver_envelope_exposes_heterogeneous_parent_failure():
    out = theory.fixed_receiver_envelope([[0.9, 0.0], [0.45, 0.45]])
    assert np.allclose(out["q"], [0.9, 0.45])
    assert out["mean"] == pytest.approx(1.35)
    assert not out["subcritical"]
    assert max(map(sum, [[0.9, 0.0], [0.45, 0.45]])) < 1.0


def test_multitype_weight_certificate_and_strict_cdf():
    cert = theory.multitype_weight_certificate(
        [[0.2, 0.1], [0.1, 0.3]], [1.0, 1.0], 0.5)
    assert cert["certified"]
    assert theory.strict_empirical_margin_cdf([0.1, 0.2, 0.2, 0.3], 0.2) \
        == pytest.approx(0.25)


def test_whitening_safe_bound_holds_without_t1_maximum_claim():
    for beta1, beta2 in ((0.9, 0.999), (0.2, 0.99), (0.9, 0.1)):
        bound = theory.whitening_transient_safe_bound(beta1)
        values = [theory.whitening_transient(t, beta1, beta2)
                  for t in range(1, 101)]
        assert max(values) <= bound + 1e-12


def test_returned_shed_conservation_and_endpoints():
    middle = theory.returned_shed(M=1.4, c=1.0, r=0.2, f=0.25)
    assert middle["local"] == pytest.approx(0.5)
    assert middle["exported"] == pytest.approx(0.9)
    assert middle["conserved"] == pytest.approx(1.4)
    assert theory.returned_shed(1.4, 1.0, 0.2, 0.0)["local"] \
        == pytest.approx(0.2)
    assert theory.returned_shed(1.4, 1.0, 0.2, 1.0)["exported"] \
        == pytest.approx(0.0)


def _endpoint_record(normalization):
    samples = np.array([
        [[0.4, 0.4], [0.8, 0.8]],
        [[0.1, 0.1], [0.2, 0.2]],
        [[0.05, 0.05], [0.1, 0.1]],
    ])
    return dict(word=np.zeros(8), act=np.zeros(8),
                probe_steps=np.array([3, 5, 8]),
                rowmap_samples=samples, normalization=normalization)


def test_endpoint_hold_is_missing_before_first_probe_and_normalizers_agree():
    scalar = theory.derive_capture_endpoint(_endpoint_record(2.0), 0.11, 2)
    layer = theory.derive_capture_endpoint(
        _endpoint_record(np.array([2.0, 2.0])), 0.11, 2)
    time_layer = theory.derive_capture_endpoint(
        _endpoint_record(np.full((3, 2), 2.0)), 0.11, 2)
    assert np.isnan(scalar["jet_step"][:2]).all()
    assert not scalar["observation_mask"][:2].any()
    assert scalar["status"] == "finite_horizon_entry"
    assert scalar["entry"] == 5
    assert np.allclose(scalar["all_layer_probe"], layer["all_layer_probe"])
    assert np.allclose(layer["all_layer_probe"], time_layer["all_layer_probe"])
    with pytest.raises(theory.DecisionSchemaError, match="global step 1"):
        theory.derive_capture_endpoint(
            _endpoint_record(2.0), 0.11, 2, require_step_one=True)
