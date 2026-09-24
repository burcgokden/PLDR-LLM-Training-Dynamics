"""Finite categorical geometry must preserve gauges and signed contrasts."""
import numpy as np
from scripts.analyze_finite_emission_transport import integrate,direct

def test_finite_kl_contrast_with_changed_metric_and_increment():
    rng=np.random.default_rng(290314)
    a=rng.normal(size=(9,13))*.4
    b=a+rng.normal(size=a.shape)*.3
    terms=integrate(a,b,32)
    np.testing.assert_allclose(terms[:,:2],np.stack([direct(a),direct(b)],-1),atol=2e-14,rtol=1e-12)
    np.testing.assert_allclose(terms[:,2:].sum(-1),direct(b)-direct(a),atol=2e-14,rtol=1e-12)
    assert np.max(np.abs(terms[:,4]))>1e-5

def test_time_dependent_constant_logit_gauges_leave_finite_terms_unchanged():
    rng=np.random.default_rng(310914);a=rng.normal(size=(7,11))*.3;b=a+rng.normal(size=a.shape)*.2
    shifted_a=a+rng.normal(size=(7,1));shifted_b=b+rng.normal(size=(7,1))
    np.testing.assert_allclose(integrate(shifted_a,shifted_b,32),integrate(a,b,32),atol=2e-14,rtol=2e-12)
