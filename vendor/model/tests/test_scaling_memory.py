"""Independent covariance and passive-observation checks for the scale limits."""
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_scaling_collectives import arithmetic_comparison


def test_correlated_feedback_variance_from_augmented_lyapunov_equation():
    for a in [.8,.2,.01,.0001]:
        for rho in [0.,.5,.95,.999]:
            sigma=.07
            transition=np.array([[1-a,sigma],[0,rho]])
            covariance=solve_discrete_lyapunov(transition,np.diag([0.,1-rho*rho]))
            predicted=sigma*sigma/(a*(2-a))*(1+(1-a)*rho)/(1-(1-a)*rho)
            np.testing.assert_allclose(covariance[0,0],predicted,rtol=2e-11,atol=1e-12)
            np.testing.assert_allclose(covariance[1,1],1,rtol=1e-11)


def test_paired_arithmetic_bound_retains_seed_and_coordinate_conditioning():
    rng=np.random.default_rng(650211)
    reference=rng.normal(size=(4,17,3))*np.array([1.,1e-5,1e-10])
    # A common offset must cancel in the conditional seed variance difference.
    error=.001*reference+rng.normal(size=(4,17,3))*1e-6+7.
    report=arithmetic_comparison({'q':reference+error},{'q':reference},14)['q']
    assert report['rms_field_difference']>6.9
    assert report['centered_error_susceptibility']<1e-4
    assert report['absolute_difference']<=report['susceptibility_error_bound']*(1+1e-12)
