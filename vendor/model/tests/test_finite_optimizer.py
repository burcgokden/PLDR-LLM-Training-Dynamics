import numpy as np
import pytest
from model_rg.finite_optimizer import fixed_gradient_step, first_moment_displacement, pulse_parts


def test_moment_pulse_changes_future_weights_without_changing_incoming_weights():
    rng=np.random.default_rng(1729)
    theta,m,g=rng.normal(size=(3,127)); v=rng.uniform(.01,2,127)
    args=dict(beta1=.9,beta2=.95,counter=17,rate=.003,epsilon=1e-8,decay=.01)
    p0,m0,v0=fixed_gradient_step(theta,m,v,g,**args)
    delta=.1*m
    p1,m1,v1=fixed_gradient_step(theta,m+delta,v,g,**args)
    predicted=first_moment_displacement(delta,v0,**{k:x for k,x in args.items() if k!='decay'})
    np.testing.assert_allclose(p1-p0,predicted,rtol=1e-9,atol=5e-16)
    np.testing.assert_allclose(m1-m0,.9*delta,atol=5e-16)
    np.testing.assert_array_equal(v0,v1)
    assert np.linalg.norm(p1-p0)>0


def test_log_second_moment_preserves_support_and_has_finite_nonlinear_response():
    x=np.array([1.,2.,3.]); m=np.array([.2,0.,-.1]); v=np.array([.3,0.,.7]); g=np.array([-.5,0.,.2])
    args=dict(beta1=.8,beta2=.95,counter=4,rate=.02,epsilon=1e-8,decay=.03)
    zero=fixed_gradient_step(x,m,v,g,**args)
    plus=fixed_gradient_step(x,m,np.exp(.2)*v,g,**args)
    minus=fixed_gradient_step(x,m,np.exp(-.2)*v,g,**args)
    even,odd=pulse_parts(zero[0],plus[0],minus[0])
    np.testing.assert_allclose(zero[0]+even+odd,plus[0])
    np.testing.assert_array_equal(plus[2]==0,v==0)
    assert np.linalg.norm(even)>1e-7 and np.linalg.norm(odd)>1e-7
    with pytest.raises(ValueError): fixed_gradient_step(x,m,-v,g,**args)


def test_linear_observation_preserves_all_pulse_covariance_sectors():
    rng=np.random.default_rng(314159)
    x=rng.normal(size=(13,5)); even=.4*x+rng.normal(size=x.shape)
    odd=-.3*x+.7*even+rng.normal(size=x.shape); b=rng.normal(size=(2,5))
    fields=[(a-a.mean(0))@b.T for a in [x,even,odd]]
    for sign in [-1,1]:
        a=[fields[0],fields[1],sign*fields[2]]
        direct=np.cov((x+even+sign*odd)@b.T,rowvar=False)
        expanded=sum(u.T@v/12 for u in a for v in a)
        np.testing.assert_allclose(direct,expanded,rtol=1e-13,atol=1e-13)
        diagonal=sum(u.T@u/12 for u in a)
        assert np.linalg.norm(direct-diagonal)>1
