import numpy as np
import pytest
from model_rg.rg import (AffineJet, block_sum, fisher_pullback, gaussian_schur,
                         finite_susceptibility, tilted_mean)


def test_combined_block_semigroup():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(96,7))
    c = rng.normal(size=(7,4))
    d = rng.normal(size=(4,2))
    np.testing.assert_allclose(block_sum(block_sum(x,3)@c,4)@d,
                               block_sum(x,12)@(c@d),atol=2e-14)
    with pytest.raises(ValueError):
        block_sum(x,5)


def test_affine_noncommuting_source_transport():
    a = AffineJet(np.array([[1.,2.],[0.,1.]]),np.eye(2))
    b = AffineJet(np.array([[1.,0.],[3.,1.]]),2*np.eye(2))
    c = AffineJet(np.array([[2.,1.],[1.,0.]]),np.ones((2,2)))
    x,s = np.array([.3,-.7]),np.array([.2,.9])
    np.testing.assert_allclose(c.after(b).after(a).act(x,s),c.act(b.act(a.act(x,s),s),s))
    assert not np.allclose(a.after(b).A,b.after(a).A)


def test_schur_equals_marginal_covariance_and_nested_elimination():
    rng=np.random.default_rng(4)
    x=rng.normal(size=(9,9))
    q=x.T@x+np.eye(9)
    effective=gaussian_schur(q,[0,2,5])
    np.testing.assert_allclose(np.linalg.inv(effective),np.linalg.inv(q)[np.ix_([0,2,5],[0,2,5])],rtol=1e-12)
    np.testing.assert_allclose(gaussian_schur(gaussian_schur(q,[0,2,5,7]),[0,1,2]),effective,atol=1e-12)


def test_fisher_psd_gauge_and_kl_curvature():
    rng=np.random.default_rng(7)
    z=rng.normal(size=17);j=rng.normal(size=(17,5))
    h=fisher_pullback(z,j)
    assert np.linalg.eigvalsh(h).min()>0
    np.testing.assert_allclose(fisher_pullback(z+3,j+np.ones((17,1))@rng.normal(size=(1,5))),h,atol=1e-14)
    from scipy.special import log_softmax
    d=rng.normal(size=5);eps=1e-4
    p=np.exp(log_softmax(z))
    kl=p@(log_softmax(z)-log_softmax(z+eps*j@d))
    assert abs(kl/(eps**2/2*(d@h@d))-1)<1e-3


def test_tilt_derivative_is_covariance():
    rng=np.random.default_rng(8)
    x=rng.normal(size=(91,6));v=rng.normal(size=6);eps=1e-5
    deriv=(tilted_mean(x,eps*v)-tilted_mean(x,-eps*v))/(2*eps)
    np.testing.assert_allclose(deriv,np.cov(x,rowvar=False,ddof=0)@v,rtol=1e-8)


def test_correlations_are_not_discarded_by_blocking():
    rng=np.random.default_rng(40)
    x=np.repeat(rng.normal(size=(256,3)),8,axis=0)
    s1=finite_susceptibility(x,1);s8=finite_susceptibility(x,8)
    np.testing.assert_allclose(s8/s1,8*2047/(8*255),rtol=1e-12)
