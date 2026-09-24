import numpy as np
import pytest

from model_rg.finetuning import mixture_stream, paired_order, quadratic_mixture, covariance_split


def test_mixture_distinctness_and_endpoints():
    a=np.arange(1000);b=np.arange(1000,2000);u=np.random.default_rng(4).uniform(size=(8,32))
    np.testing.assert_array_equal(mixture_stream(a,b,u,1),a[:256].reshape(8,32))
    np.testing.assert_array_equal(mixture_stream(a,b,u,0),b[:256].reshape(8,32))
    for rho in [.125,.25,.5,.75,.875]:
        out=mixture_stream(a,b,u,rho)
        assert len(np.unique(out))==256
        np.testing.assert_array_equal(out<1000,u<rho)


def test_mixture_rejects_reuse_and_exhaustion():
    u=np.full((2,32),.5)
    with pytest.raises(ValueError):mixture_stream(np.arange(32),np.arange(64),u,1)
    u[:,::2]=.1
    with pytest.raises(ValueError):mixture_stream(np.arange(64),np.arange(64),u,.5)


def test_pair_normalization_keeps_zero_mean_undefined():
    result=paired_order(np.array([4.,4.]),np.array([0.,2.]),np.array([8.,8.]),2)
    np.testing.assert_array_equal(result['mean_defined'],[False,True])
    np.testing.assert_allclose(result['rms_ratio'],np.sqrt(.5))
    assert result['mean_ratio'][1]==np.sqrt(2)


def test_paired_discrepancy_depends_on_coupling():
    x=np.array([-1.,1.]);independent=np.mean((x[:,None]-x[None,:])**2)
    assert independent==2*np.var(x)
    assert np.mean((x-x)**2)==0
    assert np.mean((x+x)**2)==4*np.var(x)


def test_quadratic_interaction_predicts_unseen_mixtures():
    rng=np.random.default_rng(2);a,b,c=rng.normal(size=(3,11))
    law=lambda r:a+r*b+r*r*c
    for rho in [.125,.25,.75,.875]:
        np.testing.assert_allclose(quadratic_mixture(law(0),law(.5),law(1),rho),law(rho),atol=1e-14)


def test_total_covariance_keeps_shared_collective():
    x=np.random.default_rng(8).normal(size=(4,9,3))+np.arange(4)[:,None,None]
    within,between=covariance_split(x)
    flat=x.reshape(-1,3);center=flat-flat.mean(0)
    np.testing.assert_allclose(within+between,center.T@center/len(center),atol=1e-13)


def test_finite_mixture_law_score_identity():
    # Exhaust a native-independent finite three-choice nonlinear response law.
    histories=np.array([[i>>k&1 for k in range(3)] for i in range(8)])
    k=histories.sum(1);value=(histories[:,0]+2*histories[:,1]-histories[:,2])**2
    r=.37;p=r**k*(1-r)**(3-k)
    derivative=np.sum(p*value*(k-3*r)/(r*(1-r)))
    def mean(q):return np.sum(q**k*(1-q)**(3-k)*value)
    np.testing.assert_allclose(derivative,(mean(r+1e-5)-mean(r-1e-5))/2e-5,rtol=1e-9)


def test_two_visible_modes_need_more_than_a_scalar_projection():
    a=np.array([1.,1.,0.]);b=np.array([1.,-1.,0.])
    gram=np.array([[a@a,a@b],[b@a,b@b]])
    assert np.linalg.det(gram)>0
    assert a[0]==b[0]


def test_covariance_eigenvalue_transport_bounds():
    rng=np.random.default_rng(24);x=rng.normal(size=(2000,3))*np.array([1,.1,.01])
    x-=x.mean(0);a=rng.normal(size=(5,3));e=rng.normal(size=(2000,5))*1e-4;e-=e.mean(0)
    y=x@a.T+e
    sx=np.linalg.svd(x/np.sqrt(len(x)),compute_uv=False)
    sy=np.linalg.svd(y/np.sqrt(len(y)),compute_uv=False)[:3]
    sa=np.linalg.svd(a,compute_uv=False);error=np.sqrt(np.mean(np.sum(e*e,axis=1)))
    assert np.all(sy>=sa[-1]*sx-error)
    assert np.all(sy<=sa[0]*sx+error)
