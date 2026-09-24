import numpy as np
import pytest
from model_rg.potential_symmetry import potential_allocations


def test_endpoint_reversal_and_blocked_signed_energy():
    rng=np.random.default_rng(9132201)
    b=rng.normal(size=(17,23));p=rng.normal(size=(17,23))
    a=potential_allocations(b,p);r=potential_allocations(b[::-1],p[::-1])
    u,v=a['symmetric']
    np.testing.assert_allclose(u+v,np.diff(p*b,axis=0),atol=2e-15)
    np.testing.assert_allclose(r['symmetric'][0],-u[::-1],atol=2e-15)
    np.testing.assert_allclose(r['symmetric'][1],-v[::-1],atol=2e-15)
    np.testing.assert_allclose(u-a['forward'][0],a['corner']/2,atol=2e-15)
    w=rng.dirichlet(np.ones(23));d=a['increment']
    blocked=(p[-1]*b[-1]-p[0]*b[0])
    np.testing.assert_allclose(np.sum(w*blocked**2),np.einsum('i,ti,si->',w,d,d),rtol=2e-14)
    assert not np.isclose(np.sum(w*blocked**2),np.sum(w*d**2))


def test_unaligned_observations_are_rejected():
    with pytest.raises(ValueError):potential_allocations(np.ones((3,4)),np.ones((2,4)))
    with pytest.raises(ValueError):potential_allocations(np.full((3,4),np.nan),np.ones((3,4)))
