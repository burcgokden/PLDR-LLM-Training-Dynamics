"""Finite resource covariance including signed, noncommuting transport."""
import itertools
import numpy as np
from scripts.check_consuming_bridge import formula


def test_signed_finite_population_transport():
    u=np.array([[1.,0.],[-1.,1.],[2.,-1.],[-2.,0.]])
    D=np.array([[[1.,2.],[0.,1.]],[[0.,-1.],[3.,0.]]])
    z=np.stack([sum(D[k]@u[list(p[2*k:2*k+2])].mean(0) for k in range(2)) for p in itertools.permutations(range(4))])
    actual=z.T@z/24
    np.testing.assert_allclose(formula(u,D,2),actual,rtol=0,atol=1e-14)
    iid=sum(d@(u.T@u/4)@d.T for d in D)/2
    assert np.linalg.norm(iid-actual)>1


def test_full_consumption_cancels_population_mean():
    u=np.arange(8.)[:,None]-3.5
    np.testing.assert_array_equal(formula(u,np.ones((4,1,1)),2),np.zeros((1,1)))
