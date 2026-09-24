"""Independent SVD tests of fixed-Fisher projection, including quotient gauges."""
import numpy as np
import pytest


@pytest.mark.parametrize('dependent',[False,True])
def test_fisher_projection_separates_representation_and_coefficients(dependent):
    rng=np.random.default_rng(91801)
    probabilities=rng.dirichlet(np.arange(1,8),size=19)
    dictionary=rng.normal(size=(19,7,4));targets=rng.normal(size=(19,7,2))
    if dependent:dictionary[:,:,3]=2*dictionary[:,:,0]-dictionary[:,:,1]
    def embed(x):
        centered=x-(probabilities[:,:,None]*x).sum(1,keepdims=True)
        return (centered*np.sqrt(probabilities[:,:,None]/19)).reshape(-1,x.shape[-1])
    d,y=embed(dictionary),embed(targets)
    # Singular values supply the projection independently of invertible-Gram formulas.
    u,s,v=np.linalg.svd(d,full_matrices=False);u=u[:,s>1e-12*s[0]]
    projected=u@(u.T@y);orthogonal=y-projected
    np.testing.assert_allclose(d.T@orthogonal,0,atol=2e-14)
    coefficients=rng.normal(size=(4,2))
    residual=y-d@coefficients;excess=projected-d@coefficients
    np.testing.assert_allclose(residual.T@residual,orthogonal.T@orthogonal+excess.T@excess,atol=2e-14)
    assert np.linalg.norm(residual)>=np.linalg.norm(orthogonal)
    # The same logit contrast modulo documentwise constants must give the same geometry.
    np.testing.assert_allclose(embed(targets+rng.normal(size=(19,1,2))),y,atol=2e-15)
    if not dependent:
        g=d.T@d;c=d.T@y
        np.testing.assert_allclose(orthogonal.T@orthogonal,y.T@y-c.T@np.linalg.solve(g,c),atol=2e-14)


def test_projection_hypotheses_are_necessary():
    # Dropping orthogonality removes the Pythagorean equality.
    y,q,z=np.array([1.]),np.array([2.]),np.array([0.])
    assert np.sum((y-z)**2)!=np.sum((y-q)**2)+np.sum((q-z)**2)
    # A negative weight turns a nonzero coefficient excess into a negative term.
    assert -1.*(0.-0.)**2 > -1.*(0.-1.)**2
