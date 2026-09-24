import numpy as np
from model_rg.training import mixture_covariance, shared_checkpoint_susceptibility


def test_shared_checkpoint_retains_between_state_cross_covariance():
    # Four equiprobable within-state offsets have covariance diag(2, 8).
    offsets=np.array([[-2.,0.],[2.,0.],[0.,-4.],[0.,4.]])
    means=np.array([[-3.,-6.],[3.,6.]])
    fields=means[:,None,:]+offsets[None,:,:]
    a,b,total=mixture_covariance(fields)
    np.testing.assert_allclose(a,np.diag([2.,8.]))
    np.testing.assert_allclose(b,np.array([[9.,18.],[18.,36.]]))
    # Enumerate every pair of independent documents sharing one state.
    blocks=np.stack([(fields[:,i]+fields[:,j])/np.sqrt(2)
                     for i in range(4) for j in range(4)],axis=1).reshape(-1,2)
    np.testing.assert_allclose(np.cov(blocks,rowvar=False,bias=True),
                               shared_checkpoint_susceptibility(a,b,2))
    np.testing.assert_allclose(total,a+b)


def test_temporal_projection_requires_hidden_state_closure():
    p=np.array([[1.,0.],[1.,0.],[0.,1.],[0.,1.]])
    k=np.array([[.4,.4,.1,.1],[.6,.2,.15,.05],[.2,.1,.4,.3],[.1,.2,.2,.5]])
    q=np.array([[.8,.2],[.3,.7]])
    np.testing.assert_allclose(k@p,p@q)
    np.testing.assert_allclose(np.linalg.matrix_power(k,7)@p,p@np.linalg.matrix_power(q,7))
    # Same retained starting state, different hidden optimizer states.
    bad=k.copy();bad[1]=[.1,.1,.4,.4]
    assert not np.allclose((bad@p)[0],(bad@p)[1])
