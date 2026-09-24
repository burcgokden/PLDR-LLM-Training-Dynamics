import numpy as np

from model_rg.source_basis import covariance_rotation
from model_rg.visible_noise import visible_basis


def test_covariance_priority_preserves_noise_when_total_priority_tracks_drift():
    # Large constant drift is orthogonal to a two-dimensional innovation.
    # The response and covariance optimal first coordinates must differ.
    noise = np.array([[1,1],[1,-1],[-1,1],[-1,-1]], dtype=float)
    y = np.column_stack([np.full(4,100.),3*noise[:,0],noise[:,1],np.zeros(4)])
    total, _ = visible_basis(y)
    rotation, eigenvalues = covariance_rotation(y,total)
    covariance_basis = rotation@total
    np.testing.assert_allclose(abs(total[0]),[1,0,0,0],atol=1e-12)
    np.testing.assert_allclose(abs(covariance_basis[0]),[0,1,0,0],atol=1e-12)
    np.testing.assert_allclose(eigenvalues,[12.,4/3,0],atol=1e-12)
    np.testing.assert_allclose(covariance_basis@covariance_basis.T,np.eye(3),atol=1e-12)
    # The entire span remains unchanged, including its separate drift sector.
    np.testing.assert_allclose(y@covariance_basis.T@covariance_basis,y,atol=1e-12)
    centered = y-y.mean(0)
    retained = (centered@covariance_basis[:1].T)@covariance_basis[:1]
    np.testing.assert_allclose(np.sum((centered-retained)**2)/np.sum(centered**2),.1,atol=1e-12)
