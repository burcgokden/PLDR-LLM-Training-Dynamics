"""Independent duality and held-out covariance checks for predictive source reduction."""
import numpy as np

from model_rg.visible_noise import fisher_image, fisher_adjoint, visible_basis, projection_statistics


def test_fisher_duality_includes_context_weights_and_softmax_gauge():
    rng = np.random.default_rng(650331)
    p = rng.random((3, 11));p /= p.sum(-1, keepdims=True)
    v, w = rng.normal(size=(2, 3, 11))
    np.testing.assert_allclose(np.sum(fisher_image(v, p)*w), np.sum(v*fisher_adjoint(w, p)), rtol=1e-13)
    np.testing.assert_allclose(fisher_adjoint(w, p).sum(-1), 0, atol=2e-16)
    np.testing.assert_allclose(fisher_image(v+23., p), fisher_image(v, p), atol=2e-15)


def test_frozen_visible_subspace_retains_its_rank_and_rejects_unseen_noise():
    training = np.array([[3., 0., 0.], [0., 2., 0.], [-3., 0., 0.], [0., -2., 0.]])
    basis, _ = visible_basis(training)
    assert len(basis) == 2
    held = np.array([[1., 2., 3.], [-1., 2., -3.], [2., -2., 3.], [-2., -2., -3.]])
    report = projection_statistics(held, held@basis.T, basis, 2)
    np.testing.assert_allclose(report['residual_covariance_trace'], 12.)
    np.testing.assert_allclose(report['covariance_trace'], report['predicted_covariance_trace']+12.)
    np.testing.assert_allclose(report['residual_second_moment'], 9.)
