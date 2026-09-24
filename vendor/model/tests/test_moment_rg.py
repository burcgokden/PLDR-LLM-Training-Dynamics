import numpy as np
import pytest

from model_rg.moment_rg import block_maps, forecast, moments, score, transport


def test_transport_matches_observations_and_composes():
    rng = np.random.default_rng(9923)
    x = rng.normal(size=(23, 8))
    x[:, 7] += 3*x[:, 0]
    mean, cov = moments(x)
    maps = block_maps(2)
    pair_mean, pair_cov = transport(mean, cov, maps['paired'])
    actual_mean, actual_cov = moments(x @ maps['paired'].T)
    np.testing.assert_allclose(pair_mean, actual_mean, atol=1e-14)
    np.testing.assert_allclose(pair_cov, actual_cov, atol=1e-14)
    last = np.kron(np.eye(2), np.ones((1, 2)))
    for a, b in zip(transport(pair_mean, pair_cov, last), transport(mean, cov, maps['endpoint'])):
        np.testing.assert_allclose(a, b, atol=1e-14)


def test_score_regret_for_non_gaussian_finite_law():
    x = np.array([[-2., 0.], [1., 3.], [0., -1.], [4., 2.]])
    mu = x.mean(0); centered = x-mu
    cov = centered.T @ centered/len(x)  # exact population of four equally weighted atoms
    proposed_mean = np.array([1., 2.]); proposed_cov = np.array([[3., .4], [.4, 2.]])
    measured = np.mean(score(x, proposed_mean, proposed_cov)-score(x, mu, cov))
    delta = mu-proposed_mean
    expected = (np.linalg.slogdet(proposed_cov)[1]-np.linalg.slogdet(cov)[1]
                + np.trace(np.linalg.solve(proposed_cov, cov))-2
                + delta @ np.linalg.solve(proposed_cov, delta))
    assert measured > 0
    np.testing.assert_allclose(measured, expected)


def test_common_origin_and_variable_origin_have_different_estimands():
    increments = np.array([[1., 2., 3., 4.], [-1., 4., 2., 3.], [2., -3., 0., 1.]])
    _, cov = moments(increments)
    endpoints = increments.sum(1)+7
    np.testing.assert_allclose(cov.sum(), endpoints.var(ddof=1))
    assert not np.isclose(cov.trace(), endpoints.var(ddof=1))
    assert not np.isclose(cov.sum(), (endpoints+np.arange(3)).var(ddof=1))


def test_regularization_is_positive_definite_even_for_singular_samples():
    x = np.zeros((4, 8)); x[:, :4] = np.arange(4)[:, None]
    fit = forecast(x)
    scales = np.array(fit['scale'])
    for key in ['regularized', 'diagonal']:
        covariance = np.array(fit[key])*np.outer(scales, scales)
        assert np.linalg.eigvalsh(covariance).min() > 0
        assert np.isfinite(score(x, np.array(fit['mean']), covariance)).all()


def test_scores_reject_indefinite_covariance_and_short_samples():
    with pytest.raises(np.linalg.LinAlgError):
        score(np.ones((3, 2)), np.zeros(2), np.diag([1., -1.]))
    with pytest.raises(ValueError):
        moments(np.zeros((1, 2)))


def test_relative_precision_bound_after_rectangular_blocking():
    rng = np.random.default_rng(37481)
    for _ in range(40):
        d = 8
        a = rng.normal(size=(d, d))
        true_cov = a @ a.T + np.eye(d)
        values, vectors = np.linalg.eigh(true_cov)
        root = (vectors*np.sqrt(values)) @ vectors.T
        q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        epsilon = .4
        error = (q*rng.uniform(-epsilon, epsilon, d)) @ q.T
        predicted = root @ (np.eye(d)+error) @ root
        v = rng.normal(size=d); v = .3*v/np.linalg.norm(v)
        delta = root @ v
        for block in block_maps(2).values():
            truth = block @ true_cov @ block.T
            guess = block @ predicted @ block.T
            offset = block @ delta
            dimension = len(block)
            regret = (np.linalg.slogdet(guess)[1]-np.linalg.slogdet(truth)[1]
                      +np.trace(np.linalg.solve(guess, truth))-dimension
                      +offset @ np.linalg.solve(guess, offset))
            upper = dimension*epsilon**2/(2*(1-epsilon)**2)+.3**2/(1-epsilon)
            assert -1e-12 <= regret <= upper
