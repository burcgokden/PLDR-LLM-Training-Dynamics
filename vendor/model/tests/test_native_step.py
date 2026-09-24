"""A nonlinear source example in which signed covariance cross terms matter."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_scaling_native_step import decompose


def test_coupled_native_corner_statistics_keep_cancellation_and_mixed_term():
    # F(g,b)=g+b+g*b at (0,0), with perfectly anticorrelated displacements.
    g = np.array([-2., -1., 1., 2.])[:, None]
    b = -g
    full = g+b+g*b
    result = decompose(np.zeros(1), g, b, full)
    expected_covariance = np.array([[10/3, -10/3, 0.], [-10/3, 10/3, 0.], [0., 0., 3.]])
    np.testing.assert_allclose(result['component_covariance'], expected_covariance, atol=1e-14)
    np.testing.assert_allclose(result['full_covariance_trace'], 3.)
    np.testing.assert_allclose(result['full_mean_squared_norm'], 6.25)
    np.testing.assert_allclose(result['component_cohort_mean_increments'], [0., 0., -2.5])
    np.testing.assert_allclose(result['component_cohort_mean_covariance'], expected_covariance, atol=1e-14)
    np.testing.assert_allclose(result['full_second_moment'], 8.5)
    np.testing.assert_allclose(result['mixed_relative_second_moment'], 1.)
    np.testing.assert_allclose(result['generator_only_relative_covariance_error'], 19/9)
