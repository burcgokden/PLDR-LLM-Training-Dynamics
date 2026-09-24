import numpy as np
import pytest
from scripts.analyze_critical_joint import crossing_draws, peak_draws
from scripts.analyze_critical_scaling import width_models
from model_rg.critical_onepass import peak_profile
from model_rg.critical_resampling import binary_variance_decomposition


def test_vectorized_peaks_retain_unresolved_boundaries():
    controls = [0, 1, 2, 3, 4]
    rows = np.array([[0, 1, 2, 1, 0], [0, 1, 2, 3, 4], [2, 0, 2, 0, 1], [0, 0, 0, 0, 0]], dtype=float)
    result = peak_draws(controls, rows)
    for k, row in enumerate(rows):
        expected = peak_profile(controls, row)
        assert result['controls'][k] == expected['sampled_maximum_control']
        assert result['interior'][k] == expected['interior']
        if expected['half_width'] is None:
            assert np.isnan(result['width'][k])
        else:
            assert result['width'][k] == pytest.approx(expected['half_width'])


def test_zero_bracket_and_multiple_crossings_are_explicit():
    rows = np.array([[1, .2, .1, 0], [1, .9, .1, 0], [1, 0, 1, 0]], dtype=float)
    values, interior, count = crossing_draws([0, 1, 2, 3], rows, .5)
    np.testing.assert_array_equal(count, [1, 1, 2])
    np.testing.assert_array_equal(interior, [False, True, False])
    assert np.isnan(values[2])
    assert values[1] == pytest.approx(1.5)


def test_regular_variance_null_can_increase_with_width():
    widths = np.array([2, 4, 8, 14], dtype=float)
    law = 1-1/widths
    result = width_models(widths, law)
    np.testing.assert_allclose(result['regular']['fitted'], law, atol=2e-14)
    assert result['regular']['withheld_largest_prediction'] == pytest.approx(law[-1])
    shared = width_models(widths, .3+.2*widths)
    np.testing.assert_allclose(shared['shared_mode']['coefficients'], [.3, .2], atol=2e-14)


def test_timing_mixture_uses_whole_realizations_and_one_divisor():
    x = np.random.default_rng(79).normal(size=(7, 4, 3))
    labels = np.array([0, 1, 0, 0, 1, 1, 1])
    stat = binary_variance_decomposition(x, labels, 8)
    assert stat['within_susceptibility']+stat['between_susceptibility'] == pytest.approx(8*np.var(x, axis=0, ddof=1).mean())
    assert stat['decomposition_residual'] == pytest.approx(0., abs=1e-14)


def test_shared_variance_reference_allows_a_negative_finite_correction():
    widths = np.array([2, 4, 8, 14], dtype=float)
    law = .2*widths-.3
    result = width_models(widths, law)['shared_with_signed_correction']
    np.testing.assert_allclose(result['fitted'], law, atol=2e-14)
    assert result['asymptotic_intensive_variance'] == pytest.approx(.2)
    assert result['susceptibility_intercept'] == pytest.approx(-.3)
    assert result['withheld_largest_prediction'] == pytest.approx(law[-1])
