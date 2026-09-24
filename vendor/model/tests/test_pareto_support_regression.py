"""Compare the selected support with an independent inclusive-support reducer."""
import numpy as np
from model_rg.potential_avalanches import pareto_fit, tail_diagnostics

def independent_fit(x, xmin):
    tail = np.sort(x[np.isfinite(x) & (x >= xmin)])
    alpha = 1 + len(tail) / np.log(tail / xmin).sum()
    cdf = 1 - (tail / xmin) ** (1 - alpha)
    ranks = np.arange(len(tail)) / len(tail)
    ks = max(np.max(ranks + 1 / len(tail) - cdf), np.max(cdf - ranks))
    return len(tail), alpha, ks

def test_reported_cutoff_includes_all_ties():
    x = np.round(1 + np.random.default_rng(9131401).pareto(1.5, 180), 1)
    fit = tail_diagnostics(x, np.random.default_rng(222), bootstraps=19)
    count, alpha, ks = independent_fit(x, fit['xmin'])
    assert fit['n_tail'] == count
    np.testing.assert_allclose([fit['alpha'], fit['ks']], [alpha, ks], rtol=1e-12)

def test_capped_search_preserves_tied_support_and_scale():
    rng = np.random.default_rng(9131402)
    x = np.repeat(np.geomspace(1., 1000., 240), rng.integers(1, 6, 240))
    fit = pareto_fit(x)
    count, alpha, ks = independent_fit(x, fit['xmin'])
    assert fit['n_tail'] == count
    np.testing.assert_allclose([fit['alpha'], fit['ks']], [alpha, ks], rtol=1e-12)
    scaled = pareto_fit(8 * x)
    np.testing.assert_allclose([scaled['xmin'], scaled['alpha'], scaled['ks']],
                               [8 * fit['xmin'], fit['alpha'], fit['ks']], rtol=1e-12)

def test_unrounded_data_matches_independent_complete_search():
    x = 1 + np.random.default_rng(9131403).pareto(1.5, 120)
    expected = []
    for xmin in np.unique(x):
        if np.sum(x >= xmin) >= 50:
            count, alpha, ks = independent_fit(x, xmin)
            expected.append((ks, xmin, count, alpha))
    ks, xmin, count, alpha = min(expected)
    fit = pareto_fit(x)
    assert fit['xmin'] == xmin and fit['n_tail'] == count
    np.testing.assert_allclose([fit['alpha'], fit['ks']], [alpha, ks], rtol=1e-12)
