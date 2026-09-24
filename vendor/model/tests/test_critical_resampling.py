import numpy as np
import pytest
from model_rg.critical_resampling import replica_distances, replica_counts, resampled_variance, vector_statistics


def test_gram_resampling_matches_full_realization_resampling():
    rng = np.random.default_rng(40)
    x = rng.normal(size=(7, 5, 2, 8))
    counts = replica_counts(7, 50)
    actual = resampled_variance(replica_distances(x), counts)
    expected = []
    for row in counts:
        sample = x[np.repeat(np.arange(7), row)]
        expected.append(np.var(sample, axis=0, ddof=1).mean())
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


def test_joint_resampling_retains_a_paired_scaling_identity():
    x = np.random.default_rng(1).normal(size=(6, 4, 2))
    counts = replica_counts(6, 70)
    base = resampled_variance(replica_distances(x), counts)
    shifted = resampled_variance(replica_distances(3*x+1000), counts)
    np.testing.assert_allclose(shifted, 9*base, rtol=3e-13, atol=3e-13)


def test_rotation_changes_no_vector_susceptibility_or_radial_split():
    rng = np.random.default_rng(21)
    x = rng.normal(size=(6, 4, 2, 3, 8))+2
    rotation = np.linalg.qr(rng.normal(size=(8, 8)))[0]
    a = vector_statistics(x)
    b = vector_statistics(x@rotation)
    for key in ['susceptibility', 'one_head_variance', 'parallel_susceptibility', 'perpendicular_susceptibility']:
        assert a[key] == pytest.approx(b[key])
    assert a['parallel_susceptibility']+a['perpendicular_susceptibility'] == pytest.approx(a['susceptibility'])


def test_complete_vectors_resolve_fluctuations_hidden_by_scalar_amplitude():
    x = np.zeros((4, 1, 1, 3, 2))
    x[:, 0, 0, :, :] = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])[:, None, :]
    assert np.var(np.linalg.norm(x, axis=-1)) == 0
    stat = vector_statistics(x)
    assert stat['susceptibility'] > 0
    assert stat['susceptibility'] == pytest.approx(3*stat['one_head_variance'])


def test_binary_onset_mixture_can_produce_extensive_susceptibility():
    from model_rg.critical_resampling import binary_variance_decomposition
    labels = np.array([0, 0, 0, 1, 1, 1])
    x = labels[:, None]*np.array([[1., 2., 3.]])
    first = binary_variance_decomposition(x, labels, 2)
    second = binary_variance_decomposition(x, labels, 10)
    assert first['within_susceptibility'] == 0
    assert first['between_fraction'] == pytest.approx(1.)
    assert second['susceptibility'] == pytest.approx(5*first['susceptibility'])


def test_empty_phase_does_not_remove_within_phase_fluctuations():
    from model_rg.critical_resampling import binary_variance_decomposition
    x = np.random.default_rng(7).normal(size=(5, 4, 2))
    result = binary_variance_decomposition(x, np.zeros(5), 4)
    assert result['between_susceptibility'] == 0
    assert result['within_susceptibility'] == pytest.approx(4*np.var(x, axis=0, ddof=1).mean())


def test_degenerate_exact_mass_and_direct_difference_reference():
    from model_rg.critical_resampling import exact_replica_counts
    x=2**40+np.arange(66).reshape(6,11)/8
    counts,mass=exact_replica_counts(6)
    assert len(counts)==462 and mass.sum()==6**6
    d=replica_distances(x)
    assert np.array_equal(d,replica_distances(x-2**40))
    values=resampled_variance(d,counts)
    assert mass[values==0].sum()==6
    assert np.array_equal(resampled_variance(replica_distances(np.ones((6,11))),counts),np.zeros(len(counts)))
    for count,value in zip(counts,values):
        sample=np.repeat(x-2**40,count,axis=0)
        assert value==pytest.approx(sample.var(axis=0,ddof=1).mean(),rel=1e-13,abs=1e-13)


def test_count_validation_and_zero_peak():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from analyze_critical_joint import peak_draws
    d=replica_distances(np.arange(6).reshape(3,2))
    for c in [np.ones((1,3)),np.array([[1,-1,3]]),np.array([[1,1,2]])]:
        with pytest.raises(ValueError):resampled_variance(d,c)
    p=peak_draws([0,1,2],np.zeros((1,3)))
    assert not p['interior'][0] and np.isnan(p['width'][0])


def test_predictive_trace_has_no_vocabulary_divisor():
    from scipy.special import softmax
    x=np.random.default_rng(2601).normal(size=(6,8,17))
    embedding=2*np.sqrt(softmax(x,axis=-1))
    d=replica_distances(embedding)*embedding.shape[-1]
    value=resampled_variance(d,np.ones((1,6),dtype=np.int64))[0]
    expected=np.var(embedding,axis=0,ddof=1).sum(-1).mean()
    assert value==pytest.approx(expected,rel=1e-14)
