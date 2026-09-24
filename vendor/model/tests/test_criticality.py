import numpy as np
import scipy.linalg as la
import torch
from model_rg.criticality import conditional_covariances, predictive_kl, sample_batches


def test_conditional_covariance_separates_independent_and_shared_heads():
    orthogonal = la.hadamard(8).astype(float)
    independent = orthogonal[:, 1:5][:, None, None, :]
    chi, diagonal, context = conditional_covariances(independent)
    np.testing.assert_allclose(chi, diagonal, atol=1e-14)
    np.testing.assert_array_equal(context, np.zeros((1, 1)))
    shared = np.repeat(orthogonal[:, 1, None, None, None], 4, axis=-1)
    chi, diagonal, _ = conditional_covariances(shared)
    np.testing.assert_allclose(chi, 4 * diagonal, atol=1e-14)
    # Changing only the document mean cannot create conditional seed variance.
    two_contexts = np.repeat(independent, 2, axis=1)
    two_contexts[:, 1] += 7
    after, _, context = conditional_covariances(two_contexts)
    np.testing.assert_allclose(after, conditional_covariances(independent)[0], atol=1e-14)
    assert context[0, 0] > 0


def test_sampling_streams_have_identical_prefixes_across_horizons():
    tokens = np.broadcast_to(np.arange(513), (3072, 513))
    short = sample_batches(tokens, 640101, 19)
    long = sample_batches(tokens, 640101, 47)
    for first, second in zip(short, long):
        np.testing.assert_array_equal(first, second[:19])
    np.testing.assert_array_equal(short[2][:, :, 64], short[1] + 64)


def test_predictive_kl_resolves_small_and_extreme_logit_changes():
    first = torch.tensor([[.2, -.3, .7], [1000., -1000., 0.]], dtype=torch.float64)
    second = first + torch.tensor([[1e-8, -2e-8, 3e-8], [-2000., 2000., 0.]], dtype=torch.float64)
    result = predictive_kl(first, second)
    assert torch.isfinite(result).all()
    assert 0 < result[0] < 1e-14
    assert abs(float(result[1]) - 2000.) < 1e-10
    torch.testing.assert_close(predictive_kl(first + 51, second + 51), result, rtol=1e-6, atol=1e-20)


def test_multilayer_covariance_is_invariant_under_independent_head_relabeling():
    rng = np.random.default_rng(761)
    fields = rng.normal(size=(6, 7, 3, 5))
    fields[:, :, 2] += 2 * fields[:, :, 0].mean(-1, keepdims=True)
    before = conditional_covariances(fields)
    changed = fields.copy()
    for layer in range(3):
        changed[:, :, layer, :] = fields[:, :, layer, :][..., rng.permutation(5)]
    after = conditional_covariances(changed)
    for first, second in zip(before, after):
        np.testing.assert_allclose(first, second, rtol=1e-12, atol=1e-12)
    chi, reference, _ = before
    assert la.eigvalsh(5 * reference - chi).min() >= -1e-12


def test_width_prediction_is_stable_in_tiny_covariance_units():
    from scripts.analyze_criticality import width_fits
    sizes = np.array([2., 4., 8., 14.])
    for scale in [1., 1e-24]:
        result = width_fits(sizes, scale * (2 + 6 / sizes))['regular_inverse_width']
        np.testing.assert_allclose(np.array(result['parameters']) / scale, [2., 6.], rtol=1e-6, atol=1e-6)
        assert result['relative_rmse'] < 1e-7
    zero = width_fits(sizes, np.zeros(4))
    assert zero['power']['parameters'][0] == 0
    assert not zero['power']['exponent_identified']


def test_initialization_roles_survive_width_vocabulary_dimension_collision():
    from model_rg.variance_family import reference_dimensions, linear_factor
    width, ffn, vocabulary = 32000, (32000 * 8) // 3, 32000
    readout = reference_dimensions('final_layer', width, vocabulary, width, ffn, vocabulary)
    attention = reference_dimensions('decoder.dec_layers.0.mha1.wq', width, width, width, ffn, vocabulary)
    readout_variance = 2 / (width + vocabulary) * linear_factor(width, vocabulary, *readout)**2
    attention_variance = 2 / (2 * width) * linear_factor(width, width, *attention)**2
    np.testing.assert_allclose(readout_variance * width, 256 / 32128, rtol=1e-14)
    np.testing.assert_allclose(attention_variance * width, 1., rtol=1e-14)
    for heads in [2, 4, 8, 14, 24]:
        width = 64 * heads
        ffn = (width * 8) // 3
        previous = {width: 128, ffn: 341, vocabulary: vocabulary}
        for name, inputs, outputs in [('final_layer', width, vocabulary), ('query', width, width),
                                      ('gate', width, ffn), ('project', ffn, width)]:
            actual = reference_dimensions(name, inputs, outputs, width, ffn, vocabulary)
            assert linear_factor(inputs, outputs, *actual) == linear_factor(inputs, outputs, previous[inputs], previous[outputs])
