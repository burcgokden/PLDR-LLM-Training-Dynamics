import numpy as np

from analysis.radial_tangential import (
    closing_masks,
    face_affine_coefficients,
    face_affine_endpoint,
    paired_decomposition,
    radial_decomposition,
    source_resolved_decomposition,
)


def test_zero_energy_is_explicitly_ineligible_and_face_reopening_is_exact():
    result = radial_decomposition([[0.0, 0.0]], [[1.0, -2.0]])
    assert not result["eligible"][0]
    assert np.isnan(result["alpha"][0])
    cocycle = face_affine_coefficients([0.0, 5.0, 10.0, 0.0, 3.0])
    np.testing.assert_array_equal(cocycle["step_residual"], 0.0)
    assert face_affine_endpoint(
        0.0, cocycle["gain_squared"], cocycle["face_reopening"]
    ) == 3.0


def test_pure_radial_motion_has_zero_tangent_and_exact_gain():
    source = np.array([[1.0, 2.0]])
    increment = -0.25 * source
    result = radial_decomposition(source, increment)
    np.testing.assert_allclose(result["alpha"], [0.25])
    np.testing.assert_allclose(result["tangent"], 0.0, atol=1.0e-16)
    np.testing.assert_allclose(result["gain_squared"], [0.75**2])
    assert closing_masks(result)["contract"][0]


def test_pure_tangent_motion_reopens_and_is_orthogonal():
    result = radial_decomposition([[1.0, 0.0]], [[0.0, 0.5]])
    np.testing.assert_allclose(result["alpha"], [0.0])
    np.testing.assert_allclose(result["tau_squared"], [0.25])
    np.testing.assert_allclose(result["orthogonality_residual"], [0.0])
    masks = closing_masks(result)
    assert masks["reopen"][0]
    assert masks["radial_failure"][0]


def test_radial_overshoot_and_tangent_excess_are_distinct_failures():
    source = np.array([[1.0, 0.0], [1.0, 0.0]])
    increment = np.array([[-2.5, 0.0], [-0.25, 1.0]])
    masks = closing_masks(radial_decomposition(source, increment))
    np.testing.assert_array_equal(masks["radial_failure"], [True, False])
    np.testing.assert_array_equal(masks["tangent_excess"], [False, True])


def test_source_radial_cancellation_and_tangent_gram_recompose():
    source = np.array([[1.0, 0.0]])
    components = np.array([[[1.0, 1.0], [-1.0, 2.0], [0.0, -1.0], [0.0, 0.0]]])
    resolved = source_resolved_decomposition(source, components)
    np.testing.assert_allclose(resolved["alpha"], [[-1.0, 1.0, 0.0, 0.0]])
    np.testing.assert_allclose(resolved["total_alpha"], [0.0])
    np.testing.assert_allclose(
        resolved["total_tangent"], np.sum(resolved["tangents"], axis=1)
    )
    gram = resolved["tangent_gram_normalized"][0]
    tangent_energy = np.sum(resolved["total_tangent"][0] ** 2)
    np.testing.assert_allclose(np.sum(gram), tangent_energy)


def test_paired_four_term_identity_matches_native_contrast():
    source = np.array([[2.0, -1.0, 0.5]])
    natural = np.array([[-0.2, 0.3, 0.4]])
    control = np.array([[0.1, -0.5, 0.2]])
    result = paired_decomposition(source, natural, control)
    np.testing.assert_allclose(result["identity_residual"], [0.0], atol=1.0e-15)
    np.testing.assert_allclose(
        result["native_contrast"] / result["energy"],
        np.sum(result["normalized_terms"], axis=1),
        atol=1.0e-15,
    )


def test_positive_excursion_product_and_affine_formula_agree():
    energies = np.array([4.0, 3.0, 6.0, 1.5])
    cocycle = face_affine_coefficients(energies)
    assert np.prod(cocycle["gain_squared"]) * energies[0] == energies[-1]
    assert face_affine_endpoint(
        energies[0], cocycle["gain_squared"], cocycle["face_reopening"]
    ) == energies[-1]
