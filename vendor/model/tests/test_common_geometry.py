"""Solved directional, radial and zero-radius common-sector examples."""
import numpy as np

from model_rg.common_geometry import radial_directional


def test_centroid_geometry_distinguishes_radius_from_direction():
    # The coordinate inner product is a mean, so these all have radius one.
    directions = np.sqrt(2)*np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
    first = radial_directional(directions[:,None,None], 14)
    np.testing.assert_allclose(first['radial_susceptibility'], 0., atol=1e-14)
    np.testing.assert_allclose(first['directional_susceptibility'], 14*4/3)
    np.testing.assert_allclose(first['across_seed_direction_cosine'], -1/3)
    # All nonzero vectors now share a direction; the zero vector is harmless.
    radial = np.sqrt(2)*np.array([[0.,0.],[1.,0.],[2.,0.],[3.,0.]])
    second = radial_directional(radial[:,None,None], 14)
    np.testing.assert_allclose(second['radial_susceptibility'], 14*5/3)
    np.testing.assert_allclose(second['directional_susceptibility'], 0., atol=1e-14)
    assert second['zero_radius_vectors'] == 1
    zero = radial_directional(np.zeros((4,2)), 14)
    assert zero['directional_fraction'] is None

    # For these small perturbations around (1,1), the limiting directional
    # fraction is 1/10 and the radial fraction is 9/10. Fixed relative
    # tolerances on tiny pair distances must not mistake roundoff for physics.
    pattern = np.array([[1.,2.],[2.,1.],[-1.,-2.],[-2.,-1.]])
    for amplitude in [1e-6,1e-8,1e-10]:
        near = radial_directional((1+amplitude*pattern)[:,None,None],14)
        np.testing.assert_allclose(near['directional_fraction'],.1,rtol=1e-6,atol=1e-9)
        np.testing.assert_allclose(near['total_susceptibility'],14*10*amplitude**2/3,rtol=1e-6)
