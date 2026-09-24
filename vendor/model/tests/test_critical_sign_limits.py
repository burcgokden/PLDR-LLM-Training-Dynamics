"""Sign integration distinguishes regular Gaussian mixtures from critical claims."""
import itertools
import numpy as np
import scripts.analyze_critical_sign_limits as sign


def test_native_projection_orbit_matches_all_sign_representatives():
    x=np.array([1.,2.,-3.,4.])
    signs=np.array(list(itertools.product([-1.,1.],repeat=4)))
    y=signs@x/2
    moments,comparisons=sign.orbit_projection_statistics(x)
    np.testing.assert_allclose(moments['variance'],np.mean(y*y),rtol=0,atol=1e-14)
    np.testing.assert_allclose(moments['fourth_moment'],np.mean(y**4),rtol=0,atol=1e-13)
    for record in comparisons:
        exact=np.mean(np.cos(record['frequency']*y/np.sqrt(np.mean(y*y))))
        np.testing.assert_allclose(record['exact'],exact,rtol=0,atol=2e-15)
        if record['bound_domain']:
            assert record['absolute_error']<=record['bound']+2e-15


def test_positive_fourth_cumulant_can_come_from_orbit_variance_mixing():
    x=np.array([[1.,1.],[3.,3.]])
    budget=sign.mixture_fourth_budget(x)
    assert budget['variance']==5
    assert budget['conditional_sign_contribution']==-41
    assert budget['variance_mixture_contribution']==48
    assert budget['fourth_cumulant']==7
    signs=np.array(list(itertools.product([-1.,1.],repeat=2)))
    y=(signs@x.T/np.sqrt(2)).ravel()
    np.testing.assert_allclose(np.mean(y**4)-3*np.mean(y*y)**2,7,atol=1e-13)


def test_large_head_count_does_not_remove_one_dominant_normalized_sign():
    x=np.zeros(64);x[0]=1
    moments,_=sign.orbit_projection_statistics(x)
    assert moments['effective_heads']==1
    assert moments['standardized_orbit_excess']==-2
    zero,comparisons=sign.orbit_projection_statistics(np.zeros((3,64)))
    np.testing.assert_array_equal(zero['variance'],0)
    for record in comparisons:
        np.testing.assert_array_equal(record['exact'],1)
        np.testing.assert_array_equal(record['gaussian'],1)
        np.testing.assert_array_equal(record['absolute_error'],0)
