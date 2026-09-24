"""Independent endpoint tests of the finite metric enclosures."""
import numpy as np
from scipy.special import softmax
from scipy.stats import entropy
from scripts.analyze_finite_metric_budget import budget, factor
from scripts.refine_finite_metric_budget import midpoint_bounds


def test_constant_gauge_has_zero_motion():
    a=np.array([[1., -2., 4., .2]])
    result=budget(a,a+3.)
    assert abs(result['kl'][0])<1e-14
    assert result['quadratic'][0]<1e-28
    assert abs(factor(np.array([0.]))[0]-1.)<1e-15


def test_skewed_distributions_and_nonlocal_spans_are_enclosed():
    rng=np.random.default_rng(240914)
    a=rng.normal(size=(20,31))*4
    b=a+rng.normal(size=a.shape)*2
    result=budget(a,b)
    direct=entropy(softmax(a,axis=-1),softmax(b,axis=-1),axis=-1)
    np.testing.assert_allclose(result['kl'],direct,rtol=1e-12,atol=1e-14)
    assert np.all(result['lower']<=direct+1e-13)
    assert np.all(result['upper']>=direct-1e-13)
    shifted=budget(a+np.arange(20)[:,None],b-np.arange(20)[:,None])
    for key in result: np.testing.assert_allclose(result[key],shifted[key],rtol=1e-10,atol=1e-10)


def test_refinement_controls_large_fisher_error():
    # A finite change for which treating the initial metric as constant is poor.
    logits=np.array([[0.,0.,0.],[5.,-3.,1.],[-1.,2.,4.]])
    direct=entropy(softmax(logits[:-1],axis=-1),softmax(logits[1:],axis=-1),axis=-1)
    widths=[]
    for m in [1,4,16,64]:
        lower,upper=midpoint_bounds(logits,m)
        assert np.all(lower<=direct+1e-12)
        assert np.all(upper>=direct-1e-12)
        widths.append(np.sum(upper-lower))
    assert widths[-1] < widths[0]/100
