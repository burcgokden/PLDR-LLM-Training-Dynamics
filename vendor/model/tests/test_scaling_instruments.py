"""Numerical controls independent of the measured native trajectories."""
import math
from pathlib import Path
import sys

import numpy as np
from scipy.special import log_ndtr

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from freeze_scaling_forecasts import log_interval_probability
from measure_scaling_noise import fit_common_tangent


def test_censored_likelihood_retains_extreme_gaussian_tails():
    lower=np.array([-np.inf,35.,-1.])
    upper=np.array([-35.,np.inf,1.])
    actual=log_interval_probability(lower,upper)
    expected=np.array([log_ndtr(-35.),log_ndtr(-35.),math.log(math.erf(1/math.sqrt(2)))])
    np.testing.assert_allclose(actual,expected,rtol=1e-13,atol=1e-13)
    assert np.isfinite(actual).all()


def test_common_forcing_predicts_unseen_normalized_rows():
    rng=np.random.default_rng(651)
    rows=rng.normal(size=(16,2,3,7,7))
    rows-=rows.mean(-1,keepdims=True)
    rows/=np.linalg.norm(rows,axis=-1,keepdims=True)
    forcing=rng.normal(size=(2,7))
    forcing-=forcing.mean(-1,keepdims=True)
    v=forcing[None,:,None,None,:]
    tangent=v-rows*np.sum(rows*v,axis=-1,keepdims=True)
    fitted,records=fit_common_tangent(rows,tangent)
    np.testing.assert_allclose(fitted,forcing,rtol=1e-12,atol=1e-12)
    assert all(r['validation_residual_squared']<1e-24 for r in records)
    changed=tangent.copy()
    changed[8:]*=2
    _,records=fit_common_tangent(rows,changed)
    # Fitting never sees the changed validation contexts.
    for r in records:
        np.testing.assert_allclose(r['validation_residual_squared']/r['validation_tangent_squared'],.25,rtol=1e-13)
