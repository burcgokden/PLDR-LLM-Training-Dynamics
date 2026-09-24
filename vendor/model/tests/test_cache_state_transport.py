"""Finite vector transport and independent predictive covariance checks."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_operator_cache import paired_metrics
from verify_cache_state_transfer import predictive


def test_state_transport_with_signed_calibration_cross():
    rng=np.random.default_rng(32001)
    initial=rng.normal(size=(13,17));current=.2*initial+rng.normal(size=(13,17))
    weights=rng.uniform(size=13);weights/=weights.sum();m0=weights@initial;mt=weights@current
    for sign in [-1.,1.]:
        cache=m0+sign*(mt-m0)
        variance=np.sum(weights[:,None]*(current-mt)**2)
        risk=np.sum(weights[:,None]*(current-cache)**2)
        delta=mt-m0;offset=m0-cache;cross=2*delta@offset
        assert np.sign(cross)==-sign
        assert risk==pytest.approx(variance+delta@delta+offset@offset+cross,abs=1e-12)
        assert np.sqrt(risk)<=np.sqrt(variance)+np.linalg.norm(delta)+np.linalg.norm(offset)+1e-12


def test_independent_pair_reduction_with_large_and_small_errors():
    rng=np.random.default_rng(32002);native=rng.normal(size=(6,64,7)).astype(np.float32);targets=rng.integers(7,size=64)
    for amplitude in [.001,.1,3.]:
        cached=native+amplitude*rng.normal(size=native.shape)
        primary=paired_metrics(native,cached,targets);independent=predictive(native,cached,targets)
        for key,value in independent.items():np.testing.assert_allclose(value,primary[key],atol=4e-14,rtol=1e-12)
