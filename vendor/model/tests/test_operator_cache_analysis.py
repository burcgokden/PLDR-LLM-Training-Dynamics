"""Independent consequences of paired categorical errors and complete replicas."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pytest
from analyze_operator_cache import paired_metrics

def test_exact_cache_preserves_every_fluctuation():
    z=np.array([[[0.,1.,2.],[2.,1.,0.]],[[2.,1.,0.],[0.,1.,2.]],[[1.,0.,2.],[0.,2.,1.]]])
    r=paired_metrics(z,z.copy(),np.array([0,2]))
    assert r['relative_centered_rms']==0
    assert r['retained_variance_fraction']==pytest.approx(1)
    assert r['mean_kl']==0 and r['mean_nll_change']==0

def test_common_output_erases_replica_fluctuations():
    rng=np.random.default_rng(18);z=rng.normal(size=(6,4,5))
    q=np.broadcast_to(z.mean(0),z.shape)
    r=paired_metrics(z,q,np.arange(4))
    assert r['relative_centered_rms']==pytest.approx(1)
    assert r['retained_variance_fraction']==pytest.approx(0,abs=1e-28)
    assert r['contexts_over_target']==4

def test_logit_gauge_and_replica_permutation_preserve_result():
    rng=np.random.default_rng(25);z=rng.normal(size=(6,4,5));q=z+rng.normal(size=z.shape)*.03
    a=paired_metrics(z,q,np.arange(4));b=paired_metrics(z[::-1]+3,q[::-1]-5,np.arange(4))
    for key in ['relative_centered_rms','mean_kl','mean_nll_change','signed_cross_covariance']:
        assert a[key]==pytest.approx(b[key],abs=1e-12)

def test_zero_native_variance_has_no_relative_accuracy():
    z=np.zeros((6,4,5))
    with pytest.raises(ValueError,match='zero native variance'):paired_metrics(z,z,np.arange(4))
