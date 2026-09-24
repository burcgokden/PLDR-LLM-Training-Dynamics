"""The accelerated uncertainty estimator preserves complete training realizations."""
import importlib.util
from pathlib import Path
import numpy as np
from model_rg.criticality import conditional_covariances


def test_whole_seed_gram_resampling_matches_materialized_model_replicas():
    import sys
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    sys.path.insert(0,str(scripts))
    try:
        spec=importlib.util.spec_from_file_location('replication_analysis',scripts/'analyze_dynamics_replication.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(scripts))
    rng=np.random.default_rng(702)
    # Shared seed offsets ensure that treating individual contexts as seeds would fail.
    values=rng.normal(size=(7,13,5,4))+.8*rng.normal(size=(7,1,1,1))
    sampled=rng.integers(0,7,size=(23,7))
    counts=np.array([np.bincount(indices,minlength=7) for indices in sampled])
    expected=np.array([np.trace(conditional_covariances(values[indices])[0])/5 for indices in sampled])
    np.testing.assert_allclose(module.bootstrap_trace(values,counts),expected,rtol=1e-13,atol=1e-13)


def test_nearly_concentrated_field_moments_and_resamples_keep_the_small_variance():
    import sys
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    sys.path.insert(0,str(scripts))
    try:
        spec=importlib.util.spec_from_file_location('replication_analysis_small_variance',scripts/'analyze_dynamics_replication.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(scripts))
    values=np.ones((3,1,1,2),dtype=np.float32)
    values[2]=np.nextafter(np.float32(1),np.float32(0))
    delta=float(values[0,0,0,0])-float(values[2,0,0,0])
    result=module.summarize(values)
    # The finite three-point law {1,1,1-delta} has these exact central moments.
    np.testing.assert_allclose(result['centered_second_moment'],2*delta**2/9,rtol=1e-8,atol=0)
    np.testing.assert_allclose(result['centered_fourth_moment'],2*delta**4/27,rtol=1e-8,atol=0)
    counts=np.array([[1,1,1],[3,0,0],[2,0,1],[0,1,2],[0,0,3]])
    # Mixed resamples have unbiased N=2 susceptibility 2*delta^2/3.
    expected=(2*delta**2/3)*np.array([1,0,1,1,0],dtype=float)
    np.testing.assert_allclose(module.bootstrap_trace(values,counts),expected,rtol=1e-8,atol=1e-28)
