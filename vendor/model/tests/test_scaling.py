"""Independent finite identities used in collective size-time analysis."""
import importlib.util
from pathlib import Path
import itertools

import numpy as np
import torch

from model_rg.scaling import metric_coordinates, collective_moments


spec=importlib.util.spec_from_file_location('analyze_size_time_test',Path(__file__).resolve().parents[1]/'scripts/analyze_size_time.py')
analysis=importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_metric_row_orthogonality_on_nonconstant_matrices():
    a=torch.tensor([[[[1.,2.,3.],[4.,5.,6.],[9.,7.,2.]]]],dtype=torch.float64)
    c,e=metric_coordinates(a)
    projected=torch.ones((3,3),dtype=torch.float64)/3@a
    normal=a-projected
    torch.testing.assert_close(e[...,0],torch.linalg.matrix_norm(normal).square()/9)
    torch.testing.assert_close(e[...,1],e[...,0]+c.square().mean(-1),rtol=1e-14,atol=1e-14)


def test_exact_seed_resampling_includes_variance_increase():
    q=np.array([0.,0.,0.,1.])[:,None]
    stats,empirical=analysis.whole_seed_statistics(q,1)
    exact=np.array([np.var(q[list(ids),0],ddof=1) for ids in itertools.product(range(4),repeat=4)])
    np.testing.assert_allclose(empirical,exact,rtol=1e-14,atol=1e-15)
    assert np.max(empirical)>stats['susceptibility']


def test_conditional_and_cohort_covariance_are_distinct():
    q=np.array([[-1.,1.],[1.,-1.],[-2.,2.],[2.,-2.]])
    stats,_=analysis.whole_seed_statistics(q,7)
    assert stats['cohort_susceptibility']==0
    np.testing.assert_allclose(stats['susceptibility'],7*10/3)


def test_linear_matrix_sector_variances_add():
    rng=np.random.default_rng(17)
    a=rng.normal(size=(8,6,2,4,4))+3*rng.normal(size=(8,1,2,1,4))
    common=a.mean(-2,keepdims=True)
    total=collective_moments(a,13)['susceptibility']
    normal=collective_moments(a-common,13)['susceptibility']
    constant=collective_moments(common,13)['susceptibility']
    np.testing.assert_allclose(total,normal+constant,rtol=1e-13)


def test_finite_seed_binder_diagnostic_detects_one_seed_dominance():
    stats,_=analysis.whole_seed_statistics(np.array([0.,0.,0.,1.])[:,None],1)
    np.testing.assert_allclose(stats['cohort_binder'],2/9)
    np.testing.assert_allclose(stats['finite_seed_gaussian_binder_mean'],2/5)


def test_quadratic_observable_has_different_size_exponent():
    x=np.array([-2.,-1.,1.,3.])
    n1,n2=16,256
    q1,q2=x*n1**(-.25),x*n2**(-.25)
    linear1=collective_moments(q1[:,None],n1)['susceptibility']
    linear2=collective_moments(q2[:,None],n2)['susceptibility']
    square1=collective_moments((q1*q1)[:,None],n1)['susceptibility']
    square2=collective_moments((q2*q2)[:,None],n2)['susceptibility']
    np.testing.assert_allclose(linear2/linear1,(n2/n1)**.5)
    np.testing.assert_allclose(square2,square1)
