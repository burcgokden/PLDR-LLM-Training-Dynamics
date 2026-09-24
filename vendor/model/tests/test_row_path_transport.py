"""Nontrivial controls for finite versus differential chronological transport."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_row_path_confirmation import transport


def test_outgoing_and_incoming_denominators_are_distinct():
    a=np.array([[1.,0.],[0.,0.]])
    r=transport(a,np.array([[1.,0.],[1.,0.]]))
    assert r['finite_cross']==-.5 and r['matrix_derivative']==-1.
    assert r['increment']==-.5 and np.isnan(r['error_bound'])


def test_pure_transverse_creation_needs_quadratic_term():
    a=np.ones((2,2));b=a+np.array([[.2,-.2],[-.2,.2]])
    r=transport(a,b)
    assert r['finite_cross']==0 and r['matrix_derivative']==0
    assert r['increment']>0
    np.testing.assert_allclose(r['increment'],r['quadratic'],atol=1e-15)


def test_radial_rescaling_preserves_row_coordinate():
    a=np.array([[1.,3.],[-2.,.5]])
    r=transport(a,1.7*a)
    np.testing.assert_allclose([r['increment'],r['finite_cross'],r['quadratic'],r['matrix_derivative']],0,atol=1e-14)


def test_path_error_bound_and_all_aligned_compositions():
    rng=np.random.default_rng(91736)
    a=rng.normal(size=(9,3,4,4));a[1:]=a[0]+np.cumsum(.025*a[1:],axis=0)
    step=transport(a[:-1],a[1:])
    error=abs(np.cumsum(step['increment']-step['matrix_derivative'],axis=0))
    assert np.all(error<=np.cumsum(step['error_bound'],axis=0)+1e-14)
    for block in [1,2,4,8]:
        starts=np.arange(0,8,block);r=transport(a[starts],a[starts+block])
        chronological=np.stack([(step['finite_cross'][k:k+block]+step['quadratic'][k:k+block]).sum(0) for k in starts])
        np.testing.assert_allclose(r['increment'],chronological,atol=1e-14)


@pytest.mark.parametrize('bad',[np.zeros((2,2)),np.full((2,2),np.nan)])
def test_undefined_row_quotients_refused(bad):
    with pytest.raises(ValueError):transport(bad,np.ones((2,2)))
