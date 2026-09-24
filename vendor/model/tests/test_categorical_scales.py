"""Finite partition laws against independent small probability panels."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_categorical_scales import reduce_panel


def test_nonuniform_nested_reference_and_risk():
    rng=np.random.default_rng(2727)
    p=rng.dirichlet(np.ones(9),size=(6,3))
    r=rng.dirichlet(np.ones(9),size=3)
    cells,edges=reduce_panel(p,r,np.array([8,1,5,0,2,7,6,3,4]),[1,3,6,8])
    assert all(e['mean_kl_gain']>=0 and e['variance_gain']>=0 for e in edges)
    assert all(c['maximum_algebra_error']<1e-13 for c in cells)
    assert all(c['centered_remainder_variance']<=6/5*c['uncentered_energy']+1e-13 for c in cells)
    assert all(c['relative_centered_rms']<=c['sufficient_kl_relative_bound']+1e-13 for c in cells)


def test_zero_variance_does_not_define_relative_accuracy():
    p=np.full((6,3,5),.2)
    with pytest.raises(ValueError,match='undefined'):
        reduce_panel(p,np.full(5,.2),np.arange(5),[1,3])


def test_inconsistent_reference_has_no_absorption_law():
    p=np.array([.1,.2,.3,.4]);r=np.array([.4,.3,.2,.1]);s=r[::-1]
    def lift(a,ref,k):
        q=a.copy();q[k:]=a[k:].sum()*ref[k:]/ref[k:].sum();return q
    assert not np.allclose(lift(lift(p,r,1),s,2),lift(p,r,1))
