"""Finite-flow controls exercise cancellation, domain failure and energy weighting."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_equal_time_flux import coordinates
from verify_equal_time_flux import reduce


def test_ordered_energy_weights_are_necessary():
    x=np.array([[1.,0.],[0.,1.]])
    y=np.array([[2.,0.],[0.,1.]])
    z=np.array([[2.,3.],[1.,0.]])
    def qr(x,y):
        _,u,a,b,c=coordinates(x,y)
        return a+b+u*c,1+c
    q1,r1=qr(x,y);q2,r2=qr(y,z);q,r=qr(x,z)
    assert np.allclose((q1+r1*q2,r1*r2),(q,r))
    assert not np.isclose(q1+q2,q)


def test_signed_denominator_error_budget():
    x=np.array([[1.,2.],[-1.,3.]])
    y=np.array([[3.,1.],[2.,-1.]])
    values,u,a,b,c=coordinates(x,y)
    ah,bh,ch=a+.03,b-.02,c+.12
    f=(ah+bh)/(1+ch)
    remainder=((a-ah)+(b-bh)-(c-ch)*f)/(1+c)
    assert np.isclose(values['increment']-f,remainder)
    assert abs(remainder)<=(abs(a-ah)+abs(b-bh)+abs(f)*abs(c-ch))/(1+c)+1e-15


def test_independent_common_row_reduction_near_collapse():
    rng=np.random.default_rng(737)
    x=np.ones((20,4,4))+1e-5*rng.normal(size=(20,4,4))
    y=x+.02*rng.normal(size=x.shape)
    primary,*_=coordinates(x,y);independent=reduce(x,y)
    for name in ['increment','finite_cross','quadratic','matrix_derivative']:
        assert np.allclose(primary[name],independent[name],rtol=0,atol=1e-14)


def test_zero_energy_is_outside_domain():
    with pytest.raises(ValueError):coordinates(np.ones((2,2)),np.zeros((2,2)))
    with pytest.raises(ValueError):reduce(np.zeros((2,2)),np.ones((2,2)))
