import importlib.util
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from readout_budget import budget
from analyze_fresh_readout import gram_readout,vector
from physical_design import schedule_factor,check_draws
import pytest


def test_small_signal_relative_bound_and_cross_cancellation():
    v=np.array([[.1],[-.1]])
    aligned=budget(v,2*v)
    assert np.isclose(aligned['absolute_relative_moment_error'],3)
    assert np.isclose(aligned['sharp_relative_bound'],3)
    reflected=budget(v,-v)
    assert reflected['absolute_relative_moment_error']==0
    assert reflected['delta']>0
    assert np.isclose(reflected['signed_cross'],-reflected['squared_error'])


def test_mean_sector_does_not_control_connected_error():
    v=np.array([[.5001],[.4999]])
    result=budget(v,np.array([[.5],[.5]]))
    assert result['relative_rms']<.001
    assert np.isclose(result['connected_relative_rms'],1.)
    assert np.isclose(result['estimated_connected_variance'],0.)


def test_analytic_gram_recovers_simplex_and_bounds_missing_site():
    q,L=3,4;rng=np.random.default_rng(13)
    x=rng.integers(0,q,(20,L,L));flat=x.reshape(20,-1)
    prefix=np.stack([(flat[:,:-1]==a).mean(1) for a in range(q)],1)
    C=np.array([[1.,2.,4.],[2.,5.,3.],[1.,1.,1.]])
    off=np.array([4.,5.,6.]);traces=off+(L*L-1)*prefix@C.T
    truth=np.stack([(flat==a).mean(1) for a in range(q)],1)
    z={'gram-coefficients-0':C,'gram-offset-0':off,'gram-traces-0':traces,
       'configurations-0':x,'fractions-0':truth,'prefix-fractions-0':prefix}
    pred,details=gram_readout(z,0,q)
    np.testing.assert_allclose(pred,prefix,atol=1e-14)
    assert details['maximum_vector_bound_residual']<=1e-10
    assert np.linalg.norm(vector(pred)-vector(truth),axis=1).max()<=np.sqrt(3)/L**2+1e-14


def test_schedule_convention_and_boundary():
    assert schedule_factor(0,16384)==1/128
    assert schedule_factor(127,16384)==1
    assert schedule_factor(128,16384)==1
    assert schedule_factor(16384,16384)==.2
    assert schedule_factor(16383,16384)>.2


def test_draw_identity_is_cell_local_and_refuses_repetition():
    spec={'steps':2,'stage':'qualification'}
    ds={'cells':[{'id':i,'L':8,'chains':1,'samples_per_chain':64} for i in range(2)]}
    d={'cell':np.array([0,1],dtype='int64'),'site':np.array([1,1],dtype='int64'),
       'index':np.tile(np.arange(32,dtype='int64'),(2,1))}
    check_draws(d,spec,ds)
    d['cell'][1]=0
    with pytest.raises(ValueError,match='repeated'):check_draws(d,spec,ds)
