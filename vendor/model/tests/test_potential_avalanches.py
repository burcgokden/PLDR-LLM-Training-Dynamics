import numpy as np
from model_rg.potential_avalanches import excursions,complete_events,pareto_fit,iaaft
from scripts.train_potential_avalanches import lr_at

def test_boundary_censoring_and_bin_units():
    e=excursions([3,0,2,4,0,5],1,2)
    assert len(e)==3 and len(complete_events(e))==1
    assert e[0]['left_censored'] and e[-1]['right_censored']
    assert e[1]['duration']==4 and e[1]['size']==8 and e[1]['start']==4

def test_powerlaw_recovery_is_distributional():
    r=np.random.default_rng(8)
    x=np.exp(r.exponential(1/1.5,10000))
    fit=pareto_fit(x)
    assert fit['status']=='fitted' and abs(fit['alpha']-2.5)<.15
    assert fit['n_tail']>=50

def test_surrogate_preserves_amplitudes_and_approximately_spectrum():
    r=np.random.default_rng(8);x=np.zeros(1024)
    for k in range(1,len(x)):x[k]=.85*x[k-1]+r.normal()
    y,err=iaaft(x,r)
    np.testing.assert_array_equal(np.sort(x),np.sort(y))
    assert err<.025 and not np.array_equal(x,y)

def test_schedule_phase_endpoints():
    assert lr_at(1,2048,256,8e-4,'constant')==8e-4
    assert lr_at(256,2048,256,8e-4,'warm_cosine')==8e-4
    assert abs(lr_at(2048,2048,256,8e-4,'warm_cosine')-8e-5)<1e-18
    assert lr_at(2048,2048,256,8e-4,'warm_plateau')==8e-4
