"""Independent finite geometry and declared-outcome completeness regressions."""
import copy
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from equal_time_outcomes import SCHEMA, OUTCOMES, NORMALIZATION, validate_outcomes

def fixture():
    job=dict(run_id='sample',heads=2,control=1.5,seed=9,steps=256)
    p=dict(secondary='signed cross-time matrix energy',jobs=[job],physical_block_fractions=[1,2])
    contract=dict(schema=SCHEMA,outcomes=OUTCOMES,normalization=NORMALIZATION,convention_status='explicit_post_acquisition')
    endpoint=dict(job,increment=.1,finite_cross=.08,quadratic=.02,matrix_derivative=.09,derivative_error=.01,target_nll_before=8.,target_nll_after=7.9)
    cells=[dict(job,factor=f,increment=.1,finite_cross=.08,quadratic=.02,matrix_derivative=.09,matrix_derivative_absolute_error=.01,step_squared_energy=1.,block_squared_energy=.5,signed_cross_time_energy=-.5) for f in [1,2]]
    return dict(outcome_contract=contract,endpoints=[endpoint],cells=cells),p

@pytest.mark.parametrize('change',['missing_field','missing_cell','duplicate','wrong_seed','wrong_sign','nonfinite','negative_energy','missing_contract','boolean_factor'])
def test_declared_outcome_corruptions(change):
    a,p=fixture();validate_outcomes(a,p)
    if change=='missing_field':del a['cells'][0]['signed_cross_time_energy']
    elif change=='missing_cell':a['cells'].pop()
    elif change=='duplicate':a['cells'][1]=copy.deepcopy(a['cells'][0])
    elif change=='wrong_seed':a['cells'][0]['seed']+=1
    elif change=='wrong_sign':a['cells'][0]['signed_cross_time_energy']*= -1
    elif change=='nonfinite':a['cells'][0]['block_squared_energy']=float('nan')
    elif change=='negative_energy':a['cells'][0]['step_squared_energy']=-1
    elif change=='missing_contract':del a['outcome_contract']
    elif change=='boolean_factor':a['cells'][0]['factor']=True
    with pytest.raises(ValueError):validate_outcomes(a,p)


def test_geometry_attains_both_bounds_and_associative_merge():
    rng=np.random.default_rng(318)
    def summary(ds):
        v=ds.sum(0);q=np.square(ds).sum();h=2*sum(np.vdot(x,y) for i,x in enumerate(ds) for y in ds[i+1:])
        return v,q,h
    def merge(x,y):return x[0]+y[0],x[1]+y[1],x[2]+y[2]+2*np.vdot(x[0],y[0])
    for ds in [rng.normal(size=(7,3,4)),np.ones((4,3,4)),np.stack([np.ones((3,4)),-np.ones((3,4))])]:
        v,q,h=summary(ds);np.testing.assert_allclose(np.square(v).sum(),q+h,atol=1e-12)
        assert -q-1e-12<=h<=(len(ds)-1)*q+1e-12
    ds=rng.normal(size=(9,3,4));parts=[summary(ds[:2]),summary(ds[2:6]),summary(ds[6:])]
    for x,y in zip(merge(merge(parts[0],parts[1]),parts[2]),merge(parts[0],merge(parts[1],parts[2]))):np.testing.assert_allclose(x,y,atol=1e-12)
    _,q,h=summary(np.ones((4,3,4)));assert h==3*q
    _,q,h=summary(np.array([[2.,-3.],[-2.,3.]]));assert h==-q
