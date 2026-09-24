"""The passive recorder preserves a real optimizer trajectory and its output."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'scripts'))
import run_critical_shared_paths as recorder


def test_shared_recorder_does_not_change_adamw_or_random_draws(tmp_path,monkeypatch):
    protocol={'shared_paths':{'schema':'native-shared-parameter-path-v1','cadence':2}}
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    monkeypatch.setattr(recorder.native,'admit',lambda *args,**kwargs:protocol)
    states=[];losses=[]
    def optimizer(model,multiplier):
        return torch.optim.AdamW(model.model.parameters(),lr=.0003*multiplier,betas=(.9,.95),eps=1e-8,weight_decay=.01,foreach=False)
    monkeypatch.setattr(recorder.native,'optimizer_for',optimizer)
    def native_train(study,job,device,role,destination,after_update=None):
        torch.manual_seed(job['seed'])
        module=torch.nn.Module();module.reslayerAs=torch.nn.Linear(3,2)
        wrapper=type('Wrapper',(),{'model':module})()
        opt=recorder.native.optimizer_for(wrapper,job['control'])
        trace=[]
        for _ in range(job['steps']):
            opt.zero_grad(set_to_none=True)
            loss=module.reslayerAs(torch.randn(4,3)).square().mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(module.parameters(),1.,foreach=False,error_if_nonfinite=True)
            opt.step();trace.append(float(loss.detach()))
            if after_update:after_update(len(trace))
        states.append({'model':{k:v.clone() for k,v in module.state_dict().items()},'optimizer':opt.state_dict(),'rng':torch.get_rng_state()})
        losses.append(trace)
        destination.mkdir()
        return {'status':'complete','completed_steps':job['steps'],'artifacts':{}}
    monkeypatch.setattr(recorder.native,'train',native_train)
    job={'seed':12,'control':2.,'steps':4}
    native_train(tmp_path,job,'cpu','qualification',tmp_path/'native')
    original_step=recorder.native.optimizer_for;original_clip=torch.nn.utils.clip_grad_norm_
    result=recorder.train(tmp_path,job,'cpu','qualification',tmp_path/'recorded')
    assert recorder.native.same(*states) and losses[0]==losses[1]
    assert recorder.native.optimizer_for is original_step and torch.nn.utils.clip_grad_norm_ is original_clip
    np.testing.assert_array_equal(np.load(tmp_path/'recorded/shared-steps.npy'),[0,1,2,4])
    raw=np.load(tmp_path/'recorded/shared-parameters.npy')
    expected=np.concatenate([x.numpy().reshape(-1) for x in states[-1]['model'].values()])
    np.testing.assert_array_equal(raw[-1],expected)
    assert len(result['artifacts'])==4 and result['shared_path_recorded']


from analyze_critical_shared_paths import path_budget,first_force_budget


def test_block_budget_retains_anticorrelated_cancellation():
    # Every realization moves and then returns exactly to its initial state.
    signs=np.array([-1.,1.,-2.,2.])[:,None]
    direction=np.array([[.5,1.,-1.]])
    x=np.zeros((4,4,3));x[:,1]=signs*direction/256;x[:,2]=signs*direction
    budget=path_budget(x,[0,1,256,512],.01,0.)
    end=budget['block_budgets'][-1]
    assert end['measured_variance']==0.
    assert end['weighted_diagonal_contribution']>0
    np.testing.assert_allclose(end['signed_off_diagonal_contribution'],-end['weighted_diagonal_contribution'])
    assert budget['block_correlation_matrix'][0][1]==-1.
    assert budget['path_statistics'][2]['unbiased_squared_mean_displacement']<0.


def test_weight_decay_budget_uses_exact_unequal_interval_factors():
    rng=np.random.default_rng(22);rate=.02;decay=.03;q=1-rate*decay
    times=np.array([0,1,256,512,768]);x=np.empty((6,len(times),7));x[:,0]=rng.normal(size=7)
    for k in range(1,len(times)):
        x[:,k]=q**(times[k]-times[k-1])*x[:,k-1]+rng.normal(size=(6,7))*.001
    budget=path_budget(x,times,rate,decay)
    assert budget['maximum_coordinate_telescope_error']<1e-13
    assert budget['maximum_block_variance_error']<1e-15
    for cell in budget['path_statistics']:
        np.testing.assert_allclose(cell['unbiased_squared_mean_displacement']+cell['per_coordinate_variance'],cell['mean_squared_decay_corrected_displacement'],atol=1e-18)


def test_first_force_budget_keeps_float32_arithmetic_remainder():
    rng=np.random.default_rng(3);initial=np.tile(rng.normal(size=40).astype(np.float32),(6,1)).astype(float)
    clipped=rng.normal(size=(6,40))*.001;rate=.0003;decay=.01;eps=1e-8
    after=(initial*(1-rate*decay)-rate*clipped/(np.abs(clipped)+eps)).astype(np.float32).astype(float)
    budget=first_force_budget(initial,after,clipped*10,clipped,rate,decay,eps)
    assert budget['arithmetic_remainder_variance']>0.
    assert abs(budget['actual_step_variance']-budget['ideal_step_variance'])<=budget['arithmetic_covariance_bound']
