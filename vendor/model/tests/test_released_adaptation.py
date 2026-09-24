"""Scientific invariants for repeated-corpus adaptation and causal assessment."""
import json
from pathlib import Path
import numpy as np
import pytest
from scripts.train_released_adaptation import augment,prepare_draws
from scripts.assess_released_adaptation import test_plan as make_test_plan
from model_rg.physical_native import prefix_batch
from model_rg.lattice import observables

@pytest.mark.parametrize('q',[2,3])
def test_training_symmetries_preserve_gibbs_energy_and_order(q):
    rng=np.random.default_rng(219900);x=rng.integers(q,size=(64,8,8),dtype=np.uint8)
    p=rng.random((64,q)).argsort(1);sym=np.tile(np.arange(8),8)
    y=augment(x,q,p,sym)
    ox,oy=observables(x,q),observables(y,q)
    for k in ['energy','m','m2','m4','corr1','corr2','corr4']:
        np.testing.assert_allclose(ox[k],oy[k],rtol=0,atol=1e-14)

@pytest.mark.parametrize('site',[0,1,8,15])
def test_target_and_suffix_cannot_change_native_prefix(site):
    rng=np.random.default_rng(site);a=rng.integers(3,size=(5,4,4),dtype=np.uint8);b=a.copy()
    b.reshape(5,-1)[:,site:]=(b.reshape(5,-1)[:,site:]+1)%3
    aa=prefix_batch(a,site,[1,9,8],[4,5,6],'cpu');bb=prefix_batch(b,site,[1,9,8],[4,5,6],'cpu')
    assert aa.shape==(5,3+site)
    np.testing.assert_array_equal(aa.numpy(),bb.numpy())

def test_repeated_language_epochs_never_add_or_drop_example_identities(tmp_path):
    (tmp_path/'data').mkdir();out=tmp_path/'run';out.mkdir()
    np.savez(tmp_path/'data/language.npz',**{'technical-train-ids':np.arange(4096),'technical-validation-ids':np.arange(5000,5100),'technical-test-ids':np.arange(6000,6100)})
    assert prepare_draws(tmp_path,out,'technical',3,219901)==1024
    with np.load(out/'draws.npz') as z:
        sets=[]
        for e in range(3):
            pairs=z[f'e{e}-examples'].reshape(-1,2);s=set(map(tuple,pairs));sets.append(s)
            assert len(s)==32768 and pairs[:,0].max()==4095 and set(pairs[:,1])==set(range(8))
        assert sets[0]==sets[1]==sets[2]
        assert not np.array_equal(z['e0-examples'],z['e1-examples'])

def test_physical_epochs_visit_each_identity_once_and_never_validation(tmp_path):
    (tmp_path/'data').mkdir();out=tmp_path/'run';out.mkdir()
    cells=[dict(id=0,split='train',chains=2,samples_per_chain=32,q=2,L=4),dict(id=1,split='train',chains=2,samples_per_chain=32,q=3,L=8),dict(id=2,split='validation',chains=2,samples_per_chain=32,q=3,L=8)]
    (tmp_path/'data/physical.json').write_text(json.dumps(dict(cells=cells)))
    assert prepare_draws(tmp_path,out,'physical',3,219902)==4
    with np.load(out/'draws.npz') as z:
        for e in range(3):
            seen=set()
            for cid,ids,site,colors,sym in zip(*(z[f'e{e}-{k}'] for k in ['cell','sample','site','colors','symmetry'])):
                c=cells[cid];assert cid in [0,1] and 0<=site<c['L']**2
                assert np.all(np.sort(colors[:,:c['q']],axis=1)==np.arange(c['q']))
                assert np.all((sym>=0)&(sym<8))
                for i in ids:
                    assert (cid,int(i)) not in seen;seen.add((cid,int(i)))
            assert len(seen)==128

def test_stratified_site_weights_integrate_constant_risk(tmp_path):
    (tmp_path/'data').mkdir()
    cells=[dict(id=L,split='test',q=2,L=L,temperature_ratio=1.) for L in [4,6,8,12,16,20,24]]
    (tmp_path/'data/physical.json').write_text(json.dumps(dict(cells=cells)))
    for p in make_test_plan(tmp_path):
        assert len(set(p['sites']))==8 and min(p['sites'])>=0 and max(p['sites'])<p['L']**2
        assert abs(sum(p['site_weights'])-1)<1e-14
        if p['L']==6:assert len(set(p['site_weights']))==2


def test_lowrank_export_preserves_native_emission_and_input_gradient():
    import torch
    from torch import nn
    from scripts.train_released_lowrank import LowRank,merged
    torch.manual_seed(219903)
    adapted=nn.Module();adapted.final_layer=LowRank(nn.Linear(7,11,dtype=torch.float64),rank=4)
    with torch.no_grad():adapted.final_layer.lora_B.normal_()
    native=nn.Module();native.final_layer=nn.Linear(7,11,dtype=torch.float64)
    native.load_state_dict(merged(adapted,['final_layer']),strict=True)
    x=torch.randn(3,7,dtype=torch.float64,requires_grad=True);y=x.detach().clone().requires_grad_(True)
    a=adapted.final_layer(x);b=native.final_layer(y)
    torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
    a.square().sum().backward();b.square().sum().backward();torch.testing.assert_close(x.grad,y.grad,rtol=1e-12,atol=1e-12)
