"""Exercise production control flow with real CPU autograd and AdamW."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_critical_onepass as native
import run_critical_shared_paths as recorder
from model_rg.run_outcome import inventory


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    study=tmp_path/'study';study.mkdir();corpus=tmp_path/'corpus';corpus.mkdir()
    np.save(corpus/'tokens.npy',np.zeros((4,513),dtype=np.int32))
    np.savez(study/'selection.npz',blocks_0=np.arange(32).reshape(1,32),probes=np.zeros((128,65),dtype=np.int32))
    protocol={'source_sha256':{},'observations':{'milestone_steps':[0,1]},'memory_ceiling_bytes':10**10,
              'shared_paths':{'schema':'native-shared-parameter-path-v1','cadence':256}}
    job=dict(run_id='fixture',environment=0,heads=2,seed=1,control=2.,steps=1,shared_seed=2,save_optimizer=True)
    protocol['jobs']=[job];(study/'protocol.json').write_text(json.dumps(protocol))
    class Fake:
        def __init__(self,*args):
            self.model=torch.nn.Module();self.model.reslayerAs=torch.nn.Linear(1,1,bias=False)
        def forward(self,x,capture=False):
            a=self.model.reslayerAs.weight[0,0].expand(x.shape[0],x.shape[1])
            return SimpleNamespace(logits=torch.stack((a,torch.zeros_like(a)),dim=-1))
    monkeypatch.setattr(native,'admit',lambda *a,**kw:protocol)
    monkeypatch.setattr(native,'CORPUS',corpus);monkeypatch.setattr(native,'TrainingModel',Fake)
    monkeypatch.setattr(native,'normalize_variance_initialization',lambda *a:None)
    monkeypatch.setattr(native,'fix_shared_generator',lambda *a:'fixture')
    monkeypatch.setattr(native,'parameter_digest',lambda *a:'fixture')
    monkeypatch.setattr(native,'optimizer_for',lambda model,g:torch.optim.AdamW(model.model.parameters(),lr=.001))
    obs={'heads':np.zeros((64,5,2,4)),'nll':np.zeros(128),'logits':np.zeros((8,2))}
    monkeypatch.setattr(native,'observe',lambda *a:obs)
    for name in ['set_device','init','reset_peak_memory_stats']:
        monkeypatch.setattr(torch.cuda,name,lambda *a:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda *a:0)
    monkeypatch.setattr(torch.cuda,'get_rng_state',lambda *a:torch.zeros(1,dtype=torch.uint8))
    return study,job,study/'runs'/job['run_id'],Fake


def test_success_and_empty_initial_observation(fixture,monkeypatch):
    study,job,dest,_=fixture
    def fail(*a):raise FloatingPointError('initial snapshot')
    monkeypatch.setattr(native,'observe',fail)
    m=native.train(study,job,'cpu','qualification',dest)
    assert m['status']=='numerical_failure' and m['completed_steps']==0
    assert m['available_snapshot_steps']==[] and m['checkpoint_available']
    with np.load(dest/'observations.npz') as data:assert data['heads'].shape==(0,64,5,2,4)


def test_actual_nonfinite_gradient(fixture,monkeypatch):
    study,job,dest,Fake=fixture
    class Infinite(Fake):
        def __init__(self,*a):
            super().__init__(*a)
            self.model.reslayerAs.weight.register_hook(lambda g:torch.full_like(g,float('inf')))
    monkeypatch.setattr(native,'TrainingModel',Infinite)
    m=native.train(study,job,'cpu','qualification',dest)
    assert m['status']=='numerical_failure' and m['stage']=='gradient_clipping'
    assert m['attempted_steps']==1 and m['completed_steps']==0
    assert np.isfinite(np.load(dest/'observations.npz')['heads']).all()


@pytest.mark.parametrize('exception',[MemoryError('startup'),torch.OutOfMemoryError('startup')])
def test_startup_resource_record_precedes_allocation(fixture,monkeypatch,exception):
    study,job,dest,_=fixture
    def fail(*a):
        assert json.loads((dest/'manifest.json').read_text())['status']=='initializing'
        raise exception
    monkeypatch.setattr(native,'TrainingModel',fail)
    m=native.train(study,job,'cpu','qualification',dest)
    assert m['status']=='resource_failure' and not m['checkpoint_available']


def test_completed_update_survives_observation_failure(fixture):
    study,job,dest,_=fixture
    def fail(step):raise FloatingPointError('after update')
    m=native.train(study,job,'cpu','qualification',dest,after_update=fail)
    assert m['completed_steps']==m['qualification_updates']==1
    assert m['status']=='numerical_failure' and not m['optimizer_partial_mutation']


def test_optimizer_exception_marks_partial_mutation(fixture,monkeypatch):
    study,job,dest,_=fixture
    original=native.optimizer_for
    def optimizer(model,g):
        opt=original(model,g)
        def fail():raise RuntimeError('unknown optimizer error')
        opt.step=fail
        return opt
    monkeypatch.setattr(native,'optimizer_for',optimizer)
    with pytest.raises(RuntimeError,match='unknown optimizer'):native.train(study,job,'cpu','qualification',dest)
    m=json.loads((dest/'manifest.json').read_text())
    assert m['status']=='program_error' and m['optimizer_partial_mutation'] and not m['checkpoint_resumable']


def test_save_error_preserves_original_failure(fixture,monkeypatch):
    study,job,dest,_=fixture
    def fail(*a,**kw):raise OSError('checkpoint write')
    monkeypatch.setattr(torch,'save',fail)
    def observer(step):raise FloatingPointError('original observation')
    m=native.train(study,job,'cpu','qualification',dest,after_update=observer)
    assert m['status']=='numerical_failure' and m['error']=='original observation'
    assert m['artifact_save_errors'][0]['message']=='checkpoint write'


def test_recorder_save_error_has_terminal_record(fixture,monkeypatch):
    study,job,dest,_=fixture
    def fail(*a,**kw):raise OSError('recorder write')
    monkeypatch.setattr(recorder.np,'save',fail)
    with pytest.raises(OSError,match='recorder write'):recorder.train(study,job,'cpu','qualification',dest)
    m=json.loads((dest/'manifest.json').read_text())
    assert m['status']=='program_error' and m['completed_steps']==1
    assert m['original_failure']['stage']=='recorder_save'


def test_successful_recorded_path_and_missing_job_accounting(fixture):
    study,job,dest,_=fixture
    p=json.loads((study/'protocol.json').read_text())
    assert inventory(study,p)['status_counts']['incomplete']==1
    m=recorder.train(study,job,'cpu','qualification',dest)
    assert m['status']=='complete' and m['shared_path_recorded']
    assert m['completed_steps']==m['attempted_steps']==1
    assert inventory(study,p)['status_counts']['complete']==1
