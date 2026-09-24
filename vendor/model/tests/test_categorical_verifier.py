"""Executable corruption checks for the proposed categorical verifier."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/verify_categorical_visibility.py'
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location('candidate_visibility', SCRIPT)
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


@pytest.fixture
def study(tmp_path):
    root=tmp_path/'study'; root.mkdir()
    native=tmp_path/'native'; native.mkdir()
    logits=np.random.default_rng(1801).normal(size=(3,2,5))
    jobs=[dict(heads=2,control=0,seed=s,steps=1,run_id=f's{s}') for s in range(3)]
    write(native/'protocol.json',dict(design=dict(heads=[2],controls=[0]),jobs=jobs))
    for job,z in zip(jobs,logits):
        folder=native/'runs'/job['run_id'];folder.mkdir(parents=True)
        np.savez(folder/'observations.npz',logits_1=z)
        write(folder/'manifest.json',dict(status='complete',
            artifacts={'observations.npz':v.sha256(folder/'observations.npz')}))
    reference=np.array([[.1,.2,.25,.15,.3],[.15,.25,.2,.3,.1]])
    order=np.arange(5)
    np.savez(root/'reference.npz',reference=reference,order=order)
    protocol=dict(development_checked_sha256={},evaluation_study=str(native),
        evaluation_protocol_sha256=v.sha256(native/'protocol.json'),
        reference_sha256=v.sha256(root/'reference.npz'),
        retained_tokens=[2],relative_centered_rms_target=.25)
    write(root/'protocol.json',protocol)
    p=np.exp(logits-logits.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
    full=2*np.sqrt(p);tail=p[:,:,2:].sum(-1)
    coarse=2*np.sqrt(np.concatenate([p[:,:,:2],tail[...,None]],-1))
    conditional=reference[:,2:]/reference[:,2:].sum(-1,keepdims=True)
    lifted=full.copy();lifted[:,:,2:]=2*np.sqrt(tail[...,None]*conditional[None])
    variance=lambda x:float(np.var(x,axis=0,ddof=1).sum(-1).mean())
    a=variance(full);b=variance(coarse);c=variance(full-lifted)
    kl=float(np.mean(np.sum(p[:,:,2:]*np.log(p[:,:,2:]/(tail[...,None]*conditional[None])),axis=-1)))
    row=dict(heads=2,control=0,retained_tokens=2,native_variance=a,coarse_variance=b,
        centered_remainder_variance=c,relative_centered_rms=float(np.sqrt(c/a)),
        retained_variance_fraction=b/a,mean_emission_kl=kl,target_met=bool(np.sqrt(c/a)<=.25))
    write(root/'analysis.json',dict(status='complete',cells=[row]))
    return root


def corrupt(study,key,value):
    p=study/'analysis.json';x=json.loads(p.read_text());x['cells'][0][key]=value;write(p,x)


def test_positive_reconstruction(study,tmp_path):
    output=tmp_path/'passed.json'
    v.verify(study,output)
    r=json.loads(output.read_text())
    assert r['status']=='passed' and r['cells']==1 and r['observed_paths']==3
    assert r['maximum_absolute_error'] < 1e-14


@pytest.mark.parametrize('key',v.NUMERICAL_CLAIMS)
@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf'),'NaN',None,True,[],{}])
def test_invalid_claim_is_rejected_without_certificate(study,tmp_path,key,value):
    corrupt(study,key,value)
    output=tmp_path/'must-not-exist.json'
    with pytest.raises(ValueError,match='numerical scalar'):
        v.verify(study,output)
    assert not output.exists()


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_reconstruction_is_rejected(study,tmp_path,monkeypatch,value):
    monkeypatch.setattr(v,'pair_variance',lambda x:value)
    output=tmp_path/'must-not-exist.json'
    with pytest.raises(ValueError,match='reconstructed'):
        v.verify(study,output)
    assert not output.exists()


def test_zero_denominator_is_explicit(study,tmp_path,monkeypatch):
    monkeypatch.setattr(v,'pair_variance',lambda x:0.)
    output=tmp_path/'must-not-exist.json'
    with pytest.raises(ValueError,match='Undefined relative visibility'):
        v.verify(study,output)
    assert not output.exists()


def test_finite_mismatch_still_rejected(study,tmp_path):
    corrupt(study,'native_variance',100.)
    with pytest.raises(ValueError,match='Independent reduction differs'):
        v.verify(study,tmp_path/'never.json')


def test_duplicate_cell_rejected(study,tmp_path):
    p=study/'analysis.json';r=json.loads(p.read_text());r['cells'].append(r['cells'][0]);write(p,r)
    with pytest.raises(ValueError,match='Duplicate'):
        v.verify(study,tmp_path/'never.json')


@pytest.mark.parametrize('mode',[[],['-O']])
@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
def test_cli_nonfinite_rejection_in_both_modes(study,tmp_path,mode,value):
    corrupt(study,'native_variance',value)
    output=tmp_path/'never.json'
    r=subprocess.run([sys.executable,*mode,str(SCRIPT),'--study',str(study),'--output',str(output)],
                     capture_output=True,text=True)
    assert r.returncode != 0 and 'Nonfinite numerical scalar' in r.stderr
    assert not output.exists()


@pytest.mark.parametrize('mode',[[],['-O']])
def test_cli_positive_in_both_modes(study,tmp_path,mode):
    output=tmp_path/'positive.json'
    r=subprocess.run([sys.executable,*mode,str(SCRIPT),'--study',str(study),'--output',str(output)],
                     capture_output=True,text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads(output.read_text())['status']=='passed'


def test_guard_rejects_arithmetic_overflow():
    with pytest.raises(ValueError,match='discrepancy'):
        v.discrepancy_scalar(1e308,-1e308,'overflow')
    with pytest.raises(ValueError,match='Nonfinite'):
        v.finite_scalar(10**1000,'integer')


def test_tolerance_and_replica_domain():
    assert v.discrepancy_scalar(1.,1.+1e-13,'within') < 3e-12
    for bad in [np.zeros((1,2,3)),np.full((2,2,3),np.nan)]:
        with pytest.raises(ValueError):
            v.pair_variance(bad)
