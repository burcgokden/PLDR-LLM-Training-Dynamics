import importlib.util
from pathlib import Path
import numpy as np
import pytest
from model_rg.critical_onepass import conditional_statistics, peak_profile, seed_uncertainty


def test_common_mode_is_extensive_but_independent_heads_are_not():
    # All 16 independent Rademacher configurations yield exactly zero cross covariance.
    independent=np.array([[1. if (s>>h)&1 else -1. for h in range(4)] for s in range(16)])
    independent=independent[:,None,None,:]
    stat=conditional_statistics(independent)
    assert stat['susceptibility']==pytest.approx(stat['one_head_variance'])
    assert stat['cross_head_covariance']==pytest.approx(0.)
    shared=np.broadcast_to(independent[:,:,:,0:1],independent.shape)
    common=conditional_statistics(shared)
    assert common['susceptibility']==pytest.approx(4*common['one_head_variance'])


def test_fixed_document_offsets_do_not_create_replica_susceptibility():
    rng=np.random.default_rng(7)
    x=rng.normal(size=(8,6,2,4))
    offset=rng.normal(size=(1,6,2,1))*20
    before=conditional_statistics(x);after=conditional_statistics(x+offset)
    for key in ['susceptibility','one_head_variance','cross_head_covariance','empirical_binder']:
        assert before[key]==pytest.approx(after[key])


def test_seed_resampling_preserves_repeated_context_dependence():
    x=np.arange(24,dtype=float).reshape(6,1,1,4)
    a=seed_uncertainty(x,draws=200)
    b=seed_uncertainty(np.repeat(x,15,axis=1),draws=200)
    assert a['susceptibility_percentile_95']==pytest.approx(b['susceptibility_percentile_95'])
    assert a['mean_percentile_95']==pytest.approx(b['mean_percentile_95'])


def test_half_height_requires_both_sides():
    p=peak_profile([0,1,2,3,4],[0,1,2,1,0])
    assert p['half_width']==pytest.approx(2.)
    assert p['sampled_maximum_control']==2
    p=peak_profile([0,1,2],[0,1,2])
    assert not p['interior'] and p['half_width'] is None


def test_design_rejects_repeated_or_unsupported_axes():
    path=Path(__file__).resolve().parents[1]/'scripts/run_critical_onepass.py'
    spec=importlib.util.spec_from_file_location('critical_runner',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    d=dict(name='test',heads=[2,4],controls=[0,1],seeds=[3,4],environments=[0],steps=2048,shared_seed=9)
    module.validate_design(d)
    for change in [dict(seeds=[3,3]),dict(steps=2049),dict(environments=[2]),dict(controls=[float('nan')]),dict(heads=[1])]:
        with pytest.raises(ValueError): module.validate_design(dict(d,**change))


def test_role_override_rejected_before_native_dispatch():
    path=Path(__file__).resolve().parents[1]/'scripts/run_critical_onepass.py'
    spec=importlib.util.spec_from_file_location('critical_runner_cli',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for action in ['worker','run','qualify']:
        args=[action,'--study','/tmp/no-native-dispatch','--role','scientific']
        if action=='worker':args+=['--run-id','invented']
        with pytest.raises(SystemExit) as exc: module.main(args)
        assert exc.value.code==2
