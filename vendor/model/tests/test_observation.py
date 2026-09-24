import itertools
import numpy as np
import pytest
from model_rg.observation import (sample_susceptibility, paired_horizon,
                                  observation_transfer, screen_decomposition)
from scripts.formal_compatibility import check_formal_inventory
from model_rg.provenance import sha256


def test_paired_gram_matches_all_explicit_four_seed_resamples():
    rng=np.random.default_rng(192)
    x=rng.normal(size=(4,7,5));y=.8*x+.1*rng.normal(size=x.shape)
    report,raw=paired_horizon(x,y,14,[11,12,13,14])
    brute=[]
    for idx in itertools.product(range(4),repeat=4):
        a,b=x[list(idx)],y[list(idx)]
        brute.append(14*(np.var(b,axis=0,ddof=1)-np.var(a,axis=0,ddof=1)).mean())
    np.testing.assert_allclose(raw['differences'],brute,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(report['percentiles'],np.quantile(brute,[.025,.975]))
    assert report['resamples']==256


def test_four_seed_resampling_can_exceed_point_variance():
    x=np.zeros((4,1));y=np.array([[0.],[0.],[0.],[1.]])
    report,raw=paired_horizon(x,y,1,[1,2,3,4])
    assert report['difference']==.25
    np.testing.assert_allclose(raw['differences'].max(),1/3)
    assert report['percentiles'][1]>report['difference']


def test_centered_transfer_normalization_and_screen_cross_terms():
    rng=np.random.default_rng(490)
    reference=rng.normal(size=(4,9,5));observed=reference+.05*rng.normal(size=reference.shape)
    result=observation_transfer(reference,observed,14)
    assert abs(result['signed_difference'])<=result['absolute_bound']
    assert result['error_susceptibility']<=result['uncentered_error_bound']
    np.testing.assert_allclose(result['uncentered_error_bound'],14*4/3*np.mean((observed-reference)**2))
    shifted=observation_transfer(reference,reference+3,14)
    assert shifted['error_susceptibility']<1e-28
    rows=np.abs(rng.normal(size=(4,9,5,14)))
    rows[...,1:]=1e-15
    screen=screen_decomposition(rows,14,1e-13)
    assert screen['fraction']>0.9
    np.testing.assert_allclose(screen['below']+screen['above']+screen['cross'],sample_susceptibility(rows.mean(-1),14))
    assert screen['removal_relative']<1e-12


@pytest.mark.parametrize('mutation',['none','missing','changed','root'])
def test_archived_formal_scope_allows_only_hash_preserving_extensions(tmp_path,mutation):
    (tmp_path/'ModelRG').mkdir()
    first=tmp_path/'ModelRG/First.lean';first.write_text('theorem first : True := by trivial\n')
    recorded={'ModelRG/First.lean':sha256(first)}
    (tmp_path/'ModelRG/Second.lean').write_text('theorem second : True := by trivial\n')
    (tmp_path/'ModelRG.lean').write_text('import ModelRG.First\nimport ModelRG.Second\n')
    if mutation=='missing':first.unlink()
    elif mutation=='changed':first.write_text('theorem changed : True := by trivial\n')
    elif mutation=='root':(tmp_path/'ModelRG.lean').write_text('import ModelRG.First\n')
    if mutation=='none':
        assert check_formal_inventory(tmp_path,recorded)['extensions']==['ModelRG/Second.lean']
    else:
        with pytest.raises(AssertionError):check_formal_inventory(tmp_path,recorded)
