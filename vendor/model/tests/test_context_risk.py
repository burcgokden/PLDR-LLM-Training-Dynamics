"""Regression cases for the scientific distinction between panel and context risk."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import numpy as np
import pytest
from scripts.analyze_context_risk import context_metrics


def test_aggregate_target_does_not_certify_each_context():
    result=context_metrics([100.,1.],[1.,1.],.25)
    assert result['aggregate_rms']<.25
    assert result['per_context_rms']==[.1,1.]
    assert result['contexts_over_target']==1
    assert result['weighted_mass_over_target']==pytest.approx(1/101)
    assert result['weighted_mass_over_target']<result['weighted_mass_markov_bound']


@pytest.mark.parametrize('native,residual',[
    ([1.,0.],[.1,0.]),([1.,np.nan],[.1,.2]),([1.,np.inf],[.1,.2]),
    ([1.,2.],[.1,-.2]),([1.,2.],[.1,np.inf]),([True,False],[0.,0.])])
def test_undefined_or_nonfinite_variance_is_rejected(native,residual):
    with pytest.raises(ValueError):context_metrics(native,residual,.25)


def test_pair_permutation_and_common_units_preserve_risk():
    original=context_metrics([100.,1.],[1.,1.],.25)
    rescaled=context_metrics([3.,300.],[3.,3.],.25)
    assert rescaled['aggregate_rms']==pytest.approx(original['aggregate_rms'])
    assert rescaled['weighted_mass_over_target']==pytest.approx(original['weighted_mass_over_target'])
    assert rescaled['per_context_rms']==list(reversed(original['per_context_rms']))
