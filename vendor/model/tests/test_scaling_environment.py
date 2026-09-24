"""Check all crossed finite-environment sectors in fixed observation units."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_scaling_environment import factorial_components


def test_three_factor_covariance_keeps_all_interactions():
    c=np.array([-1.,1.])[:,None,None,None]
    b=np.array([-1.,1.])[None,:,None,None]
    s=(np.array([-3.,-1.,1.,3.])/np.sqrt(5))[None,None,:,None]
    context=np.arange(7)[None,None,None,:]*100.
    q=context+3*c+5*b+2*s+.7*c*b+.5*c*s+.4*b*s+.9*c*b*s
    result=factorial_components(q)
    expected={'shared':9.,'batch':25.,'initialization':4.,'shared,batch':.49,
              'shared,initialization':.25,'batch,initialization':.16,
              'shared,batch,initialization':.81}
    assert set(result['components'])==set(expected)
    for key,value in expected.items():
        np.testing.assert_allclose(result['components'][key],value,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(result['total_conditional_population_variance'],sum(expected.values()))
