"""Independent small arrays check the geometry used in native inference analysis."""
import numpy as np
import scripts.analyze_finetuning as module


def test_fisher_matches_explicit_categorical_matrix_and_gauge():
    p=np.array([[.1,.2,.7],[.3,.4,.3]])
    x=np.array([[2.,-1.,3.],[7.,4.,-2.]])
    y=np.array([[-3.,6.,1.],[2.,-1.,1.]])
    expected=np.array([a@(np.diag(q)-np.outer(q,q))@b for a,b,q in zip(x,y,p)])
    np.testing.assert_allclose(module.fisher_product(x,y,p),expected)
    np.testing.assert_allclose(module.fisher_product(x+np.array([[10.],[-50.]]),y-9,p),expected,atol=2e-13)


def test_response_subspace_overlap_recovers_rotation_and_new_direction():
    a=np.array([[1.,0.,0.],[0.,1.,0.]])
    rot=np.array([[.6,.8],[-.8,.6]])
    b=rot@a
    np.testing.assert_allclose(module.canonical_overlap(a@a.T,b@b.T,a@b.T),[1,1],atol=1e-14)
    c=np.array([[1.,0.,0.],[0.,0.,1.]])
    np.testing.assert_allclose(module.canonical_overlap(a@a.T,c@c.T,a@c.T),[1,0],atol=1e-14)
    assert module.canonical_overlap(np.ones((2,2)),np.eye(2),np.eye(2)) is None


def test_risk_uses_the_target_outside_the_prefix_and_is_gauge_invariant():
    z=np.array([[0.,2.,-2.],[3.,0.,1.]])
    targets=np.array([2,0])
    expected=-np.log(module.probabilities(z)[np.arange(2),targets])
    np.testing.assert_allclose(module.risk(z,targets),expected)
    np.testing.assert_allclose(module.risk(z+1000,targets),expected,atol=2e-13)


def test_conditional_seed_covariance_excludes_context_means():
    cache={};cases=[]
    # Arbitrarily different input means must contribute no seed susceptibility.
    for h in [4,8,14]:
        for s in range(4):
            cases.append(dict(heads=h,seed=s))
            cache[(h,s,'incoming',0)]={'ids':np.arange(6),
                'common':np.arange(6.)[:,None,None]+np.zeros((6,5,64)),
                'attention':np.arange(6.)[:,None]+np.zeros((6,5)),
                'risk':np.arange(6.)*100}
    rows=module.susceptibilities(cache,dict(cases=cases,times=[0]),np.array([1,1,0,0,-1,-1]),np.array([0,1,0,1,0,1]))
    assert all(r['common_chi']==r['attention_chi']==r['risk_chi']==0 for r in rows)


def test_zero_fluctuation_has_no_logarithmic_exponent():
    rows=[dict(heads=h,time=0,arm='incoming',cohort=c,common_chi=0.,attention_chi=0.,risk_chi=0.)
          for h in [4,8,14] for c in ['all','heldout','heldout-fixed','technical-heldout','narrative-heldout','unassigned-heldout']]
    result=module.size_diagnostics(rows,dict(times=[0]))
    assert all(r['endpoint_slope'] is None and r['adjacent_slopes']==[None,None] for r in result)
    assert all(r['middle_prediction'] is None and r['middle_relative_error'] is None for r in result)
