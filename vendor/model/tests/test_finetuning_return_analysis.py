import numpy as np
import scripts.analyze_finetuning_return as module


def test_transport_predicts_independent_inputs_under_true_coupled_map():
    rng=np.random.default_rng(6310);x=rng.normal(size=(60,2))
    a=np.array([[.8,.2],[-.3,1.1]]);y=x@a
    g=lambda p,q:p.T@q/len(p)
    row=module.transport(g(x[:30],x[:30]),g(x[:30],y[:30]),g(x[30:],x[30:]),g(y[30:],y[30:]),g(x[30:],y[30:]))
    np.testing.assert_allclose(row['matrix'],a,atol=1e-14)
    assert row['relative_residual']<3e-8
    assert row['weak_lower_bound_positive']
    np.testing.assert_allclose(row['canonical_cosines'],[1,1],atol=1e-12)


def test_rank_loss_does_not_certify_inherited_weak_sector():
    x=np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
    y=x@np.diag([1.,0.]);g=lambda p,q:p.T@q/len(p)
    row=module.transport(g(x,x),g(x,y),g(x,x),g(y,y),g(x,y))
    assert not row['weak_lower_bound_positive']
    assert row['singular_values'][1]==0
    assert row['canonical_cosines'] is None


def test_new_orthogonal_response_contributes_to_inheritance_error():
    x=np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
    e=np.array([[2.,0.],[2.,0.],[0.,2.],[0.,2.]])
    y=x+e;g=lambda p,q:p.T@q/len(p)
    row=module.transport(g(x,x),g(x,y),g(x,x),g(y,y),g(x,y))
    np.testing.assert_allclose(row['matrix'],np.eye(2))
    np.testing.assert_allclose(row['residual_rms'],2.)
    assert row['residual_to_weak_incoming']>1
    assert not row['weak_lower_bound_positive']


def test_expanded_dictionary_recovers_a_missing_effect_on_new_inputs():
    rng=np.random.default_rng(182);x=rng.normal(size=(100,3))
    a=np.array([[.8,.1],[.3,.2],[.7,1.1]]);y=x@a
    g=lambda p,q:p.T@q/len(p)
    small=module.dictionary_transport(g(x[:50,:2],x[:50,:2]),g(x[:50,:2],y[:50]),g(x[50:,:2],x[50:,:2]),g(y[50:],y[50:]),g(x[50:,:2],y[50:]))
    full=module.dictionary_transport(g(x[:50],x[:50]),g(x[:50],y[:50]),g(x[50:],x[50:]),g(y[50:],y[50:]),g(x[50:],y[50:]))
    np.testing.assert_allclose(full['matrix'],a,atol=1e-14)
    assert full['relative_residual']<3e-8
    assert small['relative_residual']>.3
