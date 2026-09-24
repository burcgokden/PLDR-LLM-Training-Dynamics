import torch
from model_rg.adam_tangent import AdamState,AdamDirection,adam_update,adam_update_tangent,clipping_tangent


def test_augmented_adam_tangent_matches_full_differentiation_with_active_clipping():
    dtype=torch.float64
    p=torch.tensor([2.,-3.,.7],dtype=dtype)
    m=torch.tensor([.01,-.02,.03],dtype=dtype)
    v=torch.tensor([.002,.003,.004],dtype=dtype)
    dp=torch.tensor([.04,.02,-.03],dtype=dtype)
    dm=torch.tensor([.003,-.002,.001],dtype=dtype)
    dv=torch.tensor([.0001,-.0001,.0002],dtype=dtype)
    matrix=torch.tensor([[2.,.2,0.],[.2,1.,.1],[0.,.1,3.]],dtype=dtype)
    state=AdamState((p,),(m,),(v,),37);direction=AdamDirection((dp,),(dm,),(dv,))
    gradient=matrix@p;hessian_direction=matrix@dp
    updated,tangent,meta=adam_update_tangent(state,direction,(gradient,),(hessian_direction,),(.003,))
    assert meta['clip_factor']<1
    def function(alpha):
        shifted=AdamState((p+alpha*dp,),(m+alpha*dm,),(v+alpha*dv,),37)
        value,_=adam_update(shifted,(matrix@shifted.weights[0],),(.003,))
        return torch.cat([*value.weights,*value.first,*value.second])
    _,automatic=torch.func.jvp(function,(torch.zeros((),dtype=dtype),),(torch.ones((),dtype=dtype),))
    torch.testing.assert_close(torch.cat([*tangent.weights,*tangent.first,*tangent.second]),automatic,rtol=2e-12,atol=1e-13)
    for radius in [1e-3,5e-4]:
        finite=(function(torch.tensor(radius,dtype=dtype))-function(torch.tensor(-radius,dtype=dtype)))/(2*radius)
        torch.testing.assert_close(finite,automatic,rtol=3e-7,atol=1e-11)


def test_zero_moment_boundary_has_finite_weight_direction():
    zero=torch.zeros(2,dtype=torch.float64);direction=torch.tensor([.3,-.4],dtype=torch.float64)
    state=AdamState((zero,),(zero,),(zero,),0)
    tangent=AdamDirection((direction,),(zero,),(zero,))
    _,observed,_=adam_update_tangent(state,tangent,(zero,),(direction,),(.001,),epsilon=.1)
    expected=(1-.001*.01)*direction-.001*direction/.1
    torch.testing.assert_close(observed.weights[0],expected,rtol=1e-13,atol=1e-13)


def test_clipping_tangent_retains_gradient_norm_derivative():
    g=torch.tensor([3.,4.],dtype=torch.float64);v=torch.tensor([.2,-.1],dtype=torch.float64)
    _,observed,_,factor=clipping_tangent((g,),(v,))
    assert not torch.allclose(observed[0],factor*v,rtol=1e-3,atol=1e-6)
    def clipped(x):return x/(torch.linalg.vector_norm(x)+1e-6)
    _,expected=torch.func.jvp(clipped,(g,),(v,))
    torch.testing.assert_close(observed[0],expected,rtol=1e-13,atol=1e-13)
