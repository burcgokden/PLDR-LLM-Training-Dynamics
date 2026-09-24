"""Real-arithmetic tangent of the declared globally clipped AdamW update.

The loss Hessian-vector product is supplied by the full native graph. This
module retains weight, first-moment and second-moment perturbations. It does
not infer stationary criticality from a finite update Jacobian.
"""
from dataclasses import dataclass
import torch


@dataclass
class AdamState:
    weights: tuple
    first: tuple
    second: tuple
    step: int


@dataclass
class AdamDirection:
    weights: tuple
    first: tuple
    second: tuple


def global_clip(gradients, max_norm=1., offset=1e-6):
    norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in gradients]))
    factor=(max_norm/(norm+offset)).clamp(max=1.)
    return tuple(factor*g for g in gradients),norm,factor


def clipping_tangent(gradients, hessian_direction, max_norm=1., offset=1e-6):
    clipped,norm,factor=global_clip(gradients,max_norm,offset)
    dot=sum((g*h).sum() for g,h in zip(gradients,hessian_direction,strict=True))
    if float(max_norm/(norm+offset)) < 1.:
        derivative=-max_norm*dot/(norm*(norm+offset).square())
    else:
        derivative=torch.zeros_like(norm)
    return clipped,tuple(factor*h+derivative*g for g,h in zip(gradients,hessian_direction,strict=True)),norm,factor


def adam_update(state, gradients, rates, beta1=.9, beta2=.95, epsilon=1e-8, decay=.01):
    clipped,norm,factor=global_clip(gradients)
    counter=state.step+1
    first=tuple(beta1*m+(1-beta1)*g for m,g in zip(state.first,clipped,strict=True))
    second=tuple(beta2*v+(1-beta2)*g.square() for v,g in zip(state.second,clipped,strict=True))
    weights=tuple((1-rate*decay)*p-rate*(m/(1-beta1**counter))/(torch.sqrt(v/(1-beta2**counter))+epsilon)
                  for p,m,v,rate in zip(state.weights,first,second,rates,strict=True))
    return AdamState(weights,first,second,counter),dict(gradient_norm=float(norm),clip_factor=float(factor))


def adam_update_tangent(state, direction, gradients, hessian_direction, rates,
                        beta1=.9, beta2=.95, epsilon=1e-8, decay=.01):
    clipped,change,norm,factor=clipping_tangent(gradients,hessian_direction)
    counter=state.step+1;a1=1-beta1**counter;a2=1-beta2**counter
    next_weights=[];next_first=[];next_second=[];weight_changes=[];first_changes=[];second_changes=[]
    for p,m,v,dp,dm,dv,g,dg,rate in zip(state.weights,state.first,state.second,
            direction.weights,direction.first,direction.second,clipped,change,rates,strict=True):
        m1=beta1*m+(1-beta1)*g;v1=beta2*v+(1-beta2)*g.square()
        dm1=beta1*dm+(1-beta1)*dg;dv1=beta2*dv+2*(1-beta2)*g*dg
        scale=torch.sqrt(v1/a2);denominator=scale+epsilon
        if bool(((scale==0)&(m1!=0)).any()):
            raise ValueError('An inconsistent Adam state has nonzero first moment and zero second moment')
        # At a reachable zero-moment coordinate and for weight-only directions,
        # the denominator contribution vanishes. Nonzero incoming dv is not a
        # two-sided admissible direction at the boundary v=0.
        if bool(((scale==0)&(dv1!=0)).any()):
            raise ValueError('Second-moment direction is not regular at a zero boundary')
        safe_scale=scale.clamp_min(torch.finfo(scale.dtype).tiny)
        scale_change=torch.where(scale>0,dv1/(2*a2*safe_scale),torch.zeros_like(scale))
        ratio=m1/(a1*denominator)
        dratio=dm1/(a1*denominator)-(m1/a1)*scale_change/denominator.square()
        next_weights.append((1-rate*decay)*p-rate*ratio)
        next_first.append(m1);next_second.append(v1)
        weight_changes.append((1-rate*decay)*dp-rate*dratio)
        first_changes.append(dm1);second_changes.append(dv1)
    return (AdamState(tuple(next_weights),tuple(next_first),tuple(next_second),counter),
            AdamDirection(tuple(weight_changes),tuple(first_changes),tuple(second_changes)),
            dict(gradient_norm=float(norm),clip_factor=float(factor)))


def squared_norm(values):
    return sum(x.double().square().sum() for x in values)
