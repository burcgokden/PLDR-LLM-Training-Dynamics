"""Exact covariance RG for a finite set of retained geometric relaxation modes."""
import numpy as np


def geometric_block(c0, shared, tails, poles, block, exponent=.5):
    if int(block)!=block or block<1:raise ValueError('Positive integer block required')
    c0,shared,tails,poles=map(lambda x:np.asarray(x,dtype=float),(c0,shared,tails,poles))
    if np.any((poles<=0)|(poles>=1)):raise ValueError('Poles must be in (0,1)')
    if tails.shape!=(len(poles),*c0.shape) or c0.shape!=shared.shape:raise ValueError('Covariance dimensions differ')
    b=int(block);scale=b**(-2*exponent)
    zero=b*c0+b*(b-1)*shared
    for r in range(1,b):
        zero=zero+(b-r)*np.einsum('k,kij->ij',poles**(r-1),tails+tails.transpose(0,2,1))
    weights=np.array([sum(rho**i for i in range(b)) for rho in poles])**2
    return scale*zero,b*b*scale*shared,scale*weights[:,None,None]*tails,poles**b


def geometric_lag(c0,shared,tails,poles,lag):
    if lag==0:return np.asarray(c0)
    if lag<0:return geometric_lag(c0,shared,tails,poles,-lag).T
    return shared+np.einsum('k,kij->ij',poles**(lag-1),tails)
