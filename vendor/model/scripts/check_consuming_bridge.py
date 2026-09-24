#!/usr/bin/env python3
"""Independent exact enumeration and physical-clock controls for finite resources."""
import argparse
import itertools
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm


def formula(u,Ds,B):
    M=len(u);C=u.T@u/M;S=Ds.sum(0)
    return (M/B*sum(D@C@D.T for D in Ds)-S@C@S.T)/(M-1)


def check():
    rng=np.random.default_rng(9163433);cases=[]
    for M,B,T in [(4,1,3),(4,2,2),(6,2,2),(6,3,2)]:
        u=rng.normal(size=(M,3));u-=u.mean(0);D=rng.normal(size=(T,2,3))
        values=[]
        for perm in itertools.permutations(range(M)):
            values.append(sum(D[k]@u[list(perm[k*B:(k+1)*B])].mean(0) for k in range(T)))
        z=np.stack(values);direct=z.T@z/len(z);expected=formula(u,D,B)
        error=float(np.max(abs(direct-expected)))
        if error>1e-12 or np.linalg.eigvalsh(expected).min()<-1e-12:raise ValueError('Finite covariance differs')
        cases.append(dict(population=M,batch=B,steps=T,permutations=len(z),max_error=error))
    full=np.ones((6,1,1));u=np.arange(6.)[:,None]-2.5
    if np.max(abs(formula(u,full,1)))>1e-12:raise ValueError('Full-consumption sum fails')
    clock=[];kappa=.7;B=32;m=128.;tau=1.
    limit=(1-np.exp(-2*kappa*tau))/(2*B*kappa)-(1-np.exp(-kappa*tau))**2/(m*kappa*kappa)
    for N in [4,8,14,24,96,384]:
        h=1/(128*N);M=16384*N;T=128*N
        d=h*np.exp(-kappa*(tau-np.arange(T)*h))
        value=(M/B*np.dot(d,d)-d.sum()**2)/(M-1)/h
        clock.append(dict(heads=N,scaled_covariance=float(value),limit=float(limit),absolute_error=float(abs(value-limit))))
    if not all(clock[i+1]['absolute_error']<clock[i]['absolute_error'] for i in range(len(clock)-1)):raise ValueError('Clock refinement fails')
    return dict(status='passed',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Finite enumeration and deterministic covariance limits; no native training or Gaussian-limit claim.',enumerations=cases,full_consumption_zero=True,clock=clock)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=check();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
