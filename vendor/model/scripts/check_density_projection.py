#!/usr/bin/env python3
"""Independent finite partition, density-energy and chronological averaging checks."""
import argparse
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256, write_json


def check(output,seed=280915,panels=200):
    if output.exists():raise FileExistsError(output)
    rng=np.random.default_rng(seed)
    errors={k:0. for k in ['density','absorption','orthogonal_energy','kl_chain','time_average','signed_increment']}
    for _ in range(panels):
        # Positive floors keep this finite numerical check well conditioned.
        r=rng.uniform(.1,1,37);r/=r.sum()
        p=rng.uniform(.1,1,(6,3,37));p/=p.sum(-1,keepdims=True)
        coarse=np.arange(37)%3;fine=np.arange(37)%9
        def matrix(groups):
            a=np.zeros((37,37))
            for label in set(groups):
                ids=np.flatnonzero(groups==label)
                a[np.ix_(ids,ids)]=r[ids,None]/r[ids].sum()
            return a
        P,Q=matrix(coarse),matrix(fine)
        q=p@Q.T;u=p@P.T;f=p/r
        means=np.empty_like(p)
        for i in range(37):
            ids=np.flatnonzero(coarse==coarse[i])
            means[...,i]=(r[ids]*f[...,ids]).sum(-1)/r[ids].sum()
        residuals=dict(density=u/r-means,absorption=np.concatenate([(Q@P-P).ravel(),(P@Q-P).ravel()]),
            orthogonal_energy=np.sum(r*(f-u/r)**2,-1)-np.sum(r*(f-q/r)**2,-1)-np.sum(r*(q/r-u/r)**2,-1),
            kl_chain=np.sum(p*np.log(p/u),-1)-np.sum(p*np.log(p/q),-1)-np.sum(q*np.log(q/u),-1))
        transition=rng.uniform(.1,1,(6,6));transition/=transition.sum(-1,keepdims=True)
        expected=np.einsum('st,tci->sci',transition,p)
        residuals['time_average']=np.einsum('st,tci->sci',transition,u)-expected@P.T
        residuals['signed_increment']=(p[1:]-p[:-1])@P.T-(u[1:]-u[:-1])
        for name,value in residuals.items():
            error=float(np.max(np.abs(value)))
            if not np.isfinite(value).all() or error>3e-12:raise ValueError(name+' identity failed')
            errors[name]=max(errors[name],error)
    record=dict(status='passed',seed=seed,panels=panels,replicas=6,contexts=3,vocabulary=37,
        maximum_absolute_errors=errors,absolute_tolerance=3e-12,source_sha256=sha256(__file__),
        native_forward_calls=0,training_updates=0,
        scope='Finite synthetic probability panels with nested nontrivial partitions and a stochastic time kernel; numerical identity reconstruction, not native dynamics or a real-arithmetic enclosure.')
    write_json(output,record);return record

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    print(check(p.parse_args().output))
