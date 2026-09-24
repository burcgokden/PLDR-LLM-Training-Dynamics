#!/usr/bin/env python3
"""Executed finite clock/refinement and without-replacement resource controls."""
import argparse
import math
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from model_rg.provenance import sha256, write_json


def check(output):
    if output.exists():
        raise FileExistsError(output)
    kappa = .7
    rows = []
    for n in [4,8,14,24,96,384]:
        h = 1/(128*n); lam = math.exp(-kappa*h)
        update = -1/math.log(lam); physical = h*update
        # Propagation for one physical unit is evaluated as a discrete map.
        transition = np.linalg.matrix_power(np.array([[lam]]),128*n)[0,0]
        error = abs(transition-expm(np.array([[-kappa]]))[0,0])
        if abs(physical-1/kappa)>1e-9 or error>1e-10:
            raise ValueError('Physical clock comparison failed')
        innovation = -math.expm1(-2*kappa*h)/(2*kappa)
        stationary = innovation/(-math.expm1(-2*kappa*h))
        if abs(stationary-1/(2*kappa))>1e-14:
            raise ValueError('Clock-invariant covariance failed')
        m=16*32*128*n
        rows.append(dict(heads=n,step=h,eigenvalue=lam,update_relaxation=update,
            physical_relaxation=physical,physical_rate=-math.log(lam)/h,
            one_unit_response=float(transition),semigroup_difference=error,
            innovation_variance=innovation,stationary_variance=stationary,
            resource_blocks=m,remaining_blocks=15*m//16,
            linear_window_collision_ratio=(32*128*n)**2/(15*m/16)))
    # Compare exact index-law total variation with the pair-collision bound.
    resource=[]
    for m in [64,128,512,2048]:
        for k in [2,4,8,16]:
            exact=-math.expm1(sum(math.log1p(-j/m) for j in range(k)))
            union=min(1.,k*(k-1)/(2*m))
            if not 0<=exact<=union+1e-14:
                raise ValueError('Finite resource envelope failed')
            resource.append(dict(blocks=m,draws=k,total_variation=exact,union_bound=union))
    write_json(output,dict(status='passed',schema='physical-clock-resource-v1',
        source_sha256=sha256(__file__),decay=kappa,rows=rows,resource_controls=resource,
        native_training_updates=0,native_forward_calls=0,
        maximum_physical_relaxation_error=max(abs(x['physical_relaxation']-1/kappa) for x in rows),
        scope='Analytic scalar and finite sampling controls; no native critical exponent is inferred.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    check(p.parse_args().output)
