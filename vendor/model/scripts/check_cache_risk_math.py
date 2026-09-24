#!/usr/bin/env python3
"""Independent finite enumeration of calibrated Hilbert risks and dependence corrections."""
import argparse
from itertools import product, combinations
import json
from pathlib import Path
import numpy as np


def check(output):
    if output.exists(): raise FileExistsError(output)
    calibration = np.array([[-1.,2.],[2.,0.],[4.,1.]])
    cp = np.array([.2,.5,.3]); evaluation = np.array([[0.,-1.],[3.,2.]])
    ep = np.array([.4,.6]); mc = cp@calibration; me = ep@evaluation
    vc = float(cp@np.sum((calibration-mc)**2,axis=1))
    ve = float(ep@np.sum((evaluation-me)**2,axis=1)); shift = float(np.sum((me-mc)**2))
    records = []
    for size in [1,2,3,4]:
        for weights in [np.full(size,1/size), np.arange(1,size+1)/sum(range(1,size+1))]:
            exact = 0.
            for ids in product(range(3),repeat=size):
                cache = weights@calibration[list(ids)]; mass = float(np.prod(cp[list(ids)]))
                exact += mass*float(ep@np.sum((evaluation-cache)**2,axis=1))
            theory = ve+shift+vc*float(weights@weights)
            records.append(dict(size=size, weights=weights.tolist(), exact=exact, theory=theory, error=abs(exact-theory)))
    without = []
    m = len(calibration); mean = calibration.mean(0); population = float(np.mean(np.sum((calibration-mean)**2,axis=1)))
    for size in range(1,m+1):
        exact = float(np.mean([np.sum((calibration[list(ids)].mean(0)-mean)**2) for ids in combinations(range(m),size)]))
        theory = population*(m-size)/(size*(m-1))
        without.append(dict(size=size, exact=exact, theory=theory, error=abs(exact-theory)))
    x = np.array([[-3.,1.],[2.,4.],[5.,-2.]]); w = np.array([.1,.3,.6]); c = np.array([1.,-1.]); mu = w@x
    empirical = abs(float(w@np.sum((x-c)**2,axis=1))-float(w@np.sum((x-mu)**2,axis=1)+np.sum((mu-c)**2)))
    dependent = np.array([-1.,1.]); actual = float(np.mean((dependent-dependent)**2))
    corrected = float(2*np.var(dependent)-2*np.var(dependent)); omitted = float(2*np.var(dependent))
    maximum = max([r['error'] for r in records+without]+[empirical,abs(actual-corrected)])
    if maximum > 1e-12 or omitted == actual: raise AssertionError('Finite identity or dependence sensitivity failed')
    result = dict(status='passed', independent_weighted_enumeration=records,
        without_replacement=without, empirical_identity_error=empirical,
        dependence=dict(actual=actual,corrected=corrected,omitted_cross=omitted),
        maximum_absolute_error=maximum,
        scope='Finite synthetic Hilbert identities, including unequal laws, nonuniform weights, and finite-population correction. No native sampling or limit assumption is inferred.')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='passed',maximum_absolute_error=maximum)))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();check(a.output)
