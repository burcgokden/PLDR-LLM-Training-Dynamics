#!/usr/bin/env python
"""Exact finite enumeration of source transport, memory and data aggregation."""
import argparse
from fractions import Fraction as F
import itertools
from pathlib import Path

from model_rg.provenance import sha256,write_json


def mm(a,b):
    return [[sum((a[i][l]*b[l][j] for l in range(len(b))),F(0)) for j in range(len(b[0]))] for i in range(len(a))]


def transpose(a):return [list(row) for row in zip(*a)]
def scale(a,c):return [[x*c for x in row] for row in a]
def add(a,b):return [[x+y for x,y in zip(u,v)] for u,v in zip(a,b)]
def outer(x):return [[a*b for b in x] for a in x]
def zero():return [[F(0),F(0)],[F(0),F(0)]]


def check_case(m,b,t,r,matrix=False):
    population=[(F(i),F(i*i-i,3)) for i in range(m)]
    mean=[sum((x[j] for x in population),F(0))/m for j in range(2)]
    centered=[[x[j]-mean[j] for j in range(2)] for x in population]
    sigma=scale(sum_matrices([outer(x) for x in centered]),F(1,m))
    weights=[r**(t-1-k) for k in range(t)]
    transport=[[[w,F((-1)**k,3)],[F(k,7),w/2]] if matrix else [[w,F(0)],[F(0),w]]
               for k,w in enumerate(weights)]
    accumulated=zero();zmean=[F(0),F(0)];permutations=0
    for ordering in itertools.permutations(range(m)):
        values=[]
        for k in range(t):
            indices=ordering[k*b:(k+1)*b]
            values.append([sum((centered[i][j] for i in indices),F(0))/b for j in range(2)])
        z=[F(0),F(0)]
        for a,y in zip(transport,values):
            z=[z[j]+sum((a[j][l]*y[l] for l in range(2)),F(0)) for j in range(2)]
        zmean=[x+y for x,y in zip(zmean,z)];accumulated=add(accumulated,outer(z));permutations+=1
        if t==6 and b==1:
            pairs=[[sum((values[2*k+j][l] for j in range(2)),F(0))/2 for l in range(2)] for k in range(3)]
            composed=[sum((q[l] for q in pairs),F(0))/3 for l in range(2)]
            direct=[sum((q[l] for q in values),F(0))/6 for l in range(2)]
            if composed!=direct:raise AssertionError('Data aggregation failed pointwise composition')
    if any(zmean):raise AssertionError('The finite source response mean is not zero')
    empirical=scale(accumulated,F(1,permutations))
    diagonal=sum_matrices([mm(mm(a,sigma),transpose(a)) for a in transport])
    total=sum_matrices(transport)
    prediction=add(scale(diagonal,F(m,b*(m-1))),scale(mm(mm(total,sigma),transpose(total)),F(-1,m-1)))
    if empirical!=prediction:raise AssertionError('Finite permutation transport covariance differs')
    effective=None;factor=None
    if not matrix:
        effective=sum(weights,F(0))**2/sum((w*w for w in weights),F(0))
        formula=F(t) if r==1 else (1+r)/(1-r)*(1-r**t)/(1+r**t)
        if effective!=formula:raise AssertionError('Finite response memory formula differs')
        factor=F(m-b*effective,m-1)
        replacement=scale(sigma,sum((w*w for w in weights),F(0))/b)
        if empirical!=scale(replacement,factor):raise AssertionError('Finite data-resource factor differs')
    return dict(population=m,batch=b,batches=t,retention=str(r),matrix_transport=matrix,
        permutations=permutations,effective_batches=str(effective) if effective is not None else None,
        covariance_factor=str(factor) if factor is not None else None,
        covariance=[[str(x) for x in row] for row in empirical])


def sum_matrices(values):
    result=zero()
    for value in values:result=add(result,value)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='docs/finite-data-response');a=p.parse_args()
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    configurations=[(4,1,1,F(1)),(4,1,2,F(1)),(4,1,4,F(1)),(4,2,1,F(1)),(4,2,2,F(1)),
        (6,2,1,F(1)),(6,2,2,F(1)),(6,2,3,F(1)),(4,1,4,F(1,2)),(6,2,3,F(1,2)),
        (6,1,6,F(3,4)),(5,1,5,F(0))]
    records=[check_case(*c) for c in configurations]
    records.extend(check_case(*c,matrix=True) for c in configurations[6:])
    if len(records)!=18:raise AssertionError('The finite response example inventory changed')
    # A complete pass with constant weights has zero covariance. Retaining
    # replacement noise or dropping the temporal cross covariance is rejected.
    complete=check_case(4,1,4,F(1))
    if complete['covariance']!=[['0','0'],['0','0']] or complete['covariance_factor']!='0':
        raise AssertionError('The complete-pass constant temporal mode is not deterministic')
    write_json(out/'verification.json',dict(status='passed',exact_cases=18,records=records,
        arithmetic='Exact Python Fraction arithmetic over every population permutation in each finite case.',
        source_sha256=sha256(__file__),scope='Selected exact finite cases check the source transport covariance, '
        'exponential-memory factor and pointwise block-aggregation composition. The manuscript provides general '
        'stand-alone proofs. This is not a native optimizer covariance correction or a criticality measurement.'))
    print('Exact finite-data response checks passed:',len(records),'cases',flush=True)


if __name__=='__main__':main()
