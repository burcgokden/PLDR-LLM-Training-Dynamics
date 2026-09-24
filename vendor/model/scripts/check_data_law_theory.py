#!/usr/bin/env python
"""Exact rational finite checks of the manuscript's data-resource identities."""
import argparse
from collections import Counter
from fractions import Fraction as F
from itertools import combinations, product
from math import factorial
from pathlib import Path

from model_rg.provenance import sha256, write_json


def mean(values):
    return sum(values,F(0))/len(values)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    records=[]
    population=[(0,2),(3,-1),(-2,4),(1,0),(5,3)]
    center=[mean([F(x[j]) for x in population]) for j in range(2)]
    sigma=[[mean([(F(x[i])-center[i])*(F(x[j])-center[j]) for x in population])
            for j in range(2)] for i in range(2)]
    for b in range(1,6):
        values=[[mean([F(population[k][j]) for k in indices]) for j in range(2)]
                for indices in combinations(range(5),b)]
        assert [mean([x[j] for x in values]) for j in range(2)]==center
        observed=[[mean([(x[i]-center[i])*(x[j]-center[j]) for x in values])
                   for j in range(2)] for i in range(2)]
        expected=[[F(5-b,b*4)*sigma[i][j] for j in range(2)] for i in range(2)]
        assert observed==expected
        records.append(dict(identity='finite_population_covariance',population=5,batch=b,
                            enumerated_batches=len(values),factor=str(F(5-b,b*4))))
    h=[F(0),F(2),F(9)];n=[F(1),F(2),F(3)];g=sum(h)/sum(n)
    residual=[h[i]-n[i]*g for i in range(3)]
    for b in range(1,4):
        indices=list(combinations(range(3),b))
        gradients=[sum(h[k] for k in ix)/sum(n[k] for k in ix) for ix in indices]
        minimum_count=min(sum(n[k] for k in ix)/b for ix in indices)
        error=mean([(value-g)**2 for value in gradients])
        bound=F(3-b,b*2)*mean([x*x for x in residual])/minimum_count**2
        assert error<=bound
        records.append(dict(identity='masked_target_ratio',population=3,batch=b,
                            mean_gradient=str(mean(gradients)),population_gradient=str(g),
                            mean_squared_error=str(error),upper_bound=str(bound)))
    assert (F(0)/1+F(2)/2)/2 != (F(0)+F(2))/(1+2)
    for k in range(1,5):
        sequences=list(product(range(4),repeat=k));distinct=[x for x in sequences if len(set(x))==k]
        p0=F(1,len(sequences));q0=F(1,len(distinct))
        tv=sum(abs(p0-(q0 if len(set(x))==k else F(0))) for x in sequences)/2
        expected=1-F(factorial(4)//factorial(4-k),4**k)
        assert tv==expected and tv<=min(F(1),F(k*(k-1),8))
        a_counts=Counter(map(sum,sequences));b_counts=Counter(map(sum,distinct))
        pushed=sum(abs(F(a_counts[i],len(sequences))-F(b_counts[i],len(distinct)))
                   for i in set(a_counts)|set(b_counts))/2
        assert pushed<=tv
        records.append(dict(identity='path_coupling',population=4,draws=k,
                            exact_total_variation=str(tv),observed_sum_total_variation=str(pushed)))
    theta=[F(1),F(-2)];H=[[F(2),F(1)],[F(1),F(3)]]
    increments=[[F(x),F(y)] for x,y in [(1,0),(-1,2),(0,-1),(2,1)]]
    def loss(x):return sum(x[i]*H[i][j]*x[j] for i in range(2) for j in range(2))/2
    drift=[mean([d[i] for d in increments]) for i in range(2)]
    covariance=[[mean([(d[i]-drift[i])*(d[j]-drift[j]) for d in increments])
                 for j in range(2)] for i in range(2)]
    gradient=[sum(H[i][j]*theta[j] for j in range(2)) for i in range(2)]
    change=mean([loss([theta[i]+d[i] for i in range(2)])-loss(theta) for d in increments])
    first=sum(gradient[i]*drift[i] for i in range(2))
    second=(sum(drift[i]*H[i][j]*drift[j] for i in range(2) for j in range(2))+
            sum(H[i][j]*covariance[j][i] for i in range(2) for j in range(2)))/2
    assert change==first+second
    bound=2*mean([sum(x*x for x in d) for d in increments])
    assert abs(change-first)<=bound
    records.append(dict(identity='quadratic_risk_drift',expected_loss_increment=str(change),
                        linear_drift=str(first),quadratic_term=str(second),hessian_norm_upper_bound=4))
    write_json(out/'verification.json',dict(status='passed',exact_finite_cases=len(records),
        records=records,checker_sha256=sha256(__file__),arithmetic='Exact Python rational arithmetic',
        scope='Finite independent enumerations of sampling, target-mask and quadratic-risk identities. These checks supplement the standalone mathematical proofs; they are not proofs for all populations or evidence that a native training critical point exists.'))
    print('Passed',len(records),'exact finite data-law cases',flush=True)


if __name__=='__main__':main()
