"""Exact-rational finite-law check of the proposed nonstationary closure bound."""
from fractions import Fraction as Q
from pathlib import Path
import argparse
from model_rg.provenance import sha256, write_json
import random,json
p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
rng=random.Random(906103)
def distribution(n):
 v=[rng.randint(0,10) for _ in range(n)]
 if not sum(v):v[0]=1
 return [Q(x,sum(v)) for x in v]
def push(x,K):return [sum(a*row[j] for a,row in zip(x,K)) for j in range(len(K[0]))]
def prod(xs):
 y=Q(1)
 for x in xs:y*=x
 return y
records=[]
for trial in range(120):
 m=trial%12+1;rho=distribution(4);coarse=distribution(2);initial=abs(sum(rho[2:])-coarse[1]);defects=[];lips=[];errors=[]
 for t in range(m):
  K=[distribution(4) for _ in range(4)];A=[distribution(2) for _ in range(2)]
  L=abs(A[0][1]-A[1][1]);delta=sum(rho[i]*abs(sum(K[i][2:])-A[i//2][1]) for i in range(4));defects.append(delta);lips.append(L)
  rho=push(rho,K);coarse=push(coarse,A);error=abs(sum(rho[2:])-coarse[1]);bound=initial*prod(lips)+sum(defects[j]*prod(lips[j+1:]) for j in range(t+1));assert error<=bound;errors.append(error)
 E=[distribution(2) for _ in range(4)];F=[distribution(2) for _ in range(2)];LE=abs(F[0][1]-F[1][1]);emission_defect=sum(rho[i]*abs(E[i][1]-F[i//2][1]) for i in range(4));actual=abs(push(rho,E)[1]-push(coarse,F)[1]);bound=emission_defect+LE*(initial*prod(lips)+sum(defects[j]*prod(lips[j+1:]) for j in range(m)));assert actual<=bound
 records.append({'trial':trial,'horizon':m,'observation_error':str(actual),'bound':str(bound),'passed':True})
# A sharp example: fine state always 1, projected reduced state flips 1->0
# with probability d. Average one-step defect d and L=1-d give equality.
for d in [Q(1,10),Q(1,100)]:
 for m in [1,2,8,32]:
  actual=1-(1-d)**m;bound=d*sum((1-d)**j for j in range(m));assert actual==bound
  records.append({'case':'sharp binary example','d':str(d),'horizon':m,'observation_error':str(actual),'bound':str(bound),'equality':True})
out={'status':'passed','arithmetic':'fractions.Fraction exact rational','randomized_inhomogeneous_examples':120,'sharp_examples':8,'random_seed':906103,'records':records,'scope':'Finite binary-observation Markov examples verify the law-averaged bound algebra. They do not estimate a native PLDR closure error or prove an asymptotic theorem.'}
out['checker_sha256']=sha256(__file__);write_json(args.output,out);print('Passed 120 nonstationary exact examples and 8 equality examples.')
