"""Finite checks of the conditional covariance statements and their limits.
No native training, population inference, or imported project algebra routines.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse,hashlib,json
import numpy as np
parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
OUT=Path(parser.parse_args().output);OUT.parent.mkdir(parents=True,exist_ok=True)
assert not OUT.exists()
rng=np.random.default_rng(8110902026);errors={};counts={}
def check(name,a,b,atol=5e-10):
 err=float(np.max(np.abs(np.asarray(a)-np.asarray(b))));assert err<atol,(name,err)
 errors[name]=max(errors.get(name,0.),err);counts[name]=counts.get(name,0)+1
def cov(x,w):
 c=x-w@x
 return c.T@(w[:,None]*c)
for dim in [1,2,4,8]:
 for trial in range(32):
  n=41;horizon=4
  w=rng.uniform(.1,1.,n);w/=w.sum()
  joint=rng.standard_t(5,size=(n,(horizon+1)*dim))
  mix=rng.normal(size=((horizon+1)*dim,(horizon+1)*dim))/np.sqrt((horizon+1)*dim)
  joint=joint@mix.T
  q=joint[:,:dim].copy();js=[rng.normal(size=(dim,dim))*.25 for _ in range(horizon)]
  offsets=rng.normal(size=(horizon,dim))
  propagators=[np.eye(dim)]
  for j in range(horizon-1,-1,-1):propagators.insert(0,propagators[0]@js[j])
  for j in range(horizon):q=q@js[j].T+offsets[j]+joint[:,(j+1)*dim:(j+2)*dim]
  L=np.concatenate(propagators,axis=1)
  check('chronological_all_cross_terms',cov(q,w),L@cov(joint,w)@L.T)
  k=max(1,dim//2);A=rng.normal(size=(k,dim));B=rng.normal(size=(1,k))
  check('rectangular_blocking_composition',cov(q@(B@A).T,w),B@(A@cov(q,w)@A.T)@B.T)
  e=rng.normal(size=(n,k))*.1+q@A.T*.2
  x=q@A.T;c=x-w@x;ec=e-w@e
  cross=c.T@(w[:,None]*ec)
  diff=cov(x+e,w)-cov(x,w)
  check('signed_matrix_perturbation',diff,cross+cross.T+cov(e,w))
  trace=float(np.trace(cov(x,w)));rms=float(np.sqrt(np.trace(cov(e,w))))
  assert np.linalg.norm(diff,2)<=2*np.sqrt(trace)*rms+rms*rms+1e-10
  # Finite empirical divisor conversion is not lost by consolidation.
  xx=x-x.mean(0);ee=e-e.mean(0);N=17
  chi=N*np.sum(xx*xx)/(n-1);ce=N*np.sum(ee*ee)/(n-1)
  delta=N*np.sum((xx+ee)**2)/(n-1)-chi
  assert abs(delta)<=2*np.sqrt(chi*ce)+ce+1e-9
  assert ce<=N*n/(n-1)*float(np.mean(np.sum(e*e,axis=1)))+1e-9

def variance(xs):
 m=sum(xs)/len(xs);return sum((x-m)**2 for x in xs)/len(xs)
# Exact-rational counterexamples protect the distinctions in the written proofs.
J=[F(1),F(2)];X=J;Y=[j*x for j,x in zip(J,X)]
correct=variance(Y);illegal=sum(j*j for j in J)/2*variance(X)
assert correct==F(9,4) and illegal==F(5,8)
Z=[F(-1),F(1)]
assert variance([z-z for z in Z])==0 and variance(Z)+variance([-z for z in Z])==2
assert variance([-z for z in Z])==variance(Z) and variance([-2*z for z in Z])==4
assert variance([z+1000 for z in Z])==variance(Z)
# T=[-1,1], B=(1,-1)^T, A=(1,1). Exact A B T={0}; intermediate box gives [-2,2].
assert np.array_equal(np.array([[1,1]])@np.array([[1],[-1]]),[[0]])
result={'status':'passed','seed':8110902026,'finite_examples':128,'identity_checks':counts,'max_absolute_error':errors,'matrix_and_empirical_bounds_checked':256,'exact_rational_controls':{'random_propagator_true_variance':str(correct),'unjustified_unconditional_factorization':str(illegal),'conditional_variance_term':'0','between_conditional_mean_variance':'9/4','anticorrelated_forcing_true_variance':'0','independent_forcing_substitution':'2','reflection_same_variance':True,'deterministic_shift_same_variance':True,'exact_set_composition':'{0}','successive_box_enclosures':'[-2,2]'},'scope':'Finite non-Gaussian algebra/inequality checks and exact counterexamples; not proofs of population or critical asymptotics.','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
