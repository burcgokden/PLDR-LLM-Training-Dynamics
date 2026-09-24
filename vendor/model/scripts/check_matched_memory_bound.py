"""Finite scalar checks of a derived matched-memory Adam force bound, not native training."""
from pathlib import Path
import json,math,argparse,hashlib
import numpy as np
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
if args.output.exists():raise FileExistsError(args.output)
rng=np.random.default_rng(230032)
g1=-1792*math.log(.9);g2=-1792*math.log(.95)
rows=[];random_checks=0;max_violation=0.;saturation_error=0.
for n in [4,8,14,24]:
 T=128*n;h=1/T;b1=math.exp(-g1*h);b2=math.exp(-g2*h);r=b1*b1/b2
 C=(-math.expm1(-g1*h))/math.sqrt((-math.expm1(-g2*h))*(-math.expm1(-(2*g1-g2)*h)))
 for t in sorted(set([1,32,T//4,T])):
  j=np.arange(t);w1=(1-b1)*b1**j;w2=(1-b2)*b2**j
  factor=math.sqrt(1-b2**t)/(1-b1**t)
  Ct=C*math.sqrt((1-b2**t)*(1-r**t))/(1-b1**t)
  gs=rng.normal(size=(256,t))
  forces=np.abs(gs@w1)/np.sqrt((gs*gs)@w2)*factor
  if not np.isfinite(forces).all() or forces.max()>Ct+1e-12:raise ValueError('Bound failed')
  random_checks+=len(gs);max_violation=max(max_violation,float((forces-Ct).max()))
  saturating=(b1/b2)**j
  exact=float(np.dot(w1,saturating)/math.sqrt(np.dot(w2,saturating*saturating))*factor)
  saturation_error=max(saturation_error,abs(exact-Ct))
  if abs(exact-Ct)>1e-11 or Ct>C+1e-12:raise ValueError('Sharp finite factor failed')
 rows.append(dict(heads=n,horizon=T,beta1=b1,beta2=b2,force_envelope=C,physical_memory1=-h/math.log(b1),physical_memory2=-h/math.log(b2),fixed_beta_physical_memory1=-h/math.log(.9),fixed_beta_physical_memory2=-h/math.log(.95)))
C0=g1/math.sqrt(g2*(2*g1-g2))
answer=dict(status='passed',seed=230032,rows=rows,random_gradient_sequences=random_checks,extremizing_sequences=16,maximum_random_bound_violation=max_violation,maximum_extremizer_discrepancy=saturation_error,limiting_envelope=C0,gamma1=g1,gamma2=g2,scope='Float64 finite scalar formula checks. Written proof supplies the uniform-bound argument. No native acquisition, optimizer-family qualification, trained limiting law, or criticality evidence.')
answer['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
args.output.write_text(json.dumps(answer,indent=2,allow_nan=False)+'\n')
print(json.dumps(answer,indent=2))
