#!/usr/bin/env python
"""Reconstruct the fixed-mean score decomposition and empirical covariance hierarchy.
No model is fitted and no new independent observation is claimed.
"""
import argparse
from pathlib import Path
import hashlib, json, time
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--output',required=True)
a=p.parse_args();STUDY=Path(a.study).resolve();output=Path(a.output).resolve()
if output.exists():raise FileExistsError(output)
output.parent.mkdir(parents=True,exist_ok=True)
start=time.monotonic();inputs={}
def load(p):
 inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 return json.loads(p.read_text())
def moments(x):
 m=x.mean(0);z=x-m
 return m,z.T@z/len(x)
spec=load(STUDY/'protocol.json');frozen=load(STUDY/'frozen-predictions.json');reported=load(STUDY/'analysis.json')
rows=[];paths=[];max_score_error=0.;max_hierarchy_error=0.
for case in spec['cases']:
 name=case['name'];run=STUDY/'runs'/name;meta=load(run/'results.json');values=[]
 for record in meta['records']:
  raw=run/record['raw'];inputs[str(raw)]=hashlib.sha256(raw.read_bytes()).hexdigest()
  assert inputs[str(raw)]==record['sha256']
  with np.load(raw,allow_pickle=False) as a:
   ix=[list(a['horizons']).index(h) for h in [0,1,4,16,64]]
   values.append(np.stack([a['nll'][ix].mean(-1),a['entropy'][ix].mean(-1)],axis=-1))
 values=np.asarray(values);assert values.shape==(64,5,2);paths.append(values)
 for target,nobs in [('risk',1),('risk_entropy',2)]:
  fit=frozen['fits'][name]['targets'][target];mu=np.array(fit['mean']);cov=np.array(fit['covariance'])
  var=np.maximum(np.diag(cov),1e-12)
  reg=.75*cov+.25*np.diag(np.diag(cov))+.001*np.diag(var)
  diagonal=np.diag(np.diag(cov))+.001*np.diag(var)
  x=np.concatenate([np.diff(values[:,:,j],axis=1) for j in range(nobs)],axis=1)
  for tag,base in [('fine',np.eye(4)),('paired',np.array([[1,1,0,0],[0,0,1,1]])),('endpoint',np.ones((1,4)))]:
   b=np.kron(np.eye(nobs),base);y=x@b.T;ym,cpop=moments(y);bias=ym-b@mu
   r=b@reg@b.T;d=b@diagonal@b.T
   ir=np.linalg.solve(r,np.eye(len(r)));id_=np.linalg.solve(d,np.eye(len(d)));precision=ir-id_
   logdet=float(np.linalg.slogdet(r)[1]-np.linalg.slogdet(d)[1])
   covariance=float(np.trace(precision@cpop));mean=float(bias@precision@bias)
   total=logdet+covariance+mean
   reference=next(k for k in reported['records'] if (k['case'],k['target'],k['scale'])==(name,target,tag))
   error=abs(total-reference['score_difference']);max_score_error=max(max_score_error,error);assert error<1e-9
   rows.append(dict(case=name,heads=case['heads'],seed=case['seed'],step=case['step'],target=target,scale=tag,logdet=logdet,covariance=covariance,mean_error=mean,total=total,reported_difference=reference['score_difference']))
# Equal-weight empirical distribution over the eight retained states, then 64 branches.
# Population divisors are essential: this is an exact finite mixture, not an unbiased outer-law estimate.
paths=np.asarray(paths);hierarchy=[]
for target,nobs in [('risk',1),('risk_entropy',2)]:
 increments=np.concatenate([np.diff(paths[:,:,:,j],axis=2) for j in range(nobs)],axis=2)
 for scale,base in [('fine',np.eye(4)),('paired',np.array([[1,1,0,0],[0,0,1,1]])),('endpoint',np.ones((1,4)))]:
  b=np.kron(np.eye(nobs),base);x=increments@b.T
  state_means=np.mean(x,axis=1);within=np.mean([moments(y)[1] for y in x],axis=0)
  _,between=moments(state_means);_,total=moments(x.reshape(-1,x.shape[-1]))
  error=float(np.max(np.abs(within+between-total)));max_hierarchy_error=max(max_hierarchy_error,error)
  hierarchy.append(dict(target=target,scale=scale,within=within.tolist(),between=between.tolist(),total=total.tolist(),between_trace_fraction=float(np.trace(between)/np.trace(total)),error=error))
_,between_endpoint=moments(paths[:,:,-1,0].mean(1)[:,None])
within_endpoint=np.mean([moments(x[:,-1,0,None])[1] for x in paths],axis=0)
_,total_endpoint=moments(paths[:,:,-1,0].reshape(-1,1))
assert np.allclose(within_endpoint+between_endpoint,total_endpoint,atol=1e-12)
# Transparent finite Gaussian-moment example: rankings need not survive a projection.
def regret(diagonal):
 d=np.asarray(diagonal);return float(np.sum(np.log(d)+1/d-1))
example=dict(truth_covariance=[1,1],forecast_a=[1,4],forecast_b=[2,1],full_regret_a=regret([1,4]),full_regret_b=regret([2,1]),first_coordinate_regret_a=regret([1]),first_coordinate_regret_b=regret([2]))
assert example['full_regret_b']<example['full_regret_a'] and example['first_coordinate_regret_a']<example['first_coordinate_regret_b']
summary={}
for target in ['risk','risk_entropy']:
 for scale in ['fine','paired','endpoint']:
  rr=[r for r in rows if r['target']==target and r['scale']==scale]
  summary[target+'__'+scale]={key:float(np.mean([r[key] for r in rr])) for key in ['logdet','covariance','mean_error','total']}
report=dict(status='passed',description='Exploratory finite reductions using retained 512 paths, unchanged frozen forecasts and all 48 cells. No new independent observations or population covariance certificates.',sample_divisor=64,states=8,paths=512,max_score_decomposition_error=max_score_error,max_hierarchical_covariance_error=max_hierarchy_error,summary=summary,rows=rows,hierarchy=hierarchy,endpoint_level_mixture=dict(within=float(within_endpoint[0,0]),between=float(between_endpoint[0,0]),total=float(total_endpoint[0,0]),between_fraction=float(between_endpoint[0,0]/total_endpoint[0,0])),analytic_ranking_example=example,inputs_sha256=inputs,seconds=time.monotonic()-start)
report['analyzer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['status','max_score_decomposition_error','max_hierarchical_covariance_error','summary','endpoint_level_mixture']},indent=2))
