#!/usr/bin/env python3
"""Independent pair-distance reconstruction of every fixed-cache context policy."""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.special import log_softmax
from numerical_validation import load_json_strict,array_discrepancy
import argparse
parser=argparse.ArgumentParser()
for name in ['study','analysis','output']:parser.add_argument('--'+name,type=Path,required=True)
args=parser.parse_args();STUDY=args.study
if args.output.exists():raise FileExistsError(args.output)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for part in iter(lambda:f.read(8*1024**2),b''):h.update(part)
 return h.hexdigest()
def read(path):return load_json_strict(path.read_text())
p=read(STUDY/'protocol.json');a=read(args.analysis)
if a['protocol_sha256']!=sha(STUDY/'protocol.json'):raise ValueError('Protocol mismatch')
if p['selection_sha256']!=sha(STUDY/'inputs.npz'):raise ValueError('Input mismatch')
expected={(n,s,policy) for n in [8,24] for s in ['initial','g0','g1.5'] for policy in (['state_cache'] if s=='initial' else ['state_cache','initial_cache'])}
if len(a['cells'])!=10 or {(c['heads'],c['state'],c['policy']) for c in a['cells']}!=expected:raise ValueError('Cell coverage')
with np.load(STUDY/'inputs.npz') as z:targets=z['blocks'][:,64]
if targets.shape!=(16,):raise ValueError('Target shape')
inventory={};totals={'assessment':0,'qualification':0};maximum_error=0.;results=[]
for j in p['jobs']:
 folder=STUDY/'runs'/j['run_id'];m=read(folder/'manifest.json')
 if m['status']!='complete' or m['job']!=j or m['protocol_sha256']!=a['protocol_sha256']:raise ValueError('Manifest identity')
 if m['calls']!={'assessment':16,'qualification':6} or m['training_updates']!=0:raise ValueError('Job counts')
 if m['artifacts']!={'logits.npz':sha(folder/'logits.npz')}:raise ValueError('Observation identity')
 for q in m['qualification']:
  if q['own_replay_max_error']!=0 or q['restoration_max_error']!=0:raise ValueError('Qualification difference')
 for k in totals:totals[k]+=m['calls'][k]
 inventory[str((folder/'manifest.json').relative_to(STUDY))]=sha(folder/'manifest.json');inventory[str((folder/'logits.npz').relative_to(STUDY))]=sha(folder/'logits.npz')
for c in a['cells']:
 jobs=sorted([j for j in p['jobs'] if j['heads']==c['heads']],key=lambda j:j['seed'])
 if c['seeds']!=[j['seed'] for j in jobs]:raise ValueError('Seed ordering')
 native=[];cached=[]
 for j in jobs:
  with np.load(STUDY/'runs'/j['run_id']/'logits.npz') as z:
   x=np.asarray(z[c['state']+'-native'],dtype=np.float64);y=np.asarray(z[c['state']+'-'+c['policy']],dtype=np.float64)
   if x.shape!=(16,32000) or y.shape!=x.shape or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Array domain')
   native.append(log_softmax(x,axis=1));cached.append(log_softmax(y,axis=1))
 lp=np.stack(native);lq=np.stack(cached);u=2*np.exp(lp/2);v=2*np.exp(lq/2);d=u-v
 def pair(z):
  out=np.zeros(16)
  for i in range(6):
   for j in range(i):out+=np.einsum('cv,cv->c',z[i]-z[j],z[i]-z[j])/30
  return out
 vp,vq,ve=pair(u),pair(v),pair(d)
 if np.any(vp<=0):raise ValueError('Undefined denominator')
 rc=np.sqrt(ve/vp);rms=float(np.sqrt(ve.sum()/vp.sum()));kl=float(np.mean(np.einsum('scv,scv->sc',np.exp(lp),lp-lq)))
 nll=float(np.mean(lq[:,np.arange(16),targets]*(-1)));nll0=float(np.mean(lp[:,np.arange(16),targets]*(-1)))
 vals=dict(relative_centered_rms=rms,mean_kl=kl,maximum_context_rms=float(rc.max()),native_variance=float(vp.mean()),cached_variance=float(vq.mean()),residual_variance=float(ve.mean()),native_nll=nll0,cached_nll=nll,mean_nll_change=nll-nll0)
 error=array_discrepancy(np.array(list(vals.values())),np.array([c[k] for k in vals]),'fixed-cache result',atol=5e-12);maximum_error=max(maximum_error,error)
 array_discrepancy(rc,np.array(c['per_context_rms']),'fixed-cache contexts',atol=5e-12)
 if c['contexts_over_target']!=int((rc>.25).sum()) or c['aggregate_targets_met']!=bool(rms<=.25 and kl<=.03):raise ValueError('Decision mismatch')
 results.append({'heads':c['heads'],'state':c['state'],'policy':c['policy'],'max_difference':error})
if totals!={'assessment':192,'qualification':72} or a['native_forward_calls']!=264:raise ValueError('Total counts')
r=dict(status='passed',cells=10,maximum_reconstruction_difference=maximum_error,cases=results,calls=totals,analysis_sha256=sha(args.analysis),protocol_sha256=a['protocol_sha256'],verifier_source_sha256=sha(Path(__file__)),raw_artifact_sha256=inventory,scope='Independent CPU reconstruction from every raw logit archive. No imports from acquisition/production reducers, no new native execution.')
args.output.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'status':r['status'],'cells':10,'maximum_difference':maximum_error}))
