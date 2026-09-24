"""Required native regression with pinned assets: isolation, ownership and precision.
Both adapter classes are tested for lifetime and direct-model execution.
Native gain isolation covers both supported shapes and both capture orders.
"""
from pathlib import Path
import argparse,gc,hashlib,json,sys,time,weakref
import numpy as np
import torch
p=argparse.ArgumentParser()
p.add_argument('--case',choices=['isolation','lifetime','single'],required=True)
p.add_argument('--root',required=True);p.add_argument('--output',required=True)
a=p.parse_args();a.variant='candidate'
root=Path(a.root).resolve();b=Path(a.output).resolve();b.mkdir(parents=True,exist_ok=False)
(b/'evidence').mkdir();out=b
from model_rg.native import NativeModel
from model_rg.training import TrainingModel
from model_rg.precision import preserve_response_dtype
source=root/'assets/PLDR-LLM-v51-SOC-110M-1';torch.set_num_threads(4);start=time.monotonic()
ids=torch.tensor([[2,81,193,62,771,26,59,53],[2,31,83,702,511,11,48,21]]);targets=torch.tensor([81,83]);records=[]
def make(kind):return NativeModel(source,'cpu') if kind=='NativeModel' else TrainingModel(source,2,906042,'cpu')
def digest(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
if a.case=='isolation':
 for layout in ['same-source','same-basename']:
  other=source
  if layout=='same-basename':
   other=b/'scratch/source-alias'/source.name;other.mkdir(parents=True,exist_ok=True)
   for name in ['config.json','configuration_pldrllm.py','modeling_pldrllm.py','model.safetensors']:
    if not (other/name).exists():(other/name).symlink_to(source/name)
  left=NativeModel(source,'cpu');right=NativeModel(other,'cpu')
  with torch.no_grad():
   left.eta=None;right.eta=None;base=right.logits(ids).clone()
   for shape in [(5,14),(2,5,14)]:
    left.eta=None;right.eta=None;shift=left.logits(ids,torch.full(shape,.25)).clone();observed=right.logits(ids).clone()
    rec={'layout':layout,'shape':list(shape),'distinct_module':left.module is not right.module,'cross_instance_max_logit_error':float((observed-base).abs().max()),'intended_max_logit_change':float((shift-base).abs().max())}
    isolated=[]
    for first,second in [(left,right),(right,left)]:
     first.features(ids,targets);saved={k:v.clone() for k,v in first.head_outputs.items()};second.features(ids.flip(0),targets.flip(0));isolated.append(all(torch.equal(v,first.head_outputs[k]) for k,v in saved.items()))
    rec['capture_isolated_both_orders']=all(isolated);records.append(rec)
    assert rec['intended_max_logit_change']>0
    if a.variant!='baseline':assert rec['cross_instance_max_logit_error']==0 and rec['capture_isolated_both_orders'] and rec['distinct_module']
  del left,right;gc.collect()
elif a.case=='single':
 m=make('NativeModel');values={}
 with torch.no_grad():
  values['baseline']=m.logits(ids).numpy().copy()
  for shape in [(5,14),(2,5,14)]:values['eta_'+str(len(shape))]=m.logits(ids,torch.full(shape,.25)).numpy().copy()
  f,d=m.features(ids,targets);values['features']=f.numpy().copy();values.update({k:v.numpy().copy() for k,v in d.items()})
  m.eta=None;m.capture=False
  values['direct_full']=m.model(ids,use_cache=False,logits_to_keep=0).logits.numpy().copy()
 # Check that the existing override still changes the dispatched attention callable.
 preserve_response_dtype(m);m.model.double()
 with torch.no_grad():values['precision_override']=m.logits(ids).numpy().copy()
 eps=torch.zeros((5,14),dtype=torch.float64);direction=torch.ones_like(eps)
 _,derivative=torch.autograd.functional.jvp(lambda e:m.logits(ids,e),eps,direction,strict=False);values['eta_jvp']=derivative.detach().numpy().copy();assert np.linalg.norm(values['eta_jvp'])>0
 np.savez_compressed(out/(a.variant+'-single.npz'),**values)
 records=[{'field':k,'shape':list(v.shape),'sha256':hashlib.sha256(v.tobytes()).hexdigest()} for k,v in values.items()]
else:
 def once(kind,mode):
  obj=make(kind);ar=weakref.ref(obj);mr=weakref.ref(obj.model)
  if mode=='capture':
   with torch.no_grad():obj.features(ids,targets)
  if mode=='direct-survivor':
   model=obj.model;obj.eta=None;obj.capture=False
   with torch.no_grad():before=model(ids,use_cache=False,logits_to_keep=1).logits.clone()
  del obj;gc.collect();rec={'kind':kind,'mode':mode,'adapter_released':ar() is None,'model_released_initial':mr() is None}
  if mode=='direct-survivor':
   with torch.no_grad():after=model(ids,use_cache=False,logits_to_keep=1).logits
   rec['direct_model_output_equal']=torch.equal(before,after)
   if kind=='TrainingModel':
    model.zero_grad(set_to_none=True);result=model(ids,use_cache=False,logits_to_keep=1).logits;result.square().mean().backward();rec['direct_autograd_finite']=all(t.grad is None or torch.isfinite(t.grad).all().item() for t in model.parameters());del result
   del model,before,after;gc.collect();rec['model_released_final']=mr() is None
  else:rec['model_released_final']=mr() is None
  if a.variant=='candidate':
   assert rec['adapter_released'] and rec['model_released_final']
   if mode=='direct-survivor':assert rec['direct_model_output_equal'] and rec.get('direct_autograd_finite',True)
  return rec
 for kind in ['NativeModel','TrainingModel']:
  for mode in ['unused','capture','direct-survivor']:records.append(once(kind,mode))
from model_rg.provenance import sha256
repo=Path(__file__).resolve().parents[1]
result={'status':'passed','variant':a.variant,'case':a.case,'seconds':time.monotonic()-start,'torch':torch.__version__,'python':sys.version,'records':records,'sources':{n:sha256(repo/n) for n in ['scripts/check_native_adapters.py','src/model_rg/native.py','src/model_rg/training.py','src/model_rg/precision.py']},'scope':'Fresh CPU adapter qualification on short deterministic inputs; not a scientific training or population experiment.'}
(b/'evidence'/f'{a.variant}-{a.case}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
