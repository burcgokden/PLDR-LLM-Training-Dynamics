"""Small-corpus language adaptation with a low-rank parameterization of native weights.

Native checkpoint exports merge each rank-four update back into the existing
linear matrix. The comparison with full native inference is checked at every
selected export. Test data never enter training or selection.
"""
from companion_paths import configured_path
import argparse,math,sys,time,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
from model_rg.native import NativeModel
from model_rg.provenance import sha256,write_json
from train_released_adaptation import prepare_draws,language_validation
ROOT=Path(configured_path('data:model'))

class LowRank(nn.Module):
    def __init__(self,base,rank=4):
        super().__init__();self.base=base;self.base.requires_grad_(False)
        self.lora_A=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=base.weight.dtype));nn.init.normal_(self.lora_A,std=.01)
        self.lora_B=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=base.weight.dtype))
    def forward(self,x):return self.base(x)+F.linear(F.linear(x,self.lora_A),self.lora_B)


def inject(model):
    names=[]
    for name,module in list(model.named_modules()):
        if isinstance(module,nn.Linear) and (name.endswith('.wq') or name.endswith('.wv') or name=='final_layer'):
            parent,_,key=name.rpartition('.');setattr(model.get_submodule(parent) if parent else model,key,LowRank(module));names.append(name)
    if len(names)!=11:raise ValueError('Unexpected native low-rank target inventory')
    return names


def merged(model,names):
    state={}
    for k,v in model.state_dict().items():
        if '.lora_' in k:continue
        state[k.replace('.base.','.')]=v.detach().cpu().clone()
    for n in names:
        m=model.get_submodule(n);state[n+'.weight']=(m.base.weight+m.lora_B@m.lora_A).detach().cpu()
    return state


def main(a):
    from model_rg.released_qualification import admit_training,source_inventory
    study,out,qualification,qual=admit_training(a,'factor')
    assets=ROOT/'assets/PLDR-LLM-v51-SOC-110M-5'
    q=dict(weights_sha256=sha256(assets/'model.safetensors'))
    out.mkdir(parents=True,exist_ok=False)
    per_epoch=prepare_draws(study,out,a.task,a.epochs,a.seed);total=a.limit_steps or a.epochs*per_epoch
    bindings=source_inventory()
    spec=dict(schema='released-lowrank-adaptation-v1',name=a.name,task=a.task,model=5,epochs=a.epochs,steps=total,steps_per_epoch=per_epoch,seed=a.seed,learning_rate=a.lr,rank=4,targets='query and value matrices in all five layers, and native vocabulary output matrix',optimizer=dict(name='AdamW',betas=[.9,.95],eps=1e-8,weight_decay=.01,clip_norm=1.,origin='fresh moments on low-rank factors'),batch_size=32,schedule='64-step linear warmup then cosine to 0.2 peak',sources=bindings,draws_sha256=sha256(out/'draws.npz'),base_weights_sha256=q['weights_sha256'],language_data_sha256=sha256(study/'data/language.npz'),qualification=str(qual),qualification_sha256=sha256(qual),role='development' if a.limit_steps else 'primary',selection='minimum validation proper-prefix loss including incoming base; no test use; same fixed 64-prefix dataset and draws as full-parameter comparison')
    write_json(out/'protocol.json',spec)
    for n in bindings:
        dest=out/'executed-source'/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((REPO/n).read_bytes())
    torch.set_num_threads(4);torch.cuda.set_device(a.device);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(a.seed)
    adapter=NativeModel(assets,a.device);base=adapter.model;names=inject(base);params=[p for p in base.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=a.lr,betas=(.9,.95),eps=1e-8,weight_decay=.01);draws=np.load(out/'draws.npz');corpus=np.load(ROOT/'data/refinedweb-onepass-524288/tokens.npy',mmap_mode='r')
    with np.load(study/'data/language.npz') as z:probe=torch.tensor(z['technical-validation-tokens'][:8,:64],device=a.device,dtype=torch.long)
    start=time.perf_counter();observations=[];trace=[];best=math.inf;beststep=0;merge_errors=[]
    torch.cuda.reset_peak_memory_stats(a.device)
    def save(step,filename):
        state=merged(base,names)
        native=NativeModel(assets,a.device);native.model.load_state_dict(state,strict=True)
        with torch.no_grad():
            adapter.model.eval();native.model.eval();left=adapter.forward(probe).logits;right=native.forward(probe).logits
            err=float((left-right).abs().max());torch.testing.assert_close(left,right,rtol=3e-5,atol=3e-5)
        merge_errors.append(dict(step=step,max_logit_error=err));del native
        torch.save(dict(model=state,optimizer=opt.state_dict(),lowrank={n:{'A':base.get_submodule(n).lora_A.detach().cpu(),'B':base.get_submodule(n).lora_B.detach().cpu()} for n in names},step=step,epoch=step/per_epoch,protocol_sha256=sha256(out/'protocol.json')),out/filename)
    def observe(step):
        nonlocal best,beststep
        value=language_validation(adapter,study/'data/language.npz',a.task);observations.append(dict(step=step,epoch=step/per_epoch,**value))
        if value['nll']<best:best=value['nll'];beststep=step;save(step,'best.pt')
        write_json(out/'progress.json',dict(step=step,best_step=beststep,best_validation=best,observations=observations,seconds=time.perf_counter()-start));np.save(out/'trace.npy',np.asarray(trace,np.float64).reshape(-1,8))
        print(a.name,step,round(value['nll'],6),round(value['accuracy'],5),'best',beststep,flush=True)
    observe(0)
    for step in range(total):
        e,i=divmod(step,per_epoch);ex=draws[f'e{e}-examples'][i];x=corpus[ex[:,0,None],ex[:,1,None]*64+np.arange(65)];inp=torch.tensor(x[:,:64],device=a.device,dtype=torch.long);target=torch.tensor(x[:,64],device=a.device,dtype=torch.long)
        rate=a.lr*((step+1)/64 if step<64 else .2+.8*(1+math.cos(math.pi*(step-64)/max(1,total-65)))/2)
        for group in opt.param_groups:group['lr']=rate
        base.train();opt.zero_grad(set_to_none=True);logits=adapter.forward(inp).logits[:,-1];loss=F.cross_entropy(logits,target);loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('Nonfinite low-rank training')
        opt.step();trace.append([step+1,e+1,-1,64,rate,float(loss.detach()),float((logits.argmax(-1)==target).float().mean()),float(norm)])
        if (step+1)%256==0 or (step+1)%per_epoch==0 or step+1==total:observe(step+1)
    save(total,'last.pt')
    if bindings!={n:sha256(REPO/n) for n in bindings}:raise ValueError('Sources changed')
    write_json(out/'result.json',dict(status='complete',name=a.name,task=a.task,model=5,steps=total,steps_per_epoch=per_epoch,epochs_completed=total/per_epoch,selected_step=beststep,selected_epoch=beststep/per_epoch,selected_validation_nll=best,observations=observations,trainable_parameters=sum(p.numel() for p in params),merge_errors=merge_errors,scientific_updates=0 if a.limit_steps else total,development_updates=total if a.limit_steps else 0,protocol_sha256=sha256(out/'protocol.json'),draws_sha256=sha256(out/'draws.npz'),trace_sha256=sha256(out/'trace.npy'),best_sha256=sha256(out/'best.pt'),last_sha256=sha256(out/'last.pt'),seconds=time.perf_counter()-start,peak_gib=torch.cuda.max_memory_allocated(a.device)/2**30))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--name',required=True);p.add_argument('--task',choices=['technical','narrative','mixture','general'],required=True);p.add_argument('--device',required=True);p.add_argument('--epochs',type=int,default=1);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--seed',type=int,default=219100);p.add_argument('--limit-steps',type=int,default=0);main(p.parse_args())
