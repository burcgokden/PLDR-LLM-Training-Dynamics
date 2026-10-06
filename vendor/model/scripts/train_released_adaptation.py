"""Proper-prefix fine-tuning of an immutable released PLDR checkpoint.

Selection uses validation only. Test evaluation is a separate executable.
Every epoch and repeated example identity is retained in the frozen draws.
"""
from companion_paths import configured_path
import argparse,gc,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
import sentencepiece as spm
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
from model_rg.native import NativeModel
from model_rg.physical_native import selected_forward,prefix_batch,context_tokens,spin_tokens
from model_rg.provenance import sha256,write_json
from qualify_released_base import physical
ROOT=Path(configured_path('data:model'))
TASKS=['technical','narrative','mixture','general','physical']

def sources():
    from model_rg.released_qualification import source_inventory
    return source_inventory()

def augment(x,q,colors,symmetries):
    out=np.empty_like(x)
    for i,(a,c,s) in enumerate(zip(x,colors,symmetries)):
        a=np.rot90(a,int(s)%4)
        if s>=4:a=np.fliplr(a)
        out[i]=c[a]
    return out

@torch.no_grad()
def language_validation(model,data,task):
    model.model.eval();rows=[]
    domains=['technical','narrative','unassigned']
    with np.load(data) as z:
        for domain in domains:
            tokens=z[f'{domain}-validation-tokens'];nll=[];correct=[]
            for offset in [0,192]:
                for begin in range(0,len(tokens),16):
                    x=torch.tensor(tokens[begin:begin+16,offset:offset+64],device=model.device,dtype=torch.long)
                    target=torch.tensor(tokens[begin:begin+16,offset+64],device=model.device,dtype=torch.long)
                    logits=model.forward(x).logits[:,-1]
                    nll.extend(torch.nn.functional.cross_entropy(logits,target,reduction='none').cpu().tolist());correct.extend((logits.argmax(-1)==target).cpu().tolist())
            rows.append(dict(domain=domain,nll=float(np.mean(nll)),accuracy=float(np.mean(correct)),samples=len(nll)))
    selected=[r for r in rows if (task not in ['technical','narrative'] or r['domain']==task) and (task!='mixture' or r['domain']!='unassigned')]
    return dict(nll=float(np.mean([r['nll'] for r in selected])),accuracy=float(np.mean([r['accuracy'] for r in selected])),rows=rows)

def prepare_draws(study,out,task,epochs,seed):
    rng=np.random.default_rng(seed);arrays={}
    if task=='physical':
        cells=[c for c in json.loads((study/'data/physical.json').read_text())['cells'] if c['split']=='train']
        for epoch in range(epochs):
            cellids=[];sampleids=[];sites=[];colors=[];sym=[]
            for c in cells:
                n=c['chains']*c['samples_per_chain'];ids=rng.permutation(n).reshape(-1,32)
                for b in ids:
                    cellids.append(c['id']);sampleids.append(b);sites.append(int(rng.integers(c['L']**2)))
                    p=np.tile(np.arange(3),(32,1));p[:,:c['q']]=rng.random((32,c['q'])).argsort(1)
                    colors.append(p);sym.append(rng.integers(8,size=32))
            order=rng.permutation(len(cellids))
            for key,value in [('cell',cellids),('sample',sampleids),('site',sites),('colors',colors),('symmetry',sym)]:arrays[f'e{epoch}-{key}']=np.asarray(value,dtype=np.int64)[order]
        count=len(cellids)
    else:
        if task=='technical':counts={'technical':4096}
        elif task=='narrative':counts={'narrative':4096}
        elif task=='mixture':counts={'technical':2048,'narrative':2048}
        else:counts={'technical':151,'narrative':2122,'unassigned':1823}
        docs=[]
        with np.load(study/'data/language.npz') as z:
            for domain,n in counts.items():
                ids=z[f'{domain}-train-ids'];docs.extend(ids[rng.permutation(len(ids))[:n]].tolist())
        examples=np.array([(d,b) for d in docs for b in range(8)],dtype=np.int64)
        for epoch in range(epochs):arrays[f'e{epoch}-examples']=examples[rng.permutation(len(examples))].reshape(-1,32,2)
        count=len(examples)//32
    np.savez_compressed(out/'draws.npz',**arrays)
    return count

def main(a):
    from model_rg.released_qualification import admit_training,source_inventory
    study,out,qualification,qual=admit_training(a,'full')
    assets=ROOT/f'assets/PLDR-LLM-v51-SOC-110M-{a.model}'
    q=dict(weights_sha256=sha256(assets/'model.safetensors'))
    out.mkdir(parents=True,exist_ok=False);per_epoch=prepare_draws(study,out,a.task,a.epochs,a.seed)
    total=a.limit_steps if a.limit_steps else a.epochs*per_epoch
    if total>a.epochs*per_epoch:raise ValueError('Budget exceeds frozen epochs')
    spec=dict(schema='released-adaptation-training-v1',status='frozen',task=a.task,name=a.name,model=a.model,assets=str(assets),
        weights_sha256=q['weights_sha256'],qualification=str(qual),qualification_sha256=sha256(qual),epochs=a.epochs,
        steps_per_epoch=per_epoch,steps=total,seed=a.seed,batch_size=32,learning_rate=a.lr,warmup=min(64,max(16,total//8)),
        optimizer=dict(name='AdamW',betas=[.9,.95],eps=1e-8,weight_decay=.01,clip_norm=1.,origin='fresh zero moments; released optimizer unavailable'),
        schedule='linear warmup then cosine to 0.2 of peak',selection='minimum task-specific validation cross entropy; evaluate every 256 steps and at epoch boundaries; final test is separate',
        role='development' if a.limit_steps else 'primary',sources=sources(),draws_sha256=sha256(out/'draws.npz'),
        data_manifest_sha256=sha256(study/('data/physical.json' if a.task=='physical' else 'data/language.json')),
        language_data_sha256=sha256(study/'data/language.npz') if a.task!='physical' else None,
        runtime=dict(torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,dtype='float32',tf32=False),
        law='Physical: fresh target and symmetry draw on each repeat of a finite configuration. Language: each of 32768 fixed proper-prefix examples once per epoch. Validation and test identities excluded from training.')
    write_json(out/'protocol.json',spec)
    for n in spec['sources']:
        dest=out/'executed-source'/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((REPO/n).read_bytes())
    torch.set_num_threads(4);torch.cuda.set_device(a.device);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(a.seed)
    model=NativeModel(assets,a.device);model.model.requires_grad_(True)
    optimizer=torch.optim.AdamW(model.model.parameters(),lr=a.lr,betas=(.9,.95),eps=1e-8,weight_decay=.01)
    processor=spm.SentencePieceProcessor(model_file=str(assets/'tokenizer.model'));alphabet=spin_tokens(processor)
    draws=np.load(out/'draws.npz');cells=json.loads((study/'data/physical.json').read_text())['cells'];cellmap={c['id']:c for c in cells};maps={}
    if a.task=='physical':
        for c in cells:
            if c['split']=='train':
                if sha256(c['path'])!=c['sha256']:raise ValueError('Physical source changed')
                maps[c['id']]=np.memmap(c['path'],mode='r',dtype=np.uint8,shape=tuple(c['shape'])).reshape(-1,c['L'],c['L'])
    else:corpus=np.load(ROOT/'data/refinedweb-onepass-524288/tokens.npy',mmap_mode='r')
    def observe():
        if a.task!='physical':return language_validation(model,study/'data/language.npz',a.task)
        rows,_=physical(model,cells,processor)
        return dict(nll=float(np.mean([r['nll'] for r in rows])),accuracy=float(np.mean([r['accuracy'] for r in rows])),rows=rows)
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats(a.device);observations=[];trace=[]
    initial=observe();observations.append(dict(step=0,epoch=0.,**initial));best=initial['nll'];best_step=0
    print(a.name,'initial',initial['nll'],initial['accuracy'],flush=True)
    def save(step,name):
        torch.save(dict(model=model.model.state_dict(),optimizer=optimizer.state_dict(),step=step,epoch=step/per_epoch,
            protocol_sha256=sha256(out/'protocol.json'),sources=spec['sources']),out/name)
    save(0,'best.pt')
    for step in range(total):
        epoch=step//per_epoch;i=step%per_epoch
        rate=a.lr*((step+1)/spec['warmup'] if step<spec['warmup'] else .2+.8*(1+math.cos(math.pi*(step-spec['warmup'])/max(1,total-spec['warmup']-1)))/2)
        for g in optimizer.param_groups:g['lr']=rate
        model.model.train();optimizer.zero_grad(set_to_none=True)
        if a.task=='physical':
            cid=int(draws[f'e{epoch}-cell'][i]);c=cellmap[cid];site=int(draws[f'e{epoch}-site'][i]);ids=draws[f'e{epoch}-sample'][i]
            x=augment(np.asarray(maps[cid][ids]),c['q'],draws[f'e{epoch}-colors'][i],draws[f'e{epoch}-symmetry'][i])
            inp=prefix_batch(x,site,context_tokens(processor,c['q'],c['L'],c['temperature_ratio']),alphabet[:c['q']],a.device)
            target=torch.tensor(x.reshape(32,-1)[:,site].astype('int64'),device=a.device);micro=16 if c['L']>=16 else 32
        else:
            ex=draws[f'e{epoch}-examples'][i];x=corpus[ex[:,0,None],ex[:,1,None]*64+np.arange(65)]
            inp=torch.tensor(x[:,:64],device=a.device,dtype=torch.long);target=torch.tensor(x[:,64],device=a.device,dtype=torch.long);micro=32;cid=-1;site=64
        loss_value=0.;correct=0
        for begin in range(0,32,micro):
            if a.task=='physical':logits,_=selected_forward(model,inp[begin:begin+micro],alphabet[:c['q']])
            else:logits=model.forward(inp[begin:begin+micro]).logits[:,-1]
            loss=torch.nn.functional.cross_entropy(logits,target[begin:begin+micro])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training objective')
            (loss*(micro/32)).backward();loss_value+=float(loss.detach())*micro/32;correct+=int((logits.argmax(-1)==target[begin:begin+micro]).sum())
        norm=torch.nn.utils.clip_grad_norm_(model.model.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
        optimizer.step();trace.append([step+1,epoch+1,cid,site,rate,loss_value,correct/32,float(norm)])
        if (step+1)%256==0 or (step+1)%per_epoch==0 or step+1==total:
            value=observe();observations.append(dict(step=step+1,epoch=(step+1)/per_epoch,**value))
            if value['nll']<best:
                best=value['nll'];best_step=step+1;save(best_step,'best.pt')
            write_json(out/'progress.json',dict(step=step+1,best_step=best_step,best_validation=best,observations=observations,seconds=time.perf_counter()-start))
            np.save(out/'trace.npy',np.asarray(trace,np.float64))
            print(a.name,'step',step+1,'epoch',round((step+1)/per_epoch,3),'train',round(loss_value,5),'val',round(value['nll'],5),'acc',round(value['accuracy'],5),'best',best_step,'seconds',round(time.perf_counter()-start,1),flush=True)
    save(total,'last.pt')
    if sources()!=spec['sources']:raise ValueError('Training sources changed during execution')
    result=dict(status='complete',name=a.name,task=a.task,model=a.model,steps=total,steps_per_epoch=per_epoch,epochs_completed=total/per_epoch,
        selected_step=best_step,selected_epoch=best_step/per_epoch,selected_validation_nll=best,observations=observations,
        scientific_updates=0 if a.limit_steps else total,development_updates=total if a.limit_steps else 0,
        protocol_sha256=sha256(out/'protocol.json'),trace_sha256=sha256(out/'trace.npy'),draws_sha256=sha256(out/'draws.npz'),
        best_sha256=sha256(out/'best.pt'),last_sha256=sha256(out/'last.pt'),seconds=time.perf_counter()-start,peak_gib=torch.cuda.max_memory_allocated(a.device)/2**30)
    write_json(out/'result.json',result);print(out,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--name',required=True);p.add_argument('--task',choices=TASKS,required=True);p.add_argument('--model',type=int,choices=[1,4,5],default=5);p.add_argument('--device',required=True);p.add_argument('--lr',type=float,required=True);p.add_argument('--epochs',type=int,default=1);p.add_argument('--seed',type=int,default=219020);p.add_argument('--limit-steps',type=int,default=0);main(p.parse_args())
