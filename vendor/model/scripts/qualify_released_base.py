"""Native validation and disposable update qualification of released checkpoints."""
from companion_paths import configured_path
import argparse,gc,json,sys,time
from pathlib import Path
import numpy as np
import torch
import sentencepiece as spm
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from model_rg.native import NativeModel
from model_rg.physical_native import selected_forward,prefix_batch,context_tokens,spin_tokens
from model_rg.provenance import sha256,write_json
ROOT=Path(configured_path('data:model'))

@torch.no_grad()
def language(model,data,split='validation',max_documents=None):
    model.model.eval();rows=[];raw={}
    with np.load(data) as z:
        for domain in ['technical','narrative','unassigned']:
            tokens=z[f'{domain}-{split}-tokens'];ids=z[f'{domain}-{split}-ids']
            if max_documents:tokens=tokens[:max_documents];ids=ids[:max_documents]
            for offset in [0,192]:
                for length in [32,64,128]:
                    arrays=[]
                    for begin in range(0,len(tokens),8):
                        x=torch.tensor(tokens[begin:begin+8,offset:offset+length],device=model.device,dtype=torch.long)
                        target=torch.tensor(tokens[begin:begin+8,offset+length],device=model.device,dtype=torch.long)
                        logits=model.forward(x).logits[:,-1].float()
                        lse=torch.logsumexp(logits,-1);tl=logits.gather(1,target[:,None])[:,0]
                        pred=logits.argmax(-1);correct=pred==target
                        top5=logits.topk(5,-1).indices.eq(target[:,None]).any(-1)
                        arrays.append(torch.stack([lse,tl,pred.float(),target.float(),correct.float(),top5.float()],1).cpu().numpy())
                    a=np.concatenate(arrays);key=f'{domain}-o{offset}-p{length}'
                    raw[key]=a;raw[key+'-document-ids']=ids
                    rows.append(dict(domain=domain,offset=offset,prefix=length,documents=len(ids),nll=float((a[:,0]-a[:,1]).mean()),accuracy=float(a[:,4].mean()),top5=float(a[:,5].mean())))
    return rows,raw

@torch.no_grad()
def physical(model,cells,processor,limit=32):
    model.model.eval();alphabet=spin_tokens(processor);rows=[];raw={}
    for c in cells:
        if c['split']!='validation':continue
        q,L=c['q'],c['L'];x=np.asarray(np.memmap(c['path'],mode='r',dtype=np.uint8,shape=tuple(c['shape'])))[:,:4].reshape(-1,L,L)[:limit]
        for fraction in [.25,.5,.875]:
            site=int(L*L*fraction);meta=context_tokens(processor,q,L,c['temperature_ratio'])
            values=[]
            for begin in range(0,len(x),8):
                b=x[begin:begin+8];tokens=prefix_batch(b,site,meta,alphabet[:q],model.device)
                logits,_=selected_forward(model,tokens,alphabet[:q]);target=torch.tensor(b.reshape(len(b),-1)[:,site].astype('int64'),device=model.device)
                nll=torch.nn.functional.cross_entropy(logits,target,reduction='none');correct=logits.argmax(-1)==target
                values.append(torch.cat([logits,target[:,None],nll[:,None],correct[:,None]],1).cpu().numpy())
            a=np.concatenate(values);key=f"cell{c['id']}-j{site}";raw[key]=a
            rows.append(dict(cell=c['id'],q=q,L=L,ratio=c['temperature_ratio'],site=site,nll=float(a[:,-2].mean()),accuracy=float(a[:,-1].mean()),samples=len(a)))
    return rows,raw

def main(study,index,device):
    from model_rg.released_qualification import destination,assets_inventory,data_inventory,validate_design,configure
    study=destination(study);configure();validate_design(study);assets_inventory(study,index);data_inventory(study)
    out=study/'base-qualification'/f'model{index}';out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.cuda.set_device(device);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assets=ROOT/f'assets/PLDR-LLM-v51-SOC-110M-{index}';start=time.perf_counter()
    model=NativeModel(assets,device);torch.cuda.reset_peak_memory_stats(device)
    lang,lr=language(model,study/'data/language.npz');np.savez_compressed(out/'language-validation.npz',**lr)
    print('model',index,'language',np.mean([r['nll'] for r in lang]),np.mean([r['accuracy'] for r in lang]),flush=True)
    cells=json.loads((study/'data/physical.json').read_text())['cells'];processor=spm.SentencePieceProcessor(model_file=str(assets/'tokenizer.model'))
    phys,pr=physical(model,cells,processor);np.savez_compressed(out/'physical-validation.npz',**pr)
    model.model.requires_grad_(True);model.model.eval();alphabet=spin_tokens(processor)
    rng=np.random.default_rng(219011);x=rng.integers(0,3,(4,8,8),dtype=np.uint8)
    inputs=prefix_batch(x,31,context_tokens(processor,3,8,1.),alphabet,device);target=torch.tensor([0,1,2,0],device=device)
    full=model.forward(inputs).logits[:,-1,alphabet];sel,_=selected_forward(model,inputs,alphabet)
    maxerr=float((full-sel).abs().max());torch.testing.assert_close(full,sel,rtol=2e-5,atol=2e-5)
    torch.nn.functional.cross_entropy(full,target).backward();grad={n:p.grad.detach().clone() for n,p in model.model.named_parameters() if p.grad is not None}
    model.model.zero_grad(set_to_none=True);torch.nn.functional.cross_entropy(sel,target).backward()
    num=sum(float((p.grad-grad[n]).double().square().sum()) for n,p in model.model.named_parameters() if p.grad is not None)
    den=sum(float(g.double().square().sum()) for g in grad.values());rel=(num/max(den,1e-30))**.5
    if rel>2e-5:raise ValueError('Selected-output gradient mismatch')
    del grad,full,sel;model.model.zero_grad(set_to_none=True);gc.collect()
    optimizer=torch.optim.AdamW(model.model.parameters(),lr=3e-5,betas=(.9,.95),eps=1e-8,weight_decay=.01)
    profile=[]
    with np.load(study/'data/language.npz') as z:training=z['technical-train-tokens'][:96,:65].copy()
    for step in range(3):
        inp=torch.tensor(training[step*32:(step+1)*32,:64],device=device,dtype=torch.long)
        tar=torch.tensor(training[step*32:(step+1)*32,64],device=device,dtype=torch.long)
        model.model.train();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();t=time.perf_counter()
        logits=model.forward(inp).logits[:,-1];loss=torch.nn.functional.cross_entropy(logits,tar);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.model.parameters(),1.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):raise ValueError('Nonfinite qualification')
        optimizer.step();torch.cuda.synchronize();profile.append(dict(step=step+1,batch=32,tokens=64,loss=float(loss.detach()),seconds=time.perf_counter()-t,source='distinct RefinedWeb documents'))
    write_json(out/'verification.json',dict(status='passed',model=index,assets=str(assets),
        weights_sha256=sha256(assets/'model.safetensors'),tokenizer_sha256=sha256(assets/'tokenizer.model'),native_sha256=sha256(assets/'modeling_pldrllm.py'),
        producer_sha256=sha256(__file__),adapter_sha256=sha256(REPO/'src/model_rg/native.py'),physical_adapter_sha256=sha256(REPO/'src/model_rg/physical_native.py'),
        language_data_sha256=sha256(study/'data/language.npz'),physical_manifest_sha256=sha256(study/'data/physical.json'),
        language=lang,physical=phys,max_logit_error=maxerr,relative_gradient_error=rel,profile=profile,
        qualification_updates=3,scientific_updates=0,seconds=time.perf_counter()-start,peak_gib=torch.cuda.max_memory_allocated(device)/2**30,
        runtime=dict(torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,dtype='float32',tf32=False),
        files={p.name:sha256(p) for p in out.iterdir() if p.is_file()}))
    print(out,profile,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--model',type=int,choices=[1,4,5],required=True);p.add_argument('--device',required=True);a=p.parse_args();main(a.study,a.model,a.device)
