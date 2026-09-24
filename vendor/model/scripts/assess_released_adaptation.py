"""Held-out proper-prefix and generated-law assessment of selected released-model adaptations.

Selection is complete before test inference. Test configurations are never used for
checkpoint choice. Native generation recomputes the entire proper prefix without a cache.
"""
from companion_paths import legacy_path
import argparse,itertools,json,sys,time
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
from model_rg.native import NativeModel
from model_rg.physical_native import selected_forward,prefix_batch,context_tokens,spin_tokens,generate
from model_rg.provenance import sha256,write_json
from model_rg.lattice import observables,critical_temperature
from qualify_released_base import language
ROOT=Path(legacy_path('/pldr-data/model'))

def sources():
    names=['scripts/assess_released_adaptation.py','scripts/qualify_released_base.py','src/model_rg/native.py','src/model_rg/physical_native.py','src/model_rg/lattice.py']
    return {n:sha256(REPO/n) for n in names}

def load_model(study,name,device):
    from model_rg.released_qualification import admit_assessment
    qualified,qual,identity=admit_assessment(study,name)
    assets=ROOT/'assets/PLDR-LLM-v51-SOC-110M-5'
    q=dict(weights_sha256=sha256(assets/'model.safetensors'))
    model=NativeModel(assets,device)
    identity.update(base_weights=q['weights_sha256'],qualification_sha256=sha256(qual))
    if name!='base5':
        root=study/'runs'/name;result=json.loads((root/'result.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
        if result['status']!='complete' or sha256(root/'best.pt')!=result['best_sha256'] or sha256(root/'protocol.json')!=result['protocol_sha256']:raise ValueError('Incomplete or changed checkpoint')
        state=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
        if state['step']!=result['selected_step'] or state['protocol_sha256']!=result['protocol_sha256']:raise ValueError('Selection mismatch')
        model.model.load_state_dict(state['model'],strict=True);del state
        identity.update(checkpoint_sha256=result['best_sha256'],result_sha256=sha256(root/'result.json'),step=result['selected_step'],epoch=result['selected_epoch'],task=protocol['task'])
    else:identity.update(step=0,epoch=0.,checkpoint_sha256=q['weights_sha256'])
    model.model.requires_grad_(False).eval()
    return model,identity

def test_plan(study):
    cells=json.loads((study/'data/physical.json').read_text())['cells'];rng=np.random.default_rng(219301)
    plans=[]
    for c in cells:
        if c['split']!='test':continue
        V=c['L']**2;edges=np.linspace(0,V,9,dtype=int)
        sites=[int(rng.integers(edges[i],edges[i+1])) for i in range(8)]
        # Eight sites sampled independently in equal spatial quantile strata;
        # all listed sizes have volume divisible by eight except L=6 (handled by weights).
        weights=np.diff(edges)/V
        plans.append(dict(cell=c['id'],q=c['q'],L=c['L'],ratio=c['temperature_ratio'],sites=sites,site_weights=weights.tolist(),samples_per_chain=8,chains=16))
    return plans

@torch.no_grad()
def spin_scores(model,study,out,processor):
    cells={c['id']:c for c in json.loads((study/'data/physical.json').read_text())['cells']};alphabet=spin_tokens(processor);rows=[];raw={}
    for plan in test_plan(study):
        c=cells[plan['cell']];q,L=c['q'],c['L']
        if sha256(c['path'])!=c['sha256']:raise ValueError('Changed test data')
        x=np.asarray(np.memmap(c['path'],mode='r',dtype=np.uint8,shape=tuple(c['shape'])))[:,:8].reshape(-1,L,L)
        flat=x.reshape(len(x),-1)
        for site,weight in zip(plan['sites'],plan['site_weights']):
            vals=[]
            for begin in range(0,len(x),16):
                b=x[begin:begin+16];inp=prefix_batch(b,site,context_tokens(processor,q,L,c['temperature_ratio']),alphabet[:q],model.device)
                logits,_=selected_forward(model,inp,alphabet[:q]);target=torch.tensor(b.reshape(len(b),-1)[:,site].astype('int64'),device=model.device)
                nll=torch.nn.functional.cross_entropy(logits,target,reduction='none');ok=logits.argmax(-1)==target
                vals.append(torch.cat([logits,target[:,None],nll[:,None],ok[:,None]],1).cpu().numpy())
            a=np.concatenate(vals);key=f"cell{c['id']}-j{site}";raw[key]=a
            # Dirichlet-half prefix histogram baseline, no fitted parameters.
            counts=np.stack([(flat[:,:site]==color).sum(1) for color in range(q)],1);p=(counts+.5)/(site+.5*q)
            target=flat[:,site];hist_nll=-np.log(p[np.arange(len(p)),target]);hist_acc=(p.argmax(1)==target)
            raw[key+'-histogram']=np.stack([hist_nll,hist_acc],1)
            rows.append(dict(cell=c['id'],q=q,L=L,ratio=c['temperature_ratio'],site=site,site_weight=weight,nll=float(a[:,-2].mean()),accuracy=float(a[:,-1].mean()),histogram_nll=float(hist_nll.mean()),histogram_accuracy=float(hist_acc.mean()),samples=len(a),raw=key))
        print('spin',c['id'],q,L,c['temperature_ratio'],flush=True)
    np.savez_compressed(out/'spin-test.npz',**raw)
    return dict(rows=rows,plan=test_plan(study),raw_sha256=sha256(out/'spin-test.npz'),scope='Independent chains and configurations; eight stratified uniformly sampled target sites per cell. Test law weights physical cells equally and site strata by width.')

@torch.no_grad()
def generation(model,study,out,processor,name,mode):
    alphabet=spin_tokens(processor);cells=json.loads((study/'data/physical.json').read_text())['cells'];rows=[]
    for c in cells:
        if c['split']!='test':continue
        critical=c['temperature_ratio']==1.
        if mode=='critical' and not critical:continue
        if mode=='thermal' and (critical or c['L'] not in [4,8,16,24]):continue
        count=512 if critical else 128
        # Fixed independent seeds by model and physical cell; every generated sample is retained.
        offset={'base5':0,'physical-s0':1,'physical-s1':2}[name]
        seed=2194000+1000*offset+c['id'];start=time.perf_counter()
        x=generate(model,context_tokens(processor,c['q'],c['L'],c['temperature_ratio']),alphabet[:c['q']],c['L'],count,seed,batch_size=32)
        path=out/f"cell{c['id']}.npy";np.save(path,x)
        rows.append(dict(cell=c['id'],q=c['q'],L=c['L'],ratio=c['temperature_ratio'],count=count,seed=seed,path=path.name,sha256=sha256(path),seconds=time.perf_counter()-start))
        write_json(out/'progress.json',dict(rows=rows))
        print('generated',name,c['q'],c['L'],c['temperature_ratio'],count,round(rows[-1]['seconds'],1),flush=True)
    return dict(rows=rows,sampling='Native proper-prefix probabilities normalized on q tokens, temperature one, no cache or postprocessing.')

@torch.no_grad()
def exact(model,processor,out):
    rows=[];alphabet=spin_tokens(processor)
    for q in [2,3]:
        x=np.asarray(list(itertools.product(range(q),repeat=4)),np.uint8).reshape(-1,2,2);flat=x.reshape(-1,4);obs=observables(x,q)
        for ratio in [.94,1.,1.06]:
            logs=[]
            for j in range(4):
                inp=prefix_batch(x,j,context_tokens(processor,q,2,ratio),alphabet[:q],model.device)
                logits,_=selected_forward(model,inp,alphabet[:q]);logs.append(logits.double().log_softmax(-1).cpu().numpy())
            lp=np.stack(logs,1);logq=np.take_along_axis(lp,flat[:,:,None],2)[:,:,0].sum(1);modelp=np.exp(logq)
            logp=-obs['energy']/(ratio*critical_temperature(q));logp-=np.logaddexp.reduce(logp);p=np.exp(logp)
            kl=float(p@(logp-logq));tv=float(.5*np.abs(p-modelp).sum());terms=[]
            for j in range(4):
                term=0.
                for pref in set(map(tuple,flat[:,:j])):
                    mask=np.all(flat[:,:j]==np.asarray(pref,dtype=np.uint8),axis=1);mass=p[mask].sum()
                    cond=np.array([p[mask&(flat[:,j]==a)].sum()/mass for a in range(q)])
                    term+=mass*(cond@(np.log(cond)-lp[np.flatnonzero(mask)[0],j]))
                terms.append(float(term))
            errors={k:float(abs(p@obs[k]-modelp@obs[k])) for k in ['m','m2','m4']}
            if abs(modelp.sum()-1)>2e-6 or abs(sum(terms)-kl)>2e-6 or tv>min(1.,np.sqrt(max(kl,0)/2))+2e-6 or max(errors.values())>tv+2e-6:raise ValueError('Finite-law identity failed')
            path=out/f'q{q}-r{ratio:.3f}.npz';np.savez_compressed(path,configurations=x,source_logp=logp,model_logp=logq,model_log_conditionals=lp)
            rows.append(dict(q=q,L=2,ratio=ratio,kl=kl,tv=tv,prefix_kl=terms,errors=errors,raw=path.name,sha256=sha256(path)))
    return dict(rows=rows,scope='Exact finite laws on an untrained 2x2 lattice; no asymptotic inference.')

@torch.no_grad()
def main(a):
    from model_rg.released_qualification import admit_assessment,destination
    study=Path(a.study).resolve();out=destination(study,a.name,'assessment')/a.kind
    if a.kind not in ['language','spin','critical','thermal','exact'] or a.device not in ['cuda:0','cuda:1']:raise ValueError('Invalid assessment task/device')
    admit_assessment(study,a.name)
    out.mkdir(parents=True,exist_ok=False);before=sources();start=time.perf_counter()
    torch.set_num_threads(4);torch.cuda.set_device(a.device);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,identity=load_model(study,a.name,a.device);processor=spm.SentencePieceProcessor(model_file=str(model.path/'tokenizer.model'))
    write_json(out/'protocol.json',dict(identity=identity,kind=a.kind,sources=before,language_data_sha256=sha256(study/'data/language.npz'),physical_manifest_sha256=sha256(study/'data/physical.json')))
    if a.kind=='language':
        rows,raw=language(model,study/'data/language.npz',split='test');np.savez_compressed(out/'language-test.npz',**raw);data=dict(rows=rows,raw_sha256=sha256(out/'language-test.npz'))
    elif a.kind=='spin':data=spin_scores(model,study,out,processor)
    elif a.kind=='exact':data=exact(model,processor,out)
    else:data=generation(model,study,out,processor,a.name,a.kind)
    if before!=sources():raise ValueError('Sources changed during execution')
    write_json(out/'result.json',dict(status='complete',identity=identity,kind=a.kind,sources=before,seconds=time.perf_counter()-start,protocol_sha256=sha256(out/'protocol.json'),**data));print(out,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--name',required=True);p.add_argument('--device',required=True);p.add_argument('--kind',choices=['language','spin','critical','thermal','exact'],required=True);main(p.parse_args())
