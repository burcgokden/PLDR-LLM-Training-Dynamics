"""Independent reductions of held-out logits and generated physical configurations."""
import argparse,json,sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from model_rg.provenance import sha256,write_json
from model_rg.lattice import observables,summarize


def interval(values,rng,repeats=2000):
    x=np.asarray(values,dtype=np.float64)
    means=np.array([x[rng.integers(len(x),size=len(x))].mean(0) for _ in range(repeats)])
    return np.quantile(means,[.025,.975],axis=0).tolist()


def checked(path,expected,bindings):
    if sha256(path)!=expected:raise ValueError('Changed evidence: '+str(path))
    bindings[str(path)]=expected


def language_analysis(study,bindings):
    rng=np.random.default_rng(219501);bases=[];runs=[];test={};qualification={}
    for index in [1,4,5]:
        root=study/f'base-qualification/model{index}';q=json.loads((root/'verification.json').read_text());bindings[str(root/'verification.json')]=sha256(root/'verification.json')
        checked(root/'language-validation.npz',q['files']['language-validation.npz'],bindings)
        with np.load(root/'language-validation.npz') as z:
            a=[]
            for domain in ['technical','narrative','unassigned']:
                v=np.stack([z[f'{domain}-o{o}-p{p}'] for o in [0,192] for p in [32,64,128]],1).astype(np.float64)
                metrics=np.stack([(v[:,:,0]-v[:,:,1]).mean(1),v[:,:,4].mean(1),v[:,:,5].mean(1)],1);a.append(metrics)
            allv=np.concatenate(a);qualification[index]=allv
            bases.append(dict(model=index,nll=float(allv[:,0].mean()),accuracy=float(allv[:,1].mean()),top5=float(allv[:,2].mean()),document_bootstrap95=interval(allv,rng)))
    for name in ['base5']+[task+suffix for suffix in ['-s0','-lowrank'] for task in ['technical','narrative','mixture','general']]:
        root=study/'assessment'/name/'language';r=json.loads((root/'result.json').read_text());bindings[str(root/'result.json')]=sha256(root/'result.json')
        checked(root/'language-test.npz',r['raw_sha256'],bindings);panels={}
        with np.load(root/'language-test.npz') as z:
            for domain in ['technical','narrative','unassigned']:
                for lengths,label in [([32,64,128],'all'),([64],'matched64')]:
                    vs=[]
                    for o in [0,192]:
                        for p in lengths:
                            a=z[f'{domain}-o{o}-p{p}'].astype(np.float64);vs.append(np.stack([a[:,0]-a[:,1],a[:,4],a[:,5]],1))
                    panels[domain+'-'+label]=np.mean(vs,axis=0)
        test[name]=panels
        for key,a in panels.items():
            domain,prefix=key.split('-');row=dict(name=name,domain=domain,prefix=prefix,selected_step=r['identity']['step'],selected_epoch=r['identity']['epoch'],nll=float(a[:,0].mean()),accuracy=float(a[:,1].mean()),top5=float(a[:,2].mean()),document_bootstrap95=interval(a,rng))
            if name!='base5':
                delta=a-test['base5'][key];row.update(delta_nll=float(delta[:,0].mean()),delta_accuracy=float(delta[:,1].mean()),paired_document_bootstrap95=interval(delta,rng))
            runs.append(row)
    summaries=[]
    for task in ['technical','narrative','mixture','general']:
        domains=[task] if task in ['technical','narrative'] else ['technical','narrative'] if task=='mixture' else ['technical','narrative','unassigned']
        for name in ['base5',task+'-s0',task+'-lowrank']:
            for prefix in ['all','matched64']:
                v=[test[name][d+'-'+prefix] for d in domains];base=[test['base5'][d+'-'+prefix] for d in domains]
                mean=np.mean([x.mean(0) for x in v],axis=0);bm=np.mean([x.mean(0) for x in base],axis=0);delta=[x-y for x,y in zip(v,base)]
                db=np.array([np.mean([x[rng.integers(len(x),size=len(x))].mean(0) for x in delta],axis=0) for _ in range(2000)])
                selected=next(r for r in runs if r['name']==name)
                summaries.append(dict(task=task,name=name,prefix=prefix,nll=float(mean[0]),accuracy=float(mean[1]),base_nll=float(bm[0]),base_accuracy=float(bm[1]),delta_nll=float(mean[0]-bm[0]),delta_accuracy=float(mean[1]-bm[1]),paired_stratified_document_bootstrap95=np.quantile(db,[.025,.975],axis=0).tolist(),selected_step=selected['selected_step'],selected_epoch=selected['selected_epoch']))
    return dict(base_validation=bases,base5_minus4_validation_paired95=interval(qualification[5]-qualification[4],rng),test=runs,summary=summaries,scope='Float64 reductions of stored vocabulary log normalizers, target logits and prediction indices; full vocabulary vectors are not retained for this language assessment. Equal domain weighting; each document is a bootstrap unit with all token positions retained together. Multiple target positions are not independent documents.')


def spin_analysis(study,bindings):
    rng=np.random.default_rng(219502);rows=[];summary=[];byname={}
    for name in ['base5','physical-s0','physical-s1']:
        root=study/'assessment'/name/'spin';r=json.loads((root/'result.json').read_text());bindings[str(root/'result.json')]=sha256(root/'result.json');checked(root/'spin-test.npz',r['raw_sha256'],bindings)
        grouped={};byname[name]={}
        with np.load(root/'spin-test.npz') as z:
            for row in r['rows']:
                a=z[row['raw']].astype(np.float64);q=row['q'];logits=a[:,:q];target=a[:,q].astype(int)
                logp=logits-np.logaddexp.reduce(logits,axis=1)[:,None];nll=-logp[np.arange(len(a)),target];ok=logits.argmax(1)==target
                if not np.allclose(nll,a[:,-2],rtol=0,atol=2e-6) or not np.array_equal(ok,a[:,-1]):raise ValueError('Saved spin reduction disagrees with logits')
                hist=z[row['raw']+'-histogram'];metrics=np.stack([nll,ok,hist[:,0],hist[:,1]],1).reshape(16,8,4).mean(1)
                grouped.setdefault(row['cell'],[]).append((row,metrics))
            for cell,records in grouped.items():
                a=sum(row['site_weight']*v for row,v in records);row=records[0][0];byname[name][cell]=a
                value=dict(name=name,cell=cell,q=row['q'],L=row['L'],ratio=row['ratio'],nll=float(a[:,0].mean()),accuracy=float(a[:,1].mean()),histogram_nll=float(a[:,2].mean()),histogram_accuracy=float(a[:,3].mean()),chain_bootstrap95=interval(a,rng))
                if name!='base5':
                    d=a-byname['base5'][cell];value.update(delta_nll=float(d[:,0].mean()),delta_accuracy=float(d[:,1].mean()),paired_chain_bootstrap95=interval(d[:,:2],rng))
                rows.append(value)
        for q in [2,3]:
            for scope in ['all','critical','trained_sizes','untrained_sizes']:
                select=[row for row in rows if row['name']==name and row['q']==q and (scope!='critical' or row['ratio']==1.) and (scope!='trained_sizes' or row['L'] in [4,8,16]) and (scope!='untrained_sizes' or row['L'] not in [4,8,16])]
                a=np.stack([byname[name][row['cell']] for row in select]);means=a.mean((0,1))
                boots=np.array([np.mean([v[rng.integers(16,size=16)].mean(0) for v in a],axis=0) for _ in range(2000)])
                record=dict(name=name,q=q,scope=scope,cells=len(select),nll=float(means[0]),accuracy=float(means[1]),histogram_nll=float(means[2]),histogram_accuracy=float(means[3]),stratified_chain_bootstrap95=np.quantile(boots,[.025,.975],axis=0).tolist())
                if name!='base5':
                    delta=np.stack([byname[name][row['cell']]-byname['base5'][row['cell']] for row in select]);db=np.array([np.mean([v[rng.integers(16,size=16)].mean(0) for v in delta],axis=0) for _ in range(2000)])
                    record.update(delta_nll=float(delta[:,:,0].mean()),delta_accuracy=float(delta[:,:,1].mean()),paired_stratified_chain_bootstrap95=np.quantile(db,[.025,.975],axis=0).tolist())
                summary.append(record)
    return dict(cells=rows,summary=summary,scope='Site-weighted means; physical cells equally weighted. Confidence intervals resample chains independently within each physical cell, paired between native models and baselines, conditional on the frozen target-site panel.')


def physical_bootstrap(x,q,rng,repeats=1000,chain_axis=False):
    L=x.shape[-1];flat=x.reshape(-1,L,L);o=observables(flat,q);f=np.stack([(flat==a).mean((1,2)) for a in range(q)],1)
    vals=np.column_stack([o['m'],o['m2'],o['m4'],f]);vals=vals.reshape(x.shape[0],x.shape[1],-1).mean(1) if chain_axis else vals
    means=np.array([vals[rng.integers(len(vals),size=len(vals))].mean(0) for _ in range(repeats)])
    bias=q/(q-1)*np.square(means[:,3:]-1/q).sum(1)
    boot=np.stack([means[:,0],L*L*means[:,1],means[:,2]/means[:,1]**2,L*L*(means[:,1]-bias)],1)
    return boot


def generation_analysis(study,bindings):
    rng=np.random.default_rng(219503);cells={c['id']:c for c in json.loads((study/'data/physical.json').read_text())['cells']};rows=[];fits=[];boots={};reference={}
    for c in cells.values():
        if c['split']!='test':continue
        checked(c['path'],c['sha256'],bindings);x=np.asarray(np.memmap(c['path'],mode='r',dtype=np.uint8,shape=tuple(c['shape'])));flat=x.reshape(-1,c['L'],c['L']);s=summarize(flat,c['q']);b=physical_bootstrap(x,c['q'],rng,chain_axis=True)
        reference[c['id']]=s;boots[('source',c['id'])]=b
    for name in ['base5','physical-s0','physical-s1']:
        for kind in ['critical','thermal']:
            root=study/'assessment'/name/kind;r=json.loads((root/'result.json').read_text());bindings[str(root/'result.json')]=sha256(root/'result.json')
            for cell in r['rows']:
                p=root/cell['path'];checked(p,cell['sha256'],bindings);x=np.load(p);q=cell['q'];L=cell['L'];s=summarize(x,q);ref=reference[cell['cell']]
                f=np.stack([(x==a).mean((1,2)) for a in range(q)],1);bias=q/(q-1)*np.square(f.mean(0)-1/q).sum();connected=L*L*(s['m2']-bias);b=physical_bootstrap(x,q,rng);boots[(name,cell['cell'])]=b
                row=dict(name=name,**{k:cell[k] for k in ['cell','q','L','ratio','count']},m=s['m'],chi=s['chi'],connected_chi=connected,binder=s['binder_ratio'],color_bias=float(np.max(abs(f.mean(0)-1/q))),color_fractions=f.mean(0).tolist(),reference_m=ref['m'],reference_chi=ref['chi'],reference_binder=ref['binder_ratio'],relative_error_m=s['m']/ref['m']-1,relative_error_chi=s['chi']/ref['chi']-1,relative_error_binder=s['binder_ratio']/ref['binder_ratio']-1,bootstrap95=np.quantile(b,[.025,.975],axis=0).tolist(),reference_chain_bootstrap95=np.quantile(boots[('source',cell['cell'])],[.025,.975],axis=0).tolist(),correlations={k:dict(model=v,reference=ref[k],absolute_error=v-ref[k]) for k,v in s.items() if k.startswith('corr')})
                rows.append(row)
        for q in [2,3]:
            for cutoff in [4,8,12]:
                sub=sorted([r for r in rows if r['name']==name and r['q']==q and r['ratio']==1. and r['L']>=cutoff],key=lambda r:r['L']);L=np.array([r['L'] for r in sub]);d=np.column_stack([np.ones(len(L)),np.log(L)]);inv=np.linalg.pinv(d)[1]
                model=np.array([[r['m'],r['chi'],r['connected_chi']] for r in sub]);ref=np.array([[r['reference_m'],r['reference_chi']] for r in sub]);coeff=inv@np.log(model);rcoeff=inv@np.log(ref);bs=np.stack([boots[(name,r['cell'])][:,[0,1,3]] for r in sub],1);rb=np.stack([boots[('source',r['cell'])][:,:2] for r in sub],1)
                mb=np.einsum('l,blo->bo',inv,np.log(bs));mb[:,0]*=-1;rbs=np.einsum('l,blo->bo',inv,np.log(rb));rbs[:,0]*=-1
                fits.append(dict(name=name,q=q,cutoff=cutoff,sizes=L.tolist(),beta_over_nu=float(-coeff[0]),gamma_over_nu=float(coeff[1]),connected_gamma_over_nu=float(coeff[2]),reference_beta_over_nu=float(-rcoeff[0]),reference_gamma_over_nu=float(rcoeff[1]),conditional_bootstrap95=np.quantile(mb,[.025,.975],axis=0).tolist(),reference_chain_bootstrap95=np.quantile(rbs,[.025,.975],axis=0).tolist()))
    thermal=[]
    for name in ['base5','physical-s0','physical-s1']:
        for q in [2,3]:
            for L in [4,8,16,24]:
                rr={r['ratio']:r for r in rows if r['name']==name and r['q']==q and r['L']==L}
                for width in [.03,.06]:
                    lo,hi=rr[round(1-width,2)],rr[round(1+width,2)]
                    mb=(boots[(name,hi['cell'])][:,2]-boots[(name,lo['cell'])][:,2])/(2*width);rb=(boots[('source',hi['cell'])][:,2]-boots[('source',lo['cell'])][:,2])/(2*width)
                    thermal.append(dict(name=name,q=q,L=L,half_width=width,binder_contrast=(hi['binder']-lo['binder'])/(2*width),reference_binder_contrast=(hi['reference_binder']-lo['reference_binder'])/(2*width),conditional_bootstrap95=np.quantile(mb,[.025,.975]).tolist(),reference_chain_bootstrap95=np.quantile(rb,[.025,.975]).tolist()))
    return dict(cells=rows,finite_slopes=fits,temperature_contrasts=thermal,scope='Finite log slopes of generated moments, not established native or physical thermodynamic exponents. Conditional generation bootstrap and independent source-chain bootstrap; adaptation seeds remain separate.')


def main(a):
    study=Path(a.study);out=study/'analysis';out.mkdir(exist_ok=True);bindings={str(study/'assessment-plan.json'):sha256(study/'assessment-plan.json')}
    f={'language':language_analysis,'spin':spin_analysis,'generation':generation_analysis}[a.kind];data=f(study,bindings)
    write_json(out/(a.kind+'.json'),dict(status='complete',kind=a.kind,analyzer_sha256=sha256(__file__),checked_sha256=bindings,**data));print(out/(a.kind+'.json'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--kind',required=True,choices=['language','spin','generation']);main(p.parse_args())
