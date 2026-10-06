"""Independent reconstruction from raw arrays, without importing project reducers.

Recomputes every new spin score, exact law, generation moment/slope/contrast,
spatial decomposition, and compact language score. Intervals are audited
separately; these are point-estimate and algebra checks, not fresh experiments.
"""
from companion_paths import configured_path
import hashlib
import json
from pathlib import Path
import numpy as np

STUDY = Path(configured_path('data:model/released-adaptation-20260913'))
counts = {}
errors = {}
bindings = {}

def bind(path):
    path=Path(path);bindings[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    return path

def read(path):
    bind(path)
    return json.loads(path.read_text())

def compare(label, actual, expected, atol=2e-10):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError(('Nonfinite value', label))
    np.testing.assert_allclose(a, b, rtol=2e-9, atol=atol, err_msg=label)
    errors[label] = max(errors.get(label, 0.), float(np.max(np.abs(a-b))))

def physical(x, q):
    x = np.asarray(x); L = x.shape[-1]; x = x.reshape(-1,L,L)
    f = np.array([(x==a).mean((1,2)) for a in range(q)]).T
    m2 = q/(q-1)*((f-1/q)**2).sum(1)
    mu = f.mean(0)
    profile = np.array([(x==a).mean(0) for a in range(q)]).transpose(1,2,0)
    out = dict(m=np.sqrt(m2).mean(), chi=L*L*m2.mean(),
        connected_chi=L*L*q/(q-1)*np.var(f,axis=0,ddof=0).sum(),
        binder=np.mean(m2*m2)/np.mean(m2)**2,
        global_mean_sector=q/(q-1)*((mu-1/q)**2).sum(),
        position_mean_inhomogeneity=q/(q-1)*((profile-mu)**2).sum(-1).mean())
    corrs = {}
    for r in sorted({1,max(1,L//4),max(1,L//2)}):
        equal = np.mean([(x==np.roll(x,r,axis=k)).mean() for k in (1,2)])
        raw = (q*equal-1)/(q-1)
        sector = q/(q-1)*np.mean([((profile-1/q)*(np.roll(profile,r,axis=k)-1/q)).sum(-1).mean() for k in (0,1)])
        corrs[str(r)] = dict(raw=raw, connected=raw-sector, mean_sector=sector)
    out['correlations'] = corrs
    return out

def main():
    cells = {c['id']:c for c in read(STUDY/'data/physical.json')['cells']}
    refs = {}
    generated = {}
    generation_report = read(STUDY/'analysis/generation.json')
    spatial_report = {(r['name'],r['cell']):r for r in read(STUDY/'analysis/spatial-centering.json')['cells']}
    for name in ('base5','physical-s0','physical-s1'):
        for kind in ('critical','thermal'):
            root = STUDY/'assessment'/name/kind
            for r in read(root/'result.json')['rows']:
                c=cells[r['cell']]
                x=np.load(bind(root/r['path']))
                v=physical(x,c['q']); generated[name,c['id']]=v
                if c['id'] not in refs:
                    source=np.memmap(bind(c['path']),dtype=np.uint8,mode='r',shape=tuple(c['shape']))
                    refs[c['id']]=physical(source,c['q'])
                row=next(t for t in generation_report['cells'] if (t['name'],t['cell'])==(name,c['id']))
                for k in ('m','chi','connected_chi','binder'):
                    compare('generated_'+k,v[k],row[k])
                for field,values in [('model',v),('source',refs[c['id']])]:
                    reported=spatial_report[name,c['id']][field]
                    for k in ('chi','connected_chi','global_mean_sector','position_mean_inhomogeneity'):
                        compare('spatial_'+k,values[k],reported[k])
                    for rkey, cc in values['correlations'].items():
                        for k,t in cc.items():compare('spatial_'+k,t,reported['correlations'][rkey][k])
    counts['generated_cells']=len(generated)
    counts['generated_configurations']=sum(r['count'] for r in generation_report['cells'])
    for row in generation_report['finite_slopes']:
        cases=sorted([c for c in cells.values() if c['split']=='test' and c['q']==row['q'] and c['temperature_ratio']==1 and c['L']>=row['cutoff']],key=lambda c:c['L'])
        xx=np.log([c['L'] for c in cases]);xx-=xx.mean()
        for k,field,sign in [('m','beta_over_nu',-1),('chi','gamma_over_nu',1),('connected_chi','connected_gamma_over_nu',1)]:
            yy=np.log([generated[row['name'],c['id']][k] for c in cases]);slope=sign*float(xx@yy/(xx@xx))
            compare('finite_'+field,slope,row[field])
        for k,field,sign in [('m','reference_beta_over_nu',-1),('chi','reference_gamma_over_nu',1)]:
            yy=np.log([refs[c['id']][k] for c in cases]);compare(field,sign*float(xx@yy/(xx@xx)),row[field])
    counts['finite_slope_rows']=len(generation_report['finite_slopes'])
    for row in generation_report['temperature_contrasts']:
        near={c['temperature_ratio']:c['id'] for c in cells.values() if c['split']=='test' and c['q']==row['q'] and c['L']==row['L']}
        w=row['half_width'];lo=near[round(1-w,2)];hi=near[round(1+w,2)]
        compare('thermal_contrast',(generated[row['name'],hi]['binder']-generated[row['name'],lo]['binder'])/(2*w),row['binder_contrast'])
    counts['thermal_contrasts']=len(generation_report['temperature_contrasts'])

    spin_report=read(STUDY/'analysis/spin.json');spin={};nrows=0;logit_coordinates=0
    for name in ('base5','physical-s0','physical-s1'):
        root=STUDY/'assessment'/name/'spin';result=read(root/'result.json');parts={}
        with np.load(bind(root/'spin-test.npz')) as z:
            for row in result['rows']:
                q=row['q'];a=z[row['raw']].astype(float);logits=a[:,:q];target=a[:,q].astype(int)
                lp=logits-np.logaddexp.reduce(logits,axis=1)[:,None]
                nll=-lp[np.arange(len(a)),target];acc=(logits.argmax(1)==target).astype(float)
                compare('raw_native_nll',nll,a[:,-2],atol=2e-6);compare('raw_native_accuracy',acc,a[:,-1])
                pair=np.column_stack((nll,acc));parts.setdefault(row['cell'],[]).append(row['site_weight']*pair.reshape(16,8,2).mean(1))
                nrows+=1;logit_coordinates+=logits.size
        for cid,values in parts.items():
            mean=np.sum(values,axis=0);spin[name,cid]=mean
            r=next(r for r in spin_report['cells'] if (r['name'],r['cell'])==(name,cid))
            compare('cell_spin_scores',mean.mean(0),[r['nll'],r['accuracy']])
    for row in spin_report['summary']:
        selected=[c for c in cells.values() if c['split']=='test' and c['q']==row['q'] and
            (row['scope']!='critical' or c['temperature_ratio']==1) and
            (row['scope']!='trained_sizes' or c['L'] in [4,8,16]) and
            (row['scope']!='untrained_sizes' or c['L'] not in [4,8,16])]
        mean=np.mean([spin[row['name'],c['id']].mean(0) for c in selected],axis=0)
        compare('aggregate_spin_scores',mean,[row['nll'],row['accuracy']])
        if row['name']!='base5':
            delta=np.mean([spin[row['name'],c['id']]-spin['base5',c['id']] for c in selected],axis=(0,1))
            compare('aggregate_paired_delta',delta,[row['delta_nll'],row['delta_accuracy']])
    counts.update(spin_site_panels=nrows,spin_logit_coordinates=logit_coordinates,spin_cells=len(spin),spin_summary_rows=len(spin_report['summary']))

    exact_rows=0
    for name in ('base5','physical-s0','physical-s1'):
        root=STUDY/'assessment'/name/'exact'
        for row in read(root/'result.json')['rows']:
            with np.load(bind(root/row['raw'])) as z:
                x=z['configurations'];q=row['q'];flat=x.reshape(-1,4)
                energy=-sum((x==np.roll(x,1,axis=k)).sum((1,2)) for k in (1,2))
                lp=-energy*np.log(1+np.sqrt(q))/row['ratio'];lp-=np.logaddexp.reduce(lp)
                lc=z['model_log_conditionals'];lq=np.take_along_axis(lc,flat[:,:,None],axis=2)[:,:,0].sum(1)
                compare('exact_source_law',lp,z['source_logp']);compare('exact_model_chain',lq,z['model_logp'])
                P=np.exp(lp);Q=np.exp(lq);compare('exact_normalization',Q.sum(),1.,atol=2e-6)
                kl=P@(lp-lq);tv=.5*np.abs(P-Q).sum();terms=[]
                for j in range(4):
                    total=0.
                    for prefix in set(map(tuple,flat[:,:j])):
                        ix=np.flatnonzero((flat[:,:j]==np.asarray(prefix)).all(1));mass=P[ix].sum()
                        pp=np.array([P[ix[flat[ix,j]==a]].sum()/mass for a in range(q)])
                        compare('exact_prefix_consistency',lc[ix,j],np.broadcast_to(lc[ix[0],j],lc[ix,j].shape),atol=2e-6)
                        total+=mass*(pp@(np.log(pp)-lc[ix[0],j]))
                    terms.append(total)
                compare('exact_KL_TV',[kl,tv],[row['kl'],row['tv']]);compare('exact_KL_chain',sum(terms),kl,atol=2e-6)
                f=np.stack([(flat==a).mean(1) for a in range(q)],axis=1);m2=q/(q-1)*((f-1/q)**2).sum(1)
                for ob in [np.sqrt(m2),m2,m2**2]:
                    if abs((P-Q)@ob)>tv+2e-6:raise AssertionError('Moment/TV inequality')
            exact_rows+=1
    counts['exact_laws']=exact_rows

    language=read(STUDY/'analysis/language.json');panels={};samples=0
    for name in ['base5']+[task+suffix for suffix in ['-s0','-lowrank'] for task in ['technical','narrative','mixture','general']]:
        root=STUDY/'assessment'/name/'language'
        with np.load(bind(root/'language-test.npz')) as z:
            for domain in ['technical','narrative','unassigned']:
                for lengths,label in [([32,64,128],'all'),([64],'matched64')]:
                    values=[]
                    for offset in [0,192]:
                        for length in lengths:
                            a=z[f'{domain}-o{offset}-p{length}'].astype(float)
                            compare('language_argmax_indicator',(a[:,2]==a[:,3]).astype(float),a[:,4])
                            values.append(np.column_stack((a[:,0]-a[:,1],a[:,4],a[:,5])))
                    panels[name,domain,label]=np.mean(values,axis=0)
                    r=next(r for r in language['test'] if (r['name'],r['domain'],r['prefix'])==(name,domain,label))
                    compare('language_panel_scores',panels[name,domain,label].mean(0),[r['nll'],r['accuracy'],r['top5']])
                samples+=512*6
    for r in language['summary']:
        ds=[r['task']] if r['task'] in ['technical','narrative'] else ['technical','narrative'] if r['task']=='mixture' else ['technical','narrative','unassigned']
        mean=np.mean([panels[r['name'],d,r['prefix']].mean(0) for d in ds],axis=0)
        compare('language_summary',mean[:2],[r['nll'],r['accuracy']])
    counts.update(language_panels=len(panels),language_scored_examples=samples,language_summary_rows=len(language['summary']))
    # New-seed, paired chain intervals for the two noteworthy untrained-size
    # Potts accuracy changes, independent of the published bootstrap RNG.
    rng=np.random.default_rng(2026091301);intervals={}
    cases=[c for c in cells.values() if c['split']=='test' and c['q']==3 and c['L'] in [6,12,20,24]]
    for name in ('physical-s0','physical-s1'):
        delta=np.array([spin[name,c['id']]-spin['base5',c['id']] for c in cases])
        estimates=[]
        for _ in range(4000):
            indices=rng.integers(16,size=(len(cases),16))
            estimates.append(delta[np.arange(len(cases))[:,None],indices].mean((0,1)))
        intervals[name]=dict(delta=delta.mean((0,1)).tolist(),paired_chain_bootstrap95=np.quantile(estimates,[.025,.975],axis=0).tolist())
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(dict(status='passed',producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checked_sha256=bindings,counts=counts,max_abs_error=errors,
        independent_untrained_potts_intervals=intervals,
        scope='Own NumPy reducer; all retained point estimates and exact identities. Published bootstrap quantiles are not exhaustively regenerated; two selected paired intervals use a new seed. Language uses the supplied compact normalizer and target-logit records, not full vocabulary recomputation.'),indent=2)+'\n')
    print(json.dumps(counts,indent=2),flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();STUDY=args.study.resolve();OUT=args.output.resolve();main()
