#!/usr/bin/env python
"""Report all native physical-law cells, finite slopes and spatial controls."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from model_rg.lattice import observables,summarize,slope_bound
from model_rg.provenance import sha256,write_json


def generation(x,q,bootstrap,rng):
    obs=observables(x,q)
    values=np.stack([obs['m'],obs['m2'],obs['m4']],1)
    averages=np.array([values[rng.integers(len(x),size=len(x))].mean(0) for _ in range(bootstrap)])
    transformed=np.stack([averages[:,0],x.shape[-1]**2*averages[:,1],averages[:,2]/averages[:,1]**2],1)
    return obs,transformed


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',required=True);a=ap.parse_args()
    base=Path(a.study);out=base/'model-analysis';out.mkdir(exist_ok=False)
    spec=json.loads((base/'assessment/protocol.json').read_text())
    reference=json.loads((base/'reference-analysis/analysis.json').read_text())
    precision=json.loads((base/'precision-analysis/analysis.json').read_text())
    data=json.loads((base/'assessment-data/manifest.json').read_text())
    ref={c['id']:c for c in reference['rows']}
    precise={(c['q'],c['L']):c for c in precision['rows']}
    chains=np.load(base/'reference-analysis/chain-means.npz')
    spatial={}
    for c in spec['cells']:
        raw=np.memmap(c['path'],mode='r',dtype=np.uint8,shape=tuple(c['shape']))
        spatial[c['id']]=summarize(raw.reshape(-1,c['L'],c['L']),c['q'])
    rng=np.random.default_rng(1930000);rows=[];fits=[];interventions=[];development=[];budgets=[]
    checked={str(base/'assessment/protocol.json'):sha256(base/'assessment/protocol.json')}
    for case in spec['cases']:
        parent=base/'assessment'/case['name'];p=parent/'manifest.json'
        m=json.loads(p.read_text())
        if m['status']!='complete':raise ValueError('Incomplete assessment')
        checked[str(p)]=sha256(p)
        generated_boot={}
        case_rows=[]
        for checkpoint in m['checkpoints']:
            p=Path(checkpoint['results'])
            if sha256(p)!=checkpoint['sha256']:raise ValueError('Changed assessment results')
            checked[str(p)]=sha256(p)
            result=json.loads(p.read_text());folder=p.parent
            observations={c['cell']:c for c in result['observations']}
            for c in result['generated']:
                path=Path(c['path'])
                if sha256(path)!=c['sha256']:raise ValueError('Changed generated configurations')
                checked[str(path)]=sha256(path)
                x=np.load(path);obs,boot=generation(x,c['q'],1000,rng)
                generated_boot[(checkpoint['step'],c['q'],c['L'],c['ratio'])]=boot
                summary=summarize(x,c['q']);r=ref[c['cell']]
                if c['ratio']==1.:r=precise[(c['q'],c['L'])]
                record=observations[c['cell']];op=folder/record['file']
                if sha256(op)!=record['sha256']:raise ValueError('Changed native observations')
                checked[str(op)]=sha256(op)
                z=np.load(op)
                nll=np.stack([z[k] for k in z.files if k.endswith('-nll')],1)
                fractions=np.stack([(x==color).mean((1,2)) for color in range(c['q'])],1)
                connected=c['L']**2*c['q']/(c['q']-1)*np.square(fractions-fractions.mean(0)).sum(1).mean()
                row=dict(case=case['name'],heads=case['heads'],seed=case['seed'],arm=case['arm'],step=checkpoint['step'],
                    q=c['q'],L=c['L'],ratio=c['ratio'],count=len(x),
                    m=summary['m'],chi=summary['chi'],binder=summary['binder_ratio'],
                    reference_m=r['m'],reference_chi=r['chi'],reference_binder=r['binder'],
                    relative_error_m=summary['m']/r['m']-1,relative_error_chi=summary['chi']/r['chi']-1,
                    relative_error_binder=summary['binder_ratio']/r['binder']-1,
                    bootstrap95=np.quantile(boot,[.025,.975],axis=0).tolist(),
                    connected_chi=float(connected),mean_color_fractions=fractions.mean(0).tolist(),
                    color_bias=float(np.max(np.abs(fractions.mean(0)-1/c['q']))),
                    correlations={k:dict(model=v,reference=spatial[c['cell']][k]) for k,v in summary.items() if k.startswith('corr')},
                    nll=float(nll.mean()),nll_by_prefix=nll.mean(0).tolist(),
                    nll_chain_means=nll.mean(1).reshape(16,4).mean(1).tolist(),
                    row_fraction_median=float(np.median(z['row_fraction'])),row_fraction_max=float(np.max(z['row_fraction'])))
                rows.append(row);case_rows.append(row)
            for q in [2,3]:
                for minimum in [4,8]:
                    selected=sorted([r for r in case_rows if r['step']==checkpoint['step'] and r['q']==q and r['ratio']==1. and r['L']>=minimum],key=lambda r:r['L'])
                    sizes=np.array([r['L'] for r in selected]);design=np.stack([np.ones(len(sizes)),np.log(sizes)],1)
                    inverse=np.linalg.pinv(design)
                    vals=np.array([[r['m'],r['chi']] for r in selected])
                    reference_vals=np.array([[r['reference_m'],r['reference_chi']] for r in selected])
                    slopes=(inverse@np.log(vals))[1];reference_slopes=(inverse@np.log(reference_vals))[1]
                    boots=np.stack([generated_boot[(checkpoint['step'],q,r['L'],1.)][:,:2] for r in selected],1)
                    boot_slopes=np.einsum('s,bso->bo',inverse[1],np.log(boots));boot_slopes[:,0]*=-1
                    fits.append(dict(case=case['name'],heads=case['heads'],seed=case['seed'],arm=case['arm'],step=checkpoint['step'],q=q,
                        minimum_L=minimum,sizes=sizes.tolist(),beta_over_nu=float(-slopes[0]),gamma_over_nu=float(slopes[1]),
                        reference_beta_over_nu=float(-reference_slopes[0]),reference_gamma_over_nu=float(reference_slopes[1]),
                        conditional_bootstrap95=np.quantile(boot_slopes,[.025,.975],axis=0).tolist()))
                critical={r['L']:r for r in case_rows if r['step']==checkpoint['step'] and r['q']==q and r['ratio']==1.}
                for L in [4,8]:
                    lo,hi=critical[L],critical[2*L]
                    eps=max(abs(lo['relative_error_chi']),abs(hi['relative_error_chi']))
                    actual=np.log(hi['chi']/lo['chi'])/np.log(2)-np.log(hi['reference_chi']/lo['reference_chi'])/np.log(2)
                    bound=slope_bound(eps,2) if eps<1 else None
                    if bound is not None and abs(actual)>bound+1e-12:raise ValueError('Finite slope inequality failed')
                    budgets.append(dict(case=case['name'],step=checkpoint['step'],q=q,L=L,epsilon=eps,
                        slope_error=float(actual),bound=bound))
        for r in m['interventions']:
            if sha256(r['path'])!=r['sha256']:raise ValueError('Changed intervention configurations')
            checked[r['path']]=r['sha256'];interventions.append(dict(case=case['name'],**r))
        for step in [0,512,2048,16384]:
            p=base/'confirmation'/case['name']/f'observation-{step:05d}.json'
            d=json.loads(p.read_text());checked[str(p)]=sha256(p)
            for q in [2,3]:
                for L in [4,8,16]:
                    selected=[r for r in d['results'] if r['q']==q and r['L']==L]
                    development.append(dict(case=case['name'],step=step,q=q,L=L,
                        nll=float(np.mean([r['nll'] for r in selected])),
                        scope='Common development holdout, never trained on; not the independent assessment.'))
    # Five discrete temperature tags define two symmetric finite differences.
    # Their width dependence is reported, not relabelled as an analytic derivative.
    thermal=[]
    for case in spec['cases']:
        for q in [2,3]:
            for L in [4,6,8,12,16]:
                selected={r['ratio']:r for r in rows if r['case']==case['name'] and r['step']==16384 and r['q']==q and r['L']==L}
                for width in [.03,.06]:
                    lo,hi=selected[round(1-width,2)],selected[round(1+width,2)]
                    thermal.append(dict(case=case['name'],q=q,L=L,half_width=width,
                        binder_contrast=(hi['binder']-lo['binder'])/(2*width),
                        reference_binder_contrast=(hi['reference_binder']-lo['reference_binder'])/(2*width)))
    report=dict(status='complete',schema='physical-model-analysis-v1',analyzer_sha256=sha256(__file__),
        rows=rows,finite_slopes=fits,slope_budgets=budgets,interventions=interventions,
        development_risk=development,temperature_contrasts=thermal,checked_sha256=checked,
        scope='All frozen cells and cases. Monte Carlo intervals are conditional on each model; three initialization replicas are shown separately. Five sizes through L=16 give finite slopes, not established learned thermodynamic exponents. No fit to native width criticality is attempted.')
    write_json(out/'analysis.json',report)
    print('complete',len(rows),'cells,',len(fits),'finite slopes',flush=True)


if __name__=='__main__':main()
