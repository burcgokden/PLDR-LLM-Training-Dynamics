#!/usr/bin/env python
"""Independent-chain finite-size summaries of the physical reference corpus."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from model_rg.lattice import observables
from model_rg.provenance import sha256,write_json


def derived(means,L,T):
    m,m2,m4,e,m2e,m4e=means
    binder=m4/m2**2
    d2=(m2e-m2*e)/T**2;d4=(m4e-m4*e)/T**2
    derivative=d4/m2**2-2*m4*d2/m2**3
    return np.array([m,L*L*m2,binder,derivative])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',required=True);a=ap.parse_args()
    study=Path(a.study);source=study/'precision-reference/manifest.json';spec=json.loads(source.read_text())
    out=study/'precision-analysis';out.mkdir(exist_ok=False)
    rows=[];chains={}
    for c in spec['cells']:
        raw=np.memmap(c['path'],mode='r',dtype=np.uint8,shape=tuple(c['shape']))
        obs=observables(raw.reshape(-1,c['L'],c['L']),c['q'])
        a=np.stack([obs['m'],obs['m2'],obs['m4'],obs['energy'],
                    obs['m2']*obs['energy'],obs['m4']*obs['energy']],1)
        chain=a.reshape(c['chains'],c['samples_per_chain'],6).mean(1)
        chains[str(c['id'])]=chain
        estimates=derived(chain.mean(0),c['L'],c['temperature'])
        centered=obs['m2'].reshape(c['chains'],-1)
        centered=centered-centered.mean(1,keepdims=True)
        lag=float(np.mean(centered[:,:-1]*centered[:,1:])/max(np.mean(centered**2),1e-30))
        rows.append(dict(id=c['id'],q=c['q'],L=c['L'],ratio=c['temperature_ratio'],
                         temperature=c['temperature'],m=float(estimates[0]),chi=float(estimates[1]),
                         binder=float(estimates[2]),binder_derivative=float(estimates[3]),
                         energy_per_site=float(obs['energy'].mean()/c['L']**2),
                         chain_means=chain.tolist(),m2_lag1=lag))
    np.savez(out/'chain-means.npz',**chains)
    rng=np.random.default_rng(1780000);fits=[]
    for q in [2,3]:
        for minimum in [4,8,16]:
            cells=sorted([r for r in rows if r['q']==q and r['ratio']==1. and r['L']>=minimum],key=lambda r:r['L'])
            sizes=np.array([c['L'] for c in cells]);logs=np.log(sizes)
            observed=np.array([[c['m'],c['chi'],abs(c['binder_derivative'])] for c in cells])
            slopes=np.polyfit(logs,np.log(observed),1)[0]
            samples=[]
            for _ in range(1000):
                values=[]
                for c in cells:
                    cm=chains[str(c['id'])];boot=cm[rng.integers(len(cm),size=len(cm))].mean(0)
                    vals=derived(boot,c['L'],c['temperature']);values.append([vals[0],vals[1],abs(vals[3])])
                samples.append(np.polyfit(logs,np.log(values),1)[0])
            samples=np.asarray(samples)
            point=np.array([-slopes[0],slopes[1],1/slopes[2]])
            transformed=np.stack([-samples[:,0],samples[:,1],1/samples[:,2]],1)
            bounds=np.quantile(transformed,[.025,.975],axis=0)
            fits.append(dict(q=q,minimum_L=minimum,sizes=sizes.tolist(),
                beta_over_nu=float(point[0]),gamma_over_nu=float(point[1]),nu=float(point[2]),
                bootstrap95=bounds.tolist(),
                scope='Uncorrected finite-size log slopes, with independent-chain bootstrap. Window dependence measures finite-size systematic effects.'))
    result=dict(status='complete',source=str(source),source_sha256=sha256(source),
                analyzer_sha256=sha256(__file__),rows=rows,fits=fits,
                exact_targets={'2':{'beta_over_nu':1/8,'gamma_over_nu':7/4,'nu':1},
                               '3':{'beta_over_nu':2/15,'gamma_over_nu':26/15,'nu':5/6}})
    write_json(out/'analysis.json',result)
    print(json.dumps(fits,indent=2),flush=True)


if __name__=='__main__':main()
