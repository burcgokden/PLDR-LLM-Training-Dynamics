"""Correction-family and size-window sensitivity, separate from sampling error."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from model_rg.provenance import sha256,write_json


def derivative(cm,T):
    m,m2,m4,e,m2e,m4e=cm
    return ((m4e-m4*e)/m2**2-2*m4*(m2e-m2*e)/m2**3)/T**2


def matrix(L,omega):
    columns=[np.ones(len(L)),np.log(L)]
    if omega is not None:columns.append(L.astype(float)**(-omega))
    return np.column_stack(columns)


def main(study,output):
    study=Path(study);output=Path(output);output.mkdir(exist_ok=False)
    design=dict(schema='physical-fit-sensitivity-v1',sizes_min=[8,16,24],omega=[None,.8,1.,2.],
        fit='Unweighted log absolute Binder derivative = intercept + p log L + c L^(-omega); nu=1/p. No exponent is fixed or selected to recover its target.',
        validation='Leave one physical size out; report every fit and window; independent chain bootstrap conditions on the fit family.',
        bootstrap_replicates=1000,seed=218400)
    write_json(output/'design.json',design)
    source=study/'precision-analysis/analysis.json';prior=json.loads(source.read_text())
    source_manifest=Path(prior['source'])
    if sha256(source_manifest)!=prior['source_sha256']:raise ValueError('Reference manifest changed')
    chain_file=study/'precision-analysis/chain-means.npz';saved=np.load(chain_file)
    for r in prior['rows']:
        if not np.array_equal(saved[str(r['id'])],np.asarray(r['chain_means'])):raise ValueError('Reference chain reduction disagrees')
    rows=[];rng=np.random.default_rng(design['seed'])
    for q in [2,3]:
        for minimum in design['sizes_min']:
            selected=sorted([r for r in prior['rows'] if r['q']==q and r['L']>=minimum and r['ratio']==1.],key=lambda r:r['L'])
            L=np.array([r['L'] for r in selected]);means=[saved[str(r['id'])] for r in selected]
            y=np.array([np.log(abs(derivative(c.mean(0),r['temperature']))) for c,r in zip(means,selected)])
            boot=[]
            for _ in range(1000):
                boot.append([np.log(abs(derivative(c[rng.integers(len(c),size=len(c))].mean(0),r['temperature'])))
                             for c,r in zip(means,selected)])
            boot=np.asarray(boot)
            for omega in design['omega']:
                X=matrix(L,omega);coef=np.linalg.lstsq(X,y,rcond=None)[0]
                sample=np.linalg.lstsq(X,boot.T,rcond=None)[0][1]
                predicted=[]
                for i in range(len(L)):
                    keep=np.arange(len(L))!=i
                    fit=np.linalg.lstsq(X[keep],y[keep],rcond=None)[0]
                    predicted.append(float(X[i]@fit))
                rows.append(dict(q=q,minimum_L=minimum,sizes=L.tolist(),omega=omega,
                    nu=float(1/coef[1]),bootstrap95=np.quantile(1/sample,[.025,.975]).tolist(),
                    fitted_power=float(coef[1]),coefficients=coef.tolist(),
                    leave_one_size_out_log_rmse=float(np.sqrt(np.mean((y-predicted)**2))),
                    heldout_log_predictions=predicted,observed_log_derivatives=y.tolist(),
                    negative_power_bootstrap_count=int((sample<=0).sum())))
    write_json(output/'analysis.json',dict(status='complete',schema='physical-fit-sensitivity-v1',rows=rows,
        design_sha256=sha256(output/'design.json'),analyzer_sha256=sha256(__file__),
        inputs={str(p):sha256(p) for p in [source,source_manifest,chain_file]},
        scope='Reanalysis of saved reference-chain sufficient statistics. Not additional independent chains or a native critical-exponent estimate.'))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.study,a.output)
