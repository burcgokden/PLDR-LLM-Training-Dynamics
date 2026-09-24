#!/usr/bin/env python
"""Freeze nonexpansive coarse maps, then score untouched initialization identities."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

from model_rg.provenance import sha256, write_json


def load_case(study,case):
    folder=study/'runs'/case['name'];result=json.loads((folder/'results.json').read_text())
    assert result['case']==case and result['status']=='complete'
    rows=[]
    for record in result['records']:
        path=folder/record['raw']
        assert sha256(path)==record['sha256']
        with np.load(path) as data:rows.append(data['q'])
    return np.stack(rows),result


def fit(study,spec):
    if (study/'frozen-fit.json').exists():raise FileExistsError(study/'frozen-fit.json')
    if any((study/'runs'/c['name']).exists() for c in spec['cases'] if c['role']=='validation'):
        raise AssertionError('Validation started before maps were frozen')
    maps={};inputs={};calibration_metrics=[]
    for heads in [4,14]:
        for step in [8192,32768]:
            cases=[c for c in spec['cases'] if c['heads']==heads and c['step']==step and c['role']=='calibration']
            data=[]
            for c in cases:
                q,r=load_case(study,c);data.append(q)
                inputs[c['name']]=sha256(study/'runs'/c['name']/'results.json')
            full=np.concatenate(data)
            for variant in spec['analysis']['variants']:
                dimension=full.shape[-1] if variant=='optimizer_augmented' else r['row_predictive_dimension']
                y=full[:,:,:dimension]
                center=y.reshape(-1,dimension).mean(0)
                floor=1e-4*np.maximum(1,np.abs(center));floor[0]=.01
                scale=np.maximum(y.reshape(-1,dimension).std(0),floor)
                u=(y-center)/scale;matrices=[];offsets=[];norms=[]
                for k in range(4):
                    x=u[:,k];target=u[:,k+1];xc=x-x.mean(0);delta=target-x
                    if variant in ['identity','translation']:
                        matrix=np.eye(dimension)
                    else:
                        change=np.linalg.solve(xc.T@xc+np.eye(dimension),xc.T@(delta-delta.mean(0)))
                        uu,ss,vv=np.linalg.svd(np.eye(dimension)+change,full_matrices=False)
                        matrix=(uu*np.minimum(ss,1.))@vv
                    offset=np.zeros(dimension) if variant=='identity' else (target-x@matrix).mean(0)
                    matrices.append(matrix.tolist());offsets.append(offset.tolist())
                    norms.append(float(np.linalg.norm(matrix,2)))
                    assert norms[-1]<=1+1e-12
                key=f'h{heads}-t{step}-{variant}'
                maps[key]=dict(dimension=dimension,center=center.tolist(),scale=scale.tolist(),
                    matrices=matrices,offsets=offsets,lipschitz=norms)
    result=dict(status='frozen',schema='nonexpansive-law-closure-fit-v1',
        frozen_at=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha256(study/'protocol.json'),
        analyzer_sha256=sha256(__file__),calibration_inputs=inputs,maps=maps,
        rule=spec['analysis'])
    write_json(study/'frozen-fit.json',result)
    print('Frozen',len(maps),'coarse models from calibration identities only',flush=True)


def score(study,spec):
    fit_path=study/'frozen-fit.json';frozen=json.loads(fit_path.read_text())
    if frozen['analyzer_sha256']!=sha256(__file__):raise AssertionError('Frozen scoring algorithm changed')
    output=study/'analysis';output.mkdir(exist_ok=False);rows=[];input_hashes={}
    for case in spec['cases']:
        if case['role']!='validation':continue
        full,result=load_case(study,case)
        assert datetime.fromisoformat(result['started_at'])>datetime.fromisoformat(frozen['frozen_at'])
        input_hashes[case['name']]=sha256(study/'runs'/case['name']/'results.json')
        for variant in spec['analysis']['variants']:
            m=frozen['maps'][f"h{case['heads']}-t{case['step']}-{variant}"]
            dim=m['dimension'];center=np.array(m['center']);scale=np.array(m['scale'])
            exact=(full[:,:,:dim]-center)/scale
            prediction=exact[:,0].copy();bound=np.zeros(len(exact));residuals=[]
            for k,horizon in enumerate(spec['horizons'][1:]):
                matrix=np.array(m['matrices'][k]);offset=np.array(m['offsets'][k])
                residual=exact[:,k+1]-(exact[:,k]@matrix+offset)
                residuals.append(residual)
                costs=np.linalg.norm(residual,axis=-1)
                bound=m['lipschitz'][k]*bound+costs
                prediction=prediction@matrix+offset
                errors=np.linalg.norm(prediction-exact[:,k+1],axis=-1)
                if np.any(errors>bound+1e-10):raise AssertionError('Empirical telescoping inequality failed')
                risk_prediction=prediction[:,0]*scale[0]+center[0]
                risk_actual=full[:,k+1,0];risk_error=risk_prediction-risk_actual
                covariance=np.cov(exact[:,k+1],rowvar=False,bias=True)
                predicted_covariance=np.cov(prediction,rowvar=False,bias=True)
                cov_norm=float(np.linalg.norm(covariance))
                rows.append(dict(case=case['name'],heads=case['heads'],step=case['step'],seed=case['seed'],
                    variant=variant,horizon=horizon,branches=len(exact),
                    observed_mean_risk=float(risk_actual.mean()),predicted_mean_risk=float(risk_prediction.mean()),
                    mean_risk_error=float(risk_error.mean()),risk_rmse=float(np.sqrt(np.mean(risk_error**2))),
                    conditional_risk_sd=float(risk_actual.std(ddof=0)),
                    initial_to_horizon_risk_change=float((risk_actual-full[:,0,0]).mean()),
                    risk_wasserstein_empirical=float(wasserstein_distance(risk_actual,risk_prediction)),
                    retained_pairing_cost_mean=float(errors.mean()),local_cost_mean=float(costs.mean()),
                    telescoping_bound_mean=float(bound.mean()),risk_bound_mean=float(scale[0]*bound.mean()),
                    risk_mean_target_met=bool(abs(risk_error.mean())<=.01),
                    covariance_norm=cov_norm,covariance_error=float(np.linalg.norm(covariance-predicted_covariance)),
                    covariance_relative_error=float(np.linalg.norm(covariance-predicted_covariance)/cov_norm) if cov_norm>0 else None,
                    residual_adjacent_inner_mean=float(np.mean(np.sum(residuals[-2]*residuals[-1],axis=-1))) if k else None,
                    lipschitz=m['lipschitz'][k],emission_lipschitz=float(scale[0])))
    write_json(output/'results.json',dict(status='complete',schema='conditional-law-closure-analysis-v1',
        protocol_sha256=sha256(study/'protocol.json'),frozen_fit_sha256=sha256(fit_path),
        analyzer_sha256=sha256(__file__),validation_inputs=input_hashes,records=rows,
        validation_initializations=2,calibration_initializations=2,incoming_states=16,
        scientific_branches=256,scientific_updates=16384,
        scope='All prespecified models, state identities and horizons are retained. The coupling '
            'bound is exact for these paired empirical trajectories, conditional on their recorded '
            'arithmetic. Costs under the native law are estimated by branch sampling, not uniformly '
            'certified. Two held-out initialization identities do not support asymptotic coverage.'))
    print('Scored',len(rows),'complete model/state/horizon cells',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--mode',choices=['fit','score'],required=True)
    a=p.parse_args();study=Path(a.study).resolve();spec=json.loads((study/'protocol.json').read_text())
    (fit if a.mode=='fit' else score)(study,spec)


if __name__=='__main__':main()
