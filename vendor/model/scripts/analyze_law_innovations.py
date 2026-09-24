#!/usr/bin/env python
"""Calibrate joint innovation histories before new native validation starts."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

from model_rg.provenance import sha256, write_json


def read_paths(study,case):
    folder=study/'runs'/case['name'];r=json.loads((folder/'results.json').read_text());rows=[]
    for entry in r['records']:
        path=folder/entry['raw'];assert sha256(path)==entry['sha256']
        with np.load(path) as raw:rows.append(raw['q'])
    return np.stack(rows),r


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--mode',choices=['fit','score'],required=True)
    a=p.parse_args();study=Path(a.study).resolve();spec=json.loads((study/'protocol.json').read_text())
    fitpath=study/'frozen-innovations.json'
    if a.mode=='fit':
        if fitpath.exists():raise FileExistsError(fitpath)
        if any((study/'runs'/c['name']).exists() for c in spec['cases'] if c['role']=='validation'):
            raise AssertionError('Innovation law must be frozen before validation starts')
        models={};inputs={}
        for heads in [4,14]:
            for step in [8192,32768]:
                histories=[];drifts=[]
                for case in spec['cases']:
                    if (case['heads'],case['step'],case['role'])!=(heads,step,'calibration'):continue
                    q,r=read_paths(study,case);q=q[:,:,:r['row_predictive_dimension']]
                    increment=np.diff(q,axis=1);mean=increment.mean(0)
                    histories.append(increment-mean);drifts.append(mean)
                    inputs[case['name']]=sha256(study/'runs'/case['name']/'results.json')
                models[f'h{heads}-t{step}']=dict(drift=np.mean(drifts,axis=0).tolist(),
                    centered_histories=np.concatenate(histories).tolist())
        write_json(fitpath,dict(status='frozen',schema='joint-innovation-closure-v1',
            frozen_at=datetime.now(timezone.utc).isoformat(),analyzer_sha256=sha256(__file__),
            protocol_sha256=sha256(study/'protocol.json'),calibration_inputs=inputs,models=models,
            derivation='A deterministic mean kernel has no conditional branch fluctuations at a fixed incoming '
                'state. Lift translation by an independently sampled empirical innovation history, centered '
                'within each calibration initialization; retain that history as memory. A white control '
                'samples each interval marginal independently. Both use the same frozen conditional mean.',
            frozen_assessment='Retain all held-out initialization/size/time/horizon cells. Compare absolute '
                'mean error and scalar risk variance for white and joint histories; report signed cross-time '
                'variance and empirical distribution distance. These are finite predictive discrepancies, '
                'without a covariance acceptance threshold or asymptotic uncertainty claim. No critical fit.'))
        print('Frozen joint and independent innovation laws before validation',flush=True)
        return
    frozen=json.loads(fitpath.read_text())
    assert frozen['analyzer_sha256']==sha256(__file__)
    rows=[]
    for case in spec['cases']:
        if case['role']!='validation':continue
        q,r=read_paths(study,case)
        assert datetime.fromisoformat(r['started_at'])>datetime.fromisoformat(frozen['frozen_at'])
        m=frozen['models'][f"h{case['heads']}-t{case['step']}"]
        drift=np.array(m['drift'])
        histories=np.array(m['centered_histories'])
        increment=np.diff(q[:,:,0],axis=1)
        for k,horizon in enumerate(spec['horizons'][1:]):
            actual=q[:,k+1,0]
            mean_prediction=float(q[0,0,0]+drift[:k+1,0].sum())
            colored=mean_prediction+histories[:,:k+1,0].sum(1)
            white_variance=float(histories[:,:k+1,0].var(axis=0,ddof=0).sum())
            centered=increment[:,:k+1]-increment[:,:k+1].mean(0)
            covariance=centered.T@centered/len(centered)
            actual_variance=float(actual.var(ddof=0));reconstructed=float(covariance.sum())
            assert abs(actual_variance-reconstructed)<=1e-12*max(1,actual_variance)
            diagonal=float(np.trace(covariance));cross=reconstructed-diagonal
            rows.append(dict(case=case['name'],seed=case['seed'],heads=case['heads'],step=case['step'],horizon=horizon,
                mean_risk_error=float(mean_prediction-actual.mean()),
                actual_risk_variance=actual_variance,white_predicted_variance=white_variance,
                history_predicted_variance=float(colored.var(ddof=0)),
                history_wasserstein_empirical=float(wasserstein_distance(actual,colored)),
                diagonal_increment_variance=diagonal,cross_interval_variance=cross,
                cross_fraction=cross/actual_variance if actual_variance>0 else None,
                exact_increment_variance_residual=actual_variance-reconstructed))
    path=study/'analysis/innovations.json'
    if path.exists():raise FileExistsError(path)
    write_json(path,dict(status='complete',schema='joint-innovation-validation-v1',
        frozen_fit_sha256=sha256(fitpath),analyzer_sha256=sha256(__file__),records=rows,
        scope='All held-out states; conditional population divisors over the recorded branch ensembles. '
            'Joint innovation histories form a finite conditional-memory coarse process. Exact covariance '
            'reconstruction checks the transport algebra, while predictive variances use calibration alone.'))
    print('Scored',len(rows),'joint innovation cells',flush=True)


if __name__=='__main__':main()
