#!/usr/bin/env python
"""Score every frozen native clock and joint-width forecast without refitting."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linprog

from analyze_size_time import empirical_weights, whole_seed_statistics
from freeze_scaling_forecasts import full_observations
from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


def paired_score(actual,predicted,n):
    sa,ba=whole_seed_statistics(actual,n);sp,bp=whole_seed_statistics(predicted,n)
    difference=(actual-predicted).reshape(4,-1)
    per_seed=difference.mean(-1)
    means=empirical_weights(4)@per_seed/4
    return dict(actual=sa,predicted=sp,mean_error=float(per_seed.mean()),
        mean_error_empirical_percentiles=np.quantile(means,[.025,.975]).tolist(),
        mean_error_seed_se=float(per_seed.std(ddof=1)/2),
        field_rmse=float(np.sqrt(np.mean(difference**2))),
        field_rmse_by_seed=np.sqrt(np.mean(difference**2,axis=-1)).tolist(),
        susceptibility_error=sa['susceptibility']-sp['susceptibility'],
        susceptibility_error_empirical_percentiles=np.quantile(ba-bp,[.025,.975]).tolist()),ba-bp


def interval_compatible_powers(records):
    """Exact finite log-interval feasibility, not a confidence interval."""
    widths=sorted({r['heads'] for r in records});seeds=sorted({r['seed'] for r in records})
    rows=[];bounds=[]
    for record in records:
        row=[float(record['heads']==n) for n in widths]+[float(record['seed']==s) for s in seeds[1:]]
        row.append(-np.log(record['multiplier']));row=np.array(row)
        if record['upper'] is not None:
            rows.append(row);bounds.append(np.log(record['upper']))
        if record['lower']>0:
            rows.append(-row);bounds.append(-np.log(record['lower']))
    box=[(0,20)]*len(widths)+[(-5,5)]*(len(seeds)-1)+[(-2,5)]
    solutions=[]
    for sign in [1,-1]:
        objective=np.zeros(len(box));objective[-1]=sign
        result=linprog(objective,A_ub=np.array(rows),b_ub=np.array(bounds),bounds=box,method='highs')
        solutions.append(dict(success=bool(result.success),message=result.message,
                              power=float(result.x[-1]) if result.success else None))
    return dict(widths=widths,lower=solutions[0],upper=solutions[1],
        interpretation='Powers compatible with every observation interval at zero residual, using the same width and initialization fixed effects and coefficient bounds as the frozen working model. Not a statistical coverage statement.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='critical-scaling-20260906');p.add_argument('--output',required=True)
    p.add_argument('--wait',action='store_true');a=p.parse_args()
    root=Path(a.root);study=root/a.study
    predecessor=study/'launcher-prediction-tests.json'
    while True:
        status=json.loads(predecessor.read_text())['status'] if predecessor.exists() else 'waiting'
        if status=='complete':break
        if not a.wait or status not in ['waiting','running']:raise RuntimeError('Prediction executor: '+status)
        time.sleep(30)
    frozen=study/'analysis/frozen-forecasts';forecast=json.loads((frozen/'results.json').read_text())
    fm=json.loads((frozen/'manifest.json').read_text())
    for filename,key in [('results.json','results_sha256'),('measurements.npz','raw_sha256')]:
        if sha256(frozen/filename)!=fm[key]:raise AssertionError('Frozen predictions changed')
    paths={};inputs=[predecessor,*[frozen/name for name in ['manifest.json','results.json','measurements.npz','binding.json']]]
    for protocol_name in ['clock-holdout.json','diffusive-size-map.json']:
        protocol=study/'protocols'/protocol_name
        if sha256(protocol)!=forecast['protocol_sha256'][protocol_name]:raise AssertionError('Frozen target protocol changed')
        inputs.append(protocol)
        for job in json.loads(protocol.read_text())['jobs']:
            path=study/'runs'/job['run_id']/'manifest.json';meta=json.loads(path.read_text())
            if meta['status']!='complete':raise AssertionError('An expected target is incomplete')
            if sha256(path.parent/'measurements.npz')!=meta['raw_sha256']:raise AssertionError('Target observations changed')
            binding=json.loads((path.parent/'binding.json').read_text())
            if datetime.fromisoformat(forecast['frozen_at']) >= datetime.fromisoformat(binding['environment']['utc']):
                raise AssertionError('Forecast binding does not precede the target producer')
            paths[protocol_name,job['heads'],job['seed']]=path
            inputs.extend([path,path.parent/'measurements.npz',path.parent/'binding.json'])
    refs=[]
    for n,g,t in [(2,2,2048),(8,1,8192)]:
        candidates=[]
        for directory in ['criticality-study-20260905','criticality-dynamics-20260906','observation-closure-20260906']:
            for path in (root/directory/'runs').glob('*/manifest.json'):
                meta=json.loads(path.read_text());c=meta.get('arguments',{})
                if (meta.get('schema') in ['criticality-training-v1','criticality-dynamics-training-v1']
                    and meta['status']=='complete' and c.get('heads')==n and c.get('multiplier')==g
                    and c.get('seed') in range(640101,640105) and c.get('shared_seed')==640011
                    and c.get('stream_seed')==640001 and (n==2 or c.get('normalization')=='variance')
                    and t in meta['milestones']):
                    candidates.append((meta['completed_step'],str(path.parent),path,c['seed']))
        selected={}
        for _,_,path,seed in sorted(candidates):selected[seed]=path
        if sorted(selected)!=list(range(640101,640105)):raise AssertionError('Missing frozen joint reference')
        reference=[full_observations(selected[s],t) for s in range(640101,640105)]
        refs.append({key:np.stack([r[key] for r in reference]) for key in reference[0]})
        for path in selected.values():inputs.extend([path,path.parent/'measurements.npz'])
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    bind_run(out,list(dict.fromkeys(inputs)),vars(a))
    records=[];raw={};actual_cache={}
    with np.load(frozen/'measurements.npz') as z:
        for item in forecast['forecasts']:
            key=item['protocol'],item['heads'],item['step']
            record={k:item[k] for k in ['protocol','heads','multiplier','step','model','power','reference_step','status']}
            if item['status']!='frozen_prediction':
                records.append(record);continue
            if key not in actual_cache:
                rows=[full_observations(paths[item['protocol'],item['heads'],s],item['step']) for s in range(640101,640105)]
                actual_cache[key]={field:np.stack([r[field] for r in rows]) for field in rows[0]}
            record['observables']={}
            for field in item['observables']:
                label=f"{Path(item['protocol']).stem}_h{item['heads']}_t{item['step']}_{item['model']}_{field}"
                predicted=z[label];actual=actual_cache[key][field]
                score,bootstrap=paired_score(actual,predicted,item['heads'])
                for moment in ['mean','susceptibility']:
                    np.testing.assert_allclose(score['predicted'][moment],item['observables'][field][moment],rtol=1e-12,atol=1e-20)
                record['observables'][field]=score;raw[label+'_error_bootstrap']=bootstrap
                raw[label+'_actual']=actual
            records.append(record)
    joint=[]
    for item in forecast['joint_clock_forecasts']:
        n=item['heads'];x=(1/n-1/8)/(1/2-1/8)
        actual=actual_cache['diffusive-size-map.json',n,item['step']]
        record=dict(heads=n,multiplier=item['multiplier'],step=item['step'],observables={})
        for field,target in item['observables'].items():
            stats,bootstrap=whole_seed_statistics(actual[field],n)
            reference_stats=[whole_seed_statistics(r[field],n) for r in refs]
            prediction=(1-x)*reference_stats[1][0]['susceptibility']+x*reference_stats[0][0]['susceptibility']
            predicted_bootstrap=(1-x)*reference_stats[1][1]+x*reference_stats[0][1]
            expected_mean=(1-x)*refs[1][field].mean()+x*refs[0][field].mean()
            np.testing.assert_allclose([prediction,expected_mean],[target['susceptibility'],target['mean']],rtol=1e-12,atol=1e-20)
            difference=actual[field].reshape(4,-1).mean(-1)-((1-x)*refs[1][field].reshape(4,-1).mean(-1)+x*refs[0][field].reshape(4,-1).mean(-1))
            mean_bootstrap=empirical_weights(4)@difference/4
            record['observables'][field]=dict(actual=stats,predicted=target,
                mean_error=float(difference.mean()),mean_error_empirical_percentiles=np.quantile(mean_bootstrap,[.025,.975]).tolist(),
                susceptibility_error=stats['susceptibility']-prediction,
                susceptibility_error_empirical_percentiles=np.quantile(bootstrap-predicted_bootstrap,[.025,.975]).tolist())
            raw[f'joint_h{n}_{field}_error_bootstrap']=bootstrap-predicted_bootstrap
        joint.append(record)
    primary=[r for r in forecast['crossing_intervals'] if r['threshold']==.5]
    calibration=[interval_compatible_powers(primary)]+[interval_compatible_powers([r for r in primary if r['heads']==n]) for n in [2,4,8,14]]
    result=dict(schema='frozen-scaling-forecast-scores-v1',status='complete',forecasts=records,joint_forecasts=joint,
        interval_identifiability=calibration,
        uncertainty='Exact 256 whole-initialization empirical resamples, paired across actual and reference fields, all contexts and all widths. Percentiles have no guaranteed population coverage.',
        validation='Every frozen alternative and target horizon is scored without fitting target observations. Joint inverse-N forecasts use only the two frozen calibration widths.',
        scope='Finite conditional rate-time and joint-clock prediction tests, not an identification of a thermodynamic critical surface or a native universality class.')
    write_json(out/'results.json',result);np.savez_compressed(out/'measurements.npz',**raw)
    write_json(out/'manifest.json',dict(status='complete',binding_sha256=sha256(out/'binding.json'),
        results_sha256=sha256(out/'results.json'),raw_sha256=sha256(out/'measurements.npz'),
        forecasts=len(records),joint_forecasts=len(joint)))


if __name__=='__main__':main()
