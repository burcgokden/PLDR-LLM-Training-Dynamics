#!/usr/bin/env python
"""Score frozen clock predictions on all eight completed schedule-only paths."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from analyze_size_time import empirical_weights, whole_seed_statistics
from freeze_scaling_forecasts import crossing_interval, log_interval_probability
from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True)
    p.add_argument('--study', default='scheduled-training-20260908'); a = p.parse_args()
    study = Path(a.root).resolve()/a.study; frozen = study/'analysis/frozen-clocks'
    analysis_protocol=study/'protocols/scheduled-analysis-selection.json'
    contract=json.loads(analysis_protocol.read_text())
    for name,digest in contract['producer_sources'].items():
        if sha256(Path(__file__).resolve().parents[1]/name)!=digest:
            raise AssertionError('A recorded scheduled clock analysis implementation changed')
    selected = study/'protocols/regime-training-selection.json'
    spec = json.loads(selected.read_text()); jobs = {j['run_id']:j for j in spec['jobs'] if j['recipe']=='controlled'}
    forecast = json.loads((frozen/'results.json').read_text()); fm = json.loads((frozen/'manifest.json').read_text())
    if forecast['status'] != 'frozen_before_controlled_targets' or set(forecast['controlled_target_ids']) != set(jobs) or len(jobs)!=8:
        raise AssertionError('The exact eight target identities must have frozen predictions')
    for name, key in [('results.json','results_sha256'),('measurements.npz','raw_sha256'),('binding.json','binding_sha256')]:
        if sha256(frozen/name) != fm[key]: raise AssertionError('A frozen clock prediction changed')
    inputs = [selected,analysis_protocol,*[frozen/n for n in ['manifest.json','binding.json','results.json','measurements.npz']]]
    observations = json.loads((study/'protocols/regime-observation-selection.json').read_text())
    states = {(c['heads'],c['seed'],c['step']):c for c in observations['cases'] if c['recipe']=='controlled'}
    dense = {}
    for name, job in jobs.items():
        folder = study/'runs'/name; meta = json.loads((folder/'manifest.json').read_text())
        binding = json.loads((folder/'binding.json').read_text())
        if meta['status'] != 'complete' or meta['completed_step'] != 262144:
            raise AssertionError('Every controlled native target trajectory must be complete')
        if datetime.fromisoformat(forecast['frozen_at']) >= datetime.fromisoformat(binding['environment']['utc']):
            raise AssertionError('The target began before its forecast was frozen')
        if sha256(folder/'measurements.npz') != meta['raw_sha256']: raise AssertionError('A completed clock target changed')
        inputs += [folder/'manifest.json',folder/'binding.json',folder/'measurements.npz']
        with np.load(folder/'measurements.npz') as z:
            dense[job['heads'],job['seed']] = (z['steps'], z['dense_heads'][...,2].astype(float).mean(axis=(1,2,3)))
    for case in states.values():
        folder = study/'measurements'/case['name']; verification = study/'verification/observations'/(case['name']+'.json')
        check = json.loads(verification.read_text())
        if check['status'] != 'passed' or check['case'] != case: raise AssertionError('Every target CPU observation must be verified')
        for name in ['manifest.json','measurements.npz']:
            path = folder/name
            if sha256(path) != check['checked_sha256'][str(path)]: raise AssertionError('A verified clock target changed')
            inputs.append(path)
        inputs.append(verification)
    out = study/'analysis/clock-scores'; out.mkdir(parents=True,exist_ok=False)
    bind_run(out,inputs,vars(a)); scores = []; raw = {}; weights = empirical_weights(4)
    with np.load(frozen/'measurements.npz') as predictions:
        for record in forecast['forecasts']:
            result = {k:v for k,v in record.items() if k!='observables'}
            if record['status'] != 'frozen_prediction': scores.append(result); continue
            n = record['heads']; step = record['step']; model = record['model']; arrays = []
            for seed in range(640101,640105):
                case = states[n,seed,step]
                with np.load(study/'measurements'/case['name']/'measurements.npz') as z:
                    h=z['float32_heads'].astype(float); f=z['float32_fields'].astype(float)
                    c=z['float32_centroids']; e=z['float32_energies']
                arrays.append(dict(row=h[...,2].mean(-1),common_centroid=c.mean(-2),absolute_row_energy=e[...,0].mean(-1),
                    attention=h[...,0].mean(-1),operator_rms=h[...,3].mean(-1),nll=f[...,25],
                    prediction_entropy=f[...,24],logit_projection=f[...,16:24]))
            result['observables'] = {}
            for field in arrays[0]:
                actual = np.stack([r[field] for r in arrays]); label = f'h{n}_t{step}_{model}_{field}'
                predicted = predictions[label]; error = actual-predicted
                astat, aboot = whole_seed_statistics(actual,n); pstat, pboot = whole_seed_statistics(predicted,n)
                estat, _ = whole_seed_statistics(error,n)
                seed_means=error.reshape(4,-1).mean(-1); boot_means=weights@seed_means/4
                result['observables'][field] = dict(actual=astat,predicted=pstat,mean_error=float(error.mean()),
                    paired_field_rmse=float(np.sqrt(np.mean(error*error))),
                    mean_error_empirical_percentiles=np.quantile(boot_means,[.025,.975]).tolist(),
                    susceptibility_error=astat['susceptibility']-pstat['susceptibility'],
                    susceptibility_error_empirical_percentiles=np.quantile(aboot-pboot,[.025,.975]).tolist(),
                    error_susceptibility=estat['susceptibility'])
                raw[label+'_paired_mean_error_bootstrap']=boot_means
                raw[label+'_paired_susceptibility_error_bootstrap']=aboot-pboot
            scores.append(result)
        passages=[]
        for record in forecast['first_passages']:
            n=record['heads']; seed=record['seed']; threshold=record['threshold']; model=record['model']
            times,values=dense[n,seed]; low,high=crossing_interval(times,values,threshold)
            clock=predictions[f'first_passage_{threshold}_{model}_clock']
            lo=np.log(max(float(clock[int(low)]),1e-30)); hi=np.log(float(clock[int(high)])) if high is not None else np.inf
            mean=np.log(record['median_reference_clock']); scale=record['working_log_scale']
            log_probability=float(log_interval_probability(np.array([(lo-mean)/scale]),np.array([(hi-mean)/scale]))[0])
            passages.append(dict(**record,observed_lower=low,observed_upper=high,
                observed_lower_clock=float(clock[int(low)]),observed_upper_clock=float(clock[int(high)]) if high is not None else None,
                negative_log_working_probability=-log_probability))
    write_json(out/'results.json',dict(status='complete',schema='scheduled-clock-scores-v1',
        trajectories=8,forecasts=scores,first_passages=passages,
        scope='Every frozen path-clock alternative and interval-censored first-passage prediction is retained, including predictions outside the available reference horizon. No scheduled target was used to refit the clocks. Paired field RMSE uses the declared initialization coupling and is not an optimized marginal transport distance.',
        uncertainty='Exact enumeration of all 256 whole-seed empirical resamples preserves each initialization across prediction and target. Percentile ranges have no guaranteed population coverage. First-passage probabilities belong to the inherited marginal working model; correlated thresholds and shared paths are not treated as independent populations.'))
    np.savez_compressed(out/'measurements.npz',**raw)
    write_json(out/'manifest.json',dict(status='complete',binding_sha256=sha256(out/'binding.json'),
        results_sha256=sha256(out/'results.json'),raw_sha256=sha256(out/'measurements.npz')))
    print('Scored all eight controlled scheduled trajectories',flush=True)


if __name__=='__main__':main()
