#!/usr/bin/env python
"""Reconstruct fixed-window temporal laws of passive native update observations."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


def covariance_diagnostics(values, spacing):
    """Finite temporal second moments, with both mean and linear trend removal."""
    x=np.asarray(values,dtype=float).reshape(len(values),-1)
    length=len(x)
    if length<4:raise ValueError('Insufficient temporal observations')
    mean=x.mean(0);centered=x-mean
    coordinate=np.arange(length,dtype=float)-(length-1)/2
    slope=coordinate@centered/(coordinate@coordinate)
    detrended=centered-coordinate[:,None]*slope[None]
    lags=[0]+[2**k for k in range(20) if 2**k<=length//4]
    result=dict(samples=length,spacing=spacing,coordinates=x.shape[1],
        mean_squared=float(np.mean(mean**2)),variance=float(np.mean(centered**2)),
        endpoint_displacement_squared=float(np.mean((x[-1]-x[0])**2)),
        linear_slope_squared_per_update=float(np.mean(slope**2)/spacing**2),
        linear_trend_variance=float(np.mean((coordinate[:,None]*slope[None])**2)),
        detrended_variance=float(np.mean(detrended**2)),lags=[lag*spacing for lag in lags])
    for label,signal in [('centered',centered),('detrended',detrended)]:
        covariance=[float(np.mean(signal*signal)) if lag==0 else
                    float(np.mean(signal[:-lag]*signal[lag:])) for lag in lags]
        result[label+'_covariance']=covariance
        result[label+'_correlation']=[value/covariance[0] if covariance[0]>0 else None for value in covariance]
    return result


def blocks(start,end,width):
    first=((start+width-1)//width)*width
    return [(begin,begin+width) for begin in range(first,end-width+1,width)]


def excursion_intervals(steps,values,threshold):
    """Observed above-threshold intervals on the fixed probe mesh, including censoring."""
    steps=np.asarray(steps);values=np.asarray(values)
    active=values>threshold;records=[];begin=None
    for i,state in enumerate(active):
        if state and begin is None:begin=i
        if begin is not None and (not state or i==len(active)-1):
            finish=i-1 if not state else i
            left=None if begin==0 else int(steps[begin-1])
            right=None if finish==len(active)-1 else int(steps[finish+1])
            records.append(dict(first_above=int(steps[begin]),last_above=int(steps[finish]),
                onset_lower=left,onset_upper=int(steps[begin]),offset_lower=int(steps[finish]),offset_upper=right,
                mesh_duration=int(steps[finish]-steps[begin]),peak=float(np.max(values[begin:finish+1])),
                left_censored=left is None,right_censored=right is None))
            begin=None
    return records


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='critical-scaling-20260906');p.add_argument('--output',required=True)
    p.add_argument('--wait',action='store_true');a=p.parse_args();study=Path(a.root)/a.study
    protocol=study/'protocols/shared-update-observation.json';spec=json.loads(protocol.read_text())
    paths=[study/'runs'/name/'manifest.json' for name in spec['run_ids']]
    while not all(path.exists() for path in paths):
        if not a.wait:raise RuntimeError('Passive native update targets incomplete')
        time.sleep(30)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    inputs=[protocol,Path(spec['projection']),Path(spec['qualification'])]
    for path in paths:inputs.extend([path,path.parent/'measurements.npz'])
    bind_run(out,inputs,vars(a));records=[];raw={}
    for path in paths:
        meta=json.loads(path.read_text());condition=meta['arguments'];info=meta['shared_update_observation']
        if meta['status']!='complete' or info is None:raise AssertionError('Missing completed passive observation')
        if info['protocol_sha256']!=sha256(protocol) or info['projection_sha256']!=spec['projection_sha256']:
            raise AssertionError('Passive observation units changed')
        if sha256(path.parent/'measurements.npz')!=meta['raw_sha256']:raise AssertionError('Passive observation raw file changed')
        start=meta['start_step'];end=meta['completed_step'];name=path.parent.name
        record=dict(run_id=name,condition=condition,start_step=start,completed_step=end,windows=[],excursions={})
        with np.load(path.parent/'measurements.npz') as z:
            updates=z['shared_update_projection'];gradients=z['shared_clipped_gradient_projection'];moments=z['shared_update_moments']
            if updates.shape!=(end-start,16) or gradients.shape!=updates.shape or moments.shape!=(end-start,3):
                raise AssertionError('A completed native update was omitted')
            if np.any(moments[:,:2]<0):raise AssertionError('Squared native update moment is negative')
            if condition['multiplier']==0 and (np.any(updates!=0) or np.any(moments[:,0]!=0) or np.any(moments[:,2]!=0)):
                raise AssertionError('A zero-rate shared parameter moved')
            steps=z['steps'];dense_heads=z['dense_heads'];fields=z['dense_fields']
            observed=dict(common_centroid=z['dense_centroids'],absolute_row_energy=z['dense_energies'][...,0],
                total_metric_energy=z['dense_energies'][...,1],row=dense_heads[...,2].mean(-1),
                attention=dense_heads[...,0].mean(-1),prediction_entropy=fields[...,24],
                nll=fields[...,25],logit_projection=fields[...,16:24])
            if not np.array_equal(steps,np.arange(start,end+1,64)):
                # Joint-clock endpoints also lie on this mesh for the frozen widths.
                raise AssertionError('Temporal cohort mesh changed')
            for width in [2048,8192,32768]:
                for begin,finish in blocks(start,end,width):
                    sl=slice(begin-start,finish-start)
                    row=dict(begin=begin,end=finish,width=width,observables={})
                    for label,values in [('shared_update_projection',updates[sl]),('shared_clipped_gradient_projection',gradients[sl])]:
                        row['observables'][label]=covariance_diagnostics(values,1)
                    m=moments[sl].mean(0)
                    row['parameter_moments']=dict(mean_squared_update=float(m[0]),mean_squared_weight=float(m[1]),
                        mean_update_times_weight=float(m[2]),
                        radial_rate=float(m[2]/m[1]) if m[1]>0 else None,
                        nominal_decay_rate=3e-6*condition['multiplier'])
                    mask=(steps>=begin)&(steps<finish)
                    if np.sum(mask)!=width//64:raise AssertionError('A temporal window changed its probe count')
                    for label,values in observed.items():
                        row['observables'][label]=covariance_diagnostics(values[mask],64)
                    record['windows'].append(row)
            row_path=observed['row'].mean((1,2))
            for threshold in [.1,.01,.001]:
                record['excursions'][str(threshold)]=excursion_intervals(steps,row_path,threshold)
            record['whole_path']=dict(row_minimum=float(row_path.min()),row_maximum=float(row_path.max()),
                row_last=float(row_path[-1]),mean_loss=float(z['losses'].mean()),
                projected_weight_displacement_squared=float(np.mean(updates.sum(0)**2)),
                mean_squared_native_update=float(moments[:,0].mean()))
            raw[name+'_steps']=steps;raw[name+'_row']=row_path
            raw[name+'_common_centroid_mean']=observed['common_centroid'].mean(1)
            raw[name+'_prediction_entropy_mean']=observed['prediction_entropy'].mean(1)
        records.append(record);print(name,'temporal reconstruction complete',flush=True)
    result=dict(schema='passive-native-temporal-laws-v1',status='complete',runs=records,
        windows='Every complete absolute-time-aligned block of 2048, 8192 and 32768 updates. Probe windows are left-closed and right-open on the fixed 64-update mesh.',
        covariance_scope='Finite within-window temporal covariances at fixed coordinates, with separately reported mean removal and linear detrending. These are neither independent pretraining replicates nor certified stationary covariance estimates.',
        excursions='All above-threshold row-cohort intervals at thresholds 0.1, 0.01 and 0.001, with mesh censoring. They are concentration excursions, not an identified avalanche process; no avalanche exponent is inferred.',
        conditioning='All 60 declared targets retain the same signed parameter projections and fixed probe cohort. Generator and body clocks, environment and incoming moments remain part of each condition.')
    write_json(out/'results.json',result);np.savez_compressed(out/'measurements.npz',**raw)
    write_json(out/'manifest.json',dict(status='complete',binding_sha256=sha256(out/'binding.json'),
        results_sha256=sha256(out/'results.json'),raw_sha256=sha256(out/'measurements.npz'),runs=len(records)))


if __name__=='__main__':main()
