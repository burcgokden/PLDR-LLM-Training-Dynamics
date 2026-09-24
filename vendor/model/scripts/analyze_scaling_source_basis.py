#!/usr/bin/env python
"""Compare all frozen source bases on the same second fresh-batch sample."""
import argparse
from datetime import datetime
import itertools
import json
from pathlib import Path
import time

import numpy as np

from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


def read(path):
    return json.loads(Path(path).read_text())


def projection(teacher, coefficients, basis, k, construction):
    k_actual = min(k,len(basis))
    prediction = coefficients[:,:k_actual]@basis[:k_actual]
    residual = teacher-prediction
    mean, residual_mean = teacher.mean(0), residual.mean(0)
    covariance_trace = float(np.sum((teacher-mean)**2)/(len(teacher)-1))
    residual_trace = float(np.sum((residual-residual_mean)**2)/(len(teacher)-1))
    second = float(np.mean(np.sum(teacher**2,axis=1)))
    residual_second = float(np.mean(np.sum(residual**2,axis=1)))
    return dict(construction=construction,requested_dimension=k,retained_dimension=k_actual,
        total_response_second_moment=second,residual_second_moment=residual_second,
        total_response_residual_fraction=residual_second/second if second else None,
        covariance_trace=covariance_trace,residual_covariance_trace=residual_trace,
        predicted_covariance_trace=float(np.var(prediction,axis=0,ddof=1).sum()),
        covariance_residual_fraction=residual_trace/covariance_trace if covariance_trace else None,
        mean_response_squared=float(mean@mean),mean_response_error_squared=float(residual_mean@residual_mean))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='critical-scaling-20260906');p.add_argument('--output',required=True)
    p.add_argument('--wait',action='store_true');a=p.parse_args();study=Path(a.root)/a.study
    launcher=study/'launcher-source-basis.json'
    while not launcher.exists() or read(launcher)['status']!='complete':
        status=read(launcher)['status'] if launcher.exists() else 'waiting'
        if not a.wait or status not in ['waiting','running']:
            raise RuntimeError('The expanded source executor requires resolution: '+status)
        time.sleep(30)
    protocol=study/'protocols/source-basis.json';spec=read(protocol)
    reference_path=study/'protocols/source-basis-reference.json';reference=read(reference_path)
    if sha256(protocol)!=reference['target_protocol_sha256']:raise AssertionError('The paired reference target changed')
    lookup={r['name']:r for r in reference['cases']}
    if set(lookup)!={c['name'] for c in spec['cases']}:raise AssertionError('The reference selection differs from the expanded selection')
    inputs=[launcher,protocol,reference_path]
    for case in spec['cases']:
        folder=study/'measurements'/case['name'];old=Path(lookup[case['name']]['reference'])
        inputs.extend(folder/name for name in ['manifest.json','results.json','predictor.json','predictor.npz',
                                               'measurements.npz','validation-directions.npy','binding.json'])
        inputs.extend(old/name for name in ['manifest.json','predictor.json','predictor.npz'])
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);bind_run(out,inputs,vars(a))
    rows=np.random.default_rng(spec['validation_batch_seed']+1000).integers(0,3072,size=(spec['validation_batches'],32))
    offsets=np.random.default_rng(spec['validation_batch_seed']+1001000).integers(0,449,size=rows.shape)
    records,raw=[],{}
    dimensions={**{name:spec['dimensions'] for name in spec['constructions']},
                'eight_batch_reference':reference['dimensions']}
    locality={str(x):dict(total=0,passed=0) for x in spec['amplitudes']}
    for case in spec['cases']:
        path=study/'measurements'/case['name'];meta=read(path/'manifest.json');result=read(path/'results.json')
        if meta['status']!='complete' or meta['case']!=case:raise AssertionError('A selected source outcome is incomplete')
        for file,key in [('results.json','results_sha256'),('predictor.npz','predictor_sha256'),
                         ('predictor.json','predictor_record_sha256'),('measurements.npz','raw_sha256'),
                         ('validation-directions.npy','directions_sha256')]:
            if sha256(path/file)!=meta[key]:raise AssertionError('A source artifact changed')
        if datetime.fromisoformat(reference['frozen_at'])>=datetime.fromisoformat(read(path/'binding.json')['environment']['utc']):
            raise AssertionError('The paired eight-batch reference was selected after an expanded target began')
        with np.load(path/'predictor.npz') as predictor:
            basis,adjoints,rotation,images=[predictor[k] for k in ['basis','adjoints','covariance_rotation','calibration_images']]
            eigenvalues=predictor['covariance_eigenvalues']
            np.testing.assert_allclose(basis@basis.T,np.eye(len(basis)),rtol=1e-10,atol=1e-12)
            np.testing.assert_allclose(rotation@rotation.T,np.eye(len(rotation)),rtol=1e-10,atol=1e-12)
            image=images.reshape(len(images),-1);calibration_coefficients=image@basis.T
            centered=calibration_coefficients-calibration_coefficients.mean(0)
            rotated=centered@rotation.T
            np.testing.assert_allclose(rotated.T@rotated/(len(images)-1),np.diag(eigenvalues),rtol=2e-8,atol=1e-13)
            cal_error=float(np.linalg.norm(image-calibration_coefficients@basis)/max(np.linalg.norm(image),1e-300))
            if cal_error>2e-11:raise AssertionError('The calibration response span is not preserved')
        directions=np.load(path/'validation-directions.npy',mmap_mode='r')
        coefficients=np.asarray(directions)@adjoints.T
        with np.load(path/'measurements.npz') as z:
            np.testing.assert_array_equal(z['rows'],rows);np.testing.assert_array_equal(z['offsets'],offsets)
            np.testing.assert_array_equal(z['cohort'],np.arange(512,528))
            teacher=z['teacher']
            np.testing.assert_allclose(coefficients,z['source_coefficients'],rtol=2e-8,atol=1e-11)
            oracle=teacher@basis.T
            error=np.linalg.norm(coefficients-oracle,axis=1)
            if np.any(error>spec['duality_absolute_tolerance']+spec['duality_relative_tolerance']*np.linalg.norm(oracle,axis=1)):
                raise AssertionError('The expanded source adjoint fails a fresh direction')
            computed=[]
            for construction in spec['constructions']:
                b,c=(basis,coefficients) if construction=='total_response' else (rotation@basis,coefficients@rotation.T)
                for k in dimensions[construction]:
                    r=projection(teacher,c,b,k,construction)
                    original=next(x for x in result['projections'] if x['construction']==construction and x['requested_dimension']==k)
                    for key,value in r.items():
                        if isinstance(value,str):continue
                        if value is None:
                            if original[key] is not None:raise AssertionError('Undefined source error changed')
                        else:np.testing.assert_allclose(value,original[key],rtol=2e-8,atol=1e-16)
                    computed.append(r)
            for item in result['numerical']:
                k=item['batch'];scale=float(teacher[k]@teacher[k])
                for finite in item['finite_responses']:
                    amp=finite['amplitude'];secant=z[f'b{k}_a{amp}_secant'];kl=float(z[f'b{k}_a{amp}_symmetric_kl'])
                    first=float(np.linalg.norm(secant-teacher[k])/max(np.sqrt(scale),1e-300))
                    fisher=abs(kl-scale)/max(scale,1e-300)
                    np.testing.assert_allclose([first,fisher],[finite['first_relative_error'],finite['fisher_relative_error']],rtol=2e-8,atol=1e-14)
                    passed=first<=.01 and fisher<=.05
                    if finite['local_diagnostic']!=passed:raise AssertionError('The local response criterion changed')
                    locality[str(amp)]['total']+=1;locality[str(amp)]['passed']+=int(passed)
        old_spec=lookup[case['name']];old=Path(old_spec['reference']);old_meta=read(old/'manifest.json')
        for file,key in [('manifest.json','manifest_sha256'),('predictor.npz','predictor_sha256'),
                         ('predictor.json','predictor_record_sha256')]:
            if sha256(old/file)!=old_spec[key]:raise AssertionError('The frozen eight-batch reference changed')
        if old_meta['case']['parent_sha256']!=case['parent_sha256']:raise AssertionError('The paired incoming state changed')
        with np.load(old/'predictor.npz') as q:
            old_basis,old_adjoints=q['basis'],q['adjoints']
        old_coefficients=np.asarray(directions)@old_adjoints.T
        oracle=teacher@old_basis.T;err=np.linalg.norm(old_coefficients-oracle,axis=1)
        if np.any(err>spec['duality_absolute_tolerance']+spec['duality_relative_tolerance']*np.linalg.norm(oracle,axis=1)):
            raise AssertionError('The original source adjoint fails a new fresh direction')
        for k in dimensions['eight_batch_reference']:
            computed.append(projection(teacher,old_coefficients,old_basis,k,'eight_batch_reference'))
        result['projections']=computed
        result['paired_reference']=dict(**old_spec,maximum_adjoint_duality_error=float(np.max(err)))
        result['calibration_reconstruction_relative_error']=cal_error
        records.append(result)
        raw[case['name']+'_reference_coefficients']=old_coefficients
        raw[case['name']+'_expanded_coefficients']=coefficients
        print(case['name'],'all paired source bases reconstructed',flush=True)
    summaries=[]
    for n in [4,14]:
        for t in [2048,16384]:
            selected=sorted([r for r in records if r['case']['heads']==n and r['case']['step']==t],key=lambda r:r['case']['seed'])
            if [r['case']['seed'] for r in selected]!=list(range(640101,640105)):
                raise AssertionError('The matched state panel changed')
            for construction,dims in dimensions.items():
                for k in dims:
                    values=[next(x for x in r['projections'] if x['construction']==construction and x['requested_dimension']==k) for r in selected]
                    summaries.append(dict(heads=n,step=t,construction=construction,requested_dimension=k,
                        total_response_residual_fractions=[r['total_response_residual_fraction'] for r in values],
                        covariance_residual_fractions=[r['covariance_residual_fraction'] for r in values],
                        pooled_total_response_residual_fraction=sum(r['residual_second_moment'] for r in values)/sum(r['total_response_second_moment'] for r in values),
                        pooled_covariance_residual_fraction=sum(r['residual_covariance_trace'] for r in values)/sum(r['covariance_trace'] for r in values)))
    draws=np.array(list(itertools.product(range(4),repeat=4)))
    comparisons=[]
    for n in [4,14]:
        for t in [2048,16384]:
            baseline=next(r for r in summaries if r['heads']==n and r['step']==t and r['construction']=='eight_batch_reference' and r['requested_dimension']==8)
            for construction in spec['constructions']:
                for k in [8,16,32]:
                    expanded=next(r for r in summaries if r['heads']==n and r['step']==t and r['construction']==construction and r['requested_dimension']==k)
                    for field in ['total_response_residual_fractions','covariance_residual_fractions']:
                        change=np.array(expanded[field])-np.array(baseline[field])
                        comparisons.append(dict(heads=n,step=t,construction=construction,requested_dimension=k,field=field,
                            reference_dimension=8,per_initialization_change=change.tolist(),mean_change=float(change.mean()),
                            empirical_change_percentiles=np.quantile(change[draws].mean(1),[.025,.975]).tolist()))
    result=dict(schema='paired-source-basis-analysis-v1',status='complete',states=records,conditions=summaries,
        dimensions=dimensions,paired_calibration_changes=comparisons,locality=locality,
        reference_selection_sha256=sha256(reference_path),
        scope='Every16 selected incoming state is validated on the same32 new minibatches for both32-calibration basis orderings and the frozen8-calibration reference. All dimensions and response amplitudes remain. Paired empirical ranges condition on the fixed batch sample and have no population coverage guarantee. Neither finite response rank nor local source fidelity identifies autonomous training closure or criticality.')
    write_json(out/'results.json',result);np.savez_compressed(out/'measurements.npz',**raw)
    write_json(out/'manifest.json',dict(status='complete',binding_sha256=sha256(out/'binding.json'),
        results_sha256=sha256(out/'results.json'),raw_sha256=sha256(out/'measurements.npz'),states=len(records)))


if __name__=='__main__':main()
