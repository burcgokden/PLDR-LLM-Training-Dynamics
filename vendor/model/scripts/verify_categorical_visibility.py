#!/usr/bin/env python3
"""Independently reduce native endpoint pairs and categorical tail contrasts."""
import argparse
import json
from numerical_claims import finite_greater
from numerical_validation import load_json_strict, discrepancy, array_discrepancy, finite_array, finite_scalar
import math
from numbers import Real
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256,write_json


NUMERICAL_CLAIMS = ('native_variance', 'coarse_variance',
    'centered_remainder_variance', 'relative_centered_rms',
    'retained_variance_fraction', 'mean_emission_kl')


def finite_scalar(value, label):
    """Require a finite real scalar without coercing strings or booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError('Invalid numerical scalar: ' + label)
    try:
        valid = math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError('Nonfinite numerical scalar: ' + label)
    return value


def discrepancy_scalar(actual, claimed, label):
    finite_scalar(actual, 'reconstructed ' + label)
    finite_scalar(claimed, 'claimed ' + label)
    discrepancy = abs(actual - claimed)
    finite_scalar(discrepancy, 'discrepancy ' + label)
    if finite_greater(discrepancy, 3e-12, 'scripts/verify_categorical_visibility.py:50'):
        raise ValueError('Independent reduction differs: ' + label)
    return discrepancy


def pair_variance(x):
    if x.ndim != 3 or len(x) < 2 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError('Invalid finite replica observations')
    return sum(float(np.sum((x[i]-x[j])**2)) for i in range(len(x)) for j in range(i))/(len(x)*(len(x)-1)*x.shape[1])


def verify(study,output):
    if output.exists():raise FileExistsError(output)
    spec=load_json_strict((study/'protocol.json').read_text());result=load_json_strict((study/'analysis.json').read_text())
    if result['status']!='complete':raise ValueError('Incomplete visibility analysis')
    for row in result['cells']:
        for key in NUMERICAL_CLAIMS:
            finite_scalar(row[key], 'claimed ' + key)
    finite_scalar(spec['relative_centered_rms_target'], 'relative_centered_rms_target')
    checked={str(study/'analysis.json'):sha256(study/'analysis.json'),str(study/'protocol.json'):sha256(study/'protocol.json')}
    for path,digest in spec['development_checked_sha256'].items():
        if sha256(path)!=digest:raise ValueError('Development source changed')
        checked[path]=digest
    with np.load(study/'reference.npz',allow_pickle=False) as data:reference=data['reference'];order=data['order']
    if not np.isfinite(reference).all() or np.any(reference <= 0):
        raise ValueError('Invalid positive finite tail reference')
    if sha256(study/'reference.npz')!=spec['reference_sha256']:raise ValueError('Dictionary changed')
    parent=Path(spec['evaluation_study']);p=load_json_strict((parent/'protocol.json').read_text())
    if sha256(parent/'protocol.json')!=spec['evaluation_protocol_sha256']:raise ValueError('Evaluation protocol changed')
    indexed={(r['heads'],r['control'],r['retained_tokens']):r for r in result['cells']}
    if len(indexed) != len(result['cells']):raise ValueError('Duplicate native cell')
    seen=set();maximum=0.
    for n in p['design']['heads']:
        for g in p['design']['controls']:
            jobs=sorted([j for j in p['jobs'] if j['heads']==n and j['control']==g],key=lambda j:j['seed'])
            probabilities=[]
            for job in jobs:
                folder=parent/'runs'/job['run_id'];m=load_json_strict((folder/'manifest.json').read_text());path=folder/'observations.npz'
                if m['status']!='complete' or sha256(path)!=m['artifacts']['observations.npz']:raise ValueError('Native input changed')
                checked[str(path)]=m['artifacts']['observations.npz']
                with np.load(path,allow_pickle=False) as a:z=a[f'logits_{job["steps"]}'].astype(float)
                if not np.isfinite(z).all():raise ValueError('Nonfinite native logits')
                w=np.exp(z-z.max(-1,keepdims=True));probabilities.append(w/w.sum(-1,keepdims=True))
            probabilities=np.stack(probabilities);full=2*np.sqrt(probabilities)
            for k in spec['retained_tokens']:
                keep=order[:k];tail=order[k:];q=probabilities[:,:,tail].sum(-1)
                coarse=np.concatenate([2*np.sqrt(probabilities[:,:,keep]),2*np.sqrt(q)[...,None]],axis=-1)
                conditional=reference[:,tail]/reference[:,tail].sum(-1,keepdims=True)
                reconstructed=full.copy();reconstructed[:,:,tail]=2*np.sqrt(q[...,None]*conditional[None])
                native=pair_variance(full);reduced=pair_variance(coarse);error=pair_variance(full-reconstructed)
                for key,value in [('native_variance',native), ('coarse_variance',reduced), ('centered_remainder_variance',error)]:
                    finite_scalar(value, 'reconstructed ' + key)
                    if value < 0:raise ValueError('Negative reconstructed variance: ' + key)
                if native == 0:
                    raise ValueError('Undefined relative visibility: native variance is zero')
                row=indexed[n,g,k];seen.add((n,g,k))
                values={'native_variance':native,'coarse_variance':reduced,'centered_remainder_variance':error,
                        'relative_centered_rms':np.sqrt(error/native),'retained_variance_fraction':reduced/native}
                kl=np.mean(np.sum(probabilities[:,:,tail]*np.log(probabilities[:,:,tail]/(q[...,None]*conditional[None])),axis=-1))
                values['mean_emission_kl']=kl
                for key,value in values.items():
                    discrepancy=discrepancy_scalar(value,row[key],key)
                    maximum=max(maximum,discrepancy)
                if row['target_met']!=bool(values['relative_centered_rms']<=spec['relative_centered_rms_target']):raise ValueError('Threshold outcome changed')
    if seen!=set(indexed):raise ValueError('Incomplete native cell coverage')
    write_json(output,dict(status='passed',cells=len(seen),observed_paths=len(p['jobs']),maximum_absolute_error=maximum,
        scientific_updates=0,checked_sha256=checked,verifier_sha256=sha256(__file__),
        scope='Independent direct-pair variance and conditional-tail KL reduction of every declared endpoint cell.'))
    print('Independent categorical visibility verification passed',len(seen),'cells',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify(a.study,a.output)
