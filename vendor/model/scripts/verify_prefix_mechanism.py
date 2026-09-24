#!/usr/bin/env python
"""Independently reconstruct the frozen prefix probes and intervention checks."""
import argparse
import json
from pathlib import Path

import numpy as np

from model_rg.provenance import sha256, write_json


def same(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def log_prob(z):
    z=z.astype(np.float64); z=z-z.max(-1,keepdims=True)
    return z-np.log(np.exp(z).sum(-1,keepdims=True))


def close(a,b):
    np.testing.assert_allclose(a,b,rtol=3e-10,atol=3e-14)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-20260908');a=p.parse_args()
    root=Path(a.root).resolve();study=root/a.study
    output=study/'verification/prefix-mechanism-initial.json'
    if output.exists():raise FileExistsError(output)
    checked={};cache={}
    def check(path,want=None):
        path=Path(path).resolve();stat=path.stat();key=(str(path),stat.st_size,stat.st_mtime_ns)
        if key not in cache:cache[key]=sha256(path)
        got=cache[key]
        if want is not None and got!=want:raise AssertionError('Changed prefix evidence: '+str(path))
        checked[str(path)]=got;return got
    def artifact(folder):
        m=json.loads((folder/'manifest.json').read_text());check(folder/'manifest.json')
        if m['status']!='complete':raise AssertionError('Incomplete prefix artifact')
        for file,key in [('binding.json','binding_sha256'),('measurements.npz','raw_sha256'),('results.json','results_sha256')]:
            check(folder/file,m[key])
        b=json.loads((folder/'binding.json').read_text())
        for file,digest in b['inputs'].items():check(file,digest)
        for file,digest in b['source_files'].items():check(folder/'source'/file,digest)
        with np.load(folder/'measurements.npz') as z:q={k:z[k] for k in z.files}
        return m,json.loads((folder/'results.json').read_text()),q
    selection=study/'protocols/prefix-mechanism-initial-selection.json';check(selection)
    spec=json.loads(selection.read_text());rows=np.array(spec['rows']);probe=root/'controlled-study-20260905/data/short'
    tokens=np.load(probe/'tokens.npy',mmap_mode='r');offsets=np.load(probe/'offsets.npy')
    crops=tokens[rows[:,None],offsets[rows,None]+np.arange(65)];records=[];conditions=0;elements=0
    for case in spec['cases']:
        for file,digest in case['input_sha256'].items():check(file,digest)
        m,r,q=artifact(study/'measurements'/case['name'])
        if m['case']!=case or r['case']!=case:raise AssertionError('A prefix condition changed')
        np.testing.assert_array_equal(q['rows'],rows);np.testing.assert_array_equal(q['crops'],crops)
        donor=np.roll(np.arange(len(rows)),1);np.testing.assert_array_equal(q['donor_index'],donor)
        expected=[]
        for length in spec['prefix_lengths']:
            base=f'L{length}_full';lp0=log_prob(q[base+'_logits'])
            nll0=-lp0[np.arange(len(rows)),crops[:,length]]
            variants={'prefix':crops[:,:length].copy(),'noop':crops[:,:64].copy()}
            if length<64:
                target=crops[:,:64].copy();target[:,length]=crops[donor,length]
                suffix=crops[:,:64].copy();suffix[:,length:]=crops[donor,length:64]
                variants.update(target_swap=target,suffix_swap=suffix)
            for name,ids in variants.items():
                label=f'L{length}_{name}';np.testing.assert_array_equal(q[label+'_inputs'],ids)
                for field in ['logits','A','G']:
                    if q[label+'_'+field].dtype!=np.float32 or not np.isfinite(q[label+'_'+field]).all():
                        raise AssertionError('Invalid native prefix output')
                    if name=='noop' and not same(q[label+'_'+field],q[base+'_'+field]):raise AssertionError('No-op replay differs')
                lp=log_prob(q[label+'_logits']);nll=-lp[np.arange(len(rows)),crops[:,length]]
                kl=np.sum(np.exp(lp0)*(lp0-lp),axis=-1)
                aa=q[label+'_A'].astype(float);g=q[label+'_G'].astype(float);g0=q[base+'_G'].astype(float)
                row=np.mean((aa-aa.mean(-2,keepdims=True))**2,axis=(-2,-1))/np.maximum(np.mean(aa*aa,axis=(-2,-1)),1e-30)
                delta=np.sqrt(np.mean((g-g0)**2,axis=(-2,-1))/np.maximum(np.mean((g*g+g0*g0)/2,axis=(-2,-1)),1e-30))
                for field,value in [('kl',kl),('nll',nll),('row_ratio',row),('relative_G_difference',delta)]:close(q[label+'_'+field],value)
                row_result=dict(length=length,variant=name,full_window_nll=float(nll0.mean()),nll=float(nll.mean()),
                    nll_change=float((nll-nll0).mean()),mean_predictive_kl=float(kl.mean()),
                    maximum_logit_difference=float(np.max(np.abs(q[label+'_logits'].astype(float)-q[base+'_logits']))),
                    mean_row_ratio=float(row.mean()),mean_relative_G_difference=float(delta.mean()))
                expected.append(row_result);conditions+=1;elements+=q[label+'_logits'].size
        if len(expected)!=len(r['conditions']):raise AssertionError('A prefix comparison is missing')
        for left,right in zip(expected,r['conditions'],strict=True):
            if set(left)!=set(right):raise AssertionError('Prefix statistics schema changed')
            for key,want in left.items():
                if isinstance(want,str):
                    if right[key]!=want:raise AssertionError('Prefix variant changed')
                else:close(right[key],want)
        records.append(dict(name=case['name'],manifest_sha256=check(study/'measurements'/case['name']/'manifest.json')))
    ledger=study/'launcher-prefix-mechanism-initial.json';check(ledger);ld=json.loads(ledger.read_text())
    if ld['status']!='complete' or ld['records']!=records or ld['protocol_sha256']!=check(selection):
        raise AssertionError('Prefix selection or completion ledger differs')
    qm,qr,q=artifact(study/'qualification/inference-interventions')
    np.testing.assert_array_equal(q['inputs'],crops[:2,:64])
    altered=crops[:2,:64].copy();altered[:,32:]=crops[:2,:64][::-1,32:]
    np.testing.assert_array_equal(q['altered'],altered)
    if [r['name'] for r in qr['records']]!=[c['name'] for c in spec['cases']]:raise AssertionError('Missing intervention qualification')
    for r in qr['records']:
        key=r['name'];base=q[key+'_native_logits'];replay=q[key+'_replay_logits'];suffix=q[key+'_suffix_logits']
        values=dict(native_external_G_replay=same(base,replay),fixed_G_suffix_prefix=same(replay[:,:32],suffix[:,:32]),
            restored_after_fixed_G=same(base,q[key+'_restored_logits']),restored_after_projection=same(base,q[key+'_restored2_logits']))
        if values!=r['checks'] or not all(values.values()) or r['checked_logit_elements']!=base.size:
            raise AssertionError('Incorrect intervention byte-comparison claim')
        aa=q[key+'_projected_A']
        if not same(aa,np.broadcast_to(aa[:,:,:,:1,:],aa.shape)) or not r['projected_rows_identical']:
            raise AssertionError('Projected rows differ')
    write_json(output,dict(status='passed',states=len(records),conditions=conditions,retained_logit_elements=elements,
        qualified_states=len(qr['records']),checked_sha256=checked,
        scope='Independent NumPy reconstruction of all retained prefix logits, target losses, KL, row and operator statistics; exact input interventions; independent reconstruction of native replay and restored/fixed-operator byte comparisons. Model forwards are not independently rerun here.'))
    print('passed',len(records),'states',conditions,'conditions',flush=True)


if __name__=='__main__':main()
