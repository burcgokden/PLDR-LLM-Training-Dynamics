#!/usr/bin/env python
"""Independently reconstruct native row/context transport and exact endpoints."""
import argparse
import json
from pathlib import Path

import numpy as np

from model_rg.provenance import sha256,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-20260908');a=p.parse_args()
    root=Path(a.root).resolve();study=root/a.study
    output=study/'verification/row-input-transport.json'
    if output.exists():raise FileExistsError(output)
    checked={};cache={}
    def check(path,want=None):
        path=Path(path).resolve();stat=path.stat();key=(str(path),stat.st_size,stat.st_mtime_ns)
        if key not in cache:cache[key]=sha256(path)
        got=cache[key]
        if want is not None and want!=got:raise AssertionError('Changed row/input evidence: '+str(path))
        checked[str(path)]=got;return got
    def close(x,y):np.testing.assert_allclose(x,y,rtol=3e-11,atol=1e-25)
    selection=study/'protocols/row-input-transport-selection.json';check(selection);spec=json.loads(selection.read_text())
    for path,digest in spec['inputs_sha256'].items():check(path,digest)
    probe=root/'controlled-study-20260905/data/short';rows=np.array(spec['rows'])
    pt=np.load(probe/'tokens.npy',mmap_mode='r');po=np.load(probe/'offsets.npy')
    crops=pt[rows[:,None],po[rows,None]+np.arange(64)];suffix=crops.copy();suffix[:,32:]=crops[np.roll(np.arange(len(rows)),1),32:]
    records=[];summaries=[];elements=0
    for case in spec['cases']:
        folder=study/'measurements'/case['name'];meta=json.loads((folder/'manifest.json').read_text());check(folder/'manifest.json')
        if meta['status']!='complete' or meta['case']!=case:raise AssertionError('Incorrect row/input state')
        for filename,key in [('binding.json','binding_sha256'),('results.json','results_sha256'),('measurements.npz','raw_sha256')]:check(folder/filename,meta[key])
        binding=json.loads((folder/'binding.json').read_text())
        for path,digest in binding['inputs'].items():check(path,digest)
        for path,digest in binding['source_files'].items():check(folder/'source'/path,digest)
        for path,digest in spec['producer_sources'].items():check(folder/'source'/path,digest)
        for path,digest in case['input_sha256'].items():check(path,digest)
        result=json.loads((folder/'results.json').read_text())
        if result['case']!=case or not result['native_prefix_replay_byte_equal']:raise AssertionError('Incorrect passive capture qualification')
        with np.load(folder/'measurements.npz') as z:q={k:z[k] for k in z.files}
        for key,value in [('rows',rows),('full_inputs',crops),('suffix_inputs',suffix)]:np.testing.assert_array_equal(q[key],value)
        reference=study/'measurements'/case['reference_prefix'];rm=json.loads((reference/'manifest.json').read_text())
        check(reference/'manifest.json');check(reference/'measurements.npz',rm['raw_sha256'])
        with np.load(reference/'measurements.npz') as z:
            for label,ref in [('full','L32_full'),('suffix','L32_suffix_swap')]:
                if q[label+'_stages'].dtype!=np.float32 or q[label+'_stages'].shape[:3]!=(8,5,9):raise AssertionError('Native stage layout changed')
                if not np.isfinite(q[label+'_stages']).all():raise AssertionError('Nonfinite native row path')
                for field,value in [('A',q[label+'_stages'][:,:,8]),('G',q[label+'_G']),('logits',q[label+'_logits'])]:
                    target=z[ref+'_'+field]
                    if value.dtype!=target.dtype or value.shape!=target.shape or value.tobytes()!=target.tobytes():raise AssertionError('Native stage endpoint differs bytewise')
                    elements+=value.size
        x=q['full_stages'].astype(float);y=q['suffix_stages'].astype(float)
        energies=dict(row=np.mean((x-x.mean(-2,keepdims=True))**2,axis=(0,3,4,5)),
            context=np.mean((x-x.mean(0,keepdims=True))**2,axis=(0,3,4,5)),
            suffix=np.mean((x-y)**2,axis=(0,3,4,5)))
        if result['stages']!=['normalized_input_Gram',*[f'residual_unit_{i}' for i in range(1,9)]]:raise AssertionError('Native stage names changed')
        summary=dict(name=case['name'])
        for name,energy in energies.items():
            close(q[name+'_energy'],energy);close(result[name+'_energy'],energy)
            defined=energy[:,:-1]>0;np.testing.assert_array_equal(q[name+'_factor_defined'],defined)
            factors=np.sqrt(np.divide(energy[:,1:],energy[:,:-1],out=np.zeros_like(energy[:,1:]),where=defined))
            close(q[name+'_secant_factors'],factors)
            overall=[float(np.sqrt(v[-1]/v[0])) if v[0]>0 else None for v in energy]
            got=result['overall_'+name+'_factor']
            if len(got)!=5:raise AssertionError('A decoder transport summary is missing')
            for left,right in zip(got,overall,strict=True):
                if right is None:
                    if left is not None:raise AssertionError('Undefined stage factor changed')
                else:close(left,right)
            summary[name+'_overall_factors']=overall
        records.append(dict(name=case['name'],manifest_sha256=check(folder/'manifest.json')));summaries.append(summary)
    ledger=study/'launcher-row-input-transport.json';check(ledger);ld=json.loads(ledger.read_text())
    if ld['status']!='complete' or ld['records']!=records or ld['selection_sha256']!=check(selection):raise AssertionError('Incomplete row/input selection')
    write_json(output,dict(status='passed',states=len(records),decoder_stage_paths=len(records)*5,
        native_endpoint_elements_compared_bytewise=elements,summaries=summaries,checked_sha256=checked,
        scope='Independent reconstruction of every retained stage row/context/suffix energy and contraction factor, explicit undefined-factor masks, and bytewise endpoint correspondence to the independently completed prefix observations. These finite secant ratios are not uniform Lipschitz bounds.'))
    print('passed',len(records),'states',len(records)*5,'decoder stage paths',flush=True)


if __name__=='__main__':main()
