#!/usr/bin/env python
"""Independently reconstruct every selected proper-prefix row-stage observation."""
import argparse
import json
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-feasible-20260908');a=p.parse_args()
    root=Path(a.root).resolve();study=root/a.study;out=study/'verification/onepass-input-domain.json'
    if out.exists():raise FileExistsError(out)
    checked={};cache={}
    def check(path,want=None):
        path=Path(path).resolve();st=path.stat();key=(str(path),st.st_size,st.st_mtime_ns)
        if key not in cache:cache[key]=sha256(path)
        digest=cache[key]
        if want is not None and digest!=want:raise AssertionError('Changed input-domain evidence: '+str(path))
        checked[str(path)]=digest;return digest
    def load(path):
        check(path);return json.loads(Path(path).read_text())
    spec_path=study/'protocols/onepass-input-domain-selection.json';spec=load(spec_path)
    for name,digest in spec['inputs_sha256'].items():check(name,digest)
    if len(spec['cases'])!=4 or spec['lengths']!=[16,32,48,64] or spec['rows']!=list(range(512,520)):
        raise AssertionError('The complete input-domain selection changed')
    probe=root/'controlled-study-20260905/data/short';tokens=np.load(probe/'tokens.npy',mmap_mode='r');offsets=np.load(probe/'offsets.npy')
    rows=np.array(spec['rows']);crops=tokens[rows[:,None],offsets[rows,None]+np.arange(65)]
    records=[];summaries=[];elements=0
    for case in spec['cases']:
        folder=study/'measurements'/case['name'];meta=load(folder/'manifest.json')
        if meta['status']!='complete' or meta['case']!=case:raise AssertionError('A selected input-domain state is incomplete')
        for name,key in [('binding.json','binding_sha256'),('results.json','results_sha256'),('measurements.npz','raw_sha256')]:check(folder/name,meta[key])
        binding=load(folder/'binding.json')
        for name,digest in binding['inputs'].items():check(name,digest)
        for name,digest in binding['source_files'].items():check(folder/'source'/name,digest)
        for name,digest in spec['producer_sources'].items():check(folder/'source'/name,digest)
        reference=study/'measurements'/case['reference_prefix'];pr=load(case['prefix_verification'])
        if pr['status']!='passed':raise AssertionError('The original prefix observations require independent verification')
        for name in ['manifest.json','measurements.npz']:check(reference/name,pr['checked_sha256'][str(reference/name)])
        result=load(folder/'results.json')
        if result['case']!=case or result['native_prefix_replay_byte_equal'] is not True:raise AssertionError('Passive native endpoint replay is required')
        observed_elements=0;results=[]
        with np.load(folder/'measurements.npz') as raw,np.load(reference/'measurements.npz') as old:
            np.testing.assert_array_equal(raw['rows'],rows);np.testing.assert_array_equal(raw['crops'],crops)
            full=raw['L64_stages'].astype(float)
            expected_arrays={'rows','crops'}
            for length in spec['lengths']:
                x_native=raw[f'L{length}_stages']
                if x_native.dtype!=np.float32 or x_native.shape!=(8,5,9,14,64,64) or not np.isfinite(x_native).all():
                    raise AssertionError('The complete native row-stage axes changed')
                for field,actual in [('A',x_native[:,:,8]),('G',raw[f'L{length}_G']),('logits',raw[f'L{length}_logits'])]:
                    target=old[f'L{length}_prefix_'+field]
                    if actual.dtype!=target.dtype or actual.shape!=target.shape or actual.tobytes()!=target.tobytes():
                        raise AssertionError('A passive endpoint differs from its independently verified prefix')
                    observed_elements+=actual.size
                x=x_native.astype(float)
                e={
                    'row':np.mean((x-np.mean(x,axis=4,keepdims=True))**2,axis=(0,3,4,5)),
                    'context':np.mean((x-np.mean(x,axis=0,keepdims=True))**2,axis=(0,3,4,5)),
                    'prefix':np.mean((x-full)**2,axis=(0,3,4,5))}
                report=result['lengths'][spec['lengths'].index(length)]
                if report['length']!=length:raise AssertionError('Prefix-domain identity changed')
                for name,energy in e.items():
                    key='prefix_separation_energy' if name=='prefix' else name+'_energy'
                    np.testing.assert_allclose(report[key],energy,rtol=3e-11,atol=1e-25)
                    ratios=[]
                    for layer in range(5):
                        v=float(np.sqrt(energy[layer,8]/energy[layer,0])) if energy[layer,0]>0 else None;ratios.append(v)
                        got=report[name+'_transport_factor'][layer]
                        if v is None:
                            if got is not None:raise AssertionError('An undefined transport factor changed')
                        else:np.testing.assert_allclose(got,v,rtol=3e-11,atol=1e-25)
                results.append(report);expected_arrays.update(f'L{length}_'+name for name in ['stages','G','logits'])
            if set(raw.files)!=expected_arrays or len(result['lengths'])!=4:raise AssertionError('An input-domain observation was omitted or added')
        if observed_elements!=result['native_endpoint_elements']:raise AssertionError('Native endpoint element count changed')
        elements+=observed_elements;summaries.append(dict(name=case['name'],lengths=results))
        records.append(dict(name=case['name'],manifest_sha256=check(folder/'manifest.json')))
        print('Reconstructed every proper-prefix stage and native endpoint',case['name'],flush=True)
    ledger_path=study/'launcher-onepass-input-domain.json';ledger=load(ledger_path)
    if ledger['status']!='complete' or ledger['records']!=records or ledger['selection_sha256']!=check(spec_path):
        raise AssertionError('The input-domain execution is incomplete')
    write_json(out,dict(status='passed',states=4,prefix_domains=16,decoder_stage_paths=80,
        native_endpoint_elements_compared_bytewise=elements,summaries=summaries,checked_sha256=checked,
        verifier_sha256=sha256(__file__),scope='Independent finite-cloud reduction and bytewise replay correspondence for every proper-prefix input-domain observation. No uniform Lipschitz or criticality claim is certified.'))


if __name__=='__main__':main()
