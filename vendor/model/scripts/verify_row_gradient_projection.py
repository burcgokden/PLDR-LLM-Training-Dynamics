#!/usr/bin/env python
"""Independently check shared-gradient arrays, projection choices, and predictive KL."""
from companion_paths import required_input
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from model_rg.provenance import sha256, write_json


def integrated_kl(base, other, order):
    # log-partition Taylor remainder: KL = integral_0^1 (1-s) Var_s(dz) ds.
    # The positive variance integral avoids subtracting nearby log probabilities.
    z=np.asarray(base,dtype=np.float64);delta=np.asarray(other,dtype=np.float64)-z
    nodes,weights=np.polynomial.legendre.leggauss(order);total=np.zeros(z.shape[0])
    for x,w in zip(nodes,weights,strict=True):
        s=(x+1)/2;logits=z+s*delta;p=np.exp(logits-logits.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
        centered=delta-(p*delta).sum(-1,keepdims=True)
        total+=w*(1-s)*(p*centered**2).sum(-1)/2
    return total


def verify_gradient_projection(study,bound):
    study=Path(study);repo=Path(__file__).resolve().parents[1];torch.set_num_threads(4)
    protocol=study/'protocols/row-gradient-projection.json';spec=json.loads(protocol.read_text());ledger=json.loads((study/'row-gradient-projection.json').read_text())
    bound(Path(required_input('dynamics-protocols')) / 'row-gradient-projection.json',sha256(protocol))
    bound(protocol,ledger['protocol_sha256'])
    if ledger['status']!='complete' or len(ledger['records'])!=48 or any(r['returncode'] for r in ledger['records']):raise AssertionError('Incomplete paired gradient design')
    if {r['name'] for r in ledger['records']}!={c['name'] for c in spec['cases']}:raise AssertionError('Gradient ledger case identities changed')
    identities={(c['heads'],c['seed'],c['step'],c['batch_step']) for c in spec['cases']}
    if len(spec['cases'])!=48 or identities!={(n,s,t,b) for n in [4,8,14] for s in range(640101,640105) for t in [2048,8192] for b in [2048,8192]}:raise AssertionError('Gradient source crossing changed')
    vectors=0;kl_checks=0;index_checks=0
    for case in spec['cases']:
        folder=study/'analysis'/case['name'];meta=json.loads((folder/'manifest.json').read_text());result=json.loads((folder/'results.json').read_text())
        if meta['status']!='complete' or result['case']!=case:raise AssertionError('Gradient result identity changed')
        bound(folder/'binding.json',meta['binding_sha256'])
        binding=json.loads((folder/'binding.json').read_text())
        for p,h in binding['inputs'].items():bound(p,h)
        for p,h in binding['source_files'].items():bound(folder/'source'/p,h)
        bound(folder/'results.json',meta['results_sha256']);bound(folder/'measurements.npz',meta['raw_sha256'])
        raw=np.load(folder/'measurements.npz');reference=study/'analysis'/case['adjoint_reference'];original=np.load(reference/'measurements.npz')
        if [r['variant'] for r in result['records']]!=spec['variants']:raise AssertionError('A gradient variant was omitted')
        for name in ['rows','offsets','crops']:np.testing.assert_array_equal(raw[name],original[name])
        np.testing.assert_array_equal(raw['base_logits'],original['native_logits'])
        np.testing.assert_array_equal(raw['terminal_identity_logits'],raw['base_logits'])
        original.close();layout=result['shared_parameter_layout'];offset=0
        for entry in layout:
            if 'reslayerAs' not in entry['name'] or entry['start']!=offset or entry['stop']-offset!=int(np.prod(entry['shape'])):raise AssertionError('Shared gradient coordinate layout changed')
            offset=entry['stop']
        if len({entry['name'] for entry in layout})!=len(layout):raise AssertionError('Repeated shared gradient coordinate')
        for record in result['records']:
            variant=record['variant']
            for label,key in [('unclipped','gradient'),('clipped','clipped_gradient')]:
                base=raw['base_'+key].astype(float);value=raw[variant+'_'+key].astype(float)
                if base.shape!=(offset,) or value.shape!=base.shape or not np.isfinite(value).all():raise AssertionError('Gradient vector size or finiteness changed')
                a=np.linalg.norm(base);b=np.linalg.norm(value);difference=np.linalg.norm(value-base);dot=float(np.sum(base*value));reported=record[label]
                np.testing.assert_allclose([a*a,b*b,difference*difference,dot],[reported[k] for k in ['reference_squared_norm','squared_norm','squared_difference','inner_product']],rtol=1e-9,atol=1e-30)
                if a>0:np.testing.assert_allclose(difference/a,reported['relative_error'],rtol=1e-9,atol=1e-16)
                elif reported['relative_error'] is not None:raise AssertionError('Zero force reference needs explicit handling')
                if a*b>0:np.testing.assert_allclose(dot/(a*b),reported['cosine'],rtol=1e-9,atol=1e-14)
                elif reported['cosine'] is not None:raise AssertionError('Zero-force cosine needs explicit handling')
                if variant=='terminal_identity':np.testing.assert_array_equal(value,base)
                vectors+=1
            factor=min(np.float32(1),np.float32(1)/np.float32(np.float32(record['total_gradient_norm'])+np.float32(1e-6)))
            np.testing.assert_array_equal(raw[variant+'_clipped_gradient'],raw[variant+'_gradient']*factor)
            if variant in ['base','terminal_identity']:
                kl=np.zeros(32);np.testing.assert_array_equal(raw[variant+'_kl'],0)
            else:
                kl=integrated_kl(raw['base_logits'],raw[variant+'_logits'],16)
                finer=integrated_kl(raw['base_logits'],raw[variant+'_logits'],32)
                np.testing.assert_allclose(kl,finer,rtol=1e-9,atol=1e-25)
                np.testing.assert_allclose(finer,raw[variant+'_kl'],rtol=1e-7,atol=1e-24)
            np.testing.assert_allclose([float(kl.mean()),float(kl.max())],[record['mean_kl'],record['maximum_kl']],rtol=1e-7,atol=1e-24)
            kl_checks+=len(kl)
            if variant!='base':
                count={'terminal_keep4':4,'terminal_mean':0,'terminal_identity':64}[variant]
                for layer in range(5):
                    rows=raw[variant+f'_L{layer}_terminal_rows'];indices=raw[variant+f'_L{layer}_indices']
                    # Reconstruct the declared native selection. NumPy and Torch
                    # float32 centroids need not give the same order near equal rows.
                    # Fixed-index geometry requires distinct indices, not an
                    # unmeasured real-arithmetic ordering of a different centroid.
                    if indices.shape!=rows.shape[:-2]+(count,) or np.any(np.diff(np.sort(indices,-1),axis=-1)==0):raise AssertionError('Exception index registry changed')
                    if count and (indices.min()<0 or indices.max()>=64):raise AssertionError('Out-of-range exception index')
                    tensor=torch.from_numpy(rows)
                    native_distance=(tensor-tensor.mean(-2,keepdim=True)).square().sum(-1)
                    native_indices=native_distance.argsort(dim=-1,descending=True,stable=True)[...,:count].numpy()
                    np.testing.assert_array_equal(indices,native_indices)
                    index_checks+=1
        raw.close();print('Verified shared-force case',case['name'],flush=True)
    return dict(cases=48,gradient_vectors=vectors,predictive_kl_values=kl_checks,exception_index_registries=index_checks,identity_controls=48)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    signature=sha256(__file__);cache={}
    def bound(path,h):
        path=Path(path).resolve()
        if path not in cache:cache[path]=sha256(path)
        if cache[path]!=h:raise AssertionError('Gradient binding changed: '+str(path))
    checks=verify_gradient_projection(Path(args.root)/'criticality-dynamics-20260906',bound)
    if signature!=sha256(__file__):raise AssertionError('Gradient verifier changed during execution')
    write_json(args.output,dict(status='passed',verifier_sha256=signature,hashed_files=len(cache),checks=checks))


if __name__=='__main__':main()
