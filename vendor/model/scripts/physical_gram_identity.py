#!/usr/bin/env python
"""Check the native first-query Gram histogram identity before normalization."""
from companion_paths import configured_path
import argparse
import gc
import json
from pathlib import Path
import sys
import numpy as np
import sentencepiece as spm
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from model_rg.training import TrainingModel
from model_rg.physical_native import spin_tokens,context_tokens,prefix_batch,selected_forward
from model_rg.provenance import sha256,write_json
ROOT=Path(configured_path('data:model'))


class GramCaptured(Exception):pass


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',required=True);ap.add_argument('--device',required=True);a=ap.parse_args()
    base=Path(a.study);out=base/'gram-identity';out.mkdir(exist_ok=False)
    spec=json.loads((base/'collective-validation/protocol.json').read_text())
    torch.set_num_threads(4);torch.cuda.set_device(a.device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=ROOT/'assets/PLDR-LLM-v51-SOC-110M-1'
    processor=spm.SentencePieceProcessor(model_file=str(source/'tokenizer.model'));alphabet=spin_tokens(processor)
    rows=[];arrays={}
    for case in spec['cases']:
        for origin in ['random','pretrained','adapted']:
            adapter=TrainingModel(source,case['heads'],case['seed'],a.device)
            checkpoint=None
            if origin!='random':
                checkpoint=case['state'] if origin=='pretrained' else spec['endpoints'][case['name']]['state']['path']
                expected=case['state_sha256'] if origin=='pretrained' else spec['endpoints'][case['name']]['state']['sha256']
                if sha256(checkpoint)!=expected:raise ValueError('Changed Gram state')
                state=torch.load(checkpoint,map_location='cpu',weights_only=False)
                adapter.model.load_state_dict(state['model']);del state;gc.collect()
            adapter.model.requires_grad_(False).eval();decoder=adapter.model.decoder;attention=decoder.dec_layers[0].mha1
            traces=[]
            def hook(module,args):
                traces.append(args[0].double().diagonal(dim1=-2,dim2=-1).sum(-1).cpu().numpy())
                raise GramCaptured()
            handle=attention.layernorm1.register_forward_pre_hook(hook)
            try:
                for c in spec['cells']:
                    if sha256(c['path'])!=c['sha256']:raise ValueError('Changed Gram configurations')
                    raw=np.memmap(c['path'],mode='r',dtype=np.uint8,shape=tuple(c['shape']))
                    x=np.asarray(raw[:,::8]).reshape(-1,c['L'],c['L']);flat=x.reshape(len(x),-1)
                    meta=context_tokens(processor,c['q'],c['L'],1.);tokens=meta+alphabet[:c['q']]
                    ids=torch.tensor(tokens,device=a.device)
                    embedded=decoder.embedding(ids)*torch.sqrt(torch.tensor(decoder.d_model).to(dtype=decoder.embedding.weight.dtype))
                    query=attention.wq(decoder.layernorm1(embedded)).reshape(len(tokens),case['heads'],64)
                    norms=query.double().square().sum(-1).cpu().numpy()
                    baseline=norms[:len(meta)].sum(0);coefficients=norms[len(meta):].T
                    tangent=coefficients[:,:-1]-coefficients[:,-1,None]
                    singular=np.linalg.svd(tangent,compute_uv=False)
                    if singular[-1]<1e-8*singular[0]:raise ValueError('Histogram direction is numerically invisible')
                    for site in [1,c['L']**2//2,c['L']**2-1]:
                        traces.clear()
                        for start in range(0,len(x),32):
                            inputs=prefix_batch(x[start:start+32],site,meta,alphabet[:c['q']],a.device)
                            try:selected_forward(adapter,inputs,alphabet[:c['q']])
                            except GramCaptured:pass
                        observed=np.concatenate(traces)
                        counts=np.stack([(flat[:,:site]==color).sum(1) for color in range(c['q'])],1)
                        expected_trace=baseline+counts@coefficients.T
                        relative=float(np.max(np.abs(observed-expected_trace)/np.maximum(np.abs(expected_trace),1e-30)))
                        recovered=np.linalg.lstsq(tangent,((observed-baseline)/site-coefficients[:,-1]).T,rcond=None)[0].T
                        fractions=np.column_stack([recovered,1-recovered.sum(1)])
                        error=float(np.max(np.abs(fractions-counts/site)))
                        if relative>2e-5 or error>1e-3:raise ValueError('Native histogram identity exceeds qualified arithmetic tolerance')
                        key=f'h{case["heads"]}-{origin}-q{c["q"]}-L{c["L"]}-j{site}'
                        arrays[key+'-traces']=observed;arrays[key+'-counts']=counts;arrays[key+'-coefficients']=coefficients;arrays[key+'-metadata']=baseline
                        rows.append(dict(heads=case['heads'],origin=origin,q=c['q'],L=c['L'],site=site,
                            tangent_rank=len(singular),smallest_singular_value=float(singular[-1]),
                            condition_number=float(singular[0]/singular[-1]),trace_relative_error=relative,
                            maximum_fraction_error=error,samples=len(x)))
            finally:handle.remove()
            print('qualified native Gram',case['heads'],origin,flush=True)
            del adapter,decoder,attention;gc.collect();torch.cuda.empty_cache()
    np.savez_compressed(out/'raw.npz',**arrays)
    sources={n:sha256(REPO/n) for n in ['scripts/physical_gram_identity.py','src/model_rg/physical_native.py','src/model_rg/training.py','src/model_rg/native.py','src/model_rg/provenance.py']}
    write_json(out/'verification.json',dict(status='passed',schema='physical-native-gram-v1',rows=rows,
        producer_sha256=sha256(__file__),sources=sources,raw_sha256=sha256(out/'raw.npz'),
        reference_protocol_sha256=sha256(base/'collective-validation/protocol.json'),native_updates=0,
        scope='Native first-layer Gram measurement stops at a read-only hook before Gram normalization. The identity is not asserted after LayerNorm or the metric generator. Finite floating-point tolerances are not an all-size arithmetic certificate.'))


if __name__=='__main__':main()
