#!/usr/bin/env python
"""Check raw training, update-response, derivative, and mixture-analysis artifacts."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from model_rg.provenance import sha256, write_json
from model_rg.training import mixture_covariance


def check(condition,message):
    if not condition:raise AssertionError(message)


def bound(path,digest):
    check(path.is_file(),f'Missing {path}')
    check(sha256(path)==digest,f'Hash mismatch: {path}')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    root=Path(args.data_root);dm=root/'data/refinedweb-4608/manifest.json'
    data_meta=json.loads(dm.read_text());bound(dm.parent/'tokens.npy',data_meta['tokens_sha256'])
    tokens=np.load(dm.parent/'tokens.npy');summary=[];covariance_error=0.
    analysis=root/'analysis/training';am=json.loads((analysis/'analysis-manifest.json').read_text())
    bound(analysis/'results.json',am['results_sha256'])
    for name,digest in am['derived_arrays'].items():bound(analysis/name,digest)
    for h in [2,4,8,14]:
        trajectories=[]
        for seed in [12001,12002,12003]:
            run=root/'executed'/f'training-h{h}-s{seed}'
            m=json.loads((run/'manifest.json').read_text())
            bound(run/'training.npz',m['raw_sha256']);bound(run/'sampling.npz',m['sampling_sha256'])
            for name,digest in m['training_state_sha256'].items():bound(run/name,digest)
            check(m['data_manifest_sha256']==sha256(dm),'Training data boundary changed')
            bound(Path(m['arguments']['source'])/'modeling_pldrllm.py',m['native_source_sha256'])
            bound(Path(m['arguments']['source'])/'configuration_pldrllm.py',m['native_config_source_sha256'])
            config=json.loads((run/'config.json').read_text())
            check(config['num_attention_heads']==h and config['hidden_size']==64*h,'Width mismatch')
            check(config['num_hidden_layers']==5 and config['num_reslayerA']==8 and config['num_denseA']==2,'Architecture changed')
            check(m['parameters']==m['trainable_parameters'],'A parameter subset was frozen')
            raw=np.load(run/'training.npz');sampling=np.load(run/'sampling.npz')
            check(np.array_equal(raw['steps'],[0,32,128,256]),'Training horizon mismatch')
            check(raw['features'].shape==(4,512,10*h+34),'Incomplete all-head observations')
            check(len(m['feature_names'])==10*h+34,'Feature labels mismatch')
            for name in raw.files:check(np.isfinite(raw[name]).all(),f'Nonfinite training artifact: {name}')
            check(sampling['rows'].shape==(260,32),'Missing training/replay batch indices')
            check(sampling['rows'].min()>=0 and sampling['rows'].max()<3072,'Training/probe overlap')
            check(sampling['offsets'].min()>=0 and sampling['offsets'].max()<=448,'Target exceeds stored crop')
            check(np.array_equal(sampling['probe_rows'],np.arange(3584,4096)),'Probe identity mismatch')
            check(m['temporal_blocking']['parameter_max_abs_error']==0,'Temporal parameter mismatch')
            check(m['temporal_blocking']['optimizer_max_abs_error']==0,'Temporal optimizer mismatch')
            trajectories.append(raw['features'])
            rr=root/'executed'/f'training-response-h{h}-s{seed}'
            rm=json.loads((rr/'manifest.json').read_text());bound(rr/'response.npz',rm['raw_sha256'])
            check(rm['training_manifest_sha256']==sha256(run/'manifest.json'),'Response state mismatch')
            r=np.load(rr/'response.npz')
            check(r['baseline_logits'].shape==(16,32000),'Vocabulary coverage mismatch')
            check(np.array_equal(r['probe_rows'],np.arange(4096,4112)),'Response probe boundary mismatch')
            z=r['baseline_logits'];lp=z-logsumexp(z,axis=-1,keepdims=True);prob=np.exp(lp)
            dz=r['logit_derivative'];centered=dz-(prob*dz).sum(-1,keepdims=True)
            expected=(prob*centered**2).sum(-1)
            check(np.allclose(expected,r['curvature'],rtol=1e-10,atol=1e-12),'Predictive curvature mismatch')
            for i,zz in enumerate(r['perturbed_logits']):
                lq=zz-logsumexp(zz,axis=-1,keepdims=True)
                actual=(prob*(lp-lq)).sum(-1)
                check(np.allclose(actual,r['kl'][i],rtol=1e-8,atol=1e-12),'Predictive KL mismatch')
            dr=root/'executed'/f'training-derivative-h{h}-s{seed}'
            d=json.loads((dr/'manifest.json').read_text());bound(dr/'duality.npz',d['raw_sha256'])
            check(d['response_manifest_sha256']==sha256(rr/'manifest.json'),'Derivative response boundary mismatch')
            check(d['max_normalized_duality_error']<1e-5,'Forward/reverse derivative disagreement')
            check(d['baseline_max_abs_error']<2e-5,'Primal precision boundary exceeded')
            for folder in [run,rr,dr]:
                check(am['input_manifests'][str(folder.relative_to(root))]==sha256(folder/'manifest.json'),'Analysis input changed')
            summary.append({'width':64*h,'seed':seed,'training':m['raw_sha256'],'response':rm['raw_sha256'],'derivative':d['raw_sha256']})
        x=np.stack(trajectories).astype(np.float64);saved=np.load(analysis/f'covariances-h{h}.npz')
        for k in range(4):
            a,b,c=mixture_covariance(x[:,k]);error=float(np.linalg.norm(a+b-c)/np.linalg.norm(c))
            covariance_error=max(covariance_error,error)
            check(error<1e-12,'Training/context covariance split failed')
            for key,value in [('within',a),('between',b),('total',c)]:
                check(np.allclose(saved[key][k],value,rtol=1e-12,atol=1e-12),'Saved covariance mismatch')
    report={'schema':'model-rg-training-verification-v1','status':'passed','training_runs':12,
            'native_response_runs':12,'independent_derivative_runs':12,
            'max_covariance_relative_residual':covariance_error,'results_sha256':am['results_sha256'],
            'data_manifest_sha256':sha256(dm),'runs':summary}
    write_json(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k!='runs'},indent=2))


if __name__=='__main__':main()
