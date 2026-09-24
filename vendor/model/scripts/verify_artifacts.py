#!/usr/bin/env python
"""Verify the measurement boundary independently of manuscript rendering."""
import argparse
import json
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256, write_json


def require(condition,message):
    if not condition:raise AssertionError(message)


def bound_file(path,digest):
    require(path.is_file(),f'Missing artifact: {path}')
    require(sha256(path)==digest,f'Hash mismatch: {path}')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True)
    ap.add_argument('--data');ap.add_argument('--assets');ap.add_argument('--output',required=True);args=ap.parse_args()
    root=Path(args.data_root);data=Path(args.data) if args.data else root/'data/refinedweb-4608'
    assets=Path(args.assets) if args.assets else root/'assets'
    dm=json.loads((data/'manifest.json').read_text())
    bound_file(data/'tokens.npy',dm['tokens_sha256']);bound_file(data/'records.json',dm['records_sha256'])
    tokens=np.load(data/'tokens.npy');records=json.loads((data/'records.json').read_text())
    require(tokens.shape==(4608,513),'Unexpected token inventory')
    require(tokens.min()>=0 and tokens.max()<32000,'Out-of-vocabulary token')
    require(len({r['content_sha256'] for r in records})==4608,'Document overlap')
    tokenizers=[sha256(assets/f'PLDR-LLM-v51-SOC-110M-{i}'/'tokenizer.model') for i in [1,2]]
    require(tokenizers[0]==tokenizers[1]==dm['tokenizer_sha256'],'Tokenizer mismatch')
    checks=[]
    for i in [1,2]:
        model=assets/f'PLDR-LLM-v51-SOC-110M-{i}'
        model_digest=sha256(model/'model.safetensors')
        for kind,name,array,shape in [
            ('features',f'features-soc{i}-s128','features',(4608,174)),
            ('segments',f'segments-soc{i}-s64','features',(4608,8,174)),
            ('response',f'response-replication-soc{i}-s128','fisher',(256,70,70))]:
            run=root/'executed'/name;meta=json.loads((run/'manifest.json').read_text());raw=run/(kind+'.npz')
            bound_file(raw,meta[kind+'_sha256'])
            require(meta['model_sha256']==model_digest,'Checkpoint mismatch')
            require(meta['data_manifest_sha256']==sha256(data/'manifest.json'),'Data mismatch')
            with np.load(raw) as archive:
                x=archive[array]
                require(x.shape==shape,f'Unexpected dimensions: {name}')
                require(np.isfinite(x).all(),f'Nonfinite observations: {name}')
                if kind in ('features','segments'):
                    require(len(meta['feature_names'])==174,'Feature labels mismatch')
                if kind=='response':
                    require(np.max(abs(x-x.swapaxes(-1,-2)))<1e-10,'Fisher asymmetry')
                    require(np.linalg.eigvalsh(x).min()>-1e-10,'Fisher negativity')
                    require(archive['jacobian'].shape==(256,32000,70),'Incomplete source/vocabulary coverage')
                    require(archive['kl'].shape==(256,4,8),'Incomplete perturbation inventory')
                    require(np.isfinite(archive['kl']).all(),'Nonfinite KL')
                    require(np.min(archive['kl'])>-1e-10,'KL below numerical tolerance')
            native_hash=meta['source_files'].get('src/model_rg/native.py')
            require(native_hash==sha256(Path(__file__).resolve().parents[1]/'src/model_rg/native.py'),
                    'Native model adapter differs from producer')
            checks.append({'run':name,'sha256':meta[kind+'_sha256'],'shape':shape})
        run=root/'executed'/f'native-validation-soc{i}';meta=json.loads((run/'result.json').read_text())
        bound_file(run/'native-checks.npz',meta['raw_sha256'])
        require(meta['model_sha256']==model_digest,'Native validation checkpoint mismatch')
        require(len(meta['composition_checks'])==8,'Missing composition controls')
        require(max(c['primal_max_abs_error'] for c in meta['composition_checks'])==0,'Native loop mismatch')
        require(max(c['tangent_relative_fisher_error'] for c in meta['composition_checks'])<1e-5,'Source transport mismatch')
        checks.append({'run':run.name,'sha256':meta['raw_sha256']})
    result=root/'analysis/main/results.json'
    analysis=json.loads((result.parent/'analysis-manifest.json').read_text())
    bound_file(result,analysis['results_sha256'])
    summary=json.loads(result.read_text())
    for i in [1,2]:
        require(summary[f'soc{i}']['segments']['covariance_identity_relative_residual']<1e-12,
                'Blocked covariance does not retain measured cross terms')
    report={'schema':'model-rg-verification-v1','status':'passed','data_manifest_sha256':sha256(data/'manifest.json'),
            'unique_documents':len(records),'tokenizer_sha256':tokenizers[0],
            'runs':checks,'results_sha256':sha256(result)}
    write_json(args.output,report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
