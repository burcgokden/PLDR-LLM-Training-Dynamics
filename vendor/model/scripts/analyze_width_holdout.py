#!/usr/bin/env python
"""Score frozen width predictions against the new architecture."""
import argparse
import json
from pathlib import Path
import numpy as np
from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json
from analyze_criticality import susceptibility, jackknife_trace


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    study=Path(args.root)/'criticality-study-20260905';out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    prediction=study/'analysis/width-prediction/results.json'
    paths=sorted((study/'runs').glob('width-holdout-*/manifest.json'))
    if len(paths)!=4:raise RuntimeError('All four wider-model initializations are required')
    bind_run(out,[prediction,*[p for m in paths for p in [m,m.parent/'measurements.npz']]],vars(args))
    arrays=[];metas=[]
    for path in paths:
        meta=json.loads(path.read_text());metas.append(meta)
        if meta['status']!='complete':raise RuntimeError('A numerical failure must be retained and assessed')
        if sha256(path.parent/'measurements.npz')!=meta['raw_sha256']:raise AssertionError('Raw hash mismatch')
        arrays.append(np.load(path.parent/'measurements.npz'))
    heads=np.stack([a['heads_2048'] for a in arrays]).astype(float);fields={}
    for name,index in [('attention_entropy',0),('row_energy',2)]:
        values=heads[...,index];stat=susceptibility(values);stat['jackknife']=jackknife_trace(values)
        fields[name]=stat
    scores=[]
    for row in json.loads(prediction.read_text())['rows']:
        observed=fields[row['field']]['trace'];error=row['prediction']-observed
        scores.append(dict(**row,observed=observed,error=error,relative_error=error/observed if observed>0 else None))
    write_json(out/'results.json',dict(schema='width-holdout-analysis-v1',heads=24,width=1536,
        seeds=[m['arguments']['seed'] for m in metas],fields=fields,prediction_scores=scores,
        nll_by_seed=[float(a['fields_2048'][:,25].mean()) for a in arrays],
        context_kl_by_seed=[float(a['context_kl_2048'].mean()) for a in arrays],
        peak_cuda_gb=[m['peak_cuda_gb'] for m in metas],seconds=[m['seconds'] for m in metas],
        interpretation='One withheld larger size at a prespecified intermediate rate. Predictive superiority over four fitted widths does not establish a singular limit or critical exponent.'))
    write_json(out/'manifest.json',dict(results_sha256=sha256(out/'results.json'),binding_sha256=sha256(out/'binding.json')))


if __name__=='__main__':main()
