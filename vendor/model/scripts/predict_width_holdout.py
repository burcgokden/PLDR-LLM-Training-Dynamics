#!/usr/bin/env python
"""Freeze four-width trained-field predictions before the larger-width training."""
import argparse
import json
from pathlib import Path
from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    study=Path(args.root)/'criticality-study-20260905';out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    path=study/'analysis/variance/results.json';meta=json.loads((path.parent/'manifest.json').read_text())
    if sha256(path)!=meta['results_sha256']:raise AssertionError('Analysis hash mismatch')
    bind_run(out,[path,path.parent/'manifest.json',study/'protocols/width-holdout.json'],vars(args))
    fits=json.loads(path.read_text())['width_fit_diagnostics'];rows=[]
    for name,fit in fits.items():
        if '-g1-' not in name:continue
        for model,values in fit['models'].items():
            a,b=values['parameters'];prediction=a+b/24 if model=='regular_inverse_width' else a*24**b
            rows.append(dict(field=name.rsplit('-g1-',1)[1],model=model,heads=24,prediction=prediction,
                             fitted_heads=fit['heads'],parameters=values['parameters']))
    if len(rows)!=4:raise AssertionError('Missing two-model predictions for both fixed fields')
    write_json(out/'results.json',dict(schema='width-prediction-v1',rows=rows,
        interpretation='Predictions fixed using only the four base widths at g=1. A power fit is a competing finite model, not a critical exponent estimate.'))
    write_json(out/'manifest.json',dict(results_sha256=sha256(out/'results.json'),binding_sha256=sha256(out/'binding.json')))


if __name__=='__main__':main()
