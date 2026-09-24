#!/usr/bin/env python
"""Bounded magnetic transport on the completed fresh fine/coarse pairs."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from physical_collective_validation import simplex_projection
from analyze_physical_mechanisms import regression,apply
from model_rg.provenance import sha256,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',required=True);a=ap.parse_args()
    base=Path(a.study);out=base/'transport-analysis';out.mkdir(exist_ok=False)
    spec=json.loads((base/'collective-validation/protocol.json').read_text());rows=[];raw={};checked={}
    for case in spec['cases']:
        folder=base/'collective-validation'/case['name']
        proof=json.loads((folder/'verification.json').read_text())
        if proof['status']!='passed':raise ValueError('Fresh collective observations incomplete')
        for cell in spec['cells']:
            path=folder/f'q{cell["q"]}-L{cell["L"]}.npz'
            if sha256(path)!=proof['files'][path.name]:raise ValueError('Fresh observation changed')
            checked[str(path)]=sha256(path);z=np.load(path);cal=z['chain']<8;test=~cal
            f0=z['fractions-0'];f1=z['fractions-1'];fit=regression(f0,f1,cal)
            lip=float(np.linalg.svd(fit[0],compute_uv=False)[0])
            predicted=simplex_projection(apply(fit,f0));normalization=np.sqrt(cell['q']/(cell['q']-1))
            norm=lambda x:float(np.sqrt(np.mean(np.sum((normalization*x[test])**2,axis=1))))
            closure=norm(predicted-f1)
            for family in ['hidden','A_common','G_common']:
                h0=z[f'{family}-predicted-fractions-0'];h1=z[f'{family}-predicted-fractions-1']
                transported=simplex_projection(apply(fit,h0));e0=norm(h0-f0);e1=norm(h1-f1)
                prediction=norm(transported-f1);route=norm(transported-h1)
                prediction_bound=lip*e0+closure;route_bound=prediction_bound+e1
                if prediction>prediction_bound+1e-12 or route>route_bound+1e-12:raise ValueError('Magnetic transport bound failed')
                key=f'{case["name"]}-q{cell["q"]}-L{cell["L"]}-{family}'
                raw[key+'-transported']=transported;raw[key+'-weights']=fit[0];raw[key+'-intercept']=fit[1]
                rows.append(dict(case=case['name'],q=cell['q'],L=cell['L'],family=family,lipschitz_bound=lip,
                    physical_closure_rms=closure,fine_readout_rms=e0,coarse_readout_rms=e1,
                    transported_prediction_rms=prediction,prediction_bound=prediction_bound,
                    route_discrepancy_rms=route,route_bound=route_bound))
    np.savez_compressed(out/'transport.npz',**raw)
    write_json(out/'analysis.json',dict(status='passed',schema='physical-bounded-transport-v1',
        analyzer_sha256=sha256(__file__),checked_sha256=checked,rows=rows,
        raw_sha256=sha256(out/'transport.npz'),
        method='An affine physical fine-to-coarse color-fraction map is fitted on the first eight fresh chains and projected onto the simplex. Its spectral matrix norm bounds its Lipschitz constant. All reported RMS errors use the other eight chains and the normalized zero-sum color vector.',
        scope='Descriptive finite transport on the existing fresh validation cohort. Its exact coupled empirical error bounds do not bound population TV or identify a fixed-point RG eigenvalue.'))
    print('passed',len(rows),'finite magnetic transport bounds',flush=True)


if __name__=='__main__':main()
