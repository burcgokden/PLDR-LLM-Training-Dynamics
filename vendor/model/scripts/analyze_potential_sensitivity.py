#!/usr/bin/env python3
"""Passive readout sensitivity to small positive bases on the retained entry panel.

Raised floors below are postprocessing of an unchanged native trajectory, not
retrained architectures or evidence about counterfactual training outcomes.
"""
from companion_paths import legacy_path
import json
from pathlib import Path
import numpy as np
from model_rg.provenance import write_json,sha256
ROOT=Path(legacy_path('/pldr-data/model/potential-avalanche-20260913'))

def main():
    records=[]
    paths=sorted((ROOT/'runs').glob('*/manifest.json'))+sorted((ROOT/'continuations').glob('*/manifest.json'))
    for mf in paths:
        m=json.loads(mf.read_text());z=np.load(mf.parent/'activity.npz')
        logbase=z['logbase'][:,0].astype(float);power=z['power'].astype(float)
        q=logbase*power;delta=np.diff(q,axis=0)
        e=delta**2;ee=e.reshape(len(e),-1);total=ee.sum(1)
        neff=total**2/np.sum(ee**2,axis=1).clip(1e-300)
        lo=0 if mf.parent.parent.name=='continuations' else 256
        adjacent=np.minimum(logbase[:-1],logbase[1:]);allenergy=e[lo:].sum()
        row=dict(run_id=mf.parent.name,kind=mf.parent.parent.name,manifest_sha256=sha256(mf),
            sample_coordinates=int(ee.shape[1]),activity_participation_fraction_median=float(np.median(neff[lo:]/ee.shape[1])),
            fraction_energy_at_small_base={str(c):float(e[lo:][adjacent[lo:]<np.log(c)].sum()/allenergy) for c in [1e-8,1e-7,1e-6,1e-5]},
            raised_floor_readout={})
        # Native base is iSwiGLU(z)+1e-9 in FP32. Subtracting that offset
        # reconstructs the positive activation approximately, with a floor only
        # for at-most-roundoff negative values after float32 log storage.
        activation=np.maximum(np.exp(logbase)-1e-9,0.)
        original=np.sqrt(ee.mean(1))
        for floor in [1e-8,1e-6]:
            qq=power*np.log(activation+floor)
            act=np.sqrt(np.mean(np.diff(qq,axis=0).reshape(len(original),-1)**2,axis=1))
            row['raised_floor_readout'][str(floor)]=dict(
                integrated_squared_activity_ratio=float(np.sum(act[lo:]**2)/np.sum(original[lo:]**2)),
                q99_activity_ratio=float(np.quantile(act[lo:],.99)/np.quantile(original[lo:],.99)))
        records.append(row)
    write_json(ROOT/'analysis/base-sensitivity.json',dict(status='complete_for_available_paths',producer_sha256=sha256(__file__),
        scope='Fixed128 entries per head, primary fixed probe. Effective participation and base-floor sensitivity are passive observation diagnostics, not independent-site statistics or altered-model training.',records=records))
    for r in records:print(r['run_id'],round(r['activity_participation_fraction_median'],4),r['fraction_energy_at_small_base']['1e-06'])
if __name__=='__main__':main()
