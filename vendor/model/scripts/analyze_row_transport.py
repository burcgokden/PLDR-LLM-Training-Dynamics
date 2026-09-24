#!/usr/bin/env python
"""Summarize local row transport while retaining aggregate nonlinear budgets."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();study=Path(args.root)/'criticality-dynamics-20260906';out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False);cases=[];inputs=[];radii=None
    for name in ['row-transport','row-transport-transfer']:
        protocol=study/'protocols'/(name+'.json');ledger=study/(name+'.json')
        spec=json.loads(protocol.read_text());record=json.loads(ledger.read_text())
        if record['status']!='complete' or any(r['returncode'] for r in record['records']):raise AssertionError('Incomplete row transport study')
        if len(record['records'])!=len(spec['cases']):raise AssertionError('Row transport inventory changed')
        if radii is not None and radii!=spec['radii']:raise AssertionError('Row transport amplitude grid changed')
        radii=spec['radii'];inputs.extend([protocol,ledger]);cases.extend(spec['cases'])
    for case in cases:
        folder=study/'analysis'/case['name'];meta=json.loads((folder/'manifest.json').read_text())
        if meta['status']!='complete':raise AssertionError('Incomplete transport case')
        for name,key in [('results.json','results_sha256'),('measurements.npz','raw_sha256')]:
            if sha256(folder/name)!=meta[key]:raise AssertionError('Changed row transport input')
        inputs.extend([folder/'manifest.json',folder/'results.json',folder/'measurements.npz'])
    bind_run(out,inputs,vars(args));groups=defaultdict(list);rows=[]
    for case in cases:
        raw=np.load(study/'analysis'/case['name']/'measurements.npz');units=[]
        for unit in range(8):
            def values(label):return np.concatenate([raw[f'L{layer}_U{unit}_{label}'].reshape(-1) for layer in range(5)])
            vin=values('variance_input');vout=values('variance_output');linear=values('variance_linear');cross=values('variance_cross');residual=values('variance_residual')
            norms=values('spectral_norm');gains=values('empirical_gain');directional=values('directional_gain')
            errors=np.concatenate([raw[f'L{layer}_U{unit}_amplitude_relative_errors'].reshape(len(raw['radii']),-1) for layer in range(5)],axis=1)
            resolved=values('response_resolved').astype(bool)
            record=dict(unit=unit,points=len(norms),spectral_norm_median=float(np.median(norms)),spectral_norm_max=float(norms.max()),
                fraction_centroid_norm_below_one=float(np.mean(norms<1)),empirical_gain_median=float(np.median(gains)),directional_gain_median=float(np.median(directional)),
                variance_input=float(vin.mean()),variance_output=float(vout.mean()),variance_linear=float(linear.mean()),variance_cross=float(cross.mean()),variance_residual=float(residual.mean()),
                aggregate_linear_ratio=float(linear.sum()/vout.sum()),aggregate_signed_cross_ratio=float(cross.sum()/vout.sum()),aggregate_residual_ratio=float(residual.sum()/vout.sum()),
                point_residual_ratio_median=float(np.median(residual/np.maximum(vout,1e-300))),
                amplitude_max_relative_errors=[float(e[resolved].max()) for e in errors],
                native_smooth_max_rms_difference=float(values('native_smooth_rms_difference').max()))
            np.testing.assert_allclose(record['variance_output'],record['variance_linear']+record['variance_cross']+record['variance_residual'],rtol=1e-9,atol=1e-15)
            units.append(record)
            groups[case['heads'],case['step'],unit].append((case,record,norms,gains))
        rows.append(dict(case=case,units=units,native_mean_row_energy=float(raw['native_head_fields'][...,2].mean())))
        raw.close()
    summaries=[]
    for (heads,step,unit),members in sorted(groups.items()):
        if len(members)!=4 or {m[0]['seed'] for m in members}!=set(range(640101,640105)):raise AssertionError('Four paired seed identities required')
        norms=np.concatenate([m[2] for m in members]);gains=np.concatenate([m[3] for m in members]);records=[m[1] for m in members]
        summaries.append(dict(heads=heads,step=step,unit=unit,seeds=[m[0]['seed'] for m in members],
            measured_points=len(norms),spectral_norm_median=float(np.median(norms)),spectral_norm_max=float(norms.max()),
            fraction_centroid_norm_below_one=float(np.mean(norms<1)),empirical_gain_median=float(np.median(gains)),
            aggregate_residual_ratio_by_seed=[r['aggregate_residual_ratio'] for r in records],
            aggregate_linear_ratio_by_seed=[r['aggregate_linear_ratio'] for r in records],
            aggregate_signed_cross_ratio_by_seed=[r['aggregate_signed_cross_ratio'] for r in records],
            amplitude_max_relative_errors=np.max([r['amplitude_max_relative_errors'] for r in records],axis=0).tolist(),
            native_smooth_max_rms_difference=max(r['native_smooth_max_rms_difference'] for r in records)))
    write_json(out/'results.json',dict(schema='row-transport-analysis-v1',cases=rows,groups=summaries,radii=radii,
        interpretation='Four contexts and four initialization identities per width-time condition. Centroid derivatives are local smooth-map measurements. The ratio of total residual energy to total output row variance is reported separately from the median local ratio: rare excursions can dominate aggregate fluctuations. Cases, contexts, heads and internal units do not supply additional independently trained seeds.'))
    write_json(out/'manifest.json',dict(status='complete',binding_sha256=sha256(out/'binding.json'),results_sha256=sha256(out/'results.json')))
    print('Analyzed',len(rows),'row transport checkpoints and',len(summaries),'unit-condition groups')


if __name__=='__main__':main()
