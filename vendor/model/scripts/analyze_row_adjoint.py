#!/usr/bin/env python
"""Summarize full-loss adjoints without replacing their actual row weighting."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--include-source-control',action='store_true')
    args=parser.parse_args();study=Path(args.root)/'criticality-dynamics-20260906'
    inputs=[];cases=[];results=[]
    for name in ['row-adjoint']+(['row-adjoint-source-control'] if args.include_source_control else []):
        protocol=study/'protocols'/(name+'.json');ledger=study/(name+'.json')
        spec=json.loads(protocol.read_text());execution=json.loads(ledger.read_text())
        if execution['status']!='complete' or len(execution['records'])!=24 or any(r['returncode'] for r in execution['records']):raise AssertionError('Incomplete training-adjoint inventory')
        if sha256(protocol)!=execution['protocol_sha256']:raise AssertionError('Adjoint protocol changed')
        inputs.extend([protocol,ledger]);cases.extend(spec['cases'])
    for case in cases:
        folder=study/'analysis'/case['name'];meta=json.loads((folder/'manifest.json').read_text())
        if meta['status']!='complete':raise AssertionError('Incomplete adjoint case')
        for name,key in [('results.json','results_sha256'),('measurements.npz','raw_sha256')]:
            if sha256(folder/name)!=meta[key]:raise AssertionError('Changed adjoint input')
        inputs.extend([folder/'manifest.json',folder/'results.json',folder/'measurements.npz']);result=json.loads((folder/'results.json').read_text())
        if result['case']!=case:raise AssertionError('Adjoint source identity changed')
        results.append(result)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);bind_run(out,inputs,vars(args));rows=[];groups=defaultdict(list)
    for result in results:
        units=[]
        for unit in range(8):
            records=[r for r in result['records'] if r['unit'].endswith('_U'+str(unit))]
            if len(records)!=5:raise AssertionError('Incomplete decoder coverage')
            vin=sum(r['input_adjoint_energy'] for r in records);vout=sum(r['output_adjoint_energy'] for r in records)
            gradient=sum(r['parameter_gradient_norm']**2 for r in records)**.5
            units.append(dict(unit=unit,input_adjoint_energy=vin,output_adjoint_energy=vout,
                aggregate_adjoint_gain=(vin/vout)**.5 if vout>0 else None,
                parameter_gradient_norm=gradient,parameter_gradient_fraction=gradient/result['total_gradient_norm'],
                maximum_local_vjp_relative_error=max((r['maximum_local_vjp_relative_error'] or 0) for r in records),
                retained4_input_adjoint_energy_fraction=sum(r['input_adjoint_energy']*(r['retained4_input_adjoint_energy_fraction'] or 0) for r in records)/vin if vin>0 else None))
        block=(units[0]['input_adjoint_energy']/units[7]['output_adjoint_energy'])**.5 if units[7]['output_adjoint_energy']>0 else None
        if block is not None:
            np.testing.assert_allclose(np.prod([r['aggregate_adjoint_gain'] for r in units]),block,rtol=1e-12,atol=1e-15)
        row=dict(case=result['case'],training_loss=result['training_loss'],total_gradient_norm=result['total_gradient_norm'],clipping_factor=result['clipping_factor'],
            units=units,block_adjoint_gain=block,batch_step=result['case'].get('batch_step',result['case']['step']));rows.append(row)
        groups[result['case']['heads'],result['case']['step'],row['batch_step']].append(row)
    summaries=[]
    for (heads,step,batch_step),members in sorted(groups.items()):
        members.sort(key=lambda r:r['case']['seed'])
        if [r['case']['seed'] for r in members]!=list(range(640101,640105)):raise AssertionError('Adjoint seed identities changed')
        if any(r['block_adjoint_gain'] is None for r in members):raise AssertionError('A source-free adjoint requires explicit reporting')
        summaries.append(dict(heads=heads,step=step,batch_step=batch_step,seeds=[r['case']['seed'] for r in members],
            final_unit_gains=[r['units'][7]['aggregate_adjoint_gain'] for r in members],block_gains=[r['block_adjoint_gain'] for r in members],
            unit_gains=[[u['aggregate_adjoint_gain'] for u in r['units']] for r in members],
            unit_gradient_fractions=[[u['parameter_gradient_fraction'] for u in r['units']] for r in members],
            maximum_local_vjp_relative_error=max(u['maximum_local_vjp_relative_error'] for r in members for u in r['units'])))
    paired=[]
    if args.include_source_control:
        lookup={(r['case']['heads'],r['case']['seed'],r['case']['step'],r['batch_step']):r for r in rows}
        if len(lookup)!=48:raise AssertionError('The matched two-state by two-batch design is incomplete')
        for heads in [4,8,14]:
            for batch_step in [2048,8192]:
                members=[]
                for seed in range(640101,640105):
                    early=lookup[heads,seed,2048,batch_step];late=lookup[heads,seed,8192,batch_step]
                    members.append(dict(seed=seed,block_gain_ratio=late['block_adjoint_gain']/early['block_adjoint_gain'],
                        final_unit_gain_ratio=late['units'][7]['aggregate_adjoint_gain']/early['units'][7]['aggregate_adjoint_gain'],
                        unit_gain_ratios=[b['aggregate_adjoint_gain']/a['aggregate_adjoint_gain'] for a,b in zip(early['units'],late['units'],strict=True)]))
                paired.append(dict(heads=heads,batch_step=batch_step,by_seed=members))
    write_json(out/'results.json',dict(schema='metric-adjoint-analysis-v2',include_source_control=args.include_source_control,cases=rows,groups=summaries,paired_state_comparisons=paired,
        interpretation='Native CPU full-loss adjoints at four seeds per width and time. With source controls, the same two sampler minibatches are crossed with both checkpoint horizons; these measurements add no training seeds. Gains are source-weighted Frobenius-norm ratios, not operator norms or stationary training eigenvalues. Parameter-gradient fractions are relative to the complete model gradient before clipping. Local smooth64 comparisons hold the native output adjoint fixed and resolve relative errors only above input-adjoint energy1e-40.'))
    write_json(out/'manifest.json',dict(status='complete',results_sha256=sha256(out/'results.json'),binding_sha256=sha256(out/'binding.json')))


if __name__=='__main__':main()
