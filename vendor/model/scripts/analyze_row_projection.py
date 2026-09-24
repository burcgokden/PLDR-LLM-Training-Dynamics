#!/usr/bin/env python
"""Measure predictive and conditional-fluctuation fidelity of native row reductions."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from model_rg.controlled import bind_run
from model_rg.criticality import conditional_covariances
from model_rg.provenance import sha256, write_json


def trace(values):
    return float(np.trace(conditional_covariances(values)[0])/values.shape[-2])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();study=Path(args.root)/'criticality-dynamics-20260906';out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False);protocol=study/'protocols/row-projection.json';ledger=study/'row-projection.json'
    spec=json.loads(protocol.read_text());state=json.loads(ledger.read_text())
    if state['status']!='complete' or any(r['returncode'] for r in state['records']):raise AssertionError('Incomplete full-graph row reduction study')
    if len(state['records'])!=len(spec['cases']):raise AssertionError('Projection inventory changed')
    inputs=[protocol,ledger];groups=defaultdict(list);released=[]
    for case in spec['cases']:
        folder=study/'analysis'/case['output_name'];meta=json.loads((folder/'manifest.json').read_text())
        if meta['status']!='complete':raise AssertionError('Incomplete projection case')
        for name,key in [('results.json','results_sha256'),('measurements.npz','raw_sha256')]:
            if sha256(folder/name)!=meta[key]:raise AssertionError('Changed row reduction input')
        inputs.extend([folder/'manifest.json',folder/'results.json',folder/'measurements.npz'])
        if case.get('model_dir'):released.append(case)
        else:groups[case['heads'],case['step']].append(case)
    bind_run(out,inputs,vars(args));rows=[];individual=[];implementation=[]
    for case in spec['cases']:
        with np.load(study/'analysis'/case['output_name']/'measurements.npz') as data:
            broadcast=data['initial_mean_broadcast_logits'];compressed=data['initial_mean_compressed_logits']
            implementation.append(dict(case=case['output_name'],logits_identical=bool(np.array_equal(broadcast,compressed)),
                maximum_logit_difference=float(np.max(np.abs(broadcast-compressed))),maximum_kl=float(data['compressed_broadcast_kl'].max())))
    for (heads,step),cases in sorted(groups.items()):
        cases.sort(key=lambda c:c['seed']);seeds=[c['seed'] for c in cases]
        if seeds!=list(range(640101,640105)):raise AssertionError('Projection initialization identities changed')
        data=[np.load(study/'analysis'/c['output_name']/'measurements.npz') for c in cases];base=np.stack([d['base_heads'] for d in data]).astype(float)
        base_nll=np.stack([d['base_fields'][:,25] for d in data]).astype(float)
        for variant in spec['variants']:
            field=np.stack([d[variant+'_heads'] for d in data]).astype(float);kls=np.stack([d[variant+'_kl'] for d in data])
            entry=dict(heads=heads,step=step,seeds=seeds,variant=variant,cases=[c['output_name'] for c in cases],
                mean_kl=float(kls.mean()),maximum_kl=float(kls.max()),mean_kl_by_seed=kls.mean(1).tolist(),
                nll_difference_by_seed=(np.stack([d[variant+'_fields'][:,25] for d in data]).astype(float)-base_nll).mean(1).tolist())
            for name,index in [('entropy',0),('row_energy',2)]:
                values=field[...,index];reference=base[...,index];chi=trace(values);base_chi=trace(reference)
                error=values.mean(-1)-reference.mean(-1);epsilon2=float(np.mean(error**2));penalty=heads*len(seeds)/(len(seeds)-1)*epsilon2
                bound=2*np.sqrt(base_chi*penalty)+penalty
                difference=abs(chi-base_chi)
                if difference>bound+1e-12:raise AssertionError('Sample susceptibility transfer inequality failed')
                entry[name]=dict(mean=float(values.mean()),susceptibility=chi,base_susceptibility=base_chi,
                    susceptibility_ratio=chi/base_chi if base_chi>0 else None,collective_squared_error=epsilon2,
                    absolute_susceptibility_error=difference,susceptibility_error_bound=float(bound),
                    relative_error_bound=float(bound/base_chi) if base_chi>0 else None)
            if variant=='initial_mean_compressed':entry['maximum_broadcast_compressed_kl']=max(float(d['compressed_broadcast_kl'].max()) for d in data)
            rows.append(entry)
        for d in data:d.close()
    for case in released:
        data=np.load(study/'analysis'/case['output_name']/'measurements.npz')
        for variant in spec['variants']:
            kls=data[variant+'_kl'];fields=data[variant+'_fields'];heads=data[variant+'_heads']
            row=dict(model=Path(case['model_dir']).name,case=case['output_name'],variant=variant,mean_kl=float(kls.mean()),maximum_kl=float(kls.max()),
                mean_nll_difference=float(np.mean(fields[:,25].astype(float)-data['base_fields'][:,25].astype(float))),
                mean_row_energy=float(heads[...,2].mean()),
                base_row_energy_mean=float(data['base_heads'][...,2].astype(float).mean()),
                base_row_energy_median=float(np.median(data['base_heads'][...,2].astype(float))),
                base_row_energy_max=float(data['base_heads'][...,2].max()),
                cohort_shape=list(data['base_heads'][...,2].shape))
            if variant=='initial_mean_compressed':row['maximum_broadcast_compressed_kl']=float(data['compressed_broadcast_kl'].max())
            individual.append(row)
        data.close()
    write_json(out/'results.json',dict(schema='row-projection-analysis-v1',rows=rows,released=individual,variants=spec['variants'],implementation_comparison=implementation,
        interpretation='Native CPU float32 complete-graph interventions on the same 64 contexts. Conditional susceptibility averages unbiased seed covariance at fixed context and decoder; four initialization identities are available per controlled condition. Released models are individual frozen laws, not a trained-seed ensemble. Small predictive KL alone does not transfer an internal susceptibility. All declared projections are retained, including early states and centroid-only variants.'))
    write_json(out/'manifest.json',dict(status='complete',binding_sha256=sha256(out/'binding.json'),results_sha256=sha256(out/'results.json')))
    print('Analyzed',len(rows),'controlled projections and',len(individual),'released-model projections')


if __name__=='__main__':main()
