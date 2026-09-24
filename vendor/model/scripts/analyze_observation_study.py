#!/usr/bin/env python
"""Analyze completed arithmetic cells and paired horizons without altering their estimands."""
import argparse
import json
from pathlib import Path
import numpy as np
from model_rg.controlled import bind_run
from model_rg.observation import paired_horizon
from model_rg.provenance import sha256, write_json


def analyze_horizons(root,study):
    protocol_path=study/'protocols/paired-horizon.json'
    protocol=json.loads(protocol_path.read_text());ledger=json.loads((study/'launcher-paired-horizon.json').read_text())
    if ledger['status']!='complete' or len(ledger['records'])!=len(protocol['jobs']):
        raise AssertionError('The paired continuation inventory is not complete')
    dynamics=root/'criticality-dynamics-20260906';data={};inputs=[protocol_path,study/'launcher-paired-horizon.json']
    for h in [2,4,8,14,24]:
        for seed in range(640101,640105):data[h,seed]=dynamics/'runs'/f'long-h{h}-g1-s{seed}'
    for job in protocol['jobs']:data[14,job['seed']]=study/'runs'/job['run_id']
    loaded={}
    for key,path in data.items():
        meta=json.loads((path/'manifest.json').read_text());args=meta['arguments']
        if meta['status']!='complete' or meta['completed_step']!=16384 or meta['start_step']!=8192:
            raise AssertionError('Invalid paired horizon outcome')
        if (args['heads'],args['seed'])!=key or args['multiplier']!=1 or args['shared_seed']!=640011 or args['stream_seed']!=640001:
            raise AssertionError('Changed conditional family')
        if sha256(path/'measurements.npz')!=meta['raw_sha256']:raise AssertionError('Changed paired observations')
        inputs.extend([path/'manifest.json',path/'measurements.npz',path/'sampling.npz'])
        with np.load(path/'measurements.npz') as z:
            loaded[key]={name:z[name].astype(float) for t in [8192,16384]
                for name in [f'heads_{t}',f'fields_{t}',f'context_kl_{t}']}
    records=[];arrays={}
    groups=[(f'four-N{h}',h,list(range(640101,640105))) for h in [2,4,8,14,24]]
    groups += [('all-sixteen-N14',14,list(range(640101,640117))),
               ('additional-twelve-N14',14,list(range(640105,640117)))]
    for name,h,seeds in groups:
        before=np.stack([loaded[h,s]['heads_8192'][...,2].mean(-1) for s in seeds])
        after=np.stack([loaded[h,s]['heads_16384'][...,2].mean(-1) for s in seeds])
        result,raw=paired_horizon(before,after,h,seeds,protocol['analysis']['resamples'],protocol['analysis']['bootstrap_seed'])
        result.update(group=name,mean_before=float(before.mean()),mean_after=float(after.mean()),
            nll_before=[float(loaded[h,s]['fields_8192'][:,25].mean()) for s in seeds],
            nll_after=[float(loaded[h,s]['fields_16384'][:,25].mean()) for s in seeds],
            context_kl_before=[float(loaded[h,s]['context_kl_8192'].mean()) for s in seeds],
            context_kl_after=[float(loaded[h,s]['context_kl_16384'].mean()) for s in seeds])
        records.append(result)
        arrays[name+'-before']=before;arrays[name+'-after']=after
        arrays.update({name+'-'+key:value for key,value in raw.items()})
    return records,arrays,inputs


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='observation-closure-20260906');p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.root);study=root/a.study;out=Path(a.output)
    out.mkdir(parents=True,exist_ok=False)
    horizons,raw,inputs=analyze_horizons(root,study)
    protocol_path=study/'protocols/arithmetic.json';protocol=json.loads(protocol_path.read_text())
    ledger=json.loads((study/'launcher-arithmetic.json').read_text())
    if ledger['status']!='complete' or len(ledger['records'])!=len(protocol['cells']):raise AssertionError('Incomplete arithmetic inventory')
    inputs += [protocol_path,study/'launcher-arithmetic.json'];arithmetic=[];arithmetic_energies=[]
    for cell in protocol['cells']:
        path=study/'runs'/f"arithmetic-h{cell['heads']}-g{cell['multiplier']}-t{cell['step']}"
        meta=json.loads((path/'manifest.json').read_text())
        for name,key in [('results.json','results_sha256'),('measurements.npz','raw_sha256')]:
            if sha256(path/name)!=meta[key]:raise AssertionError('Changed arithmetic result')
            inputs.append(path/name)
        inputs.append(path/'manifest.json');arithmetic.append(json.loads((path/'results.json').read_text()))
        with np.load(path/'measurements.npz') as z:
            energy=dict(cell=cell)
            for key in ['E32','T32','R32','E64','T64','R64','Rarchive']:
                fields=np.stack([z[f's{seed}_{key}'] for seed in protocol['seeds']])
                energy['mean_'+key]=float(fields.mean())
            total=np.stack([z[f's{seed}_T64'] for seed in protocol['seeds']])
            energy['minimum_T64']=float(total.min());energy['fraction_T64_below_floor']=float(np.mean(total<1e-30))
            arithmetic_energies.append(energy)
    row_projection=study/'analysis/row-projection/results.json'
    inputs.append(row_projection);released=json.loads(row_projection.read_text())['released']
    # Entry-resolution fractions in the existing rate-time and saturation plotting cohorts.
    # These remain diagnostics on the actual plotted cohort, distinct from the 512-context controls.
    screen_inputs=[];screens=[]
    for h in [4,8,14]:
        for g in [1.5,2,3,4]:
            arrays=[];endpoints=[]
            for seed in range(640101,640105):
                path=root/'criticality-dynamics-20260906/runs'/f'pilot-h{h}-g{g}-s{seed}'/'measurements.npz'
                screen_inputs.append(path)
                with np.load(path) as z:
                    arrays.append((z['steps'].copy(),z['dense_heads'][...,2].astype(float)))
                    endpoints.append(z['heads_8192'][...,2].astype(float))
            steps=arrays[0][0]
            if not all(np.array_equal(t,steps) for t,_ in arrays):raise AssertionError('Dense cohort times differ')
            fields=np.stack([field for _,field in arrays])
            screens.append(dict(heads=h,multiplier=g,steps=steps.tolist(),
                fraction=np.mean(fields<protocol['primary_screen'],axis=(0,2,3,4)).tolist(),
                threshold=protocol['primary_screen'],contexts=fields.shape[2],seeds=4,
                endpoint_fraction=float(np.mean(np.stack(endpoints)<protocol['primary_screen'])),endpoint_contexts=512))
    replication_protocol=root/'criticality-dynamics-20260906/protocols/replication.json'
    replication_spec=json.loads(replication_protocol.read_text());inputs.append(replication_protocol)
    groups={h:[] for h in [2,4,8,14,24]};replication_screens=[]
    parents=[root/'criticality-dynamics-20260906/runs'/j['run_id'] for j in replication_spec['jobs']]+[Path(p) for p in replication_spec['additional_bound_runs']]
    for parent in parents:
        meta=json.loads((parent/'manifest.json').read_text());h=meta['arguments']['heads']
        inputs += [parent/'manifest.json',parent/'measurements.npz']
        with np.load(parent/'measurements.npz') as z:
            groups[h].append({t:z[f'heads_{t}'][...,2].astype(float) for t in [2048,4096,8192]})
    for h,group in groups.items():
        for t in [2048,4096,8192]:
            fields=np.stack([r[t] for r in group])
            replication_screens.append(dict(heads=h,step=t,seeds=len(group),contexts=512,
                threshold=protocol['primary_screen'],fraction=float(np.mean(fields<protocol['primary_screen']))))
    inputs+=screen_inputs
    bind_run(out,inputs,vars(a));np.savez_compressed(out/'measurements.npz',**raw)
    result=dict(schema='observation-study-analysis-v1',horizons=horizons,arithmetic=arithmetic,
        arithmetic_energies=arithmetic_energies,released=released,dense_entry_screens=screens,replication_entry_screens=replication_screens,primary_screen=protocol['primary_screen'],
        protocol_scope='All declared continuations and arithmetic cells; no outcome-dependent omissions. Sixteen-seed horizon selection follows an unresolved four-seed comparison; the additional-twelve result is retained separately.',
        interpretation='Finite conditional row dynamics and arithmetic-qualified observations. Empirical agreement is not a certified real-arithmetic result. No critical exponent or autonomous finite row-only training law is inferred.')
    write_json(out/'results.json',result)
    write_json(out/'manifest.json',dict(status='complete',binding_sha256=sha256(out/'binding.json'),
        results_sha256=sha256(out/'results.json'),raw_sha256=sha256(out/'measurements.npz')))
    print('Analyzed',len(horizons),'paired groups and',len(arithmetic),'complete arithmetic conditions',flush=True)


if __name__=='__main__':main()
