#!/usr/bin/env python
"""Reduce every frozen state, budget, fit replicate, target and scale."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
from model_rg.moment_rg import block_maps, score
from model_rg.provenance import sha256, write_json
from model_rg.refresh import archive_fit, increments


def reduce(study):
    dest=study/'analysis.json'
    if dest.exists():
        raise FileExistsError(dest)
    spec=json.loads((study/'protocol.json').read_text())
    assert json.loads((study/'launcher.json').read_text())['status']=='complete'
    archive=json.loads((study/'archive-forecast.json').read_text())['fits']
    inputs={}; records=[]; comparisons=[]; means=[]; coverage=[]; costs=[]; gates=[]; scores_out={}
    def read(p):
        inputs[str(p)]=sha256(p);return json.loads(p.read_text())
    for ci,case in enumerate(spec['cases']):
        folder=study/'runs'/case['name'];meta=read(folder/'results.json')
        assert meta['status']=='complete'
        state_fits={};state_tubes={}
        for state in ['parent','successor']:
            f=folder/state; state_fits[state]={};state_tubes[state]={}
            for rep in range(2):
                state_fits[state][rep]=read(f/f'fit-{rep}.json')['fits']
                state_tubes[state][rep]=read(f/f'calibration-{rep}.json')['tubes']
        for si,state in enumerate(['parent','successor']):
            f=folder/state; state_meta=read(f/'results.json')
            inputs[str(f/'paths.npy')]=sha256(f/'paths.npy')
            paths=np.load(f/'paths.npy');held=paths[48:];n=len(held)
            assert n==spec['assessment'] and np.all(paths[:,0]==paths[0,0])
            bootstrap=np.random.default_rng(spec['bootstrap_seed']+ci*2+si).integers(n,size=(spec['bootstrap_samples'],n))
            observed=held[:,1:,0]-held[:,0,None,0]
            # Fresh conditional assessment randomness; fit and panel are fixed.
            bootstrap_mean=observed[bootstrap].mean(1)
            predictions={};tubes={}
            for source in (['current'] if state=='parent' else ['current','parent_reuse']):
                src=state if source=='current' else 'parent'
                for rep in range(2):
                    for budget in spec['budgets']:
                        key=f'{source}_r{rep}_m{budget}'
                        predictions[key]=dict(source=source,replicate=rep,budget=budget,fit=state_fits[src][rep][str(budget)])
                        tubes[key]=state_tubes[src][rep][str(budget)]
            for kind in ['archive','no_change']:
                predictions[kind]=dict(source=kind,replicate=-1,budget=0,fit=archive_fit(archive[str(case['heads'])],kind=='no_change'))
            mean_by_key={}
            for key,pred in predictions.items():
                mu=np.cumsum(pred['fit']['risk']['mean'])
                error=np.abs(observed.mean(0)-mu);boot_error=np.max(np.abs(bootstrap_mean-mu),axis=1)
                row=dict(case=case['name'],heads=case['heads'],state=state,key=key,source=pred['source'],
                         replicate=pred['replicate'],budget=pred['budget'],prediction=mu.tolist(),
                         assessment_mean=observed.mean(0).tolist(),maximum_error=float(error.max()),
                         bootstrap_interval=np.quantile(boot_error,[.025,.975]).tolist())
                means.append(row);mean_by_key[key]=row
                if key in tubes:
                    tube=tubes[key];scale=np.asarray(tube['scale']);radius=tube['radius']
                    z=np.max(np.abs(observed-np.asarray(tube['mean']))/scale,axis=1);flags=z<=radius
                    images={};difference=np.eye(4)-np.eye(4,k=-1)
                    for name,b in block_maps(1).items():
                        mapped=b@difference;widths=radius*(np.abs(mapped)@scale)
                        image_flags=np.all(np.abs((observed-np.asarray(tube['mean']))@mapped.T)<=widths+1e-12,axis=1)
                        assert np.all(image_flags[flags])
                        images[name]=dict(covered=int(image_flags.sum()),half_widths=widths.tolist())
                    coverage.append(dict(case=case['name'],state=state,key=key,source=pred['source'],
                        replicate=pred['replicate'],budget=pred['budget'],covered=int(flags.sum()),assessment=n,
                        indicators=flags.tolist(),half_widths=(radius*scale).tolist(),scale_enclosures=images))
            for rep in range(2):
                for budget in spec['budgets']:
                    acquired=sum(r['seconds'] for r in state_meta['records'][rep*24:rep*24+budget])
                    cal=sum(r['seconds'] for r in state_meta['records'][rep*24+16:rep*24+24])
                    costs.append(dict(case=case['name'],state=state,replicate=rep,budget=budget,
                        adaptation_updates=budget*spec['horizon'],calibration_updates=8*spec['horizon'],
                        adaptation_seconds=acquired,fit_seconds=state_meta['fit_seconds'][str(rep)][str(budget)],
                        calibration_path_seconds=cal,calibration_seconds=state_meta['calibration_seconds'][str(rep)][str(budget)],
                        assessment_seconds=sum(r['seconds'] for r in state_meta['records'][48:])))
            fine_differences={};query_times={}
            for count,target in [(1,'risk'),(2,'risk_entropy')]:
                x=increments(held,count)
                for scale,b in block_maps(count).items():
                    y=x@b.T;cell_scores={}
                    for key,pred in predictions.items():
                        fitted=pred['fit'][target];mu=b@np.asarray(fitted['mean'])
                        for candidate,c in fitted['covariances'].items():
                            covariance=b@np.asarray(c)@b.T
                            tick=time.perf_counter();values=score(y,mu,covariance);elapsed=time.perf_counter()-tick
                            label=key+'__'+candidate;cell_scores[label]=values
                            fullkey='__'.join([case['name'],state,target,scale,label])
                            scores_out[fullkey]=values
                            records.append(dict(case=case['name'],state=state,target=target,scale=scale,
                                dimension=y.shape[1],key=key,source=pred['source'],replicate=pred['replicate'],
                                budget=pred['budget'],candidate=candidate,mean_score=float(values.mean()),
                                query_seconds=elapsed,score_key=fullkey))
                            if target=='risk' and scale=='fine':
                                query_times[label]=elapsed
                    pairs=[]
                    for key,pred in predictions.items():
                        if pred['budget']==0:
                            continue
                        pairs.extend([('covariance',key+'__local',key+'__diagonal'),
                                      ('covariance',key+'__transferred',key+'__diagonal'),
                                      ('covariance',key+'__transferred',key+'__local')])
                        for candidate in ['local','diagonal','transferred']:
                            if pred['budget']!=16:
                                reference=f"{pred['source']}_r{pred['replicate']}_m16__{candidate}"
                                pairs.append(('budget',key+'__'+candidate,reference))
                            if pred['source']=='parent_reuse':
                                reference=f"current_r{pred['replicate']}_m{pred['budget']}__{candidate}"
                                pairs.append(('reuse',key+'__'+candidate,reference))
                    for kind,first,second in pairs:
                        delta=cell_scores[first]-cell_scores[second]
                        interval=np.quantile(delta[bootstrap].mean(1),[.025,.975])
                        comparisons.append(dict(case=case['name'],state=state,target=target,scale=scale,
                            kind=kind,first=first,second=second,mean=float(delta.mean()),
                            mcse=float(delta.std(ddof=1)/np.sqrt(n)),bootstrap_interval=interval.tolist()))
                        if target=='risk' and scale=='fine' and kind=='budget' and first.startswith('current'):
                            fine_differences[first]=dict(mean=float(delta.mean()),interval=interval)
            for rep in range(2):
                for budget in [4,8]:
                    key=f'current_r{rep}_m{budget}';m=mean_by_key[key]
                    cost=next(c for c in costs if c['case']==case['name'] and c['state']==state and c['replicate']==rep and c['budget']==budget)
                    reference=next(c for c in costs if c['case']==case['name'] and c['state']==state and c['replicate']==rep and c['budget']==16)
                    for candidate in ['local','diagonal','transferred']:
                        label=key+'__'+candidate;d=fine_differences[label]
                        cheap=cost['adaptation_seconds']+cost['fit_seconds']+query_times[label] < reference['adaptation_seconds']+reference['fit_seconds']+query_times[f'current_r{rep}_m16__{candidate}']
                        observed_pass=m['maximum_error']<=.01 and d['mean']<=.05 and cheap
                        upper_pass=m['bootstrap_interval'][1]<=.01 and d['interval'][1]<=.05 and cheap
                        gates.append(dict(case=case['name'],state=state,replicate=rep,budget=budget,candidate=candidate,
                            observed_pass=bool(observed_pass),upper_bound_pass=bool(upper_pass),cheaper=bool(cheap),
                            mean_error=m['maximum_error'],mean_error_interval=m['bootstrap_interval'],
                            score_excess=d['mean'],score_excess_interval=d['interval'].tolist()))
    np.savez_compressed(study/'scores.npz',**scores_out)
    summary=[]
    for state in ['parent','successor']:
        for budget in spec['budgets']:
            subset=[r for r in means if r['state']==state and r['source']=='current' and r['budget']==budget]
            cov=[r for r in coverage if r['state']==state and r['source']=='current' and r['budget']==budget]
            summary.append(dict(state=state,budget=budget,fit_cells=len(subset),
                mean_tolerance_count=sum(r['maximum_error']<=.01 for r in subset),
                mean_error_range=[min(r['maximum_error'] for r in subset),max(r['maximum_error'] for r in subset)],
                covered=sum(r['covered'] for r in cov),assessment_events=sum(r['assessment'] for r in cov),
                endpoint_half_width_range=[min(r['half_widths'][-1] for r in cov),max(r['half_widths'][-1] for r in cov)]))
    write_json(dest,dict(schema='onepass-refresh-results-v1',status='complete',completed_at=datetime.now(timezone.utc).isoformat(),
        qualification=spec['qualification'],primary=spec['primary'],records=records,comparisons=comparisons,
        means=means,coverage=coverage,costs=costs,gates=gates,summary=summary,
        scientific_updates=spec['scientific_updates'],conditional_paths=len(spec['cases'])*2*(48+spec['assessment']),
        successor_paths=len(spec['cases']),assessment_paths=len(spec['cases'])*2*spec['assessment'],
        protocol_sha256=sha256(study/'protocol.json'),inputs_sha256=inputs,scores_sha256=sha256(study/'scores.npz'),
        analyzer_sha256=sha256(__file__)))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True)
    reduce(Path(p.parse_args().study).resolve())
