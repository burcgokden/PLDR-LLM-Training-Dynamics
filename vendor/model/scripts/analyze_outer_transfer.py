#!/usr/bin/env python
"""Analyze every frozen outer-transfer candidate, state, target, and scale."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from model_rg.moment_rg import block_maps, score
from model_rg.provenance import sha256, write_json


def empirical(x):
    mean=x.mean(0);z=x-mean
    return mean,z.T@z/len(x)


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);a=p.parse_args()
    study=Path(a.study).resolve();dest=study/'analysis.json'
    if dest.exists():raise FileExistsError(dest)
    spec=json.loads((study/'protocol.json').read_text());archive=json.loads((study/'archive-forecast.json').read_text())
    assert json.loads((study/'launcher.json').read_text())['status']=='complete'
    inputs={};records=[];means=[];coverage=[];inference=[];paths_by_state=[];all_scores={};performance=[];primary=[]
    def read(path):
        inputs[str(path)]=sha256(path);return json.loads(path.read_text())
    for case in spec['cases']:
        folder=study/'runs'/case['name'];meta=read(folder/'results.json')
        fit=read(folder/'adapted-forecast.json');tube=read(folder/'calibration.json')
        assert meta['status']=='complete' and meta['replay_bitwise'] and meta['restoration_bitwise']
        assert len(meta['records'])==56
        inputs[str(folder/'primary.npz')]=sha256(folder/'primary.npz')
        assert inputs[str(folder/'primary.npz')]==meta['files']['primary.npz']
        with np.load(folder/'primary.npz') as raw:
            primary.append(dict(case=case['name'],heads=case['heads'],corpus=case['corpus'],identity=case['identity'],
                initial_risk=float(raw['initial_nll'].mean()),incoming_risk=float(raw['incoming_nll'].mean()),
                initial_entropy=float(raw['initial_entropy'].mean()),incoming_entropy=float(raw['incoming_entropy'].mean()),
                initial_rows=raw['initial_rows'].tolist(),incoming_rows=raw['incoming_rows'].tolist()))
        values=[]
        for rec in meta['records']:
            raw=folder/rec['raw'];assert sha256(raw)==rec['sha256'];inputs[str(raw)]=rec['sha256']
            with np.load(raw,allow_pickle=False) as f:
                assert list(f['horizons'])==spec['horizons']
                values.append(np.stack([f['nll'].mean(1),f['entropy'].mean(1)],axis=1))
        paths=np.array(values);assert paths.shape==(56,5,2)
        assert np.all(paths[:,0]==paths[0,0])
        held=paths[24:];paths_by_state.append(held)
        for count,target in [(1,'risk'),(2,'risk_entropy')]:
            x=np.diff(held[:,:,:count],axis=1).transpose(0,2,1).reshape(32,-1)
            fitted=fit['fits'][target];mu=np.array(fitted['mean'])
            covs={key:np.array(value) for key,value in fitted['covariances'].items()}
            for scale,b in block_maps(count).items():
                y=x@b.T;ym,cpop=empirical(y);pred=b@mu;cov={key:b@value@b.T for key,value in covs.items()}
                scores={key:score(y,pred,value) for key,value in cov.items()}
                key=case['name']+'__'+target+'__'+scale
                all_scores.update({key+'__'+name:value for name,value in scores.items()})
                comparisons={}
                for first,second in [('local','diagonal'),('transferred','diagonal'),('transferred','local')]:
                    delta=scores[first]-scores[second];precision=np.linalg.inv(cov[first])-np.linalg.inv(cov[second]);bias=ym-pred
                    components=dict(logdet=float(np.linalg.slogdet(cov[first])[1]-np.linalg.slogdet(cov[second])[1]),
                        covariance=float(np.trace(precision@cpop)),mean_error=float(bias@precision@bias))
                    assert abs(sum(components.values())-delta.mean())<1e-8
                    comparisons[first+'__'+second]=dict(mean=float(delta.mean()),mcse=float(delta.std(ddof=1)/np.sqrt(32)),components=components)
                records.append(dict(case=case['name'],heads=case['heads'],corpus=case['corpus'],identity=case['identity'],
                    target=target,scale=scale,dimension=y.shape[1],comparisons=comparisons,
                    empirical_mean=ym.tolist(),empirical_covariance=cpop.tolist(),
                    mean_scores={key:float(value.mean()) for key,value in scores.items()}))
        observed=held[:,1:,0]-held[:,0,None,0]
        avg=observed.mean(0);se=observed.std(0,ddof=1)/np.sqrt(32)
        candidates=dict(identity=np.zeros(4),archive=np.cumsum(archive['fits'][str(case['heads'])]['risk']['mean']),
                        adapted=np.cumsum(fit['fits']['risk']['mean']))
        means.append(dict(case=case['name'],heads=case['heads'],corpus=case['corpus'],identity=case['identity'],
            incoming_risk=float(held[0,0,0]),observed_mean_change=avg.tolist(),mcse=se.tolist(),
            candidates={key:dict(prediction=value.tolist(),absolute_error=np.abs(value-avg).tolist(),
                maximum_absolute_error=float(np.max(np.abs(value-avg))),endpoint_absolute_error=float(abs(value[-1]-avg[-1]))) for key,value in candidates.items()}))
        z=np.max(np.abs(observed-np.array(tube['mean']))/np.array(tube['scale']),axis=1)
        covered=z<=tube['radius']
        images={}
        difference=np.eye(4)-np.eye(4,k=-1)
        for scale_name,block in block_maps(1).items():
            mapped=block@difference
            widths=tube['radius']*(abs(mapped)@np.array(tube['scale']))
            errors=abs((observed-np.array(tube['mean']))@mapped.T)
            flags=np.all(errors<=widths+1e-12,axis=1)
            assert np.all(flags[covered])
            images[scale_name]=dict(covered=int(flags.sum()),half_widths=widths.tolist())
        coverage.append(dict(case=case['name'],covered=int(covered.sum()),assessment=32,
            radius=tube['radius'],half_widths=(tube['radius']*np.array(tube['scale'])).tolist(),scores=z.tolist(),
            indicators=covered.tolist(),scale_enclosures=images))
        for length,entry in meta['inference'].items():
            raw=folder/entry['raw'];assert sha256(raw)==entry['sha256'];inputs[str(raw)]=entry['sha256']
            with np.load(raw) as f:arr={k:f[k] for k in f.files}
            z=arr['native'].astype(float);z-=z.max(-1,keepdims=True);lz=z-np.log(np.exp(z).sum(-1,keepdims=True));prob=np.exp(lz)
            for name in ['fixed','projected']:
                w=arr[name].astype(float);w-=w.max(-1,keepdims=True);lw=w-np.log(np.exp(w).sum(-1,keepdims=True))
                delta=lw-lz;labels=arr['targets'];index=np.arange(len(labels))
                kl=np.sum(prob*(lz-lw),axis=1);risk=-delta[index,labels]
                oscillation=np.ptp(arr[name].astype(float)-arr['native'].astype(float),axis=1)
                assert np.all(np.abs(risk)<=oscillation+1e-12)
                inference.append(dict(case=case['name'],heads=case['heads'],prefix=int(length),operation=name,
                    native_risk=float(-lz[index,labels].mean()),risk_change=float(risk.mean()),
                    maximum_absolute_target_score_change=float(np.abs(risk).max()),mean_forward_kl=float(kl.mean()),
                    maximum_oscillation=float(oscillation.max()),top1_changed=int(np.sum(arr['native'].argmax(-1)!=arr[name].argmax(-1)))))
        performance.append(dict(case=case['name'],seconds=meta['seconds'],peak_cuda_bytes=meta['peak_cuda_bytes'],
            initial_component_sha256=meta['initial_component_sha256']))
    summaries={}
    for target in ['risk','risk_entropy']:
        for scale in ['fine','paired','endpoint']:
            selected=[r for r in records if r['target']==target and r['scale']==scale]
            summaries[target+'__'+scale]={}
            for comp in ['local__diagonal','transferred__diagonal','transferred__local']:
                dif=[r['comparisons'][comp]['mean'] for r in selected]
                summaries[target+'__'+scale][comp]=dict(equal_state_mean=float(np.mean(dif)),favorable_states=sum(x<0 for x in dif),states=len(dif))
    hierarchy=[];paths=np.array(paths_by_state)
    for heads in [4,14]:
        indices=[i for i,c in enumerate(spec['cases']) if c['heads']==heads]
        # Corpus -> initialization -> continuation. Population divisors specify
        # an exact empirical law, rather than unbiased outer-population estimates.
        x=paths[indices,:,-1,0].reshape(2,2,32)
        within=float(np.mean(np.var(x,axis=2)));init_means=x.mean(2)
        initialization=float(np.mean(np.var(init_means,axis=1)));corpus=float(np.var(init_means.mean(1)))
        total=float(np.var(x));assert abs(total-within-initialization-corpus)<1e-12
        hierarchy.append(dict(heads=heads,within=within,initialization=initialization,corpus=corpus,total=total))
    np.savez_compressed(study/'scores.npz',**all_scores)
    write_json(dest,dict(schema='outer-transfer-results-v1',status='complete',completed_at=datetime.now(timezone.utc).isoformat(),
        primary_transitions=primary,records=records,summaries=summaries,mean_predictions=means,coverage=coverage,inference=inference,hierarchy=hierarchy,performance=performance,
        primary_paths=8,primary_updates=16384,conditional_paths=448,conditional_updates=28672,assessment_paths=256,
        fitting_paths=128,calibration_paths=64,corpus_draws=2,nested_full_initializations=4,
        total_covered=sum(x['covered'] for x in coverage),protocol_sha256=sha256(study/'protocol.json'),
        inputs_sha256=inputs,scores_sha256=sha256(study/'scores.npz'),analyzer_sha256=sha256(__file__)))
    print(json.dumps(dict(summaries=summaries,coverage=sum(x['covered'] for x in coverage),assessment=256),indent=2))


if __name__=='__main__':main()
