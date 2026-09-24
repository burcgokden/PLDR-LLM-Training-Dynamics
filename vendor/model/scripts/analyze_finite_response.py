#!/usr/bin/env python
"""Independent raw-vocabulary reduction and complete locked pulse assessment."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from model_rg.provenance import sha256, write_json


def require(flag, message):
    if not flag:
        raise ValueError(message)


def analyze(study):
    study=Path(study).resolve(); spec=json.loads((study/'protocol.json').read_text())
    output=study/'verification.json'; require(not output.exists(), 'Fresh output required')
    checked={}
    def check(path,digest=None):
        path=Path(path); value=sha256(path)
        require(digest is None or digest==value,'Changed artifact: '+str(path))
        checked[str(path)]=value
    check(study/'protocol.json')
    check(__file__,spec['analysis_source_sha256'])
    for path,digest in {**spec['inputs'],**spec['sources']}.items(): check(path,digest)
    root=study.parents[1]
    with np.load(root/'outer-transfer-20260911/data/panels.npz') as f: targets=f['evaluation'][:,64]
    rows=[];prefixes=[];implementation=[];max_error=0.;identity_error=0.;coordinates=0;raw_files=0
    clips=[];moment_zeros=[];activation_mins=[];replays=[];timings=[]
    for case in spec['cases']:
        folder=study/case['name'];check(folder/'results.json');meta=json.loads((folder/'results.json').read_text())
        require(meta['status']=='complete' and meta['protocol_sha256']==sha256(study/'protocol.json'),'Incomplete worker')
        check(folder/'directions.pt',meta['directions_sha256'])
        timings.append(dict(case=case['name'],seconds=meta['seconds']))
        arrays={}
        order=np.random.default_rng(case['stream_seed']).permutation(524288)
        rng=np.random.default_rng(spec['source_seed_base']+case['heads'])
        want=np.stack([rng.choice(order[65536:],32*8,replace=False).reshape(8,32) for _ in range(4)])
        require(np.array_equal(np.load(study/(case['name']+'-blocks.npy')),want),'Wrong source law')
        require(len(meta['records'])==74,'Incomplete arm matrix')
        for record in meta['records']:
            path=Path(record['file']);check(path,record['sha256']);raw_files+=1
            with np.load(path,allow_pickle=False) as f:
                require(np.array_equal(f['times'],spec['times']) and np.array_equal(f['blocks'],want[record['source']]),'Wrong path coordinates')
                logits=f['logits'];coordinates+=logits.size
                require(logits.shape==(5,32,32000) and np.isfinite(logits).all(),'Invalid vocabulary observations')
                lp=logits-logsumexp(logits,axis=-1,keepdims=True)
                nll=-np.take_along_axis(lp,targets[None,:,None],axis=-1)[...,0]
                entropy=-(np.exp(lp)*lp).sum(-1)
                common=f['common_ms']; transverse=f['transverse_ms']
                q=np.concatenate([nll.mean(1)[:,None],entropy.mean(1)[:,None],
                    np.log1p(np.sqrt(common.mean((2,3)))),np.log1p(np.sqrt(transverse.mean((2,3))))],axis=1)
                error=max(np.max(abs(nll-f['nll'])),np.max(abs(entropy-f['entropy'])),np.max(abs(q-f['q'])))
                max_error=max(max_error,float(error));require(error<1e-10,'Raw reduction disagrees')
                arrays[record['mode'],record['source'],record['arm']]=f['q'].copy()
                require(np.isfinite(f['losses']).all() and len(f['losses'])==8,'Invalid training path')
                clips.extend(f['gradnorms'].tolist());moment_zeros.extend(f['moment_stats'][:,1].tolist())
                activation_mins.extend(f['activation_diagnostics'][:,1].tolist())
                require((f['activation_diagnostics'][:,2]==0).all(),'Nonfinite activation')
        for mode in spec['modes']:
            check(folder/mode/'metadata.json');info=json.loads((folder/mode/'metadata.json').read_text())
            require(info['replay_bitwise'],'Replay missing')
            zero=next(r for r in meta['records'] if r['mode']==mode and r['source']==0 and r['arm']=='zero')
            replay=next(r for r in meta['records'] if r['mode']==mode and r['arm']=='zero-replay')
            require(zero['final_digest']==replay['final_digest'] and zero['gradnorms']==replay['gradnorms'],'Final-state replay differs')
            require(np.array_equal(arrays[mode,0,'zero'],arrays[mode,0,'zero-replay']),'Observed replay differs')
            replays.append(dict(case=case['name'],mode=mode,peak_cuda_bytes=info['peak_cuda_bytes']))
            for family in ['generator','body']:
                for source in range(4):
                    a={tag:arrays[mode,source,f'{family}-3e-07-{tag}'] for tag in ['plus','minus','halfplus','halfminus']}
                    odd=(a['plus']-a['minus'])/2; half=(a['halfplus']-a['halfminus'])/2
                    x=arrays[mode,source,'zero']; even=(a['plus']+a['minus'])/2-x
                    identity_error=max(identity_error,float(np.max(abs(a['plus']-x-even-odd))))
                    for index,t in enumerate(spec['times']):
                        select=slice(0,index+1)
                        signal=float(np.sqrt(np.mean(half[select]**2)))
                        numerator=float(np.sqrt(np.mean((odd[select]-2*half[select])**2)))
                        denom=2*signal
                        discrepancy=numerator/denom if denom>0 else None
                        resolved=signal>spec['gate']['signal_floor']
                        row=dict(case=case['name'],heads=case['heads'],mode=mode,family=family,source=source,horizon=t,
                            discrepancy=discrepancy,half_signal_rms=signal,resolved=resolved,
                            passes=bool(resolved and discrepancy<=spec['gate']['relative_tolerance']),
                            even_over_odd=float(np.linalg.norm(even[select])/max(np.linalg.norm(odd[select]),1e-30)))
                        prefixes.append(row)
                        if t==8: rows.append(row)
        for source in range(4):
            d=arrays['native32',source,'zero']-arrays['arithmetic64',source,'zero']
            implementation.append(dict(case=case['name'],source=source,time_rms=np.sqrt(np.mean(d*d,axis=1)).tolist()))
    native=[r for r in rows if r['mode']=='native32']; controls=[r for r in rows if r['mode']=='arithmetic64']
    summary=dict(native_passes=sum(r['passes'] for r in native),native_cells=len(native),
                 control_passes=sum(r['passes'] for r in controls),control_cells=len(controls),
                 all_native_pass=all(r['passes'] for r in native),
                 native_updates=576,arithmetic_control_updates=576,replay_updates=32,
                 gradient_norm_range=[min(clips),max(clips)],all_clipping_active=min(clips)>1,
                 second_moment_zero_count_range=[min(moment_zeros),max(moment_zeros)],
                 smooth_power_activation_min=min(activation_mins),raw_files=raw_files,vocabulary_coordinates=coordinates,
                 maximum_reduction_error=max_error,maximum_identity_error=identity_error)
    write_json(output,dict(status='passed',schema='finite-response-assessment-verification-v1',summary=summary,
        primary=rows,prefixes=prefixes,implementation=implementation,replays=replays,timings=timings,
        checked_sha256=checked,verifier_sha256=sha256(__file__),
        scope='All prescribed new trajectories, source arrays, raw logits and fixed metric summaries. Metric tensors are not independently reconstructed. Successful verification means complete valid data, not acceptance of the response hypothesis. Time-zero observations reuse incoming states.'))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);a=p.parse_args();analyze(a.study)
