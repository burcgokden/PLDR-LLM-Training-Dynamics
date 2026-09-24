#!/usr/bin/env python
"""Compare unperturbed native replays with the finite physical-pulse separation."""
import argparse
import json
from pathlib import Path
import numpy as np
from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


def rms(values):
    return np.sqrt(np.mean(np.asarray(values,dtype=float)**2,axis=tuple(range(1,values.ndim))))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    study=Path(args.root)/'criticality-study-20260905';out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    protocol=study/'protocols/training-replay.json';parents=json.loads(protocol.read_text())['parents']
    pairs=[];inputs=[protocol]
    for parent in parents:
        suffix=parent.split('-',1)[1]
        replay=study/'training-replays'/('replay-'+suffix)
        original=study/'runs'/('restoration-'+suffix)
        pairs.append((replay,original))
        inputs.extend([p/name for p in [replay,original] for name in ['manifest.json','measurements.npz','sampling.npz']])
    bind_run(out,inputs,vars(args));results=[]
    for replay,original in pairs:
        meta=json.loads((replay/'manifest.json').read_text());oldmeta=json.loads((original/'manifest.json').read_text())
        if meta['status']!='complete' or oldmeta['status']!='complete':raise AssertionError('An input branch is incomplete')
        for folder,record in [(replay,meta),(original,oldmeta)]:
            if sha256(folder/'measurements.npz')!=record['raw_sha256']:raise AssertionError('A branch measurement changed')
        raw=np.load(replay/'measurements.npz');old=np.load(original/'measurements.npz')
        np.testing.assert_array_equal(raw['steps'],old['steps'])
        if meta['condition']!=oldmeta['condition']:raise AssertionError('Replay changes the parent condition')
        initial_errors={name:float(np.max(np.abs(raw[name+'_fields'][0].astype(float)-old['base_fields'][0].astype(float)))) for name in ['base','replay']}
        first=np.load(replay/'sampling.npz');second=np.load(original/'sampling.npz')
        for name in ['rows','offsets']:np.testing.assert_array_equal(first[name],second[name])
        fields={}
        for field,index in [('attention_entropy',0),('row_energy',2)]:
            get=lambda data,name:data[name+'_heads'][...,index].mean(-1).astype(float)
            repeated=get(raw,'replay')-get(raw,'base')
            external=get(raw,'base')-get(old,'base')
            pulsed=get(old,'plus')-get(old,'minus')
            repeat_norm=rms(repeated);external_norm=rms(external);pulse_norm=rms(pulsed)
            fields[field]=dict(replay_separation_rms=repeat_norm.tolist(),original_baseline_difference_rms=external_norm.tolist(),
                               pulse_separation_rms=pulse_norm.tolist(),
                               final_replay_to_pulse_ratio=float(repeat_norm[-1]/pulse_norm[-1]) if pulse_norm[-1]>1e-14 else None,
                               final_original_to_pulse_ratio=float(external_norm[-1]/pulse_norm[-1]) if pulse_norm[-1]>1e-14 else None)
        results.append(dict(run_id=replay.name,heads=meta['condition']['heads'],seed=meta['condition']['seed'],steps=raw['steps'].tolist(),fields=fields,
            initial_field_max_errors=initial_errors,losses_bitwise_equal=bool(np.array_equal(raw['base_losses'],raw['replay_losses'])),
            heads_bitwise_equal=bool(np.array_equal(raw['base_heads'],raw['replay_heads'])),
            original_losses_bitwise_equal=bool(np.array_equal(raw['base_losses'],old['base_losses'])),
            replay_kl_mean=raw['replay_kl'].mean(1).tolist(),replay_kl_max=raw['replay_kl'].max(1).tolist(),
            peak_cuda_gb=meta['peak_cuda_gb'],seconds=meta['seconds']))
    write_json(out/'results.json',dict(schema='training-replay-analysis-v1',runs=results,
        interpretation='Two nominally identical native branches per parent, plus comparison to the recorded baseline. These controls assess numerical repeatability of finite trajectories; they do not add independent seeds or establish a critical instability.'))
    write_json(out/'manifest.json',dict(results_sha256=sha256(out/'results.json'),binding_sha256=sha256(out/'binding.json')))


if __name__=='__main__':main()
