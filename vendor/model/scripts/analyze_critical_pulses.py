#!/usr/bin/env python3
"""Amplitude consistency and predictive visibility of the complete paired pulse grid."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from model_rg.provenance import sha256,write_json


def rms(x,axes):return np.sqrt(np.mean(x*x,axis=axes))


def analyze(root,out):
    if out.exists():raise FileExistsError(out)
    p=json.loads((root/'protocol.json').read_text());ph=sha256(root/'protocol.json')
    checked={str(root/'protocol.json'):ph};cases=[];scientific=qualification=0
    for case in p['cases']:
        folder=root/'cases'/case['case_id'];mp=folder/'manifest.json'
        m=json.loads(mp.read_text());checked[str(mp)]=sha256(mp)
        if m['protocol_sha256']!=ph or m['case']!=case:raise ValueError('Pulse case binding differs')
        if m['status']=='unresolved_direction':
            cases.append(dict(case=case,status=m['status'],reason=m['reason']));continue
        if m['status']!='complete' or not m['full_state_baseline_replay']:raise ValueError('Incomplete pulse')
        if sha256(folder/'direction.pt')!=m['direction_sha256']:raise ValueError('Direction changed')
        checked[str(folder/'direction.pt')]=m['direction_sha256']
        arrays={}
        for arm in m['arms']:
            f=folder/(arm['arm']+'.npz');digest=sha256(f)
            if digest!=arm['artifact_sha256']:raise ValueError('Pulse artifact changed')
            checked[str(f)]=digest;arrays[arm['arm']]=np.load(f)
            if arm['role']=='scientific':scientific+=arm['updates']
            elif arm['role']=='qualification':qualification+=arm['updates']
            else:raise ValueError('Unknown pulse role')
        base=arrays['baseline'];replay=arrays['baseline_replay']
        if base.files!=replay.files or not all(np.array_equal(base[k],replay[k]) for k in base.files):
            raise ValueError('Baseline arrays do not replay')
        if m['arms'][0]['full_final_state_sha256']!=m['arms'][-1]['full_final_state_sha256']:
            raise ValueError('Full final state replay differs')
        radii=[r*m['generator_weight_norm'] for r in p['design']['radii']]
        large=(arrays['plus_large']['fields']-arrays['minus_large']['fields'])/(2*radii[0])
        small=(arrays['plus_small']['fields']-arrays['minus_small']['fields'])/(2*radii[1])
        norms=rms(small,1);error=rms(large-small,1)
        qualified=(error<=.05*norms)&(norms>1e-10)
        field_records=[]
        for i,name in enumerate(['row','attention','common_projection','operator_rms']):
            initial=norms[0,i]
            ratios=(norms[:,i]/initial).tolist() if initial>1e-10 else [None]*len(norms)
            field_records.append(dict(field=name,small_secant_rms=norms[:,i].tolist(),
                secant_discrepancy_rms=error[:,i].tolist(),five_percent_consistent=qualified[:,i].tolist(),
                normalized_response=ratios,stationary_gap=None,
                scope='Finite conditional pulse propagation; consistency at two radii does not prove an infinitesimal derivative or a stationary eigenmode.'))
        zb=base['logits'].astype(float);prob=np.exp(zb-logsumexp(zb,axis=-1,keepdims=True))
        dl=(arrays['plus_large']['logits'].astype(float)-arrays['minus_large']['logits'].astype(float))/(2*radii[0])
        ds=(arrays['plus_small']['logits'].astype(float)-arrays['minus_small']['logits'].astype(float))/(2*radii[1])
        dl-=np.sum(prob*dl,axis=-1,keepdims=True);ds-=np.sum(prob*ds,axis=-1,keepdims=True)
        fisher=np.sqrt(np.mean(np.sum(prob*ds*ds,axis=-1),axis=-1))
        ferr=np.sqrt(np.mean(np.sum(prob*(dl-ds)**2,axis=-1),axis=-1))
        initial_fields=base['fields'][0]
        result=dict(case=case,status='complete',steps=base['steps'].tolist(),fields=field_records,
            initial_mean_fields=initial_fields.mean(0).tolist(),
            predictive_fisher_norm=fisher.tolist(),predictive_secant_discrepancy=ferr.tolist(),
            predictive_five_percent_consistent=((ferr<=.05*fisher)&(fisher>1e-10)).tolist(),
            relative_radii=p['design']['radii'],absolute_radii=radii,
            gradient_norm=m['gradient_norm'],generator_weight_norm=m['generator_weight_norm'],
            source_prefix_updates=case['job']['steps'],source_suffix_updates=p['design']['steps'],
            scientific_updates=m['scientific_updates'],qualification_updates=m['qualification_updates'])
        cases.append(result)
        for a in arrays.values():a.close()
    result=dict(schema='critical-native-pulse-analysis-v1',status='complete',protocol_sha256=ph,
        role=p['design']['role'],cases=cases,scientific_updates=scientific,qualification_updates=qualification,
        checked_sha256=checked,analyzer_sha256=sha256(__file__),
        inference='Native conditional finite responses. Frozen generator coordinates at g=0 supply a neutral reference and do not establish intrinsic criticality.')
    write_json(out,result)
    print(json.dumps({k:result[k] for k in ['status','role','scientific_updates','qualification_updates']}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();analyze(a.study.resolve(),a.output.resolve())

if __name__=='__main__':main()
