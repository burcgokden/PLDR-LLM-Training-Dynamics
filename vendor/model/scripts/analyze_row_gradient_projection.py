#!/usr/bin/env python
"""Summarize the complete paired source-gradient fidelity measurements."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    study=Path(args.root)/'criticality-dynamics-20260906';protocol=study/'protocols/row-gradient-projection.json';ledger=study/'row-gradient-projection.json'
    spec=json.loads(protocol.read_text());execution=json.loads(ledger.read_text())
    if execution['status']!='complete' or len(execution['records'])!=48 or any(r['returncode'] for r in execution['records']):raise AssertionError('Incomplete gradient reduction design')
    if sha256(protocol)!=execution['protocol_sha256']:raise AssertionError('Gradient reduction protocol changed')
    inputs=[protocol,ledger];cases=[];groups=defaultdict(list)
    for case in spec['cases']:
        folder=study/'analysis'/case['name'];meta=json.loads((folder/'manifest.json').read_text())
        if meta['status']!='complete':raise AssertionError('Incomplete gradient reduction case')
        for name,key in [('results.json','results_sha256'),('measurements.npz','raw_sha256')]:
            if sha256(folder/name)!=meta[key]:raise AssertionError('Changed gradient reduction input')
        inputs.extend([folder/'manifest.json',folder/'results.json',folder/'measurements.npz'])
        result=json.loads((folder/'results.json').read_text())
        if result['case']!=case or [r['variant'] for r in result['records']]!=spec['variants']:raise AssertionError('Gradient reduction case or variants changed')
        cases.append(result)
        for record in result['records']:
            groups[case['heads'],case['step'],record['variant']].append((case,record))
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);bind_run(out,inputs,vars(args));summaries=[]
    for (heads,step,variant),members in sorted(groups.items()):
        members.sort(key=lambda x:(x[0]['seed'],x[0]['batch_step']))
        expected={(s,b) for s in range(640101,640105) for b in [2048,8192]}
        if len(members)!=8 or {(c['seed'],c['batch_step']) for c,_ in members}!=expected:raise AssertionError('Unbalanced reduction sources')
        records=[dict(seed=c['seed'],batch_step=c['batch_step'],**r) for c,r in members]
        summaries.append(dict(heads=heads,step=step,variant=variant,records=records,
            maximum_mean_kl=max(r['mean_kl'] for _,r in members),maximum_context_kl=max(r['maximum_kl'] for _,r in members),
            maximum_unclipped_relative_error=max(r['unclipped']['relative_error'] for _,r in members),
            maximum_clipped_relative_error=max(r['clipped']['relative_error'] for _,r in members),
            median_clipped_relative_error=float(np.median([r['clipped']['relative_error'] for _,r in members])),
            minimum_clipped_cosine=min(r['clipped']['cosine'] for _,r in members),
            maximum_clipped_absolute_error=max(r['clipped']['squared_difference']**.5 for _,r in members),
            minimum_clipped_reference_norm=min(r['clipped']['reference_squared_norm']**.5 for _,r in members)))
    write_json(out/'results.json',dict(schema='row-gradient-projection-summary-v1',cases=cases,groups=summaries,interpretation=spec['interpretation']))
    write_json(out/'manifest.json',dict(status='complete',binding_sha256=sha256(out/'binding.json'),results_sha256=sha256(out/'results.json')))
    print('Summarized 48 matched-gradient cases and 24 width-time-variant groups',flush=True)


if __name__=='__main__':main()
