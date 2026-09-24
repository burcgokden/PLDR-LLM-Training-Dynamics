#!/usr/bin/env python3
"""Separate completed native row continuations from internal optimizer replay."""
from companion_paths import legacy_path
import argparse
from pathlib import Path
from model_rg.provenance import sha256,write_json
from numerical_validation import load_json_strict
ROOT=Path(legacy_path('/pldr-data/model'))


def check():
    checked={};records=[]
    def bind(path,expected=None):
        path=Path(path).resolve();key=str(path)
        if key not in checked:checked[key]=sha256(path)
        if expected is not None and expected!=checked[key]:raise ValueError('Changed execution member '+key)
        return checked[key]
    for name,key,updates,calls in [('collective-row-clock-20260916','scientific_updates',8,40),('collective-row-quadrature-20260916','optimizer_replay_updates',8,272)]:
        study=ROOT/name;p=load_json_strict((study/'protocol.json').read_text());ph=bind(study/'protocol.json')
        if p[key]!=updates or p['native_forwards']!=calls or len(p['jobs'])!=8:raise ValueError('Wrong execution design')
        for member,h in p['source_sha256'].items():bind(study/'executed-source'/member,h)
        for member,h in p['input_sha256'].items():bind(member,h)
        counted_updates=0;counted_calls=0;seconds=0.;peak=0
        for j in p['jobs']:
            folder=study/'runs'/j['run_id'];bind(folder/'manifest.json');m=load_json_strict((folder/'manifest.json').read_text())
            if m['status']!='complete' or m['job']!=j or m['protocol_sha256']!=ph:raise ValueError('Incomplete acquired job')
            if key=='scientific_updates':
                if m['incoming_row_replay_error']!=0 or m['endpoint_restored'] is not True:raise ValueError('Changed incoming/outgoing states')
            elif m['transport_replay_error']!=0:raise ValueError('Diagnostic replay differs')
            for member,h in m['artifacts'].items():bind(folder/member,h)
            counted_updates+=m[key];counted_calls+=m['native_forwards'];seconds+=m['elapsed_seconds'];peak=max(peak,m['peak_allocated_bytes'])
        if counted_updates!=updates or counted_calls!=calls:raise ValueError('Accounting differs')
        records.append(dict(study=str(study),role='native_scientific' if key=='scientific_updates' else 'internal_diagnostic_replay',paths=8,scientific_updates=updates if key=='scientific_updates' else 0,optimizer_replay_updates=updates if key!='scientific_updates' else 0,native_forward_calls=calls,worker_seconds=seconds,peak_allocated_bytes=peak,protocol_sha256=ph))
    return dict(status='passed',schema='collective-row-execution-v1',scientific_updates=8,optimizer_replay_updates=8,native_forward_calls=312,records=records,checked_sha256=checked,checker_sha256=sha256(__file__),scope='Complete acquisition and accounting only. Internal derivative replay does not provide a successful approximation or a new scientific trajectory.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    r=check();write_json(a.output,r);print({k:r[k] for k in ['status','scientific_updates','optimizer_replay_updates','native_forward_calls']})
