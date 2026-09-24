#!/usr/bin/env python
"""Render and check the single source of implementation status; reject unready stages."""
import argparse
import json
from pathlib import Path
from model_rg.provenance import sha256
from model_rg.qualification import validate_qualification, need, REQUIRED, TERMINALS


def bindings(value):
    if isinstance(value,dict):
        if set(['path','sha256']) <= set(value):
            yield value
        else:
            for key in sorted(value):yield from bindings(value[key])
    elif isinstance(value,list):
        for child in value:yield from bindings(child)


def validate(status):
    need(status.get('schema') == 'rg-confirmation-status-v3', 'Unsupported status schema')
    stages=status.get('stages', [])
    need(len(stages)==4 and {s['id'] for s in stages}=={'P1','P2','P3','P4'}, 'Incomplete stage status')
    repo=Path(__file__).resolve().parents[1]
    for stage in stages:
        for record in bindings(stage.get('historical_evidence', {})):
            need(sha256(record['path'])==record['sha256'], 'Historical evidence changed: '+record['path'])
        qualification=stage.get('executable_qualification')
        if qualification is not None:
            current=validate_qualification(qualification['protocol'],repo,stage['id'])
            need(current.record()==qualification, 'Current qualification changed')
        if stage['id'] in ['P2','P4']:
            need(qualification is None, 'Unimplemented stage cannot be qualified')


def render(status):
    lines=['# Native execution status', '',
        'Historical scientific completion and qualification of the current executable are separate records.', '',
        '| Stage | Historical scientific updates | Current executable |',
        '|---|---:|---|']
    for stage in status['stages']:
        current='qualified' if stage.get('executable_qualification') else 'not qualified'
        lines.append(f"| {stage['id']}: {stage['name']} | {stage.get('executed_scientific_updates',0):,} | {current} |")
    lines += ['', status['launch_rule'], '', status['publication_rule'], '']
    for stage in status['stages']:
        lines += [stage['id']+': '+stage.get('limits',''), '']
    return '\n'.join(lines)


def require_qualification(producer,verifier,protocol,evidence):
    producer,verifier,protocol,evidence=map(lambda p:Path(p).resolve(),[producer,verifier,protocol,evidence])
    repo=Path(__file__).resolve().parents[1]
    record=validate_qualification(protocol,repo)
    need(producer==repo/REQUIRED[record.stage][0], 'Unrelated producer identity')
    need(verifier==repo/TERMINALS[record.stage][0][1], 'Unrelated terminal identity')
    need(str(evidence) in record.terminal, 'Unrelated terminal evidence')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['render','check','require-ready','require-qualification'])
    p.add_argument('--status',default='docs/confirmation-status.json');p.add_argument('--output',default='docs/CONFIRMATION_STATUS.md')
    p.add_argument('--stage')
    for name in ['producer','verifier','protocol','evidence']:p.add_argument('--'+name)
    a=p.parse_args()
    if a.action=='require-qualification':
        need(all(getattr(a,key) for key in ['producer','verifier','protocol','evidence']), 'Missing qualification arguments')
        require_qualification(a.producer,a.verifier,a.protocol,a.evidence)
        print('Stage-specific terminal qualification passed',flush=True);return
    status=json.loads(Path(a.status).read_text())
    validate(status);content=render(status)
    if a.action=='render':Path(a.output).write_text(content)
    elif a.action=='check':need(Path(a.output).read_text()==content,'Status rendering is stale')
    else:
        stage=next(s for s in status['stages'] if s['id']==a.stage)
        if not stage.get('executable_qualification'):
            raise SystemExit('Stage '+stage['id']+' has no current complete native qualification; execution refused.')
        validate_qualification(stage['executable_qualification']['protocol'],Path(__file__).resolve().parents[1],stage['id'])
        print(REQUIRED[stage['id']][0])
    print('Status bindings and consistency passed',flush=True)

if __name__=='__main__':main()
