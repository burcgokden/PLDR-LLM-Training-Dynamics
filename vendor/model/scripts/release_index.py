#!/usr/bin/env python3
"""Build, inspect or execute the bounded reconstruction routes of the release index."""
from companion_paths import child_pythonpath
from companion_paths import configured_path
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from model_rg.provenance import sha256,write_json
from numerical_validation import load_json_strict

REPO=Path(__file__).resolve().parents[1]
DATA=Path(configured_path('data:model'))


def generate(output):
    spec=load_json_strict((REPO/'docs/release-routes.json').read_text())
    entries=[]
    for route in spec['routes']:
        sources={}
        for command in route['commands']:
            script=command[1].replace('{repo}/','')
            sources[script]=sha256(REPO/script)
        protocols={}
        for member in route['protocols']:
            path=DATA/member;p=load_json_strict(path.read_text())
            protocols[str(path)]=dict(sha256=sha256(path),status=p.get('status'),
                executed_source_sha256=p.get('source_sha256',p.get('producer_sources',p.get('sources',{}))),
                bound_input_sha256=p.get('input_sha256',p.get('inputs_sha256',{})))
        for doc in route['documentation']:
            if not (REPO/'docs'/doc).is_file():raise ValueError('Missing guide')
        entries.append(dict(route,current_entrypoint_sha256=sources,protocol_identity=protocols,
            executable=route.get('executable',True),
            dependency_note='Native reconstruction requires the recorded local raw arrays and every input demanded by its validator. Full acquisition also needs parent optimizer/RNG states, source/tokenizer/native assets and matching packages; inference weights alone are insufficient.',
            public_raw_data_retrieval=None))
    toolchain=dict(python=platform.python_version(),packages={n:importlib.metadata.version(n) for n in ['numpy','torch','transformers','safetensors','pytest']},
        lean=(REPO/'lean-toolchain').read_text().strip(),lake_manifest_sha256=sha256(REPO/'lake-manifest.json'),
        observed_requirements_sha256=sha256(REPO/'requirements-observed.txt'))
    out=dict(schema='modelrg-release-index-v1',status='complete',routes=entries,toolchain=toolchain,
        scope='Source identity, scientific completion, reconstruction qualification and public availability are distinct. Retained qualification is not labeled as a fresh execution.',
        source_sha256={'scripts/release_index.py':sha256(__file__),'docs/release-routes.json':sha256(REPO/'docs/release-routes.json')},
        deferred_status_sha256=sha256(REPO/'docs/confirmation-status.json'))
    write_json(output,out)
    return out


def load_current(path):
    index=load_json_strict(path.read_text())
    if index['schema']!='modelrg-release-index-v1':raise ValueError('Foreign release index')
    for n,h in index['source_sha256'].items():
        if sha256(REPO/n)!=h:raise ValueError('Changed release-index source')
    specification=load_json_strict((REPO/'docs/release-routes.json').read_text())['routes']
    expected={r['id']:r for r in specification}
    if len(index['routes'])!=len(expected) or {r['id'] for r in index['routes']}!=set(expected):raise ValueError('Incomplete route census')
    for route in index['routes']:
        fixed=expected[route['id']]
        if any(route.get(k)!=v for k,v in fixed.items()) or route['executable']!=fixed.get('executable',True):raise ValueError('Changed command or route policy')
        scripts={c[1].replace('{repo}/','') for c in fixed['commands']}
        if set(route['current_entrypoint_sha256'])!=scripts or set(route['protocol_identity'])!={str(DATA/n) for n in fixed['protocols']}:raise ValueError('Incomplete source or protocol identity')
        for n,h in route['current_entrypoint_sha256'].items():
            if sha256(REPO/n)!=h:raise ValueError('Changed reconstruction entrypoint')
        for n,p in route['protocol_identity'].items():
            if Path(n).exists() and sha256(n)!=p['sha256']:raise ValueError('Changed scientific protocol')
    if sha256(REPO/'docs/confirmation-status.json')!=index['deferred_status_sha256']:raise ValueError('Changed deferred-route status')
    return index


def execute(index,route_id,output):
    matches=[r for r in index['routes'] if r['id']==route_id]
    if len(matches)!=1:raise ValueError('Unknown route')
    route=matches[0]
    if not route['executable'] or not route['commands']:raise ValueError('Unqualified route; no execution permitted')
    for name,identity in route.get('protocol_identity',{}).items():
        if not Path(name).is_file():raise FileNotFoundError('Required external protocol: '+name)
        if sha256(name)!=identity['sha256']:raise ValueError('Changed selected scientific protocol')
    output=output.resolve()
    if output.exists():raise FileExistsError(output)
    values=dict(python=sys.executable,repo=str(REPO),data=str(DATA),output=str(output))
    commands=[[arg.format(**values) for arg in command] for command in route['commands']]
    output.mkdir(parents=True)
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
    records=[]
    for i,command in enumerate(commands):
        with (output/f'command-{i}.log').open('x') as log:
            result=subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
        records.append(dict(command=command,returncode=result.returncode,log_sha256=sha256(output/f'command-{i}.log')))
        if result.returncode:
            write_json(output/'execution.json',dict(status='failed',route=route_id,commands=records))
            raise RuntimeError('Reconstruction failed; retained logs identify the failure')
    write_json(output/'execution.json',dict(status='passed',route=route_id,commands=records,new_native_updates=0,new_native_forwards=0))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['generate','check','run']);p.add_argument('--index',type=Path,default=REPO/'docs/RELEASE_INDEX.json')
    p.add_argument('--route');p.add_argument('--output',type=Path);a=p.parse_args()
    if a.action=='generate':result=generate(a.index);print({'routes':len(result['routes'])})
    else:
        result=load_current(a.index)
        if a.action=='check':print({'status':'passed','routes':len(result['routes']),'unqualified':[r['id'] for r in result['routes'] if not r['executable']],'missing_external_protocols':{r['id']:[n for n in r['protocol_identity'] if not Path(n).is_file()] for r in result['routes'] if any(not Path(n).is_file() for n in r['protocol_identity'])}})
        else:
            if a.route is None or a.output is None:p.error('--route and --output are required')
            execute(result,a.route,a.output)
