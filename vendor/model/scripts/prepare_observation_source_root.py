#!/usr/bin/env python3
"""Assemble a small, exact source root for acquired observations and a current reducer.

Only named source files with their declared digests are copied. Conflicting
producer/reducer identities fail; no checkpoint or observation payload is copied.
"""
import argparse
from pathlib import Path
from model_rg.provenance import sha256,write_json
from numerical_validation import load_json_strict

REPO=Path(__file__).resolve().parents[1]


def prepare(protocol,analysis,snapshot,output):
    if output.exists():raise FileExistsError(output)
    expected={}
    for path in [protocol,analysis]:
        record=load_json_strict(path.read_text())
        for name,digest in record['source_sha256'].items():
            if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Source name must be relative')
            if name in expected and expected[name]!=digest:raise ValueError('Conflicting source identities')
            expected[name]=digest
    selected={}
    for name,digest in expected.items():
        candidates=[snapshot/name,REPO/name]
        matches=[p for p in candidates if p.is_file() and sha256(p)==digest]
        if not matches:raise ValueError('Exact source bytes unavailable: '+name)
        selected[name]=matches[0]
    output.mkdir(parents=True)
    for name,path in selected.items():
        target=output/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
        if sha256(target)!=expected[name]:raise ValueError('Copied source identity differs')
    write_json(output/'source-bridge.json',dict(status='passed',producer_protocol_sha256=sha256(protocol),analysis_sha256=sha256(analysis),
        helper_sha256=sha256(__file__),source_sha256=expected,resolved_from={n:str(p) for n,p in selected.items()},
        scope='Small exact producer/reducer source bridge; no raw data, model, or optimizer-state copy.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['protocol','analysis','snapshot','output']:p.add_argument('--'+name,type=Path,required=True)
    prepare(**vars(p.parse_args()))
