#!/usr/bin/env python
"""Preserve content-addressed copies of the source hashes actually recorded by runs."""
import argparse
import hashlib
import json
from pathlib import Path
from model_rg.provenance import sha256, write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True);args=ap.parse_args()
    root=Path(args.data_root);repo=Path(__file__).resolve().parents[1]
    objects=root/'provenance/source-objects';objects.mkdir(parents=True,exist_ok=True)
    byhash={}
    for base in [repo/'src',repo/'scripts',repo/'internal']:
        for p in base.rglob('*.py'):
            data=p.read_bytes();byhash[hashlib.sha256(data).hexdigest()]=data
    # Exact reconstruction of the small pilot's pre-offset CLI, retained as an internal source object.
    p=repo/'scripts/measure_response.py';data=p.read_text()
    data=data.replace('    ap.add_argument("--offset", type=int, default=512)\n','').replace('[args.offset:args.offset+args.documents, :args.length]','[512:512+args.documents, :args.length]').encode()
    byhash[hashlib.sha256(data).hexdigest()]=data
    records=[];missing=[]
    paths=sorted((root/'executed').glob('*/manifest.json'))+sorted((root/'executed').glob('*/result.json'))+sorted((root/'analysis').glob('*/analysis-manifest.json'))
    for p in paths:
        meta=json.loads(p.read_text())
        for name,digest in meta.get('source_files',{}).items():
            target=objects/(digest+'.py')
            if not target.exists():
                if digest not in byhash:
                    missing.append({'record':str(p.relative_to(root)),'source':name,'sha256':digest});continue
                target.write_bytes(byhash[digest])
            if sha256(target)!=digest:raise RuntimeError('Corrupt source object')
            records.append({'record':str(p.relative_to(root)),'source':name,'sha256':digest,'object':str(target.relative_to(root))})
    write_json(root/'provenance/source-index.json',{'source_bindings':records,'missing':missing})
    if missing:raise RuntimeError(f'{len(missing)} recorded source bindings unavailable; see source-index.json')
    print(f'Archived {len(set(r["sha256"] for r in records))} unique source objects for {len(records)} bindings')


if __name__=='__main__':main()
