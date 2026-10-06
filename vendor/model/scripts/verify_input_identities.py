#!/usr/bin/env python
"""Verify logical input identities after explicit directory remapping."""
from companion_paths import configured_path
import argparse,json
from pathlib import Path
from model_rg.provenance import sha256,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',default='docs/response-inputs.json')
    p.add_argument('--data-root',required=True);p.add_argument('--refinedweb-root',default=configured_path('assets:refinedweb'))
    p.add_argument('--include-upstream-shards',action='store_true');p.add_argument('--output',required=True);a=p.parse_args()
    spec=json.loads(Path(a.manifest).read_text());roots={'data':Path(a.data_root),'refinedweb':Path(a.refinedweb_root)};checked={}
    for record in spec['artifacts']:
        if record['root']=='refinedweb' and not a.include_upstream_shards:continue
        root=roots[record['root']].resolve();path=(root/record['relative_path']).resolve()
        if not path.is_relative_to(root):raise ValueError('Input identity escapes its mapped root')
        if not path.is_file() or sha256(path)!=record['sha256']:raise ValueError('Missing or changed input: '+record['role'])
        checked[record['role']]=dict(path=str(path),sha256=record['sha256'])
    write_json(a.output,dict(status='passed',checked=checked,upstream_shards_rehashed=a.include_upstream_shards,
        manifest_sha256=sha256(a.manifest),verifier_sha256=sha256(__file__)))
    print('Verified',len(checked),'logical input identities')

if __name__=='__main__':main()
