#!/usr/bin/env python3
"""Describe exact local inputs and distinguish measured from retained hashes."""
from companion_paths import configured_path
import argparse
import json
from pathlib import Path
from model_rg.provenance import sha256,write_json
ROOT=Path(configured_path('data:model'))
REPO=Path(__file__).resolve().parents[1]


def inventory(output,uncertainty,qualification):
    entries={}
    def add(path,digest,role,basis):
        path=Path(path);key=str(path.resolve())
        if key in entries and entries[key]['sha256']!=digest:raise ValueError('Conflicting asset identity')
        entries[key]=dict(path=key,bytes=path.stat().st_size,sha256=digest,role=role,
                          hash_basis=basis,access='Local maintained research archive; no public download asserted')
    records=[ROOT/'exact-replica-profiles-20260915/analysis.json',ROOT/'categorical-visibility-20260915/analysis.json',
             ROOT/'categorical-visibility-20260915/verification.json']
    records += [uncertainty/label/name for label in ['coarse','refinement'] for name in
                ['joint-analysis.json','prediction-analysis.json','collective-analysis.json','operator-moments.json']]
    freshly_checked={}
    for path in records:
        r=json.loads(path.read_text())
        if r['status'] not in ['passed','complete']:raise ValueError('Incomplete reconstruction')
        for name,digest in r.get('checked_sha256',{}).items():freshly_checked[name]=digest
        add(path,sha256(path),'Completed reconstruction record','Hashed by this inventory')
    for label in ['coarse','refinement']:
        study=ROOT/f'critical-onepass-{label}-20260914';p=json.loads((study/'protocol.json').read_text())
        for name in ['protocol.json','selection.npz']:
            path=study/name;add(path,sha256(path),'Native design and source selection','Hashed by this inventory')
        for name,digest in p['input_sha256'].items():
            add(name,digest,'Frozen corpus, tokenizer/native-source prerequisite','Protocol identity, separately checked by native admission')
        for job in p['jobs']:
            folder=study/'runs'/job['run_id'];mp=folder/'manifest.json';m=json.loads(mp.read_text())
            if m['status']!='complete':raise ValueError('Uncompleted acquisition')
            add(mp,sha256(mp),'Native trajectory manifest','Hashed by this inventory')
            for name,digest in m['artifacts'].items():
                path=folder/name
                if str(path) in freshly_checked:
                    if freshly_checked[str(path)]!=digest:raise ValueError('Reconstruction/acquisition mismatch')
                    basis='Raw bytes checked by the completed reconstruction records'
                else:basis='Retained acquisition manifest; not rehashed by this inventory'
                add(path,digest,'Native observation or complete checkpoint',basis)
    for name,digest in freshly_checked.items():
        if Path(name).is_relative_to(ROOT):add(name,digest,'Reconstruction prerequisite','Raw bytes checked by the completed reconstruction records')
    for route in ['base','shared','matched']:
        p=qualification/route/'protocol.json';r=json.loads(p.read_text())
        add(p,sha256(p),'Fresh source-qualified native protocol','Hashed by this inventory')
        q=qualification/route/'qualification/verification.json';v=json.loads(q.read_text())
        if v['status']!='passed':raise ValueError('Unqualified route')
        add(q,sha256(q),'Logical-byte native replay certificate','Hashed by this inventory')
        for path,digest in v['checked_sha256'].items():add(path,digest,'Fresh native qualification artifact','Bytes checked by executed native qualification')
    native=ROOT/'assets/PLDR-LLM-v51-SOC-110M-1'
    for name in ['tokenizer.model','config.json','configuration_pldrllm.py','modeling_pldrllm.py']:
        path=native/name
        if path.exists():add(path,sha256(path),'Pinned tokenizer or native model definition','Hashed by this inventory')
    records_dir=ROOT/'scheduled-training-feasible-20260908/protocols'
    for path in records_dir.glob('*data-selection.json'):
        add(path,sha256(path),'Corpus regeneration selection and prerequisites','Hashed by this inventory')
    write_json(output,dict(schema='replica-asset-access-v1',assets=list(entries.values()),
        reconstruction_records={str(p):sha256(p) for p in records},inventory_sha256=sha256(__file__),
        public_upstream_references=['https://huggingface.co/fromthesky/PLDR-LLM-v51-SOC-110M-1',
                                    'https://huggingface.co/datasets/tiiuae/falcon-refinedweb'],
        regeneration='Use the frozen expanded and one-pass selection protocols and their enumerated prerequisite hashes with prepare_expanded_refinedweb.py and prepare_onepass_refinedweb.py. Exact master-corpus regeneration requires the recorded local shard strata and tokenizer; arbitrary public downloads need not reproduce them.',
        read_only_source=configured_path('assets:refinedweb'),
        access_boundary='Source ZIP builds the manuscript and contains compact evidence/code. Large local assets require the maintained research archive. No public deposit, DOI, or independently tested external clean-asset reconstruction is claimed.',
        storage_bytes=sum(r['bytes'] for r in entries.values())))
    print('Inventoried',len(entries),'assets',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--uncertainty',type=Path,required=True);p.add_argument('--qualification',type=Path,required=True)
    a=p.parse_args();inventory(a.output,a.uncertainty,a.qualification)
