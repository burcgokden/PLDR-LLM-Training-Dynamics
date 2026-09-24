#!/usr/bin/env python3
"""Record access class and actual identities for the native factorial inputs."""
from companion_paths import legacy_path
import json
from pathlib import Path
from model_rg.provenance import sha256,write_json
REPO=Path(__file__).resolve().parents[1]
ROOT=Path(legacy_path('/pldr-data/model'))

def main():
    entries={}
    def add(path,category,access='locally retained; no public deposit claimed'):
        path=Path(path).resolve();entries[str(path)]=dict(path=str(path),category=category,access=access,bytes=path.stat().st_size,sha256=sha256(path))
    for folder in ['potential-factorial-20260913','potential-factorial-disjoint-20260914']:
        study=ROOT/folder;spec=json.loads((study/'protocol.json').read_text())
        for name in ['protocol.json','selection.npz']:add(study/name,'Protocol or source/evaluation selection')
        for case in spec['cases']:
            add(case['checkpoint'],'Complete model/Adam/scheduler incoming state')
            add(case['parent_manifest'],'Incoming-state manifest')
            add(Path(case['checkpoint']).parent/('activity.npz' if case['kind']=='early' else 'sampling.npz'),'Ordered consumed-source history')
        for path in (study/'runs').rglob('*.npz'):add(path,'Native paired observations and replay')
        for path in (study/'profile').rglob('*.npz'):add(path,'Qualification observations and replay')
        for f in spec['native_sources']:add(f,'Pinned native code','public upstream; exact local content pinned by SHA-256')
    corpus=ROOT/'data/refinedweb-onepass-524288'
    for n in ['tokens.npy','manifest.json','records.json']:add(corpus/n,'Tokenized single-pass corpus and document identities')
    matched=ROOT/'matched-onepass-20260914'
    for n in ['protocol.json','selection.npz','qualification/verification.json']:
        add(matched/n,'Matched one-pass protocol, source selection or qualification')
    for path in (matched/'runs').rglob('*'):
        if path.is_file() and path.name in ['manifest.json','observations.npz','final-state.pt']:
            add(path,'Matched single-pass scientific observations and complete trained states')
    result=dict(schema='native-study-asset-access-v2',assets=list(entries.values()),
        bundled=['Standalone LaTeX, figures, compact numerical evidence','Selected Lean modules, pinned dependency lockfiles and exact statement registry','Current native producers, analyzers, independent verifiers and executed producer snapshot','Formal, numerical, admission and reconstruction records'],
        public_upstream=[dict(url='https://huggingface.co/fromthesky/PLDR-LLM-v51-SOC-110M-1',role='Native architecture and released inference assets; not the incoming training/optimizer states'),dict(url='https://huggingface.co/datasets/tiiuae/falcon-refinedweb',role='Source corpus distribution; exact tokenization and selected local bytes remain separately identified')],
        access_limit='No durable public archive is claimed for complete incoming training states, local token arrays, source histories or native branch arrays. Complete scientific reruns require these identified local assets. The manuscript itself builds from the source bundle.',
        restriction='Read-only source corpus at /pldr-assets/refinedweb. Exact historical pretraining streams of released checkpoints are not supplied by public inference weights.',
        inventory_sha256=sha256(__file__))
    write_json(REPO/'docs/ASSET_ACCESS.json',result)
    print(len(entries),'identified native inputs')
if __name__=='__main__':main()
