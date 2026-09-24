#!/usr/bin/env python
"""Check exact document separation from every retained text-evaluation cohort."""
import argparse
import json
from pathlib import Path

from model_rg.provenance import sha256,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-feasible-20260908');a=p.parse_args()
    root=Path(a.root).resolve();study=root/a.study;out=study/'verification/onepass-cohort-separation.json'
    if out.exists():raise FileExistsError(out)
    checked={}
    def records(folder,count):
        path=folder/'manifest.json';checked[str(path)]=sha256(path);meta=json.loads(path.read_text())
        path=folder/'records.json';checked[str(path)]=sha256(path)
        if checked[str(path)]!=meta['records_sha256']:raise AssertionError('A bound source-record table changed')
        rows=json.loads(path.read_text())
        if len(rows)!=count:raise AssertionError('A selected document cohort changed size')
        return rows
    fresh=records(root/'data/refinedweb-onepass-524288',524288)
    hashes={r['content_sha256'] for r in fresh}
    if len(hashes)!=524288:raise AssertionError('The training corpus contains an exact duplicate document')
    panels=[]
    for name,folder,count in [('original',root/'data/refinedweb-4608',4608),
            ('short',root/'controlled-study-20260905/data/short',2048),
            ('long',root/'controlled-study-20260905/data/long',1024)]:
        old=records(folder,count);overlap=[i for i,r in enumerate(old) if r['content_sha256'] in hashes]
        if overlap:raise AssertionError('Exact training/evaluation document overlap: '+name+' '+str(overlap))
        panels.append(dict(cohort=name,documents=count,overlapping_rows=overlap,exact_document_overlap=0))
    write_json(out,dict(status='passed',training_documents=524288,panels=panels,
        checked_sha256=checked,verifier_sha256=sha256(__file__),
        scope='Exact document-content hash separation of the complete new training corpus from all '
        'three retained RefinedWeb observation cohorts. The separately checked record tables bind '
        'their actual source documents. This does not assert semantic deduplication, absence of '
        'naturally repeated phrases, benchmark contamination freedom, or separation from the '
        'released models\' historical pretraining.'))
    print('No exact training-document overlap with 4,608 original, 2,048 short or 1,024 long documents',flush=True)


if __name__=='__main__':main()
