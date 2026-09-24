#!/usr/bin/env python
"""Independently retokenize every selected source document from read-only Arrow."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import sentencepiece as spm
from model_rg.provenance import sha256,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--study',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.root).resolve();study=Path(a.study).resolve();out=Path(a.output).resolve()
    if out.exists():raise FileExistsError(out)
    begin=time.time();data=study/'data';spec=json.loads((data/'protocol.json').read_text());manifest=json.loads((data/'manifest.json').read_text())
    checked={str(data/'protocol.json'):sha256(data/'protocol.json'),str(data/'manifest.json'):sha256(data/'manifest.json')}
    for name,digest in manifest['files'].items():
        assert sha256(data/name)==digest;checked[str(data/name)]=digest
    for name,digest in spec['sources'].items():assert sha256(name)==digest;checked[name]=digest
    corpus=[np.load(data/f'corpus-{c}.npy',mmap_mode='r') for c in range(2)]
    records=[json.loads((data/f'records-{c}.json').read_text()) for c in range(2)]
    panel_records=json.loads((data/'panel-records.json').read_text())
    with np.load(data/'panels.npz') as f:panels={k:f[k] for k in f.files}
    tokenizer=root/'assets/PLDR-LLM-v51-SOC-110M-1/tokenizer.model'
    assert sha256(tokenizer)==spec['tokenizer_sha256'];checked[str(tokenizer)]=spec['tokenizer_sha256']
    tok=spm.SentencePieceProcessor(model_file=str(tokenizer));count=0;tokens_checked=0
    selections=[(r,corpus[c][i]) for c in range(2) for i,r in enumerate(records[c])]
    selections += [(r,panels[key][i]) for key in ['evaluation','donors'] for i,r in enumerate(panel_records[key])]
    for s,entry in enumerate(spec['strata']):
        path=Path(entry['path']);assert sha256(path)==entry['sha256'];checked[str(path)]=entry['sha256']
        with pa.memory_map(str(path),'r') as mm:table=pa.ipc.open_stream(mm).read_all()
        chosen=[x for x in selections if x[0]['stratum']==s]
        for start in range(0,len(chosen),512):
            chunk=chosen[start:start+512];contents=table.column('content').take(pa.array([r['row'] for r,_ in chunk])).to_pylist()
            encoded=tok.encode(contents,num_threads=4)
            for (r,stored),text,ids in zip(chunk,contents,encoded,strict=True):
                assert r['shard']==path.name and r['content_sha256']==hashlib.sha256(text.encode()).hexdigest()
                assert r['document_tokens']==len(ids)>=513
                offset=r['token_offset'];assert 0<=offset<=len(ids)-513
                assert np.array_equal(stored,np.array(ids[offset:offset+len(stored)],dtype=np.int32))
                count+=1;tokens_checked+=len(stored)
        print('verified source stratum',s,'documents',count,flush=True)
    old={r['content_sha256'] for r in json.loads((root/'data/refinedweb-4608/records.json').read_text())}
    original={r['content_sha256'] for r in json.loads((root/'data/refinedweb-onepass-524288/records.json').read_text())}
    for rows in records:
        content={r['content_sha256'] for r in rows};assert len(content)==65536 and not content&old
    for rows in panel_records.values():assert not {r['content_sha256'] for r in rows}&(old|original)
    assert count==131136 and tokens_checked==67244096
    write_json(out,dict(status='passed',schema='outer-transfer-source-data-verification-v1',document_crops=count,
        token_coordinates=tokens_checked,maximum_token_difference=0,source_document_exclusion=True,
        checked_sha256=checked,verifier_sha256=sha256(__file__),seconds=time.time()-begin,
        scope='Every retained training and observation token reconstructed from its recorded read-only Arrow document and SentencePiece crop. Exact-content exclusion verified; semantic overlap is not asserted absent.'))


if __name__=='__main__':main()
