#!/usr/bin/env python
"""Independent array-level check of the native null-sector prediction."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from model_rg.provenance import sha256,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);a=p.parse_args();study=Path(a.study).resolve()
    checked={};rows=[];coordinates=0
    def check(path,digest=None):
        value=sha256(path)
        if digest is not None and value!=digest:raise ValueError('Changed artifact: '+str(path))
        checked[str(path)]=value
    spec=json.loads((study/'protocol.json').read_text());check(study/'protocol.json')
    for path,digest in {**spec['inputs'],**spec['sources']}.items():check(path,digest)
    with np.load(study.parents[1]/'outer-transfer-20260911/data/panels.npz') as f:targets=f['evaluation'][:,64]
    for case in spec['cases']:
        folder=study/case['name'];meta=json.loads((folder/'results.json').read_text());check(folder/'results.json')
        if meta['status']!='complete' or meta['protocol_sha256']!=sha256(study/'protocol.json'):raise ValueError('Incomplete native result')
        data={};records={}
        for r in meta['records']:
            check(r['path'],r['sha256']);records[r['source'],r['arm']]=r
            with np.load(r['path']) as f:data[r['source'],r['arm']]={k:f[k] for k in f.files}
            x=data[r['source'],r['arm']];coordinates+=x['logits'].size
            lp=x['logits']-logsumexp(x['logits'],axis=-1,keepdims=True)
            nll=-np.take_along_axis(lp,targets[None,:,None],axis=-1)[...,0]
            if not np.allclose(nll,x['nll'],atol=1e-12,rtol=0):raise ValueError('Risk reduction differs')
            if not np.array_equal(x['embedding_rows'],x['predicted_rows']):raise ValueError('Pre-update row forecast differs')
            # Match the binary32 multiplication with its scalar cast independently.
            factor=np.float32(r['decay_factor'])
            for t in range(8):
                if not np.array_equal(x['embedding_rows'][t]*factor,x['embedding_rows'][t+1]):raise ValueError('Native multiplication differs')
        for source in range(2):
            x=data[source,'zero'];y=data[source,'pulse']
            if any(not np.array_equal(x[k],y[k]) for k in ['logits','q','losses']):raise ValueError('Non-null emission')
            if records[source,'zero']['unaffected_digests']!=records[source,'pulse']['unaffected_digests']:raise ValueError('Other native state differs')
            require_delta=float(np.linalg.norm(y['embedding_rows'][0]-x['embedding_rows'][0]))
            if require_delta<=0:raise ValueError('Pulse vanished')
            rows.append(dict(case=case['name'],source=source,decay_factor=records[source,'zero']['decay_factor'],
                initial_displacement_norm=require_delta,final_displacement_norm=float(np.linalg.norm(y['embedding_rows'][-1]-x['embedding_rows'][-1])),
                maximum_logit_difference=float(np.max(abs(x['logits']-y['logits']))),bitwise_unaffected=True,bitwise_row_forecast=True))
        if any(not np.array_equal(data[0,'zero'][k],data[0,'replay'][k]) for k in data[0,'zero']):raise ValueError('Replay differs')
    write_json(study/'verification.json',dict(status='passed',schema='native-invisible-sector-verification-v1',rows=rows,
        native_updates=64,replay_updates=16,vocabulary_coordinates=coordinates,checked_sha256=checked,verifier_sha256=sha256(__file__),
        scope='All saved vocabulary arrays, risk reductions and pre-update native row forecasts. Other parameter/moment equality uses producer digests, not independently saved complete tensor states. Finite declared input support only.'))
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
