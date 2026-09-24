#!/usr/bin/env python3
"""Independent trapezoidal reconstruction of all retained and paired areas."""
from companion_paths import legacy_path
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
ROOT=Path(legacy_path('/pldr-data/model'))

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main(out):
    if out.exists():raise FileExistsError(out)
    start=time.monotonic();records=[];bound={}
    sources=[ROOT/'potential-factorial-20260913/analysis/block-area.json',ROOT/'potential-factorial-disjoint-20260914/analysis/replication-area.json']
    for source in sources:
        summary=json.loads(source.read_text());bound[str(source)]=sha(source)
        for record in summary['records']:
            path=Path(record['raw']);bound[str(path)]=sha(path)
            if bound[str(path)]!=record['raw_sha256']:raise ValueError('Changed native panel')
            kind=record.get('kind','paired');skip=256 if kind=='early' else 0
            with np.load(path,allow_pickle=False) as z:b=z['logbase'][skip:,0].astype(float);p=z['power'][skip:].astype(float)
            db,dp=np.diff(b,axis=0),np.diff(p,axis=0)
            for cell in record['scales']:
                width=cell['width'];shape=(-1,width)+b.shape[1:]
                if (len(b)-1)%width:raise ValueError('Unaligned retained window')
                u=(dp*(b[:-1]+b[1:])/2).reshape(shape).sum(1);v=(db*(p[:-1]+p[1:])/2).reshape(shape).sum(1)
                d=p[width::width]-p[:-width:width];e=b[width::width]-b[:-width:width]
                U=d*(b[width::width]+b[:-width:width])/2;V=e*(p[width::width]+p[:-width:width])/2
                bmid=((b[:-1]+b[1:])/2).reshape(shape)-b[:-width:width,None]
                pmid=((p[:-1]+p[1:])/2).reshape(shape)-p[:-width:width,None]
                area=(bmid*dp.reshape(shape)-pmid*db.reshape(shape)).sum(1)/2
                residual=max(float(np.max(np.abs(u-U-area))),float(np.max(np.abs(v-V+area))))
                np.testing.assert_allclose(u,U+area,rtol=5e-9,atol=5e-13);np.testing.assert_allclose(v,V-area,rtol=5e-9,atol=5e-13)
                total=p[width::width]*b[width::width]-p[:-width:width]*b[:-width:width]
                ratio=float(np.linalg.norm(area)/np.linalg.norm(total))
                np.testing.assert_allclose(ratio,cell['area_relative_increment_norm'],rtol=5e-9,atol=5e-13)
                records.append(dict(path=str(path),kind=kind,width=width,relative_area_norm=ratio,absolute_identity_residual=residual))
    if len(records)!=326:raise ValueError('Incomplete area inventory')
    result=dict(status='passed',raw_paths=38,scale_cells=326,retained_cells=198,paired_cells=128,records=records,
        maximum_absolute_identity_residual=max(c['absolute_identity_residual'] for c in records),checked_sha256=bound,verifier_sha256=sha(__file__),seconds=time.monotonic()-start,
        scope='Independent trapezoidal area expression, fixed first-probe panels, every declared temporal scale. No native-size exponent or predictive closure inferred.')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k not in ['records','checked_sha256']})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)
