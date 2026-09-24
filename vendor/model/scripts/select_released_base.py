"""Select a released checkpoint using matched validation scores only."""
import argparse,json,sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from model_rg.provenance import sha256,write_json
REVISIONS={1:'7a34e2ca9aa78038683677cfda17fe3a9fe6da8a',4:'c377be06f2294aeccb56d19ea5f52640b1850daf',5:'de8e539c0ba1829072f4b8c2c5fae3bde0a3a2d2'}
def main(a):
    s=Path(a.study);rows=[]
    for index in [1,4,5]:
        p=s/f'base-qualification/model{index}/verification.json';d=json.loads(p.read_text())
        if d['status']!='passed':raise ValueError('All candidates require qualification')
        rows.append(dict(model=index,language_nll=float(np.mean([r['nll'] for r in d['language']])),language_accuracy=float(np.mean([r['accuracy'] for r in d['language']])),qualification_sha256=sha256(p)))
    selected=min(rows,key=lambda r:r['language_nll'])['model']
    value=dict(selected=selected,revision=REVISIONS[selected],candidates=rows,criterion='Minimum equal-domain proper-prefix validation NLL among released models 1,4,5. Test data unused.',producer_sha256=sha256(__file__))
    dest=s/'base-selection.json'
    if dest.exists():
        existing=json.loads(dest.read_text())
        if existing['selected']!=selected:raise ValueError('Recorded choice differs from validation reduction')
        print('Existing choice verified:',selected);return
    write_json(dest,value);print('Selected:',selected)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);main(p.parse_args())
