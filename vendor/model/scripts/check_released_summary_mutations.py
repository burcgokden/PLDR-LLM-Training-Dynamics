"""Require independent reconstruction to reject perturbed reported aggregates."""
import argparse,importlib.util,json,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from model_rg.provenance import sha256,write_json

def main(a):
    records=[]
    for family in ['spin','language','generation']:
        p=REPO/'scripts/reconstruct_released_independent.py';spec=importlib.util.spec_from_file_location('raw_'+family,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        m.STUDY=Path(a.study);m.OUT=Path(a.output).parent/('unexpected-'+family+'.json');original=m.read
        def read(path):
            d=original(path)
            if path==m.STUDY/'analysis'/(family+'.json'):
                if family=='spin':d['summary'][0]['nll']+=.1
                elif family=='language':d['summary'][0]['nll']+=.1
                else:d['cells'][0]['binder']+=.1
            return d
        m.read=read
        try:m.main()
        except (AssertionError,ValueError):records.append(dict(family=family,rejected=True))
        else:raise ValueError('Altered aggregate escaped independent reconstruction: '+family)
    write_json(a.output,dict(status='passed',mutations=records,checker_sha256=sha256(__file__),reducer_sha256=sha256(REPO/'scripts/reconstruct_released_independent.py')))
    print(records)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--output',required=True);main(p.parse_args())
