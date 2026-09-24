"""Verify or acquire all pinned released-model assets, including model 1."""
import argparse,json,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from model_rg.provenance import sha256
from model_rg.released_qualification import ROOT

def main(a):
    from huggingface_hub import hf_hub_download
    pins=json.loads((REPO/'docs/templates/released-pinned-assets.json').read_text());count=0
    for name,record in pins.items():
        folder=ROOT/'assets'/name
        for filename,h in record['files'].items():
            p=folder/filename
            if not p.exists() or sha256(p)!=h:
                if not a.download:raise ValueError('Missing or changed pinned asset: '+str(p))
                hf_hub_download(record['repository'],filename,revision=record['revision'],local_dir=folder,force_download=True)
            if sha256(p)!=h:raise ValueError('Downloaded identity mismatch: '+str(p))
            count+=1
    print('Verified',count,'pinned assets across',len(pins),'released models')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');main(p.parse_args())
