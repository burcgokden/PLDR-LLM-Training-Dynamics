#!/usr/bin/env python
"""Independent precision reference at the known critical point, both classes."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import sys
import json
from prepare_lattice_data import compile_sampler,generate
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from model_rg.provenance import sha256,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',required=True);a=ap.parse_args()
    study=Path(a.study);dest=study/'precision-reference';dest.mkdir(exist_ok=False)
    binary=compile_sampler(study)
    jobs=[(q,L) for q in [2,3] for L in [4,6,8,12,16,24,32,48,64]]
    cells=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        tasks={pool.submit(generate,binary,dest/f'q{q}-L{L}',q,L,1.,1790000+j,16,8192):j
               for j,(q,L) in enumerate(jobs)}
        for future in as_completed(tasks):
            cell=future.result();cell['id']=tasks[future];cells.append(cell)
            print('precision',cell['q'],cell['L'],round(cell['runtime_seconds'],1),flush=True)
    write_json(dest/'manifest.json',dict(status='complete',schema='physical-corpus-v1',
        stage='precision-reference',cells=sorted(cells,key=lambda x:x['id']),
        sampler_source_sha256=sha256(REPO/'scripts/potts_mc.cpp'),
        protocol='Both classes, all nine predeclared sizes, 16 independent chains and 8192 configurations per chain. Fixed 12 cluster-update spacing, 4096 burn-in updates. Independent from all training and assessment streams.'))


if __name__=='__main__':main()
