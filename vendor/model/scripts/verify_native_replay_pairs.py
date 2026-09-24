#!/usr/bin/env python3
"""Check retained full native states against the maintained byte contract."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from model_rg.provenance import sha256,write_json
from model_rg.replay_equality import require_replay,require_observations,contract


def verify(pairs,output):
    if output.exists():raise FileExistsError(output)
    rows=[]
    for left,right in pairs:
        start=time.monotonic();checked={}
        states=[]
        for folder in [left,right]:
            for name in ['final-state.pt','observations.npz','manifest.json']:
                checked[str(folder/name)]=sha256(folder/name)
            m=json.loads((folder/'manifest.json').read_text())
            if m['status']!='complete':raise ValueError('A successful retained state is required')
            for name in ['final-state.pt','observations.npz']:
                if m['artifacts'][name]!=checked[str(folder/name)]:raise ValueError('Retained acquisition changed')
            states.append(torch.load(folder/'final-state.pt',map_location='cpu',weights_only=False))
        require_replay(*states)
        with np.load(left/'observations.npz') as a,np.load(right/'observations.npz') as b:
            require_observations(a,b);fields=sorted(a.files)
        del states
        rows.append(dict(left=str(left),right=str(right),status='passed',observation_fields=fields,
                         checked_sha256=checked,seconds=time.monotonic()-start))
        print('Retained pair passed',left.parent.parent.name,flush=True)
    write_json(output,dict(status='passed',pairs=rows,replay_contract=contract(),native_updates=0,scientific_updates=0,
        source_sha256=sha256(__file__),scope='Fresh logical-byte and finite-state check of saved acquisitions; no new producer qualification or native trajectory.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pair',type=Path,nargs=2,action='append',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify(a.pair,a.output)
