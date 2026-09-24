#!/usr/bin/env python3
"""Check common factorial replay observations and declare recorder-only fields."""
import argparse
import json
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256,write_json
from model_rg.replay_equality import require_replay,finite_state,contract

COMMON={'final_logits','initial_intervened_logits','initial_native_logits','loss','lr','offsets','rows','targets'}
RECORDED={'fields','logbase','power','predictive_step_kl','probe_nll','step_logits','symmetric','tensor_samples','tensor_statistics'}


def verify(study,output):
    if output.exists():raise FileExistsError(output)
    checked={};rows=[]
    for folder in [study/'profile/reference1',study/'runs/early-h2']:
        path=folder/'manifest.json';checked[str(path)]=sha256(path);m=json.loads(path.read_text())
        if m['status']!='complete' or m['role']!='qualification':raise ValueError('Completed qualification required')
        for branch in ['native_keep','replay']:
            path=folder/(branch+'.npz');checked[str(path)]=sha256(path)
            if checked[str(path)]!=m['branches'][branch]['artifact_sha256']:raise ValueError('Changed observations')
        with np.load(folder/'native_keep.npz') as native,np.load(folder/'replay.npz') as replay:
            if set(replay.files)!=COMMON or set(native.files)!=COMMON|RECORDED:raise ValueError('Undeclared observation schema')
            for key in sorted(COMMON):require_replay(native[key],replay[key])
            for key in sorted(RECORDED):
                if not finite_state(native[key]):raise ValueError('Nonfinite recorded observation')
        rows.append(dict(folder=str(folder),common_fields=sorted(COMMON),recorder_only_fields=sorted(RECORDED)))
    write_json(output,dict(status='passed',pairs=rows,checked_sha256=checked,native_updates=0,
        replay_contract=contract(),source_sha256=sha256(__file__),
        scope='Exact bytes of every declared common observation and finiteness of recorder-only fields. Full optimizer-state digests remain separately certified by the native producer.'))
    print('Both factorial observation pairs passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify(a.study.resolve(),a.output.resolve())
