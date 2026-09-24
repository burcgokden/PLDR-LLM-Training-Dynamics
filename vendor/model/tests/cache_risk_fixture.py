"""Geometry-correct synthetic observations; no native qualification is claimed."""
from pathlib import Path
import numpy as np
from scripts.run_cache_risk_study import arms, SOURCES
from model_rg.provenance import sha256, write_json


def make_fixture(study):
    repo=Path(__file__).resolve().parents[1];study.mkdir()
    jobs=[dict(heads=h,control=g,seed=s,run_id=f'h{h}-g{g}-s{s}',checkpoint_sha256='fixture') for h in [8,24] for g in [0,1.5] for s in range(6)]
    np.savez_compressed(study/'inputs.npz',blocks=np.zeros((128,129),dtype=np.int64))
    p=dict(schema='cache-risk-v1',status='frozen_before_acquisition',arms=arms(),jobs=jobs,contexts=64,
        vocabulary=3,prefix_lengths=[32,64,128],document_hashes=[f'fixture-{i}' for i in range(128)],
        selection_sha256=sha256(study/'inputs.npz'),source_sha256={n:sha256(repo/n) for n in SOURCES},
        memory_ceiling_bytes=1024,parent='synthetic',expected_calls=dict(calibration=576,native_assessment=576,cached_assessment=5376,qualification=336))
    for n in SOURCES:
        dest=study/'executed-source'/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((repo/n).read_bytes())
    write_json(study/'protocol.json',p);ph=sha256(study/'protocol.json');rng=np.random.default_rng(320023)
    for job in jobs:
        out=study/'runs'/job['run_id'];out.mkdir(parents=True);artifacts=[];ca={}
        for a in p['arms']:
            key=f"C{a['calibration_length']}-P{a['panel']}-M{a['size']}"
            ca[key]=np.full((5,3,1,job['heads'],64,64),.1,dtype=np.float32)
        np.savez_compressed(out/'caches.npz',**ca);artifacts.append('caches.npz')
        for length in p['prefix_lengths']:
            z=rng.normal(size=(64,3));name=f'native-L{length}.npy';np.save(out/name,z);artifacts.append(name)
            mean=np.full((5,job['heads'],64,64),.25)
            op=dict(mean=mean,powers=np.full((5,64),1.25),scatter=np.full(5,1.1875))
            for a in p['arms']:
                if a['length']!=length:continue
                cg=ca[f"C{a['calibration_length']}-P{a['panel']}-M{a['size']}"][:,2,0].astype(float)
                op[a['name']+'-bias']=np.mean((mean-cg)**2,axis=(1,2,3))
                op[a['name']+'-risk']=op['scatter']+op[a['name']+'-bias']
                name=a['name']+'.npy';np.save(out/name,z+.01*rng.normal(size=z.shape));artifacts.append(name)
            name=f'operators-L{length}.npz';np.savez_compressed(out/name,**op);artifacts.append(name)
        write_json(out/'manifest.json',dict(status='complete',job=job,protocol_sha256=ph,training_updates=0,
            qualification=dict(exact_own_operator_replay=True,exact_native_restoration=True,native_generator_calls=40,cached_generator_calls=0,prefix_lengths=[32,64,128],longest_batch_one=True),
            peak_allocated_bytes=1,elapsed_seconds=1,calls={k:v//24 for k,v in p['expected_calls'].items()},artifacts={n:sha256(out/n) for n in artifacts}))
    return p
