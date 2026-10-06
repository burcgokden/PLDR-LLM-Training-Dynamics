#!/usr/bin/env python
"""Qualify all near/subcritical scalar drives against the public scheduler."""
from companion_paths import required_input
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model_rg.controlled import bind_run
from model_rg.provenance import sha256,write_json
from model_rg.scheduled_regimes import recipe,RECIPE_NAMES
from model_rg.schedules import multiplier
from check_reference_schedule import public_functions
from verify_scheduled_raw import independent_recipe,reference_multiplier


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-20260908');a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];study=Path(a.root).resolve()/a.study
    table=Path(required_input('reference-schedule')) / 'regime-table.json';spec=json.loads(table.read_text())
    reference=Path(required_input('reference-schedule')) / 'pldr_run_model_v510.py'
    out=study/'qualification/reference-regimes';out.mkdir(parents=True,exist_ok=False)
    files=[table,Path(spec['source']),reference,study/'qualification/reference-cpu-final/manifest.json']
    if sha256(spec['source'])!=spec['source_sha256']:raise AssertionError('The reference paper changed')
    bind_run(out,files,vars(a));public=public_functions(reference);raw={};records=[]
    table_profiles={r['recipe']:r for r in spec['profiles']}
    for name in RECIPE_NAMES:
        for n in [4,14]:
            profile=recipe(name,n)
            if profile!=independent_recipe(name,n):raise AssertionError('Independent reference recipe differs')
            if name in table_profiles:
                row=table_profiles[name]
                if (profile['generator_peak']!=row['peak_rate'] or profile['body_peak']!=row['peak_rate'] or
                    profile['warmup_steps']!=row['warmup_steps']):raise AssertionError('The paper recipe differs')
        parameter=torch.nn.Parameter(torch.ones(1));opt=torch.optim.AdamW([parameter],lr=profile['generator_peak'])
        scheduler=public['LinearWarmupCosineLRSchedule'](opt,250000,profile['warmup_steps'],.1)
        expected=np.array([scheduler.lr_lambdas[0](k) for k in range(262146)])
        actual=np.array([multiplier(k,profile['warmup_steps'],250000,.1) for k in range(262146)])
        checked=np.array([reference_multiplier(k,profile) for k in range(262146)])
        if expected.tobytes()!=actual.tobytes() or expected.tobytes()!=checked.tobytes():
            raise AssertionError('A producer or verifier scalar curve differs from public code')
        raw[name+'_curve']=actual
        records.append(dict(recipe=name,heads_checked=[4,14],phase_values=len(actual),byte_equal=True,
            drift_multiplier_sum=float(actual[:262144].sum()),noise_multiplier_sum=float(np.square(actual[:262144]).sum()),
            floor_updates=12144,warmup_steps=profile['warmup_steps']))
    np.savez_compressed(out/'measurements.npz',**raw)
    write_json(out/'results.json',dict(schema='reference-regime-qualification-v1',status='complete',recipes=records,
        native_program_qualification=str(study/'qualification/reference-cpu-final/manifest.json'),
        scope='All five recipes and both head counts are checked. The two added subcritical recipes use the same qualified optimizer, clipping and all-token objective, with different peak rates and warmup lengths. This record qualifies their scalar drives and Table1 correspondence; it assigns no empirical regime classification.'))
    write_json(out/'manifest.json',dict(status='complete',binding_sha256=sha256(out/'binding.json'),
        results_sha256=sha256(out/'results.json'),raw_sha256=sha256(out/'measurements.npz')))
    print('All five near/subcritical comparison recipes qualified through the floor tail',flush=True)


if __name__=='__main__':main()
