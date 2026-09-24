#!/usr/bin/env python3
"""Freeze the bounded potential-avalanche assay before any scientific path runs."""
from companion_paths import legacy_path
import argparse
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256,write_json,environment
from model_rg.onepass_regimes import sample_batches
ROOT=Path(legacy_path('/pldr-data/model'))
STUDY=ROOT/'potential-avalanche-20260913'
REPO=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args()
    seeds=[913101,913102]
    if not (STUDY/'selection.npz').exists():
        token=np.load(ROOT/'data/refinedweb-onepass-524288/tokens.npy',mmap_mode='r')
        selected={}
        for seed in seeds:
            rows,offsets,_=sample_batches(token,seed+500000,2048)
            selected[f'rows_{seed}']=rows;selected[f'offsets_{seed}']=offsets
        probe=ROOT/'controlled-study-20260905/data/short'
        pt=np.load(probe/'tokens.npy',mmap_mode='r');po=np.load(probe/'offsets.npy')
        rr=np.arange(512,576)
        selected['probes']=pt[rr[:,None],po[rr,None]+np.arange(65)]
        selected['probe_rows']=rr
        selected['coordinates']=np.sort(np.random.default_rng(913001).choice(4096,128,replace=False))
        np.savez_compressed(STUDY/'selection.npz',**selected)
    sources=['scripts/train_potential_avalanches.py','src/model_rg/training.py',
      'src/model_rg/native.py','src/model_rg/variance_family.py','src/model_rg/criticality.py',
      'src/model_rg/controlled.py','src/model_rg/provenance.py']
    spec=dict(schema='potential-avalanche-assay-v1',environment=environment(),
        producer_sources={x:sha256(REPO/x) for x in sources},
        data=dict(training_manifest_sha256=sha256(ROOT/'data/refinedweb-onepass-524288/manifest.json'),
          probe_manifest_sha256=sha256(ROOT/'controlled-study-20260905/data/short/manifest.json'),
          selection_sha256=sha256(STUDY/'selection.npz')),
        observation='Every optimizer update; two fixed unseen 64-token inputs; 128 fixed matrix entries per head plus all-entry summaries. All 64 evaluation inputs are excluded from gradients.',
        objective='One external next token per distinct 64-token RefinedWeb prefix; batch32; no block reuse within a path.',
        initialization='Native family, five layers, head dimension64, eight residual metric units; variance-normalized head family; independent full initialization seeds; matched data order per seed.',
        statistical_plan=dict(phases=[[1,256],[257,768],[769,1280],[1281,2048]],
            scalar='RMS of all-coordinate fixed-probe log-potential update; primary probe0, probe1 replication',
            thresholds=[.8,.9,.95],threshold_rule='Within-phase quantiles for descriptive excursions, plus frozen calibration from steps257..512 applied to subsequent steps.',
            time_bins=[1,2,4],units='optimizer updates; repeat for activity divided by applied learning rate',
            nulls=['whole-phase time permutation','within-64-update-block permutation','phasewise IAAFT correlated amplitude-preserving surrogate'],
            events='Maximal consecutive above-threshold bins; size is integrated excess activity; duration in updates; exclude boundary-censored events and retain censoring flags.',
            tails='Continuous Pareto MLE with cutoff selected by KS and >=50 tail events; semiparametric bootstrap refits cutoff; compare conditional exponential and lognormal; all settings retained.',
            evidence='Power-law compatibility alone is not evidence of SOC. Need temporal/causal propagation beyond common drive, size-dependent scaling and endogenous control feedback; this bounded assay may remain inconclusive.',
            inferential_units='Training realizations, not coordinates, heads, events, or time bins. Tail bootstrap is conditional diagnostic under iid tail assumption; serial nulls evaluate dependence.'),
        save_steps=[1024])
    if a.freeze:
        path=STUDY/'protocols/training.json'
        if path.exists():raise ValueError('Frozen scientific selection already exists')
        spec['jobs']=[dict(run_id=f'h{h}-{sc}-s{s}',heads=h,schedule=sc,seed=s,steps=2048,warmup=256,peak=8e-4)
                      for h in [2,4,8] for sc in ['constant','warm_plateau','warm_cosine'] for s in seeds]
    else:
        path=STUDY/'protocols/pilot.json'
        spec['jobs']=[dict(run_id='h8-pilot',heads=8,schedule='warm_cosine',seed=seeds[0],steps=32,warmup=8,peak=8e-4)]
    write_json(path,spec)
    print(path)
if __name__=='__main__':main()
