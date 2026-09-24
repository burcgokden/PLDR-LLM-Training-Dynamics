#!/usr/bin/env python
"""Bind complete single-pass outcome and temporal analysis to the selected study."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from onepass_analysis import GROUP_KEYS, group_key, group_record, windows
from model_rg.provenance import sha256, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-feasible-20260908');a=p.parse_args()
    study=Path(a.root).resolve()/a.study;repo=Path(__file__).resolve().parents[1]
    out=study/'protocols/onepass-analysis-selection.json'
    if out.exists():raise FileExistsError(out)
    training=study/'protocols/onepass-training-selection.json';selection=study/'protocols/onepass-observation-selection.json'
    jobs=json.loads(training.read_text())['jobs'];cases=json.loads(selection.read_text())['cases'];groups=defaultdict(list)
    for case in cases:groups[group_key(case)].append(case['seed'])
    if len(cases)!=284 or len(groups)!=80 or sum(c['inference_stability'] for c in cases)!=94:
        raise AssertionError('The full current single-pass selection is required')
    if any(len(v)!=len(set(v)) or len(v) not in (1,2,4) for v in groups.values()):
        raise AssertionError('Unexpected drive and initialization coverage')
    contrasts=[]
    for first in ['reference1','reference2']:
        for second in ['subcritical1','subcritical2']:
            for key in sorted(groups):
                h,r,n,t,c,b=key
                if r!=first:continue
                other=(h,second,n,t,c,b)
                if other not in groups:continue
                seeds=sorted(groups[key])
                if seeds!=sorted(groups[other]):raise AssertionError('Paired recipe coverage differs')
                contrasts.append(dict(first=first,second=second,schedule_horizon=h,heads=n,step=t,
                    shared_seed=c,stream_seed=b,first_key=list(key),second_key=list(other),seeds=seeds))
    qa=study/'qualification/onepass-analysis/verification.json';check=json.loads(qa.read_text())
    if check['status']!='passed':raise AssertionError('Independent statistical qualification is required')
    for name,digest in check['source_sha256'].items():
        if sha256(repo/name)!=digest:raise AssertionError('The qualified implementation changed')
    inputs=[training,selection,qa,*[study/'protocols'/name for name in
        ['scheduled-prefix-selection.json','reasoning-mechanism-selection.json','onepass-shared-update-observation.json']]]
    known=[]
    for job in jobs:
        folder=study/'runs'/job['run_id'];path=folder/'manifest.json'
        if path.exists():
            meta=json.loads(path.read_text());known.append(dict(run_id=job['run_id'],status=meta['status'],
                completed_step=meta['completed_step'],manifest_sha256=sha256(path)))
        elif folder.exists():known.append(dict(run_id=job['run_id'],status='started_without_final_manifest'))
    sources=['onepass_analysis','analyze_onepass_collectives','verify_onepass_collectives','analyze_onepass_predictions',
        'verify_onepass_predictions','analyze_onepass_paths','verify_onepass_paths','analyze_scaling_collectives',
        'analyze_scaling_updates','verify_scheduled_collectives','check_onepass_analysis','prepare_onepass_analysis',
        'run_onepass_analyses']
    files=['scripts/'+n+'.py' for n in sources]+['src/model_rg/controlled.py','src/model_rg/provenance.py']
    spec=dict(schema='onepass-outcome-analysis-selection-v1',frozen_at=datetime.now(timezone.utc).isoformat(),
        run_ids=[j['run_id'] for j in jobs],selected_states=284,groups=[dict(**group_record(k),seeds=sorted(v)) for k,v in sorted(groups.items())],
        group_keys=list(GROUP_KEYS),precisions=['float32','float64'],recipe_contrasts=contrasts,
        phase_local_block_widths=[2048,8192,32768],thresholds=[.1,.01,.001,.0001,.00001],
        selected_temporal_windows=sum(len(windows(j,[2048,8192,32768])) for j in jobs),
        development_scope='Outcome aggregation extends the prior scheduled finite-ensemble and temporal methods to the selected nonrepeated data law and exact drive horizons. Two fresh source-schedule paths and retained repeated outcomes were already observed when this implementation was selected. The completed and started identities are explicitly listed; no claim of selection before all scientific execution is made.',
        known_native_states=known,
        uncertainty='Whole-initialization empirical resampling enumerates all s^s ordered resamples for the declared s=2 or4. Percentile ranges are conditional on the fixed shared initialization, complete batch history and evaluation cohort; they carry no guaranteed population coverage or multiple-testing interpretation. Single-seed mean values are retained without an invented uncertainty interval. Contexts and heads are not substituted for initialization replicas.',
        single_seed_scope='One-seed covariances, fourth ratios, effective head counts and uncertainty are unavailable. Two-seed covariance estimates are retained as small-ensemble diagnostics; deleting one of two seeds cannot estimate a variance. Finite-sample Gaussian Binder offsets are explicit.',
        criticality_scope='The controlled widths N4,N8,N14 and externally imposed drive histories define finite conditional comparisons. Scaling a common learned mode with N, decreasing row contrast, improved prediction and a schedule-driven transition do not separately establish a critical surface, thermodynamic exponent or self-organized criticality. No label inherited from a reference recipe assigns a physical phase in this family.',
        predictive_laws='All284 selected frozen risks and proper-prefix states are retained, including all94 selected four-mode task and paired-generation endpoints. Recipe contrasts use the same drive horizon, head count, saved update, shared state, stream and initialization identities at every common saved time. No unequal warmup time is interpolated.',
        endpoint_coverage='At constant N4/N14 update32768, all four identities have core risk/prefix observations, while only seeds640103/640104 have endpoint generation/task observations; seeds640101/640102 receive those observations at131072. Each field records its selected and unselected identities. This preselected coverage is not treated as missing random data or extrapolated to four endpoint replicas.',
        undefined_ratios='A selected undefined scalar remains null and prevents the selected-identity mean. An unselected endpoint field is recorded separately from an undefined measured ratio. No selected seed is dropped to make a finite summary.',
        phase_rule='Constant paths use a single observed constant phase. Scheduled paths use nonempty warmup, annealing and floor intersections with the exact selected stop. The two250000-horizon paths end at65536 and have no completed cosine or floor phase. Consecutive fixed-width blocks start at each phase boundary and never cross it; incomplete residual blocks are not counted as full blocks.',
        probe_rule='Every-update native projected changes and clipped gradients are analyzed separately from fixed-context emissions on the regular64-update grid. Non-grid warmup/saved probes are retained by the raw path but excluded from regular-spacing covariance. Intervals are left-closed and right-open.',
        floor_rule='Each compact path has8192 selected updates at the nonzero floor, divided additionally into two equal4096-update halves. A constant externally set rate does not remove the changing finite remaining-data resource.',
        temporal_inference='Centered and linearly detrended finite-window covariances and geometric-lag correlations are reported together. No fitted relaxation time is assigned to zero variance, a deterministic drift or a window longer than the observed regime. Threshold intervals retain onset/offset censoring and are not labeled avalanche laws. These finite diagnostics cannot certify stationarity or SOC.',
        inputs_sha256={str(p):sha256(p) for p in inputs},producer_sources={n:sha256(repo/n) for n in files})
    write_json(out,spec)
    print('Recorded80 groups,',len(contrasts),'paired recipe conditions and',spec['selected_temporal_windows'],'temporal windows',sha256(out),flush=True)


if __name__=='__main__':main()
