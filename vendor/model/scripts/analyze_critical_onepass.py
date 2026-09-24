#!/usr/bin/env python3
"""Reduce all native criticality cells, with no publication side effects."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
import numpy as np
from model_rg.critical_onepass import conditional_statistics, peak_profile, seed_uncertainty
from model_rg.provenance import sha256, write_json

REPO = Path(__file__).resolve().parents[1]


def analyze(study, output, partial=False):
    started = time.monotonic()
    if output.exists(): raise FileExistsError(output)
    p = json.loads((study/'protocol.json').read_text())
    if p['schema'] != 'critical-onepass-v1' or p['role'] != 'scientific':
        raise ValueError('Expected a frozen scientific native protocol')
    ph = sha256(study/'protocol.json')
    groups = defaultdict(list)
    missing, failed, inventory, checked = [], [], [], {str(study/'protocol.json'): ph}
    completed_steps = 0
    for job in p['jobs']:
        root = study/'runs'/job['run_id']
        mp = root/'manifest.json'
        if not mp.exists():
            missing.append(job['run_id']); continue
        m = json.loads(mp.read_text())
        if m['job'] != job or m['role'] != 'scientific' or m['protocol_sha256'] != ph:
            raise ValueError('Manifest/protocol mismatch: '+job['run_id'])
        checked[str(mp)] = sha256(mp)
        completed_steps += m.get('scientific_updates',0)
        inventory.append(dict(run_id=job['run_id'], status=m['status'],
                              steps=m['completed_steps'], runtime_seconds=m.get('runtime_seconds',0),
                              peak_bytes=m.get('maximum_cuda_memory_bytes') or 0))
        if m['status'] != 'complete':
            failed.append(dict(run_id=job['run_id'], status=m['status'], reason=m.get('error')))
            if m['status'] not in ['numerical_failure','resource_failure','program_error','incomplete']:
                missing.append(job['run_id'])
            continue
        ap = root/'observations.npz'
        ah = sha256(ap)
        if ah != m['artifacts']['observations.npz']:
            raise ValueError('Observation hash mismatch')
        checked[str(ap)] = ah
        if m['completed_steps'] != job['steps']:
            raise ValueError('Incomplete successful run')
        groups[(job['environment'], job['heads'], job['control'])].append((job, root, m))
    if missing and not partial: raise ValueError(f'Missing {len(missing)} scientific paths')
    cells, arrays_by_cell = [], {}
    expected_seeds = set(p['design']['seeds'])
    for (e, h, g), rows in sorted(groups.items()):
        if {j['seed'] for j, _, _ in rows} != expected_seeds:
            if partial: continue
            # Failed seeds remain explicit. A smaller success ensemble is not substituted.
            continue
        rows.sort(key=lambda r:r[0]['seed'])
        if len({m['initial_shared_sha256'] for _, _, m in rows}) != 1:
            raise ValueError('Shared initial environment changed within cell')
        native = [np.load(root/'observations.npz', allow_pickle=False) for _,root,_ in rows]
        steps = native[0]['steps']
        if not all(np.array_equal(a['steps'], steps) for a in native):
            raise ValueError('Observation grids differ')
        if not all(np.array_equal(a['blocks'], native[0]['blocks']) for a in native):
            raise ValueError('Source differs across conditional replicas')
        data = np.stack([a['heads'] for a in native])
        if data.shape != (len(rows), len(steps), 64, 5, h, 4):
            raise ValueError('Unexpected head-field shape')
        for i, t in enumerate(steps):
            for label, index in [('row',0), ('attention',1), ('operator',2), ('common_amplitude',None)]:
                if index is None:
                    energy = data[:,i,...,3]
                    x = np.sqrt(np.maximum(energy-data[:,i,...,0]*np.maximum(energy,1e-30),0.))
                else:
                    x = data[:,i,...,index]
                stat = conditional_statistics(x)
                stat.update(environment=e, heads=h, control=g, step=int(t), field=label,
                            seed_ids=[j['seed'] for j,_,_ in rows],
                            analysis_role='primary' if label=='row' else 'secondary' if label=='attention' else 'exploratory',
                            generator_clock=float(t*3e-4*g), other_clock=float(t*3e-4*2/h),
                            consumed_fraction=float(t*32/p['source']['population_blocks']),
                            below_row_denominator_fraction=float(np.mean(data[:,i,...,3] < 1e-30)),
                            nll_by_seed=[float(a['nll_path'][i].mean()) for a in native])
                if int(t) in p['observations']['milestone_steps']:
                    stat['seed_uncertainty'] = seed_uncertainty(x)
                    stat['evaluation_nll_by_seed'] = [float(a[f'evaluation_nll_{t}'].mean()) for a in native]
                cells.append(stat)
                if int(t) == p['design']['steps']:
                    arrays_by_cell[(e,h,g,label)] = x
        for a in native: a.close()
    curves = defaultdict(list)
    for row in cells:
        curves[(row['environment'],row['heads'],row['step'],row['field'])].append(row)
    peaks = []
    for (e,h,t,field), rows in sorted(curves.items()):
        rows.sort(key=lambda r:r['control'])
        if len(rows) != len(p['design']['controls']): continue
        q = peak_profile([r['control'] for r in rows],[r['susceptibility'] for r in rows])
        q.update(environment=e,heads=h,step=t,field=field)
        peaks.append(q)
    result = dict(schema='critical-onepass-analysis-v1', status='partial' if missing else 'complete',
        study=str(study), protocol_sha256=ph, design=p['design'], planned_paths=len(p['jobs']),
        completed_paths=sum(r['status']=='complete' for r in inventory), accounted_paths=len(inventory), scientific_updates=completed_steps,
        missing=missing, failed_paths=failed, numerical_failures=[r for r in failed if r['status']=='numerical_failure'], inventory=inventory, cells=cells, peaks=peaks,
        summed_job_seconds=sum(r['runtime_seconds'] for r in inventory),
        maximum_cuda_memory_bytes=max((r['peak_bytes'] for r in inventory),default=0),
        checked_sha256=checked,
        source_sha256={name:sha256(REPO/name) for name in
            ['scripts/analyze_critical_onepass.py','src/model_rg/critical_onepass.py','src/model_rg/provenance.py']},
        analysis_seconds=time.monotonic()-started,
        interpretation='Complete conditional finite seed statistics. A peak or fitted finite-size slope is not a criticality certificate.')
    write_json(output,result)
    print(json.dumps({k:result[k] for k in ['status','completed_paths','scientific_updates','analysis_seconds']}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--partial',action='store_true')
    a=p.parse_args()
    analyze(a.study.resolve(),a.output.resolve(),a.partial)

if __name__=='__main__': main()
