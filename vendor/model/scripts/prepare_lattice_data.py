#!/usr/bin/env python
"""Build and qualify independent finite Ising/Potts Monte Carlo corpora."""
from companion_paths import legacy_path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from model_rg.lattice import critical_temperature, exact_small, observables
from model_rg.provenance import sha256, write_json

ROOT=Path(legacy_path('/pldr-data/model'))


def compile_sampler(study):
    build=study/'sampler';build.mkdir(parents=True,exist_ok=True)
    source=REPO/'scripts/potts_mc.cpp'; binary=build/'potts_mc'
    if not binary.exists():
        subprocess.run(['g++','-O3','-std=c++17',str(source),'-o',str(binary)],check=True)
    return binary


def generate(binary, dest, q, L, ratio, seed, chains, samples, kernel='wolff', burn=4096, thin=12):
    dest.mkdir(parents=True,exist_ok=False)
    path=dest/'configurations.bin'; temperature=critical_temperature(q)*ratio
    args=[str(binary),str(q),str(L),repr(temperature),str(seed),str(chains),str(samples),
          str(burn),str(thin),kernel,str(path)]
    start=time.perf_counter();run=subprocess.run(args,check=True,capture_output=True,text=True)
    if path.stat().st_size!=chains*samples*L*L:raise ValueError('Wrong output size')
    spec=dict(q=q,L=L,temperature=temperature,temperature_ratio=ratio,seed=seed,chains=chains,
              samples_per_chain=samples,burn_updates=burn,updates_between_samples=thin,kernel=kernel,
              path=str(path),shape=[chains,samples,L,L],dtype='uint8',sha256=sha256(path),
              runtime_seconds=time.perf_counter()-start,execution=json.loads(run.stdout))
    write_json(dest/'manifest.json',spec)
    return spec


def qualification(study):
    binary=compile_sampler(study);dest=study/'data-qualification';dest.mkdir(exist_ok=False)
    jobs=[]
    for q,L in [(2,4),(3,3)]:
        for r in [.8,1.,1.3]:
            for kernel in ['wolff','heatbath']:
                name=f'q{q}-L{L}-r{r:.3f}-{kernel}'
                seed=1700100+len(jobs)
                jobs.append((name,q,L,r,kernel,seed))
    reports=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures={pool.submit(generate,binary,dest/name,q,L,r,seed,8,4096,kernel):name
                 for name,q,L,r,kernel,seed in jobs}
        for future in as_completed(futures):
            spec=future.result();raw=np.memmap(spec['path'],dtype=np.uint8,mode='r',shape=tuple(spec['shape']))
            exact=exact_small(spec['q'],spec['L'],spec['temperature'])
            obs=observables(raw.reshape(-1,spec['L'],spec['L']),spec['q'])
            checks={}
            for key in ['energy','m','m2','m4','field','corr1']:
                chain_means=obs[key].reshape(spec['chains'],-1).mean(1)
                se=float(chain_means.std(ddof=1)/math.sqrt(len(chain_means)))
                discrepancy=float(abs(chain_means.mean()-exact[key]))
                tolerance=max(7*se, .002 if key!='energy' else .025)
                checks[key]=dict(observed=float(chain_means.mean()),exact=exact[key],
                                 chain_standard_error=se,absolute_error=discrepancy,tolerance=tolerance,
                                 passed=discrepancy<=tolerance)
            reports.append(dict(spec=spec,checks=checks))
            print(f"qualified {futures[future]} {all(c['passed'] for c in checks.values())}",flush=True)
    passed=all(c['passed'] for r in reports for c in r['checks'].values())
    write_json(dest/'verification.json',dict(status='passed' if passed else 'failed',
        sampler_source_sha256=sha256(REPO/'scripts/potts_mc.cpp'),sampler_binary_sha256=sha256(binary),
        checker_sha256=sha256(__file__),cases=reports,
        scope='Exact finite-volume equilibrium moments using independent local and cluster kernels; chain-level Monte Carlo errors. No dynamical exponent inference.'))
    if not passed:raise ValueError('Sampler did not qualify')


def corpus(study, stage):
    proof=json.loads((study/'data-qualification/verification.json').read_text())
    if proof['status']!='passed' or proof['sampler_source_sha256']!=sha256(REPO/'scripts/potts_mc.cpp'):
        raise ValueError('Current sampler qualification required')
    binary=compile_sampler(study);dest=study/stage;dest.mkdir(exist_ok=False)
    if stage=='development-data': sizes=[4,8,16];ratios=[.85,1.,1.15];chains=8;samples=2048;origin=1710000
    elif stage=='training-data':sizes=[4,8,16];ratios=[.85,.925,1.,1.075,1.15];chains=16;samples=2048;origin=1720000
    elif stage=='assessment-data':sizes=[4,6,8,12,16,24,32,48,64];ratios=[.94,.97,1.,1.03,1.06];chains=16;samples=512;origin=1730000
    else:raise ValueError('Unknown data stage')
    jobs=[(q,L,r) for q in [2,3] for L in sizes for r in ratios]
    specs=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures={pool.submit(generate,binary,dest/f'q{q}-L{L}-r{r:.3f}',q,L,r,origin+j,
                             chains,samples):j for j,(q,L,r) in enumerate(jobs)}
        for future in as_completed(futures):
            spec=future.result();spec['id']=futures[future];specs.append(spec)
            print(f"{stage} q={spec['q']} L={spec['L']} r={spec['temperature_ratio']} {spec['runtime_seconds']:.1f}s",flush=True)
    spec=dict(schema='physical-corpus-v1',status='complete',stage=stage,
        created_at=datetime.now(timezone.utc).isoformat(),cells=sorted(specs,key=lambda s:s['id']),
        law='H=-sum_equal_right_down_bonds, periodic square lattice; independent seeded chains; fixed cluster-update spacing; q=2 temperature is half standard Ising spin-product units.',
        identities='Each (cell,chain,sample) is an observation identity. Duplicate configurations allowed by the physical law are not rejected.',
        qualification_sha256=sha256(study/'data-qualification/verification.json'),
        sampler_source_sha256=sha256(REPO/'scripts/potts_mc.cpp'),sampler_binary_sha256=sha256(binary))
    write_json(dest/'manifest.json',spec)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',required=True)
    ap.add_argument('--stage',choices=['qualification','development-data','training-data','assessment-data'],required=True)
    a=ap.parse_args();study=Path(a.study).resolve()
    if not study.is_relative_to(ROOT):raise ValueError('Unauthorized data destination')
    study.mkdir(parents=True,exist_ok=True)
    if a.stage=='qualification':qualification(study)
    else:corpus(study,a.stage)


if __name__=='__main__':main()
