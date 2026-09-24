#!/usr/bin/env python
"""Frozen CPU-only source coupling, with complete saved pairs and direct checks."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
from model_rg.provenance import sha256, write_json
from model_rg.resource_coupling import draw_pair, edit_moments


def run(study, native, repo):
    study.mkdir(parents=True, exist_ok=False)
    spec = json.loads((native/'protocol.json').read_text())
    inputs = {str(native/'protocol.json'): sha256(native/'protocol.json')}
    for c in spec['cases']:
        p = native/'sampling'/(c['name']+'.npz'); inputs[str(p)] = sha256(p)
    sources = {name: sha256(repo/name) for name in
               ['scripts/resource_coupling_study.py', 'src/model_rg/resource_coupling.py', 'src/model_rg/provenance.py']}
    for name in sources:
        p = study/'executed-source'/name; p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo/name, p)
    protocol = dict(status='frozen', frozen_at=datetime.now(timezone.utc).isoformat(),
        draws_per_case=4096, length=2048, seed=115201000, native=str(native), cases=spec['cases'],
        inputs_sha256=inputs, sources=sources, native_updates=0,
        target='Hamming edits under a uniform remaining-resource sequence coupling; no model observations or sensitivity estimates.')
    write_json(study/'protocol.json', protocol)
    tick = time.perf_counter(); records = []
    for ci, c in enumerate(spec['cases']):
        with np.load(native/'sampling'/(c['name']+'.npz')) as f:
            used = f['primary']; removed = f['successor'].ravel()
        remaining = np.setdiff1d(np.arange(524288, dtype=np.int32), used)
        later = np.setdiff1d(remaining, removed); removed_set = set(int(v) for v in removed)
        shape = (protocol['draws_per_case'], 2, protocol['length'])
        path = study/(c['name']+'.npy')
        pairs = np.lib.format.open_memmap(path, mode='w+', dtype=np.int32, shape=shape)
        rng = np.random.default_rng(protocol['seed']+ci)
        for i in range(shape[0]):
            pairs[i, 0], pairs[i, 1] = draw_pair(rng, remaining, later, removed_set, shape[-1])
        pairs.flush()
        edits = np.sum(pairs[:, 0] != pairs[:, 1], axis=1)
        mean, var = edit_moments(len(remaining), len(removed), shape[-1])
        log_survival = sum(math.log1p(-len(removed)/(len(remaining)-j)) for j in range(shape[-1]))
        records.append(dict(case=c['name'], population=len(remaining), removed=len(removed), length=shape[-1],
            draws=shape[0], theoretical_mean=mean, theoretical_variance=var, index_law_tv=-math.expm1(log_survival),
            mean=float(edits.mean()), variance=float(edits.var(ddof=1)), mcse=float(edits.std(ddof=1)/math.sqrt(len(edits))),
            zero_edits=int(np.sum(edits==0)), maximum_edits=int(edits.max()), path=str(path), sha256=sha256(path)))
        print('Completed source coupling', c['name'], flush=True)
        del pairs
    write_json(study/'results.json', dict(status='complete', protocol_sha256=sha256(study/'protocol.json'),
        records=records, source_coupling_draws=sum(r['draws'] for r in records), native_updates=0,
        seconds=time.perf_counter()-tick, completed_at=datetime.now(timezone.utc).isoformat()))


def verify(study, repo):
    dest = study/'verification.json'
    if dest.exists(): raise FileExistsError(dest)
    spec = json.loads((study/'protocol.json').read_text()); result = json.loads((study/'results.json').read_text())
    checked = {}
    def check(p, h=None):
        p = Path(p); value = sha256(p); checked[str(p.resolve())] = value
        assert h is None or value == h
    check(study/'protocol.json', result['protocol_sha256']); check(study/'results.json')
    for p, h in spec['inputs_sha256'].items(): check(p, h)
    for p, h in spec['sources'].items(): check(repo/p, h); check(study/'executed-source'/p, h)
    for r in result['records']:
        check(r['path'], r['sha256']); pairs = np.load(r['path'], mmap_mode='r')
        assert pairs.shape == (4096, 2, 2048)
        with np.load(Path(spec['native'])/'sampling'/(r['case']+'.npz')) as f:
            used=f['primary']; removed=f['successor'].ravel()
        assert not np.isin(pairs, used).any()
        assert np.all((pairs>=0)&(pairs<524288))
        assert not np.isin(pairs[:,1], removed).any()
        assert np.all(np.diff(np.sort(pairs, axis=-1), axis=-1)>0)
        hits=np.isin(pairs[:,0], removed)
        assert np.array_equal(hits, pairs[:,0]!=pairs[:,1])
        counts=[int(x) for x in hits.sum(axis=1)]
        mean=math.fsum(counts)/len(counts)
        variance=math.fsum((x-mean)**2 for x in counts)/(len(counts)-1)
        assert abs(mean-r['mean'])<1e-12 and abs(variance-r['variance'])<1e-12
        assert abs(math.sqrt(variance/len(counts))-r['mcse'])<1e-12
        assert r['zero_edits']==counts.count(0) and r['maximum_edits']==max(counts)
        pop=524288-len(used); k=len(removed); ell=2048
        expectation=ell*k/pop
        second=ell*k/pop+ell*(ell-1)*k*(k-1)/(pop*(pop-1))
        assert abs(r['theoretical_mean']-expectation)<1e-12
        assert abs(r['theoretical_variance']-(second-expectation**2))<1e-12
        assert abs(r['index_law_tv']+math.expm1(math.fsum(math.log1p(-k/(pop-j)) for j in range(ell))))<1e-12
    assert result['source_coupling_draws']==32768 and result['native_updates']==0
    write_json(dest, dict(status='passed', source_coupling_draws=32768, saved_sequences=65536, native_updates=0,
        inputs_sha256=checked, verifier_sha256=sha256(__file__), scope='Complete saved-source legality, exact edit identities and direct moment reconstruction; no model sensitivities or native forecasts.'))
    print('Passed complete source-coupling reconstruction', flush=True)




