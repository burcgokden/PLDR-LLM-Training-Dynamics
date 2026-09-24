#!/usr/bin/env python
"""Check the entire no-repetition stream and every selected scalar drive phase."""
import argparse
import json
from pathlib import Path

import numpy as np

from model_rg.controlled import bind_run
from model_rg.onepass_regimes import sample_batches, recipe, multiplier
from model_rg.provenance import sha256, write_json
from verify_onepass_raw import sample_indices, independent_recipe, reference_multiplier


def unique_blocks(rows, offsets):
    if np.any(offsets < 0) or np.any(offsets > 448) or np.any(offsets % 64):
        raise AssertionError('A target block is not aligned to the nonoverlapping partition')
    if np.any(rows < 0) or np.any(rows >= 524288):
        raise AssertionError('A source row is outside the verified corpus')
    ids = (8*rows+offsets//64).ravel()
    if len(np.unique(ids)) != len(ids):
        raise AssertionError('A supervised source position is repeated')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--study', default='scheduled-training-feasible-20260908')
    a = p.parse_args()
    root = Path(a.root).resolve()
    study = root/a.study
    corpus = root/'data/refinedweb-onepass-524288'
    proof = study/'verification/onepass-data.json'
    if json.loads(proof.read_text())['status'] != 'passed':
        raise AssertionError('The complete independent corpus replay is required')
    out = study/'qualification/onepass-stream'
    out.mkdir(parents=True, exist_ok=False)
    bind_run(out, [proof, corpus/'manifest.json', corpus/'tokens.npy',
                   study/'protocols/onepass-data-selection.json'], vars(a))
    tokens = np.load(corpus/'tokens.npy', mmap_mode='r')
    rows, offsets, batches = sample_batches(tokens, 640001, 131072)
    rr, oo = sample_indices(640001, 131072)
    if rows.tobytes() != rr.tobytes() or offsets.tobytes() != oo.tobytes():
        raise AssertionError('The independent full-stream permutation differs')
    unique_blocks(rows, offsets)
    for start in range(0, 131072, 256):
        expected = tokens[rr[start:start+256,:,None], oo[start:start+256,:,None]+np.arange(65)]
        if batches[start:start+256].tobytes() != expected.tobytes():
            raise AssertionError('A native batch differs from its unique source block')
    prefix_rows, prefix_offsets, prefix_batches = sample_batches(tokens, 640001, 1024)
    for full, prefix in [(rows,prefix_rows), (offsets,prefix_offsets), (batches,prefix_batches)]:
        if full[:1024].tobytes() != prefix.tobytes():
            raise AssertionError('The training stream depends on the requested horizon')
    rejected = []
    for name, r, o in [('duplicate', rows[:2].copy(), offsets[:2].copy()),
                       ('overlap', rows[:2].copy(), offsets[:2].copy())]:
        if name == 'duplicate':
            r[1,0], o[1,0] = r[0,0], o[0,0]
        else:
            o[1,0] = 32
        try:
            unique_blocks(r,o)
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError('A nonrepetition mutation was accepted')
    try:
        sample_batches(tokens,640001,131073)
    except ValueError:
        rejected.append('exhausted_stream')
    else:
        raise AssertionError('The sampler silently repeated an exhausted corpus')
    del batches, rr, oo, prefix_batches
    drives = []
    for horizon, warmup_override, stop, names in [
            (32768,-1,40960,['controlled','reference1','reference2','subcritical1','subcritical2']),
            (250000,-1,65536,['reference1','subcritical1']),
            (0,-1,131072,['controlled']),
            (768,128,1024,['controlled','reference1'])]:
        for name in names:
            for heads in [4,8,14]:
                profile = recipe(name,heads,horizon,warmup_override)
                if profile != independent_recipe(name,heads,horizon,warmup_override):
                    raise AssertionError('The independently specified native optimizer recipe differs')
                actual = np.array([multiplier(t,profile['warmup_steps'],profile['total_steps'],
                                              profile['floor_fraction']) for t in range(stop+1)])
                expected = np.array([reference_multiplier(t,profile) for t in range(stop+1)])
                if actual.tobytes() != expected.tobytes():
                    raise AssertionError('An applied or stored-next scalar drive phase differs')
                drives.append(dict(recipe=name,heads=heads,total_steps=horizon,
                    warmup_steps=profile['warmup_steps'],phases=stop+1,
                    multipliers_sha256=__import__('hashlib').sha256(actual.tobytes()).hexdigest()))
    write_json(out/'results.json', dict(status='passed', unique_blocks=4194304,
        unique_supervised_source_positions=268435456, maximum_updates=131072,
        source_token_values_replayed_bytewise=4194304*65, prefix_updates=1024,
        rejected_mutations=rejected, drives=drives, scalar_phases=sum(x['phases'] for x in drives),
        scope='The entire selected block population has unique source target positions. Adjacent source blocks share only the context boundary token, which is supervised once. Independent source reconstruction and scalar formula comparisons agree bytewise. No training outcome is produced by this qualification.'))
    write_json(out/'manifest.json', dict(schema='onepass-stream-qualification-v1', status='complete',
        arguments=vars(a), binding_sha256=sha256(out/'binding.json'),
        results_sha256=sha256(out/'results.json')))
    print('Qualified 4194304 nonrepeated blocks and',sum(x['phases'] for x in drives),'scalar phases',flush=True)


if __name__ == '__main__':
    main()
