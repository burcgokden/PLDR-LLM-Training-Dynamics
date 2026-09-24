#!/usr/bin/env python
"""Separate common-centroid radius and direction using the saved complete cohort."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from model_rg.common_geometry import radial_directional
from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


def main():
    p = argparse.ArgumentParser();p.add_argument('--root', required=True)
    p.add_argument('--study', default='critical-scaling-20260906')
    p.add_argument('--collectives', default='analysis/collectives');p.add_argument('--output', required=True)
    p.add_argument('--wait', action='store_true');a = p.parse_args()
    study = Path(a.root)/a.study;parent = study/a.collectives
    while not (parent/'manifest.json').exists():
        if not a.wait:raise RuntimeError('The declared complete collective snapshot is required')
        time.sleep(30)
    meta = json.loads((parent/'manifest.json').read_text())
    if meta['status'] != 'complete' or sha256(parent/'results.json') != meta['results_sha256']:
        raise AssertionError('The collective snapshot is incomplete or changed')
    report = json.loads((parent/'results.json').read_text())
    selected = sorted({Path(path) for row in report['conditions'] for path in row['sources']})
    inputs = [parent/'manifest.json', parent/'results.json']
    for path in selected:inputs.extend([path, path.parent/'measurements.npz'])
    out = Path(a.output);out.mkdir(parents=True, exist_ok=False);bind_run(out, inputs, vars(a))
    manifests = {};cache = {}
    for path in selected:
        m = json.loads(path.read_text())
        if m['status'] != 'complete' or sha256(path.parent/'measurements.npz') != m['raw_sha256']:
            raise AssertionError('A selected centroid measurement changed')
        manifests[str(path)] = m
    conditions = []
    for row in report['conditions']:
        points = []
        for path, seed in zip(row['sources'], row['seeds'], strict=True):
            m = manifests[path]
            if (m['condition']['heads'], m['condition']['seed'], m['case']['step']) != (row['heads'], seed, row['step']):
                raise AssertionError('Centroid observation identity changed')
            key = (path, row['precision'])
            if key not in cache:
                with np.load(Path(path).parent/'measurements.npz') as raw:
                    value = raw[row['precision']+'_centroids']
                    if value.shape != (512, 5, row['heads'], 64):
                        raise AssertionError('Fixed evaluation context or head dimensions changed')
                    cache[key] = value.mean(2)
            points.append(cache[key])
        geometry = radial_directional(np.stack(points), row['heads'])
        np.testing.assert_allclose(geometry['total_susceptibility'],
            row['observables']['common_centroid']['susceptibility'], rtol=2e-10, atol=1e-15)
        np.testing.assert_allclose(geometry['mean_radius_squared'],
            row['observables']['head_common_energy']['mean'], rtol=2e-10, atol=1e-15)
        condition = {key:row[key] for key in ['heads','step','multiplier','shared_seed','stream_seed',
                                             'precision','seed_count','seeds','sources']}
        condition['geometry'] = geometry;conditions.append(condition)
        print('Common-centroid geometry', row['heads'], row['step'], row['precision'], row['seed_count'], flush=True)
    result = dict(schema='common-centroid-radial-directional-v1', status='complete', conditions=conditions,
        parent_collectives=str(parent/'results.json'), parent_collectives_sha256=meta['results_sha256'],
        scope='An exact nonnegative pair-distance split in the same native centroid coordinates, on all512 fixed contexts and five decoders. It uses the saved states and adds no training or batch replicas. Directional variation is not identified with a gauge mode, a physical order parameter or a critical mode without an additional emission/response argument. Leave-one-initialization-out diagnostics have no population-coverage guarantee.')
    write_json(out/'results.json', result)
    write_json(out/'manifest.json', dict(status='complete', binding_sha256=sha256(out/'binding.json'),
        results_sha256=sha256(out/'results.json'), selected_conditions=len(conditions)))
    print('Complete common-centroid geometry:',len(conditions),'conditions',flush=True)


if __name__ == '__main__':main()
