"""Separate spatial covariance from the native generated law's color-mean field.

All frozen generation cells are required. This changes neither model selection
nor sampling; it is an exact decomposition of completed configuration arrays.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
from model_rg.provenance import sha256, write_json


def centered_observables(configurations, q):
    x = np.asarray(configurations)
    L = x.shape[-1]
    x = x.reshape(-1, L, L)
    profile = np.stack([(x == a).mean(0) for a in range(q)], -1)
    mean = profile.mean((0, 1))
    contrasts = profile - 1/q
    global_sector = q/(q-1)*np.square(mean-1/q).sum()
    inhomogeneity = q/(q-1)*np.square(profile-mean).sum(-1).mean()
    fractions = np.stack([(x == a).mean((1, 2)) for a in range(q)], -1)
    chi = L*L*q/(q-1)*np.square(fractions-1/q).sum(-1).mean()
    connected_chi = L*L*q/(q-1)*np.square(fractions-fractions.mean(0)).sum(-1).mean()
    np.testing.assert_allclose(chi, connected_chi + L*L*global_sector, rtol=1e-12, atol=1e-12)
    pairs = {}
    for r in sorted(set([1, max(1, L//4), max(1, L//2)])):
        sectors = []
        raw = []
        direct = []
        for axis in [0, 1]:
            shifted_profile = np.roll(profile, r, axis=axis)
            sectors.append(q/(q-1)*(contrasts*np.roll(contrasts, r, axis=axis)).sum(-1).mean())
            joint_equal = (x == np.roll(x, r, axis=axis+1)).mean(0)
            raw.append((q*joint_equal.mean()-1)/(q-1))
            direct.append(q/(q-1)*(joint_equal-(profile*shifted_profile).sum(-1)).mean())
        raw_value = float(np.mean(raw)); mean_sector = float(np.mean(sectors)); connected = float(np.mean(direct))
        np.testing.assert_allclose(raw_value, connected+mean_sector, rtol=1e-12, atol=1e-12)
        pairs[str(r)] = dict(raw=raw_value, connected=connected, mean_sector=mean_sector)
    return dict(samples=len(x), color_mean=mean.tolist(), global_mean_sector=float(global_sector),
                position_mean_inhomogeneity=float(inhomogeneity), chi=float(chi), connected_chi=float(connected_chi), correlations=pairs)


def main(args):
    study = Path(args.study)
    manifest = study/'data/physical.json'
    cells = {c['id']:c for c in json.loads(manifest.read_text())['cells']}
    checked = {str(manifest):sha256(manifest)}
    rows = []; source = {}
    for name in ['base5', 'physical-s0', 'physical-s1']:
        for kind in ['critical', 'thermal']:
            root = study/'assessment'/name/kind
            result = json.loads((root/'result.json').read_text())
            if result['status'] != 'complete' or len(result['rows']) != (14 if kind == 'critical' else 32):
                raise ValueError('Complete frozen generation panel required')
            checked[str(root/'result.json')] = sha256(root/'result.json')
            for row in result['rows']:
                cell = cells[row['cell']]
                if cell['id'] not in source:
                    path = Path(cell['path'])
                    if sha256(path) != cell['sha256']:raise ValueError('Changed physical source')
                    checked[str(path)] = cell['sha256']
                    x = np.asarray(np.memmap(path, mode='r', dtype=np.uint8, shape=tuple(cell['shape'])))
                    source[cell['id']] = centered_observables(x, cell['q'])
                path = root/row['path']
                if sha256(path) != row['sha256']:raise ValueError('Changed native generation')
                checked[str(path)] = row['sha256']
                model = centered_observables(np.load(path), cell['q']); ref = source[cell['id']]
                rows.append(dict(name=name, cell=cell['id'], q=cell['q'], L=cell['L'], ratio=cell['temperature_ratio'],
                    model=model, source=ref, correlation_errors={r:dict(raw=v['raw']-ref['correlations'][r]['raw'],
                    connected=v['connected']-ref['correlations'][r]['connected']) for r,v in model['correlations'].items()}))
    write_json(study/'analysis/spatial-centering.json', dict(status='complete', analyzer_sha256=sha256(__file__),
        checked_sha256=checked, cells=rows,
        scope='Exact empirical covariance/mean-field decomposition in every frozen generation cell. Spatial centering retains each position-specific color mean. Point diagnostics describe finite samples; no population confidence interval or critical exponent is inferred here.'))
    print('Decomposed all', len(rows), 'native generated-law cells.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--study',required=True);main(parser.parse_args())
