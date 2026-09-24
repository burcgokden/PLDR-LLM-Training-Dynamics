"""Finite within-chain and between-chain diagnostics for the physical source archive.

These diagnostics do not prove equilibration or independence of configurations.
Statistical intervals continue to resample complete simulation chains.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from model_rg.provenance import sha256, write_json


def diagnostics(values):
    x = np.asarray(values, dtype=np.float64)
    half = x.shape[1] // 2
    split = np.concatenate([x[:, :half], x[:, -half:]], axis=0)
    within = np.var(split, axis=1, ddof=1).mean()
    between = half * np.var(split.mean(axis=1), ddof=1)
    ratio = np.sqrt(((half-1)*within + between)/(half*within)) if within > 0 else None
    centered = x - x.mean(axis=1, keepdims=True)
    correlations = {}
    for lag in [1, 5]:
        left, right = centered[:, :-lag], centered[:, lag:]
        denominator = np.sqrt(np.square(left).sum()*np.square(right).sum())
        correlations[str(lag)] = float((left*right).sum()/denominator) if denominator > 0 else None
    return dict(mean=float(x.mean()), chain_mean_standard_error=float(np.std(x.mean(1), ddof=1)/np.sqrt(len(x))),
                classical_split_rhat=float(ratio) if ratio is not None else None,
                within_chain_lag_correlation=correlations, zero_within_variance=bool(within == 0))


def main(args):
    study = Path(args.study)
    manifest = study/'data/physical.json'
    data = json.loads(manifest.read_text())
    checked = {str(manifest): sha256(manifest)}
    rows = []
    for cell in data['cells']:
        path = Path(cell['path'])
        if sha256(path) != cell['sha256']:
            raise ValueError('Changed source configurations: '+str(path))
        checked[str(path)] = cell['sha256']
        x = np.asarray(np.memmap(path, mode='r', dtype=np.uint8, shape=tuple(cell['shape'])))
        q, L = cell['q'], cell['L']
        fractions = np.stack([(x == color).mean((-1, -2)) for color in range(q)], -1)
        m2 = (q*np.square(fractions).sum(-1)-1)/(q-1)
        energy = -sum((x == np.roll(x, 1, axis=axis)).mean((-1, -2)) for axis in [-1, -2])
        fields = dict(energy_density=energy, magnetic_second_moment=m2, color0_fraction=fractions[..., 0])
        rows.append(dict(cell=cell['id'], split=cell['split'], q=q, L=L, ratio=cell['temperature_ratio'],
                         chains=x.shape[0], retained_per_chain=x.shape[1], diagnostics={k: diagnostics(v) for k, v in fields.items()}))
    write_json(study/'analysis/source-chains.json', dict(status='complete', producer_sha256=sha256(__file__),
        checked_sha256=checked, cells=rows,
        scope='Classical split-chain variance ratios and pooled within-chain lag correlations on every source cell. These finite diagnostics do not certify equilibration or independent configurations; uncertainty uses complete simulation chains.'))
    for split in ['train', 'validation', 'test']:
        values = [d for row in rows if row['split'] == split for d in row['diagnostics'].values()]
        defined = [d['classical_split_rhat'] for d in values if d['classical_split_rhat'] is not None]
        print(split, 'largest classical split R-hat', max(defined), 'constant cells/fields', sum(d['zero_within_variance'] for d in values))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', required=True)
    main(parser.parse_args())
