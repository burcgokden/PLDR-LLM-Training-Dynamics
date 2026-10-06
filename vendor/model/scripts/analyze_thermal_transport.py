"""Direct thermal-response contrasts from every L=24 cell.

Uses no project analysis functions. Independent configurations are resampled
within each checkpoint/temperature cell. Shared cells reuse bootstrap draws.
This conditions on fitted checkpoints and does not estimate training-seed error.
"""
from companion_paths import configured_path
import hashlib
import json
from pathlib import Path
import numpy as np

STUDY = Path(configured_path('data:model/released-adaptation-20260913'))
REPEATS = 16000
SEED = 2026091303

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    rng = np.random.default_rng(SEED)
    panels, bindings = {}, {}
    for name in ['base5', 'physical-s0', 'physical-s1']:
        root = STUDY / 'assessment' / name / 'thermal'
        record = json.loads((root / 'result.json').read_text())
        bindings[str(root / 'result.json')] = digest(root / 'result.json')
        for cell in record['rows']:
            if cell['L'] != 24:
                continue
            path = root / cell['path']
            bindings[str(path)] = digest(path)
            if bindings[str(path)] != cell['sha256']:
                raise ValueError('Changed generated configurations')
            x = np.load(path)
            q = cell['q']
            f = np.stack([(x == a).mean((1, 2)) for a in range(q)], axis=1)
            m2 = q / (q - 1) * np.square(f - 1 / q).sum(1)
            point = float(np.mean(m2 ** 2) / np.mean(m2) ** 2)
            draws = []
            for start in range(0, REPEATS, 512):
                indices = rng.integers(len(m2), size=(min(512, REPEATS-start), len(m2)))
                samples = m2[indices]
                draws.append(np.mean(samples ** 2, axis=1) / np.mean(samples, axis=1) ** 2)
            panels[name, q, round(cell['ratio'], 2)] = (point, np.concatenate(draws), len(x))
    rows = []
    for q in [2, 3]:
        for width in [.03, .06]:
            estimates = {}
            for name in ['base5', 'physical-s0', 'physical-s1']:
                lo = panels[name, q, round(1-width, 2)]
                hi = panels[name, q, round(1+width, 2)]
                estimates[name] = ((hi[0]-lo[0])/(2*width), (hi[1]-lo[1])/(2*width))
            for name in ['physical-s0', 'physical-s1']:
                a, b = estimates[name], estimates['base5']
                difference = a[1] - b[1]
                rows.append(dict(q=q, L=24, half_width=width, name=name,
                    adapted_contrast=a[0], base_contrast=b[0], difference=a[0]-b[0],
                    conditional_percentile95=np.quantile(difference, [.025, .975]).tolist(),
                    bonferroni8_percentile=np.quantile(difference, [.05/16, 1-.05/16]).tolist()))
    result = dict(status='complete',role='exploratory finite-law reanalysis; no new fitting or checkpoint selection',
        seed=SEED, bootstrap_repeats=REPEATS, cells=len(panels), contrasts=len(rows),
        scope=__doc__, rows=rows, inputs=bindings, producer_sha256=digest(Path(__file__)),
        coverage='Percentile bootstrap approximations; Bonferroni quantiles target a family of eight. No exact finite-sample or adaptation-seed coverage claim.')
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(rows, indent=2))

if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();STUDY=args.study.resolve();OUT=args.output.resolve();main()
