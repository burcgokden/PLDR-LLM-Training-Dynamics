#!/usr/bin/env python
"""Run the frozen assessment after a device's training queue has completed."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', required=True)
    ap.add_argument('--heads', type=int, required=True)
    ap.add_argument('--device', required=True)
    args = ap.parse_args()
    base = Path(args.study)
    spec = json.loads((base / 'confirmation/protocol.json').read_text())
    cases = [c for c in spec['cases'] if c['heads'] == args.heads]
    started = time.monotonic()
    while not all((base / 'confirmation' / c['name'] / 'manifest.json').exists() for c in cases):
        if time.monotonic() - started > 4 * 3600:
            raise TimeoutError('Training queue did not finish within four hours')
        time.sleep(10)
    for case in cases:
        path = base / 'assessment' / case['name'] / 'manifest.json'
        if path.exists():
            if json.loads(path.read_text())['status'] != 'complete':
                raise ValueError('Incomplete assessment manifest')
            continue
        with (base / 'assessment' / (case['name'] + '.log')).open('x') as log:
            subprocess.run([sys.executable, str(Path(__file__).with_name('physical_assessment.py')),
                'worker', '--study', str(base / 'assessment'), '--name', case['name'],
                '--device', args.device], stdout=log, stderr=subprocess.STDOUT, check=True)
        print('assessment complete', case['name'], flush=True)


if __name__ == '__main__':
    main()
