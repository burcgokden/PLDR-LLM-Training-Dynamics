#!/usr/bin/env python
"""Enumerate each completed physical checkpoint on the CPU."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', required=True)
    args = ap.parse_args()
    base = Path(args.study)
    cases = json.loads((base / 'confirmation/protocol.json').read_text())['cases']
    for case in cases:
        path = base / 'exact-laws' / case['name'] / 'verification.json'
        if path.exists():
            if json.loads(path.read_text())['status'] != 'passed':
                raise ValueError('Exact-law verification failed')
            continue
        started = time.monotonic()
        while not (base / 'confirmation' / case['name'] / 'manifest.json').exists():
            if time.monotonic() - started > 4 * 3600:
                raise TimeoutError('Training checkpoint did not finish')
            time.sleep(10)
        with (base / ('exact-' + case['name'] + '.log')).open('x') as log:
            subprocess.run([sys.executable, str(Path(__file__).with_name('physical_exact_law.py')),
                '--study', str(base), '--case', case['name']], stdout=log,
                stderr=subprocess.STDOUT, check=True)
        print('exact-law verification complete', case['name'], flush=True)


if __name__ == '__main__':
    main()
