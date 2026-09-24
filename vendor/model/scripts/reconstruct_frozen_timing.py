#!/usr/bin/env python3
"""Reconstruct saved cache timing using its verified executed source snapshot."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from model_rg.provenance import sha256, write_json
from numerical_validation import load_json_strict

REPO = Path(__file__).resolve().parents[1]


def reconstruct(study, output):
    study, output = study.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(output)
    p = load_json_strict((study / 'protocol.json').read_text())
    for name, h in p['source_sha256'].items():
        path = study / 'executed-source' / name
        if not path.resolve().is_relative_to((study / 'executed-source').resolve()) or sha256(path) != h:
            raise ValueError('Frozen source mismatch: ' + name)
    output.mkdir(parents=True)
    source = output / 'source'
    shutil.copytree(study / 'executed-source', source)
    verifier = 'scripts/verify_operator_cache.py'
    shutil.copy2(REPO / verifier, source / verifier)
    write_json(output / 'source-bridge.json', dict(
        status='passed', protocol_sha256=sha256(study / 'protocol.json'),
        executed_source=str(study / 'executed-source'), source_sha256=p['source_sha256'],
        verifier_sha256=sha256(source / verifier), helper_sha256=sha256(__file__),
        current_analyzer_matches_frozen=sha256(REPO / 'scripts/analyze_operator_cache.py') == sha256(source / 'scripts/analyze_operator_cache.py'),
        scope='Saved observation reconstruction; zero native calls, zero optimizer updates.'))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(source / 'src') + os.pathsep + str(source / 'scripts'),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    records = []
    for name, script, extra in [
        ('analysis', 'analyze_operator_cache.py', []),
        ('verification', 'verify_operator_cache.py', ['--analysis', str(output / 'analysis.json')])]:
        cmd = [sys.executable, '-B', str(source / 'scripts' / script), '--study', str(study),
               *extra, '--output', str(output / (name + '.json'))]
        start = time.monotonic()
        with (output / (name + '.log')).open('x') as log:
            result = subprocess.run(cmd, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        records.append(dict(name=name, command=cmd, returncode=result.returncode, seconds=time.monotonic()-start))
        write_json(output / 'commands.json', records)
        if result.returncode:
            raise RuntimeError('Historical reconstruction failed: ' + name)
    print(dict(status='passed', output=str(output)), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--study', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    reconstruct(**vars(ap.parse_args()))
