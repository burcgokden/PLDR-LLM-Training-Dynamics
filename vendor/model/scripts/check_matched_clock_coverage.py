#!/usr/bin/env python3
"""Exercise real publication CLIs with invalid grids in both Python modes."""
from companion_paths import child_pythonpath
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from model_rg.provenance import sha256, write_json
from matched_clock_coverage import read, validate_certificate

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tests'))
from test_matched_clock_coverage import mutations


def check(study, analysis, verification, output):
    a, v = read(analysis), read(verification)
    validate_certificate(v, a, analysis)
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=child_pythonpath("model"), PYTHONDONTWRITEBYTECODE='1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='',
               MPLCONFIGDIR=str(output / 'matplotlib'))
    variants = []
    for name, mutate in mutations():
        changed = copy.deepcopy(a)
        mutate(changed)
        variants.append((name, json.dumps(changed), None))
    raw = analysis.read_text()
    variants.extend([('duplicate_json_key', raw.replace('"status":', '"status":"failed","status":', 1), None),
                     ('overflow_json_number', raw.replace('"worker_seconds":', '"overflow":1e999,"worker_seconds":', 1), None)])
    for key in ['coverage', 'coverage_sha256', 'coverage_sources_sha256']:
        bad = copy.deepcopy(v)
        bad.pop(key)
        variants.append(('missing_certificate_' + key, raw, bad))
    records = []
    for name, raw, altered_certificate in variants:
        folder = output / name
        folder.mkdir()
        ap, vp = folder / 'matched-clock-analysis.json', folder / 'matched-clock-verification.json'
        ap.write_text(raw)
        cert = copy.deepcopy(v) if altered_certificate is None else altered_certificate
        cert['analysis_sha256'] = sha256(ap)
        write_json(vp, cert)
        # This provisional record only routes the real builder to the evidence.
        # Rejection must occur at the coverage guard, before any later gate.
        proof = folder / 'publication.json'
        write_json(proof, dict(status='passed', schema='compact-publication-v1', records_root=str(folder)))
        for optimized in [False, True]:
            prefix = [sys.executable, *(['-O'] if optimized else []), '-B']
            for boundary in ['verify', 'render', 'publication', 'build']:
                if altered_certificate is not None and boundary == 'verify':
                    continue  # verifier produces a certificate; it does not consume one
                destination = folder / (boundary + ('-optimized' if optimized else '-ordinary'))
                if boundary == 'verify':
                    script, args = 'verify_matched_clock.py', ['--study', str(study), '--analysis', str(ap)]
                elif boundary == 'render':
                    script, args = 'render_matched_clock.py', ['--analysis', str(ap), '--verification', str(vp)]
                elif boundary == 'publication':
                    script, args = 'verify_compact_release.py', ['--records', str(folder)]
                else:
                    script, args = 'build_compact_release.py', ['--verification', str(proof)]
                    destination = REPO.parent / 'paper-outputs' / ('.coverage-reject-' + output.name + '-' + name + ('-O' if optimized else '-N'))
                if destination.exists():
                    raise FileExistsError(destination)
                cmd = prefix + [str(REPO / 'scripts' / script), *args, '--output', str(destination)]
                start = time.monotonic()
                result = subprocess.run(cmd, cwd=REPO, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                log = folder / (boundary + ('-O' if optimized else '-N') + '.log')
                log.write_text(result.stdout)
                passed = result.returncode != 0 and not destination.exists() and (
                    'Matched-clock coverage:' in result.stdout or 'Nonfinite numerical scalar' in result.stdout)
                records.append(dict(name=name, boundary=boundary, optimized=optimized, command=cmd,
                    returncode=result.returncode, no_output=not destination.exists(), passed=passed,
                    seconds=time.monotonic()-start, log_sha256=sha256(log)))
                if not passed:
                    write_json(output / 'failed.json', records[-1])
                    raise RuntimeError('Unexpected CLI outcome: ' + name + '/' + boundary)
        print(name, 'rejected', flush=True)
    write_json(output / 'verification.json', dict(status='passed', schema='matched-clock-cli-coverage-v1',
        case_count=len(records), cases=records, checker_sha256=sha256(__file__),
        input_sha256={str(analysis):sha256(analysis), str(verification):sha256(verification)},
        tested_sources={str(p.relative_to(REPO)):sha256(p) for p in [
            REPO / 'scripts' / n for n in ['matched_clock_coverage.py', 'verify_matched_clock.py',
            'render_matched_clock.py', 'verify_compact_release.py', 'build_compact_release.py']] +
            [REPO / 'tests/test_matched_clock_coverage.py']},
        scope='Actual verifier, renderer and compact publication boundaries; ordinary and optimized Python. Invalid fixtures are not scientific observations.'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ['study', 'analysis', 'verification', 'output']:
        ap.add_argument('--' + key, type=Path, required=True)
    check(**{k:v.resolve() for k,v in vars(ap.parse_args()).items()})
