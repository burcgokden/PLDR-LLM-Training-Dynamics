#!/usr/bin/env python
"""Run the complete numerical suite and bind its statically resolved project scope."""
from companion_paths import child_pythonpath
import argparse
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from model_rg.provenance import sha256, write_json


def main():
    p = argparse.ArgumentParser();p.add_argument('--output', required=True)
    a = p.parse_args();repo = Path(__file__).resolve().parents[1]
    out = Path(a.output).resolve();out.mkdir(parents=True, exist_ok=False)
    pending = sorted((repo/'tests').rglob('*.py'))
    pending += [repo/'scripts'/name for name in ['collect_features.py', 'measure_response.py',
                'collect_segments.py', 'validate_model.py', 'analyze.py', 'verify_artifacts.py']]

    sources = {}
    while pending:
        path = pending.pop()
        name = str(path.relative_to(repo))
        if name in sources:continue
        sources[name] = sha256(path)
        for node in ast.walk(ast.parse(path.read_text())):
            modules = ([node.module] if isinstance(node, ast.ImportFrom) and node.module else
                       [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            for module in modules:
                relative = module.replace('.', '/')
                candidates = [repo/(relative+'.py'), repo/relative/'__init__.py',
                              repo/'src'/(relative+'.py'), repo/'src'/relative/'__init__.py',
                              repo/'scripts'/(relative+'.py')]
                pending.extend(candidate for candidate in candidates if candidate.is_file())
    command = [sys.executable, '-m', 'pytest', '-q', 'tests']
    env = dict(os.environ, PYTHONPATH=child_pythonpath("model"), PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='4')
    with (out/'pytest.log').open('x') as log:
        result = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
    log = (out/'pytest.log').read_text()
    matches = re.findall(r'\b(\d+) passed\b', log)
    for name, signature in sources.items():
        if sha256(repo/name) != signature:raise AssertionError('A test dependency changed during execution: '+name)
    passed = int(matches[-1]) if matches else 0
    status = 'passed' if result.returncode == 0 and passed > 0 else 'failed'
    write_json(out/'verification.json', dict(status=status, returncode=result.returncode, command=command,
        tests_passed=passed, log=str(out/'pytest.log'), log_sha256=sha256(out/'pytest.log'),
        tested_sources=sources, checker_sha256=sha256(__file__),
        scope='Complete discovered numerical suite, with every test source and statically resolved project import bound. This is separate from full-data reconstruction and does not assert line coverage or native-training equivalence for untested code.'))
    print(log, end='')
    if status != 'passed':raise RuntimeError('The complete numerical regression did not pass')


if __name__ == '__main__':main()
