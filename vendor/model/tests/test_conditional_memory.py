"""Boundary checks for the launcher and finite conditional prediction algebra."""
import argparse
import json
from pathlib import Path
import runpy
import shlex
import subprocess
import sys

import numpy as np

from scripts.conditional_memory_study import fit_predictors, predictions
from scripts.run_training_study import main as launcher_main
from scripts.verify_training import main as verifier_main


def test_training_launcher_verifier_parser(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    assert launcher_main.__module__ == 'scripts.run_training_study'
    assert verifier_main.__module__ == 'scripts.verify_training'
    root = tmp_path/'not executed study with spaces'
    output = subprocess.run([sys.executable, str(repo/'scripts/run_training_study.py'),
                             '--run-root', str(root), '--assets', '/unused/assets',
                             '--data', '/unused/data', '--print-commands'],
                            capture_output=True, text=True, check=True)
    commands = [shlex.split(line) for line in output.stdout.splitlines() if line.startswith(sys.executable+' ')]
    assert len(commands) == 16
    command = commands[-1]
    assert Path(command[1]).name == 'verify_training.py'
    parse = argparse.ArgumentParser.parse_args
    class Parsed(BaseException):
        pass
    seen = []
    def stop(self, *args, **kwargs):
        seen.append(vars(parse(self, command[2:])))
        raise Parsed
    monkeypatch.setattr(argparse.ArgumentParser, 'parse_args', stop)
    monkeypatch.syspath_prepend(str(repo/'scripts'))
    try:
        runpy.run_path(command[1], run_name='__main__')
    except Parsed:
        pass
    assert seen == [{'data_root': str(root), 'output': str(root/'training-verification.json')}]
    assert not root.exists()


def test_frozen_memory_forecast_uses_only_past():
    rng = np.random.default_rng(12813)
    design = np.column_stack([np.full(40, 3.), 3 + rng.normal(size=(40, 4))])
    fit = fit_predictors(design)
    frozen = json.dumps(fit, sort_keys=True)
    future = np.column_stack([np.full(11, 3.), 3 + rng.normal(size=(11, 4))])
    before = predictions(fit, future)
    future[:, -1] += 1000
    after = predictions(fit, future)
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])
    assert json.dumps(fit, sort_keys=True) == frozen
    # Independent augmented least squares reproduces the declared ridge units.
    x = (design[:, 1:4] - design[:, 1:4].mean(0))/np.array(fit['history_scale'])
    y = design[:, -1] - design[:, -1].mean()
    beta = np.linalg.lstsq(np.vstack([x, np.sqrt(4.)*np.eye(3)]),
                          np.r_[y, np.zeros(3)], rcond=None)[0]
    np.testing.assert_allclose(beta, fit['regressions']['risk_history']['beta'], atol=1e-13)


def test_affine_memory_projection_and_covariance_blocking():
    rng = np.random.default_rng(120819)
    z = rng.normal(size=(31, 7)) @ rng.normal(size=(7, 7))
    z -= z.mean(0)
    x, y = z[:, :3], z[:, 3:]
    k = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y-x@k
    np.testing.assert_allclose(x.T@residual, 0, atol=1e-11)
    h = rng.normal(size=k.shape)
    np.testing.assert_allclose(np.mean(np.sum((y-x@h)**2, axis=1)),
                               np.mean(np.sum(residual**2, axis=1))+
                               np.mean(np.sum((x@(k-h))**2, axis=1)), rtol=1e-12)
    b = rng.normal(size=(5, 7)); d = rng.normal(size=(2, 5))
    cov = z.T@z/len(z)
    blocked = z@b.T@d.T
    np.testing.assert_allclose(d@(b@cov@b.T)@d.T, blocked.T@blocked/len(z), rtol=1e-12)
