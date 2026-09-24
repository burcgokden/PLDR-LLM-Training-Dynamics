"""Explicit path configuration for the bundled scientific source collections.

Historical paths are acquisition identifiers. Runtime use is redirected to
this checkout and to a caller-supplied data root, never to a source project.
"""
from pathlib import Path
import os
import shutil

CODE_ROOT = Path(__file__).resolve().parent


def data_root(family):
    specific = os.environ.get({'row': 'PLDR_ROW_DATA_ROOT', 'rg': 'PLDR_RG_DATA_ROOT',
                               'model': 'MODEL_RG_DATA_ROOT'}[family])
    if specific:
        return Path(specific).expanduser().resolve()
    base = Path(os.environ.get('PLDR_DATA_ROOT', CODE_ROOT / 'build' / 'evidence'))
    return base.expanduser().resolve() / family / 'research'


def legacy_path(value):
    """Resolve one historical runtime path without changing evidence bytes."""
    maps = {
        '/pldr-data/row': data_root('row'),
        '/pldr-data/rg': data_root('rg'),
        '/pldr-data/model': data_root('model'),
        '/pldr-code/row': CODE_ROOT / 'vendor/row',
        '/pldr-code/rg': CODE_ROOT / 'vendor/rg',
        '/pldr-code/model': CODE_ROOT / 'vendor/model',
        '/pldr-code/row': CODE_ROOT / 'vendor/row',
        '/pldr-code': CODE_ROOT / 'vendor',
        '/pldr-assets/refinedweb': Path(os.environ.get('PLDR_REFINEDWEB_ROOT', CODE_ROOT / 'build/refinedweb')),
        '/pldr-tools/elan': Path(os.environ.get('ELAN_HOME', Path.home() / '.elan')),
    }
    for old, new in maps.items():
        if value == old or value.startswith(old + '/'):
            return str(new) + value[len(old):]
    raise ValueError('Unmapped source-workspace path: ' + value)


def child_environment(family, **overrides):
    """Keep the combined imports, caller configuration, and active guard in children."""
    env = dict(os.environ)
    env.update(overrides)
    base = CODE_ROOT / 'vendor' / family
    paths = [env.get('PLDR_READ_GUARD'), str(CODE_ROOT), str(base / 'src'),
             str(base / 'scripts'), str(base), str(base / 'experiments')]
    paths.extend(env.get('PYTHONPATH', '').split(os.pathsep))
    env['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(x for x in paths if x))
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env


def child_pythonpath(family):
    return child_environment(family)['PYTHONPATH']


def dispatch_worker(command, **kwargs):
    """Production subprocess boundary shared by acquisitions and bounded validation."""
    import subprocess
    return subprocess.run(command, **kwargs)


def validate_worker_cli(parent, target=None, worker_action=True):
    """Exercise the real child imports/parser, without assets or scientific work."""
    import argparse
    import json
    import subprocess
    import sys
    if '--validate-worker' not in sys.argv:
        return
    parser = argparse.ArgumentParser(description='Bounded production worker dispatch')
    parser.add_argument('--validate-worker', action='store_true', required=True)
    parser.parse_args()
    target = Path(target or parent).resolve()
    command = [sys.executable, '-B', str(target)]
    if worker_action:
        command.append('worker')
    command.append('--help')
    result = dispatch_worker(command, env=child_environment('model',
        CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'),
        cwd=target.parents[1], timeout=45, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    print(result.stdout, end='')
    if result.returncode or 'usage:' not in result.stdout.lower():
        raise SystemExit(result.returncode or 1)
    print(json.dumps(dict(status='passed', worker=target.name,
        boundary='production dispatcher, real child imports and argument parser',
        scientific_updates=0, acquisition_calls=0,
        guard_preserved=bool(os.environ.get('PLDR_READ_GUARD')))))
    raise SystemExit(0)


def resolve_row_arguments(args, parser, *, defaults, identities=None):
    """Select large row assets only after parsing; dry-run never opens them."""
    import json
    settings = {'tokens': 'PLDR_ROW_TOKENS', 'tokenizer': 'PLDR_ROW_TOKENIZER',
                'run_root': 'PLDR_ROW_RUN_ROOT'}
    inputs = {}
    for key, env_key in settings.items():
        if not hasattr(args, key):
            continue
        value = getattr(args, key) or os.environ.get(env_key) or defaults.get(key)
        if value is None:
            parser.error('Supply --' + key.replace('_', '-') + ' or ' + env_key)
        path = Path(value).expanduser().resolve()
        setattr(args, key, path)
        inputs[key] = dict(path=str(path), exists=path.exists(), environment=env_key,
                           expected_sha256=(identities or {}).get(key))
    args.output = Path(args.output).expanduser().resolve()
    if args.dry_run:
        print(json.dumps(dict(status='dry-run', inputs=inputs, output=str(args.output),
            action='check' if args.check else 'stage', scientific_updates=0,
            identity_policy='Execution retains the frozen protocol hashes, ranges, and schema; new tokenization is a different asset.'), indent=2))
        return False
    for key, record in inputs.items():
        path = getattr(args, key)
        if not path.exists():
            parser.error('Missing ' + key + ': ' + str(path) + '; supply --' +
                         key.replace('_', '-') + ' or ' + record['environment'])
        if args.output == path or args.output.is_relative_to(path):
            parser.error('Output must be separate from the read-only input: ' + str(path))
    if args.output == CODE_ROOT or args.output.is_relative_to(CODE_ROOT / 'vendor'):
        parser.error('Output must be outside the bundled source collections')
    return True
