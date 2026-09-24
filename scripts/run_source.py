#!/usr/bin/env python3
"""Run a bundled scientific producer or reducer with portable import paths."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root', type=Path, help='Extracted evidence root or prepared experiment workspace')
    ap.add_argument('family', choices=['row', 'rg', 'model'])
    ap.add_argument('script', help='Path relative to the selected vendor directory')
    ap.add_argument('arguments', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    base = ROOT / 'vendor' / a.family
    script = (base / a.script).resolve()
    if not script.is_relative_to(base) or not script.is_file() or script.suffix != '.py':
        ap.error('Select an existing bundled Python script')
    sys.path.insert(0, str(ROOT))
    from companion_paths import child_environment
    env = child_environment(a.family)
    if a.data_root:
        env['PLDR_DATA_ROOT'] = str(a.data_root.resolve())
        key = {'row':'PLDR_ROW_DATA_ROOT','rg':'PLDR_RG_DATA_ROOT','model':'MODEL_RG_DATA_ROOT'}[a.family]
        env[key] = str(a.data_root.resolve() / a.family / 'research')
    args = a.arguments[1:] if a.arguments[:1] == ['--'] else a.arguments
    return subprocess.call([sys.executable, '-B', str(script), *args], cwd=base, env=env)


if __name__ == '__main__':
    raise SystemExit(main())
