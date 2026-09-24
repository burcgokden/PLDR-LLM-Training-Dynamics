"""Intercept CLI dispatch before any worker, queue, or prepare body executes."""
from companion_paths import legacy_path
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys

p = argparse.ArgumentParser()
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
sys.path[:0] = [str(a.repo/'src'), str(a.repo/'scripts')]
script = a.repo/'scripts/run_potential_factorial.py'
study = Path(legacy_path('/pldr-data/model/potential-factorial-disjoint-20260914'))
class StopDispatch(Exception):
    pass

records = []
for action in ['prepare', 'worker', 'profile', 'run']:
    for role in [None, 'scientific', 'qualification']:
        calls = []
        def intercept(frame, event, arg):
            if event == 'call' and frame.f_code.co_filename == str(script) and frame.f_code.co_name in ['run_case', 'queue', 'prepare']:
                calls.append(dict(function=frame.f_code.co_name, arguments={k: str(v) if isinstance(v, Path) else v for k, v in frame.f_locals.items()}))
                raise StopDispatch
            return intercept
        argv = [str(script), action, '--study', str(study)]
        if action == 'worker':
            argv += ['--case', 'early-h2']
        if role is not None:
            argv += ['--role', role]
        sys.argv = argv
        err = io.StringIO()
        code = 0
        sys.settrace(intercept)
        try:
            with contextlib.redirect_stderr(err):
                runpy.run_path(str(script), run_name='__main__')
        except StopDispatch:
            pass
        except SystemExit as exc:
            code = exc.code
        finally:
            sys.settrace(None)
        expected_rejection = action != 'prepare' and role is not None
        if expected_rejection:
            if code != 2 or calls:
                raise RuntimeError(f'Execution role was not rejected before dispatch: {argv}')
        elif code != 0 or len(calls) != 1:
            raise RuntimeError(f'Expected valid dispatch: {argv}')
        records.append(dict(action=action, requested_role=role, exit_code=code, calls=calls, stderr=err.getvalue()))

result = dict(status='passed', records=records,
              checked_cases=len(records), native_updates=0, stopped_before_native_work=True,
              python_optimized=not __debug__, script_sha256=hashlib.sha256(script.read_bytes()).hexdigest())
with a.output.open('x') as f:
    json.dump(result, f, indent=2)
    f.write('\n')
print(result['status'], len(records), 'CLI cases; zero native updates')
