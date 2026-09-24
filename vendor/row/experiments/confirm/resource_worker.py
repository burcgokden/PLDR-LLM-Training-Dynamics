"""Linux workload guardian: adopt, stop, and reap descendants before completion.

This separate interpreter keeps subreaper state local to one execution, including
when a campaign launches several executors from threads. It is not a sandbox.
"""
from __future__ import annotations
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def descendants(pid):
    found=set();pending=[pid]
    while pending:
        parent=pending.pop()
        try:
            children=Path(f'/proc/{parent}/task/{parent}/children').read_text().split()
        except (FileNotFoundError,ProcessLookupError):
            continue
        for value in children:
            child=int(value)
            if child not in found:
                found.add(child);pending.append(child)
    return found


def cleanup(seconds=1.0):
    deadline=time.monotonic()+seconds
    while True:
        for pid in descendants(os.getpid()):
            try:os.kill(pid,signal.SIGKILL)
            except ProcessLookupError:pass
        while True:
            try:pid,_=os.waitpid(-1,os.WNOHANG)
            except ChildProcessError:return True
            if pid==0:break
        if time.monotonic()>=deadline:return False
        time.sleep(.005)


def main():
    result_path=Path(sys.argv[1]);command=sys.argv[2:]
    result=dict(exit_code=None,cleanup_complete=False,surviving_descendants=[],
                detached_descendants=[],error=None)
    stopped=False
    def stop(_signum,_frame):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    try:
        # PR_SET_CHILD_SUBREAPER ensures double-forked or setsid children remain
        # waitable here when their parents exit; no process-global caller change.
        libc=ctypes.CDLL(None,use_errno=True)
        if libc.prctl(36,1,0,0,0)!=0:
            raise OSError(ctypes.get_errno(),'PR_SET_CHILD_SUBREAPER failed')
        result_path.with_suffix(".ready").touch()
        if stopped:raise RuntimeError("guardian stopped before worker launch")
        child=subprocess.Popen(command)
        result['worker_pid']=child.pid
        escaped=set()
        while child.poll() is None and not stopped:
            for pid in descendants(os.getpid()):
                try:
                    if os.getsid(pid)!=os.getsid(0):escaped.add(pid)
                except ProcessLookupError:pass
            if escaped:break
            time.sleep(.005)
        result['exit_code']=child.poll()
        # Reap the direct child before enumerating adopted descendants.
        if result['exit_code'] is not None:child.wait()
        result['surviving_descendants']=sorted(descendants(os.getpid()))
        result['detached_descendants']=sorted(escaped)
        result['interrupted']=stopped
        result['cleanup_complete']=cleanup()
        if result['exit_code'] is None:
            # Popen may no longer own the wait after cleanup; do not invent a
            # successful worker exit code.
            result['exit_code']=-signal.SIGKILL
    except Exception as error:
        result['error']=f'{type(error).__name__}: {error}'
        result['cleanup_complete']=cleanup()
    result['guardian_finished_monotonic']=time.monotonic()
    temporary=result_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,allow_nan=False)+'\n')
    temporary.replace(result_path)


if __name__=='__main__':main()
