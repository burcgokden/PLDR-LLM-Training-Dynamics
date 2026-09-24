"""Exercise the complete emitted command path without launching measurements."""
import argparse
from collections import Counter
import contextlib
import io
from pathlib import Path
import runpy
import shlex
import subprocess
import sys
from unittest.mock import patch

from scripts.run_experiments import main


class Parsed(BaseException):
    pass


def parse_command(command):
    original = argparse.ArgumentParser.parse_args
    record = {}
    def stop(parser, *args, **kwargs):
        result = original(parser, *args, **kwargs)
        record.update(vars(result))
        raise Parsed()
    with patch.object(sys, 'argv', command[1:]), \
         patch.object(argparse.ArgumentParser, 'parse_args', stop), \
         contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        try:
            runpy.run_path(command[1], run_name='__main__')
        except Parsed:
            return 0, record
        except SystemExit as exc:
            return exc.code, record
    raise AssertionError('The child parser was not exercised')


def test_confirmation_launcher_complete_command_path(tmp_path):
    root = tmp_path/'paths with spaces'
    argv = ['run_experiments.py', '--run-root', str(root), '--assets', str(root/'model assets'),
            '--data', str(root/'corpus data'), '--devices', '0,1']
    output = io.StringIO()
    with patch.object(sys, 'argv', argv+['--print-commands']), contextlib.redirect_stdout(output):
        main()
    assert not root.exists()
    commands = [shlex.split(line) for line in output.getvalue().splitlines() if line.strip()]
    assert len(commands) == 10
    for command in commands:
        status, _ = parse_command(command)
        assert status == 0, command
    assert parse_command(commands[-1])[1]['output'] == str(root/'verification.json')
    executed = []
    def intercept(command, *args, **kwargs):
        executed.append(command)
        return subprocess.CompletedProcess(command, 0)
    with patch.object(sys, 'argv', argv), patch.object(subprocess, 'run', intercept):
        main()
    assert Counter(map(tuple, executed)) == Counter(map(tuple, commands))
    assert executed[-2:] == commands[-2:]
    assert parse_command(commands[-1][:-2])[0] == 2
