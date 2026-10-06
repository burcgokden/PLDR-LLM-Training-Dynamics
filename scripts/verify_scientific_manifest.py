#!/usr/bin/env python3
"""Authenticate the complete scientific payload independently of Git HEAD."""
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'scientific-manifest.json'
IGNORED = {'.git', '.lake', 'build', 'validation', '__pycache__', '.pytest_cache', '.venv'}
SUFFIXES = {'.pyc', '.olean', '.ilean'}


def canonical(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def local(root, name):
    if not isinstance(name, str):
        raise ValueError('Unsafe manifest path: expected a string')
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name or str(p) != name:
        raise ValueError('Unsafe manifest path: ' + name)
    target = Path(root).resolve().joinpath(*p.parts)
    if target.absolute() != target.resolve():
        raise ValueError('Symlink in payload: ' + name)
    return target


def payload_files(root=ROOT):
    """Enumerate reviewed source bytes, including new files, with release exclusions."""
    root = Path(root).resolve()
    files = {}
    for path in root.rglob('*'):
        relative = path.relative_to(root)
        name = relative.as_posix()
        if set(relative.parts) & IGNORED or name == MANIFEST or path.suffix in SUFFIXES:
            continue
        if path.is_symlink():
            raise ValueError('Symlink in payload: ' + name)
        if path.is_file():
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(files.items()))


def verify(root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads(local(root, MANIFEST).read_text())
    expected = manifest['files']
    for name in expected:
        local(root, name)
    if canonical(expected) != manifest['payload_sha256']:
        raise ValueError('Scientific manifest payload digest differs')
    actual = payload_files(root)
    if set(actual) != set(expected):
        raise ValueError('Scientific payload inventory differs: missing=' + repr(sorted(set(expected) - set(actual)))
                         + '; unexpected=' + repr(sorted(set(actual) - set(expected))))
    changed = [name for name in actual if actual[name] != expected[name]]
    if changed:
        raise ValueError('Scientific manifest hash differs: ' + ', '.join(changed))
    return {'status': 'passed', 'files': len(actual), 'payload_sha256': canonical(actual),
            'scope': 'Complete prepared payload bytes; base Git HEAD is not a release identity.'}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
