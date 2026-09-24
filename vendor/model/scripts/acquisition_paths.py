"""Exact acquisition identities resolved through an explicit, hash-checked manifest.

The manifest maps files, not prefixes. Missing identities, duplicate identities,
duplicate destinations, changed bytes, and symlinks fail closed. The caller supplies the logical acquisition root explicitly. Immutable
protocols and their parent/input SHA256 fields are never rewritten.
"""
import hashlib
import json
import os
from pathlib import Path

HISTORICAL_ROOT = Path('/pldr-data/model')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('Duplicate input-manifest key: ' + key)
        result[key] = value
    return result


def manifest():
    name = os.environ.get('PLDR_INPUT_MANIFEST')
    if not name:
        return None
    data = json.loads(Path(name).read_text(), object_pairs_hook=_pairs)
    root = Path(data.get('logical_root', ''))
    if (data.get('schema') != 'pldr-acquisition-locations-v1'
            or not root.is_absolute() or '..' in root.parts
            or str(root) != data.get('logical_root')):
        raise ValueError('Unsupported acquisition location manifest')
    seen = set(); destinations = set()
    for row in data['files']:
        if set(row) != {'identity', 'path', 'sha256'}:
            raise ValueError('Incomplete acquisition alias')
        identity = Path(row['identity']); path = Path(row['path'])
        if (not identity.is_absolute() or str(identity) != row['identity'] or '..' in identity.parts
                or not identity.is_relative_to(root)):
            raise ValueError('Invalid acquisition identity')
        if (not path.is_absolute() or path.resolve() != path.absolute()
                or not path.is_file() or path.is_symlink()):
            raise ValueError('Alias must name an existing regular file without symlink components')
        if row['identity'] in seen or str(path) in destinations:
            raise ValueError('Ambiguous acquisition alias')
        if len(row['sha256']) != 64 or any(c not in '0123456789abcdef' for c in row['sha256']):
            raise ValueError('Invalid alias digest')
        seen.add(row['identity']); destinations.add(str(path))
    return data


def logical_root(runtime_root):
    data = manifest()
    return Path(data['logical_root']) if data is not None else Path(runtime_root)


def resolve_recorded(path, expected_sha256=None):
    path = Path(path)
    data = manifest()
    if data is None or not path.is_relative_to(Path(data['logical_root'])):
        physical = path
    else:
        matches = [r for r in data['files'] if r['identity'] == str(path)]
        if len(matches) != 1:
            raise ValueError('Missing or ambiguous acquisition alias: ' + str(path))
        row = matches[0]
        if expected_sha256 is not None and row['sha256'] != expected_sha256:
            raise ValueError('Alias does not bind expected acquisition bytes: ' + str(path))
        physical = Path(row['path'])
        if sha(physical) != row['sha256']:
            raise ValueError('Changed relocated input: ' + str(path))
    if expected_sha256 is not None and sha(physical) != expected_sha256:
        raise ValueError('Changed acquisition input: ' + str(path))
    return physical


def recorded_study(study):
    data = manifest()
    if data is None:
        return str(Path(study).resolve())
    row = data.get('study', {})
    if row.get('path') != str(Path(study).resolve()):
        raise ValueError('Unmapped relocated study')
    identity = row.get('identity')
    if not isinstance(identity, str) or Path(identity).parent != Path(data['logical_root']):
        raise ValueError('Invalid recorded study identity')
    # The immutable protocol, rather than its directory basename, binds the study.
    if resolve_recorded(Path(identity)/'protocol.json') != Path(study).resolve()/'protocol.json':
        raise ValueError('Study protocol alias differs')
    return identity
