"""Atomic content-identity reservations shared by all maintained context preparers.

Old protocols are imported read-only. Calibration and assessment are both reserved.
Named-panel reproduction is explicit and does not constitute fresh acquisition.
"""
from companion_paths import configured_path
from acquisition_paths import logical_root, resolve_recorded, recorded_study
import fcntl
import json
import os
from pathlib import Path
import tempfile
from model_rg.provenance import sha256
from numerical_validation import load_json_strict

ROOT = Path(configured_path('data:model'))
PROBES = ROOT/'controlled-study-20260905/data/short'
CORPUS = ROOT/'data/refinedweb-onepass-524288'
SCHEMAS = {'context-categorical-v1', 'context-categorical-v2', 'operator-cache-v1',
           'operator-cache-v2', 'cache-risk-v1', 'cache-risk-v2', 'cache-state-transfer-v1',
           'cache-state-transfer-v2', 'cache-state-transfer-v3',
           'rebuttal-heldout-cache-pilot-v1', 'matched-clock-onepass-v1'}


def read(path):
    return load_json_strict(Path(path).read_text())


def document_hashes(protocol):
    if protocol.get('schema') not in SCHEMAS:
        raise ValueError('Unsupported context protocol schema')
    values = protocol.get('document_hashes')
    if not isinstance(values, list) or not values or len(set(values)) != len(values):
        raise ValueError('Missing, duplicated or malformed context identities')
    if any(not isinstance(x, str) or len(x) != 64 or any(c not in '0123456789abcdef' for c in x) for x in values):
        raise ValueError('Malformed document digest')
    return values


def import_panels(root):
    result = {}
    for path in sorted(root.glob('*/protocol.json')):
        p = read(path)
        if 'document_hashes' in p:
            result[str(path)] = dict(sha256=sha256(path), hashes=document_hashes(p), schema=p['schema'])
    return result


def atomic_write(path, value):
    fd, name = tempfile.mkstemp(prefix='.reservation-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
            stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        if os.path.exists(name): os.unlink(name)


def reserve(study, count, start=None, *, exclusions=(), root=ROOT, probes=PROBES,
            corpus=CORPUS, reproduce=None):
    """Reserve before creating scientific output; failed reservations remain consumed."""
    root, study = Path(root).resolve(), Path(study).resolve()
    if study == root or not study.is_relative_to(root) or study.exists():
        raise ValueError('A fresh authorized study is required')
    if type(count) is not int or count <= 0 or (start is not None and (type(start) is not int or start < 0)):
        raise ValueError('Invalid context count or start')
    records = read(probes/'records.json')
    training = {r['content_sha256'] for r in read(corpus/'records.json')}
    extra = {}
    for old in exclusions:
        path = Path(old).resolve()/'protocol.json'
        extra[str(path)] = dict(sha256=sha256(path), hashes=document_hashes(read(path)))
    with (root/'.context-reservations.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root/'context-reservations-v1.json'
        index = read(path) if path.exists() else dict(schema='context-reservations-v1', reservations={})
        if index.get('schema') != 'context-reservations-v1': raise ValueError('Unsupported reservation index')
        if str(study) in index['reservations']: raise ValueError('Study already reserved')
        panels = import_panels(root); panels.update(extra)
        used = set(training)
        used.update(r['content_sha256'] for r in records[:1152])
        for item in panels.values(): used.update(item['hashes'])
        for item in index['reservations'].values(): used.update(item['document_hashes'])
        if reproduce is not None:
            named = Path(reproduce).resolve()/'protocol.json'
            original = read(named); hashes = document_hashes(original)
            rows = original.get('context_rows')
            if len(hashes) != count or not isinstance(rows, list) or len(rows) != count:
                raise ValueError('Named panel geometry differs')
            if start is not None and rows != list(range(start, start+count)):
                raise ValueError('Named panel and requested start differ')
            if set(hashes) & training: raise ValueError('Named panel overlaps training')
        else:
            candidates = range(len(records)-count+1) if start is None else [start]
            rows = None
            for first in candidates:
                proposed = list(range(first, first+count))
                if proposed[-1] >= len(records): continue
                hs = [records[i]['content_sha256'] for i in proposed]
                if len(set(hs)) == count and not used.intersection(hs):
                    rows, hashes = proposed, hs; break
            if rows is None: raise ValueError('No unreserved contiguous panel; supply a new admitted document pool')
        if any(type(i) is not int or i < 0 or i >= len(records) for i in rows):
            raise ValueError('Panel row outside source')
        if hashes != [records[i]['content_sha256'] for i in rows]: raise ValueError('Panel content identity differs')
        receipt = dict(schema='context-reservation-receipt-v1', study=str(study),
            role='named-panel-reproduction' if reproduce is not None else 'new-acquisition',
            context_rows=rows, document_hashes=hashes,
            records_sha256=sha256(probes/'records.json'), training_records_sha256=sha256(corpus/'records.json'),
            imported_protocols={name:item['sha256'] for name,item in panels.items()},
            named_panel=None if reproduce is None else str(named),
            named_panel_sha256=None if reproduce is None else sha256(named))
        index['reservations'][str(study)] = receipt
        atomic_write(path, index)
    return receipt


def validate_receipt(study, protocol, root=ROOT):
    identity = recorded_study(study)
    receipt = read(study/'reservation.json')
    if receipt.get('schema') != 'context-reservation-receipt-v1' or receipt.get('study') != identity:
        raise ValueError('Foreign reservation receipt')
    for key in ['context_rows','document_hashes']:
        if receipt.get(key) != protocol.get(key): raise ValueError('Reservation panel differs')
    index = read(resolve_recorded(logical_root(root)/'context-reservations-v1.json'))
    if index['reservations'].get(identity) != receipt: raise ValueError('Unregistered reservation')
    if protocol.get('reservation_sha256') != sha256(study/'reservation.json'):
        raise ValueError('Changed reservation receipt')
    return receipt
