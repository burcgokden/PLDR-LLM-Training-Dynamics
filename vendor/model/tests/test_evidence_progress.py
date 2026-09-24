import hashlib
import io
import json
from pathlib import Path

import pytest

from model_rg.evidence_progress import HashProgress, SourceArchive
from scripts.verify_scheduled_study import Evidence


def test_digest_progress_cache_and_mutation(tmp_path):
    data=tmp_path/'data';data.write_bytes(b'abc'*(4<<20))
    progress=HashProgress(tmp_path/'progress.json',stream=io.StringIO(),interval=0)
    evidence=Evidence(tmp_path,progress=progress)
    expected=hashlib.sha256(data.read_bytes()).hexdigest()
    assert evidence.check(data,expected)==expected
    assert evidence.check(data,expected)==expected
    assert progress.bytes_read==data.stat().st_size and progress.cache_hits==1
    assert len(progress.files_completed)==1
    data.write_bytes(b'mutation')
    with pytest.raises(AssertionError):evidence.check(data,expected)
    progress.finish('incomplete')
    status=json.loads((tmp_path/'progress.json').read_text())
    assert status['terminal'] and status['status']=='incomplete'


def test_stale_dependency_and_explicit_source_archive(tmp_path):
    repo=tmp_path/'repo';(repo/'src').mkdir(parents=True)
    source=repo/'src/entry.py';source.write_text('current')
    archive=tmp_path/'archive';(archive/'src').mkdir(parents=True)
    old=archive/'src/entry.py';old.write_text('executed')
    expected=hashlib.sha256(old.read_bytes()).hexdigest()
    (archive/'manifest.json').write_text(json.dumps(dict(commit='abc',files={
        'src/entry.py':dict(sha256=expected,git_blob='def')})))
    evidence=Evidence(repo)
    with pytest.raises(AssertionError):evidence.check(source,expected)
    evidence=Evidence(repo,source_archive=archive)
    assert evidence.check(source,expected)==expected
    assert evidence.archive.resolutions['src/entry.py']['path']==str(old)
    old.write_text('corrupted')
    with pytest.raises(AssertionError):evidence.check(source,expected)
    outside=tmp_path/'raw-array';outside.write_text('raw')
    with pytest.raises((AssertionError,ValueError)):evidence.check(outside,expected)


def test_incomplete_terminal_record(tmp_path):
    progress=HashProgress(tmp_path/'progress.json',stream=io.StringIO())
    progress.boundary('selected-inputs')
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        progress.close_incomplete()
    result=json.loads((tmp_path/'progress.json').read_text())
    assert result['status']=='incomplete' and result['component']=='selected-inputs'
    assert result['terminal']
