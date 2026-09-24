"""Streaming integrity progress and explicit, hash-bound historical source lookup."""
import atexit
import hashlib
import json
from pathlib import Path
import sys
import time

from model_rg.provenance import write_json


class HashProgress:
    def __init__(self, path=None, *, stream=None, interval=5.0):
        self.path = Path(path) if path else None
        self.stream = sys.stderr if stream is None else stream
        self.interval = interval; self.started = time.monotonic(); self.last = -float('inf')
        self.component = 'initialization'; self.bytes_read = 0; self.files_completed = set()
        self.cache_hits = 0; self.current_file = None; self.current_file_bytes = 0
        self.current_file_size = None; self.finished = False
        self.emit('running', force=True)
        atexit.register(self.close_incomplete)

    def emit(self, status='running', *, force=False, operation='hashing'):
        now = time.monotonic()
        if not force and now-self.last < self.interval: return
        self.last = now
        record = dict(schema='evidence-progress-v1',status=status,operation=operation,
            terminal=status!='running',elapsed_seconds=now-self.started,component=self.component,
            unique_files_completed=len(self.files_completed),bytes_read=self.bytes_read,
            cache_hits=self.cache_hits,current_file=self.current_file,
            current_file_bytes=self.current_file_bytes,current_file_size=self.current_file_size)
        print(json.dumps(record,sort_keys=True),file=self.stream,flush=True)
        if self.path: write_json(self.path,record)

    def boundary(self, component):
        self.component=str(component); self.emit(force=True,operation='semantic-check')

    def digest(self, path):
        path=Path(path).resolve();before=path.stat()
        self.current_file=str(path);self.current_file_size=before.st_size;self.current_file_bytes=0
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8<<20),b''):
                digest.update(chunk);self.bytes_read+=len(chunk);self.current_file_bytes+=len(chunk)
                self.emit()
        after=path.stat()
        if (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):
            raise AssertionError('File changed during integrity read: '+str(path))
        self.files_completed.add(str(path));self.emit()
        return digest.hexdigest()

    def cache_hit(self):
        self.cache_hits+=1;self.emit(operation='cache-hit')

    def finish(self, status='passed'):
        if self.finished:return
        self.finished=True;self.emit(status,force=True,operation='terminal')
        atexit.unregister(self.close_incomplete)

    def close_incomplete(self):
        if not self.finished:self.finish('incomplete')


class SourceArchive:
    """Only declared project-source bytes can resolve a historical dependency.

    Raw data and arbitrary filesystem paths never receive a fallback. The
    manifest is supplied explicitly; current-code qualification remains separate.
    """
    def __init__(self, root, repo):
        self.root=Path(root).resolve();self.repo=Path(repo).resolve()
        self.manifest=json.loads((self.root/'manifest.json').read_text())
        self.resolutions={}

    def resolve(self, path, expected):
        relative=str(Path(path).resolve().relative_to(self.repo))
        record=self.manifest['files'].get(relative)
        if record is None or record['sha256']!=expected:
            raise AssertionError('Unbound historical project dependency: '+relative)
        target=(self.root/relative).resolve()
        if not target.is_relative_to(self.root):raise AssertionError('Source archive path escape')
        self.resolutions[relative]=dict(path=str(target),sha256=expected,
            commit=self.manifest['commit'],git_blob=record['git_blob'])
        return target
