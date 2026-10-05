"""Read and authenticate the portable numerical evidence store (standard library)."""
import gzip,hashlib,json
from pathlib import Path,PurePosixPath

def digest(raw):return hashlib.sha256(raw).hexdigest()
def pairs(items):
    out={}
    for key,value in items:
        if key in out:raise ValueError('Duplicate JSON key: '+key)
        out[key]=value
    return out
def read_json(path):return json.loads(Path(path).read_text(),object_pairs_hook=pairs)
def materialized_bytes(root,name):
    raw=safe(root,name).read_bytes()
    if raw.startswith(b'version https://git-lfs.github.com/spec/v1\n') or raw.startswith(b'version https://git-lfs.github.com/spec/v1\r\n'):
        raise ValueError('Unresolved Git LFS pointer: '+name+'. From the dataset checkout, run git lfs install --local and git lfs pull origin at the intended revision; then run sha256sum -c SHA256SUMS. See the dataset README Access recipe. Downloading requires Git LFS; reading materialized records uses only the Python standard library.')
    return raw
def safe(root,name):
    p=PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name or str(p)!=name:
        raise ValueError('Unsafe relative identity: '+name)
    root=Path(root).resolve();target=root.joinpath(*p.parts)
    if target.resolve()!=target.absolute():raise ValueError('Symlink in evidence path: '+name)
    return target

class Evidence:
    def __init__(self,root):
        self.root=Path(root).resolve();self.index=read_json(self.root/'index.json')
        if self.index.get('schema')!='pldr-numerical-evidence-v2':raise ValueError('Unsupported evidence schema')
        self.records={}
        for r in self.index['records']:
            if r['id'] in self.records:raise ValueError('Duplicate evidence identity')
            safe(self.root,r['id']);safe(self.root,r['object']);self.records[r['id']]=r
    def read(self,identity):
        r=self.records[identity];compressed=materialized_bytes(self.root,r['object'])
        if digest(compressed)!=r['compressed_sha256'] or len(compressed)!=r['compressed_bytes']:raise ValueError('Compressed integrity failure: '+identity)
        raw=gzip.decompress(compressed)
        if digest(raw)!=r['sha256'] or len(raw)!=r['bytes']:raise ValueError('Record integrity failure: '+identity)
        return raw
    def json(self,identity):return json.loads(self.read(identity),object_pairs_hook=pairs)
    def verify(self,extract=None,prefix=''):
        manifest=read_json(self.root/'manifest.json')
        expected=set(manifest['files'])
        actual={str(p.relative_to(self.root)) for p in self.root.rglob('*') if p.is_file() and '.git' not in p.relative_to(self.root).parts and str(p.relative_to(self.root)) not in {'manifest.json','SHA256SUMS'}}
        if actual!=expected:
            missing=sorted(expected-actual);unexpected=sorted(actual-expected)
            detail='; missing='+repr(missing[:5])+'; unexpected='+repr(unexpected[:5])
            raise ValueError('Dataset file inventory differs'+detail+'. Use the dataset README Access recipe to create a complete checkout with regular files. Hub cache/snapshot layouts, .cache metadata, extracted records and validation outputs are not part of the dataset export; keep them outside its root. No extra files are silently ignored except Git metadata.')
        for name,h in manifest['files'].items():
            if digest(materialized_bytes(self.root,name))!=h:raise ValueError('Dataset file digest differs: '+name)
        total=0
        for identity in sorted(self.records):
            raw=self.read(identity);total+=len(raw)
            if identity.endswith('.json'):json.loads(raw,object_pairs_hook=pairs)
            if extract is not None and identity.startswith(prefix):
                path=safe(extract,identity)
                if path.exists() and path.read_bytes()!=raw:raise FileExistsError(path)
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        coverage=read_json(self.root/'coverage.json')
        for display in coverage['displays']:
            for name in display['records']:
                if name not in self.records:raise ValueError('Unresolved display evidence: '+name)
            if not display['disposition']:raise ValueError('Unspecified display coverage')
        if len({r['id'] for r in coverage['displays']})!=len(coverage['displays']):raise ValueError('Duplicate display identity')
        for claim in coverage['claims']:
            for name in claim['records']:
                if name not in self.records:raise ValueError('Unresolved claim evidence')
        return {'status':'passed','records':len(self.records),'unique_objects':len({r['object'] for r in self.records.values()}),'decompressed_bytes':total,'displays':len(coverage['displays']),'claims':len(coverage['claims']),'index_sha256':digest((self.root/'index.json').read_bytes()),'scope':'Content integrity and complete index references; not scientific replication.'}
