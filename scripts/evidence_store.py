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
        r=self.records[identity];compressed=safe(self.root,r['object']).read_bytes()
        if digest(compressed)!=r['compressed_sha256'] or len(compressed)!=r['compressed_bytes']:raise ValueError('Compressed integrity failure: '+identity)
        raw=gzip.decompress(compressed)
        if digest(raw)!=r['sha256'] or len(raw)!=r['bytes']:raise ValueError('Record integrity failure: '+identity)
        return raw
    def json(self,identity):return json.loads(self.read(identity),object_pairs_hook=pairs)
    def verify(self,extract=None,prefix=''):
        manifest=read_json(self.root/'manifest.json')
        expected=set(manifest['files'])
        actual={str(p.relative_to(self.root)) for p in self.root.rglob('*') if p.is_file() and '.git' not in p.relative_to(self.root).parts and str(p.relative_to(self.root)) not in {'manifest.json','SHA256SUMS'}}
        if actual!=expected:raise ValueError('Dataset file inventory differs')
        for name,h in manifest['files'].items():
            if digest(safe(self.root,name).read_bytes())!=h:raise ValueError('Dataset file digest differs: '+name)
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
