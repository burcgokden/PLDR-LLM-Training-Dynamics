#!/usr/bin/env python3
"""Authenticate the prepared scientific payload, independently of Git HEAD."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def verify(root=ROOT):
    root=Path(root);m=json.loads((root/'scientific-manifest.json').read_text());errors=[]
    for name,h in m['files'].items():
        p=root/name
        if p.is_symlink() or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:errors.append(name)
    if errors:raise ValueError('Scientific manifest mismatch: '+', '.join(errors))
    return {'status':'passed','files':len(m['files']),'payload_sha256':m['payload_sha256'],'scope':'Prepared payload bytes; base Git HEAD is not a release identity.'}
if __name__=='__main__':print(json.dumps(verify(),indent=2))
