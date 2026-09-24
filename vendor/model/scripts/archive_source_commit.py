#!/usr/bin/env python
"""Preserve exact executed project sources from an explicitly named Git commit."""
import argparse
import hashlib
from pathlib import Path
import subprocess

from model_rg.provenance import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();repo=Path(__file__).resolve().parents[1];out=Path(a.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    commit=subprocess.check_output(['git','rev-parse',a.commit+'^{commit}'],cwd=repo,text=True).strip()
    tree=subprocess.check_output(['git','ls-tree','-r',commit],cwd=repo,text=True)
    files={}
    for line in tree.splitlines():
        meta,name=line.split('\t',1);mode,kind,blob=meta.split()
        if kind!='blob' or not (name.startswith(('src/','scripts/','ModelRG/')) or
               name in ['ModelRG.lean','lean-toolchain','lakefile.toml','lake-manifest.json']):continue
        content=subprocess.check_output(['git','show',commit+':'+name],cwd=repo)
        path=out/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
        files[name]=dict(sha256=hashlib.sha256(content).hexdigest(),git_blob=blob)
    write_json(out/'manifest.json',dict(schema='executed-source-archive-v1',commit=commit,files=files,
        scope='Original executed source bytes; these are not qualifications of the current implementation.'))
    print('Archived',len(files),'project sources',flush=True)


if __name__=='__main__':main()
