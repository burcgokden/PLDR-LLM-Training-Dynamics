#!/usr/bin/env python3
"""Reconstruct a fixed cache study from an explicit relocated input graph.

Copies raw inputs to a fresh workspace, verifies every byte before/after,
blocks Python reads from the original workspace, and compares all archived
scientific cells exactly. The separate verifier retains its own tolerances.
No acquisition, model construction, or training is performed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(path,data):
    path.write_text(json.dumps(data,indent=2)+'\n')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-study',type=Path,required=True)
    ap.add_argument('--workspace',type=Path,required=True)
    args=ap.parse_args();src=args.source_study.resolve();base=args.workspace.resolve()
    base.mkdir(parents=True,exist_ok=False)
    code=base/'code';study=base/'inputs/study';outside=base/'outside';outside.mkdir()
    shutil.copytree(ROOT,code,ignore=shutil.ignore_patterns('.git','.lake','build','dist','__pycache__','.pytest_cache','monograph.pdf'))
    shutil.copytree(src,study)
    p=json.loads((study/'protocol.json').read_text())
    original_hashes={str(x):sha(x) for x in src.rglob('*') if x.is_file()}
    entries=[dict(identity=str(src/'protocol.json'),path=str(study/'protocol.json'),sha256=sha(study/'protocol.json'))]
    for i,(name,digest) in enumerate(sorted(p['input_sha256'].items())):
        source=Path(name);dest=base/'inputs/external'/f'{i:03d}'/source.name
        dest.parent.mkdir(parents=True);shutil.copyfile(source,dest)
        if sha(source)!=digest or sha(dest)!=digest:raise ValueError('Acquisition digest differs: '+name)
        original_hashes[name]=digest
        entries.append(dict(identity=name,path=str(dest),sha256=digest))
    # v3 receipts bind their own hash; the reservation index is a required external role.
    if p['schema']=='cache-state-transfer-v3':
        source=src.parent/'context-reservations-v1.json';dest=base/'inputs/reservations.json';shutil.copyfile(source,dest)
        original_hashes[str(source)]=sha(source)
        entries.append(dict(identity=str(source),path=str(dest),sha256=sha(dest)))
    locations=dict(schema='pldr-acquisition-locations-v1',logical_root=str(src.parent),
        study=dict(identity=str(src),path=str(study)),files=entries)
    manifest=base/'input-locations.json';write(manifest,locations)
    relocated_hashes={str(x):sha(x) for x in (base/'inputs').rglob('*') if x.is_file()}
    guard=base/'guard';guard.mkdir()
    blocked=[str(ROOT),str(src.parent)]
    (guard/'sitecustomize.py').write_text('import os,sys\nBLOCKED='+repr(blocked)+'\n'
      'def audit(event,args):\n'
      ' if event in {"open","os.listdir","os.scandir","os.chdir"} and args and isinstance(args[0],(str,bytes,os.PathLike)):\n'
      '  p=os.path.abspath(os.fsdecode(args[0]))\n'
      '  if any(p==b or p.startswith(b+os.sep) for b in BLOCKED): raise PermissionError("Original input access rejected: "+p)\n'
      'sys.addaudithook(audit)\n')
    env=dict(os.environ,PYTHONPATH=str(guard),PLDR_READ_GUARD=str(guard),PYTHONDONTWRITEBYTECODE='1',
        PLDR_INPUT_MANIFEST=str(manifest),MODEL_RG_DATA_ROOT=str(base/'unused-runtime-root'),
        CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    def run(name,command,expect=0):
        r=subprocess.run(command,cwd=outside,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=900)
        (base/(name+'.log')).write_text(r.stdout)
        if r.returncode!=expect:raise RuntimeError(name+'\n'+r.stdout[-5000:])
        return r
    run('guard-control',[sys.executable,'-c','from pathlib import Path\ntry: Path('+repr(str(src/'protocol.json'))+').read_bytes()\nexcept PermissionError: print("blocked")\nelse: raise SystemExit(1)'])
    run('reconstruction',['sh',str(code/'vendor/model/scripts/workspace-wrappers/analyze-cache-state-transfer.sh'),str(study),str(base/'analysis'),str(manifest)])
    a=json.loads((base/'analysis/analysis.json').read_text());old=json.loads((study/'analysis.json').read_text())
    if len(a['cells'])!=30 or a['cells']!=old['cells']:raise ValueError('Scientific cells differ (exact equality required)')
    cert=json.loads((base/'analysis/verification.json').read_text())
    if cert['status']!='passed':raise ValueError('Separate verifier failed')
    if original_hashes!={n:sha(n) for n in original_hashes}:raise ValueError('Original input changed')
    if relocated_hashes!={n:sha(n) for n in relocated_hashes}:raise ValueError('Relocated input changed')
    report=dict(status='passed',study=src.name,scientific_cells=30,all_scientific_cells_exactly_equal=True,
        original_input_files=len(original_hashes),relocated_input_files=len(relocated_hashes),
        external_identities=len(p['input_sha256']),original_and_relocated_hashes_unchanged=True,
        original_workspace_reads='blocked by inherited Python audit hook; negative control passed',
        protocol_sha256=sha(study/'protocol.json'),input_manifest_sha256=sha(manifest),
        analysis_sha256=sha(base/'analysis/analysis.json'),verification_sha256=sha(base/'analysis/verification.json'),
        independent_verifier=dict(status=cert['status'],tolerances=cert['tolerances'],maximum_predictive_difference=cert['maximum_predictive_difference'],maximum_operator_difference=cert['maximum_operator_difference']),
        new_training_updates=0,new_acquisition_calls=0,
        scope='Raw-array CPU reconstruction of the named fixed panel; no native model replay or training.')
    write(base/'result.json',report);write(ROOT/'validation/cache-relocation.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
