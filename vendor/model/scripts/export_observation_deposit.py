#!/usr/bin/env python3
"""Export a local complete observation deposit and reproduce an extracted copy."""
from companion_paths import configured_path
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from model_rg.provenance import sha256, write_json
from numerical_validation import load_json_strict
from observed_package_contract import inspect_package

ROOT=Path(configured_path('data:model'))


def export(package,archive,records):
    package=package.resolve();archive=archive.resolve()
    if archive.exists() or not archive.is_relative_to(ROOT) or not package.is_relative_to(ROOT):
        raise ValueError('Use a fresh archive in the authorized experiment root')
    inspect_package(package);index=load_json_strict((package/'INDEX.json').read_text())
    if not index.get('context_risk'):raise ValueError('Complete context-risk observations are required')
    names=['INDEX.json',*sorted(index['files'])]
    with tarfile.open(archive,'w') as out:
        for name in names:
            path=package/name;info=tarfile.TarInfo(name);info.size=path.stat().st_size
            info.mode=0o644;info.uid=info.gid=0;info.mtime=0
            with path.open('rb') as data:out.addfile(info,data)
    digest=sha256(archive)
    archive.with_suffix(archive.suffix+'.sha256').write_text(digest+'  '+archive.name+'\n')
    records.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='modelrg-deposit-reproduction-',dir='/tmp') as temporary:
        unpacked=Path(temporary)/'observations';unpacked.mkdir()
        with tarfile.open(archive) as data:data.extractall(unpacked,filter='data')
        if sha256(unpacked/'INDEX.json')!=sha256(package/'INDEX.json'):raise ValueError('Transported index differs')
        env=dict(os.environ,PYTHONPATH='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        for script,name in [('reproduce_categorical_observations.py','portable-verification'),
                            ('reproduce_context_risk_observations.py','portable-context-risk')]:
            with (records/(name+'.log')).open('x') as log:
                subprocess.run([sys.executable,str(unpacked/'code'/script),'--package',str(unpacked),
                    '--output',str((records/(name+'.json')).resolve())],cwd='/tmp',env=env,
                    stdout=log,stderr=subprocess.STDOUT,check=True)
    write_json(records/'deposit-verification.json',dict(status='passed',schema='local-observation-deposit-v1',
        archive=str(archive),archive_sha256=digest,archive_bytes=archive.stat().st_size,
        index_sha256=sha256(package/'INDEX.json'),members=len(names),source_sha256=sha256(__file__),
        clean_extraction_reproduced=True,empty_pythonpath=True,public_availability=False,
        native_forward_calls=0,training_updates=0,
        checks={name:sha256(records/name) for name in ['portable-verification.json','portable-context-risk.json']}))
    print('Complete local deposit and extracted-copy reproduction passed:',archive)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--records',type=Path,required=True)
    a=p.parse_args();export(a.package,a.archive,a.records)
