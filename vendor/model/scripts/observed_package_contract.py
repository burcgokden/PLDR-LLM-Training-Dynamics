"""Explicit separation of source-archive metadata and complete observed data."""
from pathlib import Path
import json,hashlib,shutil


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def inspect_package(root):
    root=Path(root).resolve()
    descriptor=root/'PACKAGE_KIND.json'
    if descriptor.exists():
        d=json.loads(descriptor.read_text())
        if d.get('kind')=='external_observed_data_metadata':
            if digest(root/'INDEX.json')!=d['external_index_sha256']:
                raise ValueError('External index differs')
            return dict(kind='metadata_only',runnable=False,external_root=d['complete_local_root'])
        raise ValueError('Unknown package descriptor')
    index=json.loads((root/'INDEX.json').read_text())
    names=index['files']
    if set(p.name for p in root.iterdir())=={'INDEX.json','README.md','clean-reproduction-verification.json'}:
        return dict(kind='legacy_metadata_only',runnable=False)
    required={'README.md','reference.npz','code/reproduce_categorical_observations.py',
              'refinement-expected.json','context-expected.json'}
    if not required.issubset(names):raise ValueError('Incomplete package index')
    observations=index['observations']
    if len({row['path'] for row in observations})!=len(observations):raise ValueError('Duplicate observation')
    if not observations or any(row['path'] not in names for row in observations):
        raise ValueError('Observation absent from index')
    for item in index.get('analysis_families',[]):
        if item['expected'] not in names:raise ValueError('Expected reduction absent from index')
    for name,expected in names.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or not p.is_file() or digest(p)!=expected:
            raise ValueError('Missing or changed complete-package member '+name)
    return dict(kind='complete_observed_data',runnable=True,indexed_files=len(names),
                observation_files=len(observations),index_sha256=digest(root/'INDEX.json'),
                verified_member_bytes=sum((root/n).stat().st_size for n in names))


def write_metadata_stub(full_root,output,verification):
    full_root=Path(full_root).resolve();output=Path(output)
    info=inspect_package(full_root)
    if not info['runnable']:raise ValueError('A complete source package is required')
    output.mkdir(parents=True,exist_ok=False)
    for source,name in [(full_root/'INDEX.json','INDEX.json'),
                        (Path(verification),'clean-reproduction-verification.json')]:
        shutil.copy2(source,output/name)
    d=dict(schema='observed-data-access-v1',kind='external_observed_data_metadata',
           complete_local_root=str(full_root),public_persistent_identifier=None,
           external_index_sha256=info['index_sha256'],external_indexed_files=info['indexed_files'],
           external_observation_files=info['observation_files'],external_member_bytes=info['verified_member_bytes'])
    (output/'PACKAGE_KIND.json').write_text(json.dumps(d,indent=2)+'\n')
    index=json.loads((full_root/'INDEX.json').read_text())
    cells=index.get('expected_scale_cells',560);chains=index.get('expected_kl_chains',23808)
    text=f'''# Metadata for the separate categorical observed-data package

This directory contains metadata only. It is not a runnable data package.
The full local package is located at:

`{full_root}`

That complete package contains {info['indexed_files']} indexed files, including
{info['observation_files']} observation files, totaling {info['verified_member_bytes']:,} bytes.
It is prepared locally and has no public persistent identifier.
The absolute location identifies local access; it is not a public download link.

`INDEX.json` describes members of that separate complete package, including its
own README. It does not describe the contents or README of this metadata directory.
The saved reconstruction certificate records an execution on the complete package;
it does not certify that its members are present here. `PACKAGE_KIND.json`
identifies this distinction explicitly.

Run the following command from any directory, after setting PACKAGE_ROOT to the
complete package location above or to a complete relocated copy. OUTPUT_JSON must
name a fresh output file. Python and NumPy are required.

    OPENBLAS_NUM_THREADS=1 python "$PACKAGE_ROOT/code/reproduce_categorical_observations.py" --package "$PACKAGE_ROOT" --output "$OUTPUT_JSON"

The complete observed-data reducer checks its indexed members and reconstructs
{cells:,} scale cells and {chains:,} KL chains with no native forwards or training updates.
Native training replay additionally needs the separately inventoried source
tokens and complete model/optimizer states described in ASSET_ACCESS.md.
'''
    (output/'README.md').write_text(text)
    return inspect_package(output)
