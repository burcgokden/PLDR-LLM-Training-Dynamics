#!/usr/bin/env python3
"""Freeze a qualified acquisition's analysis configuration before observations.

The scientific profile remains fixed. Only the independently admitted protocol
identity changes in the copied coverage configuration; numerical implementations
are copied byte for byte. Existing acquisition protocols are never rewritten.
"""
import argparse
from pathlib import Path
import shutil
import tempfile

from model_rg.provenance import sha256, write_json
from matched_clock_coverage import admitted_protocol, canonical, PROTOCOL_SHA256, read

REPO = Path(__file__).resolve().parents[1]
PROFILE = ['family','ages','cache_conditions','calibration_contexts','contexts',
           'prefix_length','vocabulary','targets','shared_seed','jobs',
           'qualification_updates','scientific_updates']
MEMBERS = ['scripts/analyze_matched_clock.py','scripts/verify_matched_clock.py',
           'scripts/render_matched_clock.py','scripts/matched_clock_coverage.py',
           'scripts/numerical_validation.py','src/model_rg/provenance.py',
           'src/model_rg/__init__.py']


def validate_profile(protocol):
    baseline = admitted_protocol()
    extra=set(protocol)-set(baseline)
    if set(baseline)-set(protocol) or extra not in [set(), {'admission_contract'}]:
        raise ValueError('Unsupported clock protocol schema')
    if extra and protocol['admission_contract']!='matched-clock-execution-v1':
        raise ValueError('Unsupported clock execution admission')
    for key in PROFILE:
        if canonical(protocol[key]) != canonical(baseline[key]):
            raise ValueError('Unsupported clock scientific profile: ' + key)


def prepare(study, output):
    study, output = study.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(output)
    # Import only for the explicit preparation action, never for CPU reduction.
    from run_matched_clock import admit
    p = admit(study, True)
    validate_profile(p)
    # Prepare prior to scientific acquisition. Qualification itself uses replay.
    if (study/'runs').exists() and any((study/'runs').iterdir()):
        raise ValueError('Freeze analysis admission before scientific trajectories')
    ph = sha256(study/'protocol.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.clock-analysis-',dir=output.parent) as tmp:
        staged=Path(tmp)/'code';staged.mkdir()
        sources={}
        for name in MEMBERS:
            path=staged/name;path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(REPO/name,path);sources[name]=sha256(REPO/name)
        config=staged/'scripts/matched_clock_coverage.py'
        original=config.read_text()
        needle="PROTOCOL_SHA256 = '"+PROTOCOL_SHA256+"'"
        if original.count(needle)!=1:
            raise ValueError('Ambiguous admission constant')
        config.write_text(original.replace(needle,"PROTOCOL_SHA256 = '"+ph+"'",1))
        protocol_path=staged/'scripts/contracts/matched-clock-observation-v1.json'
        protocol_path.parent.mkdir();shutil.copy2(study/'protocol.json',protocol_path)
        frozen={str(x.relative_to(staged)):sha256(x) for x in staged.rglob('*') if x.is_file()}
        write_json(staged/'admission.json',dict(status='passed',schema='matched-clock-analysis-bundle-v1',
            protocol_sha256=ph,qualification_sha256=sha256(study/'qualification/manifest.json'),
            scientific_profile_sha256=canonical({k:p[k] for k in PROFILE}),
            generator_sha256=sha256(__file__),maintained_sources_sha256=sources,members_sha256=frozen,
            configuration_change='Only PROTOCOL_SHA256 and the bundled immutable protocol; numerical source bytes unchanged.',
            frozen_before_scientific_acquisition=True,native_calls_during_preparation=0))
        (staged/'README.md').write_text('''# Qualified clock analysis bundle

This configuration is frozen after native qualification and before scientific
acquisition. `admission.json` identifies the qualified protocol, unchanged
numerical scripts, and the protocol-specific coverage configuration.

Use this directory's `src` on PYTHONPATH and its `scripts/analyze_matched_clock.py`,
`scripts/verify_matched_clock.py`, and `scripts/render_matched_clock.py` entry
points. The first two take the acquired study as `--study`; all require fresh
output names. Preserve this bundle with the observations. It admits the fixed
scientific profile and the exact newly reserved panel, without accepting a
submitted result as its own expected design. No native call occurs in preparation.
''')
        staged.rename(output)
    print(dict(status='passed',protocol_sha256=ph,output=str(output)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    prepare(**vars(p.parse_args()))
