#!/usr/bin/env python
"""Select a bounded single-pass mechanism extension after the paired data-law test."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from model_rg.provenance import sha256, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-feasible-20260908')
    p.add_argument('--stage',choices=['domain','jacobians'],required=True);a=p.parse_args()
    root=Path(a.root).resolve();study=root/a.study;repo=Path(__file__).resolve().parents[1]
    if a.stage=='domain':
        out=study/'protocols/onepass-input-domain-selection.json'
        if out.exists():raise FileExistsError(out)
        prefix_selection=study/'protocols/scheduled-prefix-selection.json';prefix=json.loads(prefix_selection.read_text())
        pair_proof=study/'verification/repetition-reference_schedule.json';paired=json.loads(pair_proof.read_text())
        if paired['status']!='passed' or paired['pairs']!=4:raise AssertionError('The complete matched data-law result is required')
        cases=[];inputs={str(p):sha256(p) for p in [prefix_selection,pair_proof]}
        for old in prefix['cases']:
            if old['role']!='data_schedule' or old['step'] not in [32768,65536]:continue
            proof_path=study/'verification/prefix'/(old['name']+'.json');proof=json.loads(proof_path.read_text())
            if proof['status']!='passed' or proof['case']!=old:raise AssertionError('A selected prefix reference lacks independent verification')
            folder=study/'measurements'/old['name']
            for name in ['manifest.json','measurements.npz']:
                path=folder/name;digest=sha256(path)
                if proof['checked_sha256'][str(path)]!=digest:raise AssertionError('A prefix reference changed')
                inputs[str(path)]=digest
            training=study/'verification/training'/(old['run_id']+'.json');raw=json.loads(training.read_text())
            if raw['status']!='complete':raise AssertionError('The native training parent requires independent reconstruction')
            inputs[str(proof_path)]=sha256(proof_path);inputs[str(training)]=sha256(training)
            files=[Path(old['state']),Path(old['parent_manifest']),training,proof_path]
            case=dict(old,name='input-domain-'+old['name'].removeprefix('prefix-'),reference_prefix=old['name'],
                prefix_verification=str(proof_path),input_sha256={str(p):sha256(p) for p in files})
            cases.append(case)
        if len(cases)!=4:raise AssertionError('Both source recipes and both matched horizons are required')
        probe=root/'controlled-study-20260905/data/short'
        for p in [probe/'tokens.npy',probe/'offsets.npy']:inputs[str(p)]=sha256(p)
        sources=['scripts/prepare_onepass_input_domain.py','scripts/measure_onepass_input_domain.py',
            'scripts/verify_onepass_input_domain.py','scripts/run_onepass_input_domain.py',
            'src/model_rg/native.py','src/model_rg/training.py','src/model_rg/controlled.py','src/model_rg/provenance.py']
        spec=dict(schema='onepass-prefix-input-domain-selection-v1',frozen_at=datetime.now(timezone.utc).isoformat(),
            cases=cases,rows=prefix['rows'],lengths=[16,32,48,64],states=4,prefix_domains=16,decoder_stage_paths=80,
            inputs_sha256=inputs,producer_sources={n:sha256(repo/n) for n in sources},
            prior_information='The completed single-pass/repeated source comparison and all four proper-prefix observations were known before this stage selection. In the source Below recipe at65536, a complete-position causal-risk panel has a0.395-nat proper-minus-full gap; the strongly concentrated Near endpoint has negligible measured gap. Endpoint task decisions were also inspected. No native intermediate-stage or Jacobian outcome for these four states was inspected before this selection.',
            purpose='Locate the finite input-domain transport associated with the observed prefix defect. Use the same eight held-out contexts and all four pre-existing proper-prefix lengths, retaining every decoder and head. This is a descriptive mechanism extension selected from completed outcomes, not a prediction made before those outcomes.',
            jacobian_scope='After independent stage verification, measure all64-by64 row-map Jacobians at context indices0,3, head indices0,6,13 and row indices0,31,63, at each selected length and decoder. The resulting1440 local Jacobians are not a uniform-domain or augmented-training spectrum certificate.',
            resource_scope='CPU float32 stage capture and CPU float64 local differentiation. No training updates or new training identities are added.')
    else:
        out=study/'protocols/onepass-row-jacobian-selection.json'
        if out.exists():raise FileExistsError(out)
        domain=study/'protocols/onepass-input-domain-selection.json';base=json.loads(domain.read_text())
        proof_path=study/'verification/onepass-input-domain.json';proof=json.loads(proof_path.read_text())
        if proof['status']!='passed' or (proof['states'],proof['prefix_domains'])!=(4,16):
            raise AssertionError('Complete independently checked stage inputs are required')
        inputs={str(p):sha256(p) for p in [domain,proof_path]};cases=[]
        for old in base['cases']:
            folder=study/'measurements'/old['name']
            for name in ['manifest.json','measurements.npz']:
                path=folder/name;digest=sha256(path)
                if proof['checked_sha256'][str(path)]!=digest:raise AssertionError('A verified stage input changed')
                inputs[str(path)]=digest
            for length in base['lengths']:
                cases.append(dict(old,name='onepass-row-jacobian-'+old['name'].removeprefix('input-domain-')+f'-L{length}',
                    reference_transport=old['name'],length=length))
        sources=['scripts/measure_onepass_row_jacobians.py','scripts/verify_onepass_row_jacobians.py',
            'scripts/prepare_onepass_input_domain.py','src/model_rg/native.py','src/model_rg/training.py',
            'src/model_rg/controlled.py','src/model_rg/provenance.py']
        spec=dict(schema='onepass-row-jacobian-selection-v1',frozen_at=datetime.now(timezone.utc).isoformat(),cases=cases,
            context_indices=[0,3],head_indices=[0,6,13],row_indices=[0,31,63],points_per_layer=18,
            decoder_layers=5,selected_jacobians=1440,inputs_sha256=inputs,producer_sources={n:sha256(repo/n) for n in sources},
            selection_scope='The four lengths and all differentiation indices were fixed by the parent input-domain selection before intermediate-stage outcomes. This stage binds the completed independently checked normalized-Gram inputs without changing that selection.',
            arithmetic='Exact float64 lift of stored float32 parameters and native input points; complete analytic NumPy derivative reconstruction and independent native float32 stage replay. No finite local derivative establishes a uniform contraction bound or critical exponent.')
    write_json(out,spec);print('Selected',a.stage,sha256(out),flush=True)


if __name__=='__main__':main()
