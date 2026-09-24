#!/usr/bin/env python
"""Populate durable status from bound executed artifacts; no proposed filename is executable."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
from model_rg.provenance import sha256,write_json
from confirmation_status import validate,render


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--study',required=True)
    p.add_argument('--output',default='docs');a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];root=Path(a.root).resolve();study=Path(a.study).resolve();out=Path(a.output)
    def bind(path):return dict(path=str(path.resolve()),sha256=sha256(path))
    def stage(identifier,name,producer,wrapper,folder,verifiers,updates):
        complete=all((folder/'confirmation'/file).exists() for file,_ in verifiers)
        return dict(id=identifier,name=name,status='implemented_executed_verified' if complete else 'implemented_frozen_qualified',
            executed_scientific_updates=updates if complete else 0,proposed_update_ceiling=0 if complete else updates,
            producer=bind(repo/'scripts'/producer),wrapper=bind(repo/'scripts/workspace-wrappers'/wrapper),
            protocol=bind(folder/'confirmation/protocol.json'),executed_sources=bind(folder/'confirmation/executed-source/manifest.json'),
            terminal_verifiers=[bind(repo/'scripts'/script) for _,script in verifiers],
            terminal_evidence=[bind(folder/'confirmation'/file) for file,_ in verifiers] if complete else None,
            short_qualification=dict(protocol=bind(folder/'qualification/protocol.json'),
                terminal_evidence=bind(folder/'qualification/verification.json')))
    p1=stage('P1','State refresh and adaptation cost','refresh_study.py','state-refresh-study.sh',root/'state-refresh-20260911',
        [('verification.json','verify_refresh.py'),('summary-verification.json','verify_refresh_summary.py')],82432)
    p1['limits']='Eight original lineages, two dependent fit cohorts; no uniform economical mean-and-covariance law is identified.'
    p3=stage('P3','Native paired finite-pulse response','directional_study.py','directional-response-study.sh',study,
        [('verification.json','verify_directional.py')],36864)
    p3['limits']='Complete-state finite pulses retain symmetric and antisymmetric response and signed covariance. Completion does not establish a uniform derivative domain, native response gap, forcing spectrum, or critical surface.'
    p3['development']=bind(study/'development/verification.json');p3['theory_freeze']=bind(study/'theory-freeze.json')
    p3['development_updates']=9216;p3['qualification_updates']=288;p3['all_zero_replay_updates']=2064
    def proposal(identifier,name,ceiling,design,prerequisites):
        return dict(id=identifier,name=name,status='proposed_not_implemented_not_frozen',executed_scientific_updates=0,
            proposed_update_ceiling=ceiling,producer=None,wrapper=None,terminal_verifiers=None,protocol=None,
            terminal_evidence=None,short_qualification=None,design=design,prerequisites=prerequisites)
    p2=proposal('P2','Matched crossed outer transfer',90112,
        'Four corpus draws, two complete initialization identities crossed with every corpus, and widths 4 and 14; fixed age 2048 and 16/8/32 adaptation/calibration/assessment branches.',
        ['Corpus-independent complete-initialization seed mapping and equal initial component digests across corpora.',
         'Matched-age archive fixed on independent development states before new target outcomes.',
         'Actual stage-specific producer, frozen protocol, terminal native qualification and resource budget.'])
    p4=proposal('P4','Finite size and drive study',163840,
        'Four widths and two drives crossed with two corpora and two complete initialization identities; primary age 4096 and eight 128-update continuations per state.',
        ['A validated response domain for the intended scientific claim. The completed finite-pulse study supplies no spectral qualification.',
         'Frozen shape-aware width, optimizer clocks, resource and observation specification.',
         'Actual aged width-20 pipeline, end-to-end terminal qualification and measured runtime/storage budget.'])
    status=dict(schema='rg-confirmation-status-v2',as_of=datetime.now(timezone.utc).isoformat(),stages=[p1,p2,p3,p4],
        launch_rule='A new stage must bind its own producer, frozen protocol and passed stage-specific terminal qualification. require-ready rejects an unimplemented or unrelated stage; wrappers refuse existing scientific destinations. Completed status additionally requires all terminal scientific evidence.',
        publication_rule='Only completed experiment outcomes enter the manuscript. All prespecified outcomes of a reported comparison are retained. Development that changes a hypothesis is distinct from its subsequent confirmation sample.')
    validate(status);out.mkdir(parents=True,exist_ok=True);write_json(out/'confirmation-status.json',status)
    (out/'CONFIRMATION_STATUS.md').write_text(render(status))
    print([(s['id'],s['status']) for s in status['stages']],flush=True)

if __name__=='__main__':main()
