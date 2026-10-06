#!/usr/bin/env python3
"""Internal completeness check of reconstructed tail and surrogate diagnostics."""
from companion_paths import configured_path
import argparse,json
from pathlib import Path
from model_rg.provenance import sha256,write_json
ROOT=Path(configured_path('data:model'))
KEYS=['status','n','n_tail','xmin','alpha','ks','tail_decades']


def main(output):
    before=ROOT/'potential-avalanche-20260913/analysis';after=ROOT/'potential-factorial-20260913/retained-analysis'
    direct=[];diagnostics=[];changed=[];surrogate_changes=[]
    for f in sorted(before.glob('h*.json')):
        if not (after/f.name).exists():continue
        old=json.loads(f.read_text());new=json.loads((after/f.name).read_text())
        if 'excursion_settings' not in old:continue
        for a,b in zip(old['excursion_settings'],new['excursion_settings'],strict=True):direct.append((a['size_fit'],b['size_fit']))
        for a,b in zip(old['primary'],new['primary'],strict=True):
            diagnostics.append((f.name,a['clock'],a['tail_diagnostics'],b['tail_diagnostics']))
            if a['surrogate_diagnostics']!=b['surrogate_diagnostics']:surrogate_changes.append((f.name,a['clock']))
    for f in sorted(before.glob('other-early-*.json')):
        old=json.loads(f.read_text());new=json.loads((after/f.name).read_text())
        for name in old['signals']:
            for a,b in zip(old['signals'][name],new['signals'][name],strict=True):direct.append((a['fit'],b['fit']))
    for f in sorted(before.glob('continuation-*.json')):
        old=json.loads(f.read_text());new=json.loads((after/f.name).read_text())
        for name in old['signals']:
            for a,b in zip(old['signals'][name]['settings'],new['signals'][name]['settings'],strict=True):
                direct.append((a['fit'],b['fit']))
                if a['phase']=='whole' and a['quantile']==.9:diagnostics.append((f.name,name+'/'+a['clock'],a['fit'],b['fit']))
                if a.get('surrogates')!=b.get('surrogates'):surrogate_changes.append((f.name,name+'/'+a['clock']))
    for name,condition,a,b in diagnostics:
        if a!=b:changed.append(dict(record=name,condition=condition,fields={k:dict(before=a.get(k),after=b.get(k)) for k in set(a)|set(b) if a.get(k)!=b.get(k)}))
    result=dict(status='passed',direct_fit_settings=len(direct),fitted_direct_settings=sum(a['status']=='fitted' for a,b in direct),
        changed_direct_fits=sum(any(a.get(k)!=b.get(k) for k in KEYS) for a,b in direct),tail_diagnostic_calls=len(diagnostics),
        fitted_tail_diagnostics=sum(a['status']=='fitted' for _,_,a,b in diagnostics),changed_tail_rows=len(changed),changes=changed,
        surrogate_changes=surrogate_changes,surrogate_iterations_rerun=True,checker_sha256=sha256(__file__))
    if len(direct)!=4428 or len(diagnostics)!=84 or result['changed_direct_fits'] or surrogate_changes:raise ValueError(result)
    write_json(output,result);print({k:v for k,v in result.items() if k!='changes'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
