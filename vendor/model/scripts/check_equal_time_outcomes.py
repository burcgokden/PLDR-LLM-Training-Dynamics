#!/usr/bin/env python3
"""Fail closed on incomplete protocol-to-analysis-to-table outcome coverage."""
import argparse
from pathlib import Path
from numerical_validation import load_json_strict
from model_rg.provenance import sha256, write_json
from equal_time_outcomes import validate_outcomes

def check(analysis,protocol,rendered=None):
    a=load_json_strict(analysis.read_text());p=load_json_strict(protocol.read_text());v=validate_outcomes(a,p)
    if rendered is not None:
        # Parse complete table rows independently of the renderer's number helper.
        text=(rendered/'equal-time-temporal.tex').read_text()
        actual=[]
        for line in text.splitlines():
            if line and line[0].isdigit():actual.append([x.strip() for x in line.removesuffix(r'\\').split('&')])
        if len(actual)!=len(a['cells']):raise ValueError('Rendered temporal census differs')
        def numeric(token):
            if token.startswith('$'):
                base,exponent=token.strip('$').split(r'\,10^{');return float(base)*10**int(exponent.rstrip('}'))
            return float(token)
        seen=set()
        cells={(c['heads'],c['control'],c['seed']-9163400,c['factor']):c for c in a['cells']}
        for row in actual:
            if len(row)!=7:raise ValueError('Malformed temporal row')
            key=tuple(numeric(x) for x in row[:4])
            if key in seen or key not in cells:raise ValueError('Wrong rendered identity')
            seen.add(key);c=cells[key]
            for token,field in zip(row[4:],['step_squared_energy','block_squared_energy','signed_cross_time_energy']):
                if abs(numeric(token)-c[field])>5.1e-5*max(abs(c[field]),1e-12):raise ValueError('Altered rendered outcome')
    return dict(v,analysis_sha256=sha256(analysis),protocol_sha256=sha256(protocol),checker_sha256=sha256(__file__),rendered_table_checked=rendered is not None)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['analysis','protocol','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--rendered',type=Path);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    write_json(a.output,check(a.analysis,a.protocol,a.rendered))
