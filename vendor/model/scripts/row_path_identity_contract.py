"""Canonical row-path identities and exact finite study census."""
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256

JOB = {'run_id','heads','control','seed','steps'}
ENDPOINT = JOB | {'increment','accumulated_finite_cross','accumulated_quadratic',
    'accumulated_matrix_derivative','derivative_error','maximum_inner_derivative_error',
    'maximum_one_step_delta','one_step_bound_applicable','one_step_pairs',
    'accumulated_bound_mean','target_nll_before','target_nll_after'}
CELL = JOB | {'block','aligned_blocks','pairs','increment','finite_cross','quadratic',
    'matrix_derivative','matrix_derivative_absolute_error','relative_displacement_energy',
    'signed_cross_time_energy','maximum_delta','identity_error','composition_error'}
TOP = {'status','schema','protocol_sha256','analysis_source_sha256','checked_sha256',
    'cells','endpoints','maximum_errors','scientific_updates','native_forward_calls',
    'paths','initialization_identities','one_step_pairs','all_aligned_block_pairs',
    'worker_seconds','peak_allocated_bytes','scope'}
SCOPE = 'All frozen paths and aligned scales. Matrix pairs are inner observations; finite errors are deterministic diagnostics on recorded float32 matrices reduced in float64. No new population replicas, parameter derivatives, or fitted successor forecasts.'

def integer(x, expected, label):
    if type(x) is not int or x != expected: raise ValueError('Wrong integer: '+label)

def bind(a,p,study):
    if set(a)!=TOP or a['schema']!='row-path-analysis-v1' or a['scope']!=SCOPE:
        raise ValueError('Unexpected analysis schema or scope')
    expected={(n,g,s) for n in [4,8,14,24] for g in [1.5,2.] for s in [9163402,9163403]}
    jobs=p['jobs']
    if p['schema']!='row-path-confirmation-v1' or len(jobs)!=16 or p['blocks']!=[1,2,4,8] or p['contexts']!=list(range(16,24)) or p['updates_per_path']!=8:
        raise ValueError('Wrong frozen design')
    by_id={}
    for j in jobs:
        if set(j)!=JOB or any(type(j[k]) is not int for k in ['heads','seed','steps']) or isinstance(j['control'],bool):
            raise ValueError('Malformed job identity')
        if j['run_id']!=f"h{j['heads']}-g{j['control']:g}-s{j['seed']}" or j['steps']!=128*j['heads'] or j['run_id'] in by_id:
            raise ValueError('Ambiguous frozen job')
        by_id[j['run_id']]=j
    if {(j['heads'],j['control'],j['seed']) for j in jobs}!=expected:
        raise ValueError('Wrong complete job grid')
    def index(rows, fields, blocks):
        out={}
        for row in rows:
            if set(row)!=fields: raise ValueError('Unexpected row fields')
            key=(row['run_id'],row['block']) if blocks else row['run_id']
            if key in out or row['run_id'] not in by_id: raise ValueError('Duplicate or foreign key')
            job=by_id[row['run_id']]
            if any(row[k]!=job[k] or isinstance(row[k],bool) for k in JOB):
                raise ValueError('Displayed identity differs from frozen job')
            for k in ['heads','seed','steps']: integer(row[k],job[k],k)
            if blocks:
                if type(row['block']) is not int or row['block'] not in p['blocks']: raise ValueError('Wrong block')
                integer(row['aligned_blocks'],8//row['block'],'aligned_blocks')
                integer(row['pairs'],8//row['block']*8*5*job['heads'],'pairs')
            else:
                integer(row['one_step_pairs'],8*8*5*job['heads'],'one_step_pairs')
                n=row['one_step_bound_applicable']
                if type(n) is not int or not 0<=n<=row['one_step_pairs']: raise ValueError('Invalid applicability count')
            out[key]=row
        want={(j,b) for j in by_id for b in p['blocks']} if blocks else set(by_id)
        if set(out)!=want: raise ValueError('Missing endpoint or block key')
        return out
    endpoints=index(a['endpoints'],ENDPOINT,False)
    cells=index(a['cells'],CELL,True)
    count=sum(j['heads'] for j in jobs)*8*5
    expected_counts=dict(paths=len(jobs),initialization_identities=len({j['seed'] for j in jobs}),
        scientific_updates=8*len(jobs),native_forward_calls=17*len(jobs),
        one_step_pairs=8*count,all_aligned_block_pairs=sum(8//b for b in p['blocks'])*count)
    for k,v in expected_counts.items(): integer(a[k],v,k)
    if set(a['maximum_errors'])!={'finite_identity','chronological_composition','path_error_budget'}:
        raise ValueError('Wrong residual summary keys')
    parent=Path(p['panel_reuse']['parent'])
    for name in ['selection.npz','protocol.json','reservation.json']:
        path=parent/name
        if sha256(path)!=p['input_sha256'][str(path)]: raise ValueError('Changed parent binding')
    if p['panel_reuse']['reservation_sha256']!=p['input_sha256'][str(parent/'reservation.json')]:
        raise ValueError('Changed reservation binding')
    source_root=Path(__file__).resolve().parents[1]
    for name,h in p['source_sha256'].items():
        if sha256(source_root/name)!=h: raise ValueError('Changed acquisition source')
    return endpoints,cells


def diagnostics(x,y):
    # Reconstruct reported residuals in their specified centered-sum arithmetic.
    # Main values remain independently checked using common-row energies.
    d=y-x
    e=np.square(x).sum((-2,-1));en=np.square(y).sum((-2,-1))
    px=x-x.mean(-2,keepdims=True);pd=d-d.mean(-2,keepdims=True)
    u=np.square(px).sum((-2,-1))/e
    inc=np.square(y-y.mean(-2,keepdims=True)).sum((-2,-1))/en-u
    lin=(2*(px*pd).sum((-2,-1))-2*u*(x*d).sum((-2,-1)))/en
    quad=(np.square(pd).sum((-2,-1))-u*np.square(d).sum((-2,-1)))/en
    return inc,lin,quad
