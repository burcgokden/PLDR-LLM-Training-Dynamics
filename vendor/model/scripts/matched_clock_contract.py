#!/usr/bin/env python3
"""Strict scientific design and source-selection checks for physical-clock acquisition."""
from pathlib import Path
import numpy as np
from numerical_validation import load_json_strict

CONTRACT='matched-clock-execution-v1'
FAMILY=dict(heads=[4,8,14,24],controls=[0.,1.,1.5,2.],seeds=[9163401,9163402,9163403],depth=5,head_dimension=64,
    horizon='128*N',physical_step='1/(128*N)',resource_documents='16384*N',batch=32,
    consumed_fraction=.25,generator_lr='0.0003*g*14/N',other_lr='0.0006/N',
    betas='(0.9**(14/N),0.95**(14/N))',epsilon=1e-8,weight_decay=.01,clip_norm=1.,
    source='Nested prefix of a frozen permutation of the master; first 65 tokens per document, token 64 is the sole training target; distinct documents within every trajectory.')


def validate_design(study,p,corpus,probes):
    expected=dict(admission_contract=CONTRACT,schema='matched-clock-onepass-v1',status='frozen_before_acquisition',
        family=FAMILY,shared_seed=9163499,ages=[0,.25,.5,1.],calibration_contexts=16,contexts=64,prefix_length=64,vocabulary=32000,
        cache_conditions=[[8,0.],[8,1.5],[24,0.],[24,1.5]],targets=dict(centered_rms=.25,mean_kl=.03),
        worker_hour_cap=6,memory_ceiling_bytes=22*1024**3,disk_budget_bytes=200*1024**3,
        scientific_updates=76800,qualification_updates=3,
        output_policy='Keep all finite outcomes and failed runs; no replacement seeds; final full states retained outside the manuscript bundle.')
    def same(actual,wanted):
        if isinstance(wanted,dict):return isinstance(actual,dict) and actual.keys()==wanted.keys() and all(same(actual[k],v) for k,v in wanted.items())
        if isinstance(wanted,list):return isinstance(actual,list) and len(actual)==len(wanted) and all(same(a,b) for a,b in zip(actual,wanted))
        return type(actual) is type(wanted) and actual==wanted
    allowed=set(expected)|{'jobs','document_hashes','context_rows','reservation_sha256','selection_sha256','source_sha256','input_sha256','hypotheses','interpretation'}
    if set(p)!=allowed:raise ValueError('Wrong protocol field inventory')
    for name,value in expected.items():
        if name not in p or not same(p[name],value):raise ValueError('Wrong matched-clock design field '+name)
    if any(type(p[name]) is not int for name in ['shared_seed','calibration_contexts','contexts','prefix_length','vocabulary','worker_hour_cap','memory_ceiling_bytes','disk_budget_bytes','scientific_updates','qualification_updates']):raise ValueError('Noninteger design field')
    if any(type(j['heads']) is not int or type(j['seed']) is not int or type(j['steps']) is not int or type(j['control']) not in [int,float] for j in p['jobs']):raise ValueError('Wrong job domain')
    if any(type(x) is not int for x in p['context_rows']) or len(p['context_rows'])!=80:raise ValueError('Wrong context rows')
    with np.load(Path(study)/'selection.npz') as a:
        if set(a.files)!={'order','probes','rows'}:raise ValueError('Wrong selection roles')
        order=a['order'];blocks=a['probes'];rows=a['rows']
        if order.dtype.kind not in 'iu' or not np.array_equal(order,np.random.default_rng(9163400).permutation(524288)):raise ValueError('Different frozen corpus order')
        if rows.dtype.kind not in 'iu' or not np.array_equal(rows,p['context_rows']) or len(np.unique(rows))!=80:raise ValueError('Selection/reservation rows differ')
        source=np.load(Path(probes)/'tokens.npy',mmap_mode='r')
        if not np.array_equal(blocks,source[rows,:65]):raise ValueError('Context tokens differ from source')
    tokens=np.load(Path(corpus)/'tokens.npy',mmap_mode='r')
    if tokens.shape!=(524288,513) or tokens.dtype!=np.int32:raise ValueError('Wrong master corpus geometry')
    records=load_json_strict((Path(corpus)/'records.json').read_text())
    if len(records)!=524288 or len({r['content_sha256'] for r in records})!=524288:raise ValueError('Repeated or incomplete training documents')


def validate_qualification(q,protocol_sha256):
    if q.get('status')!='complete' or q.get('protocol_sha256')!=protocol_sha256:raise ValueError('Unqualified family')
    if any(q.get(key) is not True for key in ['optimizer_replay_exact','operator_replay_exact','restoration_exact']):raise ValueError('Incomplete qualification outcomes')
    if type(q.get('optimizer_updates')) is not int or q['optimizer_updates']!=3 or type(q.get('forwards')) is not int or q['forwards']!=6:raise ValueError('Wrong qualification call ledger')
    if not 0<q.get('peak_allocated_bytes',0)<22*1024**3:raise ValueError('Qualification memory ceiling')
