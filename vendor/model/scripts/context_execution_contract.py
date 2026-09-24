"""Shared fail-closed queue/worker preflight for fixed-context acquisitions.

Scientific schema names remain stable for CPU reduction. The explicit execution
contract distinguishes new acquisitions from immutable read-only observations.
"""
from companion_paths import legacy_path
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256, write_json
from cache_state_contract import read, same, keys, hash_value, sources
from context_reservations import validate_receipt, document_hashes

REPO=Path(__file__).resolve().parents[1]
ROOT=Path(legacy_path('/pldr-data/model'))
PARENT=ROOT/'critical-onepass-refinement-20260914'
PROBES=ROOT/'controlled-study-20260905/data/short'
CORPUS=ROOT/'data/refinedweb-onepass-524288'
NATIVE=ROOT/'assets/PLDR-LLM-v51-SOC-110M-1'
DICTIONARY=ROOT/'categorical-visibility-20260915'
CONTRACT='context-execution-v1'
COMMON=['scripts/context_execution_contract.py','scripts/context_reservations.py','scripts/cache_state_contract.py']
SOURCE_ROLES={
 'context-categorical-v1':['scripts/context_categorical_study.py','scripts/numerical_validation.py','src/model_rg/training.py','src/model_rg/native.py','src/model_rg/provenance.py'],
 'operator-cache-v1':['scripts/run_operator_cache_study.py','scripts/analyze_operator_cache.py','scripts/context_categorical_study.py','scripts/numerical_validation.py','src/model_rg/training.py','src/model_rg/native.py','src/model_rg/inference_interventions.py','src/model_rg/provenance.py'],
 'cache-risk-v1':['scripts/run_cache_risk_study.py','scripts/analyze_cache_risk_study.py','scripts/cache_risk_validation.py','scripts/analyze_operator_cache.py','scripts/verify_cache_risk_study.py','scripts/context_categorical_study.py','scripts/numerical_validation.py','src/model_rg/training.py','src/model_rg/native.py','src/model_rg/inference_interventions.py','src/model_rg/provenance.py']}
SOURCE_ROLES={k:v+COMMON for k,v in SOURCE_ROLES.items()}


def attach(study,p,receipt):
    """Called only by preparation, before any prediction is obtained."""
    write_json(study/'reservation.json',receipt)
    p.update(implementation=CONTRACT,reservation_sha256=sha256(study/'reservation.json'),
             excluded_context_protocols=sorted(receipt['imported_protocols']))
    p['input_sha256'].update(receipt['imported_protocols'])
    p['source_sha256']={name:sha256(REPO/name) for name in SOURCE_ROLES[p['schema']]}
    write_json(study/'protocol.json',p)
    for name in p['source_sha256']:
        path=study/'executed-source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((REPO/name).read_bytes())


def expected_arms():
    return [dict(name=f'L{l}-C{c}-P{p}-M{m}',length=l,calibration_length=c,panel=p,size=m)
            for l in [32,64,128] for c in ([64] if l==64 else [l,64])
            for p in range(4) for m in ([4,8,16] if l==64 else [16])]


def external_roles(p):
    result={str(f) for f in [PARENT/'protocol.json',PROBES/'records.json',PROBES/'tokens.npy',CORPUS/'records.json',NATIVE/'modeling_pldrllm.py',NATIVE/'configuration_pldrllm.py']}
    if p['schema'] in ['context-categorical-v1','operator-cache-v1']:result.add(str(PROBES/'offsets.npy'))
    if p['schema']=='context-categorical-v1':result.update(str(f) for f in [DICTIONARY/'protocol.json',DICTIONARY/'reference.npz'])
    result.update(str(PARENT/'runs'/j['run_id']/'manifest.json') for j in p['jobs'])
    ex=p['excluded_context_protocols']
    if type(ex) is not list or ex!=sorted(set(ex)):raise ValueError('Noncanonical exclusion inventory')
    for n in ex:
        q=Path(n)
        if q.parent.parent!=ROOT or q.name!='protocol.json' or str(q)!=n:raise ValueError('Foreign exclusion role')
    return result|set(ex)


def admit(study,index=None):
    study=Path(study).resolve();p=read(study/'protocol.json');schema=p.get('schema')
    if schema not in SOURCE_ROLES or p.get('implementation')!=CONTRACT:raise ValueError('Current versioned execution contract required; old acquisitions are read-only')
    same(p['parent'],str(PARENT),'parent');same(p['vocabulary'],32000,'vocabulary');same(p['batch_size'],8,'batch');same(p['training_updates'],0,'training updates')
    if index is not None and (type(index) is not int or not 0<=index<24):raise ValueError('Worker index outside the complete design')
    sources(p['source_sha256'],SOURCE_ROLES[schema],REPO)
    sources(p['source_sha256'],SOURCE_ROLES[schema],study/'executed-source')
    parent=read(PARENT/'protocol.json')
    expected=[j for j in parent['jobs'] if j['heads'] in [8,24] and j['control'] in [0,1.5]]
    if len(expected)!=24 or type(p['jobs']) is not list or len(p['jobs'])!=24:raise ValueError('Incomplete checkpoint grid')
    for j,e in zip(p['jobs'],expected):
        keys(j,set(e)|{'checkpoint_sha256'},'job roles')
        same({k:v for k,v in j.items() if k!='checkpoint_sha256'},e,'ordered scientific job')
        m=read(PARENT/'runs'/j['run_id']/'manifest.json')
        same(m['status'],'complete','parent completion');same(m['job'],e,'parent job')
        hash_value(j['checkpoint_sha256']);same(j['checkpoint_sha256'],m['artifacts']['final-state.pt'],'checkpoint identity')
    keys(p['input_sha256'],external_roles(p),'external input roles')
    for n,h in p['input_sha256'].items():
        hash_value(h)
        if sha256(n)!=h:raise ValueError('Changed external input '+n)
    receipt=validate_receipt(study,p)
    same(p['excluded_context_protocols'],sorted(receipt['imported_protocols']),'reserved exclusions')
    for n,h in receipt['imported_protocols'].items():same(p['input_sha256'][n],h,'reservation input')
    count=p['contexts'];cal=0
    if schema=='context-categorical-v1':
        if type(count) is not int or count<8 or count%8:raise ValueError('Invalid context count')
        same(p['replicas'],6,'replicas');same(p['target'],.25,'target')
        sizes=p['sizes']
        if type(sizes) is not list or sizes!=sorted(set(sizes)) or not sizes or any(type(k) is not int or not 0<k<32000 for k in sizes):raise ValueError('Invalid vocabulary resolutions')
        same(p['role'],'observation','role');same(p['expected_forward_calls'],24*count//8,'call count');same(p['qualification_forward_calls'],2,'qualification count')
    else:
        same(p['status'],'frozen_before_acquisition','status');same(p['new_training_replicas'],0,'replicas')
        if schema=='operator-cache-v1':
            cal=16;same(count,128,'contexts');same(p['calibration_contexts'],16,'calibration');same(p['prefix_length'],64,'prefix');same(p['external_target_position'],64,'target position')
            same(p['calibration_rows'],p['context_rows'][:16],'calibration rows');same(p['evaluation_rows'],p['context_rows'][16:],'assessment rows')
            same(p['primary_targets'],dict(centered_rms=.25,mean_kl_nats=.03,median_latency_reduction=.10),'targets')
            same(p['expected_calls'],dict(calibration=48,assessment=768,timing=1920,qualification=96),'calls')
            t=p['timing']
            for k,v in dict(warmup_pairs=10,measured_pairs=30,alternating_order=True,synchronized=True).items():same(t[k],v,'timing '+k)
        else:
            cal=64;same(count,64,'contexts');same(p['calibration_contexts'],64,'calibration');same(p['calibration_panels'],4,'panels');same(p['prefix_lengths'],[32,64,128],'prefixes');same(p['source_prefix_start'],0,'prefix start')
            same(p['arms'],expected_arms(),'arms');same(p['primary_targets'],dict(centered_rms=.25,mean_kl_nats=.03),'targets')
            same(p['expected_calls'],dict(calibration=576,native_assessment=576,cached_assessment=5376,qualification=336),'calls')
            same(p['memory_ceiling_bytes'],22*1024**3,'memory ceiling')
        same(p['worker_hour_cap'],3,'worker cap')
    hash_value(p['selection_sha256'])
    if sha256(study/'inputs.npz')!=p['selection_sha256']:raise ValueError('Changed selected inputs')
    with np.load(study/'inputs.npz',allow_pickle=False) as a:
        wanted={'blocks','rows'}|({'reference','order'} if schema=='context-categorical-v1' else set())
        if set(a.files)!=wanted or len(a.files)!=len(wanted):raise ValueError('Input member roles differ')
        blocks,rows=a['blocks'],a['rows']
        if blocks.shape!=(count+cal,129 if schema=='cache-risk-v1' else 65) or blocks.dtype.kind not in 'iu' or np.any(blocks<0) or np.any(blocks>=32000):raise ValueError('Invalid token geometry')
        if rows.shape!=(count+cal,) or rows.dtype.kind not in 'iu' or not np.array_equal(rows,np.arange(rows[0],rows[0]+count+cal)):raise ValueError('Invalid row geometry')
        same(p['context_rows'],rows.tolist(),'rows')
        records=read(PROBES/'records.json')
        if rows[0]<0 or rows[-1]>=len(records):raise ValueError('Rows outside source')
        same(document_hashes(p),[records[i]['content_sha256'] for i in rows],'document hashes')
        tokens=np.load(PROBES/'tokens.npy',mmap_mode='r')
        original=tokens[rows,:129] if schema=='cache-risk-v1' else tokens[rows[:,None],np.load(PROBES/'offsets.npy')[rows,None]+np.arange(65)]
        if not np.array_equal(blocks,original):raise ValueError('Selected tokens differ from source')
        if schema=='context-categorical-v1':
            with np.load(DICTIONARY/'reference.npz') as d:reference=d['reference'].mean(0)
            if reference.shape!=(32000,) or not np.isfinite(reference).all() or np.any(reference<=0) or not np.array_equal(a['reference'],reference) or not np.array_equal(a['order'],np.argsort(-reference,kind='stable')):raise ValueError('Changed dictionary/reference')
    # Even a direct worker must reject a changed payload before creating output.
    if index is not None:
        job=p['jobs'][index]
        if sha256(PARENT/'runs'/job['run_id']/'final-state.pt')!=job['checkpoint_sha256']:raise ValueError('Changed incoming checkpoint payload')
    return p
