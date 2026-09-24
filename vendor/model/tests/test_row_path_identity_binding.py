"""Early refusal of mislabeled row outcomes, before touching native payloads."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pytest
from scripts.check_row_path_identity_cli import mutations
from scripts.row_path_identity_contract import bind, JOB, ENDPOINT, CELL, TOP, SCOPE


def fixture():
    jobs=[dict(run_id=f'h{n}-g{g:g}-s{s}',heads=n,control=g,seed=s,steps=128*n)
          for n in [4,8,14,24] for g in [1.5,2.] for s in [9163402,9163403]]
    endpoints=[dict.fromkeys(ENDPOINT,0) for _ in jobs];cells=[]
    for j,e in zip(jobs,endpoints):
        e.update(j,one_step_pairs=8*8*5*j['heads'])
        for b in [1,2,4,8]:
            c=dict.fromkeys(CELL,0);c.update(j,block=b,aligned_blocks=8//b,pairs=8//b*8*5*j['heads']);cells.append(c)
    a=dict.fromkeys(TOP,0);a.update(schema='row-path-analysis-v1',scope=SCOPE,endpoints=endpoints,cells=cells,
        paths=16,initialization_identities=2,scientific_updates=128,native_forward_calls=272,one_step_pairs=64000,all_aligned_block_pairs=120000,
        maximum_errors=dict(finite_identity=0.,chronological_composition=0.,path_error_budget=0.))
    p=dict(schema='row-path-confirmation-v1',jobs=jobs,blocks=[1,2,4,8],contexts=list(range(16,24)),updates_per_path=8)
    return a,p

@pytest.mark.parametrize('name',['endpoint-seed','cell-label-swap','duplicate-endpoint','duplicate-cell','missing-endpoint','missing-cell',
    'endpoint-heads','endpoint-control','endpoint-steps','cell-steps','endpoint-foreign-id','endpoint-seed-string','endpoint-pairs','cell-pairs',
    'cell-block-bool','path-count','identity-count','schema','scope','extra-field'])
def test_reject_identity_before_native_access(tmp_path,name):
    a,p=fixture();mutations(a)[name](a)
    with pytest.raises(ValueError):bind(a,p,tmp_path)
