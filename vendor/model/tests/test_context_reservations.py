"""Independent reservation invariants, including competing OS processes."""
import json
import multiprocessing
from pathlib import Path
import pytest
from scripts.context_reservations import reserve, document_hashes


def fixture(tmp_path):
    root=tmp_path/'root';root.mkdir();probes=tmp_path/'probes';probes.mkdir();corpus=tmp_path/'corpus';corpus.mkdir()
    records=[{'content_sha256':f'{i:064x}'} for i in range(1200)]
    (probes/'records.json').write_text(json.dumps(records));(corpus/'records.json').write_text(json.dumps([records[-1]]))
    old=root/'calibration';old.mkdir();(old/'protocol.json').write_text(json.dumps(dict(schema='operator-cache-v1',document_hashes=[records[1152]['content_sha256']])))
    return root,probes,corpus


def contender(root,probes,corpus,name,queue):
    try:reserve(root/name,8,1160,root=root,probes=probes,corpus=corpus);queue.put('reserved')
    except ValueError:queue.put('rejected')


def test_atomic_collision(tmp_path):
    root,probes,corpus=fixture(tmp_path);ctx=multiprocessing.get_context('fork');queue=ctx.Queue()
    processes=[ctx.Process(target=contender,args=(root,probes,corpus,name,queue)) for name in ['first','second']]
    for p in processes:p.start()
    for p in processes:p.join(15);assert p.exitcode==0
    assert sorted(queue.get(timeout=1) for _ in processes)==['rejected','reserved']
    assert len(json.loads((root/'context-reservations-v1.json').read_text())['reservations'])==1


def test_training_calibration_and_used_content_excluded(tmp_path):
    root,probes,corpus=fixture(tmp_path)
    for start in [0,1152,1199]:
        with pytest.raises(ValueError):reserve(root/f'p{start}',1,start,root=root,probes=probes,corpus=corpus)
    r=reserve(root/'fresh',8,root=root,probes=probes,corpus=corpus)
    assert r['context_rows']==list(range(1153,1161))
    assert not (root/'fresh').exists()


def test_unknown_schema_refused(tmp_path):
    with pytest.raises(ValueError):document_hashes({'schema':'unknown','document_hashes':['a'*64]})
    root,probes,corpus=fixture(tmp_path)
    (root/'calibration/protocol.json').write_text(json.dumps(dict(schema='unknown',document_hashes=['a'*64])))
    with pytest.raises(ValueError):reserve(root/'fresh',8,root=root,probes=probes,corpus=corpus)


def test_named_reproduction_is_explicit(tmp_path):
    root,probes,corpus=fixture(tmp_path);p=root/'named';p.mkdir();records=json.loads((probes/'records.json').read_text());rows=list(range(1160,1168))
    (p/'protocol.json').write_text(json.dumps(dict(schema='cache-risk-v1',context_rows=rows,document_hashes=[records[i]['content_sha256'] for i in rows])))
    with pytest.raises(ValueError):reserve(root/'new',8,1160,root=root,probes=probes,corpus=corpus)
    r=reserve(root/'replay',8,root=root,probes=probes,corpus=corpus,reproduce=p)
    assert r['role']=='named-panel-reproduction' and r['context_rows']==rows
