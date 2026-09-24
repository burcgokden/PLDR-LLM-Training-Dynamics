"""A publication must reject missing routes or a newly added package dependency."""
import hashlib,json
from pathlib import Path
import pytest
import scripts.verify_execution_routes as gate


def test_publication_refuses_an_omitted_advertised_route(tmp_path):
    p=tmp_path/'routes.json'
    p.write_text(json.dumps(dict(schema='current-execution-routes-v2',status='passed',routes=[])))
    with pytest.raises(ValueError,match='Missing advertised route'):gate.verify(p)


def test_new_package_module_invalidates_publication_before_native_loading(tmp_path,monkeypatch):
    repo=tmp_path/'repo';package=repo/'src/model_rg';package.mkdir(parents=True)
    (repo/'scripts').mkdir();driver=repo/'scripts/refresh_execution_routes.py';driver.write_text('# frozen driver\n')
    old=package/'existing.py';old.write_text('VALUE=1\n')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    r=dict(schema='current-execution-routes-v2',status='passed',
        routes=[dict(name=n) for n in ['primary','physical','continuations','optimizer','released','factorial']],
        checker_sha256=sha(driver),source_identity={'src/model_rg/existing.py':sha(old)})
    # Every recorded file still matches; adding a file must nevertheless refuse.
    (package/'added_analysis.py').write_text('VALUE=2\n')
    p=tmp_path/'routes.json';p.write_text(json.dumps(r));monkeypatch.setattr(gate,'REPO',repo)
    with pytest.raises(ValueError,match='Package changed after admission'):gate.verify(p)


def test_publication_requires_factorial_in_six_route_contract(tmp_path):
    p=tmp_path/'routes.json'
    p.write_text(json.dumps(dict(schema='current-execution-routes-v2',status='passed',
        routes=[dict(name=n) for n in ['primary','physical','continuations','optimizer','released']])))
    with pytest.raises(ValueError,match='Missing advertised route'):gate.verify(p)
