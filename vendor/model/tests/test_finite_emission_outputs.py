"""CLI analysis must isolate requested outputs and publication must be explicit."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from scripts import analyze_finite_emission_transport as analyzer

def test_analysis_writes_only_requested_destination(tmp_path, monkeypatch):
    rng=np.random.default_rng(1616)
    for folder in ['potential-factorial-20260913','potential-factorial-disjoint-20260914']:
        study=tmp_path/'inputs'/folder
        run=study/'runs/subcritical1';run.mkdir(parents=True)
        (study/'protocol.json').write_text(json.dumps({'cases':[{'name':'subcritical1'}]}))
        branches={}
        for name in ['native_keep','raised_keep','native_reset','raised_reset']:
            raw=run/(name+'.npz')
            np.savez(raw,step_logits=rng.normal(size=(3,1,7))*.1)
            branches[name]={'artifact_sha256':hashlib.sha256(raw.read_bytes()).hexdigest()}
        (run/'manifest.json').write_text(json.dumps({'branches':branches}))
    publication=tmp_path/'repo/manuscript/generated';publication.mkdir(parents=True)
    sentinel=publication/'finite-kl-transport.tex';sentinel.write_text('Preserve this publication.')
    monkeypatch.setattr(analyzer,'ROOT',tmp_path/'inputs')
    monkeypatch.setattr(analyzer,'REPO',tmp_path/'repo')
    out=tmp_path/'analysis/result.json'
    analyzer.main(out)
    assert out.is_file()
    assert list(publication.iterdir())==[sentinel]
    assert sentinel.read_text()=='Preserve this publication.'
    with pytest.raises(FileExistsError):
        analyzer.main(out)


