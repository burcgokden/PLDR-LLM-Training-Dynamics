"""Publication pointers must resolve from all four delivered README locations."""
import json
from pathlib import Path
import pytest
from scripts.release_documentation import readmes, check_links


def test_location_specific_links(tmp_path):
    repo=tmp_path/'repo';repo.mkdir();(repo/'docs').mkdir()
    output=tmp_path/'paper-outputs/release';output.mkdir(parents=True)
    (repo/'publication-release.json').write_text(json.dumps(dict(schema='modelrg-release-v1',release_root='../paper-outputs/release',body='docs/body.md')))
    (repo/'docs/body.md').write_text('Completed science.\n')
    for name in ['main.pdf','arxiv-source.zip','MANIFEST.sha256']:(output/name).write_text('fixture')
    source=output/'arxiv-source';source.mkdir();code=source/'code';code.mkdir()
    documents=readmes(repo,output)
    for location,folder in [('repository',repo),('release',output),('source',source),('code',code)]:
        assert len(check_links(documents[location],folder))==3
    (output/'main.pdf').unlink()
    with pytest.raises(ValueError,match='Broken'):check_links(documents['repository'],repo)
    (output/'main.pdf').write_text('fixture')
    staged=tmp_path/'staged';output.rename(staged)
    for location,folder in [('repository',repo),('release',staged),('source',staged/'arxiv-source'),('code',staged/'arxiv-source/code')]:
        report=check_links(documents[location],folder,{output:staged})
        assert len(report)==3 and all(isinstance(link,str) for link in report)
        assert json.loads(json.dumps(report))==report
    (staged/'main.pdf').unlink()
    with pytest.raises(ValueError,match='Broken'):check_links(documents['repository'],repo,{output:staged})


def test_descriptor_does_not_allow_another_destination(tmp_path):
    (tmp_path/'publication-release.json').write_text(json.dumps(dict(schema='modelrg-release-v1',release_root='correct',body='body.md')))
    with pytest.raises(ValueError,match='destination'):readmes(tmp_path,tmp_path/'wrong')
