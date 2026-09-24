"""Location-aware publication links from one release descriptor and body."""
import json
import re
from pathlib import Path


def readmes(repo, output):
    spec=json.loads((repo/'publication-release.json').read_text())
    if spec['schema']!='modelrg-release-v1' or (repo/spec['release_root']).resolve()!=output.resolve():
        raise ValueError('Release descriptor and destination differ')
    body=(repo/spec['body']).read_text().strip()
    prefixes={'repository':spec['release_root'].rstrip('/')+'/',
              'release':'','source':'../','code':'../../'}
    return {location:'# Model-wide PLDR renormalization\n\n'+
        f'[Manuscript PDF]({prefix}main.pdf) · [Source archive]({prefix}arxiv-source.zip) · [Checksums]({prefix}MANIFEST.sha256)\n\n'+
        body+'\n' for location,prefix in prefixes.items()}


def check_links(text, directory, replacements=None):
    checked=[]
    for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',text):
        if '://' in target or target.startswith('#'):continue
        path=(directory/target.split('#',1)[0]).resolve()
        for release_root, staged in (replacements or {}).items():
            if path.is_relative_to(release_root):path=staged/path.relative_to(release_root)
        if not path.is_file():raise ValueError('Broken generated publication link: '+str(path))
        checked.append(target)
    return checked
