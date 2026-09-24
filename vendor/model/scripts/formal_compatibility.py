"""Hash-preserving compatibility of a scoped formal record with an extended library."""
from pathlib import Path
from model_rg.provenance import sha256


def check_formal_inventory(repo, recorded):
    repo=Path(repo)
    current={str(p.relative_to(repo)):sha256(p) for p in sorted(repo.glob('ModelRG/**/*.lean'))}
    if not recorded or not current:
        raise AssertionError('Empty formal inventory')
    for name,digest in recorded.items():
        if name not in current:
            raise AssertionError('Missing recorded formal module: '+name)
        if current[name]!=digest:
            raise AssertionError('Changed recorded formal module: '+name)
    expected=''.join('import '+'.'.join(Path(name).with_suffix('').parts)+'\n' for name in sorted(current))
    if (repo/'ModelRG.lean').read_text()!=expected:
        raise AssertionError('Current root imports do not match the current module inventory')
    return dict(recorded_modules=recorded,current_modules=current,
                extensions=sorted(set(current)-set(recorded)),
                compatibility='Every recorded module is present with its recorded hash; added modules are checked by the current build and gate.')
