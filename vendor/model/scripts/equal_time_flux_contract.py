"""Complete acquisition admission for the equal-physical-duration matrix study."""
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256
from numerical_validation import load_json_strict

REPO = Path(__file__).resolve().parents[1]

def inventory(study):
    from run_equal_time_flux import DESIGN, SOURCES
    p = load_json_strict((study/'protocol.json').read_text())
    if any(p.get(k) != v for k, v in DESIGN.items()) or set(p['source_sha256']) != set(SOURCES):
        raise ValueError('Changed complete equal-time design')
    checked = {'protocol.json': sha256(study/'protocol.json')}
    for n, h in p['source_sha256'].items():
        if sha256(REPO/n) != h or sha256(study/'executed-source'/n) != h:
            raise ValueError('Changed acquisition source')
    for n, h in p['input_sha256'].items():
        if sha256(n) != h: raise ValueError('Changed acquisition input')
    parent = Path(p['panel_reuse']['parent'])
    with np.load(parent/'selection.npz') as z:
        order = z['order'].copy()
    manifests = []
    for j in p['jobs']:
        folder = study/'runs'/j['run_id']; m = load_json_strict((folder/'manifest.json').read_text())
        t = 8*j['heads']
        if m['status'] != 'complete' or m['job'] != j or m['protocol_sha256'] != checked['protocol.json']:
            raise ValueError('Incomplete or foreign path')
        for name, expected in {'executed_updates': t, 'scientific_updates': t-8, 'replay_updates': 8,
                               'native_forwards': 2*t+1, 'outgoing_consumed_documents': 32*(j['steps']+t)}.items():
            if type(m[name]) is not int or m[name] != expected: raise ValueError('Wrong path count')
        if set(m['artifacts']) != {'observations.npz','matrices.npy'}: raise ValueError('Wrong artifact census')
        for n, h in m['artifacts'].items():
            if sha256(folder/n) != h: raise ValueError('Changed raw acquisition')
            checked[str((folder/n).relative_to(study))] = h
        checked[str((folder/'manifest.json').relative_to(study))] = sha256(folder/'manifest.json')
        with np.load(folder/'observations.npz') as z:
            rows = z['document_rows'].ravel()
            if not np.array_equal(rows, order[32*j['steps']:32*(j['steps']+t)]) or len(np.unique(rows)) != len(rows):
                raise ValueError('Wrong or repeated source suffix')
            if not np.array_equal(z['optimizer_steps'], np.arange(j['steps'],j['steps']+t+1)):
                raise ValueError('Wrong optimizer clock')
            if z['target_nll'].shape != (t+1,8) or not np.isfinite(z['target_nll']).all(): raise ValueError('Invalid target observations')
        a = np.load(folder/'matrices.npy',mmap_mode='r')
        if a.dtype != np.float32 or a.shape != (t+1,8,5,j['heads'],64,64): raise ValueError('Wrong matrix census')
        if m['peak_allocated_bytes'] >= p['memory_ceiling_bytes'] or m['elapsed_seconds'] >= p['seconds_per_path']:
            raise ValueError('Resource qualification failed')
        manifests.append(m)
    q=load_json_strict((study/'qualification.json').read_text())
    if q['status']!='passed' or q['protocol_sha256']!=checked['protocol.json'] or q['manifest_sha256']!=checked['runs/'+p['jobs'][-1]['run_id']+'/manifest.json']:
        raise ValueError('Wrong widest-path qualification')
    checked['qualification.json']=sha256(study/'qualification.json')
    return p, manifests, checked
