"""Native factorial entrypoint tests, ordinary and optimized Python.

Temporary profile/protocol modifications below are adverse fixtures only.
They are never written to the retained study or used for native updates.
"""
from companion_paths import child_pythonpath
import argparse
import os
import subprocess
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--mode',action='store_true');args=parser.parse_args()
STUDY=args.study;HERE=args.output.parent
if not args.mode:
    args.output.mkdir(parents=True,exist_ok=False)
    for label,flags in [('ordinary',[]),('optimized',['-O'])]:
        cmd=[sys.executable,*flags,str(Path(__file__).resolve()),'--study',str(STUDY),'--output',str(args.output/(label+'.json')),'--mode']
        with (args.output/(label+'.log')).open('w') as log:
            subprocess.run(cmd,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=child_pythonpath("model"),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
    import hashlib
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    modes=[json.loads((args.output/(label+'.json')).read_text()) for label in ['ordinary','optimized']]
    result=dict(status='passed',schema='factorial-admission-v1',checks=[v for m in modes for v in m['checks']],checker_sha256=sha(__file__),files={label+'.json':sha(args.output/(label+'.json')) for label in ['ordinary','optimized']},scientific_updates=0,study=str(STUDY),protocol_sha256=sha(STUDY/'protocol.json'),profile_sha256=sha(STUDY/'profile/reference1/manifest.json'))
    (args.output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(len(result['checks']),'factorial admission checks passed');sys.exit(0)

sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

worker = module('factorial_worker', REPO / 'scripts/run_potential_factorial.py')
verifier = module('factorial_verifier', REPO / 'scripts/verify_potential_factorial.py')
guard = __import__('factorial_admission')
original = json.loads((STUDY / 'protocol.json').read_text())
profile_path = STUDY / 'profile/reference1'
profile_original = json.loads((profile_path / 'manifest.json').read_text())
cuda_calls = []
class NativeBoundary(Exception):
    pass
def stop(*args, **kwargs):
    raise NativeBoundary('Positive control reached TrainingModel')
worker.TrainingModel = stop
for name in ['set_device', 'init', 'reset_peak_memory_stats']:
    setattr(worker.torch.cuda, name, lambda *args, _name=name, **kwargs: cuda_calls.append(_name))

cases = [
    'valid_worker', 'valid_profile', 'valid_queue', 'missing_profile', 'stale_profile',
    'incomplete_profile', 'nonprofile_role', 'profile_step_count', 'profile_width_identity',
    'profile_bad_gradient', 'profile_bad_forward', 'profile_bad_replay', 'profile_bad_digest',
    'profile_bad_artifact', 'profile_source', 'profile_memory', 'profile_branch_family',
    'scientific_two_steps', 'profile_two_steps', 'wrong_profile_case', 'wrong_case_family',
    'duplicate_case', 'wrong_intervention', 'missing_source_inventory', 'stale_source',
    'selection_digest', 'incoming_digest', 'corpus_digest', 'corpus_shape', 'corpus_dtype',
    'corpus_manifest', 'profile_corpus_digest', 'verifier_corpus_digest',
    'verifier_corpus_shape', 'verifier_corpus_dtype', 'verifier_corpus_manifest',
    'invalid_role','destination_role_conflict','changed_policy','reserved_source','source_policy',
    'profile_array_shape','profile_targets','coordinate_shape','probe_domain','source_suffix',
    'parent_history','existing_destination','verifier_corpus_bytes','verifier_actual_shape','verifier_actual_dtype']
records = []
start = time.monotonic()
for name in cases:
    spec, profile = copy.deepcopy(original), copy.deepcopy(profile_original)
    is_profile = name in ['valid_profile', 'profile_two_steps', 'wrong_profile_case', 'profile_corpus_digest']
    selected_case = 'reference1' if is_profile else 'early-h2'
    expected = name.startswith('valid_')
    if name == 'invalid_role':spec['role']='diagnostic'
    if name == 'changed_policy':spec['runtime_policy']['tf32']=True
    if name == 'reserved_source':spec['reserved_updates']=0
    if name == 'source_policy':spec['source_policy']='replacement'
    if name == 'parent_history':next(c for c in spec['cases'] if c['name']=='early-h2')['parent_history_sha256']='0'*64
    if name == 'scientific_two_steps': spec['steps'] = 2
    if name == 'profile_two_steps': spec['profile_steps'] = 2
    if name == 'wrong_profile_case': selected_case = 'early-h2'
    if name == 'wrong_case_family': spec['cases'][0]['name'] = 'unknown'
    if name == 'duplicate_case': spec['cases'][0] = spec['cases'][1]
    if name == 'wrong_intervention': spec['branches'][0]['offset'] = 1e-3
    if name == 'missing_source_inventory': spec['producer_sources'].pop('scripts/factorial_admission.py')
    if name == 'stale_source': spec['producer_sources']['scripts/factorial_admission.py'] = '0' * 64
    if name == 'selection_digest': spec['selection_sha256'] = '0' * 64
    if name == 'incoming_digest':
        next(c for c in spec['cases'] if c['name'] == 'early-h2')['checkpoint_sha256'] = '0' * 64
    if name.endswith('corpus_digest'): spec['token_sha256'] = '0' * 64
    if name.endswith('corpus_shape'): spec['corpus']['shape'] = [1, 1]
    if name.endswith('corpus_dtype'): spec['corpus']['dtype'] = 'int64'
    if name.endswith('corpus_manifest'): spec['corpus']['manifest_sha256'] = '0' * 64
    if name == 'incomplete_profile': profile['status'] = 'incomplete'
    if name == 'nonprofile_role': profile['profile'] = False
    if name == 'profile_step_count': profile['steps_per_branch'] = 2
    if name == 'profile_width_identity': profile['case']['name'] = 'early-h2'
    if name == 'profile_bad_gradient': profile['qualification']['unchanged_offset_gradient_bitwise'] = False
    if name == 'profile_bad_forward': profile['qualification']['unchanged_offset_forward_bitwise'] = False
    if name == 'profile_bad_replay': profile['branches']['replay']['complete_state_replay_bitwise'] = False
    if name == 'profile_bad_digest': profile['branches']['replay']['final_state_sha256'] = '0' * 64
    if name == 'profile_bad_artifact': profile['branches']['native_keep']['artifact_sha256'] = '0' * 64
    if name == 'profile_source': profile['producer_sources'] = {}
    if name == 'profile_memory': profile['max_cuda_memory_bytes'] = 21 * 1024**3
    if name == 'profile_branch_family': profile['branches']['extra'] = profile['branches']['replay']
    cuda_calls.clear()
    with tempfile.TemporaryDirectory(prefix='admission-fixture-', dir=HERE) as td:
        dest = Path(td)
        shutil.copy2(STUDY / 'selection.npz', dest / 'selection.npz')
        if name in ['coordinate_shape','probe_domain','source_suffix']:
            import numpy as np
            with np.load(dest/'selection.npz') as z:selected={k:z[k] for k in z.files}
            if name=='coordinate_shape':selected['coordinates']=selected['coordinates'][:-1]
            if name=='probe_domain':selected['probes'][0,0]=-1
            if name=='source_suffix':selected['early-h2_rows'][0,0]=(selected['early-h2_rows'][0,0]+1)%524288
            np.savez_compressed(dest/'selection.npz',**selected)
            spec['selection_sha256']=guard.sha256(dest/'selection.npz')
        if name=='destination_role_conflict':(dest/'ROLE.json').write_text(json.dumps(dict(role='scientific' if spec['role']=='qualification' else 'qualification')))
        real_root=verifier.ROOT
        if name in ['verifier_corpus_bytes','verifier_actual_shape','verifier_actual_dtype']:
            import numpy as np
            root=dest/'inputs';corpus=root/'data/refinedweb-onepass-524288';corpus.mkdir(parents=True)
            cp=corpus/'tokens.npy';mp=corpus/'manifest.json'
            dtype='int64' if name=='verifier_actual_dtype' else 'int32'
            if name=='verifier_actual_dtype':
                array=np.lib.format.open_memmap(cp,mode='w+',dtype=dtype,shape=(524288,513));array.flush();del array
            else:np.save(cp,np.zeros((1,1),dtype=dtype))
            manifest=dict(status='complete',shape=[524288,513],dtype='int32',tokens_sha256=guard.sha256(cp))
            mp.write_text(json.dumps(manifest))
            spec['corpus'].update(path=str(cp.resolve()),manifest=str(mp.resolve()),manifest_sha256=guard.sha256(mp),bytes=cp.stat().st_size)
            spec['token_sha256']=spec['corpus']['sha256']=guard.sha256(cp)
            if name=='verifier_corpus_bytes':
                with cp.open('r+b') as f:f.seek(-1,2);f.write(b'X')
            verifier.ROOT=root
        (dest / 'protocol.json').write_text(json.dumps(spec))
        profile['protocol_sha256'] = '0' * 64 if name == 'stale_profile' else guard.sha256(dest / 'protocol.json')
        if name != 'missing_profile' and not is_profile:
            out = dest / 'profile/reference1'
            out.mkdir(parents=True)
            (out / 'manifest.json').write_text(json.dumps(profile))
            for p in profile_path.glob('*.npz'):
                if name in ['profile_array_shape','profile_targets'] and p.name=='native_keep.npz':
                    import numpy as np
                    with np.load(p) as z:raw={k:z[k] for k in z.files}
                    if name=='profile_array_shape':raw['final_logits']=raw['final_logits'][:1]
                    else:raw['targets']=raw['targets']+1
                    np.savez_compressed(out/p.name,**raw)
                    profile['branches']['native_keep']['artifact_sha256']=guard.sha256(out/p.name)
                    (out/'manifest.json').write_text(json.dumps(profile))
                else:(out / p.name).symlink_to(p)
        if name=='existing_destination':(dest/'runs/early-h2').mkdir(parents=True)
        try:
            if name.startswith('verifier_'):
                verifier.main(dest, dest / 'verification.json')
            elif name == 'valid_queue':
                # All queue preflights run before the first logs-directory creation.
                original_mkdir = Path.mkdir
                def sentinel_mkdir(path, *args, **kwargs):
                    if path == dest / 'logs':
                        raise NativeBoundary('Positive queue passed all case preflights before log creation')
                    return original_mkdir(path, *args, **kwargs)
                Path.mkdir = sentinel_mkdir
                try: worker.queue(dest)
                finally: Path.mkdir = original_mkdir
            else:
                worker.run_case(dest, selected_case, 'cpu', profile=is_profile)
        except NativeBoundary as exc:
            outcome, detail = 'admitted', str(exc)
        except (ValueError, KeyError, FileNotFoundError, FileExistsError) as exc:
            outcome, detail = 'rejected', str(exc)
        else:
            raise RuntimeError('Unexpected completed native execution')
        verifier.ROOT=real_root
        created = ((dest / 'runs').exists() and name!='existing_destination') or (dest / 'logs').exists() or (dest / 'verification.json').exists()
        if (outcome == 'admitted') != expected or created or (not expected and cuda_calls):
            raise RuntimeError((name, outcome, detail, created, cuda_calls))
        if 'corpus_' in name and 'orpus' not in detail:
            raise RuntimeError('Corpus test was rejected for an unrelated reason: ' + detail)
        records.append(dict(name=name, admitted=(outcome=='admitted'), expected_admit=expected, detail=detail,
                            before_native_and_cuda=(not cuda_calls), output_created=created))
result = dict(status='passed', optimized=not __debug__, tests=len(records),
              positives=sum(r['admitted'] for r in records),
              negatives=sum(not r['admitted'] for r in records), checks=records,
              optimizer_updates=0, scientific_updates=0, seconds=time.monotonic()-start,
              protocol_sha256=guard.sha256(STUDY / 'protocol.json'),
              profile_manifest_sha256=guard.sha256(profile_path / 'manifest.json'))
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
