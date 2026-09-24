"""Atomic native-run accounting, independent of model allocation."""
import traceback
import uuid
from pathlib import Path
import torch
from model_rg.provenance import sha256, write_json

TERMINAL = {'complete', 'numerical_failure', 'resource_failure', 'program_error', 'incomplete'}


def failure_kind(exc):
    if isinstance(exc, FloatingPointError):
        return 'numerical_failure'
    if isinstance(exc, (MemoryError, torch.OutOfMemoryError)):
        return 'resource_failure'
    return 'program_error'


def clip_finite_norm(parameters):
    """Translate only PyTorch's explicit nonfinite norm diagnostic."""
    try:
        return torch.nn.utils.clip_grad_norm_(parameters, 1., foreach=False, error_if_nonfinite=True)
    except RuntimeError as exc:
        if 'non-finite' in str(exc) and 'cannot be clipped' in str(exc):
            raise FloatingPointError(str(exc)) from exc
        raise


def start_record(destination, binding):
    record = dict(binding, outcome_schema='native-run-outcome-v1', attempt_id=uuid.uuid4().hex,
                  retry_parent=None, status='initializing', stage='initialization',
                  attempted_steps=0, completed_steps=0, scientific_updates=0,
                  qualification_updates=0, available_snapshot_steps=[],
                  checkpoint_available=False, checkpoint_resumable=False,
                  optimizer_partial_mutation=False, artifacts={}, artifact_save_errors=[],
                  original_failure=None, error=None)
    write_json(destination/'binding.json', binding)
    write_json(destination/'manifest.json', record)
    return record


def record_failure(record, exc):
    info = dict(type=type(exc).__name__, message=str(exc), stage=record['stage'],
                traceback=''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    if record['original_failure'] is None:
        record.update(status=failure_kind(exc), original_failure=info, error=str(exc))
    return info


def save_artifact(destination, record, name, writer):
    """Preserve the first failure if an optional artifact writer also fails."""
    try:
        writer()
        record['artifacts'][name] = sha256(destination/name)
        return None
    except Exception as exc:
        info = record_failure(record, exc)
        record['artifact_save_errors'].append(dict(artifact=name, **info))
        write_json(destination/'manifest.json', record)
        return exc


def inventory(study, protocol):
    """Reconcile every declared job without inventing terminal worker output."""
    rows = []
    ph = sha256(Path(study)/'protocol.json')
    for job in protocol['jobs']:
        path = Path(study)/'runs'/job['run_id']/'manifest.json'
        if not path.exists():
            rows.append(dict(run_id=job['run_id'], status='incomplete', completed_steps=None,
                             reason='missing terminal manifest'))
            continue
        import json
        m = json.loads(path.read_text())
        if m['job'] != job or m['protocol_sha256'] != ph:
            raise ValueError('Foreign run manifest')
        status = m['status'] if m['status'] in TERMINAL else 'incomplete'
        rows.append(dict(run_id=job['run_id'], status=status, completed_steps=m['completed_steps']))
    counts = {s: sum(r['status'] == s for r in rows) for s in sorted(TERMINAL)}
    result = dict(schema='native-outcome-inventory-v1', protocol_sha256=ph,
                  declared_paths=len(rows), outcomes=rows, status_counts=counts,
                  status='complete' if counts['complete'] == len(rows) else 'incomplete')
    write_json(Path(study)/'execution-outcomes.json', result)
    return result
