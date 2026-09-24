"""Observed resource admission with a monotonic deadline and terminal accounting.

See docs/RESOURCE_EXECUTION.md for scope, versioning, and payload boundaries.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import multiprocessing
import numbers
import stat
import sys
import tempfile
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
from typing import Any


DEFAULT_RESOURCE_RECORD_SCHEMA = "pldr-block-normal-resource-record-v1"
EXECUTOR_CONTRACT = "pldr-resource-execution-v2"


@dataclass(frozen=True)
class ResourceCaps:
    wall_seconds: float
    host_rss_bytes: int
    gpu_reserved_bytes: int
    output_bytes: int
    poll_seconds: float = 1.0

    def validate(self) -> None:
        for name in ('wall_seconds', 'poll_seconds'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,numbers.Real) or not math.isfinite(value):
                raise ValueError(f'{name} must be a finite real number')
        if self.wall_seconds<=0 or not .01<=self.poll_seconds<=60:
            raise ValueError('wall time must be positive; poll interval must be in [0.01, 60]')
        for name in ('host_rss_bytes','gpu_reserved_bytes','output_bytes'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int):
                raise ValueError(f'{name} must be an integer byte count, not a Boolean or float')
            if value < (0 if name=='gpu_reserved_bytes' else 1):
                raise ValueError(f'{name} is outside its domain')


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def directory_bytes(path: str | Path) -> int:
    """Sum logical regular-file bytes. Required reads fail closed; no symlinks."""
    root=Path(path)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('monitored payload must be an existing real directory')
    total=0
    def failed(error):raise error
    for base,dirs,names in os.walk(root,followlinks=False,onerror=failed):
        for name in dirs+names:
            item=Path(base)/name
            mode=item.lstat()
            if stat.S_ISLNK(mode.st_mode):
                raise ValueError(f'payload symlink is outside the accounting contract: {item}')
            if stat.S_ISREG(mode.st_mode):total+=mode.st_size
            elif not stat.S_ISDIR(mode.st_mode):
                raise ValueError(f'non-regular payload entry: {item}')
    return total


def control_root(monitored: str | Path) -> Path:
    root=Path(monitored).resolve()
    return root.with_name(root.name+'.resource-control')


def resource_record_path(monitored: str | Path, node_id: str) -> Path:
    if not node_id or Path(node_id).name!=node_id or node_id in ('.','..'):
        raise ValueError('resource node identifier must be a single path component')
    return control_root(monitored)/'runtime'/node_id/'resource.json'


def resource_records(monitored: str | Path):
    """Read-only discovery of both historical and current family records."""
    return sorted(set((Path(monitored)/'runtime').glob('*/resource.json')) |
                  set((control_root(monitored)/'runtime').glob('*/resource.json')))


def write_json_atomic(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _process_rss_bytes(process_id: int) -> int:
    status = Path(f"/proc/{process_id}/status")
    if not status.is_file():
        return 0
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


def _process_tree_pids(process_id: int) -> set[int]:
    """Return the live Linux process tree rooted at the given process."""

    pending = [int(process_id)]
    selected: set[int] = set()
    while pending:
        current = pending.pop()
        if current in selected:
            continue
        selected.add(current)
        children = Path(f"/proc/{current}/task/{current}/children")
        if not children.is_file():
            continue
        try:
            pending.extend(int(value) for value in children.read_text().split())
        except (FileNotFoundError, ProcessLookupError):
            continue
    return selected


def _pid_namespace_candidates(
    process_ids: set[int],
) -> tuple[set[int], dict[str, list[int]]]:
    """Return current and best-effort ancestor-namespace PIDs from ``NSpid``."""

    candidates = {int(value) for value in process_ids}
    translations: dict[str, list[int]] = {}
    for process_id in sorted(process_ids):
        status = Path(f"/proc/{process_id}/status")
        try:
            lines = status.read_text(encoding="utf-8").splitlines()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        for line in lines:
            if not line.startswith("NSpid:"):
                continue
            values = [int(value) for value in line.partition(":")[2].split()]
            if values:
                translations[str(process_id)] = values
                candidates.update(values)
            break
    return candidates, translations


def _nvidia_smi_process_bytes(
    device: str, process_ids: set[int], deadline: float | None = None
) -> tuple[int, int, dict[str, Any]]:
    """Return native CUDA bytes for the selected process tree and devices."""

    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError("CUDA cap enforcement requires nvidia-smi")
    indices = {int(value) for value in re.findall(r"cuda:(\d+)", device)}
    if not indices:
        raise ValueError("CUDA device must contain an explicit index")
    gpu_command = [
        executable,
        "--query-gpu=index,uuid",
        "--format=csv,noheader,nounits",
    ]
    gpu_result = subprocess.run(
        gpu_command,
        check=True,
        capture_output=True,
        text=True,
        timeout=_probe_timeout(deadline),
    )
    target_uuids = set()
    for line in gpu_result.stdout.splitlines():
        index_text, separator, uuid = line.partition(",")
        if separator and int(index_text.strip()) in indices:
            target_uuids.add(uuid.strip())
    if len(target_uuids) != len(indices):
        raise RuntimeError("CUDA device index is not visible to nvidia-smi")
    app_command = [
        executable,
        "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    app_result = subprocess.run(
        app_command,
        check=True,
        capture_output=True,
        text=True,
        timeout=_probe_timeout(deadline),
    )
    pid_candidates, namespace_translation = _pid_namespace_candidates(
        process_ids
    )
    used_mib = 0
    parsed_rows = 0
    target_rows = 0
    matching_rows = 0
    for line in app_result.stdout.splitlines():
        parts = [value.strip() for value in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[1])
            memory = int(parts[2])
        except ValueError:
            continue
        parsed_rows += 1
        if parts[0] not in target_uuids:
            continue
        target_rows += 1
        if pid in pid_candidates:
            matching_rows += 1
            used_mib += memory
    if matching_rows:
        query_status = "matched-monitored-process"
    elif parsed_rows == 0:
        query_status = "empty-process-query"
    elif target_rows == 0:
        query_status = "no-row-on-target-device"
    else:
        query_status = "no-row-matching-monitored-process"
    native_bytes = used_mib * 1024**2
    observation = {
        "gpu_query_command": gpu_command,
        "gpu_query_stdout": gpu_result.stdout,
        "compute_apps_query_command": app_command,
        "compute_apps_query_stdout": app_result.stdout,
        "target_device_indices": sorted(indices),
        "target_gpu_uuids": sorted(target_uuids),
        "monitored_process_ids": sorted(process_ids),
        "namespace_pid_candidates": sorted(pid_candidates),
        "namespace_pid_translation": namespace_translation,
        "namespace_translation_status": (
            "nspid-candidates-added"
            if pid_candidates != process_ids else "no-additional-nspid-candidate"
        ),
        "compute_apps_parsed_row_count": parsed_rows,
        "compute_apps_target_device_row_count": target_rows,
        "compute_apps_matching_process_row_count": matching_rows,
        "process_query_status": query_status,
        "parsed_process_memory_mib": used_mib,
        "namespace_visible": True,
    }
    return native_bytes, native_bytes, observation


def _probe_timeout(deadline):
    remaining=5.0 if deadline is None else min(5.0,deadline-time.monotonic())
    if remaining<=0:raise subprocess.TimeoutExpired('resource probe',0)
    return remaining


def _probe_child(connection, probe, device):
    try:connection.send(('ok',probe(device)))
    except Exception as error:connection.send(('error',f'{type(error).__name__}: {error}'))
    finally:connection.close()


def _bounded_injected_probe(probe, device, deadline):
    # Injection is a CPU-test interface. Isolate it so a stalled callback cannot
    # outlive the deadline or prevent the caller from exiting.
    context=multiprocessing.get_context('fork')
    receive,send=context.Pipe(duplex=False)
    child=context.Process(target=_probe_child,args=(send,probe,device))
    child.start();send.close()
    try:
        if not receive.poll(_probe_timeout(deadline)):
            raise subprocess.TimeoutExpired('injected GPU probe',0)
        status,value=receive.recv()
        if status!='ok':raise RuntimeError(value)
        return value
    finally:
        if child.is_alive():child.kill()
        child.join(timeout=1)
        receive.close()


def _stop_guardian(process):
    if process.poll() is None:
        process.terminate()
    try:process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        try:process.wait(timeout=1)
        except subprocess.TimeoutExpired:return False
        return False
    return True


def deadline_reached(observed_seconds, caps):
    return observed_seconds >= caps.wall_seconds


def record_is_admissible(record, *, require_current=True):
    """Cross-field admission; historical records are never upgraded to v2."""
    if record.get('technical_valid') is not True:return False
    if record.get('executor_contract')!=EXECUTOR_CONTRACT:
        return not require_current and 'executor_contract' not in record
    try:
        caps=ResourceCaps(**record['caps']);caps.validate()
        elapsed=record['completion_observed_seconds']
        wall=record.get('wall_seconds',record.get('attempt_elapsed_wall_seconds'))
        observations=[elapsed,wall,record['total_supervisor_seconds']]
        if any(isinstance(x,bool) or not isinstance(x,numbers.Real) or not math.isfinite(x) or x<0 for x in observations):return False
        output=record['final_output_bytes'];peak=record['peak_output_bytes']
        host=record['peak_host_rss_bytes']
        gpu=record.get('peak_gpu_reserved_bytes',record.get('monitor_peak_gpu_reserved_bytes'))
        if any(isinstance(x,bool) or not isinstance(x,int) or x<0 for x in [output,peak,host,gpu]):return False
        return bool(record['cap_status']=='within_caps' and record['exit_code']==0
            and record['cleanup_complete'] is True and record['deadline_outcome']=='before_deadline'
            and record['measurement_error'] is None and record['gpu_probe_error'] is None
            and wall==elapsed and elapsed<caps.wall_seconds
            and record['total_supervisor_seconds']>=elapsed
            and output<=peak<caps.output_bytes and host<caps.host_rss_bytes
            and (caps.gpu_reserved_bytes==0 or gpu<caps.gpu_reserved_bytes)
            and not record['surviving_descendants'] and not record['detached_descendants'])
    except (KeyError,TypeError,ValueError):return False


def run_capped(command: Sequence[str], *, device: str, output_root: str | Path,
               record_path: str | Path, caps: ResourceCaps,
               environment: dict[str,str] | None = None,
               gpu_probe: Callable[[str],tuple[int,int]] | None = None,
               monitor_root: str | Path | None = None,
               record_schema_version: str = DEFAULT_RESOURCE_RECORD_SCHEMA) -> dict[str,Any]:
    """Admit only observed completion before deadline with quiescent payload.

    Equality reaches a cap. RSS/GPU and directory high-water values are sampled,
    not hard maxima. Worker output and logs belong to the monitored tree; control
    records must be outside it. Linux /proc and a child subreaper are required.
    """
    caps.validate()
    if not isinstance(record_schema_version,str) or not record_schema_version:
        raise ValueError('resource record schema version must be nonempty')
    if isinstance(command,(str,bytes)) or not command or any(not isinstance(x,str) or not x or '\0' in x for x in command):
        raise ValueError('executor command must be a nonempty argv sequence')
    root=Path(output_root).resolve();monitored=root if monitor_root is None else Path(monitor_root).resolve()
    target=Path(record_path).resolve()
    if not root.is_relative_to(monitored):raise ValueError('worker output root must lie within monitored payload')
    if target.is_relative_to(monitored):raise ValueError('supervisor record must be outside the monitored payload')
    if target.exists():raise FileExistsError(f'resource record already exists: {target}')
    if not sys.platform.startswith('linux'):raise RuntimeError('resource supervision requires Linux /proc and subreaper support')
    root.mkdir(parents=True,exist_ok=True)
    initial=directory_bytes(monitored)
    if initial>=caps.output_bytes:raise ValueError('output cap is already exhausted before launch')
    target.parent.mkdir(parents=True,exist_ok=True)
    child_env={**os.environ,**(environment or {})}
    peaks=dict(host=0,allocated=0,reserved=0,output=initial)
    status='within_caps';measurement_error=None;probe_error=None;probe_observation=None
    completion=None;process=None;guardian={};cleanup_complete=False;final_output=None;sample_count=0
    probe_kind=('injected' if gpu_probe is not None else 'nvidia-smi-process-tree'
                if device.startswith('cuda') and caps.gpu_reserved_bytes else 'disabled')
    started_ns=time.time_ns();started=time.monotonic();deadline=started+caps.wall_seconds
    with tempfile.TemporaryDirectory(prefix='guardian-',dir=target.parent) as temp:
        result=Path(temp)/'result.json'
        try:
            with (root/'worker.stdout.log').open('ab') as stdout, (root/'worker.stderr.log').open('ab') as stderr:
                started_ns=time.time_ns();started=time.monotonic();deadline=started+caps.wall_seconds
                process=subprocess.Popen([sys.executable,'-B',str(Path(__file__).with_name('resource_worker.py')),
                                          str(result),*command],cwd=root,env=child_env,
                                         start_new_session=True,stdout=stdout,stderr=stderr)
                while True:
                    exited=process.poll() is not None
                    observed=time.monotonic()-started
                    if exited:completion=observed
                    if deadline_reached(observed,caps):
                        status='reached_wall_seconds';break
                    if exited:break
                    if not result.with_suffix('.ready').is_file():
                        try:process.wait(timeout=min(.005,max(0,deadline-time.monotonic())))
                        except subprocess.TimeoutExpired:pass
                        continue
                    try:
                        size=directory_bytes(monitored)
                        pids=_process_tree_pids(process.pid)
                        host=sum(_process_rss_bytes(pid) for pid in pids)
                        peaks['output']=max(peaks['output'],size);peaks['host']=max(peaks['host'],host)
                        sample_count+=1
                    except Exception as error:
                        measurement_error=f'{type(error).__name__}: {error}';status='measurement_failed';break
                    try:
                        if gpu_probe is not None:
                            allocated,reserved=_bounded_injected_probe(gpu_probe,device,deadline)
                        elif probe_kind=='nvidia-smi-process-tree':
                            allocated,reserved,probe_observation=_nvidia_smi_process_bytes(device,pids,deadline)
                        else:allocated,reserved=0,0
                        if any(isinstance(x,bool) or not isinstance(x,int) or x<0 for x in (allocated,reserved)):
                            raise ValueError('GPU probe must return nonnegative integer bytes')
                        peaks['allocated']=max(peaks['allocated'],allocated);peaks['reserved']=max(peaks['reserved'],reserved)
                    except Exception as error:
                        probe_error=f'{type(error).__name__}: {error}'
                        status='reached_wall_seconds' if time.monotonic()>=deadline else 'gpu_probe_failed';break
                    if time.monotonic()>=deadline:status='reached_wall_seconds';break
                    for key,cap,name in [('host',caps.host_rss_bytes,'host_rss_bytes'),
                                         ('reserved',caps.gpu_reserved_bytes,'gpu_reserved_bytes'),
                                         ('output',caps.output_bytes,'output_bytes')]:
                        if cap and peaks[key]>=cap:status='reached_'+name;break
                    if status!='within_caps':break
                    remaining=deadline-time.monotonic()
                    if remaining<=0:status='reached_wall_seconds';break
                    try:process.wait(timeout=min(caps.poll_seconds,remaining))
                    except subprocess.TimeoutExpired:pass
        except Exception as error:
            measurement_error=f'{type(error).__name__}: {error}';status='launch_failed' if process is None else 'monitoring_failed'
        finally:
            stopped=True if process is None else _stop_guardian(process)
            if result.is_file():
                try:guardian=json.loads(result.read_text())
                except (OSError,ValueError) as error:
                    measurement_error=f'{type(error).__name__}: {error}'
            cleanup_complete=stopped and (process is None or guardian.get('cleanup_complete') is True)
            if completion is None and process is not None and process.poll() is not None:
                completion=time.monotonic()-started
            if status=='within_caps':
                if completion is None or deadline_reached(completion,caps):status='reached_wall_seconds'
                elif not cleanup_complete:status='cleanup_failed'
                elif guardian.get('error'):status='launch_failed'
                elif guardian.get('detached_descendants'):status='detached_descendants'
                elif guardian.get('surviving_descendants'):status='surviving_descendants'
                elif guardian.get('exit_code')!=0:status='nonzero_exit'
            # Always attempt the terminal sample, after bounded workload cleanup.
            try:
                final_output=directory_bytes(monitored)
                peaks['output']=max(peaks['output'],final_output)
                if peaks['output']>=caps.output_bytes and status=='within_caps':status='reached_output_bytes'
            except Exception as error:
                measurement_error=f'{type(error).__name__}: {error}'
                if status=='within_caps':status='measurement_failed'
    elapsed=time.monotonic()-started
    record=dict(schema_version=record_schema_version,executor_contract=EXECUTOR_CONTRACT,
        command=list(command),command_sha256=canonical_digest(list(command)),device=device,
        started_at_ns=started_ns,finished_at_ns=time.time_ns(),
        wall_seconds=completion if completion is not None else elapsed,
        completion_observed_seconds=completion,total_supervisor_seconds=elapsed,
        deadline_outcome='before_deadline' if completion is not None and completion<caps.wall_seconds else 'reached_or_unobserved',
        peak_host_rss_bytes=peaks['host'],peak_gpu_allocated_bytes=peaks['allocated'],
        peak_gpu_reserved_bytes=peaks['reserved'],peak_output_bytes=peaks['output'],
        final_output_bytes=final_output,initial_output_bytes=initial,sample_count=sample_count,
        output_root=str(root),monitored_output_root=str(monitored),record_path=str(target),
        gpu_probe_kind=probe_kind,gpu_probe_error=probe_error,gpu_probe_observation=probe_observation,
        measurement_error=measurement_error,cleanup_complete=cleanup_complete,
        surviving_descendants=guardian.get('surviving_descendants',[]),
        detached_descendants=guardian.get('detached_descendants',[]),guardian_error=guardian.get('error'),
        caps=asdict(caps),cap_status=status,exit_code=guardian.get('exit_code'),technical_valid=False,
        measurement_scope='Logical regular-file bytes including pre-existing payload and worker logs; control records external. RSS/GPU and disk high-water are observations, not transient maxima. Linux subreaper quiescence required.')
    record['technical_valid']=status=='within_caps'
    record['technical_valid']=record_is_admissible(record)
    write_json_atomic(target,record)
    return record
