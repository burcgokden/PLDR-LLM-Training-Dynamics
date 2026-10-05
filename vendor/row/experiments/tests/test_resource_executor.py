"""Bounded CPU tests of resource admission, terminal accounting, and cleanup."""
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from confirm import resource_executor as executor

BASE=executor.ResourceCaps(5.0,2*1024**3,0,1024**2,.5)


def run(tmp_path, code, caps=BASE, **kwargs):
    return executor.run_capped([sys.executable,'-c',code],device='cpu',
        output_root=tmp_path/'payload',record_path=tmp_path/'control'/'resource.json',
        caps=caps,**kwargs)


@pytest.mark.parametrize('field',['wall_seconds','poll_seconds','host_rss_bytes','gpu_reserved_bytes','output_bytes'])
@pytest.mark.parametrize('value',[float('nan'),float('inf'),float('-inf'),True,False,'1',None])
def test_invalid_domains_before_creation_or_launch(tmp_path,field,value):
    with patch.object(executor.subprocess,'Popen') as launch:
        with pytest.raises(ValueError):run(tmp_path,'pass',replace(BASE,**{field:value}))
        launch.assert_not_called()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('field',['host_rss_bytes','gpu_reserved_bytes','output_bytes'])
@pytest.mark.parametrize('value',[1.5,1.0,-1])
def test_invalid_byte_caps(tmp_path,field,value):
    with pytest.raises(ValueError):run(tmp_path,'pass',replace(BASE,**{field:value}))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('field,value',[('wall_seconds',0),('wall_seconds',-1),('poll_seconds',.009),('poll_seconds',61),('host_rss_bytes',0),('output_bytes',0)])
def test_invalid_signs_and_poll_range(tmp_path,field,value):
    with pytest.raises(ValueError):run(tmp_path,'pass',replace(BASE,**{field:value}))
    assert not list(tmp_path.iterdir())


def test_exit_between_polls_includes_final_4096_bytes(tmp_path):
    record=run(tmp_path,"import time;from pathlib import Path;time.sleep(.15);Path('data').write_bytes(b'x'*4096)",replace(BASE,output_bytes=1024))
    assert record['cap_status']=='reached_output_bytes'
    assert record['final_output_bytes']==record['peak_output_bytes']==4096
    assert not record['technical_valid']


def test_wall_deadline_shorter_than_poll(tmp_path):
    start=time.monotonic()
    record=run(tmp_path,"import time;from pathlib import Path;time.sleep(.15);Path('data').write_bytes(b'x'*4096)",replace(BASE,wall_seconds=.05))
    assert record['cap_status']=='reached_wall_seconds'
    assert record['technical_valid'] is False
    assert record['cleanup_complete']
    assert time.monotonic()-start<3


def guardian_startup_hook(tmp_path, *, release_on_cancel):
    hooks=tmp_path/'startup-hooks';hooks.mkdir()
    entered=tmp_path/'guardian-startup.pid'
    (hooks/'sitecustomize.py').write_text(f'''import os, sys, time
from pathlib import Path
if Path(sys.argv[0]).name == 'resource_worker.py':
    Path({str(entered)!r}).write_text(str(os.getpid()))
    cancel=Path(sys.argv[1]).with_suffix('.cancel')
    limit=time.monotonic()+10
    while time.monotonic()<limit:
        if {release_on_cancel!r} and cancel.is_file():
            break
        time.sleep(.005)
''')
    return entered, {'PYTHONPATH':str(hooks),'PYTHONDONTWRITEBYTECODE':'1'}


@pytest.mark.parametrize('wall',[.05,.1])
def test_deadline_during_guardian_startup_cancels_before_launch(tmp_path,wall):
    entered,environment=guardian_startup_hook(tmp_path,release_on_cancel=True)
    start=time.monotonic()
    record=run(tmp_path,"from pathlib import Path;Path('worker-started').touch()",
               replace(BASE,wall_seconds=wall),environment=environment)
    assert record['cap_status']=='reached_wall_seconds'
    assert not record['technical_valid'] and record['cleanup_complete']
    assert record['deadline_outcome']=='reached_or_unobserved'
    assert record['sample_count']==0 and record['exit_code'] is None
    assert not (tmp_path/'payload/worker-started').exists()
    assert not Path('/proc',entered.read_text()).exists()
    assert time.monotonic()-start<3


def test_unresponsive_guardian_startup_is_bounded_and_not_admitted(tmp_path):
    entered,environment=guardian_startup_hook(tmp_path,release_on_cancel=False)
    start=time.monotonic()
    record=run(tmp_path,"from pathlib import Path;Path('worker-started').touch()",
               replace(BASE,wall_seconds=.05),environment=environment)
    assert record['cap_status']=='reached_wall_seconds'
    assert not record['technical_valid'] and not record['cleanup_complete']
    assert not (tmp_path/'payload/worker-started').exists()
    assert not Path('/proc',entered.read_text()).exists()
    assert time.monotonic()-start<4


def test_guardian_rejects_launch_permission_after_deadline(tmp_path):
    result=tmp_path/'result.json'
    result.with_suffix('.launch').touch()
    worker=Path(executor.__file__).with_name('resource_worker.py')
    subprocess.run([sys.executable,'-B',str(worker),str(result),
                    str(time.monotonic()-1),sys.executable,'-c',
                    "from pathlib import Path;Path('worker-started').touch()"],
                   cwd=tmp_path,check=True,timeout=5)
    record=json.loads(result.read_text())
    assert record['cleanup_complete'] and record['exit_code'] is None
    assert record['error'].startswith('TimeoutError:')
    assert not (tmp_path/'worker-started').exists()


def test_success_and_control_accounting(tmp_path):
    record=run(tmp_path,"from pathlib import Path;Path('data').write_bytes(b'x'*7);print('abc')")
    assert executor.record_is_admissible(record)
    assert record['final_output_bytes']==11
    assert record['peak_output_bytes']==11
    assert record['wall_seconds']==record['completion_observed_seconds']
    assert record['total_supervisor_seconds']>=record['wall_seconds']
    assert json.loads((tmp_path/'control/resource.json').read_text())==record


def test_nonzero_exit(tmp_path):
    record=run(tmp_path,'raise SystemExit(7)')
    assert record['exit_code']==7 and record['cap_status']=='nonzero_exit'
    assert not record['technical_valid'] and record['final_output_bytes']==0


def test_launch_failure_is_recorded(tmp_path):
    record=executor.run_capped(['/no/such/pldr-worker'],device='cpu',output_root=tmp_path/'payload',record_path=tmp_path/'resource.json',caps=BASE)
    assert record['cap_status']=='launch_failed' and not record['technical_valid']
    assert record['cleanup_complete'] and record['guardian_error']


def test_outer_launch_failure_is_recorded(tmp_path):
    with patch.object(executor.subprocess,'Popen',side_effect=OSError('cannot launch')):
        record=run(tmp_path,'pass')
    assert record['cap_status']=='launch_failed' and record['measurement_error']
    assert not record['technical_valid'] and record['final_output_bytes']==0


def test_final_equality_reaches_output_cap(tmp_path):
    record=run(tmp_path,"from pathlib import Path;Path('data').write_bytes(b'x'*1024)",replace(BASE,output_bytes=1024))
    assert record['cap_status']=='reached_output_bytes'
    assert not record['technical_valid']


def test_deadline_exact_boundary_and_cleanup_clock(tmp_path):
    assert not executor.deadline_reached(.049999,replace(BASE,wall_seconds=.05))
    assert executor.deadline_reached(.05,replace(BASE,wall_seconds=.05))
    assert executor.deadline_reached(.050001,replace(BASE,wall_seconds=.05))
    record=run(tmp_path,'pass')
    record['total_supervisor_seconds']=100
    assert executor.record_is_admissible(record)  # cleanup clock is separate
    record['completion_observed_seconds']=record['wall_seconds']=record['caps']['wall_seconds']
    assert not executor.record_is_admissible(record)


def test_alternate_monitor_and_preexisting_payload(tmp_path):
    payload=tmp_path/'payload';payload.mkdir();(payload/'existing').write_bytes(b'e'*512)
    record=executor.run_capped([sys.executable,'-c',"from pathlib import Path;Path('new').write_bytes(b'n'*600)"],
        device='cpu',output_root=payload/'worker',record_path=tmp_path/'control.json',
        monitor_root=payload,caps=replace(BASE,output_bytes=1024))
    assert record['initial_output_bytes']==512 and record['final_output_bytes']==1112
    assert record['cap_status']=='reached_output_bytes'


def test_exhausted_payload_no_launch(tmp_path):
    payload=tmp_path/'payload';payload.mkdir();(payload/'existing').write_bytes(b'e'*1024)
    with patch.object(executor.subprocess,'Popen') as launch:
        with pytest.raises(ValueError,match='exhausted'):run(tmp_path,'pass',replace(BASE,output_bytes=1024))
        launch.assert_not_called()


def test_record_and_worker_domains(tmp_path):
    with pytest.raises(ValueError,match='outside'):
        executor.run_capped(['true'],device='cpu',output_root=tmp_path/'payload',record_path=tmp_path/'payload/control.json',caps=BASE)
    with pytest.raises(ValueError,match='within'):
        executor.run_capped(['true'],device='cpu',output_root=tmp_path/'worker',monitor_root=tmp_path/'payload',record_path=tmp_path/'control.json',caps=BASE)


@pytest.mark.parametrize('detach',[False,True])
def test_surviving_descendant_cannot_write_after_record(tmp_path,detach):
    code="""import os,time
from pathlib import Path
pid=os.fork()
if pid==0:
    DETACH
    time.sleep(30)
    Path('late').write_text('must not appear')
    os._exit(0)
Path('descendant.pid').write_text(str(pid))
""".replace('DETACH','os.setsid()' if detach else 'pass')
    record=run(tmp_path,code)
    assert not record['technical_valid']
    assert record['cap_status'] in ('surviving_descendants','detached_descendants')
    assert record['cleanup_complete']
    pid=int((tmp_path/'payload/descendant.pid').read_text())
    with pytest.raises(ProcessLookupError):os.kill(pid,0)
    assert not (tmp_path/'payload/late').exists()


def test_failed_gpu_probe(tmp_path):
    def probe(_):raise RuntimeError('probe unavailable')
    record=run(tmp_path,'import time;time.sleep(2)',gpu_probe=probe)
    assert record['cap_status']=='gpu_probe_failed' and record['gpu_probe_error']
    assert not record['technical_valid'] and record['cleanup_complete']


def test_stalled_probe_obeys_deadline(tmp_path):
    def probe(_):time.sleep(10);return 0,0
    start=time.monotonic()
    record=run(tmp_path,'import time;time.sleep(2)',replace(BASE,wall_seconds=.1),gpu_probe=probe)
    assert record['cap_status']=='reached_wall_seconds'
    assert not record['technical_valid'] and record['cleanup_complete']
    assert time.monotonic()-start<3


def test_missing_cuda_probe(tmp_path):
    with patch.object(executor.shutil,'which',return_value=None):
        record=executor.run_capped([sys.executable,'-c','import time;time.sleep(1)'],device='cuda:0',
            output_root=tmp_path/'payload',record_path=tmp_path/'record.json',caps=replace(BASE,gpu_reserved_bytes=100))
    assert record['cap_status']=='gpu_probe_failed' and not record['technical_valid']


def test_external_gpu_queries_have_remaining_deadline(monkeypatch):
    monkeypatch.setattr(executor.shutil,'which',lambda _:'/nvidia-smi')
    timeouts=[]
    def query(command,**kwargs):
        timeouts.append(kwargs['timeout'])
        return subprocess.CompletedProcess(command,0,'0, GPU-A\n' if len(timeouts)==1 else '')
    monkeypatch.setattr(executor.subprocess,'run',query)
    executor._nvidia_smi_process_bytes('cuda:0',{1},time.monotonic()+.1)
    assert len(timeouts)==2 and all(0<t<=.1 for t in timeouts)


def test_failed_terminal_measurement(tmp_path,monkeypatch):
    measure=executor.directory_bytes
    def fail_on_result(path):
        if (Path(path)/'done').exists():raise PermissionError('terminal payload unreadable')
        return measure(path)
    monkeypatch.setattr(executor,'directory_bytes',fail_on_result)
    record=run(tmp_path,"from pathlib import Path;Path('done').touch()")
    assert record['final_output_bytes'] is None
    assert not record['technical_valid'] and record['measurement_error']


def test_bad_rss_measurement(tmp_path,monkeypatch):
    def fail(_):raise PermissionError('RSS unreadable')
    monkeypatch.setattr(executor,'_process_rss_bytes',fail)
    record=run(tmp_path,'import time;time.sleep(1)')
    assert record['cap_status']=='measurement_failed' and not record['technical_valid']
    assert record['cleanup_complete']


@pytest.mark.parametrize('key,value',[('technical_valid',False),('cap_status','reached_output_bytes'),('cleanup_complete',False),('final_output_bytes',None),('exit_code',1),('deadline_outcome','reached_or_unobserved'),('measurement_error','failed'),('peak_output_bytes',10**12),('peak_host_rss_bytes',10**12),('surviving_descendants',[123]),('completion_observed_seconds',float('nan'))])
def test_validator_rejects_inconsistent_success(tmp_path,key,value):
    record=run(tmp_path,'pass');assert executor.record_is_admissible(record)
    record[key]=value
    assert not executor.record_is_admissible(record)


def test_legacy_record_never_upgraded():
    old={'technical_valid':True,'schema_version':executor.DEFAULT_RESOURCE_RECORD_SCHEMA}
    assert not executor.record_is_admissible(old)
    assert executor.record_is_admissible(old,require_current=False)


def load_script(name):
    path=Path(__file__).resolve().parents[2]/'scripts'/name
    if str(path.parent) not in sys.path:sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location('resource_consumer_'+path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('name',['execute_block_normal_plan.py','execute_source_resolved_plan.py','execute_direct_work_plan.py','execute_direct_work_replication.py','execute_radial_context_holdout.py'])
def test_campaign_consumers_require_consistent_current_admission(tmp_path,name):
    module=load_script(name);campaign=tmp_path/'campaign';node_id='bounded';output=campaign/'result'
    schema=getattr(module,'RESOURCE_SCHEMA',executor.DEFAULT_RESOURCE_RECORD_SCHEMA)
    command=[sys.executable,'-c',"from pathlib import Path;Path('result').write_text('ok')"]
    path=executor.resource_record_path(campaign,node_id)
    record=executor.run_capped(command,device='cpu',output_root=campaign,record_path=path,caps=BASE,record_schema_version=schema)
    record.update(node_id=node_id,campaign_id=getattr(module,'CAMPAIGN_ID','fixture'),
                  expected_output_sha256={str(output):__import__('hashlib').sha256(output.read_bytes()).hexdigest()})
    node=dict(id=node_id,node_id=node_id,command=command,expected_outputs=[str(output)],output=str(output),device='cpu',role='fixture',phase='qualification')
    if name=='execute_direct_work_plan.py':
        module._rewrite_resource_clock(record,node,path)
        admitted=lambda:module._completed(campaign,node)
    elif name in ('execute_direct_work_replication.py','execute_radial_context_holdout.py'):
        module.rewrite_resource(record,node,path,{'write_json_atomic':executor.write_json_atomic})
        admitted=lambda:module.completed(campaign,node,schema)
    elif name=='execute_block_normal_plan.py':admitted=lambda:module._completed_node_valid(campaign,node)
    else:admitted=lambda:module._completed(campaign,node)
    executor.write_json_atomic(path,record)
    assert admitted()
    record['cap_status']='reached_output_bytes'  # even a contradictory true flag fails
    executor.write_json_atomic(path,record)
    assert not admitted()
    record['cap_status']='within_caps';record.pop('executor_contract')
    executor.write_json_atomic(path,record)
    assert not admitted()  # old acquisition is not an execution of this contract


@pytest.mark.parametrize('name,function',[('analyze_direct_work_confirmation','_resource_summary'),('analyze_direct_work_replication','resource_summary'),('analyze_radial_context_holdout','resource_summary')])
def test_reducer_cannot_admit_current_failed_execution(tmp_path,name,function):
    module=__import__('analysis.'+name,fromlist=[function])
    campaign=tmp_path/'campaign';campaign.mkdir()
    path=executor.resource_record_path(campaign,'invalid')
    executor.write_json_atomic(path,{'executor_contract':executor.EXECUTOR_CONTRACT,'technical_valid':False})
    with pytest.raises(ValueError,match='resource admission failed'):
        getattr(module,function)(campaign,[])


@pytest.mark.parametrize('kind',['host','gpu'])
def test_sampled_memory_equality_reaches_cap(tmp_path,monkeypatch,kind):
    if kind=='host':
        monkeypatch.setattr(executor,'_process_rss_bytes',lambda _:BASE.host_rss_bytes)
        record=run(tmp_path,'import time;time.sleep(1)')
        assert record['cap_status']=='reached_host_rss_bytes'
    else:
        record=run(tmp_path,'import time;time.sleep(1)',replace(BASE,gpu_reserved_bytes=128),gpu_probe=lambda _:(128,128))
        assert record['cap_status']=='reached_gpu_reserved_bytes'
    assert record['cleanup_complete'] and not record['technical_valid']


def test_incomplete_cleanup_cannot_admit(tmp_path,monkeypatch):
    stop=executor._stop_guardian
    def incomplete(process,result):stop(process,result);return False
    monkeypatch.setattr(executor,'_stop_guardian',incomplete)
    record=run(tmp_path,'pass')
    assert record['cap_status']=='cleanup_failed' and not record['technical_valid']


def test_current_record_matches_required_schema(tmp_path):
    schema=json.loads((Path(__file__).resolve().parents[1]/'protocols/block_normal_confirmation/resource_record_schema.json').read_text())
    record=run(tmp_path,'pass')
    assert set(schema['required'])<=set(record)
    assert record['executor_contract']==schema['properties']['executor_contract']['const']


def test_new_frozen_sources_include_guardian():
    for name in ('stage_direct_work_confirmation.py','stage_direct_work_replication.py','stage_radial_context_holdout.py'):
        module=load_script(name)
        sources=module.source_files() if hasattr(module,'source_files') else module.SOURCE_FILES
        assert Path('experiments/confirm/resource_executor.py') in sources
        assert Path('experiments/confirm/resource_worker.py') in sources
