"""Bounded campaign accounting checks; CUDA labels allocate no GPU."""
import copy
from decimal import Decimal
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from confirm import campaign_budget as budget
from confirm.campaign_plan_executor import execute_plan
from confirm.resource_executor import ResourceCaps, canonical_digest, write_json_atomic


def plan_at(tmp_path, *, seconds=60, device='cuda:0', code=None):
    bundle = tmp_path/'campaign'
    bundle.mkdir(parents=True, exist_ok=True)
    nodes = []
    for i in range(2):
        output = bundle/f'output{i}.txt'
        command = code or f"import time;from pathlib import Path;time.sleep(.04);Path({str(output)!r}).write_text('ok')"
        nodes.append(dict(id=f'n{i}', stage=f's{i}', depends_on=[] if i == 0 else ['n0'],
                          command=[sys.executable, '-c', command], device=device,
                          caps=vars(ResourceCaps(2, 2*1024**3, 0, 20*1024**2, .02)),
                          required_inputs=[], expected_outputs=[str(output)], projected_output_bytes=256))
    return dict(campaign_id='bounded-fixture', bundle_root=str(bundle), nodes=nodes,
                resource_budget=dict(hard_aggregate_gpu_hours=seconds/3600,
                                     persistent_output_cap_bytes=20*1024**2))


def run(plan, ids=None, dry=False, cont=False):
    return execute_plan(plan, stages=None, node_ids=ids, dry_run=dry,
                        continue_on_failure=cont, resource_schema='fixture-resource-v1', execution_schema='fixture-report-v1')


def fake_settle(journal, node, elapsed, *, status='within_caps', cleanup=True):
    identity, folder = journal.reserve(node)
    journal.mark_launch(identity)
    record = dict(campaign_contract=budget.CONTRACT, campaign_id=journal.plan['campaign_id'],
                  plan_sha256=journal.digest, node_id=node['id'], attempt_id=identity,
                  command=node['command'], command_sha256=canonical_digest(node['command']),
                  caps=vars(ResourceCaps(**node['caps'])), device=node['device'],
                  total_supervisor_seconds=elapsed, cleanup_complete=cleanup,
                  cap_status=status, deadline_outcome='before_deadline')
    write_json_atomic(folder/'resource.json', record)
    return journal.settle(node, identity, record, False)


def test_split_resume_real_processes(tmp_path):
    plan = plan_at(tmp_path)
    first = run(plan, {'n0'})
    assert first['nodes'][0]['status'] == 'passed'
    second = run(plan, {'n1'})
    assert second['nodes'][0]['status'] == 'passed'
    assert second['aggregate_gpu_seconds'] > first['aggregate_gpu_seconds'] > 0
    resume = run(plan)
    assert [n['status'] for n in resume['nodes']] == ['already_completed']*2
    assert resume['aggregate_gpu_seconds'] == second['aggregate_gpu_seconds']
    assert resume['attempt_count'] == 2


@pytest.mark.parametrize('launcher',['block_normal','source_resolved','orbitwise'])
def test_feasible_separate_cli_invocations(tmp_path,launcher):
    plan = plan_at(tmp_path)
    script = Path(__file__).resolve().parents[2]/'scripts'/('execute_'+launcher+'_plan.py')
    if str(script.parent) not in sys.path:sys.path.insert(0,str(script.parent))
    spec = importlib.util.spec_from_file_location('block_cli', script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    plan['campaign_id'] = module.CAMPAIGN_ID
    plan['schema_version'] = 'pldr-block-normal-launch-plan-v1' if launcher=='block_normal' else module.LAUNCH_PLAN_SCHEMA
    bundle = Path(plan['bundle_root']); protocol = bundle/'protocol'; protocol.mkdir()
    asset = bundle/'input'; asset.write_text('bound input')
    for name in ('dataset', 'tokenizer', 'registry'):
        plan[name] = dict(path=str(asset), sha256=budget.file_sha(asset))
    plan['orders'] = [] if launcher=='block_normal' else {}
    path = protocol/'launch_plan.json'; path.write_text(json.dumps(plan))
    reports = []
    for stage in ('s0', 's1'):
        output = bundle/f'{stage}.json'
        result = subprocess.run([sys.executable,'-B',str(script),'--plan',str(path),
                                 '--stages',stage,'--output',str(output)],capture_output=True,text=True,timeout=15)
        assert result.returncode == 0, result.stderr
        reports.append(json.loads(output.read_text()))
    assert reports[1]['aggregate_gpu_seconds'] > reports[0]['aggregate_gpu_seconds'] > 0
    assert reports[1]['attempt_count'] == 2
    plan['resource_budget']['hard_aggregate_gpu_hours'] = 1
    path.write_text(json.dumps(plan))
    rejected = bundle/'rejected.json'
    result = subprocess.run([sys.executable,'-B',str(script),'--plan',str(path),
                             '--nodes','n0','--continue-on-failure','--output',str(rejected)],
                            capture_output=True,text=True,timeout=15)
    assert result.returncode == 1
    assert json.loads(rejected.read_text())['campaign_decision'] == 'rejected'



@pytest.mark.parametrize('split',[False,True])
def test_original_overbudget_envelope_never_launches(tmp_path, split):
    plan = plan_at(tmp_path,seconds=1)
    for node in plan['nodes']:node['caps']['wall_seconds'] = 5
    reports = [run(plan, {'n0'},cont=True), run(plan, {'n1'},cont=True)] if split else [run(plan,cont=True)]
    assert all(r['aggregate_gpu_seconds'] == 0 and r['attempt_count'] == 0 for r in reports)
    assert reports[0]['campaign_decision'] == 'rejected'
    assert not (Path(plan['bundle_root'])/'output0.txt').exists()


@pytest.mark.parametrize('device,count',[('cpu',0),('cuda:0',1),('cuda:0,cuda:1',2)])
def test_exact_charge_including_failures_and_retries(tmp_path, device, count):
    plan=plan_at(tmp_path, device=device)
    with budget.Journal(plan).locked() as journal:
        fake_settle(journal,plan['nodes'][0],.5,status='nonzero_exit')
        fake_settle(journal,plan['nodes'][0],.25)
        assert journal.charged == Decimal('.75')*count
        assert len(journal.meta['attempts']) == 2
    with budget.Journal(plan).locked() as journal:
        assert journal.charged == Decimal('.75')*count


@pytest.mark.parametrize('remainder,allowed',[('11.999999999',False),('12',True),('12.000000001',True),('0',False)])
def test_reservation_exact_boundary(tmp_path,remainder,allowed):
    plan=plan_at(tmp_path)
    journal=budget.Journal(plan)
    journal.budget=Decimal(20);journal.charged=Decimal(20)-Decimal(remainder)
    if allowed:assert journal.feasible(plan['nodes'][0]) == 12
    else:
        with pytest.raises(budget.CampaignError,match='remaining'):journal.feasible(plan['nodes'][0])


def test_split_and_combined_controlled_cost(tmp_path):
    for split in (True,False):
        plan=plan_at(tmp_path/str(split),seconds=15)
        if split:
            with budget.Journal(plan).locked() as journal:fake_settle(journal,plan['nodes'][0],4)
            with budget.Journal(plan).locked() as journal:
                assert journal.charged == 4
                with pytest.raises(budget.CampaignError):journal.reserve(plan['nodes'][1])
        else:
            with budget.Journal(plan).locked() as journal:
                fake_settle(journal,plan['nodes'][0],4)
                with pytest.raises(budget.CampaignError):journal.reserve(plan['nodes'][1])


@pytest.mark.parametrize('phase',['reserved','launched','recorded'])
def test_interrupted_settlement_blocks_restart(tmp_path,phase):
    plan=plan_at(tmp_path)
    with budget.Journal(plan).locked() as journal:
        identity, folder=journal.reserve(plan['nodes'][0])
        if phase in ('launched','recorded'):journal.mark_launch(identity)
        if phase=='recorded':(folder/'resource.json').write_text('{}')
    report=run(plan,cont=True)
    assert report['campaign_decision']=='rejected'
    assert 'unresolved' in report['campaign_diagnostic']
    assert Decimal(report['reserved_device_seconds_exact']) == 12
    assert report['attempt_count']==1


def test_concurrent_invocation_rejected(tmp_path):
    plan=plan_at(tmp_path)
    with budget.Journal(plan).locked():
        report=run(plan,cont=True)
        assert report['campaign_decision']=='rejected'
        assert 'busy' in report['campaign_diagnostic']
    assert not (Path(plan['bundle_root'])/'output0.txt').exists()


@pytest.mark.parametrize('change',['command','input','device','caps','budget','id'])
def test_full_plan_identity_including_unselected_nodes(tmp_path,change):
    plan=plan_at(tmp_path);assert run(plan,{'n0'})['campaign_decision']=='admissible'
    altered=copy.deepcopy(plan)
    if change=='command':altered['nodes'][1]['command'][-1]+=';pass'
    elif change=='input':altered['dataset']={'path':'foreign','sha256':'0'*64}
    elif change=='device':altered['nodes'][1]['device']='cuda:1'
    elif change=='caps':altered['nodes'][1]['caps']['wall_seconds']=3
    elif change=='budget':altered['resource_budget']['hard_aggregate_gpu_hours']=1
    else:altered['campaign_id']='foreign'
    report=run(altered,{'n1'})
    assert report['campaign_decision']=='rejected' and 'changed' in report['campaign_diagnostic']


@pytest.mark.parametrize('damage',['missing','malformed','duplicate','orphan','resource','settlement','nonobject'])
def test_corrupt_history_fails_closed(tmp_path,damage):
    plan=plan_at(tmp_path);run(plan,{'n0'})
    journal=budget.Journal(plan);meta=budget._read(journal.root/'journal.json')
    folder=journal.root/'attempts'/meta['attempts'][0]
    if damage=='missing':(folder/'reservation.json').unlink()
    elif damage=='malformed':(journal.root/'journal.json').write_text('bad')
    elif damage=='duplicate':
        meta['attempts']*=2;budget._save(journal.root/'journal.json',meta)
    elif damage=='orphan':(journal.root/'attempts'/'orphan').mkdir()
    elif damage=='resource':(folder/'resource.json').write_text('{}')
    elif damage=='nonobject':(folder/'resource.json').write_text('null')
    else:(folder/'settlement.json').write_text('{}')
    assert run(plan,cont=True)['campaign_decision']=='rejected'


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1,True,'1',None])
def test_invalid_budget_before_launch(tmp_path,value):
    plan=plan_at(tmp_path);plan['resource_budget']['hard_aggregate_gpu_hours']=value
    with pytest.raises(budget.CampaignError):run(plan)


@pytest.mark.parametrize('device',['cuda','cuda:-1','cuda:0junk','cuda:0,cuda:0','gpu:1',None])
def test_invalid_device_before_launch(tmp_path,device):
    plan=plan_at(tmp_path,device=device)
    with pytest.raises(budget.CampaignError):run(plan)


@pytest.mark.parametrize('value',[float('nan'),float('inf'),True,-1])
def test_invalid_cap_before_launch(tmp_path,value):
    plan=plan_at(tmp_path);plan['nodes'][1]['caps']['wall_seconds']=value
    with pytest.raises(budget.CampaignError):run(plan,{'n0'})


@pytest.mark.parametrize('kind',['excess','cleanup','timeout'])
def test_resource_invalidity_persists_and_cost_is_not_clamped(tmp_path,kind):
    plan=plan_at(tmp_path)
    with budget.Journal(plan).locked() as journal:
        with pytest.raises(budget.CampaignError):
            fake_settle(journal,plan['nodes'][0],13 if kind=='excess' else .5,
                        cleanup=kind!='cleanup',status='reached_wall_seconds' if kind=='timeout' else 'within_caps')
        assert journal.charged == (13 if kind=='excess' else Decimal('.5'))
    report=run(plan,cont=True)
    assert report['campaign_decision']=='rejected' and report['attempt_count']==1


def test_failed_child_is_charged_and_retry_does_not_overwrite(tmp_path):
    plan=plan_at(tmp_path,code="import sys;sys.exit(2)")
    first=run(plan,{'n0'});second=run(plan,{'n0'})
    assert first['nodes'][0]['status']==second['nodes'][0]['status']=='failed'
    assert second['aggregate_gpu_seconds']>first['aggregate_gpu_seconds']>0
    assert second['attempt_count']==2
    assert run(plan,{'n1'})['nodes'][0]['status']=='blocked_dependency'


def test_real_timeout_halts_continue_on_failure(tmp_path):
    plan=plan_at(tmp_path,code='import time;time.sleep(3)')
    plan['nodes'][0]['caps']['wall_seconds']=.1
    report=run(plan,cont=True)
    assert report['campaign_decision']=='rejected'
    assert report['aggregate_gpu_seconds']>0 and report['attempt_count']==1


def test_dry_run_does_not_consume_or_launch(tmp_path):
    plan=plan_at(tmp_path)
    report=run(plan,dry=True)
    assert [n['status'] for n in report['nodes']]==['ready','ready']
    assert report['aggregate_gpu_seconds']==0 and report['attempt_count']==0
    assert not budget.Journal(plan).root.exists()
    assert not (Path(plan['bundle_root'])/'runtime').exists()
    run(plan,{'n0'})
    root=budget.Journal(plan).root
    before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    run(plan,dry=True)
    assert before=={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_legacy_latest_node_is_not_zero_cost_history(tmp_path):
    plan=plan_at(tmp_path)
    path=budget.control_root(plan['bundle_root'])/'runtime/n0/resource.json'
    path.parent.mkdir(parents=True);path.write_text('{}')
    assert 'legacy' in run(plan)['campaign_diagnostic']


def test_zero_budget_blocks_cpu_too(tmp_path):
    plan=plan_at(tmp_path,seconds=0,device='cpu')
    report=run(plan,cont=True)
    assert report['campaign_decision']=='rejected' and report['attempt_count']==0


def test_dry_run_reports_real_unresolved_reservation(tmp_path):
    plan=plan_at(tmp_path)
    with budget.Journal(plan).locked() as journal:journal.reserve(plan['nodes'][0])
    report=run(plan,dry=True)
    assert report['campaign_decision']=='rejected'
    assert Decimal(report['reserved_device_seconds_exact'])==12


def test_frozen_bundles_include_campaign_modules():
    scripts=Path(__file__).resolve().parents[2]/'scripts'
    for name in ('stage_block_normal_confirmation.py','stage_source_resolved_confirmation.py'):
        text=(scripts/name).read_text()
        assert '"experiments/confirm/campaign_budget.py"' in text
        assert '"experiments/confirm/campaign_plan_executor.py"' in text
