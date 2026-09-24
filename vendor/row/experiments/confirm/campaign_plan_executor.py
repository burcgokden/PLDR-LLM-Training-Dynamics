"""Shared archival-plan executor with durable campaign admission."""
from decimal import Decimal
from pathlib import Path
from .campaign_budget import Journal, CampaignError, CONTRACT
from .resource_executor import (ResourceCaps, directory_bytes, run_capped,
                                write_json_atomic, resource_record_path, record_is_admissible)


def execute_plan(plan, *, stages, node_ids, dry_run, continue_on_failure,
                 resource_schema, execution_schema):
    selected = [n for n in plan['nodes'] if (stages is None or n['stage'] in stages)
                and (node_ids is None or n['id'] in node_ids)]
    if not selected:
        raise CampaignError('the launch selection contains no nodes')
    journal = Journal(plan, dry_run)
    bundle = journal.bundle
    statuses, rows = {}, []
    by_id = {n['id']: n for n in plan['nodes']}
    selected_ids = {n['id'] for n in selected}
    planned_outputs = {p for n in selected for p in n['expected_outputs']}
    decision, diagnostic = 'admissible', None
    projected_reservations = Decimal(0)
    try:
        with journal.locked():
            for node in selected:
                row = dict(id=node['id'], stage=node['stage'])
                if journal.completed(node):
                    status = 'already_completed'
                else:
                    blocked = [d for d in node['depends_on'] if
                               not (statuses.get(d) in ({'passed', 'already_completed', 'ready'} if dry_run else {'passed', 'already_completed'})
                                    if d in selected_ids else journal.completed(by_id[d]))]
                    missing = [p for p in node['required_inputs'] if not
                               (Path(p).is_file() or (Path(p).is_dir() and any(Path(p).rglob('*')))
                                or (dry_run and p in planned_outputs))]
                    if blocked:
                        status = 'blocked_dependency'
                        row['blocked_dependencies'] = blocked
                    elif missing:
                        status = 'blocked_missing_input'
                        row['missing_inputs'] = missing
                    else:
                        journal.feasible(node)
                        if directory_bytes(bundle) + node.get('projected_output_bytes', 0) >= plan['resource_budget']['persistent_output_cap_bytes']:
                            raise CampaignError('campaign storage allowance exhausted before launch')
                        if dry_run:
                            status = 'ready'
                            # Simulate the full envelope of the selection, without writing attempts.
                            reservation = journal.feasible(node)
                            journal.reserved += reservation
                            projected_reservations += reservation
                            row['command'] = node['command']
                        else:
                            identity, folder = journal.reserve(node)
                            journal.mark_launch(identity)
                            record = run_capped(node['command'], device=node['device'],
                                output_root=bundle/'runtime'/node['id']/identity,
                                record_path=folder/'resource.json', caps=ResourceCaps(**node['caps']),
                                environment={'PYTHONPATH': str(bundle/'source'/'experiments'),
                                             'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPYCACHEPREFIX': '/dev/null'},
                                monitor_root=bundle, record_schema_version=resource_schema)
                            record.update(campaign_contract=CONTRACT, campaign_id=plan['campaign_id'],
                                          plan_sha256=journal.digest, node_id=node['id'], attempt_id=identity)
                            write_json_atomic(folder/'resource.json', record)
                            # fsync the bound record before the immutable settlement is committed.
                            with (folder/'resource.json').open('rb') as stream:
                                __import__('os').fsync(stream.fileno())
                            outputs = all(Path(p).is_file() for p in node['expected_outputs'])
                            passed = record_is_admissible(record) and outputs and directory_bytes(bundle) < plan['resource_budget']['persistent_output_cap_bytes']
                            settlement = journal.settle(node, identity, record, passed)
                            record['expected_output_sha256'] = settlement['expected_output_sha256']
                            # Compatibility copy only; admission always consults the complete journal.
                            write_json_atomic(resource_record_path(bundle, node['id']), record)
                            status = 'passed' if settlement['node_passed'] else 'failed'
                            row.update(attempt_id=identity, resource_record=str(folder/'resource.json'),
                                       expected_outputs_present=outputs, aggregate_gpu_seconds=float(journal.charged))
                row['status'] = status
                rows.append(row)
                statuses[node['id']] = status
                if status == 'failed' and not continue_on_failure:
                    break
    except (CampaignError, OSError, ValueError, KeyError, TypeError) as error:
        decision, diagnostic = 'rejected', str(error)
        # Reservation survives an exception after launch. Never infer zero cost.
        rows.append(dict(id=node['id'] if 'node' in locals() else None,
                         stage=node['stage'] if 'node' in locals() else None,
                         status='failed_campaign_admission', reason=diagnostic))
    report = dict(schema_version=execution_schema, campaign_id=plan['campaign_id'],
                  dry_run=dry_run, nodes=rows, campaign_decision=decision,
                  campaign_diagnostic=diagnostic, scientific_outcomes_used_for_execution=False,
                  **journal.summary())
    # Dry-run reservations are hypothetical and are not consumption.
    if dry_run:
        report['prospective_reserved_device_seconds_exact'] = report['reserved_device_seconds_exact']
        report['reserved_device_seconds_exact'] = str(journal.reserved - projected_reservations)
    return report
