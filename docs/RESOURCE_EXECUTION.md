# Resource execution contract

The inherited row executor remains an archival campaign interface. Its bounded
CPU regression tests exercise resource admission, not a new scientific campaign.
New records carry `executor_contract: pldr-resource-execution-v2` in addition to
the unchanged family `schema_version`. Historical acquisition records retain
their original bytes and lack this execution version; they are not certified by
the new executor. Current resume/admission requires the v2 contract.

Wall time must be finite and positive; the finite polling interval is in
[0.01, 60] seconds. Byte caps are Python integers, never Boolean, string, or
floating inputs. Host and output caps are positive. A zero GPU cap disables GPU
measurement and admission unless a probe is explicitly injected for testing.
Equality reaches a cap. The completion of the guardian and workload must be
observed strictly before the monotonic deadline, even for a zero worker exit.
`wall_seconds` and `completion_observed_seconds` mean this observation duration;
`total_supervisor_seconds` also includes cleanup and terminal measurement.
Subsequent serialization is outside these timings. Family clock adapters retain
this duration as `attempt_elapsed_wall_seconds`.

All worker payloads and stdout/stderr logs must be inside `monitor_root`;
`output_root` must be inside it. Pre-existing logical file bytes count. Symlinks
and special files in this tree are unsupported and invalidate measurement.
A required unreadable measurement fails admission. Supervisor records live
outside the payload. Campaigns use the sibling `CAMPAIGN.resource-control/`
directory, keeping worker logs in `CAMPAIGN/runtime/NODE/`. Both directories
are needed when transporting new campaign records. Read-only resource summaries
can discover historical in-tree records and new external records. A current
invalid record cannot be admitted merely by a successful numerical output file.

The executor launches a separate Linux subreaper guardian, preserving caller
process settings even for threaded campaign queues. It adopts orphan descendants
and reaps all descendants before successful completion. Surviving or observed
detached descendants invalidate execution and are killed. Cleanup is bounded;
missing guardian evidence or incomplete cleanup cannot produce a valid record.
This is supervision of cooperating scientific programs, not a security sandbox:
workers must not hand writes to unrelated services or write outside the declared
tree. The guardian is part of each newly frozen source bundle.

The wall deadline limits waits independently of the resource sampling interval.
External GPU queries and injected test probes have bounded waits. The final
payload sample runs after cleanup on every termination path. A reached-cap or
failed-probe decision survives finalization. `peak_output_bytes` is the maximum
of initial, sampled, and final sizes; it does not account for files deleted
between observations. RSS and GPU peaks are also sampled observations, not
hard memory ceilings. Workload startup may be shorter than a resource sampling
interval. No claim of a measured continuous-time maximum follows.

`record_is_admissible` checks exit status, completion deadline, cleanup, failure
fields, final/peak accounting, and cap consistency together. The five campaign
families and the orbitwise alias apply this gate to newly executed records.
The family resource schema requires the new fields when creating records; it
is not applied retroactively to old acquisition data.

```sh
PYTHONPATH=vendor/row/experiments python3 -m pytest -q vendor/row/experiments/tests/test_resource_executor.py
```

For the combined suite, use `python3 scripts/run_family_checks.py --family row`.

## Persistent campaign admission

The block-normal and source-resolved launchers, including the orbitwise alias,
share `confirm/campaign_budget.py` and `confirm/campaign_plan_executor.py`.
Their accounting contract is `pldr-campaign-budget-v1`. These remain archival
scientific launchers with bounded CPU accounting validation. Historical
acquisition records are never rewritten or upgraded.

The sibling `CAMPAIGN.resource-control/campaign/` directory binds the **full**
canonical plan digest, campaign identity, device declarations, commands, input
hashes, caps and total budget. Selections do not affect that identity. An
exclusive nonblocking `flock` rejects another invocation of the same campaign.
Distinct campaigns have distinct declared budgets; this is not a machine-wide
scheduler. Preserve the payload and its entire resource-control directory.

Each attempt has a unique directory with durable immutable `reservation.json`,
`launch.json` and `settlement.json` events and a hash-bound `resource.json`.
Atomic writes fsync the file and containing directory. The journal lists every
attempt, so missing, duplicate or orphan attempts fail closed. Digests detect
accidental alteration, not a hostile writer with access to the control tree.
The latest per-node resource copy exists for legacy reducers only; it cannot
supply a campaign history or satisfy execution dependencies by itself.

Write the reservation before marking launch and calling the process executor.
For device count d, reserve d times (wall cap + 10 seconds). The ten seconds
allow guardian cleanup and terminal measurement; actual charged duration is
`total_supervisor_seconds`, from monotonic supervisor start through cleanup
and terminal measurement. Record serialization is outside that interval.
`wall_seconds` remains completion observation and is not renamed. Filesystem
latency and OS scheduling cannot be hard bounded: an observed excess is charged
in full, invalidates the campaign, and blocks future work. No clamp or refund
of unobserved time is permitted. Sampled memory/storage limitations above still
apply. A CPU declaration has d=0; `cuda:0,cuda:1` has d=2 and reserves both for
the whole interval, irrespective of utilization. Duplicate or malformed labels
are rejected; a disabled GPU probe does not disable declared-device charging.

Admission requires C + R + r <= B, for settled cost C, unresolved reservations
R and next reservation r. Decimal arithmetic uses the decimal representation
of each JSON number, with no tolerance: equality is admitted, a larger sum is
rejected. Budgets must be finite nonnegative JSON numbers, never Boolean or
string; all node caps and declarations are validated before any node launches.
An exhausted budget admits no new node, even a CPU node. The conservative
full-envelope rule can reject a job that would have finished quickly.

Successful nodes with unchanged output hashes are skipped. Re-reading history
charges nothing twice. A nonzero-exit attempt retains its charge, and a retry
is a new attempt; its output cannot validate a failed dependency. Timeout,
late completion, failed resource observation, incomplete cleanup or excess
charge invalidates the campaign. `--continue-on-failure` can continue after a
settled nonzero-exit node when unrelated work remains admissible, but cannot
override campaign rejection. The CLI returns nonzero for rejected campaigns,
failed nodes or blocked selections, and the report includes a separate
`campaign_decision` and exact decimal totals.

An interruption before launch, after launch or before settlement leaves an
unresolved reservation which blocks all further launch. A durable settlement
is read idempotently after restart, even if the latest-node copy was not yet
written. There is deliberately no automatic recovery from ambiguous history.
Preserve the journal, establish descendant quiescence and a complete measured
interval externally, and document a separately reviewed reconciliation before
reuse. Do not delete the journal, change the budget or start a replacement
campaign to certify the interrupted history. Legacy overwritten attempts cannot
be reconstructed from surviving latest-node records. Dry runs may create the
external lock file, but write no journal events, launch no child and charge no
consumption; their full-envelope feasibility projection is explicitly separate.

`provenance/campaign-accounting.json` inventories all fourteen row `execute*.py`
entry points and the neighboring family scan. It names inspected functions and
limits. Only the three shared interfaces above have this cumulative contract.
Other source launchers retain their archival execution semantics and offer no
validated interruption-safe cumulative-budget guarantee. In particular,
row-map and layer-cocycle totals are retrospective observations and cannot
certify prelaunch reservation or complete failed-attempt accounting.

```sh
python3 scripts/check_campaign_budget.py
python3 scripts/check_resource_execution.py
```

These routes use deterministic boundaries and tiny temporary CPU children.
CUDA strings in fixtures test declared-device accounting without allocation.
The row family suite discovers both sets of tests; counts overlap.
