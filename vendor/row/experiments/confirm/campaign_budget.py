"""Durable, exclusive campaign admission; see docs/RESOURCE_EXECUTION.md.

The journal authenticates local records against accidental corruption, not a
hostile writer. Unresolved attempts block all launches; recovery is deliberately
manual and cannot be inferred from a missing process or a latest-node record.
"""
from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

from .resource_executor import (ResourceCaps, canonical_digest, control_root,
                                resource_records, record_is_admissible)

CONTRACT = 'pldr-campaign-budget-v1'
# Includes bounded guardian waits and an allowance for terminal measurement.
# A measured overrun invalidates the campaign; this is not an OS hard cutoff.
CLEANUP_SECONDS = Decimal('10')


class CampaignError(ValueError):
    """A global admission failure which continuation cannot override."""


def amount(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise CampaignError('budget/time must be a finite nonnegative number')
    try:
        result = Decimal(str(value))
    except InvalidOperation as error:
        raise CampaignError('invalid budget/time') from error
    if not result.is_finite() or result < 0:
        raise CampaignError('budget/time must be finite and nonnegative')
    return result


def device_count(device):
    if device == 'cpu':
        return 0
    if not isinstance(device, str) or not re.fullmatch(r'cuda:\d+(?:,cuda:\d+)*', device):
        raise CampaignError('device must be cpu or comma-separated cuda:N declarations')
    ids = [int(x.split(':')[1]) for x in device.split(',')]
    if len(ids) != len(set(ids)):
        raise CampaignError('duplicate device declaration')
    return len(ids)


def file_sha(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise CampaignError('expected regular file: ' + str(path))
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _save(path, payload, *, immutable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps({'payload': payload, 'sha256': canonical_digest(payload)},
                         sort_keys=True, allow_nan=False).encode() + b'\n'
    tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('xb') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    if immutable:
        os.link(tmp, path)  # O_EXCL semantics; never replace an event.
        tmp.unlink()
    else:
        os.replace(tmp, path)
    _sync_dir(path.parent)


def _read(path):
    try:
        if path.is_symlink():
            raise CampaignError('journal symlink')
        value = json.loads(path.read_text())
        if not isinstance(value, dict) or set(value) != {'payload', 'sha256'} or not isinstance(value['payload'], dict) or canonical_digest(value['payload']) != value['sha256']:
            raise CampaignError('journal digest mismatch')
        return value['payload']
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise CampaignError('missing or corrupt journal record: ' + str(path)) from error


def validate_plan(plan):
    """Bind the entire plan, including unselected nodes and input identities."""
    try:
        hours = plan['resource_budget']['hard_aggregate_gpu_hours']
        if isinstance(hours, bool) or not isinstance(hours, (int, float)):
            raise CampaignError('campaign budget must be numeric, not Boolean or string')
        budget = amount(hours) * 3600
        output = plan['resource_budget']['persistent_output_cap_bytes']
        if isinstance(output, bool) or not isinstance(output, int) or output <= 0:
            raise CampaignError('invalid campaign output cap')
        if not isinstance(plan['campaign_id'], str) or not plan['campaign_id']:
            raise CampaignError('missing campaign identity')
        seen = set()
        for node in plan['nodes']:
            name = node['id']
            if not isinstance(name, str) or not name or Path(name).name != name or name in ('.', '..') or name in seen:
                raise CampaignError('invalid or duplicate node identity')
            if not isinstance(node['depends_on'], list) or not set(node['depends_on']) <= seen:
                raise CampaignError('dependencies must precede their node')
            seen.add(name)
            ResourceCaps(**node['caps']).validate()
            device_count(node['device'])
            if not isinstance(node['command'], list) or not node['command'] or any(not isinstance(x, str) or not x or '\0' in x for x in node['command']):
                raise CampaignError('invalid command')
            for key in ('required_inputs', 'expected_outputs'):
                if not isinstance(node[key], list) or any(not isinstance(x, str) or not x for x in node[key]):
                    raise CampaignError('invalid input/output list')
            projected = node.get('projected_output_bytes', 0)
            if isinstance(projected, bool) or not isinstance(projected, int) or projected < 0:
                raise CampaignError('invalid projected output size')
        return budget, canonical_digest(plan)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise CampaignError('invalid campaign plan: ' + str(error)) from error


def campaign_record_valid(record, charge, reserve):
    """Node nonzero exit may be retried; resource failures invalidate the campaign."""
    return bool(charge <= reserve and record.get('cleanup_complete') is True
                and record.get('cap_status') in ('within_caps', 'nonzero_exit')
                and record.get('deadline_outcome') == 'before_deadline'
                and record.get('measurement_error') is None
                and record.get('gpu_probe_error') is None
                and not record.get('surviving_descendants')
                and not record.get('detached_descendants'))


class Journal:
    def __init__(self, plan, dry_run=False):
        self.plan = plan
        self.budget, self.digest = validate_plan(plan)
        self.bundle = Path(plan['bundle_root']).resolve()
        self.root = control_root(self.bundle) / 'campaign'
        self.dry_run = dry_run
        self.charged = Decimal(0)
        self.reserved = Decimal(0)
        self.latest = {}
        self.meta = dict(contract=CONTRACT, plan_sha256=self.digest,
                         campaign_id=plan['campaign_id'], budget_seconds=str(self.budget),
                         cleanup_seconds=str(CLEANUP_SECONDS), attempts=[])

    @contextmanager
    def locked(self):
        control = self.root.parent
        control.mkdir(parents=True, exist_ok=True)
        with (control / 'campaign.lock').open('a+b') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise CampaignError('campaign busy: another invocation owns the lock') from error
            try:
                self.load()
                yield self
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def load(self):
        self.charged = Decimal(0)
        self.reserved = Decimal(0)
        self.latest = {}
        path = self.root / 'journal.json'
        if not path.exists():
            if self.root.exists() or resource_records(self.bundle) or any((self.bundle/'runtime').glob('*')):
                raise CampaignError('missing journal or legacy history; explicit reconciliation required')
            return
        meta = _read(path)
        if {k: v for k, v in meta.items() if k != 'attempts'} != {k: v for k, v in self.meta.items() if k != 'attempts'}:
            raise CampaignError('campaign identity, plan, budget or timing contract changed')
        ids = meta.get('attempts')
        if not isinstance(ids, list) or any(not isinstance(x, str) or not re.fullmatch('[0-9a-f]{32}', x) for x in ids) or len(ids) != len(set(ids)):
            raise CampaignError('invalid or duplicate attempt identity')
        attempt_root = self.root / 'attempts'
        actual = {p.name for p in attempt_root.iterdir()} if attempt_root.exists() else set()
        if actual != set(ids):
            raise CampaignError('missing or orphan attempt; reconciliation required')
        self.meta = meta
        invalid = []
        nodes = {n['id']: n for n in self.plan['nodes']}
        for identity in ids:
            folder = attempt_root / identity
            reservation = _read(folder / 'reservation.json')
            node = nodes.get(reservation.get('node_id'))
            if node is None or reservation != self.reservation(node, identity):
                raise CampaignError('attempt reservation identity mismatch')
            reserve = amount(reservation['reserved_seconds'])
            if not (folder / 'settlement.json').exists():
                self.reserved += reserve
                invalid.append('unresolved attempt ' + identity)
                continue
            launch = _read(folder / 'launch.json')
            if launch != dict(attempt_id=identity, plan_sha256=self.digest):
                raise CampaignError('launch identity mismatch')
            settlement = _read(folder / 'settlement.json')
            record_path = folder / 'resource.json'
            record = json.loads(record_path.read_text())
            self._record_identity(node, identity, record)
            if settlement.get('resource_sha256') != file_sha(record_path):
                raise CampaignError('resource record changed')
            charge = amount(record['total_supervisor_seconds']) * device_count(node['device'])
            if settlement.get('charged_seconds') != str(charge) or settlement.get('attempt_id') != identity:
                raise CampaignError('settlement charge/identity mismatch')
            expected_valid = campaign_record_valid(record, charge, reserve)
            if settlement.get('campaign_valid') is not expected_valid:
                raise CampaignError('settlement admission mismatch')
            if not expected_valid:
                invalid.append('excess charge or incomplete cleanup ' + identity)
            self.charged += charge
            self.latest[node['id']] = (settlement, record, record_path)
        if self.charged + self.reserved > self.budget:
            invalid.append('campaign budget exceeded')
        if invalid:
            raise CampaignError('; '.join(invalid))

    def reservation(self, node, identity):
        return dict(attempt_id=identity, node_id=node['id'], plan_sha256=self.digest,
                    command_sha256=canonical_digest(node['command']), device=node['device'],
                    reserved_seconds=str(device_count(node['device']) * (amount(node['caps']['wall_seconds']) + CLEANUP_SECONDS)))

    def feasible(self, node):
        reserve = amount(self.reservation(node, 'unused')['reserved_seconds'])
        # Exhaustion stops CPU work too, as specified by the campaign contract.
        if self.charged + self.reserved >= self.budget or self.charged + self.reserved + reserve > self.budget:
            raise CampaignError('insufficient remaining campaign budget before launch')
        return reserve

    def reserve(self, node):
        if self.dry_run:
            raise CampaignError('dry run cannot reserve')
        reserve = self.feasible(node)
        if not self.root.exists():
            self.root.mkdir()
            _sync_dir(self.root.parent)
            _save(self.root / 'journal.json', self.meta)
        identity = uuid.uuid4().hex
        folder = self.root / 'attempts' / identity
        folder.mkdir(parents=True)
        _sync_dir(folder.parent)
        _sync_dir(self.root)
        _save(folder / 'reservation.json', self.reservation(node, identity), immutable=True)
        self.meta['attempts'].append(identity)
        _save(self.root / 'journal.json', self.meta)
        self.reserved += reserve
        return identity, folder

    def mark_launch(self, identity):
        _save(self.root / 'attempts' / identity / 'launch.json',
              dict(attempt_id=identity, plan_sha256=self.digest), immutable=True)

    def _record_identity(self, node, identity, record):
        if not isinstance(record, dict):
            raise CampaignError('resource record must be an object')
        if any(record.get(k) != v for k, v in dict(campaign_contract=CONTRACT,
                campaign_id=self.plan['campaign_id'], plan_sha256=self.digest,
                node_id=node['id'], attempt_id=identity, device=node['device'],
                command=node['command'], command_sha256=canonical_digest(node['command']),
                caps=vars(ResourceCaps(**node['caps']))).items()):
            raise CampaignError('resource identity or caps mismatch')
        amount(record['total_supervisor_seconds'])

    def settle(self, node, identity, record, node_passed):
        self._record_identity(node, identity, record)
        folder = self.root / 'attempts' / identity
        resource = folder / 'resource.json'
        charge = amount(record['total_supervisor_seconds']) * device_count(node['device'])
        reserve = amount(self.reservation(node, identity)['reserved_seconds'])
        valid = campaign_record_valid(record, charge, reserve)
        settlement = dict(attempt_id=identity, charged_seconds=str(charge), campaign_valid=valid,
                          node_passed=bool(node_passed and valid), resource_sha256=file_sha(resource),
                          expected_output_sha256={p: file_sha(p) for p in node['expected_outputs'] if Path(p).is_file()})
        _save(folder / 'settlement.json', settlement, immutable=True)
        self.load()  # Verify persisted state, including failure cost, before continuing.
        return settlement

    def completed(self, node):
        item = self.latest.get(node['id'])
        if item is None:
            return False
        settlement, record, _ = item
        hashes = settlement.get('expected_output_sha256', {})
        return bool(settlement.get('node_passed') is True and record_is_admissible(record)
                    and set(hashes) == set(node['expected_outputs'])
                    and all(Path(p).is_file() and file_sha(p) == h for p, h in hashes.items()))

    def summary(self):
        return dict(campaign_contract=CONTRACT, plan_sha256=self.digest,
                    aggregate_gpu_seconds=float(self.charged), charged_device_seconds_exact=str(self.charged),
                    reserved_device_seconds_exact=str(self.reserved), hard_aggregate_gpu_seconds=float(self.budget),
                    attempt_count=len(self.meta['attempts']))
