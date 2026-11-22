"""Read the fixed 429 cohort and preserve every historical Judge request."""

import csv
import os
import sqlite3
import tempfile
from collections import defaultdict, deque
from pathlib import Path

from harness_awm.io import digest, file_hash, read_json, write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = ROOT / 'runs/pending_judge_audit_20261110_v1/pending_inventory.csv'
DEFAULT_REPAIRS = ROOT / 'runs/sft_judge_recovery_20261109_v1'
MAX_CUMULATIVE_REQUESTS = 32
REQUEST_COLUMNS = ('request_key', 'stage', 'reserved', 'status', 'usage', 'started', 'finished')
REQUEST_SCHEMA = '''CREATE TABLE requests (
    id INTEGER PRIMARY KEY, request_key TEXT UNIQUE, stage TEXT, reserved INTEGER,
    status TEXT, usage TEXT, started TEXT, finished TEXT)'''
HOLD_SCHEMA = 'CREATE TABLE holds (hold_key TEXT PRIMARY KEY, reserved INTEGER NOT NULL)'


def load_cohort(inventory_path=DEFAULT_INVENTORY, *, repairs_root=DEFAULT_REPAIRS,
                expected_count=1061):
    """Return the audited 429 rows, interleaving candidates deterministically.

    CSV task_id is a short display identifier. The full task_instance_id from
    task.json is used for the existing recovery-job key. Source paths deliberately
    remain lexical: archived episode symlinks do not move the parent job's budget.
    """
    with Path(inventory_path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'candidate', 'task_id', 'latest_failure', 'source_episode', 'recovery_job'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('cohort_missing_columns')
        selected = [dict(row) for row in reader if row['latest_failure'] == 'RateLimitError']
    if len(selected) != expected_count:
        raise ValueError(f'cohort_count_mismatch:{len(selected)}!={expected_count}')

    # Same record set as sft_judge_recovery.collect_repairs(), without importing
    # training, archiving or execution modules during input preparation.
    historical = {}
    for record_path in sorted((Path(repairs_root) / 'judge_jobs_v1').glob('*/recovery.json')):
        record = read_json(record_path)
        source = record['source_episode']
        if source in historical:
            raise ValueError('duplicate_historical_source:' + source)
        historical[source] = (record, record_path.parent / 'budget.sqlite')

    groups, seen = defaultdict(deque), set()
    for row in selected:
        source = Path(row['source_episode'])
        task = read_json(source / 'task.json')
        task_instance_id = task['task_instance_id']
        if not isinstance(task_instance_id, str) or not task_instance_id:
            raise ValueError('invalid_task_instance_id:' + str(source))
        short_id = f"{task['scenario']}:{task['native_task_id']}"
        if row['task_id'] not in (short_id, task_instance_id):
            raise ValueError('cohort_task_id_mismatch:' + str(source))
        key = digest([row['candidate'], task_instance_id, str(source)])[:24]
        recovery_job = Path(row['recovery_job'])
        if recovery_job.name != key:
            raise ValueError('cohort_recovery_key_mismatch:' + str(recovery_job))
        if key in seen:
            raise ValueError('duplicate_cohort_key:' + key)
        seen.add(key)

        original_budget = source.parents[2] / 'budget.sqlite'
        budgets = [original_budget]
        previous = historical.get(str(source))
        if previous:
            record, budget_path = previous
            if (record['candidate'] != row['candidate']
                    or record['task_instance_id'] != task_instance_id):
                raise ValueError('historical_task_mismatch:' + str(source))
            budgets.append(budget_path)
        for path in budgets:
            if not path.is_file():
                raise FileNotFoundError('missing_historical_budget:' + str(path))
        new_budget = recovery_job / 'budget.sqlite'
        if new_budget.exists():
            budgets.append(new_budget)
        row.update(key=key, task_instance_id=task_instance_id,
                   budget_sources=list(dict.fromkeys(str(path) for path in budgets)))
        groups[row['candidate']].append(row)

    ordered = []
    while any(groups.values()):
        for candidate in sorted(groups):
            if groups[candidate]:
                ordered.append(groups[candidate].popleft())
    return ordered


def merge_budget(paths, destination):
    """Create a new ledger containing the union of immutable source requests.

    Source-local integer IDs are reassigned; every request field is copied
    verbatim. Duplicate request_key values must agree in all other fields.
    The destination must not exist, so reruns cannot overwrite newly spent budget.
    Returned baseline counts include all history, even if the 32-call cap is met.
    The .baseline.json sidecar is committed before the complete SQLite ledger is
    published, so a visible destination always has recoverable baseline metadata.
    """
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('budget_destination_exists:' + str(destination))
    paths = list(dict.fromkeys(str(Path(path).absolute()) for path in paths))
    if not paths:
        raise ValueError('budget_sources_empty')

    merged, sources = {}, []
    for name in paths:
        path = Path(name)
        before = file_hash(path)
        with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
            db.execute('PRAGMA query_only=ON')
            db.execute('BEGIN')
            columns = tuple(row[1] for row in db.execute('PRAGMA table_info(requests)'))
            if columns != ('id',) + REQUEST_COLUMNS:
                raise ValueError('unexpected_budget_schema:' + name)
            if db.execute('SELECT count(*) FROM holds').fetchone()[0]:
                raise ValueError('budget_has_active_holds:' + name)
            rows = db.execute('SELECT * FROM requests ORDER BY id').fetchall()
        if file_hash(path) != before:
            raise RuntimeError('source_budget_changed:' + name)
        for row in rows:
            payload = row[1:]
            key = payload[0]
            if not isinstance(key, str) or not key:
                raise ValueError('invalid_budget_request_key:' + name)
            if key in merged and merged[key] != payload:
                raise ValueError('budget_request_conflict:' + key)
            merged[key] = payload
        sources.append({'path': name, 'sha256': before,
                        'logical_sha256': digest({'requests': rows, 'holds': []}),
                        'physical_requests': len(rows)})

    baseline = {'physical_requests': len(merged),
                'tokens_accounted_or_reserved': sum(row[2] for row in merged.values()),
                'unknown_usage_requests': sum(row[3] != 'response' for row in merged.values()),
                'held_requests': 0, 'held_tokens': 0}
    metadata = {'baseline': baseline, 'sources': sources,
                'max_cumulative_requests': MAX_CUMULATIVE_REQUESTS}
    sidecar = destination.with_suffix('.baseline.json')
    if sidecar.exists() and read_json(sidecar) != metadata:
        raise ValueError('budget_baseline_sidecar_conflict:' + str(sidecar))
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=destination.name + '.',
                                                 suffix='.tmp', dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with sqlite3.connect(temporary) as db:
            db.execute(REQUEST_SCHEMA)
            db.execute(HOLD_SCHEMA)
            db.executemany('INSERT INTO requests (' + ','.join(REQUEST_COLUMNS)
                           + ') VALUES (?,?,?,?,?,?,?)', merged.values())
        if not sidecar.exists():
            write_json(sidecar, metadata)
        # Same-directory hard link publishes atomically and fails if another
        # preparer has already created the destination; it never replaces it.
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return metadata
