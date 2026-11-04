"""Offline preflight and explicitly authorized recovery of frozen batch03.

No arguments means preflight. Only `run` dispatches model requests.
Frozen v1 implementations and business prompts remain unchanged.
"""
import argparse
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path

import rubric_luna_batch03_production_v1 as p

APPROVAL = p.ROOT/'configs/rubric/luna_high_3000/network_retry_20261102.json'
ACCEPTANCE = p.ROOT/'configs/rubric/luna_high_3000/resume_v2_acceptance.json'
STAGE_TIMEOUT = 1800


def request_key(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def archive_path(row):
    return p.RUN/'interrupted_attempts'/row['request_key']


def matches(folder, row):
    return (folder.is_dir() and
            {f.name:p.digest(f) for f in folder.iterdir() if f.is_file()} == row['files'])


def check_processes(state):
    # Inspect processes separately from unresolved requests: v1 conflates the two.
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().split(b'\0')
        except (OSError, ProcessLookupError):
            continue
        if len(args)>2 and b'codex' in Path(os.fsdecode(args[0])).name.encode() and b'exec' in args[1:3]:
            if any(str(p.RUN/'tasks').encode() in arg for arg in args):
                raise RuntimeError('prior_model_process_still_alive:'+path.parent.name)
    identities = list(state.get('active', {}).values()) + list(state.get('remote_holds', {}).values())
    identities += [p.read(f)['process'] for f in p.RUN.glob('tasks/*/round_*/*/started.json')]
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    for identity in identities:
        if identity.get('boot_id') and identity['boot_id'] != boot:
            continue
        if identity.get('pgid') and p.group_alive(identity['pgid']):
            raise RuntimeError('prior_process_group_still_alive:'+str(identity['pgid']))
        if identity.get('pid') and not p.process_identity(identity['pid']).get('already_exited'):
            raise RuntimeError('prior_process_still_alive:'+str(identity['pid']))


def inspect(approval):
    """Read-only plan. An approval applies to one exact old attempt, never its retry."""
    if approval['run_id'] != p.RUN_ID or approval['manifest_sha256'] != p.digest(p.RUN/'manifest.json'):
        raise RuntimeError('retry_approval_wrong_batch')
    state = p.read(p.RUN/'scheduler_state.json')
    check_processes(state)
    rows = {}
    for row in approval['requests']:
        rel = Path(row['folder'])
        if rel.is_absolute() or '..' in rel.parts or len(rel.parts)!=4 or rel.parts[0]!='tasks':
            raise RuntimeError('invalid_retry_path')
        if request_key(row['identity']) != row['request_key'] or row['request_key'] in rows:
            raise RuntimeError('invalid_retry_identity')
        rows[row['request_key']] = row
    approved, completed, blocked, recovered = [], set(), [], []
    for path in sorted(p.RUN.glob('tasks/*/round_*/*/identity.json')):
        folder = path.parent
        identity = p.read(path)
        key = request_key(identity)
        end = p.read(folder/'completed.json') if (folder/'completed.json').exists() else None
        if end and end['status']=='completed':
            if end['identity'] != identity or p.digest(folder/'response.txt') != end['response_sha256']:
                raise RuntimeError('cached_response_mismatch:'+str(folder))
            usage = p.read(folder/'usage.json')
            if not usage.get('actual_config_verified') or usage.get('usage_unknown') or usage.get('tool_call_count'):
                raise RuntimeError('cached_usage_not_verified:'+str(folder))
            completed.add(key)
            continue
        # Recover a fully persisted response before considering any retry.
        tid, _, done = p.legacy.parse_events(folder)
        response = next((folder/n for n in ('response.txt','response.pending.txt') if (folder/n).exists()), None)
        if done and response:
            usage = p.read(folder/'usage.json') if (folder/'usage.json').exists() else p.legacy.base.thread_usage(tid)
            if usage.get('actual_config_verified') and not usage.get('usage_unknown') and not usage.get('tool_call_count') and end is None:
                recovered.append(dict(folder=str(folder), identity=identity, usage=usage, response=str(response), thread_id=tid))
                completed.add(key)
                continue
        row = rows.get(key)
        if row and not archive_path(row).exists() and matches(folder, row) and str(folder.relative_to(p.RUN))==row['folder']:
            if (folder.parent.parent/'status.json').exists():
                raise RuntimeError('cannot_retry_terminal_task')
            approved.append(row)
        else:
            blocked.append(str(folder.relative_to(p.RUN)))
    # Finish an interrupted archive transaction without moving/retrying twice.
    for key, row in rows.items():
        archive = archive_path(row)
        if archive.exists():
            if not matches(archive, row):
                raise RuntimeError('interruption_archive_changed:'+str(archive))
            if not (p.RUN/row['folder']).exists():
                approved.append(row)
    releasable = completed | {r['request_key'] for r in approved}
    held = set(state.get('active', {})) | set(state.get('remote_holds', {}))
    blocked += ['unmatched_hold:'+key for key in sorted(held-releasable)]
    allowed_stops = {None, 'pilot_scope_complete', 'trial_call_limit', 'trial_dispatch_deadline',
                     'user_stop', 'batch_complete', 'queue_interrupted', 'unresolved_remote_request'}
    if state.get('stop_reason') not in allowed_stops:
        blocked.append('stop_requires_reconciliation:'+str(state['stop_reason']))
    terminal = len(list(p.RUN.glob('tasks/*/status.json')))
    size = p.read(p.RUN/'manifest.json')['sample_size']
    return dict(ready=not blocked, terminal=terminal, remaining=size-terminal,
                retry_requests=approved, recover_completed=recovered, blocked=blocked,
                archived_attempts=len(list((p.RUN/'interrupted_attempts').glob('*/identity.json'))),
                state=state, paid_calls=0)


def apply_recovery(plan, approval_path):
    if not plan['ready']:
        raise RuntimeError('resume_blocked:'+repr(plan['blocked']))
    # Rename entire directories on the same filesystem; all original bytes survive.
    for row in plan['retry_requests']:
        source, archive = p.RUN/row['folder'], archive_path(row)
        if not archive.exists():
            p.batch.durable_mkdir(archive.parent)
            os.rename(source, archive)
            p.batch.fsync_dir(source.parent)
            p.batch.fsync_dir(archive.parent)
        p.freeze(archive.parent/(row['request_key']+'.authorization.json'),
                 dict(approval_ref=str(approval_path.resolve()), approval_sha256=p.digest(approval_path), request=row))
    for row in plan['recover_completed']:
        folder = Path(row['folder'])
        p.freeze(folder/'response.txt', Path(row['response']).read_text(), raw=True)
        p.freeze(folder/'usage.json', {**row['identity'], **row['usage']})
        p.freeze(folder/'completed.json', dict(identity=row['identity'], status='completed',
                 thread_id=row['thread_id'], response_sha256=p.digest(folder/'response.txt'),
                 server_completion_evidence=True, recovered_offline=True))
    state = copy.deepcopy(plan['state'])
    if plan['retry_requests'] or state.get('active') or state.get('remote_holds'):
        p.append(p.RUN/'reconciliation_events.jsonl', dict(at=p.now(),
                 authorization_ref=str(approval_path.resolve()), previous_state=state,
                 retried_request_keys=[r['request_key'] for r in plan['retry_requests']],
                 reason='explicit_user_authorization_unknown_usage_retained'))
        state.update(active={}, remote_holds={}, stop_reason=None, limit=3,
                     recent_infrastructure_failures=[], consecutive_infrastructure_failures=0)
        state.pop('autoscale', None)
        p.snapshot(p.RUN/'scheduler_state.json', state)


def archived_usage():
    rows = []
    for path in sorted((p.RUN/'interrupted_attempts').glob('*/identity.json')):
        folder = path.parent
        if (folder/'usage.json').exists():
            rows.append(p.read(folder/'usage.json'))
        else:
            tid, _, _ = p.legacy.parse_events(folder)
            rows.append(dict(**p.read(path), thread_id=tid, actual_usage=None,
                             usage_unknown=True, error='network_interrupted_attempt', archive_ref=str(folder)))
    return rows


class Production(p.Production):
    async def call(self, *args, **kwargs):
        try:
            return await asyncio.wait_for(super().call(*args, **kwargs), STAGE_TIMEOUT)
        except asyncio.TimeoutError:
            await self.gate.stop('stage_timeout_requires_reconciliation')
            raise p.DispatchPaused('stage_timeout_requires_reconciliation')

    def compact_progress(self):
        for index, row in enumerate(archived_usage()):
            self.usage_rows['interrupted_attempt:'+str(index)] = row
        super().compact_progress()


def report(manifest):
    # v1's report reads canonical stages once; add archived calls to that summary.
    # Restore immediately, so historical/cumulative summaries are not augmented.
    original = p.legacy.usage_summary
    def include_archive(entries):
        p.legacy.usage_summary = original
        return original(entries + archived_usage())
    p.legacy.usage_summary = include_archive
    try:
        return p.report(manifest)
    finally:
        p.legacy.usage_summary = original


def freeze_execution(acceptance_path, approval_path):
    manifest, chain = p.legacy.execution_manifest(p.RUN)
    acceptance = p.read(acceptance_path)
    required = [Path(__file__).resolve(), p.ROOT/'tests/test_rubric_luna_resume_v2.py', approval_path.resolve()]
    if acceptance.get('passed') is not True or acceptance.get('paid_calls') != 0:
        raise RuntimeError('offline_acceptance_required')
    for file in required:
        if acceptance['implementation_hashes'].get(str(file)) != p.digest(file):
            raise RuntimeError('offline_acceptance_stale:'+str(file))
    if manifest.get('production_entry') == str(Path(__file__).resolve()):
        return manifest
    execution = copy.deepcopy(manifest)
    execution.update(version='batch03-resume-v2', previous_execution_manifest_ref=chain[-1]['path'],
                     original_frozen_manifest_ref=str(p.RUN/'manifest.json'),
                     original_frozen_manifest_sha256=p.digest(p.RUN/'manifest.json'),
                     production_entry=str(Path(__file__).resolve()), created_for_execution_at=p.now(),
                     production_acceptance_ref=str(acceptance_path.resolve()),
                     production_acceptance_sha256=p.digest(acceptance_path),
                     interruption_retry_approval_ref=str(approval_path.resolve()),
                     interruption_retry_approval_sha256=p.digest(approval_path), stage_timeout_seconds=STAGE_TIMEOUT)
    execution['implementation_hashes'].update({str(f):p.digest(f) for f in required})
    p.freeze(p.RUN/f'execution_manifest_v{len(chain)}.json', execution)
    with p.legacy.locked(p.REGISTRY/'.lock'):
        batches, _ = p.legacy.replay()
        record = p.legacy.batch_record(execution, batches[p.RUN_ID])
        record.update(status='running', production_entry=execution['production_entry'])
        p.legacy.commit('原批次断网恢复入口；仅重试明确授权的旧请求；保留未知用量', [record])
    return execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='preflight', choices=['preflight','run','finalize'])
    parser.add_argument('--approval', type=Path, default=APPROVAL)
    parser.add_argument('--acceptance', type=Path, default=ACCEPTANCE)
    args = parser.parse_args()
    p.batch.configure()
    with p.legacy.locked(p.RUN/'.runner.lock', blocking=False):
        manifest, _ = p.legacy.execution_manifest(p.RUN)
        p.legacy.verify(manifest)
        audit = p.batch.historical_audit()
        assert len(manifest['tasks'])==3000 and not set(audit['ids']) & {t['task_id'] for t in manifest['tasks']}
        for name, sha in manifest['source_file_hashes'].items():
            assert p.digest(p.batch.DATA/(name+'.jsonl')) == sha
        plan = inspect(p.read(args.approval))
        public = {k:v for k,v in plan.items() if k not in ('state','retry_requests','recover_completed')}
        public.update(authorized_retries=len(plan['retry_requests']), recoverable_completed=len(plan['recover_completed']))
        print(json.dumps(public, ensure_ascii=False), flush=True)
        if args.action=='preflight':
            return 0 if plan['ready'] else 2
        if not plan['ready']:
            raise RuntimeError('resume_blocked:'+repr(plan['blocked']))
        if args.action=='run':
            manifest = freeze_execution(args.acceptance, args.approval)
            apply_recovery(plan, args.approval)
            try:
                asyncio.run(Production(manifest).run())
            except BaseException:
                report(manifest)
                raise
        result = report(manifest)
        if result['completed']:
            from rubric_luna_bank_v1 import publish
            bank = publish()
            p.snapshot(p.RUN/'completion.json', dict(at=p.now(), all_frozen_tasks_terminal=True,
                terminal_count=result['terminal_count'], passed=result['final_passed'],
                manifest_sha256=p.digest(p.RUN/'manifest.json'), bank=bank,
                bank_manifest_ref=str(p.REGISTRY/'bank_manifest.json'),
                bank_manifest_sha256=p.digest(p.REGISTRY/'bank_manifest.json')))
        return 0


if __name__=='__main__':
    raise SystemExit(main())
