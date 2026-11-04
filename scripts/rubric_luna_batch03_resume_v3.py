"""Batch03 recovery with raw-session configuration evidence; preflight is read-only.

`repair` repairs saved final-review metadata offline. Only `run` calls models.
Frozen v1/v2 code and raw request/response files are never rewritten.
"""
import argparse
import asyncio
import copy
import json
from contextlib import contextmanager
from pathlib import Path

import rubric_luna_batch03_resume_v2 as v2

p = v2.p
ACCEPTANCE = p.ROOT/'configs/rubric/luna_high_3000/resume_v3_acceptance.json'
RAW_THREAD_USAGE = p.legacy.base.thread_usage


def verify_usage(usage, thread_id):
    """NULL index fields may be filled by a matching, single-turn raw session.

    Non-null conflicts are never overridden. Absence of raw evidence fails closed.
    """
    result = copy.deepcopy(usage)
    sessions, contexts = [], []
    path = Path(usage['rollout_ref']) if usage.get('rollout_ref') else None
    if path:
        for line in path.read_text().splitlines():
            record = json.loads(line)
            payload = record.get('payload', {})
            if record.get('type') == 'session_meta':
                sessions.append(payload.get('id'))
            if record.get('type') == 'turn_context':
                contexts.append(dict(model=payload.get('model'), effort=payload.get('effort')))
    expected = dict(model=p.MODEL, effort=p.EFFORT)
    index = dict(model=usage.get('model'), effort=usage.get('reasoning_effort'))
    raw_ok = sessions == [thread_id] and len(contexts) == 1 and contexts[0] == expected
    index_ok = all(index[k] is None or index[k] == expected[k] for k in expected)
    verified = raw_ok and index_ok and usage.get('thread_id') == thread_id
    result['actual_config_verified'] = verified
    result['configuration_evidence'] = dict(
        policy='single_raw_session_required_null_index_allowed_conflicts_rejected',
        thread_index=index, session_ids=sessions, turn_contexts=contexts,
        rollout_sha256=p.digest(path) if path else None,
        verified=verified, used_missing_index_fallback=verified and None in index.values())
    if verified:
        result.update(model=p.MODEL, reasoning_effort=p.EFFORT)
    return result


def thread_usage(thread_id):
    return verify_usage(RAW_THREAD_USAGE(thread_id), thread_id)


@contextmanager
def verified_usage_reader():
    original = p.legacy.base.thread_usage
    p.legacy.base.thread_usage = thread_usage
    try:
        yield
    finally:
        p.legacy.base.thread_usage = original


class Production(v2.Production):
    async def call(self, role, revision, task_id, payload, unused_sem):
        try:
            return await super().call(role, revision, task_id, payload, unused_sem)
        except p.legacy.base.CallFailure:
            folder = p.RUN/'tasks'/task_id.replace(':', '__')/f'round_{revision}'/role
            usage_path = folder/'usage.json'
            if usage_path.exists() and not p.read(usage_path).get('actual_config_verified'):
                # v1 only stops on an explicit non-null mismatching model name.
                await self.gate.stop('model_configuration')
                raise p.legacy.base.ModelUnavailable('configuration_evidence_missing_or_conflicting')
            raise

    async def run(self):
        with verified_usage_reader():
            return await super().run()


def repair_plan(folder):
    """Re-evaluate an already completed final review using the frozen evaluator.

    No prompt changes, semantic judgments, model calls, or further revisions.
    """
    identity = p.read(folder/'identity.json')
    end = p.read(folder/'completed.json')
    old_usage = p.read(folder/'usage.json')
    if (identity.get('stage') != 'fresh_final_judge' or identity.get('revision') != 1 or
        end.get('status') != 'call_failed' or end.get('returncode') != 0 or
        not end.get('server_completion_evidence') or end.get('identity') != identity or
        old_usage.get('actual_config_verified') is not False):
        return None
    tid, unavailable, done = p.legacy.parse_events(folder)
    if not done or unavailable or tid != end.get('thread_id') or tid != old_usage.get('thread_id'):
        return None
    if p.digest(folder/'response.txt') != end.get('response_sha256'):
        raise RuntimeError('response_hash_mismatch:'+str(folder))
    if p.digest(folder/'input.json') != identity['input_sha256']:
        raise RuntimeError('input_hash_mismatch:'+str(folder))
    fresh = thread_usage(tid)
    if not fresh['actual_config_verified'] or fresh.get('usage_unknown') or fresh.get('tool_call_count'):
        return None
    for key in ('actual_usage','response_usage','tool_call_count','per_response_sum_matches_cumulative'):
        if fresh.get(key) != old_usage.get(key):
            raise RuntimeError('recovery_usage_changed:'+key)
    if not fresh.get('per_response_sum_matches_cumulative'):
        return None
    task_folder = folder.parent.parent
    state = p.read(task_folder/'status.json')
    if (state.get('status') != 'call_failed' or state.get('task_id') != identity['task_id'] or
        len(state.get('rounds', [])) != 2 or state['rounds'][-1].get('failed_stage') != 'judge'):
        return None
    rubric = p.read(folder.parent/'rubric.json')
    pack = p.read(folder/'input.json')
    if pack['data']['candidate_rubric'] != rubric:
        raise RuntimeError('recovery_rubric_mismatch')
    manifest, _ = p.legacy.execution_manifest(p.RUN)
    task = next(t for t in manifest['tasks'] if t['task_id'] == identity['task_id'])
    audit = p.read(task['audit_ref'])
    response = (folder/'response.txt').read_text()
    try:
        judgment = json.loads(response)
    except ValueError:
        judgment = dict(invalid_raw_content=response)
    verdict = p.legacy.base.evaluate(judgment, identity['task_id'],
        {item['id'] for item in rubric['rubric']}, {c['candidate_id'] for c in audit['candidates']})
    row = state['rounds'][-1]
    row.pop('error', None)
    row.pop('failed_stage', None)
    row.update(verdict)
    state['status'] = verdict['status']
    if state['status'] == 'passed':
        state['accepted_revision'] = 1
    new_usage = {**old_usage, **fresh}
    new_end = {**end, 'status':'completed', 'metadata_recovered_offline':True}
    values = {
        folder/'usage.json':new_usage, folder/'completed.json':new_end,
        folder/'output.json':judgment, folder.parent/'evaluation.json':verdict,
        task_folder/'status.json':state,
    }
    if state['status'] == 'passed':
        values[p.RUN/'accepted'/(identity['task_id'].replace(':','__')+'.json')] = rubric
    changes = []
    for path, value in values.items():
        content = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
        # Preserve the exact old derived records, as well as all raw evidence.
        before = path.read_text() if path.exists() else None
        changes.append(dict(path=str(path.relative_to(p.RUN)), before=before, after=content))
    return dict(thread_id=tid, task_id=identity['task_id'], folder=str(folder.relative_to(p.RUN)),
                evidence=fresh['configuration_evidence'], changes=changes, paid_calls=0,
                raw_hashes={str((folder/n).relative_to(p.RUN)):p.digest(folder/n)
                            for n in ('input.json','identity.json','started.json','events.jsonl','response.txt')})


def pending_repairs():
    plans = []
    journal_root = p.RUN/'metadata_repairs'
    # Replay interrupted transactions before reading partially updated task state.
    for path in sorted(journal_root.glob('*/plan.json')):
        if not (path.parent/'committed.json').exists():
            plans.append(p.read(path))
    pending_folders = {plan['folder'] for plan in plans}
    for path in sorted(p.RUN.glob('tasks/*/round_1/judge/completed.json')):
        if str(path.parent.relative_to(p.RUN)) in pending_folders:
            continue
        if p.read(path).get('status') == 'call_failed':
            plan = repair_plan(path.parent)
            if plan:
                plans.append(plan)
    return plans


def apply_metadata_repairs(plans):
    for plan in plans:
        for name, sha in plan['raw_hashes'].items():
            if p.digest(p.RUN/name) != sha:
                raise RuntimeError('recovery_raw_evidence_changed:'+name)
        for change in plan['changes']:
            path = p.RUN/change['path']
            current = path.read_text() if path.exists() else None
            if current not in (change['before'], change['after']):
                raise RuntimeError('recovery_derived_record_changed:'+change['path'])
        journal = p.RUN/'metadata_repairs'/plan['thread_id']
        p.freeze(journal/'plan.json', plan)
        for change in plan['changes']:
            p.batch.atomic(p.RUN/change['path'], change['after'])
        p.freeze(journal/'committed.json', dict(thread_id=plan['thread_id'], paid_calls=0,
                 policy='restore_completed_final_judge_from_raw_session', plan_sha256=p.digest(journal/'plan.json')))


def inspect(approval):
    repairs = pending_repairs()
    # During an interrupted transaction, cached completed.json may precede usage.json;
    # apply order ensures usage is committed first. v2 still validates all other data.
    plan = v2.inspect(approval)
    folders = {row['folder'] for row in repairs}
    plan['blocked'] = [reason for reason in plan['blocked'] if reason not in folders]
    plan['ready'] = not plan['blocked']
    plan['metadata_repairs'] = repairs
    return plan


def freeze_execution(acceptance_path, approval_path):
    manifest, chain = p.legacy.execution_manifest(p.RUN)
    acceptance = p.read(acceptance_path)
    required = [Path(__file__).resolve(), p.ROOT/'tests/test_rubric_luna_resume_v3.py', approval_path.resolve()]
    if acceptance.get('passed') is not True or acceptance.get('paid_calls') != 0:
        raise RuntimeError('offline_acceptance_required')
    for path in required:
        if acceptance['implementation_hashes'].get(str(path)) != p.digest(path):
            raise RuntimeError('offline_acceptance_stale:'+str(path))
    if manifest.get('production_entry') == str(Path(__file__).resolve()):
        return manifest
    execution = copy.deepcopy(manifest)
    execution.update(version='batch03-resume-v3', previous_execution_manifest_ref=chain[-1]['path'],
        production_entry=str(Path(__file__).resolve()), created_for_execution_at=p.now(),
        production_acceptance_ref=str(acceptance_path.resolve()),
        production_acceptance_sha256=p.digest(acceptance_path),
        configuration_policy='raw_single_session_required_null_index_allowed_conflicts_rejected')
    execution['implementation_hashes'].update({str(path):p.digest(path) for path in required})
    p.freeze(p.RUN/f'execution_manifest_v{len(chain)}.json', execution)
    return execution


def refresh_reports(manifest):
    result = v2.report(manifest)
    progress_path = p.RUN/'progress.json'
    if progress_path.exists():
        progress = p.read(progress_path)
        progress.update(at=p.now(), terminal=result['terminal_count'], passed=result['final_passed'],
            statuses={**result['failure_types'], 'passed':result['final_passed'],
                      'pending':manifest['sample_size']-result['terminal_count']},
            luna_known_tokens=result['totals']['total_tokens'], unknown_usage_calls=result['unknown_usage_calls'],
            last_offline_metadata_repair=True)
        p.snapshot(progress_path, progress)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='preflight', choices=['preflight','repair','run','finalize'])
    parser.add_argument('--approval', type=Path, default=v2.APPROVAL)
    parser.add_argument('--acceptance', type=Path, default=ACCEPTANCE)
    args = parser.parse_args()
    p.batch.configure()
    with p.legacy.locked(p.RUN/'.runner.lock', blocking=False):
        manifest, _ = p.legacy.execution_manifest(p.RUN)
        p.legacy.verify(manifest)
        audit = p.batch.historical_audit()
        assert len(manifest['tasks']) == 3000 and not set(audit['ids']) & {t['task_id'] for t in manifest['tasks']}
        for name, sha in manifest['source_file_hashes'].items():
            assert p.digest(p.batch.DATA/(name+'.jsonl')) == sha
        plan = inspect(p.read(args.approval))
        print(json.dumps({**{k:plan[k] for k in ('ready','terminal','remaining','blocked','paid_calls')},
            'authorized_retries':len(plan['retry_requests']), 'offline_metadata_repairs':len(plan['metadata_repairs'])}), flush=True)
        if args.action == 'preflight':
            return 0 if plan['ready'] else 2
        if not plan['ready']:
            raise RuntimeError('resume_blocked:'+repr(plan['blocked']))
        manifest = freeze_execution(args.acceptance, args.approval)
        apply_metadata_repairs(plan['metadata_repairs'])
        if args.action == 'repair':
            refresh_reports(manifest)
            print(json.dumps(dict(repaired=len(plan['metadata_repairs']), paid_calls=0, batch_started=False)), flush=True)
            return 0
        if args.action == 'run':
            plan = v2.inspect(p.read(args.approval))
            v2.apply_recovery(plan, args.approval)
            try:
                asyncio.run(Production(manifest).run())
            except BaseException:
                refresh_reports(manifest)
                raise
        result = refresh_reports(manifest)
        if result['completed']:
            from rubric_luna_bank_v1 import publish
            bank = publish()
            p.snapshot(p.RUN/'completion.json', dict(at=p.now(), all_frozen_tasks_terminal=True,
                terminal_count=result['terminal_count'], passed=result['final_passed'], bank=bank,
                manifest_sha256=p.digest(p.RUN/'manifest.json'),
                bank_manifest_ref=str(p.REGISTRY/'bank_manifest.json'),
                bank_manifest_sha256=p.digest(p.REGISTRY/'bank_manifest.json')))
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
