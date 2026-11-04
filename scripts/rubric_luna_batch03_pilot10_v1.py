"""Fixed ten-task pilot in the existing batch03; default is offline preflight."""
import argparse
import asyncio
import copy
import json
import os
from pathlib import Path
import signal

import rubric_luna_batch03_resume_v3 as v3

p, v2 = v3.p, v3.v2
PILOT_ID = 'pilot10_20261102'
ACCEPTANCE = p.ROOT/'configs/rubric/luna_high_3000/pilot10_v1_acceptance.json'


def pilot_dir():
    return p.RUN/'pilot_tests'/PILOT_ID


def select_scope(manifest):
    path = pilot_dir()/'scope.json'
    if path.exists():
        scope = p.read(path)
        if scope['manifest_sha256'] != p.digest(p.RUN/'manifest.json'):
            raise RuntimeError('pilot_manifest_changed')
    else:
        terminal = {p.read(f)['task_id']:p.digest(f) for f in p.RUN.glob('tasks/*/status.json')}
        selected = [t['task_id'] for t in manifest['tasks'] if t['task_id'] not in terminal][:10]
        if len(selected) != 10:
            raise RuntimeError('ten_pending_tasks_required')
        usage = p.legacy.usage_summary([p.read(f) for f in p.RUN.glob('tasks/*/round_*/*/usage.json')] + v2.archived_usage())
        scope = dict(pilot_id=PILOT_ID, task_ids=selected, task_limit=10, initial_concurrency=3,
            manifest_sha256=p.digest(p.RUN/'manifest.json'), terminal_before=terminal,
            known_usage_before=usage['totals'], unknown_usage_before=len(usage['unknown_usage']),
            thread_ids_before=[u['thread_id'] for u in usage['threads'] if u.get('thread_id')],
            stage_files_before={str(f.relative_to(p.RUN)):p.digest(f)
                for f in p.RUN.glob('tasks/*/round_*/*/*') if f.is_file()},
            created_at=p.now(), counts_in_original_batch=True, full_batch_authorized=False)
    ids = scope['task_ids']
    if len(ids) != 10 or len(set(ids)) != 10 or not set(ids) <= {t['task_id'] for t in manifest['tasks']}:
        raise RuntimeError('invalid_pilot_scope')
    return scope


class Pilot(v3.Production):
    def __init__(self, manifest, task_ids, launcher=None):
        super().__init__(manifest, launcher=launcher)
        self.selected = set(task_ids)
        if not 1 <= len(self.selected) <= 10 or len(self.selected) != len(task_ids):
            raise ValueError('pilot_scope_exceeded')
        if not self.selected <= set(self.tasks):
            raise ValueError('pilot_task_not_in_manifest')

    async def call(self, role, revision, task_id, payload, unused_sem):
        if task_id not in self.selected:
            raise RuntimeError('task_outside_pilot_scope')
        return await super().call(role, revision, task_id, payload, unused_sem)

    async def run(self):
        p.batch.configure()
        self.writer = p.Writer(maxsize=32)
        prior = p.read(p.RUN/'scheduler_state.json') if (p.RUN/'scheduler_state.json').exists() else None
        self.gate = p.AdaptiveGate(self.save_controller, state=prior)
        self.gate.limit = 3
        self.gate.max_calls = self.gate.launched + 4*len(self.selected)
        self.completed_rows = {s['task_id']:s for s in (p.read(f) for f in p.RUN.glob('tasks/*/status.json'))}
        self.usage_rows = {str(f):p.read(f) for f in p.RUN.glob('tasks/*/round_*/*/usage.json')}
        self.baseline_terminal = len(self.completed_rows)
        queue = asyncio.Queue()
        for task in self.m['tasks']:
            if task['task_id'] in self.selected and task['task_id'] not in self.completed_rows:
                queue.put_nowait(task)
        original_call = p.legacy.base.call
        p.legacy.base.call = self.call
        async def worker():
            while not queue.empty() and not self.gate.requested_stop and not self.gate.stop_reason:
                task = queue.get_nowait()
                tid = task['task_id']
                try:
                    if tid in self.claimed:
                        raise RuntimeError('duplicate_task_claim')
                    self.claimed.add(tid)
                    await self.writer.submit(lambda: p.snapshot(p.RUN/'task_claims'/tid.replace(':','__'),
                        dict(task_id=tid, owner_pid=os.getpid(), process=p.process_identity(os.getpid()), at=p.now(), pilot_id=PILOT_ID)))
                    state = await p.legacy.RAW_PROCESS(task, None)
                    fresh = {str(f):p.read(f) for f in (p.RUN/'tasks'/tid.replace(':','__')).glob('round_*/*/usage.json')}
                    if any(u.get('usage_unknown') for u in fresh.values()):
                        state['status'] = 'usage_unknown'
                        await self.writer.submit(lambda: p.snapshot(p.RUN/'tasks'/tid.replace(':','__')/'status.json', state))
                    self.usage_rows.update(fresh)
                    self.completed_rows[tid] = self.results[tid] = state
                    await self.writer.submit(lambda: p.batch.registry_event('10题试跑终态计入原批次', tasks=[self.record(tid)]))
                    await self.writer.submit(self.compact_progress)
                except p.DispatchPaused as exc:
                    self.suspended[tid] = str(exc)
                except p.legacy.base.ModelUnavailable as exc:
                    self.suspended[tid] = str(exc)
                    await self.gate.stop('model_configuration')
                except Exception as exc:
                    self.suspended[tid] = repr(exc)
                    await self.gate.stop('infrastructure_or_persistence_error')
                finally:
                    queue.task_done()
        async def monitor():
            while True:
                await asyncio.sleep(15)
                await self.writer.submit(self.compact_progress)
        loop = asyncio.get_running_loop()
        def stop():
            self.gate.requested_stop = 'user_stop'
            asyncio.create_task(self.gate.stop('user_stop'))
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop)
        workers = []
        watcher = None
        try:
            await self.writer.submit(lambda: p.append(p.RUN/'production_runs.jsonl', dict(at=p.now(),
                pilot_id=PILOT_ID, pid=os.getpid(), pending=queue.qsize(), selected_ids=sorted(self.selected),
                existing_terminal=self.baseline_terminal, user_authorized_full_continuation=False)))
            with v3.verified_usage_reader():
                workers = [asyncio.create_task(worker()) for _ in range(3)]
                watcher = asyncio.create_task(monitor())
                # Monitor failure must stop dispatch, but in-flight calls still drain.
                def monitor_done(task):
                    if not task.cancelled() and task.exception():
                        self.gate.requested_stop = 'monitor_or_writer_failure'
                        asyncio.create_task(self.gate.stop('monitor_or_writer_failure'))
                watcher.add_done_callback(monitor_done)
                await asyncio.gather(*workers)
                if not self.gate.stop_reason:
                    await self.gate.stop('pilot_scope_complete')
        finally:
            self.gate.requested_stop = self.gate.requested_stop or self.gate.stop_reason or 'runner_exiting'
            await asyncio.gather(*workers, return_exceptions=True)
            if watcher:
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.remove_signal_handler(sig)
            p.legacy.base.call = original_call
            self.usage_rows = {str(f):p.read(f) for f in p.RUN.glob('tasks/*/round_*/*/usage.json')}
            try:
                await self.writer.submit(self.compact_progress)
            finally:
                await p.finish_cleanup(self.writer.close())
        return dict(selected_ids=sorted(self.selected), new_terminal=len(self.results),
                    stop_reason=self.gate.stop_reason, suspended=self.suspended,
                    active=self.gate.active, remote_holds=self.gate.remote_holds)


def register_execution(manifest, scope):
    acceptance = p.read(ACCEPTANCE)
    required = [Path(__file__).resolve(), p.ROOT/'tests/test_rubric_luna_pilot10_v1.py']
    if acceptance.get('passed') is not True or acceptance.get('paid_calls') != 0:
        raise RuntimeError('offline_acceptance_required')
    for path in required:
        if acceptance['implementation_hashes'].get(str(path)) != p.digest(path):
            raise RuntimeError('stale_pilot_acceptance')
    p.freeze(pilot_dir()/'scope.json', scope)
    if manifest.get('pilot_entry') == str(Path(__file__).resolve()):
        return manifest
    _, chain = p.legacy.execution_manifest(p.RUN)
    execution = copy.deepcopy(manifest)
    execution.update(previous_execution_manifest_ref=chain[-1]['path'], version='batch03-pilot10-v1',
        pilot_entry=str(Path(__file__).resolve()), pilot_scope_ref=str(pilot_dir()/'scope.json'),
        pilot_scope_sha256=p.digest(pilot_dir()/'scope.json'), pilot_acceptance_ref=str(ACCEPTANCE),
        pilot_acceptance_sha256=p.digest(ACCEPTANCE))
    execution['implementation_hashes'].update({str(path):p.digest(path) for path in required})
    p.freeze(p.RUN/f'execution_manifest_v{len(chain)}.json', execution)
    with p.legacy.locked(p.REGISTRY/'.lock'):
        batches, _ = p.legacy.replay()
        record = p.legacy.batch_record(execution, batches[p.RUN_ID])
        record.update(status='running', active_pilot=PILOT_ID)
        p.legacy.commit('用户授权原批次固定10题试跑；完成后停止', [record])
    return execution


def summarize(scope, outcome):
    statuses = {tid:p.read(p.RUN/'tasks'/tid.replace(':','__')/'status.json')
                for tid in scope['task_ids'] if (p.RUN/'tasks'/tid.replace(':','__')/'status.json').exists()}
    entries = [p.read(f) for f in p.RUN.glob('tasks/*/round_*/*/usage.json')]
    new = [u for u in entries if u.get('thread_id') not in scope['thread_ids_before']]
    prior_intact = all(p.digest(p.RUN/'tasks'/tid.replace(':','__')/'status.json')==sha
                       for tid,sha in scope['terminal_before'].items())
    stages_intact = all(p.digest(p.RUN/path)==sha for path,sha in scope['stage_files_before'].items())
    outside = [u['task_id'] for u in new if u['task_id'] not in scope['task_ids']]
    usage = p.legacy.usage_summary(new)
    result = dict(at=p.now(), pilot_id=PILOT_ID, scope_ref=str(pilot_dir()/'scope.json'), outcome=outcome,
        selected_ids=scope['task_ids'], statuses={tid:s['status'] for tid,s in statuses.items()},
        terminal=len(statuses), passed=sum(s['status']=='passed' for s in statuses.values()),
        new_calls=len(new), new_usage=usage, original_terminal_unchanged=prior_intact,
        existing_stage_files_unchanged=stages_intact, calls_outside_scope=outside,
        all_new_calls_verified=all(u.get('actual_config_verified') for u in new),
        missing_index_fallback_calls=sum(bool(u.get('configuration_evidence',{}).get('used_missing_index_fallback')) for u in new),
        counts_in_original_batch=True)
    result['execution_success'] = (len(statuses)==10 and prior_intact and stages_intact and not outside and
        not usage['unknown_usage'] and result['all_new_calls_verified'] and
        all(s['status'] in ('passed','rejected','invalid_generation','invalid_judge') for s in statuses.values()) and
        not outcome.get('active') and not outcome.get('remote_holds') and not outcome.get('suspended'))
    p.snapshot(pilot_dir()/'results.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='preflight', choices=['preflight','run'])
    args = parser.parse_args()
    p.batch.configure()
    with p.legacy.locked(p.RUN/'.runner.lock', blocking=False):
        manifest, _ = p.legacy.execution_manifest(p.RUN)
        p.legacy.verify(manifest)
        plan = v3.inspect(p.read(v2.APPROVAL))
        if not plan['ready'] or plan['metadata_repairs']:
            raise RuntimeError('pilot_preflight_blocked:'+repr(plan['blocked']))
        scope = select_scope(manifest)
        if any(row['identity']['task_id'] not in scope['task_ids'] for row in plan['retry_requests']):
            raise RuntimeError('recovery_outside_pilot_scope')
        print(json.dumps(dict(ready=True, selected=scope['task_ids'], task_limit=10,
                              existing_terminal=plan['terminal'], remaining=plan['remaining'])), flush=True)
        if args.action=='preflight':
            return 0
        manifest = register_execution(manifest, scope)
        v2.apply_recovery(plan, v2.APPROVAL)
        outcome = {}
        try:
            outcome = asyncio.run(Pilot(manifest, scope['task_ids']).run())
        finally:
            v3.refresh_reports(manifest)
            result = summarize(scope, outcome)
            print(json.dumps({k:result[k] for k in ('execution_success','terminal','passed','new_calls','missing_index_fallback_calls')}), flush=True)
        return 0 if result['execution_success'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
