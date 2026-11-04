"""User-authorized batch04 concurrency: 12, with a persistent fallback to 6."""
import copy
import random
from pathlib import Path

import rubric_luna_batch04_production_v2 as previous
from rubric_luna_autoscale_v1 import compare, summarize


def actual_usage(directory):
    """Include archived interrupted attempts using batch03's frozen accounting rule."""
    rows = [previous.read(path) for path in directory.glob('tasks/*/round_*/*/usage.json')]
    for path in sorted(directory.glob('interrupted_attempts/*/identity.json')):
        folder = path.parent
        if (folder/'usage.json').exists():
            rows.append(previous.read(folder/'usage.json'))
        else:
            tid, _, _ = previous.legacy.parse_events(folder)
            rows.append(dict(**previous.read(path), thread_id=tid, actual_usage=None,
                usage_unknown=True, error='network_interrupted_attempt', archive_ref=str(folder)))
    ids = [row['thread_id'] for row in rows]
    assert len(ids) == len(set(ids)), 'duplicate_usage_thread'
    return {row['thread_id']: row for row in rows}


def verify_registered_batches(batches, registered):
    """Retain every legacy check and reconcile both live and archived usage."""
    legacy = previous.legacy
    for bid, batch in batches.items():
        if bid == legacy.RUN_ID:
            continue
        directory = Path(batch['manifest_ref']).parent
        manifest, chain = legacy.execution_manifest(directory)
        legacy.verify(manifest, directory, implementations=False)
        assert previous.digest(directory/'manifest.json') == batch['manifest_sha256']
        assert chain == batch['execution_chain']
        records = {t['task_id']: t for t in registered.values() if t['run_id'] == bid}
        assert set(records) == {t['task_id'] for t in manifest['tasks']}
        assert len(records) == batch['sample_size']
        for task in manifest['tasks']:
            assert records[task['task_id']]['task_sha256'] == task['task_sha256']
        if batch['status'] != 'completed':
            continue
        result = previous.read(directory/'results.json')
        states = {t['task_id']: t for t in result['tasks']}
        assert set(states) == set(records)
        assert result['completed'] and result['denominator'] == len(records)
        retention = previous.read(directory/'retention_manifest.json')
        assert retention['frozen_manifest_sha256'] == batch['manifest_sha256']
        assert {t['task_id'] for t in retention['tasks']} == set(records)
        for tid, state in states.items():
            assert records[tid]['status'] == state['status']
            assert previous.read(directory/'tasks'/tid.replace(':', '__')/'status.json') == state
        accepted = previous.read(directory/'accepted_manifest.json')
        assert {t['task_id'] for t in accepted} == {tid for tid,s in states.items() if s['status'] == 'passed'}
        for item in accepted:
            if 'sha256' in item:
                assert previous.digest(item['path']) == item['sha256']
            else:
                state = states[item['task_id']]
                source = directory/'tasks'/item['task_id'].replace(':', '__')/f"round_{state['accepted_revision']}"/'rubric.json'
                assert Path(item['path']).read_bytes() == source.read_bytes()
        usage = previous.read(directory/'usage.json')
        assert legacy.usage_summary(usage['threads'])['totals'] == usage['totals'] == result['totals'] == batch['luna_actual_usage']
        saved = {u['thread_id']: u for u in usage['threads'] if u.get('task_id') is not None}
        assert saved == actual_usage(directory), 'live_and_archived_usage_mismatch:'+bid


class Batch04Gate(previous.Batch04Gate):
    def __init__(self, persist, state=None, **kwargs):
        super().__init__(persist, state=state, **kwargs)
        prior = state or {}
        manual = prior.get('user_concurrency', {})
        self.fallback_reason = manual.get('fallback_reason')
        self.baseline = manual.get('baseline_6')
        if not self.baseline:
            rows = [row for row in prior.get('stage_completions', [])
                    if row.get('launch_limit') == 6]
            if len(rows) >= 30:
                self.baseline = summarize(rows, min(r['completed_at'] for r in rows),
                                          self.clock(), 6)
        self.limit = 3 if self.parallel_cap else 6 if self.fallback_reason else 12
        self.last_decision = ('batch03_parallel_cap_3' if self.parallel_cap else
                              'user_fallback_6_latched' if self.fallback_reason else
                              'user_requested_12_with_fallback_6')

    def state(self):
        value = super().state()
        value['user_concurrency'] = dict(target=12, fallback=6,
            fallback_reason=self.fallback_reason, baseline_6=self.baseline,
            automatic_repromotion=False)
        return value

    def fallback(self, reason):
        self.fallback_reason = self.fallback_reason or reason
        self.limit = 3 if self.parallel_cap else 6
        self.last_decision = 'user_fallback_6_latched'

    async def complete(self, key, *, infrastructure_failure=False, model_bad=False,
                       unresolved=False, rate_limited=False, retry_after=None, detail=None):
        # The fallback and permit release share one critical section, so no new
        # request can slip through at 12 after an error has been observed.
        async with self.condition:
            active = self.active.pop(key)
            detail = dict(detail or {})
            detail.update(launch_limit=active.get('launch_limit', self.limit),
                          rate_limited=bool(rate_limited))
            if infrastructure_failure or rate_limited or unresolved or model_bad:
                reason = ('rate_limit' if rate_limited else 'unresolved_request' if unresolved
                          else 'model_configuration' if model_bad else 'infrastructure_failure')
                self.fallback(reason)
            if unresolved:
                self.remote_holds[key] = active
                self.stop_reason = self.stop_reason or 'unresolved_remote_request'
            self.recent.append(bool(infrastructure_failure))
            self.consecutive = self.consecutive + 1 if infrastructure_failure else 0
            row = dict(completed_at=self.clock(), key=key,
                       infrastructure_failure=infrastructure_failure, **detail)
            self.history.append(row)
            if detail['launch_limit'] == self.limit:
                self.window_rows.append(row)
            if model_bad:
                self.stop_reason = 'model_configuration'
            elif self.consecutive >= 3 or sum(self.recent) >= 5:
                self.stop_reason = 'infrastructure_circuit_open'
            if rate_limited:
                self.backoff_attempt += 1
                base = min(300, 30 * 2 ** min(self.backoff_attempt - 1, 4))
                delay = max(retry_after or 0, base * random.uniform(1, 1.2))
                self.cooldown_until = max(self.cooldown_until, self.clock() + delay)
            if self.stop_reason:
                self.stopped.set()
            await self.save('stage_completed')
            self.condition.notify_all()

    async def observe(self):
        async with self.condition:
            if not previous.batch03_clear():
                self.parallel_cap = True
                self.limit = 3
                self.last_decision = 'batch03_parallel_cap_3'
                await self.save('batch03_parallel_cap')
                self.condition.notify_all()
                return
            if self.parallel_cap:
                self.parallel_cap = False
                self.limit = 6 if self.fallback_reason else 12
                self.reset_level()
            end = self.clock()
            if self.stop_reason or self.requested_stop or end < self.cooldown_until:
                return
            if end - self.window_start < 300 or len(self.window_rows) < 30:
                return
            window = summarize(self.window_rows, self.window_start, end, self.limit)
            comparison = compare(self.baseline, window) if self.baseline and self.limit == 12 else None
            window['comparison'] = comparison
            self.windows.append(window)
            degraded = (comparison and comparison['qualified'] and
                        comparison['p95_ratio'] > 2 and comparison['throughput_ratio'] <= 1)
            self.bad_windows = self.bad_windows + 1 if degraded else 0
            if self.bad_windows >= 2:
                self.fallback('two_qualified_slow_windows')
            self.window_start, self.window_rows = end, []
            await self.save('user_concurrency_window')
            self.condition.notify_all()


def freeze_execution(acceptance):
    m, chain = previous.legacy.execution_manifest(previous.RUN)
    entry = str(Path(__file__).resolve())
    if m.get('production_entry') == entry:
        return m
    execution = copy.deepcopy(m)
    execution.update(previous_execution_manifest_ref=chain[-1]['path'],
        created_for_execution_at=previous.now(), production_entry=entry,
        version='batch04-production-v3', production_acceptance_ref=str(acceptance),
        production_acceptance_sha256=previous.digest(acceptance))
    execution['policy'].update(levels=[12, 6], initial_concurrency=12,
        target_concurrency=12, fallback_concurrency=6, automatic_repromotion=False,
        user_override='2026-11-03: raise to 12; fall back to 6 on problems',
        rate_limit_backoff_preserved=True, unknown_requests_stop_dispatch=True)
    for path in (Path(__file__).resolve(), previous.ROOT/'tests/test_rubric_luna_batch04_v3.py'):
        execution['implementation_hashes'][str(path)] = previous.digest(path)
    previous.freeze(previous.RUN/f'execution_manifest_v{len(chain)}.json', execution)
    with previous.legacy.locked(previous.REGISTRY/'.lock'):
        batches, _ = previous.legacy.replay()
        record = previous.legacy.batch_record(execution, batches[previous.RUN_ID])
        record.update(status='running', production_entry=entry,
                      production_started_at=previous.now(), production_policy=execution['policy'])
        previous.legacy.commit('用户要求batch04提升12并发，异常锁定退回6并发', [record])
    return execution


def main():
    # Bind only inside this new process; frozen v2 files remain reproducible.
    previous.Batch04Gate = Batch04Gate
    previous.freeze_execution = freeze_execution
    previous.legacy.verify_registered_batches = verify_registered_batches
    previous.main()


if __name__ == '__main__':
    main()
