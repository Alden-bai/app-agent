"""Bounded validation rescoring: 16 workers, 4096 output with exact-input fallback.

Keep the frozen Judge's prompts, evidence, rubric, query/review limits and reward
contract. Only the transport output allowance and invalid-result scheduling change.
"""
import asyncio
from collections import defaultdict, deque
import fcntl
import os
from pathlib import Path
import signal
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import sft_adaptive_judge_20261110 as adaptive

base, r = adaptive.base, adaptive.r
RUN = adaptive.RUN
DRAIN = False
MAX_ASSESSMENTS = 2
OUTPUT_LIMIT = 4096
FALLBACK_LIMIT = 8192


async def request_with_fallback(api, gate, config, kwargs, budget, key, stage, folder, **kw):
    """Discard truncated output; retry identical messages once at the original cap.

    Both physical attempts (including 429 retries) use the existing cumulative
    budget. No partial output enters the Judge's evidence or repair conversation.
    """
    wire = dict(kwargs, max_completion_tokens=OUTPUT_LIMIT)
    response = await adaptive.request_with_backoff(
        api, gate, config, wire, budget, key, stage, folder, **kw)
    if response['choices'][0]['finish_reason'] != 'length':
        return response
    fallback_folder = Path(folder) / 'output_fallback_8192'
    r.write_json(Path(folder) / 'output_fallback.json', {
        'at': r.now(), 'from': OUTPUT_LIMIT, 'to': FALLBACK_LIMIT,
        'messages_sha256': r.digest(kwargs['messages']),
        'partial_output_discarded': True, 'budget_preserved': True,
    })
    wire = dict(kwargs, max_completion_tokens=FALLBACK_LIMIT)
    # A second length response is deliberately returned to the unchanged runtime,
    # which rejects it. It must never be silently treated as a complete assessment.
    return await adaptive.request_with_backoff(
        api, gate, config, wire, budget, key + '/output_fallback_8192',
        stage, fallback_folder, **kw)


def record_score(record):
    if record is None:
        return None
    path = Path(record['episode']) / 'episode_result.json'
    score = r.evaluation_score(r.read_json(path))
    if score != record.get('score'):
        raise RuntimeError('recovery_score_does_not_match_validated_episode:' + str(path))
    return score


def request_count(job):
    path = Path(job) / 'budget.sqlite'
    if not path.exists():
        return 0
    with sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True) as db:
        return db.execute('SELECT count(*) FROM requests').fetchone()[0]


def disposition(record, tries, used_requests, result=None):
    """Retry by failure category, never by the numerical score."""
    if record is not None and record.get('score') is not None:
        return 'valid'
    if used_requests >= 32:
        return 'held_request_budget'
    if record is None:
        return 'held_assessment_retry_limit' if len(tries) >= MAX_ASSESSMENTS else 'pending'
    errors = ' '.join(map(str, record.get('errors', [])))
    status = (result or {}).get('evaluation_status', record.get('evaluation_status'))
    if status in ('rubric_error', 'invalid_sample'):
        return 'held_' + status
    if any(s in errors for s in ('global_budget_exhausted', 'recovery_call_budget_exhausted')):
        return 'held_request_budget'
    if len(tries) >= MAX_ASSESSMENTS:
        return 'held_assessment_retry_limit'
    retryable = ('query_round_budget_exhausted', 'required_evidence_incomplete',
                 'judge_output_invalid_after_repair', 'incomplete_model_output',
                 'APIConnectionError', 'APITimeoutError', 'RateLimitError',
                 'InternalServerError', 'assessment_not_finished')
    if any(s in errors for s in retryable):
        return 'retry_pending'
    return 'held_requires_inspection'


def archive_record(job, record):
    if record is not None:
        path = Path(job) / 'recovery_history' / (r.digest(record) + '.json')
        if path.exists():
            if r.read_json(path) != record:
                raise RuntimeError('recovery_history_changed:' + str(path))
        else:
            r.write_json(path, record)


def inventory():
    pending, records, held = [], {}, []
    total = original_scored = untrusted = 0
    for name in r.read_json(r.REPAIRS / 'validation/candidates.json'):
        manifest = r.read_json(r.REPAIRS / 'validation' / name / 'manifest.json')
        for item in manifest['candidates']:
            total += 1
            source = Path(item['path'])
            if not (source / 'episode_result.json').exists():
                raise RuntimeError('missing_validation_episode:' + str(source))
            result = r.read_json(source / 'episode_result.json')
            if r.evaluation_score(result) is not None:
                original_scored += 1
                continue
            if result['execution_integrity'] != 'valid' or result['terminal_state_status'] != 'trusted':
                untrusted += 1
                held.append({'source': str(source), 'reason': 'held_untrusted_execution'})
                continue
            key = r.digest([name, item['task_instance_id'], str(source)])[:24]
            job = RUN / 'judge_jobs' / key
            row = (name, item, source, job)
            path = job / 'recovery.json'
            record = r.read_json(path) if path.exists() else None
            if record is not None:
                record_score(record)
                records[str(job)] = record
                result = r.read_json(Path(record['episode']) / 'episode_result.json')
            tries = r.read_json(job / 'attempts.json') if (job / 'attempts.json').exists() else []
            state = disposition(record, tries, request_count(job), result)
            if state in ('pending', 'retry_pending'):
                pending.append(row)
            elif state != 'valid':
                held.append({'job': str(job), 'reason': state, 'errors': (record or {}).get('errors', [])})
    # Mix candidates without using scores. Existing invalid attempts go first
    # within each candidate so they cannot disappear behind a completed count.
    groups = defaultdict(deque)
    for row in sorted(pending, key=lambda x: (str(x[3]) not in records, str(x[3]))):
        groups[row[0]].append(row)
    ordered = []
    while any(groups.values()):
        for name in sorted(groups):
            if groups[name]:
                ordered.append(groups[name].popleft())
    return ordered, records, held, dict(total=total, original_scored=original_scored, untrusted=untrusted)


def progress(records, held, counts):
    valid = sum(record.get('score') is not None for record in records.values())
    return dict(completed=valid, valid_completed=valid, attempts_completed=len(records),
                invalid_completed=len(records)-valid, held=len(held),
                total_validation_scored=counts['original_scored']+valid,
                remaining=counts['total']-counts['original_scored']-valid)


async def assess(row, cfg, runtime, historical, gate):
    job = row[3]
    previous = r.read_json(job / 'recovery.json') if (job / 'recovery.json').exists() else None
    if previous is not None and record_score(previous) is not None:
        return previous
    archive_record(job, previous)
    # Same seal verification, evidence copies, outcome validation and cumulative
    # baseline accounting as the established scorer. Only invalid results requeue.
    record = await base.assess_one(row, cfg, runtime, historical, gate)
    record_score(record)
    archive_record(job, record)
    return record


async def drive(plan, policy):
    cfg = r.RunConfig.model_validate(plan['config'])
    cfg = cfg.model_copy(update={'judge_v2': cfg.judge_v2.model_copy(update={'query_rounds': 12})})
    assert cfg.judge_v2.max_output_tokens == FALLBACK_LIMIT
    gate = adaptive.AdaptiveGate(RUN / 'provider_rate.sqlite')
    # Preserve the existing shared rate ledger and 429 backoff. Normal target is
    # still 16; no additional channel or quota bypass is introduced.
    runtime = r.recovery_judge()
    api, original_hash = runtime.api_request, runtime.config_hash
    runtime.config_hash = lambda config: r.digest({
        'original': original_hash(config), 'transport_policy_sha256': policy['implementation_sha256'],
        'max_output_tokens': OUTPUT_LIMIT, 'fallback_output_tokens': FALLBACK_LIMIT,
    })
    async def paced(config, kwargs, budget, key, stage, folder, **kw):
        return await request_with_fallback(api, gate, config, kwargs, budget, key, stage, folder, **kw)
    runtime.api_request = paced
    historical = {x['source_episode']: x for x in r.recovery.collect_repairs()}
    pending, records, held, counts = await asyncio.to_thread(inventory)
    active, fatal = {}, []
    queued = 0
    phase = 'pilot_4096'
    pilot_path = RUN / 'optimized_pilot.json'
    pilot = r.read_json(pilot_path) if pilot_path.exists() else None
    if pilot is None:
        pilot = {'at': r.now(), 'stage': 'running', 'jobs': [str(x[3]) for x in pending[:16]],
                 'implementation_sha256': policy['implementation_sha256']}
        r.write_json(pilot_path, pilot)
    elif pilot['implementation_sha256'] != policy['implementation_sha256']:
        raise RuntimeError('pilot_implementation_changed')

    async def report():
        while True:
            rate = gate.snapshot()
            stage = 'needs_attention' if fatal else ('draining' if DRAIN else (
                'waiting_rate_limit' if rate['cooldown_until'] > time.time() else phase))
            r.status('judge', stage, concurrency=rate['target_concurrency'], initial_concurrency=16,
                     active=list(active.values()), queued=queued, errors=fatal,
                     rate=rate, max_output_tokens=OUTPUT_LIMIT, fallback_output_tokens=FALLBACK_LIMIT,
                     **progress(records, held, counts))
            await asyncio.sleep(5)

    async def batch(rows, allow_retry=False):
        nonlocal queued
        queue = deque(rows)
        queued = len(queue)
        async def worker(index):
            nonlocal queued
            while queue and not fatal and not DRAIN:
                gate.refresh()
                if index >= gate.target:
                    return
                if min(base.shutil.disk_usage(RUN / 'judge_jobs').free,
                       base.shutil.disk_usage(RUN).free) < 2 * 1024**3:
                    fatal.append('insufficient_local_disk_less_than_2GiB')
                    return
                row = queue.popleft()
                queued = len(queue)
                active[index] = str(row[3])
                try:
                    record = await assess(row, cfg, runtime, historical, gate)
                    records[str(row[3])] = record
                    if allow_retry and record.get('score') is None:
                        job = row[3]
                        state = disposition(record, r.read_json(job / 'attempts.json'),
                                            request_count(job), r.read_json(Path(record['episode']) / 'episode_result.json'))
                        if state == 'retry_pending':
                            queue.append(row)
                            queued = len(queue)
                        else:
                            held.append({'job': str(job), 'reason': state, 'errors': record.get('errors', [])})
                except Exception as exc:
                    fatal.append(f'{type(exc).__name__}:{exc}')
                finally:
                    active.pop(index, None)
        await asyncio.gather(*(worker(i) for i in range(gate.target)))

    reporter = asyncio.create_task(report())
    try:
        if pilot['stage'] != 'complete':
            await batch([x for x in pending if str(x[3]) in pilot['jobs']])
            if DRAIN:
                return None
            if fatal:
                raise RuntimeError(';'.join(fatal))
            pilot_records = [records[j] for j in pilot['jobs'] if j in records]
            if len(pilot_records) != len(pilot['jobs']):
                raise RuntimeError('pilot_incomplete')
            # Admission is based on protocol validity, never on high/low scores.
            if pilot_records and not any(x.get('score') is not None for x in pilot_records):
                raise RuntimeError('pilot_has_no_valid_assessments')
            pilot.update(stage='complete', completed_at=r.now(),
                         valid=sum(x.get('score') is not None for x in pilot_records),
                         invalid=sum(x.get('score') is None for x in pilot_records),
                         rubric_and_review_unchanged=True,
                         output_fallbacks=sum(len(list(Path(j).glob('episode/evaluations_v2/*/attempt_*/calls/*/output_fallback.json')))
                                              for j in pilot['jobs']))
            r.write_json(pilot_path, pilot)
        phase = 'rescoring'
        while not DRAIN:
            pending, records, held, counts = await asyncio.to_thread(inventory)
            r.write_json(RUN / 'optimized_held.json', {'at': r.now(), 'jobs': held})
            if not pending:
                if held:
                    r.status('judge', 'needs_attention', reason='invalid_scores_held_after_bounded_retries',
                             **progress(records, held, counts))
                    return 'held'
                return list(records.values())
            await batch(pending, allow_retry=True)
            if fatal:
                raise RuntimeError(';'.join(fatal))
        return None
    finally:
        reporter.cancel()
        try:
            await reporter
        except asyncio.CancelledError:
            pass


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    def drain(signum, frame):
        global DRAIN
        DRAIN = True
    signal.signal(signal.SIGUSR1, drain)
    r.write_json(RUN / 'optimized_launch.json', {'pid': os.getpid(), 'at': r.now(), 'stage': 'waiting_for_previous_judge'})
    with (RUN / 'judge.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        plan = r.selection.checked_plan(r.REPAIRS)
        policy = {'at': r.now(), 'initial_concurrency': 16, 'max_output_tokens': OUTPUT_LIMIT,
                  'fallback_output_tokens': FALLBACK_LIMIT, 'max_assessments': MAX_ASSESSMENTS,
                  'max_cumulative_requests_per_job': 32, 'budget_preserved': True,
                  'valid_results_never_retried': True, 'review_all_decisions': True,
                  'query_rounds': 12, 'review_query_rounds': 4,
                  'semantic_config_preserved': True, 'output_limit_is_transport_only': True,
                  'implementation_sha256': r.file_hash(__file__),
                  'frozen_plan_sha256': r.file_hash(r.REPAIRS / 'plan.json')}
        r.write_json(RUN / 'optimized_policy.json', policy)
        r.write_json(RUN / 'optimized_launch.json', {'pid': os.getpid(), 'at': r.now(), 'stage': 'running'})
        try:
            records = asyncio.run(drive(plan, policy))
            if records is None:
                r.status('judge', 'paused_at_episode_boundary')
            elif records != 'held':
                base.finalize(plan, records)
        except BaseException as exc:
            r.status('judge', 'needs_attention', reason=f'{type(exc).__name__}:{exc}')
            raise


if __name__ == '__main__':
    main()
