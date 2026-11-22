"""One scoped, resumable recheck of the 1,061 latest-429 validation episodes.

This entry point never selects a checkpoint or starts training/test evaluation.
"""
import argparse
import asyncio
from collections import Counter, deque
from contextlib import ExitStack
from email.utils import parsedate_to_datetime
import fcntl
import os
from pathlib import Path
import random
import re
import shutil
import signal
import sqlite3
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import sft_parallel_judge_20261110 as base

r = base.r
RUN = ROOT / 'runs/sft_rate_limited_recheck_20261110_v1'
DATA = Path('/root/autodl-tmp/awm-rate-limited-recheck-20261110-v1')
DRAIN = False


def transport_settings():
    path = RUN / 'transport_settings.json'
    return r.read_json(path) if path.exists() else {'rpm': 60, 'workers': 8}


class TransportPaused(RuntimeError):
    pass


def retry_seconds(error, attempt, now=None):
    now = time.time() if now is None else now
    delay = min(60, 2 ** (attempt + 2))
    match = re.search(r'try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?', str(error), re.I)
    if match:
        delay = max(delay, sum(float(v or 0) * f for v, f in zip(match.groups(), (3600, 60, 1))))
    response = getattr(error, 'response', None)
    if response is not None:
        value = response.headers.get('retry-after')
        if value:
            try:
                delay = max(delay, float(value))
            except ValueError:
                try:
                    delay = max(delay, parsedate_to_datetime(value).timestamp() - now)
                except (TypeError, ValueError, OverflowError):
                    pass
    return max(0, delay)


def error_policy(error):
    status = getattr(error, 'status_code', None)
    body = str(getattr(error, 'body', '')).lower()
    if status in (400, 401, 403, 404, 422) or any(s in body for s in (
            'insufficient_quota', 'billing_hard_limit', 'billing_not_active',
            'organization_spend_limit_exceeded', 'project_spend_limit_exceeded',
            'organization_usage_limit_exceeded', 'project_usage_limit_exceeded',
            'usage_limit_reached', 'quota_exceeded')):
        return 'pause'
    if status in (408, 409, 429) or (isinstance(status, int) and status >= 500):
        return 'retry'
    if type(error).__name__ in ('APIConnectionError', 'APITimeoutError'):
        return 'retry'
    return 'fail'


class SmoothGate(base.RateGate):
    """Reserve a full window, and space sends by the declared token allowance."""
    def __init__(self, path):
        super().__init__(path, tpm=85000, rpm=transport_settings()['rpm'])
        with sqlite3.connect(self.path) as db:
            db.execute('CREATE TABLE IF NOT EXISTS pacing(id INTEGER PRIMARY KEY,next_send REAL,tpm INTEGER,limited_until REAL)')
            db.execute('INSERT OR IGNORE INTO pacing VALUES(1,0,85000,0)')

    def delay(self, now, upper):
        with sqlite3.connect(self.path) as db:
            next_send, self.tpm, _ = db.execute('SELECT next_send,tpm,limited_until FROM pacing WHERE id=1').fetchone()
        if upper > 95000:
            raise ValueError('single_request_above_safe_provider_allowance')
        target = self.tpm
        # A single larger request waits for an empty window, then its longer
        # spacing still respects the reduced average rate after a 429.
        self.tpm = max(target, upper)
        try:
            wait = super().delay(now, upper)
        finally:
            self.tpm = target
        return max(wait, next_send - now, 0)

    async def acquire(self, upper, check=lambda: None):
        while True:
            check()
            async with self.lock:
                wait = self.delay(time.time(), upper)
                if not wait:
                    at = time.time()
                    with sqlite3.connect(self.path) as db:
                        event = db.execute("INSERT INTO events(started,tokens,status) VALUES(?,?,'reserved')", (at, upper)).lastrowid
                        db.execute('UPDATE pacing SET next_send=? WHERE id=1', (at + max(60 / self.rpm, upper * 60 / self.tpm),))
                    return event
            await asyncio.sleep(min(wait, 1))

    def finish(self, event, usage=None):
        with sqlite3.connect(self.path) as db:
            if isinstance(usage, int) and usage >= 0:
                db.execute("UPDATE events SET tokens=max(tokens,?),status='response' WHERE id=?", (usage, event))
            else:
                db.execute("UPDATE events SET status='unknown_reserved' WHERE id=?", (event,))

    def limited(self, until):
        with sqlite3.connect(self.path) as db:
            tpm, wave = db.execute('SELECT tpm,limited_until FROM pacing WHERE id=1').fetchone()
            if time.time() >= wave:
                tpm = max(40000, int(tpm * .85))
            db.execute('UPDATE pacing SET tpm=?,limited_until=max(limited_until,?) WHERE id=1', (tpm, until))
            db.execute('UPDATE control SET until=max(until,?) WHERE id=1', (until,))
        self.tpm = tpm


class Transport:
    def __init__(self, api, gate, run):
        self.api, self.gate, self.run = api, gate, Path(run)
        self.stopped = None

    def check(self):
        if self.stopped:
            raise TransportPaused(self.stopped['reason'])

    def pause(self, reason, **details):
        if self.stopped is None:
            self.stopped = dict(reason=reason, at=r.now(), **details)
            r.write_json(self.run / 'transport_stop.json', self.stopped)

    async def exact(self, config, wire, budget, key, stage, folder, **kw):
        cache = budget.path.parent / 'response_cache'
        fingerprint = r.digest({'api': config.model_dump(mode='json'), 'wire': wire, 'stage': stage})
        entry = cache / (fingerprint + '.json')
        intents = cache / 'pending' / fingerprint
        if not entry.exists():
            # A process may die after the backend saved its response but before
            # the cache index was committed. Recover that already-paid response.
            for intent_path in sorted(intents.glob('*.json')):
                saved = r.read_json(intent_path)
                response_path = Path(saved['response_path'])
                if response_path.exists() and budget.request_record(saved['physical_key']):
                    r.read_json(response_path)  # Reject incomplete/corrupt files.
                    saved['response_sha256'] = r.file_hash(response_path)
                    r.write_json(entry, saved)
                    break
        if entry.exists():
            saved = r.read_json(entry)
            response_path = Path(saved['response_path'])
            if r.file_hash(response_path) != saved['response_sha256']:
                raise ValueError('cached_response_hash_mismatch')
            r.write_json(Path(folder) / 'cache_replay.json', saved)
            return r.read_json(response_path)
        upper = base.quota_tokens(wire['messages'], wire.get('max_completion_tokens', wire.get('max_tokens', 0)))
        for attempt in range(4):
            self.check()
            event = await self.gate.acquire(upper, self.check)
            physical_key = key + '/physical_' + uuid.uuid4().hex
            physical_folder = Path(folder) / ('physical_' + physical_key.rsplit('_', 1)[-1])
            r.write_json(intents / (physical_key.rsplit('_', 1)[-1] + '.json'), dict(
                request_sha256=fingerprint, response_path=str(physical_folder / 'response.json'), physical_key=physical_key))
            try:
                response = await self.api(config, wire, budget, physical_key, stage, physical_folder, **kw)
                self.gate.finish(event, (response.get('usage') or {}).get('total_tokens'))
                path = physical_folder / 'response.json'
                r.write_json(entry, {'request_sha256': fingerprint, 'response_path': str(path),
                                     'response_sha256': r.file_hash(path), 'physical_key': physical_key})
                return response
            except Exception as error:
                self.gate.finish(event)
                policy = error_policy(error)
                if policy == 'fail':
                    raise
                delay = retry_seconds(error, attempt) + random.uniform(.2, 1.5)
                status = getattr(error, 'status_code', None)
                if status == 429:
                    self.gate.limited(time.time() + delay)
                r.write_json(physical_folder / 'retry_policy.json', dict(
                    status=status, exception=type(error).__name__, policy=policy,
                    wait_seconds=delay, retry=attempt + 1, same_request_sha256=fingerprint))
                if policy == 'pause' or delay > 900 or attempt == 3:
                    self.pause('provider_' + type(error).__name__, retry_after=time.time() + delay,
                               evidence=str(physical_folder), policy=policy)
                    raise TransportPaused(self.stopped['reason']) from error
                if status != 429:
                    await asyncio.sleep(delay)
        raise AssertionError('unreachable')

    async def request(self, config, kwargs, budget, key, stage, folder, **kw):
        for cap in (4096, 8192):
            wire = dict(kwargs, max_completion_tokens=cap)
            target = Path(folder) / str(cap)
            response = await self.exact(config, wire, budget, key + '/' + str(cap), stage, target, **kw)
            if response['choices'][0]['finish_reason'] != 'length' or cap == 8192:
                return response
            r.write_json(Path(folder) / 'output_fallback.json', dict(
                messages_sha256=r.digest(kwargs['messages']), partial_output_discarded=True, from_cap=4096, to_cap=8192))


def sealed_link_copy(source, target):
    """Hardlink only sealed read-only inputs; result and seal are separate files."""
    source, target = Path(source).resolve(), Path(target)
    seal = r.verify_seal(source)
    if target.exists():
        raise ValueError('destination_exists')
    temp = target.with_name(target.name + '.preparing')
    temp.mkdir(parents=True, exist_ok=True)
    linked = []
    for name in set(seal) | {'evidence_seal.json', 'episode_result.json'}:
        src, dst = (source / name).resolve(), temp / name
        if not src.is_relative_to(source) or not dst.resolve().is_relative_to(temp.resolve()):
            raise ValueError('evidence_path_escape')
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            if name in seal and name not in ('episode_result.json', 'evidence_seal.json'):
                os.link(src, dst)
            else:
                shutil.copy2(src, dst)
        if name in seal:
            linked.append(name)
    if r.verify_seal(temp) != seal:
        raise ValueError('copy_seal_mismatch')
    os.rename(temp, target)
    return dict(original_episode=str(source), original_result_sha256=r.file_hash(source / 'episode_result.json'),
                original_seal_sha256=r.file_hash(source / 'evidence_seal.json'), sealed_hardlinks=sorted(linked))


def source_hashes():
    hashes = r.selection.sources()
    for name in ('sft_rate_limited_recheck_20261110.py', 'sft_rate_recheck_inputs.py',
                 'recovery_judge_runtime.py', 'sft_recovery_support.py', 'sft_parallel_judge_20261110.py',
                 'sft_server_recovery_20261110.py', 'sft_judge_recovery.py'):
        path = ROOT / 'scripts' / name
        hashes[str(path.relative_to(ROOT))] = r.file_hash(path)
    return hashes


def prepare():
    from scripts.sft_rate_recheck_inputs import load_cohort
    RUN.mkdir(parents=True, exist_ok=True)
    if (RUN / 'plan.json').exists():
        plan = checked_plan()
        prepare_storage()
        return plan
    original = r.REPAIRS / 'plan.json'
    if r.file_hash(original) != r.read_json(r.REPAIRS / 'plan.sha256.json')['sha256']:
        raise ValueError('original_plan_changed')
    cohort = load_cohort()
    for row in cohort:
        source = Path(row['source_episode'])
        result = r.read_json(source / 'episode_result.json')
        if r.evaluation_score(result) is not None or result['execution_integrity'] != 'valid' or result['terminal_state_status'] != 'trusted':
            raise ValueError('cohort_no_longer_pending_and_trusted:' + row['key'])
        row.update(source_result_sha256=r.file_hash(source / 'episode_result.json'),
                   source_seal_sha256=r.file_hash(source / 'evidence_seal.json'))
    config = r.read_json(original)['config']
    config['judge_v2']['query_rounds'] = 12
    plan = dict(at=r.now(), original_plan=str(original), original_plan_sha256=r.file_hash(original),
                cohort_audit_sha256=r.file_hash(ROOT / 'runs/pending_judge_audit_20261110_v1/pending_inventory.csv'),
                cohort=cohort, config=config, source_hashes=source_hashes(),
                policy=dict(initial_tpm=85000, workers=8, pilot=4, pilot_min_valid=3,
                            output_tokens=4096, fallback_output_tokens=8192, max_historical_requests=32,
                            max_transient_attempts=4, semantic_assessments=1,
                            changed_evidence_view='snapshot provenance, protected runtime/trajectory/final_answer, pagination cursors',
                            resume='exact request response cache, original inputs sealed', automatic_selection=False))
    r.write_json(RUN / 'plan.json', plan)
    r.write_json(RUN / 'plan.sha256.json', {'sha256': r.file_hash(RUN / 'plan.json')})
    prepare_storage()
    return plan


def prepare_storage():
    DATA.mkdir(parents=True, exist_ok=True)
    link = RUN / 'judge_jobs'
    if not link.exists():
        link.symlink_to(DATA, target_is_directory=True)
    if link.resolve() != DATA.resolve():
        raise ValueError('unexpected_job_storage_path')


def checked_plan():
    plan = r.read_json(RUN / 'plan.json')
    expected = dict(plan['source_hashes'])
    amendment_path = RUN / 'transport_settings.json'
    if amendment_path.exists():
        amendment = r.read_json(amendment_path)
        script = 'scripts/sft_rate_limited_recheck_20261110.py'
        if amendment['previous_controller_sha256'] != expected[script] or amendment['base_plan_sha256'] != r.file_hash(RUN / 'plan.json'):
            raise ValueError('transport_amendment_lineage_mismatch')
        expected[script] = amendment['controller_sha256']
    if r.file_hash(RUN / 'plan.json') != r.read_json(RUN / 'plan.sha256.json')['sha256'] or expected != source_hashes():
        raise ValueError('recheck_plan_or_implementation_changed')
    return plan


async def assess(row, config, runtime):
    from scripts.sft_rate_recheck_inputs import merge_budget
    job = RUN / 'judge_jobs' / row['key']
    job.mkdir(parents=True, exist_ok=True)
    source, target = Path(row['source_episode']), job / 'episode'
    if r.file_hash(source / 'episode_result.json') != row['source_result_sha256'] or r.file_hash(source / 'evidence_seal.json') != row['source_seal_sha256']:
        raise ValueError('source_changed:' + row['key'])
    if not target.exists():
        r.write_json(job / 'provenance.json', sealed_link_copy(source, target))
    if not (job / 'baseline_budget.json').exists():
        sidecar = job / 'budget.baseline.json'
        metadata = r.read_json(sidecar) if (job / 'budget.sqlite').exists() and sidecar.exists() else merge_budget(row['budget_sources'], job / 'budget.sqlite')
        r.write_json(job / 'baseline_budget.json', metadata)
    baseline = r.read_json(job / 'baseline_budget.json')['baseline']
    budget = r.Budget(job / 'budget.sqlite', 32, None)
    task = r.TaskSpec.model_validate(r.read_json(target / 'task.json'))
    bundle = r.input_from_manifest(target, task, config.judge_inputs_manifest)
    assessment = await runtime.evaluate_input(bundle, config, budget, asyncio.Semaphore(1), target / 'evaluations_v2')
    if any('TransportPaused' in str(e) for e in assessment.get('errors', [])):
        r.write_json(job / 'deferred.json', dict(at=r.now(), errors=assessment['errors'], assessment=assessment['attempt']))
        return None
    outcome = r.settle_reward(assessment, bundle, task, config)
    r.write_json(target / 'rewards_v2' / (r.digest(outcome) + '.json'), outcome)
    result = r.apply_evaluation(target, outcome)
    r.verify_seal(target)
    if r.file_hash(source / 'episode_result.json') != row['source_result_sha256']:
        raise ValueError('source_result_changed')
    cumulative = budget.summary()
    record = dict(candidate=row['candidate'], task_instance_id=task.task_instance_id,
                  task_id=row['task_id'], source_episode=str(source), episode=str(target),
                  score=r.evaluation_score(result), errors=assessment['errors'],
                  evaluation_status=result.get('evaluation_status'), at=r.now(),
                  usage={k: v - baseline.get(k, 0) for k, v in cumulative.items()},
                  cumulative_usage=cumulative)
    r.write_json(job / 'recovery.json', record)
    return record


async def drive(plan):
    config = r.RunConfig.model_validate(plan['config'])
    if not os.environ.get(config.judge_api.api_key_env, '').strip():
        raise ValueError('missing_api_key_environment_variable')
    gate = SmoothGate(RUN / 'provider_rate.sqlite')
    gate.cooldown(time.time() + 60)  # Account for any previous process's last window.
    runtime = r.recovery_judge()
    transport = Transport(runtime.api_request, gate, RUN)
    runtime.api_request = transport.request
    original_hash = runtime.config_hash
    runtime.config_hash = lambda cfg: r.digest({'runtime': original_hash(cfg), 'recheck_plan': r.file_hash(RUN / 'plan.json')})
    records, active, fatal = {}, {}, []
    for row in plan['cohort']:
        path = RUN / 'judge_jobs' / row['key'] / 'recovery.json'
        if path.exists():
            records[row['key']] = r.read_json(path)
    phase = 'pilot'

    def report():
        valid = sum(x['score'] is not None for x in records.values())
        rate = gate.snapshot()
        stage = 'needs_attention' if fatal or transport.stopped else ('draining' if DRAIN else phase)
        r.write_json(RUN / 'status.json', dict(at=r.now(), pid=os.getpid(), stage=stage,
                     workers=transport_settings()['workers'],
                     total=len(plan['cohort']), completed=len(records), valid=valid,
                     needs_review=len(records)-valid, remaining=len(plan['cohort'])-len(records),
                     active=active, rate=rate, fatal=fatal, transport_stop=transport.stopped,
                     scores=dict(Counter(str(x['score']) for x in records.values() if x['score'] is not None))))
        r.write_json(RUN / 'results.json', list(records.values()))

    async def reporter():
        while True:
            report()
            await asyncio.sleep(5)

    async def batch(rows, workers):
        queue = deque(x for x in rows if x['key'] not in records)
        async def worker(index):
            while queue and not (fatal or transport.stopped or DRAIN):
                if min(shutil.disk_usage(RUN).free, shutil.disk_usage(DATA).free) < 1024 ** 3:
                    fatal.append('disk_below_1GiB'); return
                row = queue.popleft()
                active[index] = dict(key=row['key'], candidate=row['candidate'], task_id=row['task_id'])
                try:
                    result = await assess(row, config, runtime)
                    if result is not None:
                        records[row['key']] = result
                except Exception as error:
                    fatal.append(type(error).__name__ + ':' + str(error))
                finally:
                    active.pop(index, None)
        await asyncio.gather(*(worker(i) for i in range(workers)))

    task = asyncio.create_task(reporter())
    try:
        pilot_rows = plan['cohort'][:4]
        await batch(pilot_rows, 4)
        pilot = [records[x['key']] for x in pilot_rows if x['key'] in records]
        pilot_valid = sum(x['score'] is not None for x in pilot)
        r.write_json(RUN / 'pilot.json', dict(at=r.now(), completed=len(pilot), valid=pilot_valid,
                                            passed=len(pilot)==4 and pilot_valid>=3, keys=[x['key'] for x in pilot_rows]))
        if len(pilot) == 4 and pilot_valid < 3:
            fatal.append('pilot_validity_below_3_of_4')
        if not (fatal or transport.stopped or DRAIN) and len(pilot) == 4:
            phase = 'rescoring'
            await batch(plan['cohort'][4:], transport_settings()['workers'])
        phase = 'complete' if len(records) == len(plan['cohort']) else 'paused'
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        report()


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'run', 'status'))
    args = parser.parse_args()
    if args.mode == 'status':
        print(r.read_json(RUN / 'status.json')); return
    def drain(*_):
        global DRAIN
        DRAIN = True
    signal.signal(signal.SIGUSR1, drain)
    signal.signal(signal.SIGTERM, drain)
    RUN.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for path in (RUN / 'run.lock', r.RUN / 'judge.lock', r.REPAIRS / 'supervisor.lock'):
            lock = stack.enter_context(path.open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare()
        if args.mode == 'prepare':
            print({'prepared': len(plan['cohort']), 'run': str(RUN)}); return
        try:
            asyncio.run(drive(plan))
        except Exception as error:
            r.write_json(RUN / 'status.json', dict(at=r.now(), stage='needs_attention', reason=type(error).__name__ + ':' + str(error)))
            raise


if __name__ == '__main__':
    main()
