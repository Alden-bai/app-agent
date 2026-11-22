"""Score completed, archived test trajectories while GPU inference continues."""
import asyncio
from collections import Counter
import fcntl
import os
from pathlib import Path
import shutil
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_sft_continuation as base

r = base.server
j = base.judge_impl
RUN = base.RUN
JUDGE = RUN / 'judge_stream'
WORKERS = 32
TPM = 2_000_000
RPM = 5_000


def discover(run, items, seen):
    """Admit only queue-done, stable archived evidence; never touch active jobs."""
    with sqlite3.connect(run / 'validation/queue.sqlite') as db:
        counts = dict(db.execute('SELECT state,count(*) FROM jobs GROUP BY state'))
        done = {task for (task,) in db.execute("SELECT task FROM jobs WHERE state='done'")}
    rows, skipped = [], {}
    for item in items:
        task = item['task_instance_id']
        if task not in done:
            continue
        source = Path(item['path'])
        if not source.is_symlink() or not (source / 'storage_archive.json').exists():
            continue  # The GPU supervisor is still archiving this episode.
        key = r.digest([item['candidate'], task, str(source)])[:24]
        if key in seen:
            continue
        result = r.read_json(source / 'episode_result.json')
        if r.evaluation_score(result) is not None:
            skipped[task] = 'already_scored'
            continue
        if result['execution_integrity'] != 'valid' or result['terminal_state_status'] != 'trusted':
            skipped[task] = 'untrusted_execution'
            continue
        rows.append(dict(key=key, candidate=item['candidate'], task_id=task.split(':', 1)[1],
            task_instance_id=task, source_episode=str(source), reason='streaming_validation_judge',
            source_result_sha256=r.file_hash(source / 'episode_result.json'),
            source_seal_sha256=r.file_hash(source / 'evidence_seal.json'), max_cumulative_requests=32))
    return rows, counts, skipped


def prepare(plan):
    JUDGE.mkdir(exist_ok=True)
    path = JUDGE / 'plan.json'
    if path.exists():
        assert r.file_hash(path) == r.read_json(JUDGE / 'plan.sha256.json')['sha256']
        policy = r.read_json(path)
        assert policy['source_hashes'] == j.hashes()
        assert policy['script_sha256'] == r.file_hash(__file__)
        return policy
    name = plan['selected']
    manifest = RUN / 'validation' / name / 'manifest.json'
    policy = dict(at=r.now(), candidate=name, config=r.read_json(RUN / 'judge_config.json'),
        items=[dict(item, candidate=name) for item in r.read_json(manifest)['candidates']],
        test_used_for_selection=False, source_hashes=j.hashes(), script_sha256=r.file_hash(__file__),
        source_manifest_sha256=r.file_hash(manifest), workers=WORKERS, tpm=TPM, rpm=RPM,
        semantics='one assessment per completed trusted trajectory; physical retries share rate gate and budget')
    r.write_json(path, policy)
    r.write_json(JUDGE / 'plan.sha256.json', {'sha256': r.file_hash(path)})
    r.write_json(JUDGE / 'transport_settings.json', dict(rpm=RPM, workers=WORKERS, target_tpm=TPM))
    return policy


async def drive(plan, policy):
    j.RUN = JUDGE
    j.previous.RUN = JUDGE
    config = r.RunConfig.model_validate(policy['config'])
    assert os.environ.get(config.judge_api.api_key_env, '').strip(), 'Judge credential missing'
    fresh = not (base.CONTROL / 'provider_rate.sqlite').exists()
    gate = j.previous.SmoothGate(base.CONTROL / 'provider_rate.sqlite')
    if fresh:
        with sqlite3.connect(gate.path) as db:
            db.execute('UPDATE pacing SET tpm=? WHERE id=1', (TPM,))
    with sqlite3.connect(gate.path) as db:
        gate.tpm = db.execute('SELECT tpm FROM pacing WHERE id=1').fetchone()[0]
    assert gate.tpm <= TPM and gate.rpm <= RPM
    runtime = r.recovery_judge()
    transport = j.FullOutputTransport(runtime.api_request, gate, JUDGE)
    runtime.api_request = transport.request
    runtime.PROMPT += j.ADDITIONAL_PROMPT
    runtime.EvidenceStore = j.ExpandedEvidenceStore
    runtime.model_view = j.expanded_view
    original_hash = runtime.config_hash
    runtime.config_hash = lambda cfg: r.digest({'original': original_hash(cfg), 'validation_policy': r.file_hash(JUDGE / 'plan.json')})
    original_copy = j.sealed_copy
    def archived_copy(source, target):
        destination = base.ARCHIVE / 'judge_stream' / target.parent.name / 'episode'
        if destination.exists():
            provenance = dict(original_episode=str(Path(source).resolve()),
                original_result_sha256=r.file_hash(Path(source) / 'episode_result.json'),
                original_seal_sha256=r.file_hash(Path(source) / 'evidence_seal.json'))
            assert r.verify_seal(destination) == r.verify_seal(source)
        else:
            provenance = original_copy(source, destination)
        target.symlink_to(destination, target_is_directory=True)
        return provenance
    j.sealed_copy = archived_copy
    records = {p.parent.name: r.read_json(p) for p in (JUDGE / 'judge_jobs').glob('*/recovery.json')}
    active, failed = {}, []
    queue_counts, skipped = {}, {}
    def report(stage='streaming'):
        value = dict(at=r.now(), pid=os.getpid(), stage=stage, total=len(policy['items']),
            completed=len(records), scored=sum(x['score'] is not None for x in records.values()),
            scores=dict(Counter(str(x['score']) for x in records.values())),
            active={row['key']: row['task_id'] for row in active.values()}, concurrency=WORKERS,
            gpu_queue=queue_counts, skipped=skipped, errors=failed, transport_stop=transport.stopped,
            rate=gate.snapshot(), remaining=len(policy['items'])-len(records)-len(skipped))
        r.write_json(JUDGE / 'status.json', value)
        r.write_json(RUN / 'judge_status.json', value)
    async def assess(row):
        if min(shutil.disk_usage(RUN).free, shutil.disk_usage(base.ARCHIVE).free) < 1024**3:
            raise RuntimeError('disk below 1 GiB')
        job = JUDGE / 'judge_jobs' / row['key']
        if not (job / 'baseline_budget.json').exists():
            r.write_json(job / 'baseline_budget.json', {'baseline': {}})
        r.write_json(job / 'admission.json', row)
        return await j.assess(row, config, runtime)
    while True:
        for task in list(active):
            if task.done():
                row = active.pop(task)
                try:
                    result = task.result()
                    if result is not None:
                        records[row['key']] = result
                except Exception as exc:
                    failed.append(f"{row['key']}: {type(exc).__name__}: {exc}")
        seen = set(records) | {row['key'] for row in active.values()}
        rows, queue_counts, skipped = discover(RUN, policy['items'], seen)
        if not failed and not transport.stopped:
            for row in rows[:max(0, WORKERS-len(active))]:
                task = asyncio.create_task(assess(row))
                active[task] = row
        report('needs_attention' if failed or transport.stopped else 'streaming')
        if failed or transport.stopped:
            if not active:
                return
        elif not active and not rows:
            gpu = r.read_json(RUN / 'gpu_status.json') if (RUN / 'gpu_status.json').exists() else {}
            if gpu.get('stage') == 'complete':
                assert queue_counts == {'done': len(policy['items'])}, queue_counts
                assert len(records)+len(skipped) == len(policy['items']), 'Completed episodes not all admitted'
                break
            state = r.read_json(RUN / 'status.json') if (RUN / 'status.json').exists() else {}
            if state.get('stage') == 'needs_attention':
                failed.append('GPU supervisor needs attention')
        if active:
            await asyncio.wait(active, timeout=5, return_when=asyncio.FIRST_COMPLETED)
        else:
            await asyncio.sleep(5)
    # Only update shared manifests/reports after GPU aggregation has released its lock.
    with (RUN / 'supervisor.lock').open('a') as lock:
        await asyncio.to_thread(fcntl.flock, lock, fcntl.LOCK_EX)
        r.recovery.overlay_repairs(RUN / 'validation', list(records.values()))
        result = r.selection.aggregate(RUN / 'validation', plan)[plan['selected']]
        for record in records.values():
            for key, value in record['usage'].items():
                result['budget'][key] = result['budget'].get(key, 0)+value
        r.write_json(RUN / 'validation' / plan['selected'] / 'metrics.json', result)
        r.write_json(RUN / 'summary.json', dict(at=r.now(), selected=plan['selected'],
            test_used_for_selection=False, validation=r.selection.compact(result), budget=result['budget']))
        r.write_json(RUN / 'stream_summary_status.json', dict(at=r.now(), stage='complete', selected=plan['selected'],
            scored=result['evaluation']['scored'], total=len(policy['items']), report=str(RUN / 'summary.json')))
        report('complete')


def persist(step):
    import tarfile
    destination = base.PERSISTENT / f'checkpoint-{step}-stream-scored.tar'
    temporary = destination.with_suffix('.tar.partial')
    with tarfile.open(temporary, 'w') as bundle:
        for path in (RUN, base.ARCHIVE / 'rollouts' / RUN.name, base.ARCHIVE / 'judge_stream'):
            if path.exists():
                bundle.add(path, arcname=str(path).lstrip('/'))
    temporary.replace(destination)
    r.write_json(destination.with_suffix('.sha256.json'), dict(sha256=r.file_hash(destination), at=r.now()))
    r.write_json(base.PERSISTENT / f'checkpoint-{step}-summary.json', r.read_json(RUN / 'summary.json'))


def main():
    global RUN, JUDGE
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, choices=base.STEPS, required=True)
    args = parser.parse_args()
    RUN = base.CONTROL / f'checkpoint-{args.step}'
    JUDGE = RUN / 'judge_stream'
    base.RUN = RUN
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    with (RUN / 'stream_judge.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = r.selection.checked_plan(RUN)
        assert r.read_json(base.CONTROL / 'scoring_authorization.json')['authorized']
        policy = prepare(plan)
        try:
            asyncio.run(drive(plan, policy))
            assert r.read_json(JUDGE / 'status.json')['stage'] == 'complete'
            persist(args.step)
        except BaseException as exc:
            value = dict(at=r.now(), pid=os.getpid(), stage='needs_attention', error=f'{type(exc).__name__}: {exc}')
            r.write_json(JUDGE / 'status.json', value)
            r.write_json(RUN / 'judge_status.json', value)
            raise


if __name__ == '__main__':
    main()
