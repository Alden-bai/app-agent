"""Frozen best-SFT test rollout, bounded GPU scheduling, and paced judging."""
import argparse
import asyncio
from collections import deque
import fcntl
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness_awm.io import read_json, write_json, file_hash, now, digest
from harness_awm.training import selection
from harness_awm.training.local import checkpoint_identity
from harness_awm.tracking import SwanLabTracker
from scripts import resume_sft_validation as scheduler
from scripts import sft_server_recovery_20261110 as server
from scripts import sft_remaining_recheck_20261110 as judge_impl

RUN = ROOT / 'runs/sft_best_test_20261110_v1'
SOURCE = server.REPAIRS
COMPARISON = ROOT / 'runs/validation_comparison_20261110/summary.json'
ARCHIVE = Path('/root/autodl-tmp/awm-sft-best-test-20261110-v1')


def status(phase, stage, **extra):
    write_json(RUN / (phase + '_status.json'), dict(at=now(), pid=os.getpid(), stage=stage, **extra))


def prepare():
    if RUN.exists():
        raise RuntimeError('Existing run preserved; inspect and use run to resume')
    original = read_json(SOURCE / 'plan.json')
    assert file_hash(SOURCE / 'plan.json') == read_json(SOURCE / 'plan.sha256.json')['sha256']
    summary = read_json(COMPARISON)['summary']
    reports = {name: {'evaluation_split': 'validation', 'evaluation': dict(
        planned=s['planned'], result_records=s['planned'], scored=s['scored'],
        score_coverage=s['coverage'], confirmed_successes=s['success'],
        eval_reward={'sum': s['mean_reward'] * s['scored']})}
        for name, s in summary.items() if name in original['sft_candidates']}
    best = selection.choose_best(reports, original['minimum_coverage'])
    assert all(s['success'] + s['missing'] < summary[best]['success']
               for name, s in summary.items() if name != best and name in reports)
    checkpoint = read_json(SOURCE / 'validation/candidates.json')[best]
    assert checkpoint_identity(checkpoint['path']) == checkpoint
    current = selection.sources()
    changed = {k: {'before': original['source_hashes'].get(k), 'after': v}
               for k, v in current.items() if original['source_hashes'].get(k) != v}
    assert set(changed) <= {'harness_awm/judge_evidence.py'}, changed
    RUN.mkdir()
    ARCHIVE.mkdir(exist_ok=True)
    selected = dict(selected=best, checkpoint=checkpoint, selected_at=now(),
        validation=summary, test_used_for_selection=False,
        selection_rule=original['selection_rule'], validation_summary_sha256=file_hash(COMPARISON))
    write_json(RUN / 'selection.json', selected)
    write_json(RUN / 'selection.sha256.json', {'sha256': file_hash(RUN / 'selection.json')})
    plan = dict(original, source_hashes=current, created_at=now(), eval_workers=18,
        source_plan_sha256=file_hash(SOURCE / 'plan.json'), source_changes=changed,
        selected=best, gpu_target_gib=85, judge_deferred=True)
    write_json(RUN / 'plan.json', plan)
    write_json(RUN / 'plan.sha256.json', {'sha256': file_hash(RUN / 'plan.json')})
    write_json(RUN / 'judge_config.json', read_json(judge_impl.RUN / 'plan.json')['config'])
    write_json(RUN / 'policy.json', dict(at=now(), selected=best, tasks=588,
        gpu_target_gib=85, initial_workers=18, sustained_high_gib=86,
        test_used_for_selection=False, judge_policy='latest validation evidence and transport fixes',
        script_sha256=file_hash(__file__), archive=str(ARCHIVE),
        judge_source_hashes=judge_impl.hashes()))
    selection.prepare_evaluation(RUN, plan, {best: checkpoint}, 'test')
    assert len(read_json(RUN / 'test' / best / 'manifest.json')['candidates']) == 588
    status('gpu', 'prepared', selected=best, total=588)


def prepare_judge(plan):
    root = RUN / 'judge'
    if (root / 'plan.json').exists():
        saved = read_json(root / 'plan.json')
        assert file_hash(root / 'plan.json') == read_json(root / 'plan.sha256.json')['sha256']
        return saved
    config = read_json(RUN / 'judge_config.json')
    rows = []
    name = plan['selected']
    for item in read_json(RUN / 'test' / name / 'manifest.json')['candidates']:
        source = Path(item['path'])
        result = read_json(source / 'episode_result.json')
        if server.evaluation_score(result) is not None:
            continue
        if result['execution_integrity'] != 'valid' or result['terminal_state_status'] != 'trusted':
            continue
        key = digest([name, item['task_instance_id'], str(source)])[:24]
        rows.append(dict(key=key, candidate=name, task_id=item['task_instance_id'].split(':', 1)[1],
            task_instance_id=item['task_instance_id'], source_episode=str(source), reason='deferred_test_judge',
            source_result_sha256=file_hash(source / 'episode_result.json'),
            source_seal_sha256=file_hash(source / 'evidence_seal.json'), max_cumulative_requests=32))
        write_json(root / 'judge_jobs' / key / 'baseline_budget.json', {'baseline': {}})
    saved = dict(at=now(), config=config, cohort=rows, semantic_attempts=1,
        max_physical_requests_per_episode=32, source_hashes=judge_impl.hashes())
    write_json(root / 'plan.json', saved)
    write_json(root / 'plan.sha256.json', {'sha256': file_hash(root / 'plan.json')})
    write_json(root / 'transport_settings.json', dict(rpm=600, workers=16, target_tpm=2000000))
    return saved


async def judge(plan, tracker):
    jp = prepare_judge(plan)
    root = RUN / 'judge'
    judge_impl.RUN = root
    judge_impl.previous.RUN = root
    config = server.RunConfig.model_validate(jp['config'])
    assert os.environ.get(config.judge_api.api_key_env, '').strip(), 'Judge credential missing'
    fresh = not (root / 'provider_rate.sqlite').exists()
    gate = judge_impl.previous.SmoothGate(root / 'provider_rate.sqlite')
    if fresh:
        with sqlite3.connect(gate.path) as db:
            db.execute('UPDATE pacing SET tpm=2000000 WHERE id=1')
    runtime = server.recovery_judge()
    transport = judge_impl.FullOutputTransport(runtime.api_request, gate, root)
    runtime.api_request = transport.request
    runtime.PROMPT += judge_impl.ADDITIONAL_PROMPT
    runtime.EvidenceStore = judge_impl.ExpandedEvidenceStore
    runtime.model_view = judge_impl.expanded_view
    original_hash = runtime.config_hash
    runtime.config_hash = lambda cfg: digest({'original': original_hash(cfg), 'test_policy': file_hash(root / 'plan.json')})
    original_copy = judge_impl.sealed_copy
    def archived_copy(source, target):
        destination = ARCHIVE / 'judge' / target.parent.name / 'episode'
        provenance = original_copy(source, destination)
        target.symlink_to(destination, target_is_directory=True)
        return provenance
    judge_impl.sealed_copy = archived_copy
    records = {}
    for row in jp['cohort']:
        path = root / 'judge_jobs' / row['key'] / 'recovery.json'
        if path.exists():
            records[row['key']] = read_json(path)
    pending = deque(row for row in jp['cohort'] if row['key'] not in records)
    fatal = []
    active = {}
    def report(stage='running'):
        status('judge', stage, completed=len(records), total=len(jp['cohort']),
               remaining=len(jp['cohort'])-len(records), active=active,
               fatal=fatal, transport_stop=transport.stopped, rate=gate.snapshot())
    async def reporter():
        while True:
            report()
            await asyncio.sleep(10)
    async def work(index):
        while pending and not fatal and not transport.stopped:
            row = pending.popleft()
            active[index] = row['task_id']
            try:
                if min(shutil.disk_usage(RUN).free, shutil.disk_usage(ARCHIVE).free) < 1024**3:
                    raise RuntimeError('disk below 1 GiB')
                result = await judge_impl.assess(row, config, runtime)
                if result is not None:
                    records[row['key']] = result
            except Exception as exc:
                fatal.append(f'{type(exc).__name__}: {exc}')
            finally:
                active.pop(index, None)
    monitor = asyncio.create_task(reporter())
    try:
        await asyncio.gather(*(work(i) for i in range(16)))
    finally:
        monitor.cancel()
        try:
            await monitor
        except asyncio.CancelledError:
            pass
        report('needs_attention' if fatal or transport.stopped else 'complete')
    server.recovery.overlay_repairs(RUN / 'test', list(records.values()))
    result = selection.aggregate(RUN / 'test', plan)[plan['selected']]
    for record in records.values():
        for key, value in record['usage'].items():
            result['budget'][key] = result['budget'].get(key, 0) + value
    write_json(RUN / 'test' / plan['selected'] / 'metrics.json', result)
    write_json(RUN / 'summary.json', dict(at=now(), selected=plan['selected'],
        test_used_for_selection=False, test=selection.compact(result), budget=result['budget']))
    tracker.log({'test_final': selection.compact(result)}, step=100000)
    stage = 'complete' if not fatal and not transport.stopped and len(records) == len(jp['cohort']) else 'needs_attention'
    status('judge', stage, scored=result['evaluation']['scored'], total=588,
           remaining_unscored=588-result['evaluation']['scored'], fatal=fatal, transport_stop=transport.stopped)
    write_json(RUN / 'status.json', dict(at=now(), stage=stage, selected=plan['selected'], report=str(RUN / 'summary.json')))


def run(gpu_only=False):
    with (RUN / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = selection.checked_plan(RUN)
        assert file_hash(__file__) == read_json(RUN / 'policy.json')['script_sha256']
        assert judge_impl.hashes() == read_json(RUN / 'policy.json')['judge_source_hashes']
        tracker = SwanLabTracker(RUN / 'tracking', mode='offline', config={'stage': 'best-sft-test', 'gpu_target_gib': 85})
        try:
            with sqlite3.connect(RUN / 'test/queue.sqlite') as db:
                counts = dict(db.execute('SELECT state,count(*) FROM jobs GROUP BY state'))
            assert not any(v for k, v in counts.items() if k not in ('pending', 'done')), counts
            if counts.get('pending'):
                assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip(), 'GPU already in use'
                from scripts.sft_fast_eval_worker import bindings
                write_json(RUN / 'kernel_preflight.json', dict(at=now(), bindings=bindings()))
                scheduler.CONTROL = RUN
                scheduler.ARCHIVE = ARCHIVE / 'rollouts'
                scheduler.ARCHIVE.mkdir(exist_ok=True)
                scheduler.__file__ = str(Path(__file__).resolve())
                status('gpu', 'running', total=588, workers=18, target_gib=85)
                scheduler.run_stage(RUN, plan, 'test', tracker, 'gpu_' + str(int(time.time())))
            status('gpu', 'complete', total=588)
            if gpu_only:
                status('judge', 'awaiting_external_scoring_authorization', total=588)
                write_json(RUN / 'status.json', dict(at=now(), stage='gpu_complete_awaiting_judge', selected=plan['selected']))
            else:
                asyncio.run(judge(plan, tracker))
        except BaseException as exc:
            write_json(RUN / 'status.json', dict(at=now(), stage='needs_attention', error=f'{type(exc).__name__}: {exc}'))
            raise
        finally:
            tracker.finish()


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run', 'gpu', 'worker'])
    parser.add_argument('--root', type=Path)
    parser.add_argument('--split', default='test')
    parser.add_argument('--index', type=int)
    parser.add_argument('--control', type=Path)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'worker':
        from scripts.sft_fast_eval_worker import bindings
        bindings()
        server.worker(args)
    else:
        run(gpu_only=args.mode == 'gpu')


if __name__ == '__main__':
    main()
