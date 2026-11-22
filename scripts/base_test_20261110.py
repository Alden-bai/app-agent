"""User-requested base-only frozen test; defer Judge until validation releases it."""
import argparse
import asyncio
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
from harness_awm.io import read_json, write_json, file_hash, now
from harness_awm.training import selection
from harness_awm.tracking import SwanLabTracker
from scripts import sft_server_recovery_20261110 as server
from scripts import sft_parallel_judge_20261110 as parallel
from scripts import resume_sft_validation as scheduler

RUN = ROOT / 'runs/base_test_20261110_v1'
VALIDATION = server.REPAIRS
VALIDATION_CONTROL = server.RUN
PIPELINE = ROOT / 'runs/grpo_pipeline_20261110_v1'


def status(mode, stage, **extra):
    write_json(RUN / (mode + '_status.json'), dict(stage=stage, at=now(), pid=os.getpid(), **extra))


def prepare():
    if RUN.exists():
        raise RuntimeError('Existing base test preserved; use run to resume after inspection')
    plan = selection.checked_plan(VALIDATION)
    RUN.mkdir()
    for name in ('plan.json', 'plan.sha256.json'):
        shutil.copy2(VALIDATION / name, RUN / name)
    candidates = read_json(VALIDATION / 'validation/candidates.json')
    selection.prepare_evaluation(RUN, plan, {'base': candidates['base']}, 'test')
    assert len(read_json(RUN / 'test/base/manifest.json')['candidates']) == 588
    write_json(RUN / 'policy.json', {
        'at': now(), 'models': ['base'], 'split': 'test', 'tasks': 588,
        'gpu_target_gib': 85, 'initial_workers': 18, 'judge_deferred': True,
        'judge_scheduling': 'wait for validation judge.lock, then four concurrent jobs',
        'frozen_plan_sha256': file_hash(RUN / 'plan.json'),
        'test_used_for_selection': False, 'script_sha256': file_hash(__file__),
    })
    write_json(PIPELINE / 'test_authorization.json', {
        'at': now(), 'base_test_run': str(RUN), 'checkpoint_tests_allowed': False,
        'checkpoint_cleanup_allowed': False,
        'reason': 'User requests base test now; all four SFT checkpoints await a later explicit test request',
    })
    status('gpu', 'prepared')


def discover_test():
    pending, records, untrusted = [], [], 0
    for item in read_json(RUN / 'test/base/manifest.json')['candidates']:
        source = Path(item['path'])
        result_path = source / 'episode_result.json'
        if not result_path.exists():
            raise RuntimeError('Missing completed test result: ' + str(source))
        result = read_json(result_path)
        if server.evaluation_score(result) is not None:
            continue
        if result['execution_integrity'] != 'valid' or result['terminal_state_status'] != 'trusted':
            untrusted += 1
            continue
        key = server.digest(['base', item['task_instance_id'], str(source)])[:24]
        job = RUN / 'judge_jobs' / key
        if (job / 'recovery.json').exists():
            records.append(read_json(job / 'recovery.json'))
        else:
            pending.append(('base', item, source, job))
    return pending, records, untrusted


def judge(plan):
    status('judge', 'waiting_for_validation_judge', lock=str(VALIDATION_CONTROL / 'judge.lock'))
    with (VALIDATION_CONTROL / 'judge.lock').open('a') as provider_lock:
        fcntl.flock(provider_lock, fcntl.LOCK_EX)
        server.RUN = RUN
        parallel.RUN = RUN
        parallel.discover = discover_test
        (RUN / 'judge_jobs').mkdir(exist_ok=True)
        write_json(RUN / 'parallel_handoff.json', {'at': time.time()})
        records = asyncio.run(parallel.drive(plan))
        server.recovery.overlay_repairs(RUN / 'test', records)
        report = selection.aggregate(RUN / 'test', plan)['base']
        for record in records:
            for key, value in record['usage'].items():
                report['budget'][key] = report['budget'].get(key, 0) + value
        write_json(RUN / 'test/base/metrics.json', report)
        summary = {'at': now(), 'model': 'base', 'test_used_for_selection': False,
                   'test': selection.compact(report), 'budget': report['budget']}
        write_json(RUN / 'summary.json', summary)
        e = report['evaluation']
        status('judge', 'complete' if e['scored'] == 588 else 'needs_attention',
               scored=e['scored'], remaining=588-e['scored'], report=str(RUN / 'summary.json'))


def run():
    with (RUN / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = selection.checked_plan(RUN)
        with sqlite3.connect(RUN / 'test/queue.sqlite') as db:
            counts = dict(db.execute('SELECT state,count(*) FROM jobs GROUP BY state'))
        if any(v for k, v in counts.items() if k not in ('pending', 'done')):
            raise RuntimeError('Test queue needs ownership audit; evidence retained')
        if counts.get('pending'):
            assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                '--format=csv,noheader,nounits'], text=True).strip(), 'GPU already in use'
            from scripts.sft_fast_eval_worker import bindings
            write_json(RUN / 'kernel_preflight.json', {'at': now(), 'bindings': bindings()})
            tracker = SwanLabTracker(RUN / 'tracking', mode='offline', config={'stage': 'base-only-test', 'gpu_target_gib': 85})
            try:
                status('gpu', 'running', tasks=588, workers=18, gpu_target_gib=85)
                scheduler.__file__ = str(Path(__file__).resolve())
                scheduler.run_stage(RUN, plan, 'test', tracker, 'base_test_' + str(int(time.time())))
                status('gpu', 'complete', judge_pending=True)
            finally:
                tracker.finish()
        else:
            status('gpu', 'complete', judge_pending=True)
        judge(plan)


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare', 'run', 'worker'])
    p.add_argument('--root', type=Path)
    p.add_argument('--split', default='test')
    p.add_argument('--index', type=int)
    p.add_argument('--control', type=Path)
    args = p.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'worker':
        server.worker(args)
    else:
        try:
            run()
        except BaseException as exc:
            phase = 'judge' if read_json(RUN / 'gpu_status.json')['stage'] == 'complete' else 'gpu'
            status(phase, 'needs_attention', reason=f'{type(exc).__name__}: {exc}')
            raise


if __name__ == '__main__':
    main()
