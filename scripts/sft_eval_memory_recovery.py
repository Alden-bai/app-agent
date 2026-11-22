"""Preserve completed validation; retry only recorded CUDA OOM interruptions.

Wait for the existing workers to finish, merge audited infrastructure retries,
then select the SFT winner and run its/base test under the same EOS protocol.
"""
import argparse
import fcntl
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness_awm.io import read_json, write_json, file_hash, now
from harness_awm.tracking import SwanLabTracker
from harness_awm.training.selection import checked_plan, sources, init_queue, aggregate, compact, choose_best, run_evaluation


def cuda_oom(result):
    return (result.get('evaluation_valid') is not True
            and result.get('execution_integrity') == 'runtime_error'
            and result.get('termination_reason') == 'runtime_error'
            and ('OutOfMemoryError' in (result.get('error') or '')
                 or 'CUDA out of memory' in (result.get('error') or '')))


def recovered_candidates(manifest, destination):
    records, jobs, audit = [], [], []
    for index, original in enumerate(manifest['candidates']):
        folder = Path(original['path'])
        result = read_json(folder / 'episode_result.json')
        if result.get('task_instance_id') != original['task_instance_id']:
            raise ValueError('source result identity mismatch')
        record = dict(original)
        if cuda_oom(result):
            output = destination / 'jobs' / f't{index:05d}'
            record['path'] = str(output / 'episodes/t00000/a0')
            record['infrastructure_retry'] = 'recorded_cuda_oom'
            record['original_attempt'] = str(folder)
            jobs.append((destination.name, original['task_instance_id'], str(output)))
        else:
            record['reused_original_attempt'] = True
        audit.append({'task': original['task_instance_id'], 'original_path': str(folder),
            'original_result_sha256': file_hash(folder / 'episode_result.json'),
            'original_seal_sha256': file_hash(folder / 'evidence_seal.json'),
            'retry_cuda_oom': cuda_oom(result)})
        records.append(record)
    return records, jobs, audit


def wait_for_source(source, root):
    pids = read_json(source / 'validation/launch.json')['worker_pids']
    while True:
        live = [pid for pid in pids if Path(f'/proc/{pid}/stat').exists()
                and Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z']
        with sqlite3.connect((source / 'validation/queue.sqlite').as_uri() + '?mode=ro', uri=True) as db:
            counts = dict(db.execute('SELECT state,count(*) FROM jobs GROUP BY state'))
        write_json(root / 'status.json', {'stage': 'waiting_for_validation_before_oom_recovery',
            'source_run': str(source), 'source_queue': counts, 'live_source_workers': len(live), 'updated_at': now()})
        if not live:
            if any(value for state, value in counts.items() if state != 'done'):
                raise RuntimeError('source has missing attempts; only completed CUDA OOM records can be retried')
            # Let the original supervisor finish its final aggregation and close.
            with (source / 'supervisor.lock').open('a') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    time.sleep(5)
                    continue
            return
        time.sleep(20)


def prepare(source, root):
    plan = checked_plan(source)
    plan = {**plan, 'source_run': str(source), 'created_at': now(), 'eval_workers': 18,
        'source_hashes': sources(), 'source_plan_sha256': file_hash(source / 'plan.json'),
        'recovery_policy': 'retry only explicit CUDA OOM; retain all other original outcomes',
        'recovery_driver_sha256': file_hash(__file__)}
    write_json(root / 'plan.json', plan)
    write_json(root / 'plan.sha256.json', {'sha256': file_hash(root / 'plan.json')})
    candidates = read_json(source / 'validation/candidates.json')
    jobs, audits = [], {}
    for name in candidates:
        original = source / 'validation' / name / 'manifest.json'
        if file_hash(original) != read_json(original.with_name('manifest.sha256.json'))['sha256']:
            raise ValueError('original manifest changed')
        manifest = read_json(original)
        destination = root / 'validation' / name
        records, new_jobs, audit = recovered_candidates(manifest, destination)
        jobs.extend(new_jobs)
        audits[name] = audit
        manifest.update(candidates=records, source_run=str(source), recovery_policy=plan['recovery_policy'],
                        created_at=now(), source_manifest_sha256=file_hash(original))
        write_json(destination / 'manifest.json', manifest)
        write_json(destination / 'manifest.sha256.json', {'sha256': file_hash(destination / 'manifest.json')})
    write_json(root / 'validation/candidates.json', candidates)
    write_json(root / 'recovery_audit.json', audits)
    init_queue(root / 'validation/queue.sqlite', jobs)
    return plan, candidates, len(jobs)


def reports_with_original_budget(source, root, plan):
    reports = aggregate(root / 'validation', plan)
    for name, report in reports.items():
        original_budget = read_json(source / 'validation' / name / 'metrics.json')['budget']
        for key, value in original_budget.items():
            report['budget'][key] = report['budget'].get(key, 0) + value
        write_json(root / 'validation' / name / 'metrics.json', report)
    return reports


def run_retries(source, root, plan, count, tracker):
    env = dict(os.environ, SWANLAB_MODE='disabled', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        PYTORCH_ALLOC_CONF='expandable_segments:True', TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    processes, logs = [], []
    step = 0
    try:
        for index in range(min(plan['eval_workers'], count)):
            log = (root / 'validation' / f'worker_{index:02d}.log').open('xb')
            logs.append(log)
            processes.append(subprocess.Popen([sys.executable, '-m', 'harness_awm.training.selection', 'worker',
                '--output', str(root), '--split', 'validation', '--index', str(index)], cwd=ROOT,
                env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT))
            time.sleep(1)
        write_json(root / 'validation/launch.json', {'worker_pids': [p.pid for p in processes], 'started_at': now()})
        while True:
            exits = [p.poll() for p in processes]
            reports = reports_with_original_budget(source, root, plan)
            with sqlite3.connect(root / 'validation/queue.sqlite') as db:
                counts = dict(db.execute('SELECT state,count(*) FROM jobs GROUP BY state'))
            write_json(root / 'status.json', {'stage': 'validation_oom_recovery', 'updated_at': now(),
                'queue': counts, 'live_workers': sum(x is None for x in exits),
                'reused_attempts': sum(len(v['evaluation']['episodes']) for v in reports.values()) - count,
                'retry_tasks': count, 'candidates': {k: compact(v) for k, v in reports.items()}})
            tracker.log({'validation': {k: compact(v) for k, v in reports.items()}, 'retry_queue': counts}, step=step)
            step += 1
            if all(x is not None for x in exits):
                if any(x != 0 for x in exits) or any(v for k, v in counts.items() if k != 'done'):
                    raise RuntimeError('infrastructure retry queue incomplete')
                return reports, step
            time.sleep(15)
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        for log in logs:
            log.close()


def execute(source, root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'launch.json').exists():
            raise ValueError('recovery already launched')
        write_json(root / 'launch.json', {'pid': os.getpid(), 'started_at': now()})
        tracker = None
        failed = True
        try:
            wait_for_source(source, root)
            plan, candidates, count = prepare(source, root)
            tracker = SwanLabTracker(root, mode='online', config={'stage': 'sft-validation-oom-recovery',
                'stop_policy': plan['stop_policy'], 'eval_workers': 18, 'infrastructure_retries': count,
                'model_failures_retried': False})
            write_json(root / 'tracking.json', {'url': tracker.url})
            reports, step = run_retries(source, root, plan, count, tracker)
            best = choose_best({k: reports[k] for k in plan['sft_candidates']}, plan['minimum_coverage'])
            selection = {'selected': best, 'checkpoint': candidates[best], 'selected_at': now(),
                'stop_policy': plan['stop_policy'], 'validation': {k: compact(v) for k, v in reports.items()},
                'selection_rule': plan['selection_rule'], 'test_used_for_selection': False,
                'recovery_policy': plan['recovery_policy'], 'infrastructure_retry_count': count}
            write_json(root / 'selection.json', selection)
            write_json(root / 'selection.sha256.json', {'sha256': file_hash(root / 'selection.json')})
            test, step = run_evaluation(root, plan, {'base': candidates['base'], best: candidates[best]}, 'test', tracker, step)
            summary = {'selected': best, 'checkpoint': candidates[best], 'stop_policy': plan['stop_policy'],
                'baseline': {'validation': compact(reports['base']), 'test': compact(test['base'])},
                'validation': compact(reports[best]), 'test': compact(test[best]),
                'test_used_for_selection': False, 'infrastructure_retry_count': count, 'completed_at': now()}
            write_json(root / 'summary.json', summary)
            tracker.log({'summary': summary}, step=step)
            write_json(root / 'status.json', {'stage': 'finished', 'selected': best, 'updated_at': now()})
            failed = False
        except BaseException as exc:
            write_json(root / 'status.json', {'stage': 'failed', 'error': f'{type(exc).__name__}: {exc}', 'updated_at': now()})
            raise
        finally:
            if tracker:
                tracker.finish(failed=failed)


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    execute(a.source.resolve(), a.output.resolve())


if __name__ == '__main__':
    main()
