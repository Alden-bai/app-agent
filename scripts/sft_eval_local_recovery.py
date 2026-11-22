"""Continue frozen evaluation with local SQLite and verified remote archiving."""
import argparse
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
from harness_awm.backend import Budget
from harness_awm.tracking import SwanLabTracker
from harness_awm.training.selection import (checked_plan, init_queue, aggregate, compact,
    choose_best, prepare_evaluation)
from scripts.sft_eval_memory_recovery import cuda_oom



def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
    except FileNotFoundError:
        return False


def tree_hashes(path):
    result = {}
    for p in sorted(path.rglob('*')):
        if p.is_symlink():
            result[str(p.relative_to(path))] = ['symlink', os.readlink(p)]
        elif p.is_file():
            result[str(p.relative_to(path))] = ['file', file_hash(p)]
    return result


def archive_episode(folder, target):
    """Only called for queue-done episodes. Verify every byte before local removal."""
    marker = folder / 'storage_archive.json'
    if marker.exists():
        return
    before = tree_hashes(folder)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        staging = target.with_name(target.name + '.copying')
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(folder, staging, symlinks=True)
        if tree_hashes(staging) != before:
            raise RuntimeError('archive hash mismatch')
        staging.rename(target)
    elif tree_hashes(target) != before:
        raise RuntimeError('existing archive differs from local evidence')
    write_json(target / 'storage_archive.json', {'path': str(target), 'file_hashes': before, 'at': now()})
    backup = folder.with_name(folder.name + '.local_backup')
    folder.rename(backup)
    try:
        folder.symlink_to(target, target_is_directory=True)
    except BaseException:
        backup.rename(folder)
        raise
    shutil.rmtree(backup)


def retry_reason(result, row, error):
    if result is not None:
        if cuda_oom(result):
            return 'recorded_cuda_oom'
        if result.get('evaluation_valid') is not True and 'database is locked' in (result.get('error') or ''):
            return 'recorded_sqlite_lock'
        return None  # Complete outcomes take precedence over queue-write errors.
    if row['state'] in {'pending', 'held_storage'}:
        return 'unstarted'
    if 'database is locked' in error:
        return 'recorded_sqlite_lock_without_result'
    raise ValueError(f"missing result without proven infrastructure error: job {row['id']}")


def prepare(source, root):
    plan = checked_plan(source)
    pids = read_json(source / 'validation/launch.json')['worker_pids']
    if any(alive(pid) for pid in pids):
        raise RuntimeError('source workers still alive')
    snapshot = root / 'source_queue.sqlite'
    for suffix in ('', '-journal', '-wal', '-shm'):
        part = Path(str(source / 'validation/queue.sqlite') + suffix)
        if part.exists():
            shutil.copy2(part, Path(str(snapshot) + suffix))
    with sqlite3.connect(snapshot, timeout=60) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute('SELECT * FROM jobs')]
    by_key = {(r['candidate'], r['task']): r for r in rows}
    candidates = read_json(source / 'validation/candidates.json')
    plan = {**plan, 'source_run': str(source), 'created_at': now(), 'eval_workers': 18,
        'source_plan_sha256': file_hash(source / 'plan.json'),
        'recovery_driver_sha256': file_hash(__file__),
        'recovery_policy': 'retain completed outcomes; only explicit CUDA OOM/SQLite locks retried; execute unstarted tasks',
        'active_storage': 'local; immutable evidence archived with SHA256 verification'}
    write_json(root / 'plan.json', plan)
    write_json(root / 'plan.sha256.json', {'sha256': file_hash(root / 'plan.json')})
    jobs, audit, budgets = [], {}, {}
    for name in candidates:
        src = source / 'validation' / name / 'manifest.json'
        if file_hash(src) != read_json(src.with_name('manifest.sha256.json'))['sha256']:
            raise ValueError('source manifest changed')
        manifest = read_json(src)
        records, audit[name], budgets[name] = [], [], {}
        for index, original in enumerate(manifest['candidates']):
            folder = Path(original['path'])
            row = by_key[name, original['task_instance_id']]
            result_path = folder / 'episode_result.json'
            result = read_json(result_path) if result_path.exists() else None
            if result is not None and result.get('task_instance_id') != original['task_instance_id']:
                raise ValueError('source task identity mismatch')
            error_path = source / 'validation/errors' / f"{row['id']}.json"
            error = read_json(error_path)['error'] if error_path.exists() else ''
            reason = retry_reason(result, row, error)
            record = dict(original)
            entry = {'task': original['task_instance_id'], 'original_path': str(folder),
                     'queue_state': row['state'], 'action': reason or 'reuse', 'queue_error': error}
            if result is not None:
                entry['result_sha256'] = file_hash(result_path)
                entry['seal_sha256'] = file_hash(folder / 'evidence_seal.json')
            if reason:
                output = root / 'validation' / name / 'jobs' / f't{index:05d}'
                record.update(path=str(output / 'episodes/t00000/a0'), recovery_action=reason,
                              original_attempt=str(folder))
                jobs.append((name, original['task_instance_id'], str(output)))
            else:
                record.update(reused_original_attempt=str(folder))
            records.append(record)
            audit[name].append(entry)
            # Copy only closed original budget DBs; never open network SQLite here.
            budget = Path(row['output']) / 'budget.sqlite'
            if budget.exists():
                cache = root / 'original_budgets' / name / f't{index:05d}' / 'budget.sqlite'
                cache.parent.mkdir(parents=True, exist_ok=True)
                for suffix in ('', '-journal', '-wal', '-shm'):
                    part = Path(str(budget) + suffix)
                    if part.exists():
                        shutil.copy2(part, Path(str(cache) + suffix))
                for key, value in Budget(cache, 0, 0).summary().items():
                    budgets[name][key] = budgets[name].get(key, 0) + value
        manifest.update(candidates=records, source_run=str(source), created_at=now(),
                        source_manifest_sha256=file_hash(src), recovery_policy=plan['recovery_policy'])
        dest = root / 'validation' / name
        write_json(dest / 'manifest.json', manifest)
        write_json(dest / 'manifest.sha256.json', {'sha256': file_hash(dest / 'manifest.json')})
    jobs.sort(key=lambda j: (Path(j[2]).name, j[0]))
    write_json(root / 'validation/candidates.json', candidates)
    write_json(root / 'recovery_audit.json', audit)
    write_json(root / 'original_budgets.json', budgets)
    init_queue(root / 'validation/queue.sqlite', jobs)
    return plan, candidates


def run_stage(root, archive, plan, split, tracker, step):
    stage = root / split
    env = dict(os.environ, SWANLAB_MODE='disabled', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        PYTORCH_ALLOC_CONF='expandable_segments:True', TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    processes, logs = [], []
    archived = set()
    try:
        for index in range(plan['eval_workers']):
            log = (stage / f'worker_{index:02d}.log').open('xb')
            logs.append(log)
            processes.append(subprocess.Popen([sys.executable, '-m', 'harness_awm.training.selection', 'worker',
                '--output', str(root), '--split', split, '--index', str(index)], cwd=ROOT,
                env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT))
            time.sleep(1)
        write_json(stage / 'launch.json', {'worker_pids': [p.pid for p in processes], 'started_at': now()})
        while True:
            exits = [p.poll() for p in processes]
            with sqlite3.connect(stage / 'queue.sqlite') as db:
                counts = dict(db.execute('SELECT state,count(*) FROM jobs GROUP BY state'))
                done = db.execute("SELECT output FROM jobs WHERE state='done'").fetchall()
            for (output,) in done:
                if output not in archived:
                    folder = Path(output) / 'episodes/t00000/a0'
                    archive_episode(folder, archive / folder.relative_to(root))
                    archived.add(output)
            reports = aggregate(stage, plan)
            if split == 'validation':
                for name, budget in read_json(root / 'original_budgets.json').items():
                    for key, value in budget.items():
                        reports[name]['budget'][key] = reports[name]['budget'].get(key, 0) + value
                    write_json(stage / name / 'metrics.json', reports[name])
            free = shutil.disk_usage(root).free
            status = {'stage': split, 'updated_at': now(), 'queue': counts,
                'live_workers': sum(x is None for x in exits), 'worker_exit_codes': exits,
                'local_free_bytes': free, 'archived_tasks': len(archived),
                'candidates': {k: compact(v) for k, v in reports.items()}}
            write_json(root / 'status.json', status)
            tracker.log({split: status['candidates'], 'queue': counts,
                         'judge_budget': {k: v['budget'] for k, v in reports.items()},
                         'live_workers': status['live_workers'], 'local_free_bytes': free}, step=step)
            step += 1
            if free < 1024**3:
                # Let owned episodes finish before stopping for disk pressure.
                with sqlite3.connect(stage / 'queue.sqlite') as db:
                    db.execute("UPDATE jobs SET state='held_disk' WHERE state='pending'")
            if not status['live_workers']:
                if any(x != 0 for x in exits) or any(v for k, v in counts.items() if k != 'done'):
                    raise RuntimeError(f'{split} incomplete; inspect preserved evidence')
                return reports, step
            time.sleep(20)
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


def execute(source, root, archive):
    root.mkdir(parents=True, exist_ok=True)
    archive.mkdir(parents=True, exist_ok=True)
    with (root / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'launch.json').exists():
            raise ValueError('already launched')
        write_json(root / 'launch.json', {'pid': os.getpid(), 'started_at': now()})
        tracker, failed = None, True
        try:
            write_json(root / 'status.json', {'stage': 'preparing_local_continuation', 'updated_at': now()})
            plan, candidates = prepare(source, root)
            tracker = SwanLabTracker(root, mode='online', config={'stage': 'sft-local-storage-recovery',
                'stop_policy': plan['stop_policy'], 'eval_workers': 18, 'recovery_policy': plan['recovery_policy']})
            write_json(root / 'tracking.json', {'url': tracker.url})
            reports, step = run_stage(root, archive, plan, 'validation', tracker, 0)
            best = choose_best({k: reports[k] for k in plan['sft_candidates']}, plan['minimum_coverage'])
            selected = {'selected': best, 'checkpoint': candidates[best], 'selected_at': now(),
                'validation': {k: compact(v) for k, v in reports.items()}, 'test_used_for_selection': False,
                'selection_rule': plan['selection_rule'], 'recovery_policy': plan['recovery_policy']}
            write_json(root / 'selection.json', selected)
            write_json(root / 'selection.sha256.json', {'sha256': file_hash(root / 'selection.json')})
            prepare_evaluation(root, plan, {'base': candidates['base'], best: candidates[best]}, 'test')
            test, step = run_stage(root, archive, plan, 'test', tracker, step)
            summary = {'selected': best, 'checkpoint': candidates[best], 'stop_policy': plan['stop_policy'],
                'baseline': {'validation': compact(reports['base']), 'test': compact(test['base'])},
                'validation': compact(reports[best]), 'test': compact(test[best]),
                'test_used_for_selection': False, 'completed_at': now(), 'swanlab_url': tracker.url}
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


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    execute(args.source.resolve(), args.output.resolve(), args.archive.resolve())
