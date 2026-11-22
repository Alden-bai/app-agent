"""Resume the exact frozen production queue after a bounded GPU benchmark."""
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
from harness_awm.training.selection import (checked_plan, aggregate, compact, choose_best,
                                           prepare_evaluation)
from harness_awm.tracking import SwanLabTracker
from scripts.sft_eval_local_recovery import archive_episode, alive


def run_stage(root, archive, plan, split, tracker, step, resume_dir):
    stage = root / split
    env = dict(os.environ, SWANLAB_MODE='disabled', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTORCH_ALLOC_CONF='expandable_segments:True', TOKENIZERS_PARALLELISM='false',
               HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    processes, logs, archived = [], [], set()
    try:
        for index in range(plan['eval_workers']):
            logfile = resume_dir / f'{split}_worker_{index:02d}.log'
            log = logfile.open('xb'); logs.append(log)
            processes.append(subprocess.Popen([sys.executable, '-m', 'harness_awm.training.selection',
                'worker', '--output', str(root), '--split', split, '--index', str(index)],
                cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT))
            time.sleep(1)
        write_json(stage / 'launch.json', {'worker_pids': [p.pid for p in processes],
            'started_at': now(), 'resume_record': str(resume_dir)})
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
                'candidates': {k: compact(v) for k, v in reports.items()},
                'resume_record': str(resume_dir)}
            write_json(root / 'status.json', status)
            tracker.log({split: status['candidates'], 'queue': counts,
                'judge_budget': {k: v['budget'] for k, v in reports.items()},
                'live_workers': status['live_workers'], 'local_free_bytes': free}, step=step)
            step += 1
            if free < 1024**3:
                with sqlite3.connect(stage / 'queue.sqlite') as db:
                    db.execute("UPDATE jobs SET state='held_disk' WHERE state='pending'")
            if not status['live_workers']:
                # Archiving/aggregation may outlive the final worker commit.
                with sqlite3.connect(stage / 'queue.sqlite') as db:
                    counts = dict(db.execute('SELECT state,count(*) FROM jobs GROUP BY state'))
                if any(x != 0 for x in exits) or any(v for k, v in counts.items() if k != 'done'):
                    raise RuntimeError(f'{split} incomplete; original evidence preserved')
                return reports, step
            time.sleep(20)
    finally:
        for p in processes:
            if p.poll() is None: p.terminate()
        for p in processes:
            try: p.wait(timeout=10)
            except subprocess.TimeoutExpired: p.kill()
        for log in logs: log.close()


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--benchmark', type=Path, required=True)
    a = parser.parse_args(); root = a.root.resolve()
    audit = read_json(a.benchmark / 'production_pause.json')
    if any(alive(pid) for pid in [audit['supervisor_pid'], *audit['worker_pids']]):
        raise RuntimeError('production processes still alive; refusing duplicate workers')
    plan = checked_plan(root)
    resume_dir = root / ('resume_' + str(int(time.time())))
    resume_dir.mkdir(exist_ok=False)
    with (root / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with sqlite3.connect(root / 'validation/queue.sqlite', timeout=60) as db:
            if db.execute("SELECT count(*) FROM jobs WHERE state='claimed'").fetchone()[0]:
                raise RuntimeError('unfinished production claims')
            db.execute("UPDATE jobs SET state='pending' WHERE state='held_benchmark'")
        write_json(resume_dir / 'audit.json', {'at': now(), 'benchmark': str(a.benchmark),
            'original_launch': audit, 'script_sha256': file_hash(__file__),
            'production_plan_sha256': file_hash(root / 'plan.json'), 'completed_results_reused': True})
        write_json(root / 'launch.json', {'pid': os.getpid(), 'started_at': now(), 'resume_record': str(resume_dir)})
        tracker = SwanLabTracker(resume_dir, mode='online', config={
            'stage': 'sft-resume-after-throughput-benchmark', 'eval_workers': 18,
            'production_inference_unchanged': True})
        write_json(root / 'tracking.json', {'url': tracker.url, 'previous': audit.get('tracking')})
        failed = True
        try:
            reports, step = run_stage(root, a.archive, plan, 'validation', tracker, 0, resume_dir)
            gate = root / 'test/JUDGE_RECOVERY_PENDING.json'
            if gate.exists():
                write_json(root / 'status.json', {'stage': 'awaiting_judge_recovery',
                    'updated_at': now(), 'handoff': read_json(gate),
                    'validation_complete': True, 'selection_not_performed': True})
                tracker.log({'validation_complete': 1, 'awaiting_judge_recovery': 1}, step=step)
                failed = False
                return
            candidates = read_json(root / 'validation/candidates.json')
            best = choose_best({k: reports[k] for k in plan['sft_candidates']}, plan['minimum_coverage'])
            selected = {'selected': best, 'checkpoint': candidates[best], 'selected_at': now(),
                'validation': {k: compact(v) for k, v in reports.items()}, 'test_used_for_selection': False,
                'selection_rule': plan['selection_rule'], 'recovery_policy': plan['recovery_policy']}
            write_json(root / 'selection.json', selected)
            write_json(root / 'selection.sha256.json', {'sha256': file_hash(root / 'selection.json')})
            prepare_evaluation(root, plan, {'base': candidates['base'], best: candidates[best]}, 'test')
            test, step = run_stage(root, a.archive, plan, 'test', tracker, step, resume_dir)
            summary = {'selected': best, 'checkpoint': candidates[best], 'stop_policy': plan['stop_policy'],
                'baseline': {'validation': compact(reports['base']), 'test': compact(test['base'])},
                'validation': compact(reports[best]), 'test': compact(test[best]),
                'test_used_for_selection': False, 'completed_at': now(), 'swanlab_url': tracker.url}
            write_json(root / 'summary.json', summary); tracker.log({'summary': summary}, step=step)
            write_json(root / 'status.json', {'stage': 'finished', 'selected': best, 'updated_at': now()})
            failed = False
        except BaseException as exc:
            write_json(root / 'status.json', {'stage': 'failed', 'error': f'{type(exc).__name__}: {exc}', 'updated_at': now()})
            raise
        finally: tracker.finish(failed=failed)


if __name__ == '__main__': main()
