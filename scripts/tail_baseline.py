"""Finish a frozen baseline using a transactional shared queue of unstarted tasks."""
import argparse
import asyncio
import fcntl
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import parallel_baseline as base
from harness_awm.contracts import RunConfig
from harness_awm.io import file_hash, now, read_json, write_json
from harness_awm.backend import Budget
from harness_awm.training.evaluate import evaluate
from harness_awm.training.local import load_checkpoint
from harness_awm.training.tasks import load_tasks


def init_queue(path, jobs):
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE jobs(task_id TEXT PRIMARY KEY, split TEXT, output TEXT UNIQUE, state TEXT, owner INTEGER, updated TEXT)")
        db.executemany("INSERT INTO jobs VALUES (?,?,?,'pending',NULL,?)", [(j['task_id'], j['split'], j['output'], now()) for j in jobs])


def claim(path, owner):
    with sqlite3.connect(path, timeout=30, isolation_level=None) as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT task_id,split,output FROM jobs WHERE state='pending' ORDER BY rowid LIMIT 1").fetchone()
        if row:
            db.execute("UPDATE jobs SET state='claimed',owner=?,updated=? WHERE task_id=? AND state='pending'", (owner, now(), row[0]))
        db.commit()
    return dict(zip(('task_id', 'split', 'output'), row)) if row else None


def finish(path, ident, owner, state):
    if state not in {'done', 'error'}:
        raise ValueError('invalid completion state')
    with sqlite3.connect(path, timeout=30) as db:
        result = db.execute("UPDATE jobs SET state=?,updated=? WHERE task_id=? AND state='claimed' AND owner=?", (state, now(), ident, owner))
        if result.rowcount != 1:
            raise ValueError('task is not owned by this worker')


def prepare(source, output, workers):
    source, output = Path(source).resolve(), Path(output).resolve()
    original = read_json(source / 'plan.json')
    kept = read_json(source / 'parallel_handoff.json')['kept']
    count = sum(len(load_tasks(original['tasks_path'], split)[0]) for split in ('validation', 'test')) - len(kept)
    workers = min(workers, max(1, count))
    plan = base.prepare(source, output, workers)
    jobs = []
    for split in ('validation', 'test'):
        manifest = read_json(output / split / 'manifest.json')
        for index, candidate in enumerate(manifest['candidates']):
            if candidate['reused_prior_result']:
                continue
            run = output / 'jobs' / split / f't{index:05d}'
            candidate['path'] = str(run / 'episodes/t00000/a0')
            jobs.append({'task_id': candidate['task_instance_id'], 'split': split, 'output': str(run)})
        manifest['scheduling'] = 'shared_transactional_queue'
        write_json(output / split / 'manifest.json', manifest)
        write_json(output / split / 'manifest.sha256.json', {'sha256': file_hash(output / split / 'manifest.json')})
    init_queue(output / 'queue.sqlite', jobs)
    plan.update(scheduling='shared_transactional_queue', queue_driver_sha256=file_hash(__file__),
                jobs=jobs, cpu_threads_per_worker=1)
    write_json(output / 'plan.json', plan)
    write_json(output / 'plan.sha256.json', {'sha256': file_hash(output / 'plan.json')})
    (output / 'queue_driver_frozen.py').write_bytes(Path(__file__).read_bytes())
    write_json(output / 'status.json', {'stage': 'prepared', 'workers': workers, 'queue_pending': len(jobs), 'reused_results': len(kept)})
    return {'workers': workers, 'pending': len(jobs), 'kept': len(kept)}


def checked_plan(output):
    plan = base.validated_plan(output)
    if file_hash(__file__) != plan['queue_driver_sha256']:
        raise ValueError('queue driver changed')
    return plan


def worker(output, index):
    import torch
    output = Path(output)
    plan = checked_plan(output)
    if not 0 <= index < plan['workers']:
        raise ValueError('unknown worker')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = RunConfig.model_validate(plan['config'])
    tasks = {}
    for split in ('validation', 'test'):
        rows, manifest, sources = load_tasks(plan['tasks_path'], split)
        if manifest != plan['task_manifest'] or sources != config.judge_inputs_manifest:
            raise ValueError('runtime task binding changed')
        tasks.update({t.task_instance_id: t for t in rows})
    model, tokenizer = load_checkpoint(plan['checkpoint']['path'], device=plan['device'], dtype=plan['dtype'])
    while job := claim(output / 'queue.sqlite', index):
        try:
            report = asyncio.run(evaluate(model, tokenizer, [tasks[job['task_id']]], config, job['output'],
                split=job['split'], checkpoint=plan['checkpoint'], task_manifest=plan['task_manifest'], sampling=plan['sampling']))
            finish(output / 'queue.sqlite', job['task_id'], index,
                   'done' if report['evaluation']['result_records'] == 1 else 'error')
            print(json.dumps({'task': job['task_id'], 'worker': index, 'scored': report['evaluation']['scored']}), flush=True)
        except Exception as exc:
            # Never silently retry an answer or choose a better sample.
            write_json(output / 'queue_errors' / job['split'] / (Path(job['output']).name + '.json'),
                       {'task': job['task_id'], 'worker': index, 'error': f'{type(exc).__name__}: {exc}'})
            finish(output / 'queue.sqlite', job['task_id'], index, 'error')
        finally:
            torch.cuda.empty_cache()


def supervise(output):
    output = Path(output).resolve()
    plan = checked_plan(output)
    with (output / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / 'launch.json').exists():
            raise ValueError('queue workers already launched')
        env = dict(os.environ, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', PYTHONUNBUFFERED='1',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', SWANLAB_MODE='offline')
        processes, logs = [], []
        started = time.monotonic()
        started_at = now()
        try:
            for index in range(plan['workers']):
                log = (output / f'worker_{index:02d}.log').open('xb')
                logs.append(log)
                proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker', '--output', str(output), '--index', str(index)],
                    cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                processes.append(proc)
                write_json(output / 'launch.json', {'supervisor_pid': os.getpid(), 'started_at': started_at, 'worker_pids': [p.pid for p in processes]})
                time.sleep(1)
            while True:
                exits = [p.poll() for p in processes]
                summary = base.aggregate(output, plan, time.monotonic() - started)
                # Include this queue's per-task ledgers in addition to all prior runs.
                for split in ('validation', 'test'):
                    report = read_json(output / split / 'metrics.json')
                    for ledger in output.glob(f'jobs/{split}/*/budget.sqlite'):
                        for key, value in Budget(ledger, 0, 0).summary().items():
                            report['budget'][key] += value
                    write_json(output / split / 'metrics.json', report)
                    from harness_awm.evaluation_metrics import write_reports
                    write_reports(output / split, report)
                with sqlite3.connect(output / 'queue.sqlite') as db:
                    counts = dict(db.execute('SELECT state,count(*) FROM jobs GROUP BY state').fetchall())
                live = sum(code is None for code in exits)
                failed = [i for i, code in enumerate(exits) if code not in (None, 0)]
                incomplete = failed or counts.get('error', 0) or counts.get('pending', 0) or counts.get('claimed', 0)
                write_json(output / 'status.json', {'stage': 'running' if live else 'incomplete' if incomplete else 'finished',
                    'updated_at': now(), 'scheduling': plan['scheduling'], 'workers': plan['workers'],
                    'live_workers': live, 'failed_workers': failed, 'worker_exit_codes': exits,
                    'queue_pending': counts.get('pending', 0), 'queue_running': counts.get('claimed', 0),
                    'queue_finished': counts.get('done', 0), 'queue_errors': counts.get('error', 0),
                    'reused_results': len(plan['kept']), 'splits': summary})
                if not live:
                    return
                time.sleep(10)
        finally:
            for log in logs:
                log.close()


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'run', 'worker'))
    p.add_argument('--source', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=20)
    p.add_argument('--index', type=int)
    a = p.parse_args()
    if a.action == 'prepare':
        print(json.dumps(prepare(a.source, a.output, a.workers)))
    elif a.action == 'worker':
        worker(a.output, a.index)
    else:
        supervise(a.output)


if __name__ == '__main__':
    main()
