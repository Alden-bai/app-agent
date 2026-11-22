"""Frozen SFT sweep, shared-queue validation, then one selected test evaluation.

Run with the same Python environment as the baseline. Training alone receives
the optional local FLA dependency path; evaluation preserves baseline kernels.
"""
import argparse
import asyncio
import fcntl
import gc
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

from harness_awm.contracts import ROOT, RunConfig
from harness_awm.io import digest, file_hash, now, read_json, write_json
from harness_awm.tracking import SwanLabTracker
from harness_awm.training.tasks import load_tasks


def sources():
    return {str(p.relative_to(ROOT)): file_hash(p)
            for p in sorted((ROOT / 'harness_awm').rglob('*.py'))}


def compact(report):
    return {k: v for k, v in report['evaluation'].items() if k != 'episodes'}


def checked_plan(root):
    plan = read_json(root / 'plan.json')
    if file_hash(root / 'plan.json') != read_json(root / 'plan.sha256.json')['sha256']:
        raise ValueError('selection plan changed')
    if sources() != plan['source_hashes']:
        raise ValueError('implementation changed after preparation')
    return plan


def choose_best(reports, minimum_coverage=0.90):
    """Validation only. Fixed planned denominator; never rank test outcomes."""
    if not reports or any(r.get('evaluation_split') != 'validation' for r in reports.values()):
        raise ValueError('selection requires validation reports only')
    planned = {r['evaluation']['planned'] for r in reports.values()}
    if len(planned) != 1 or next(iter(planned)) < 1:
        raise ValueError('candidate task counts differ')
    eligible = {}
    for name, report in reports.items():
        e = report['evaluation']
        if e['result_records'] != e['planned']:
            raise ValueError('incomplete candidate evaluation')
        if e['score_coverage'] >= minimum_coverage:
            eligible[name] = (e['confirmed_successes'], e['scored'], e['eval_reward']['sum'])
    if not eligible:
        raise ValueError('no candidate meets the frozen coverage threshold')
    # Stable name ordering breaks exact ties without consulting test results.
    return sorted(eligible, key=lambda name: (*(-v for v in eligible[name]), name))[0]


def init_queue(path, jobs):
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE jobs(id INTEGER PRIMARY KEY, candidate TEXT, task TEXT, output TEXT UNIQUE, state TEXT, owner INTEGER, UNIQUE(candidate,task))')
        db.executemany("INSERT INTO jobs(candidate,task,output,state) VALUES (?,?,?,'pending')", jobs)


def claim(path, owner, preferred=None):
    with sqlite3.connect(path, timeout=60, isolation_level=None) as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute("SELECT id,candidate,task,output FROM jobs WHERE state='pending' ORDER BY (candidate=?) DESC,id LIMIT 1", (preferred or '',)).fetchone()
        if row:
            db.execute("UPDATE jobs SET state='claimed',owner=? WHERE id=?", (owner, row[0]))
        db.commit()
    return dict(zip(('id', 'candidate', 'task', 'output'), row)) if row else None


def finish(path, job, owner, state):
    if state not in {'done', 'error'}:
        raise ValueError('invalid queue completion')
    with sqlite3.connect(path, timeout=60) as db:
        changed = db.execute("UPDATE jobs SET state=? WHERE id=? AND owner=? AND state='claimed'", (state, job, owner)).rowcount
        if changed != 1:
            raise ValueError('job ownership mismatch')


def prepare_evaluation(root, plan, candidates, split):
    from .evaluate import preflight
    tasks, manifest, bindings = load_tasks(plan['tasks_path'], split)
    config = RunConfig.model_validate(plan['config'])
    if manifest != plan['task_manifest'] or bindings != config.judge_inputs_manifest:
        raise ValueError('runtime task bundle changed')
    preflight(tasks, config, split)
    stage = root / split
    stage.mkdir(exist_ok=False)
    jobs = []
    for name, checkpoint in candidates.items():
        records = []
        for i, task in enumerate(tasks):
            output = stage / name / 'jobs' / f't{i:05d}'
            jobs.append((name, task.task_instance_id, str(output)))
            records.append({'rollout_id': f't{i:05d}', 'task_instance_id': task.task_instance_id,
                            'scenario': task.scenario, 'path': str(output / 'episodes/t00000/a0')})
        run_manifest = {'schema_version': 'awm-run-v1', 'stage': 'evaluate', 'created_at': now(),
            'config': plan['config'], 'tasks': [t.model_dump() for t in tasks], 'candidates': records,
            'checkpoint': checkpoint, 'task_manifest': manifest, 'evaluation_split': split,
            'sampling': plan['sampling'], 'seed': config.seed, 'training_updates': False,
            'stop_policy': plan.get('stop_policy'),
            'harness_source_hashes': plan['source_hashes']}
        write_json(stage / name / 'manifest.json', run_manifest)
        write_json(stage / name / 'manifest.sha256.json', {'sha256': file_hash(stage / name / 'manifest.json')})
    # Interleave candidates initially; a worker stays with its loaded model until
    # that model has no pending tasks, then helps another candidate.
    jobs.sort(key=lambda j: (j[2].split('/')[-1], j[0]))
    init_queue(stage / 'queue.sqlite', jobs)
    write_json(stage / 'candidates.json', candidates)


def worker(root, split, index):
    import torch
    from .evaluate import evaluate
    from .local import checkpoint_identity, load_checkpoint
    plan = checked_plan(root)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = RunConfig.model_validate(plan['config'])
    tasks, manifest, bindings = load_tasks(plan['tasks_path'], split)
    if manifest != plan['task_manifest'] or bindings != config.judge_inputs_manifest:
        raise ValueError('task bundle changed')
    by_id = {t.task_instance_id: t for t in tasks}
    stage = root / split
    candidates = read_json(stage / 'candidates.json')
    loaded = None
    model = tokenizer = None
    while job := claim(stage / 'queue.sqlite', index, loaded):
        try:
            if loaded != job['candidate']:
                model = tokenizer = None
                gc.collect()
                torch.cuda.empty_cache()
                checkpoint = candidates[job['candidate']]
                if checkpoint_identity(checkpoint['path']) != checkpoint:
                    raise ValueError('candidate weights changed')
                model, tokenizer = load_checkpoint(checkpoint['path'], device='cuda', dtype=plan['dtype'])
                loaded = job['candidate']
            report = asyncio.run(evaluate(model, tokenizer, [by_id[job['task']]], config,
                job['output'], split=split, checkpoint=candidates[loaded],
                task_manifest=manifest, sampling=plan['sampling']))
            finish(stage / 'queue.sqlite', job['id'], index,
                   'done' if report['evaluation']['result_records'] == 1 else 'error')
        except Exception as exc:
            write_json(stage / 'errors' / f"{job['id']}.json", {'job': job, 'error': f'{type(exc).__name__}: {exc}'})
            finish(stage / 'queue.sqlite', job['id'], index, 'error')
            # Infrastructure/model exceptions must not turn the remaining queue
            # into hundreds of immediate errors or resample existing attempts.
            raise
        finally:
            torch.cuda.empty_cache()


def aggregate(stage, plan):
    from harness_awm.stages import metrics
    from harness_awm.evaluation_metrics import write_reports
    from harness_awm.backend import Budget
    reports = {}
    for name in read_json(stage / 'candidates.json'):
        report = metrics(stage / name)
        budget = {}
        for path in (stage / name).glob('jobs/*/budget.sqlite'):
            for key, value in Budget(path, 0, 0).summary().items():
                budget[key] = budget.get(key, 0) + value
        report.update(evaluation_split=stage.name, isolation=plan['config']['isolation'],
                      sampling=plan['sampling'], training_updates=False, budget=budget,
                      stop_policy=plan.get('stop_policy'))
        write_json(stage / name / 'metrics.json', report)
        write_reports(stage / name, report)
        reports[name] = report
    return reports


def run_evaluation(root, plan, candidates, split, tracker, step):
    prepare_evaluation(root, plan, candidates, split)
    stage = root / split
    env = dict(os.environ, SWANLAB_MODE='disabled', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTORCH_ALLOC_CONF='expandable_segments:True',
               TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    processes, logs = [], []
    try:
        for index in range(plan['eval_workers']):
            log = (stage / f'worker_{index:02d}.log').open('xb')
            logs.append(log)
            processes.append(subprocess.Popen([sys.executable, '-m', 'harness_awm.training.selection', 'worker', '--output', str(root),
                '--split', split, '--index', str(index)], cwd=ROOT, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT))
            time.sleep(1)
        write_json(stage / 'launch.json', {'worker_pids': [p.pid for p in processes], 'started_at': now()})
        while True:
            exits = [p.poll() for p in processes]
            reports = aggregate(stage, plan)
            with sqlite3.connect(stage / 'queue.sqlite') as db:
                counts = dict(db.execute('SELECT state,count(*) FROM jobs GROUP BY state'))
            status = {'stage': split, 'updated_at': now(), 'queue': counts,
                      'live_workers': sum(x is None for x in exits), 'worker_exit_codes': exits,
                      'candidates': {k: compact(r) for k, r in reports.items()}}
            write_json(root / 'status.json', status)
            tracker.log({split: {k: r['evaluation'] for k, r in reports.items()},
                         'judge_budget': {k: r['budget'] for k, r in reports.items()},
                         'queue': counts, 'live_workers': status['live_workers']}, step=step)
            step += 1
            if not status['live_workers']:
                if any(x != 0 for x in exits) or any(value for key, value in counts.items() if key != 'done'):
                    raise RuntimeError(f'{split} workers incomplete; original attempts preserved, no automatic retry')
                return reports, step
            time.sleep(15)
    finally:
        # Do not leave orphaned model processes after supervisor failure.
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


def prepare(root, baseline, batch_size, accumulation, workers, no_gradient_checkpointing=False, microbatch_tokens=0):
    from .baseline import isolation_probe
    from .data import sha, verify_split
    baseline_plan = read_json(baseline / 'plan.json')
    if file_hash(baseline / 'plan.json') != read_json(baseline / 'plan.sha256.json')['sha256']:
        raise ValueError('baseline plan changed')
    if (root / 'plan.json').exists():
        raise ValueError('selection plan already exists')
    split = ROOT / 'data/training_split_20261106_v1'
    verify_split(split)
    data = ROOT / 'data/training_qwen35_2b_v1'
    if sha(data / 'train.jsonl') != read_json(data / 'report.json')['train_sha256']:
        raise ValueError('SFT data changed')
    if sha(split / 'manifest.json') != read_json(data / 'report.json')['split_manifest_sha256']:
        raise ValueError('SFT data split binding changed')
    root.mkdir(parents=True, exist_ok=True)
    config = RunConfig.model_validate(baseline_plan['config'])
    if not isolation_probe(config, root / 'isolation_probe')['passed']:
        raise RuntimeError('evaluation isolation probe failed')
    plan = {k: baseline_plan[k] for k in ('config', 'tasks_path', 'task_manifest', 'sampling', 'dtype', 'checkpoint')}
    plan.update(schema_version='awm-sft-selection-v1', created_at=now(), baseline=str(baseline),
        data=str(data), train_sha256=sha(data / 'train.jsonl'), source_hashes=sources(),
        trials=[{'name': 'lr1e-5', 'learning_rate': 1e-5}, {'name': 'lr2e-5', 'learning_rate': 2e-5}],
        epochs=2, batch_size=batch_size, gradient_accumulation=accumulation,
        microbatch_tokens=microbatch_tokens,
        pytorch_alloc_conf='expandable_segments:True',
        no_gradient_checkpointing=no_gradient_checkpointing, eval_workers=workers,
        minimum_coverage=0.90,
        selection_rule='validation only: confirmed_successes/planned, then coverage, then reward sum; stable name tie break',
        kernel_deps=str(ROOT / 'runs/sft_kernel_deps'))
    write_json(root / 'plan.json', plan)
    write_json(root / 'plan.sha256.json', {'sha256': file_hash(root / 'plan.json')})
    write_json(root / 'status.json', {'stage': 'prepared', 'updated_at': now()})


def execute(root):
    from .local import checkpoint_identity
    plan = checked_plan(root)
    with (root / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'launch.json').exists():
            raise ValueError('already launched; inspect artifacts before recovery')
        write_json(root / 'launch.json', {'pid': os.getpid(), 'started_at': now()})
        tracker = SwanLabTracker(root, mode='online', config={
            'stage': 'sft-selection', 'epochs': plan['epochs'], 'trials': plan['trials'],
            'selection_rule': plan['selection_rule'], 'minimum_coverage': plan['minimum_coverage']})
        write_json(root / 'tracking.json', {'url': tracker.url})
        failed = True
        try:
            if checkpoint_identity(plan['checkpoint']['path']) != plan['checkpoint']:
                raise ValueError('base model changed')
            baseline = {s: compact(read_json(Path(plan['baseline']) / s / 'metrics.json')) for s in ('validation', 'test')}
            tracker.log({'baseline': baseline}, step=0)
            candidates = {}
            for trial in plan['trials']:
                checked_plan(root)
                out = root / 'training' / trial['name']
                command = [sys.executable, '-m', 'harness_awm.training.sft', '--model', plan['checkpoint']['path'],
                    '--data', plan['data'], '--output', str(out), '--epochs', str(plan['epochs']),
                    '--batch-size', str(plan['batch_size']), '--gradient-accumulation', str(plan['gradient_accumulation']),
                    '--microbatch-tokens', str(plan['microbatch_tokens']),
                    '--learning-rate', str(trial['learning_rate']), '--save-strategy', 'epoch',
                    '--save-total-limit', '2', '--skip-final-save']
                if plan['no_gradient_checkpointing']:
                    command.append('--no-gradient-checkpointing')
                env = dict(os.environ, PYTHONPATH=plan['kernel_deps'], SWANLAB_MODE='online',
                    PYTORCH_ALLOC_CONF='expandable_segments:True',
                    SWANLAB_EXPERIMENT_NAME=f"sft-{trial['name']}-{root.name}", OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
                    HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
                write_json(root / 'status.json', {'stage': 'training', 'trial': trial, 'output': str(out), 'updated_at': now()})
                with (root / f"train_{trial['name']}.log").open('xb') as log:
                    subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                checkpoints = sorted(out.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
                if len(checkpoints) != plan['epochs']:
                    raise ValueError('expected one checkpoint per epoch')
                for epoch, checkpoint in enumerate(checkpoints, 1):
                    state = read_json(checkpoint / 'trainer_state.json')
                    if abs(state['epoch'] - epoch) > .01:
                        raise ValueError('unexpected checkpoint epoch')
                    candidates[f"{trial['name']}-epoch{epoch}"] = checkpoint_identity(checkpoint)
            write_json(root / 'candidates.json', candidates)
            reports, step = run_evaluation(root, plan, candidates, 'validation', tracker, 1)
            best = choose_best(reports, plan['minimum_coverage'])
            selection = {'selected': best, 'checkpoint': candidates[best], 'selected_at': now(),
                'selection_rule': plan['selection_rule'], 'minimum_coverage': plan['minimum_coverage'],
                'validation': {k: compact(r) for k, r in reports.items()}, 'test_used_for_selection': False}
            write_json(root / 'selection.json', selection)
            write_json(root / 'selection.sha256.json', {'sha256': file_hash(root / 'selection.json')})
            test, step = run_evaluation(root, plan, {best: candidates[best]}, 'test', tracker, step)
            summary = {'selected': best, 'checkpoint': candidates[best], 'baseline': baseline,
                'validation': compact(reports[best]), 'test': compact(test[best]),
                'test_used_for_selection': False, 'completed_at': now(), 'swanlab_url': tracker.url}
            write_json(root / 'summary.json', summary)
            tracker.log({'selected_model': {'validation': summary['validation'], 'test': summary['test']}}, step=step)
            write_json(root / 'status.json', {'stage': 'finished', 'selected': best, 'updated_at': now()})
            failed = False
        except BaseException as exc:
            write_json(root / 'status.json', {'stage': 'failed', 'error': f'{type(exc).__name__}: {exc}', 'updated_at': now()})
            raise
        finally:
            tracker.finish(failed=failed)


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'run', 'worker'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--gradient-accumulation', type=int, default=8)
    p.add_argument('--microbatch-tokens', type=int, default=0)
    p.add_argument('--workers', type=int, default=20)
    p.add_argument('--no-gradient-checkpointing', action='store_true')
    p.add_argument('--split', choices=('validation', 'test'))
    p.add_argument('--index', type=int)
    a = p.parse_args()
    if min(a.batch_size, a.gradient_accumulation, a.workers) < 1:
        p.error('batch, accumulation and workers must be positive')
    root = a.output.resolve()
    if a.action == 'prepare':
        if not a.baseline:
            p.error('--baseline required')
        prepare(root, a.baseline.resolve(), a.batch_size, a.gradient_accumulation, a.workers, a.no_gradient_checkpointing, a.microbatch_tokens)
    elif a.action == 'worker':
        worker(root, a.split, a.index)
    else:
        execute(root)


if __name__ == '__main__':
    main()
