"""Exclusive-GPU paired evaluation: 18 model copies vs five shared batch servers.

All benchmark outcomes are isolated from production selection. The controller
drains production at episode boundaries and always launches its exact-queue
continuation after the experiment. No training or test-set selection occurs.
"""
import argparse
import asyncio
from collections import Counter
import gc
import inspect
import json
import os
from pathlib import Path
import random
import signal
import sqlite3
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness_awm.io import read_json, write_json, file_hash, digest, now
from harness_awm.training.actor import TrainingActorBackend, stop_token_ids
from scripts.benchmark_shared_batch import batch_decode


def append(path, row):
    with Path(path).open('a') as f: f.write(json.dumps(row, ensure_ascii=False) + '\n')


def pick_tasks(tasks, count, seed):
    """Stratify by instruction length only; never inspect task outcomes."""
    ordered = sorted(tasks, key=lambda t: (len(t.instruction), t.task_instance_id))
    rng = random.Random(seed)
    return [rng.choice(ordered[i * len(tasks) // count:(i + 1) * len(tasks) // count])
            for i in range(count)]


def take_batch(pending, size, padded_limit):
    live = [r for r in pending if not r['future'].done()]
    if not live: return [], []
    # Oldest request anchors each batch; bound padding and keep output budgets equal.
    first = live[0]
    group = [first]
    for row in sorted(live[1:], key=lambda r: abs(len(r['prompt']) - len(first['prompt']))):
        width = max(len(x['prompt']) for x in [*group, row])
        if (len(group) < size and row['max_tokens'] == first['max_tokens']
                and width * (len(group) + 1) <= padded_limit): group.append(row)
    chosen = {id(r) for r in group}
    return group, [r for r in live if id(r) not in chosen]


async def serve(a):
    import torch
    from aiohttp import web
    from harness_awm.training.local import load_checkpoint
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    for name in ('torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule',
                 'causal_conv1d_fn', 'causal_conv1d_update'):
        if inspect.getclosurevars(getattr(q, name)).nonlocals.get('is_new_implementation'):
            raise RuntimeError('Reference kernels required for this batching-only comparison')
    manifest = read_json(a.run / 'experiment.json')
    checkpoint = manifest['candidates'][a.candidate]
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(12 * 1024**3 / total)
    model, tokenizer = load_checkpoint(checkpoint['path'], dtype=manifest['plan']['dtype'])
    prompt = tokenizer.encode('Hello', add_special_tokens=False)
    batch_decode(model, tokenizer, [prompt, prompt], max_tokens=2)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    pending = []; wake = asyncio.Event(); loop = asyncio.get_running_loop()
    logfile = a.stage / f'service_{a.candidate}.jsonl'

    def deliver(row, tokens=None, error=None):
        if row['future'].done(): return
        if error is not None: row['future'].set_exception(RuntimeError(error))
        else: row['future'].set_result({'completion_ids': tokens,
            'policy_version': checkpoint['sha256'], 'eos_token_ids': stop_token_ids(model, tokenizer)})

    def decode_group(group):
        try:
            batch_decode(model, tokenizer, [r['prompt'] for r in group],
                max_tokens=group[0]['max_tokens'],
                on_complete=lambda i, ids: loop.call_soon_threadsafe(deliver, group[i], ids),
                cancelled=lambda i: group[i]['future'].cancelled())
        except torch.OutOfMemoryError:
            gc.collect(); torch.cuda.empty_cache()
            append(logfile, {'event': 'oom_split', 'batch': len(group), 'at': now()})
            remaining = [r for r in group if not r['future'].done()]
            if len(group) == 1:
                for row in remaining: loop.call_soon_threadsafe(deliver, row, None, 'CUDA OOM at batch=1')
            elif remaining:
                middle = max(1, len(remaining) // 2)
                decode_group(remaining[:middle])
                if remaining[middle:]: decode_group(remaining[middle:])
        except Exception as exc:
            for row in group: loop.call_soon_threadsafe(deliver, row, None, f'{type(exc).__name__}: {exc}')

    async def schedule():
        nonlocal pending
        while True:
            if not pending: wake.clear(); await wake.wait()
            await asyncio.sleep(.015)
            group, pending = take_batch(pending, 8, 16384)
            if not group: continue
            start = time.perf_counter()
            await asyncio.to_thread(decode_group, group)
            append(logfile, {'event': 'batch', 'at': now(), 'batch': len(group),
                'prompt_tokens': [len(r['prompt']) for r in group],
                'seconds': time.perf_counter() - start,
                'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 1024**3,
                'peak_reserved_GiB': torch.cuda.max_memory_reserved() / 1024**3})

    async def generate(request):
        payload = await request.json()
        if payload['policy_version'] != checkpoint['sha256']: raise web.HTTPBadRequest()
        prompt, limit = payload['prompt'], payload['max_tokens']
        if (not prompt or any(type(x) is not int or x < 0 for x in prompt)
                or type(limit) is not int or limit < 1 or len(prompt) + limit > 32768):
            raise web.HTTPBadRequest()
        future = loop.create_future()
        pending.append({'prompt': prompt, 'max_tokens': limit, 'future': future})
        wake.set()
        try: return web.json_response(await future)
        finally:
            if not future.done(): future.cancel()

    app = web.Application(client_max_size=4 * 1024**2)
    app.router.add_post('/generate', generate)
    runner = web.AppRunner(app, handler_cancellation=True)
    await runner.setup(); site = web.TCPSite(runner, '127.0.0.1', a.port); await site.start()
    scheduler = asyncio.create_task(schedule())
    write_json(a.stage / f'service_{a.candidate}.ready.json', {'pid': os.getpid(), 'at': now()})
    try: await scheduler
    finally: await runner.cleanup()


class RemoteModel:
    """Metadata only; inference is performed by a matching CUDA service."""
    def __init__(self, checkpoint, port):
        import torch
        self.checkpoint, self.port = checkpoint, port
        config = read_json(Path(checkpoint['path']) / 'config.json')
        generation = Path(checkpoint['path']) / 'generation_config.json'
        self.generation_config = SimpleNamespace(eos_token_id=(read_json(generation) if generation.exists()
            else config.get('text_config', config)).get('eos_token_id'))
        self.parameter = SimpleNamespace(dtype=torch.bfloat16, device=torch.device('cuda'))
    def requires_grad_(self, _): return self
    def eval(self): return self
    def parameters(self): return iter([self.parameter])


class SharedActor(TrainingActorBackend):
    def __init__(self, model, tokenizer, *, policy_version, max_output_tokens,
                 context_limit, greedy, record_policy):
        if not greedy or record_policy: raise ValueError('shared probe is greedy evaluation only')
        self.model, self.tokenizer = model, tokenizer
        self.policy_version, self.max_output_tokens = policy_version, max_output_tokens
        self.context_limit, self.phase = context_limit, 'model_request'

    async def generate(self, messages, folder, turn, *, output_tokens=None, **options):
        from aiohttp import ClientSession, ClientTimeout
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, enable_thinking=False)
        if hasattr(prompt, 'keys'): prompt = prompt['input_ids']
        limit = self.max_output_tokens if output_tokens is None else output_tokens
        if not prompt or limit < 1 or len(prompt) + limit > self.context_limit:
            raise ValueError('training_context_limit_exceeded')
        async with ClientSession(timeout=ClientTimeout(total=None)) as session:
            async with session.post(f'http://127.0.0.1:{self.model.port}/generate', json={
                    'policy_version': self.policy_version, 'prompt': prompt, 'max_tokens': limit}) as response:
                response.raise_for_status(); result = await response.json()
        if result['policy_version'] != self.policy_version: raise ValueError('wrong checkpoint response')
        ids = result['completion_ids']
        if not ids: raise ValueError('empty completion')
        return {'content': self.tokenizer.decode(ids, skip_special_tokens=True),
            'finish_reason': 'stop' if ids[-1] in result['eos_token_ids'] else 'length',
            'test_only': False, 'model_version': self.policy_version,
            'usage': {'prompt_tokens': len(prompt), 'completion_tokens': len(ids),
                      'total_tokens': len(prompt) + len(ids)}}


def worker(a):
    import torch
    from transformers import AutoTokenizer
    import harness_awm.training.evaluate as ev
    from harness_awm.training.local import load_checkpoint
    from harness_awm.training.tasks import load_tasks
    from harness_awm.training.selection import claim, finish
    from harness_awm.contracts import RunConfig
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    manifest = read_json(a.run / 'experiment.json'); plan = manifest['plan']
    tasks, task_manifest, _ = load_tasks(plan['tasks_path'], 'validation')
    by_id = {t.task_instance_id: t for t in tasks}
    config = RunConfig.model_validate(plan['config'])
    loaded = a.candidate; checkpoint = manifest['candidates'][loaded]
    def load(name):
        checkpoint = manifest['candidates'][name]
        if a.mode == 'shared':
            ev.TrainingActorBackend = SharedActor
            tokenizer = AutoTokenizer.from_pretrained(checkpoint['path'], local_files_only=True)
            return RemoteModel(checkpoint, manifest['ports'][name]), tokenizer
        return load_checkpoint(checkpoint['path'], dtype=plan['dtype'])
    model, tokenizer = load(loaded)
    if a.mode == 'baseline':
        actor = TrainingActorBackend(model, tokenizer, policy_version=checkpoint['sha256'],
            max_output_tokens=2, greedy=True, record_policy=False)
        asyncio.run(actor.generate([{'role': 'user', 'content': 'Hello'}], a.stage / 'warmup', 1))
        torch.cuda.synchronize()
    write_json(a.stage / f'worker_{a.index}.ready.json', {'pid': os.getpid(), 'at': now()})
    while not (a.stage / 'go.json').exists(): time.sleep(.1)
    while job := claim(a.stage / 'queue.sqlite', a.index, loaded):
        start = time.time()
        try:
            if loaded != job['candidate']:
                model = tokenizer = None; gc.collect()
                if a.mode == 'baseline': torch.cuda.empty_cache()
                loaded = job['candidate']; model, tokenizer = load(loaded)
            report = asyncio.run(ev.evaluate(model, tokenizer, [by_id[job['task']]], config,
                job['output'], split='validation', checkpoint=manifest['candidates'][loaded],
                task_manifest=task_manifest, sampling=plan['sampling']))
            state = 'done' if report['evaluation']['result_records'] == 1 else 'error'
            finish(a.stage / 'queue.sqlite', job['id'], a.index, state)
            write_json(a.stage / 'timings' / f"{job['id']}.json", {'job': job, 'start': start,
                'end': time.time(), 'state': state, 'evaluation': report['evaluation']})
        except Exception as exc:
            write_json(a.stage / 'errors' / f"{job['id']}.json", {'error': f'{type(exc).__name__}: {exc}', 'job': job})
            finish(a.stage / 'queue.sqlite', job['id'], a.index, 'error')
            raise
        finally:
            if a.mode == 'baseline': torch.cuda.empty_cache()


def env():
    return dict(os.environ, SWANLAB_MODE='disabled', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        PYTORCH_ALLOC_CONF='expandable_segments:True', TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')


def start_child(args, log):
    return subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), *args],
        cwd=ROOT, env=env(), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)


def completed_queue_counts(queue, exits):
    """Read after workers exit: an earlier snapshot may predate their last commit."""
    if not all(x is not None for x in exits): return None
    with sqlite3.connect(queue) as db:
        counts = dict(db.execute('SELECT state,count(*) FROM jobs GROUP BY state'))
    if any(x != 0 for x in exits) or any(v for k, v in counts.items() if k != 'done'):
        raise RuntimeError(f'benchmark workers incomplete: {counts}, {exits}')
    return counts


def run_mode(a, mode, manifest):
    from harness_awm.training.selection import init_queue
    from scripts.sft_eval_local_recovery import archive_episode
    stage = a.run / mode; stage.mkdir()
    jobs = [(name, task, str(stage / name / 'jobs' / f't{i:03d}'))
            for i, task in enumerate(manifest['task_ids']) for name in manifest['candidates']]
    init_queue(stage / 'queue.sqlite', jobs)
    services, workers, logs = [], [], []
    names = list(manifest['candidates']); count = 18 if mode == 'baseline' else 32
    startup = time.time(); archived = set(); telemetry = []
    try:
        if mode == 'shared':
            for name in names:
                log = (stage / f'service_{name}.log').open('x'); logs.append(log)
                services.append(start_child(['serve', '--run', str(a.run), '--stage', str(stage),
                    '--candidate', name, '--port', str(manifest['ports'][name])], log))
        for index in range(count):
            log = (stage / f'worker_{index:02d}.log').open('x'); logs.append(log)
            workers.append(start_child(['worker', '--run', str(a.run), '--stage', str(stage),
                '--mode', mode, '--candidate', names[index % len(names)], '--index', str(index)], log))
            time.sleep(.3)
        write_json(stage / 'launch.json', {'workers': [p.pid for p in workers], 'services': [p.pid for p in services]})
        while (len(list(stage.glob('worker_*.ready.json'))) != count or
                len(list(stage.glob('service_*.ready.json'))) != len(services)):
            if any(p.poll() is not None for p in workers + services): raise RuntimeError('benchmark child failed during startup')
            if time.time() - startup > 900: raise TimeoutError('benchmark startup exceeded 15 minutes')
            write_json(a.run / 'status.json', {'stage': mode + '_loading', 'at': now(),
                'ready_workers': len(list(stage.glob('worker_*.ready.json'))),
                'ready_services': len(list(stage.glob('service_*.ready.json')))})
            time.sleep(5)
        start = time.time(); write_json(stage / 'go.json', {'epoch': start, 'at': now()})
        while True:
            with sqlite3.connect(stage / 'queue.sqlite') as db:
                counts = dict(db.execute('SELECT state,count(*) FROM jobs GROUP BY state'))
                done = db.execute("SELECT output FROM jobs WHERE state='done'").fetchall()
            for (output,) in done:
                if output not in archived:
                    folder = Path(output) / 'episodes/t00000/a0'
                    archive_episode(folder, a.archive / 'benchmark' / a.run.name / folder.relative_to(a.run))
                    archived.add(output)
            sample = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout.strip()
            telemetry.append({'at': time.time(), 'gpu': sample})
            write_json(stage / 'gpu.json', telemetry)
            write_json(a.run / 'status.json', {'stage': mode + '_evaluating', 'at': now(), 'queue': counts,
                'elapsed_seconds': time.time() - start, 'gpu': sample})
            exits = [p.poll() for p in workers]
            if completed_queue_counts(stage / 'queue.sqlite', exits) is not None:
                break
            if any(p.poll() is not None for p in services): raise RuntimeError('shared service failed')
            if time.time() - start > 2400: raise TimeoutError('benchmark mode exceeded 40 minutes')
            time.sleep(5)
        timings = [read_json(p) for p in (stage / 'timings').glob('*.json')]
        end = max(r['end'] for r in timings)
        report = {'mode': mode, 'jobs': len(jobs), 'task_workers': count,
            'model_processes': 18 if mode == 'baseline' else len(names),
            'startup_seconds': start - startup, 'wall_seconds': end - start,
            'jobs_per_minute': len(jobs) * 60 / (end - start), 'finished_at': now(),
            'confirmed_successes': sum(r['evaluation']['confirmed_successes'] for r in timings),
            'scored': sum(r['evaluation']['scored'] for r in timings),
            'peak_sampled_gpu_MiB': max(int(x['gpu'].split(',')[0]) for x in telemetry)}
        write_json(stage / 'summary.json', report)
        return report
    finally:
        for p in workers + services:
            if p.poll() is None: p.terminate()
        for p in workers + services:
            try: p.wait(timeout=15)
            except subprocess.TimeoutExpired: p.kill(); p.wait()
        for log in logs: log.close()


def pause_production(a):
    from scripts.sft_eval_local_recovery import alive
    supervisor = read_json(a.source / 'launch.json')['pid']
    workers = read_json(a.source / 'validation/launch.json')['worker_pids']
    for pid in [supervisor, *workers]:
        cmd = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        if str(a.source) not in cmd: raise RuntimeError('process identity mismatch')
    audit = {'supervisor_pid': supervisor, 'worker_pids': workers, 'at': now(),
        'tracking': read_json(a.source / 'tracking.json'), 'source': str(a.source),
        'plan_sha256': file_hash(a.source / 'plan.json')}
    write_json(a.run / 'production_pause.json', audit)
    os.kill(supervisor, signal.SIGSTOP)
    with sqlite3.connect(a.source / 'validation/queue.sqlite', timeout=60) as db:
        audit['held_job_ids'] = [r[0] for r in db.execute("SELECT id FROM jobs WHERE state='pending'")]
        db.execute("UPDATE jobs SET state='held_benchmark' WHERE state='pending'")
    write_json(a.run / 'production_pause.json', audit)
    start = time.time()
    while any(alive(pid) for pid in workers):
        write_json(a.run / 'status.json', {'stage': 'draining_production', 'at': now(),
            'remaining_workers': sum(alive(pid) for pid in workers), 'seconds': time.time() - start})
        if time.time() - start > 1800: raise TimeoutError('production drain exceeded 30 minutes')
        time.sleep(5)
    os.kill(supervisor, signal.SIGTERM); os.kill(supervisor, signal.SIGCONT)
    for _ in range(30):
        if not alive(supervisor): break
        time.sleep(1)
    if alive(supervisor): raise RuntimeError('old supervisor did not exit')
    with sqlite3.connect(a.source / 'validation/queue.sqlite') as db:
        audit['drained_queue'] = dict(db.execute('SELECT state,count(*) FROM jobs GROUP BY state'))
        if audit['drained_queue'].get('claimed', 0): raise RuntimeError('production still has claims')
    audit['drained_at'] = now(); write_json(a.run / 'production_pause.json', audit)


def resume_production(a):
    from scripts.sft_eval_local_recovery import alive
    audit = read_json(a.run / 'production_pause.json')
    if any(alive(pid) for pid in audit['worker_pids']):
        # Failed drain: return queued jobs to existing workers, then wake their original supervisor.
        with sqlite3.connect(a.source / 'validation/queue.sqlite', timeout=60) as db:
            db.execute("UPDATE jobs SET state='pending' WHERE state='held_benchmark'")
        if alive(audit['supervisor_pid']): os.kill(audit['supervisor_pid'], signal.SIGCONT)
        write_json(a.run / 'resume.json', {'at': now(), 'mode': 'original_supervisor',
            'live_workers': sum(alive(pid) for pid in audit['worker_pids'])})
        return
    if alive(audit['supervisor_pid']):
        os.kill(audit['supervisor_pid'], signal.SIGTERM); os.kill(audit['supervisor_pid'], signal.SIGCONT)
        time.sleep(2)
    log = (a.run / 'production_resume.log').open('a')
    process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'scripts/resume_sft_after_benchmark.py'),
        '--root', str(a.source), '--archive', str(a.archive), '--benchmark', str(a.run)],
        cwd=ROOT, env=env(), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    log.close()
    write_json(a.run / 'resume.json', {'at': now(), 'mode': 'exact_queue_continuation', 'pid': process.pid})
    time.sleep(5)
    if process.poll() is not None: raise RuntimeError('production continuation failed; inspect production_resume.log')


def compare(a, reports):
    baseline, shared = reports
    speedup = baseline['wall_seconds'] / shared['wall_seconds']
    pairs, summaries = [], {}
    for mode in ('baseline', 'shared'):
        outcomes = {}; metrics = Counter(); errors = Counter()
        with sqlite3.connect(a.run / mode / 'queue.sqlite') as db:
            for candidate, task, output in db.execute('SELECT candidate,task,output FROM jobs'):
                folder = Path(output) / 'episodes/t00000/a0'
                result = read_json(folder / 'episode_result.json')
                trajectory = read_json(folder / 'trajectory.json')
                outputs = [read_json(p)['content'] for p in sorted((folder / 'model_turns').glob('*/output.json'))]
                outcomes[candidate, task] = {'status': result['task_status'], 'business_label': result.get('business_label'),
                    'output_hash': digest(outputs), 'answer_hash': digest(trajectory.get('final_answer')),
                    'turns': result['turns'], 'tool_calls': result['tool_calls']}
                metrics[result['termination_reason']] += 1
                if result.get('error'): errors[result['error']] += 1
        summaries[mode] = {'outcomes': outcomes, 'termination_reasons': dict(metrics), 'errors': dict(errors)}
    for key, row in summaries['baseline']['outcomes'].items():
        other = summaries['shared']['outcomes'][key]
        pairs.append({'candidate': key[0], 'task': key[1], 'baseline': row, 'shared': other,
            'same_outputs': row['output_hash'] == other['output_hash'], 'same_status': row['status'] == other['status']})
    with sqlite3.connect(a.source / 'validation/queue.sqlite') as db:
        remaining_validation = db.execute("SELECT count(*) FROM jobs WHERE state IN ('held_benchmark','pending','claimed')").fetchone()[0]
    remaining = remaining_validation + 2 * 588
    baseline_hours = remaining / baseline['jobs_per_minute'] / 60
    shared_hours = remaining / shared['jobs_per_minute'] / 60
    result = {'at': now(), 'baseline': baseline, 'shared': shared, 'wall_speedup': speedup,
        'wall_time_saved_fraction': 1 - 1 / speedup, 'remaining_validation_jobs': remaining_validation,
        'future_test_jobs': 1176, 'estimated_baseline_hours': baseline_hours,
        'estimated_shared_hours': shared_hours, 'estimated_hours_saved': baseline_hours - shared_hours,
        'same_output_jobs': sum(r['same_outputs'] for r in pairs),
        'same_status_jobs': sum(r['same_status'] for r in pairs), 'pairs': pairs,
        'diagnostics': {k: {x: y for x, y in v.items() if x != 'outcomes'} for k, v in summaries.items()},
        'limitations': ['10 instruction-length-stratified validation tasks x 5 checkpoints; not a precision ETA',
            'Natural generation may diverge; work lengths and success rates must be examined together',
            'Shared mode also increases independent environment/Judge clients from 18 to 32',
            'One baseline then one shared pass; external Judge/network variability is not controlled',
            'ETA applies measured five-checkpoint validation mix to future two-checkpoint test mix',
            'Startup, benchmark interruption and migration costs excluded from steady-state ETA',
            'Original reference kernels; static microbatches, not vLLM continuous batching'],
        'formal_inference_changed': False}
    write_json(a.run / 'summary.json', result)
    return result


def control(a):
    from dotenv import load_dotenv
    from harness_awm.training.selection import checked_plan
    from harness_awm.training.tasks import load_tasks
    from harness_awm.training.evaluate import preflight
    from harness_awm.training.local import checkpoint_identity
    from harness_awm.contracts import RunConfig
    load_dotenv(ROOT / '.env', override=False)
    a.run.mkdir(exist_ok=False, parents=True)
    plan = checked_plan(a.source)
    tasks, _, _ = load_tasks(plan['tasks_path'], 'validation')
    selected = pick_tasks(tasks, 10, 20261109)
    preflight(selected, RunConfig.model_validate(plan['config']), 'validation')
    candidates = read_json(a.source / 'validation/candidates.json')
    for value in candidates.values():
        if checkpoint_identity(value['path']) != value: raise ValueError('checkpoint changed')
    manifest = {'at': now(), 'plan': plan, 'candidates': candidates,
        'task_ids': [t.task_instance_id for t in selected],
        'task_instruction_lengths': [len(t.instruction) for t in selected],
        'selection': '10 fixed-seed instruction-length strata; no outcome-based filtering',
        'ports': {name: 18740 + i for i, name in enumerate(candidates)},
        'code_hashes': {n: file_hash(ROOT / 'scripts' / n) for n in
            ('benchmark_shared_e2e.py', 'benchmark_shared_batch.py', 'resume_sft_after_benchmark.py')}}
    write_json(a.run / 'experiment.json', manifest)
    paused = False
    try:
        paused = True; pause_production(a)
        reports = [run_mode(a, mode, manifest) for mode in ('baseline', 'shared')]
        result = compare(a, reports)
        write_json(a.run / 'status.json', {'stage': 'finished', 'at': now(), 'speedup': result['wall_speedup']})
    except BaseException as exc:
        write_json(a.run / 'status.json', {'stage': 'failed', 'at': now(), 'error': f'{type(exc).__name__}: {exc}'})
        raise
    finally:
        if paused and (a.run / 'production_pause.json').exists(): resume_production(a)


def continue_shared(a):
    """Reuse the fully completed baseline after a controller-only reporting failure."""
    from harness_awm.training.selection import checked_plan
    manifest = read_json(a.run / 'experiment.json')
    if checked_plan(a.source) != manifest['plan']: raise RuntimeError('production plan changed')
    stage = a.run / 'baseline'
    counts = completed_queue_counts(stage / 'queue.sqlite', [0] * 18)
    timings = [read_json(p) for p in (stage / 'timings').glob('*.json')]
    if counts != {'done': 50} or len(timings) != 50: raise RuntimeError('baseline evidence incomplete')
    start = read_json(stage / 'go.json')['epoch']; end = max(r['end'] for r in timings)
    telemetry = read_json(stage / 'gpu.json')
    baseline = {'mode': 'baseline', 'jobs': 50, 'task_workers': 18, 'model_processes': 18,
        'wall_seconds': end - start, 'jobs_per_minute': 50 * 60 / (end - start),
        'confirmed_successes': sum(r['evaluation']['confirmed_successes'] for r in timings),
        'scored': sum(r['evaluation']['scored'] for r in timings),
        'peak_sampled_gpu_MiB': max(int(x['gpu'].split(',')[0]) for x in telemetry),
        'startup_seconds': None, 'startup_note': 'not recovered from the failed controller timing',
        'controller_failure_only': True, 'original_outcomes_reused_without_rerunning': True}
    write_json(stage / 'summary.json', baseline)
    for name in ('status.json', 'production_pause.json', 'resume.json'):
        path = a.run / name
        if path.exists(): path.rename(a.run / ('first_' + name))
    write_json(a.run / 'shared_continuation.json', {'at': now(),
        'reason': 'Refresh completion state after worker exits; reuse all 50 successful baseline worker records',
        'script_sha256': file_hash(__file__), 'resume_script_sha256': file_hash(ROOT / 'scripts/resume_sft_after_benchmark.py'),
        'judge_recovery_running_in_parallel': True})
    try:
        pause_production(a)
        shared = run_mode(a, 'shared', manifest)
        result = compare(a, [baseline, shared])
        write_json(a.run / 'status.json', {'stage': 'finished', 'at': now(), 'speedup': result['wall_speedup']})
    except BaseException as exc:
        write_json(a.run / 'status.json', {'stage': 'failed', 'at': now(), 'error': f'{type(exc).__name__}: {exc}'})
        raise
    finally:
        if (a.run / 'production_pause.json').exists(): resume_production(a)


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('control', 'continue-shared', 'serve', 'worker'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--stage', type=Path)
    parser.add_argument('--mode', choices=('baseline', 'shared'))
    parser.add_argument('--candidate'); parser.add_argument('--port', type=int)
    parser.add_argument('--index', type=int)
    a = parser.parse_args()
    if a.action == 'serve': asyncio.run(serve(a))
    elif a.action == 'worker': worker(a)
    else:
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'benchmark interrupted by signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        (continue_shared if a.action == 'continue-shared' else control)(a)


if __name__ == '__main__': main()
