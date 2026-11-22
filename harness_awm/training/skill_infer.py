"""Frozen skill inference only. Never loads API credentials or invokes a Judge."""
import argparse
import asyncio
from collections import Counter
import fcntl
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import time

from ..backend import Budget
from ..contracts import ROOT, RunConfig, TaskSpec
from ..io import digest, file_hash, now, read_json, verify_seal, write_json
from ..runner import EpisodeRunner
from ..skillbank import SkillBank
from ..stages import official_hashes, check_manifest
from .actor import TrainingActorBackend, STOP_POLICY
from .baseline import isolation_probe
from .evaluate import preflight
from .local import checkpoint_identity, checkpoint_filesystem, load_checkpoint
from .tasks import load_tasks


def gpu_sample():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True, timeout=15)
    used, total, utilization = map(int, raw.strip().split(','))
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                   '--format=csv,noheader,nounits'], text=True, timeout=15)
    processes = {}
    for row in raw.splitlines():
        pid, memory = row.split(',')
        processes[int(pid)] = int(memory) / 1024
    return dict(used_gib=used / 1024, total_gib=total / 1024, utilization=utilization,
                processes=processes)


def admission(gpu, live, peaks, retired):
    # Keep six GiB free for transient prefill/KV growth. Admit against the largest
    # observed replica, not just model weight size. Wait for new replicas to load.
    limit = min(89.5, gpu['total_gib'] - 6)
    replica = max(7., max(peaks.values(), default=0) + .35)
    starting = any(pid not in gpu['processes'] for pid in live.values())
    draining = any(i in retired for i in live)
    return dict(limit_gib=limit, replica_gib=replica,
                add=not starting and not draining and gpu['used_gib'] + replica < limit,
                retire=gpu['used_gib'] > limit)


async def rollout(runner, task, folder, ident, actor):
    result = await runner.run(task, folder, ident, actor, evaluate_after=False)
    verify_seal(folder)
    if result['evaluation_status'] != 'not_run' or result['evaluation_valid']:
        raise RuntimeError('Unexpected evaluation in inference-only run')
    if runner.budget.summary()['physical_requests'] != 0:
        raise RuntimeError('Unexpected API request in inference-only run')
    return result


def prepare(a):
    if a.output.exists():
        raise ValueError('Use a new output directory')
    raw = read_json(a.config)
    tasks, task_manifest, sources = load_tasks(a.tasks, a.split)
    # Source/rubric validation is local only; reserve no external request budget.
    config = RunConfig.model_validate({**raw, 'skills': read_json(a.skills),
        'judge_inputs_manifest': sources, 'max_physical_requests': 0, 'max_total_tokens': 0})
    checked = preflight(tasks, config.model_copy(update={'max_physical_requests': 1,
                         'max_total_tokens': 1}), a.split)
    bank = SkillBank(config.skills)
    checkpoint = checkpoint_identity(a.model)
    selection = read_json(a.selection)
    if checkpoint['sha256'] != selection['checkpoint']['sha256']:
        raise ValueError('Checkpoint differs from validation-selected best model')
    tasks.sort(key=lambda t: digest([config.seed, t.task_instance_id]))
    if a.limit:
        tasks = tasks[:a.limit]
    previews = []
    for task in tasks:
        catalog = read_json(Path(task.environment_ref).with_name('tools.json'))
        selected = bank.retrieve(task, catalog['tools'] if isinstance(catalog, dict) else catalog)
        previews.append(dict(task_instance_id=task.task_instance_id, selected=selected))
    a.output.mkdir(parents=True)
    if not isolation_probe(config, a.output / 'isolation_probe')['passed']:
        raise RuntimeError('Runtime isolation probe failed')
    # Freeze a copy of the bank inside the run for later review.
    shutil.copy2(config.skills.bank_path, a.output / 'bank.json')
    config = config.model_copy(update={'skills': config.skills.model_copy(
        update={'bank_path': str(a.output / 'bank.json')})})
    candidates = [dict(rollout_id=f't{i:05d}', task_instance_id=t.task_instance_id,
                      scenario=t.scenario, path=f'episodes/t{i:05d}/a0') for i, t in enumerate(tasks)]
    manifest = dict(schema_version='awm-run-v1', stage='inference', created_at=now(),
        config=config.model_dump(), tasks=[t.model_dump() for t in tasks], candidates=candidates,
        checkpoint=checkpoint, checkpoint_filesystem=checkpoint_filesystem(a.model),
        task_manifest=task_manifest, evaluation_split=a.split, sampling='greedy', dtype='bfloat16',
        stop_policy=STOP_POLICY, seed=config.seed, training_updates=False, judge_deferred=True,
        official_source_hashes=official_hashes(config),
        harness_source_hashes={str(p.relative_to(ROOT)):file_hash(p) for p in (ROOT/'harness_awm').rglob('*.py')},
        selection_sha256=file_hash(a.selection), workers=a.workers)
    write_json(a.output / 'manifest.json', manifest)
    write_json(a.output / 'manifest.sha256.json', dict(sha256=file_hash(a.output/'manifest.json')))
    write_json(a.output / 'retrieval_preview.json', previews)
    write_json(a.output / 'preflight.json', dict(**checked, checkpoint_matches_best=True,
        planned=len(tasks), skills=len(bank.skills), bank_sha256=bank.sha256,
        tasks_with_skills=sum(bool(p['selected']) for p in previews),
        tasks_with_workflows=sum(any(s['kind']=='workflow' for s in p['selected']) for p in previews),
        judge_enabled=False, api_request_budget=0))
    with sqlite3.connect(a.output/'queue.sqlite') as db:
        db.execute('CREATE TABLE jobs (id INTEGER PRIMARY KEY, state TEXT, owner INTEGER)')
        db.executemany("INSERT INTO jobs VALUES (?, 'pending', NULL)", [(i,) for i in range(len(tasks))])
    write_json(a.output / 'status.json', dict(stage='prepared', at=now(), total=len(tasks), judge_enabled=False))


def worker(root, index):
    import torch
    from transformers import set_seed
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(7*1024**3/torch.cuda.get_device_properties(0).total_memory)
    m = check_manifest(root)
    if checkpoint_filesystem(m['checkpoint']['path']) != m['checkpoint_filesystem']:
        raise ValueError('Checkpoint changed since preflight')
    for p, expected in m['harness_source_hashes'].items():
        if file_hash(ROOT/p) != expected:
            raise ValueError('Inference source changed: '+p)
    config = RunConfig.model_validate(m['config'])
    if config.max_physical_requests != 0 or config.max_total_tokens != 0:
        raise ValueError('Inference requires zero API budgets')
    model, tokenizer = load_checkpoint(m['checkpoint']['path'], dtype=m['dtype'])
    budget = Budget(root/f'worker-{index}.budget.sqlite', 0, 0)
    async def run():
        runner = EpisodeRunner(config, budget, asyncio.Semaphore(1))
        while True:
            control = root / 'worker_control.json'
            if control.exists() and index in read_json(control)['retire_indices']:
                return
            with sqlite3.connect(root/'queue.sqlite', timeout=60) as db:
                db.execute('BEGIN IMMEDIATE')
                row = db.execute("SELECT id FROM jobs WHERE state='pending' ORDER BY id LIMIT 1").fetchone()
                if row is None:
                    return
                i = row[0]
                db.execute("UPDATE jobs SET state='running',owner=? WHERE id=?", (index,i))
            task = TaskSpec.model_validate(m['tasks'][i])
            c = m['candidates'][i]
            set_seed(int(digest([config.seed,task.task_instance_id])[:8],16))
            actor = TrainingActorBackend(model,tokenizer,policy_version=m['checkpoint']['sha256'],
                max_output_tokens=config.max_output_tokens,context_limit=config.context_limit,
                greedy=True,record_policy=False)
            try:
                result = await rollout(runner,task,root/c['path'],c['rollout_id'],actor)
                if result['execution_integrity'] != 'valid' or result['terminal_state_status'] != 'trusted':
                    raise RuntimeError('Infrastructure failure: '+str(result))
                state='done'
                print(c['rollout_id'],result['termination_reason'],flush=True)
            except BaseException as exc:
                write_json(root/'failures'/f'{c["rollout_id"]}.json',dict(error=repr(exc),worker=index))
                with sqlite3.connect(root/'queue.sqlite',timeout=60) as db:
                    db.execute("UPDATE jobs SET state='error' WHERE id=?",(i,))
                raise
            with sqlite3.connect(root/'queue.sqlite',timeout=60) as db:
                db.execute('UPDATE jobs SET state=? WHERE id=?',(state,i))
    asyncio.run(run())


def execute(a):
    root=a.output
    m=check_manifest(root)
    with (root/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                 TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                 SWANLAB_MODE='disabled', PYTHONUNBUFFERED='1')
        # No credentials are needed for local inference or the local AWM tools.
        for key in list(env):
            if key.endswith('_API_KEY') or key in {'HF_TOKEN','HUGGING_FACE_HUB_TOKEN'}:
                env.pop(key)
        env['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
        processes={}
        retired=set()
        peaks={}
        last_add=-float('inf')
        control=root/'worker_control.json'
        write_json(control,dict(retire_indices=[]))
        while True:
            with sqlite3.connect(root/'queue.sqlite') as db:
                counts=dict(db.execute('SELECT state,count(*) FROM jobs GROUP BY state'))
            live={i:p for i,p in processes.items() if p.poll() is None}
            gpu=gpu_sample()
            for i,p in live.items():
                peaks[i]=max(peaks.get(i,0),gpu['processes'].get(p.pid,0))
            decision=admission(gpu,{i:p.pid for i,p in live.items()},peaks,retired)
            effective=[i for i in live if i not in retired]
            if decision['retire'] and effective and not any(i in retired for i in live):
                retired.add(max(effective))
                write_json(control,dict(retire_indices=sorted(retired),reason='GPU headroom guard'))
            failed={i:p.poll() for i,p in processes.items() if p.poll() not in (None,0)}
            if failed or counts.get('error'):
                retired.update(live)
                write_json(control,dict(retire_indices=sorted(retired),reason='Worker failure; drain remaining episodes'))
            elif (counts.get('pending',0)>len(live) and len(live)<m['workers'] and decision['add']
                  and time.monotonic()-last_add>=(10 if len(live)<12 else 30)):
                i=max(processes,default=-1)+1
                with (root/f'worker-{i}.log').open('ab') as log:
                    processes[i]=subprocess.Popen([sys.executable,'-m','harness_awm.training.skill_infer',
                        'worker','--output',str(root),'--index',str(i)],cwd=ROOT,env=env,
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                last_add=time.monotonic()
                write_json(root/'launch.json',dict(pid=os.getpid(),workers={i:p.pid for i,p in processes.items()},at=now()))
            exits=[p.poll() for p in processes.values()]
            status=dict(stage='running',at=now(),counts=counts,total=len(m['tasks']),
                        live_workers=exits.count(None),judge_enabled=False,gpu=gpu,admission=decision,
                        retired_workers=sorted(retired))
            write_json(root/'status.json',status)
            with (root/'gpu.jsonl').open('a') as log:
                log.write(__import__('json').dumps(status)+'\n')
            if not live and (not counts.get('pending') or failed or counts.get('error')):
                break
            time.sleep(3)
        if any(exits) or counts.get('done')!=len(m['tasks']):
            write_json(root/'status.json',dict(status,stage='needs_attention',exit_codes=exits))
            raise RuntimeError('Incomplete inference; inspect worker logs')
        records=[]
        injected=0
        for c in m['candidates']:
            folder=root/c['path']
            verify_seal(folder)
            records.append(read_json(folder/'episode_result.json'))
            injected+=bool(read_json(folder/'skill_selection.json')['selected'])
        budget=sum(Budget(p,0,0).summary()['physical_requests'] for p in root.glob('worker-*.budget.sqlite'))
        assert budget==0 and all(r['evaluation_status']=='not_run' for r in records)
        summary=dict(completed=len(records),total=len(m['tasks']),checkpoint_sha256=m['checkpoint']['sha256'],
            split=m['evaluation_split'],tasks_with_skills=injected,judge_enabled=False,judge_requests=budget,
            scored=0,success_rate=None,training_updates=False,
            terminations=dict(Counter(r['termination_reason'] for r in records)),
            tool_calls=sum(r['tool_calls'] for r in records),turns=sum(r['turns'] for r in records),at=now())
        write_json(root/'summary.json',summary)
        write_json(root/'status.json',dict(status,stage='inference_complete_awaiting_manual_judge'))
        if a.archive:
            a.archive.parent.mkdir(parents=True,exist_ok=True)
            if a.archive.exists():
                raise ValueError('Archive already exists')
            partial=a.archive.with_suffix('.partial')
            with tarfile.open(partial,'w:gz',compresslevel=1) as tar:
                for path in sorted(root.iterdir()):
                    tar.add(path,arcname=str(Path(root.name)/path.name))
            partial.rename(a.archive)
            write_json(a.archive.with_suffix('.sha256.json'),dict(sha256=file_hash(a.archive),at=now()))
            write_json(root/'archive.json',dict(path=str(a.archive),sha256=file_hash(a.archive)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run','worker'])
    p.add_argument('--output',type=Path,required=True)
    for name in ['config','tasks','model','skills','selection','archive']:
        p.add_argument('--'+name,type=Path)
    p.add_argument('--split',choices=['validation','test'],default='validation')
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--workers',type=int,default=12)
    p.add_argument('--index',type=int,default=0)
    a=p.parse_args()
    a.output=a.output.resolve()
    if a.limit<0 or not 1<=a.workers<=18:
        p.error('Require nonnegative limit and 1..18 workers')
    if a.action=='prepare':
        if not all(getattr(a,n) for n in ['config','tasks','model','skills','selection']):
            p.error('prepare requires config/tasks/model/skills/selection')
        prepare(a)
    elif a.action=='worker':
        worker(a.output,a.index)
    else:
        execute(a)


if __name__=='__main__':
    main()
