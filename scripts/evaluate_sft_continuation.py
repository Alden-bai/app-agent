"""Evaluate continuation checkpoints on the frozen 589-task validation split."""
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
import tarfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness_awm.io import read_json, write_json, file_hash, now, digest
from harness_awm.training import selection
from harness_awm.training.local import checkpoint_identity
from harness_awm.tracking import SwanLabTracker
from scripts import resume_sft_validation as scheduler
from scripts import sft_server_recovery_20261110 as server
from scripts import sft_remaining_recheck_20261110 as judge_impl
CONTROL = Path('/root/autodl-tmp/awm-sft-continuation-validation-20261113-v1')
PERSISTENT = Path('/autodl-fs/data/awm-sft-continuation-validation-20261113-v1')
ARCHIVE = Path('/dev/shm/awm-sft-continuation-validation-20261113-v1')
RUN = CONTROL
STEPS = (315, 420)

def status(phase, stage, **extra):
    write_json(RUN / (phase + '_status.json'), dict(at=now(), pid=os.getpid(), stage=stage, **extra))

def prepare():
    if CONTROL.exists():
        raise RuntimeError('Existing run preserved')
    original_path = server.REPAIRS / 'plan.json'
    original = read_json(original_path)
    assert file_hash(original_path) == read_json(server.REPAIRS / 'plan.sha256.json')['sha256']
    judge_config = read_json(judge_impl.RUN / 'plan.json')['config']
    config = server.RunConfig.model_validate(judge_config)
    assert os.environ.get(config.judge_api.api_key_env, '').strip(), 'Judge credential missing'
    CONTROL.mkdir()
    ARCHIVE.mkdir()
    write_json(CONTROL / 'historical_baseline.json', read_json(ROOT / 'runs/validation_comparison_20261110/summary.json'))
    for step in STEPS:
        name = f'checkpoint-{step}'
        root = CONTROL / name
        root.mkdir()
        checkpoint = checkpoint_identity(Path('/autodl-fs/data/awm-sft-continuation-20261113-v1/training') / name)
        plan = dict(original, source_hashes=selection.sources(), created_at=now(),
                    selected=name, sft_candidates={name:checkpoint}, eval_workers=18,
                    judge_deferred=True, source_plan_sha256=file_hash(original_path))
        write_json(root / 'plan.json', plan)
        write_json(root / 'plan.sha256.json', {'sha256':file_hash(root / 'plan.json')})
        write_json(root / 'judge_config.json', judge_config)
        write_json(root / 'policy.json', dict(script_sha256=file_hash(__file__),
            judge_source_hashes=judge_impl.hashes(), kernel='reference', tasks=589,
            baseline='historical checkpoint-210; no test data', archive=str(ARCHIVE)))
        selection.prepare_evaluation(root, plan, {name:checkpoint}, 'validation')
        assert len(read_json(root / 'validation' / name / 'manifest.json')['candidates']) == 589
    write_json(CONTROL / 'status.json', dict(stage='prepared', at=now(), checkpoints=list(STEPS)))

def run_one(step, gpu_only=False):
    global RUN
    RUN = CONTROL / f'checkpoint-{step}'
    with (RUN / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = selection.checked_plan(RUN)
        policy = read_json(RUN / 'policy.json')
        assert file_hash(__file__) == policy['script_sha256']
        assert judge_impl.hashes() == policy['judge_source_hashes']
        tracker = SwanLabTracker(RUN / 'tracking', mode='offline', config={'stage':'continuation-validation'})
        try:
            with sqlite3.connect(RUN / 'validation/queue.sqlite') as db:
                counts = dict(db.execute('SELECT state,count(*) FROM jobs GROUP BY state'))
            assert not any(v for k,v in counts.items() if k not in ('pending','done')), counts
            if counts.get('pending'):
                assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(), 'GPU busy'
                scheduler.CONTROL = RUN
                scheduler.ARCHIVE = ARCHIVE / 'rollouts'
                scheduler.ARCHIVE.mkdir(exist_ok=True)
                scheduler.__file__ = str(Path(__file__).resolve())
                status('gpu','running', total=589)
                scheduler.run_stage(RUN, plan, 'validation', tracker, 'gpu_' + str(int(time.time())))
            status('gpu','complete', total=589)
            persist(step, 'rollouts')
            if gpu_only:
                write_json(RUN / 'status.json', dict(stage='gpu_complete_awaiting_judge', at=now()))
                return
            asyncio.run(judge(plan, tracker))
            persist(step, 'scored')
        finally:
            tracker.finish()

def persist(step, phase):
    # Tar bundles keep evidence durable without exhausting the network disk's inodes.
    destination = PERSISTENT / f'checkpoint-{step}-{phase}.tar'
    temporary = destination.with_suffix('.tar.partial')
    with tarfile.open(temporary, 'w') as bundle:
        for path in (RUN, ARCHIVE / 'rollouts' / RUN.name, ARCHIVE / 'judge'):
            if path.exists():
                bundle.add(path, arcname=str(path).lstrip('/'), recursive=True)
    temporary.replace(destination)
    write_json(destination.with_suffix('.sha256.json'), {'sha256':file_hash(destination), 'at':now()})
    if (RUN / 'summary.json').exists():
        write_json(PERSISTENT / f'checkpoint-{step}-summary.json', read_json(RUN / 'summary.json'))

def supervise(gpu_only=False):
    with (CONTROL / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for step in STEPS:
                write_json(CONTROL / 'status.json', dict(stage='running', checkpoint=step, at=now()))
                subprocess.run([sys.executable,__file__,'one','--step',str(step)] + (['--gpu-only'] if gpu_only else []),check=True)
                state=read_json(CONTROL / f'checkpoint-{step}' / 'status.json')
                if state['stage'] != ('gpu_complete_awaiting_judge' if gpu_only else 'complete'):
                    raise RuntimeError(str(state))
            if gpu_only:
                write_json(CONTROL / 'status.json', dict(stage='gpu_complete_awaiting_judge',at=now()))
                return
            reports={str(step):read_json(CONTROL / f'checkpoint-{step}' / 'summary.json') for step in STEPS}
            write_json(CONTROL / 'comparison.json', dict(at=now(), validation=reports,
                historical_baseline=read_json(CONTROL / 'historical_baseline.json')['summary']['lr1e-5-epoch2']))
            write_json(PERSISTENT / 'comparison.json', read_json(CONTROL / 'comparison.json'))
            write_json(CONTROL / 'status.json', dict(stage='complete',at=now()))
        except BaseException as exc:
            write_json(CONTROL / 'status.json', dict(stage='needs_attention',error=str(exc),at=now()))
            raise

def prepare_judge(plan):
    root = RUN / 'judge'
    if (root / 'plan.json').exists():
        saved = read_json(root / 'plan.json')
        assert file_hash(root / 'plan.json') == read_json(root / 'plan.sha256.json')['sha256']
        return saved
    config = read_json(RUN / 'judge_config.json')
    rows = []
    name = plan['selected']
    for item in read_json(RUN / 'validation' / name / 'manifest.json')['candidates']:
        source = Path(item['path'])
        result = read_json(source / 'episode_result.json')
        if server.evaluation_score(result) is not None:
            continue
        if result['execution_integrity'] != 'valid' or result['terminal_state_status'] != 'trusted':
            continue
        key = digest([name, item['task_instance_id'], str(source)])[:24]
        rows.append(dict(key=key, candidate=name, task_id=item['task_instance_id'].split(':', 1)[1],
            task_instance_id=item['task_instance_id'], source_episode=str(source), reason='deferred_validation_judge',
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
    runtime.config_hash = lambda cfg: digest({'original': original_hash(cfg), 'validation_policy': file_hash(root / 'plan.json')})
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
    server.recovery.overlay_repairs(RUN / 'validation', list(records.values()))
    result = selection.aggregate(RUN / 'validation', plan)[plan['selected']]
    for record in records.values():
        for key, value in record['usage'].items():
            result['budget'][key] = result['budget'].get(key, 0) + value
    write_json(RUN / 'validation' / plan['selected'] / 'metrics.json', result)
    write_json(RUN / 'summary.json', dict(at=now(), selected=plan['selected'],
        test_used_for_selection=False, validation=selection.compact(result), budget=result['budget']))
    tracker.log({'validation_final': selection.compact(result)}, step=100000)
    stage = 'complete' if not fatal and not transport.stopped and len(records) == len(jp['cohort']) else 'needs_attention'
    status('judge', stage, scored=result['evaluation']['scored'], total=589,
           remaining_unscored=589-result['evaluation']['scored'], fatal=fatal, transport_stop=transport.stopped)
    write_json(RUN / 'status.json', dict(at=now(), stage=stage, selected=plan['selected'], report=str(RUN / 'summary.json')))



def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['prepare','run','one','worker'])
    p.add_argument('--root',type=Path)
    p.add_argument('--split',default='validation')
    p.add_argument('--index',type=int)
    p.add_argument('--control',type=Path)
    p.add_argument('--step',type=int,choices=STEPS)
    p.add_argument('--gpu-only',action='store_true')
    args=p.parse_args()
    if args.mode=='prepare': prepare()
    elif args.mode=='run': supervise(args.gpu_only)
    elif args.mode=='one': run_one(args.step,args.gpu_only)
    else: server.worker(args)

if __name__=='__main__': main()
