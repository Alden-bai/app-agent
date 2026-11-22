"""Resume audited validation and uniform Judge repairs after inode exhaustion."""
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
from harness_awm.training import selection
from harness_awm.tracking import SwanLabTracker
from scripts.sft_eval_local_recovery import archive_episode
from scripts import sft_judge_recovery as recovery

SOURCE = ROOT / 'runs/qwen35_sft_eval_local_20261109_v1'
REPAIRS = recovery.OUTPUT
CONTROL = ROOT / 'runs/sft_validation_resume_20261109_v1'
ARCHIVE = Path('/root/autodl-tmp/awm-sft-validation-resume-20261109')


def gpu_sample():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=10).strip().split(',')
    return {'memory_gib': int(raw[0])/1024, 'utilization': int(raw[1]), 'at': now()}


def run_stage(root, plan, split, tracker, label):
    """Existing protocol and frozen weights; only scheduling/storage changes."""
    stage = root / split
    logs, processes, archived = [], [], set()
    directory = CONTROL / label
    directory.mkdir(exist_ok=False)
    write_json(directory / 'worker_control.json', {'retire_indices': []})
    env = dict(os.environ, SWANLAB_MODE='disabled', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        PYTORCH_ALLOC_CONF='expandable_segments:True', TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    with sqlite3.connect(stage/'queue.sqlite') as db:
        pending = db.execute("SELECT count(*) FROM jobs WHERE state='pending'").fetchone()[0]
    count = min(18, pending)
    try:
        for index in range(count):
            log = (directory/f'worker_{index:02d}.log').open('x');logs.append(log)
            processes.append(subprocess.Popen([sys.executable, __file__, 'worker', '--root', str(root),
                '--split', split, '--index', str(index), '--control', str(directory)], cwd=ROOT,
                env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT))
            time.sleep(1)
        write_json(stage/'launch.json', {'worker_pids':[p.pid for p in processes], 'started_at':now(), 'resume_record':str(directory)})
        last_report = 0; step = 0; high_samples = 0; retired = set(); peak = 0
        with (directory/'gpu.jsonl').open('a', buffering=1) as telemetry:
            while True:
                exits = [p.poll() for p in processes]
                gpu = gpu_sample();peak = max(peak, gpu['memory_gib'])
                telemetry.write(__import__('json').dumps(gpu)+'\n')
                high_samples = high_samples + 1 if gpu['memory_gib'] > 86 else 0
                # Retire at an episode boundary, preserving in-flight model/Judge work.
                if (high_samples >= 3 and len(retired) < max(0,count-14)
                        and not any(exits[i] is None for i in retired)):
                    index = next((i for i in reversed(range(count)) if i not in retired and exits[i] is None), None)
                    if index is not None:
                        retired.add(index)
                        write_json(directory/'worker_control.json', {'retire_indices':sorted(retired), 'reason':'GPU above 86 GiB', 'at':now()})
                    high_samples = 0
                if time.monotonic()-last_report > 25 or all(x is not None for x in exits):
                    with sqlite3.connect(stage/'queue.sqlite') as db:
                        counts = dict(db.execute('SELECT state,count(*) FROM jobs GROUP BY state'))
                        done = db.execute("SELECT output FROM jobs WHERE state='done'").fetchall()
                    for (output,) in done:
                        if output in archived:continue
                        folder = Path(output)/'episodes/t00000/a0'
                        archive_episode(folder, ARCHIVE/root.name/folder.relative_to(root))
                        archived.add(output)
                    reports = selection.aggregate(stage, plan)
                    if root == SOURCE:
                        for name,budget in read_json(root/'original_budgets.json').items():
                            for key,value in budget.items():reports[name]['budget'][key]=reports[name]['budget'].get(key,0)+value
                            write_json(stage/name/'metrics.json', reports[name])
                    status = {'stage':split, 'updated_at':now(), 'queue':counts,
                        'live_workers':sum(x is None for x in exits), 'worker_exit_codes':exits,
                        'gpu':gpu, 'peak_sampled_gib':peak, 'retired_workers':sorted(retired),
                        'candidates':{k:selection.compact(v) for k,v in reports.items()}, 'resume_record':str(directory)}
                    write_json(root/'status.json',status);write_json(directory/'status.json',status)
                    tracker.log({split:status['candidates'], 'queue':counts, 'gpu':gpu,
                        'live_workers':status['live_workers']}, step=step);step+=1
                    if min(shutil.disk_usage(root).free,shutil.disk_usage(ARCHIVE).free)<2*1024**3:
                        with sqlite3.connect(stage/'queue.sqlite') as db:db.execute("UPDATE jobs SET state='held_disk' WHERE state='pending'")
                    last_report = time.monotonic()
                    if all(x is not None for x in exits):
                        if any(x != 0 for x in exits) or any(v for k,v in counts.items() if k!='done'):
                            raise RuntimeError('validation incomplete; evidence retained, automatic selection blocked')
                        return reports
                time.sleep(2)
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:p.kill()
        for log in logs:log.close()


def primary():
    with (SOURCE/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan = selection.checked_plan(SOURCE)
        tracker = SwanLabTracker(CONTROL/'primary_tracking', mode='online', config={
            'stage':'resume-original-checkpoint-validation', 'workers':18, 'gpu_target_gib':85,
            'completed_results_reused':True, 'inference_protocol_unchanged':True})
        write_json(SOURCE/'tracking.json',{'url':tracker.url,'resume_record':str(CONTROL)})
        write_json(SOURCE/'launch.json',{'pid':os.getpid(),'started_at':now(),'resume_record':str(CONTROL)})
        failed = True
        try:
            run_stage(SOURCE,plan,'validation',tracker,'primary')
            write_json(SOURCE/'status.json',{'stage':'awaiting_judge_recovery','validation_complete':True,'updated_at':now()})
            failed = False
        except BaseException as exc:
            write_json(SOURCE/'status.json',{'stage':'needs_attention','error':str(exc),'updated_at':now()});raise
        finally:tracker.finish(failed=failed)


def judge():
    with (REPAIRS/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=selection.checked_plan(REPAIRS)
        policy=read_json(REPAIRS/'policy.json')
        if any(file_hash(ROOT/k)!=v for k,v in policy['files'].items()):raise ValueError('frozen Judge recovery changed')
        tracker=SwanLabTracker(CONTROL/'judge_tracking',mode='online',config={'stage':'resume-uniform-judge-and-selection','policy':policy})
        write_json(REPAIRS/'tracking.json',{'url':tracker.url,'resume_record':str(CONTROL)})
        write_json(REPAIRS/'resume_launch.json',{'pid':os.getpid(),'at':now()})
        failed=True
        try:
            rows=recovery.monitor_repairs(SOURCE,'validation',plan,tracker,archive=ARCHIVE/REPAIRS.name,wait_primary=True)
            candidates=recovery.prepare_validation(plan,rows)
            run_stage(REPAIRS,plan,'validation',tracker,'snapshot_recovery')
            recovery.monitor_repairs(REPAIRS,'validation',plan,tracker,archive=ARCHIVE/REPAIRS.name)
            recovery.overlay_repairs(REPAIRS/'validation',recovery.collect_repairs())
            reports=recovery.final_reports('validation',plan)
            best=selection.choose_best({k:reports[k] for k in plan['sft_candidates']},plan['minimum_coverage'])
            bounds={k:{'lower':v['evaluation']['confirmed_successes'],
                'upper':v['evaluation']['confirmed_successes']+v['evaluation']['planned']-v['evaluation']['scored']}
                for k,v in reports.items() if k in plan['sft_candidates']}
            sensitive=[k for k,b in bounds.items() if k!=best and b['upper']>=bounds[best]['lower'] and b['upper']>b['lower']]
            write_json(REPAIRS/'selection_uncertainty.json',{'bounds':bounds,'provisional_best':best,'unresolved_could_change_winner':sensitive})
            if sensitive:raise RuntimeError('unresolved validation scores could change winner; selection held')
            selected={'selected':best,'checkpoint':candidates[best],'selected_at':now(),
                'validation':{k:selection.compact(v) for k,v in reports.items()},'test_used_for_selection':False,
                'policy_sha256':file_hash(REPAIRS/'policy.json'),'selection_rule':plan['selection_rule']}
            write_json(REPAIRS/'selection.json',selected)
            write_json(REPAIRS/'selection.sha256.json',{'sha256':file_hash(REPAIRS/'selection.json')})
            recovery.emit(tracker,{'selection':selected})
            write_json(REPAIRS/'status.json',{'stage':'validation_selected','selected':best,'test_started':False,'updated_at':now()})
            failed=False
        except BaseException as exc:
            write_json(REPAIRS/'status.json',{'stage':'needs_attention','error':str(exc),'updated_at':now()});raise
        finally:tracker.finish(failed=failed)


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env',override=False)
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['primary','judge','worker'])
    p.add_argument('--root',type=Path);p.add_argument('--split',default='validation');p.add_argument('--index',type=int);p.add_argument('--control',type=Path)
    a=p.parse_args()
    if a.mode=='worker':
        original=selection.claim
        def claim(path,owner,preferred=None):
            if owner in read_json(a.control/'worker_control.json')['retire_indices']:return None
            return original(path,owner,preferred)
        selection.claim=claim
        if a.root.resolve()==REPAIRS.resolve():recovery.install_snapshot_fix()
        selection.worker(a.root,a.split,a.index)
    elif a.mode=='primary':primary()
    else:judge()

if __name__=='__main__':main()
