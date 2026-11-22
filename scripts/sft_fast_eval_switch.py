"""Roll workers to verified fast kernels at task boundaries; keep prior evidence."""
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
from harness_awm.training.selection import checked_plan, aggregate, compact, choose_best, prepare_evaluation
from scripts.sft_eval_local_recovery import archive_episode, alive


def retirement_trigger(queue, owners):
    with sqlite3.connect(queue, timeout=60) as db:
        db.execute('DROP TRIGGER IF EXISTS fast_kernel_retirement')
        if owners:
            ids = ','.join(str(int(x)) for x in sorted(owners))
            db.execute(f"""CREATE TRIGGER fast_kernel_retirement BEFORE UPDATE OF state ON jobs
                WHEN OLD.state='pending' AND NEW.state='claimed' AND NEW.owner IN ({ids})
                BEGIN SELECT RAISE(ABORT,'planned_fast_kernel_switch_at_task_boundary'); END""")


def verify_benchmarks(paths):
    results = [read_json(p) for p in paths]
    if len(results) != 5 or any(not r['all_outputs_identical'] for r in results):
        raise ValueError('every candidate must pass paired token-output compatibility probes')
    if len({r['checkpoint']['sha256'] for r in results}) != 5:
        raise ValueError('five distinct checkpoint probes required')
    if any(not all(v['fast'] for v in r['bindings'].values()) for r in results):
        raise ValueError('incomplete kernel combination')
    return [{'path': str(p), 'sha256':file_hash(p), 'checkpoint':r['checkpoint']['sha256'],
             'paired_wall_speedup':r['median_paired_wall_speedup'], 'pairs':len(r['pairs'])}
            for p,r in zip(paths,results)]


def execute(root, archive, probe_paths):
    plan = checked_plan(root)
    probes = verify_benchmarks(probe_paths)
    with (root / 'kernel_supervisor.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'kernel_switch.json').exists():
            raise ValueError('kernel switch already started')
        previous = read_json(root / 'validation/launch.json')
        old_pids = dict(enumerate(previous['worker_pids']))
        if len(old_pids) != plan['eval_workers']:
            raise ValueError('worker count changed')
        with sqlite3.connect(root / 'validation/queue.sqlite') as db:
            snapshot = [dict(zip(('id','candidate','task','output','state','owner'),r))
                        for r in db.execute('SELECT id,candidate,task,output,state,owner FROM jobs')]
        manifest = {'at':now(), 'pid':os.getpid(), 'old_workers':previous,
            'old_supervisor':read_json(root/'launch.json'), 'old_tracking':read_json(root/'tracking.json'),
            'reference_queue_snapshot':snapshot, 'probes':probes, 'plan_sha256':file_hash(root/'plan.json'),
            'driver_sha256':file_hash(__file__), 'worker_sha256':file_hash(ROOT/'scripts/sft_fast_eval_worker.py'),
            'policy':'same weights/BF16/SDPA/greedy/EOS/limits/Judge; enable FLA 0.6.0 and causal-conv1d 1.7.0 at task boundaries',
            'past_outcomes_retained':True, 'model_failures_retried':False,
            'limitations':'Finite compatibility probes passed; floating-point kernels are not guaranteed identical for every future input. Runtime is annotated per new attempt.'}
        write_json(root/'kernel_switch.json',manifest)
        write_json(root/'kernel_switch.sha256.json',{'sha256':file_hash(root/'kernel_switch.json')})
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT/'runs/sft_eval_kernel_deps'),str(ROOT/'runs/sft_kernel_deps')]),
            SWANLAB_MODE='disabled', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
            PYTORCH_ALLOC_CONF='expandable_segments:True', TOKENIZERS_PARALLELISM='false',
            HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
        tracker = SwanLabTracker(root/'fast_kernel_tracking', mode='online', config={
            'stage':'sft-fast-kernel-continuation','kernel_policy':manifest['policy'],'compatibility_probes':probes,
            'past_outcomes_retained':True,'model_failures_retried':False,'eval_workers':plan['eval_workers']})
        failed = True
        lock = (root/'supervisor.lock').open('a')
        acquired = False
        try:
            retirement_trigger(root/'validation/queue.sqlite',old_pids)
            def drive(split, old, step):
                nonlocal acquired
                stage = root/split
                processes,logs,archived = {},[],set()
                remaining = dict(old)
                last_report = 0
                try:
                    while True:
                        for index in range(plan['eval_workers']):
                            if index in processes:continue
                            if index in remaining and alive(remaining[index]):continue
                            with sqlite3.connect(stage/'queue.sqlite') as db:
                                if db.execute("SELECT count(*) FROM jobs WHERE state='claimed' AND owner=?",(index,)).fetchone()[0]:
                                    raise RuntimeError(f'old worker {index} exited with an unfinished claim')
                            remaining.pop(index,None)
                            retirement_trigger(stage/'queue.sqlite',remaining)
                            log=(stage/f'worker_fast_{index:02d}.log').open('xb');logs.append(log)
                            processes[index]=subprocess.Popen([sys.executable,str(ROOT/'scripts/sft_fast_eval_worker.py'),
                                '--output',str(root),'--split',split,'--index',str(index)],cwd=ROOT,env=env,
                                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                            write_json(root/'kernel_transition_status.json',{'stage':'switching' if remaining else 'fast_workers_started',
                                'updated_at':now(),'old_workers_remaining':remaining,
                                'fast_workers':{str(k):p.pid for k,p in processes.items()},'swanlab_url':tracker.url})
                        if not remaining and not acquired:
                            try:
                                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                                acquired=True
                                write_json(root/'tracking.json',{'url':tracker.url,'previous_url':manifest['old_tracking']['url']})
                                write_json(root/'launch.json',{'pid':os.getpid(),'started_at':now(),'mode':'fast_kernel_continuation'})
                            except BlockingIOError:pass
                        if not remaining:
                            write_json(stage/'launch.json',{'worker_pids':[processes[i].pid for i in sorted(processes)],
                                'updated_at':now(),'mode':'fast_kernel_continuation'})
                        exits={i:p.poll() for i,p in processes.items()}
                        if any(x not in (None,0) for x in exits.values()):
                            raise RuntimeError(f'fast worker failure: {exits}')
                        if acquired and time.monotonic()-last_report>25:
                            with sqlite3.connect(stage/'queue.sqlite') as db:
                                counts=dict(db.execute('SELECT state,count(*) FROM jobs GROUP BY state'))
                                done=db.execute("SELECT output FROM jobs WHERE state='done'").fetchall()
                            for (output,) in done:
                                if output not in archived:
                                    folder=Path(output)/'episodes/t00000/a0'
                                    archive_episode(folder,archive/folder.relative_to(root));archived.add(output)
                            reports=aggregate(stage,plan)
                            if split=='validation':
                                for name,budget in read_json(root/'original_budgets.json').items():
                                    for key,value in budget.items():
                                        reports[name]['budget'][key]=reports[name]['budget'].get(key,0)+value
                                    write_json(stage/name/'metrics.json',reports[name])
                            status={'stage':split,'updated_at':now(),'kernel_mode':'fla-and-causal-conv1d',
                                'queue':counts,'live_workers':sum(x is None for x in exits.values()),
                                'worker_exit_codes':[exits[i] for i in sorted(exits)],'archived_tasks':len(archived),
                                'candidates':{k:compact(v) for k,v in reports.items()}}
                            write_json(root/'status.json',status)
                            tracker.log({split:status['candidates'],'queue':counts,
                                'judge_budget':{k:v['budget'] for k,v in reports.items()},'live_workers':status['live_workers']},step=step)
                            step+=1;last_report=time.monotonic()
                            import shutil
                            if shutil.disk_usage(root).free < 1024**3:
                                with sqlite3.connect(stage/'queue.sqlite') as db:
                                    db.execute("UPDATE jobs SET state='held_disk' WHERE state='pending'")
                            if not status['live_workers']:
                                if any(v for k,v in counts.items() if k!='done'):
                                    raise RuntimeError('incomplete queue; outcomes retained')
                                return reports,step
                        time.sleep(2)
                except BaseException:
                    with sqlite3.connect(stage/'queue.sqlite') as db:
                        db.execute("UPDATE jobs SET state='held_kernel_failure' WHERE state='pending'")
                    retirement_trigger(stage/'queue.sqlite',[])
                    # Existing requests finish naturally; no forced process kill.
                    raise
                finally:
                    for log in logs:log.close()
            reports,step=drive('validation',old_pids,0)
            candidates=read_json(root/'validation/candidates.json')
            best=choose_best({k:reports[k] for k in plan['sft_candidates']},plan['minimum_coverage'])
            selected={'selected':best,'checkpoint':candidates[best],'selected_at':now(),
                'validation':{k:compact(v) for k,v in reports.items()},'test_used_for_selection':False,
                'selection_rule':plan['selection_rule'],'kernel_switch_manifest_sha256':file_hash(root/'kernel_switch.json')}
            write_json(root/'selection.json',selected)
            write_json(root/'selection.sha256.json',{'sha256':file_hash(root/'selection.json')})
            prepare_evaluation(root,plan,{'base':candidates['base'],best:candidates[best]},'test')
            test,step=drive('test',{},step)
            summary={'selected':best,'checkpoint':candidates[best],'stop_policy':plan['stop_policy'],
                'baseline':{'validation':compact(reports['base']),'test':compact(test['base'])},
                'validation':compact(reports[best]),'test':compact(test[best]),'test_used_for_selection':False,
                'completed_at':now(),'swanlab_url':tracker.url,'kernel_switch_manifest_sha256':file_hash(root/'kernel_switch.json')}
            write_json(root/'summary.json',summary);tracker.log({'summary':summary},step=step)
            write_json(root/'status.json',{'stage':'finished','selected':best,'updated_at':now()});failed=False
        except BaseException as exc:
            write_json(root/'kernel_transition_status.json',{'stage':'failed','error':f'{type(exc).__name__}: {exc}','updated_at':now()})
            if acquired:write_json(root/'status.json',{'stage':'failed','error':str(exc),'updated_at':now()})
            raise
        finally:
            lock.close();tracker.finish(failed=failed)


if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env',override=False)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--probe',type=Path,action='append',required=True)
    a=p.parse_args()
    execute(a.output.resolve(),a.archive.resolve(),[x.resolve() for x in a.probe])
