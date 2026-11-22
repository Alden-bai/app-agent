"""One uniform repair pass for unscored validation/test; immutable source evidence."""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dotenv import load_dotenv
from harness_awm.io import read_json,write_json,file_hash,now,digest,verify_seal
from harness_awm.contracts import RunConfig
from harness_awm.backend import Budget
from harness_awm.judge_inputs import input_from_manifest
from harness_awm.reward_policy import settle_reward
from harness_awm.evaluation import apply_evaluation
from harness_awm.evaluation_metrics import evaluation_score
from harness_awm.training.tasks import load_tasks
from harness_awm.training import selection
from harness_awm.tracking import SwanLabTracker
from scripts.audit_sft_unscored import inspect_result
from scripts.sft_recovery_support import recovery_judge,copy_sealed_episode,install_snapshot_fix
from scripts.sft_eval_local_recovery import archive_episode,alive
SOURCE=ROOT/'runs/qwen35_sft_eval_local_20261109_v1'
OUTPUT=ROOT/'runs/sft_judge_recovery_20261109_v1'
ARCHIVE=Path('/autodl-fs/data/awm-sft-judge-recovery-20261109')


def repair_key(row):
    return digest({'candidate':row['candidate'],'task':row['task_instance_id'],'episode':row['episode']})[:24]


def repair_one(row,plan,root=OUTPUT,archive=ARCHIVE):
    key=repair_key(row);job=root/'judge_jobs_v1'/key;record=job/'recovery.json'
    if record.exists():return read_json(record)
    if job.exists():raise RuntimeError('interrupted repair requires audit before retry: '+key)
    source=Path(row['episode']);target=job/'episode'
    provenance=copy_sealed_episode(source,target)
    write_json(job/'started.json',{'at':now(),'source':row,**provenance})
    config=RunConfig.model_validate(plan['config'])
    config=config.model_copy(update={'judge_v2':config.judge_v2.model_copy(update={'query_rounds':12})})
    from harness_awm.contracts import TaskSpec
    task=TaskSpec.model_validate(read_json(target/'task.json'))
    bundle=input_from_manifest(target,task,config.judge_inputs_manifest)
    budget=Budget(job/'budget.sqlite',32,None)
    async def run():
        return await recovery_judge().evaluate_input(bundle,config,budget,asyncio.Semaphore(1),target/'evaluations_v2')
    assessment=asyncio.run(run())
    outcome=settle_reward(assessment,bundle,task,config)
    write_json(target/'rewards_v2'/f'{digest(outcome)}.json',outcome)
    result=apply_evaluation(target,outcome)
    verify_seal(target)
    if file_hash(source/'episode_result.json')!=provenance['original_result_sha256'] or file_hash(source/'evidence_seal.json')!=provenance['original_seal_sha256']:
        raise RuntimeError('source changed during repair')
    value={'key':key,'candidate':row['candidate'],'task_instance_id':row['task_instance_id'],
        'source_episode':str(source),'episode':str(target),'reason':row['kind'],'at':now(),'split':row.get('split','validation'),
        'in_previous_116':row.get('in_previous_116',False),'evaluation_status':result['evaluation_status'],
        'task_status':result.get('task_status'),'score':evaluation_score(result),'errors':assessment['errors'],
        'usage':budget.summary(),**provenance}
    archive_episode(target,archive/'judge_jobs_v1'/key/'episode')
    write_json(record,value)
    return value


def closed_records(source,split):
    stage=source/split
    with sqlite3.connect(stage/'queue.sqlite') as db:
        counts=dict(db.execute('SELECT state,count(*) FROM jobs GROUP BY state'))
        done={str(Path(r[0])/'episodes/t00000/a0') for r in db.execute("SELECT output FROM jobs WHERE state='done'")}
    rows=[]
    for name in read_json(stage/'candidates.json'):
        for item in read_json(stage/name/'manifest.json')['candidates']:
            folder=Path(item['path'])
            if str(folder) not in done and not item.get('reused_original_attempt'):continue
            try:
                result=read_json(folder/'episode_result.json')
                if evaluation_score(result) is None:
                    rows.append({'candidate':name,'task_instance_id':item['task_instance_id'],'episode':str(folder),'split':split,**inspect_result(folder)})
            except FileNotFoundError:
                # Archiver atomically replaces a completed directory with its verified link.
                continue
    return rows,counts


def available(row):
    return row['execution_integrity']=='valid' and row['terminal_state_status']=='trusted'


def collect_repairs(root=OUTPUT):
    return [read_json(p) for p in sorted((root/'judge_jobs_v1').glob('*/recovery.json'))]


def monitor_repairs(source,split,plan,tracker=None,root=OUTPUT,archive=ARCHIVE,wait_primary=False):
    futures={};seen=set();step=0
    original={r['episode'] for r in read_json(OUTPUT/'initial_audit.json')['attempts'] if r['in_previous_116']}
    with ThreadPoolExecutor(max_workers=4) as pool:
        while True:
            rows,counts=closed_records(source,split)
            for row in rows:
                key=repair_key(row)
                if key in seen or not available(row):continue
                row['in_previous_116']=row['episode'] in original
                if (root/'judge_jobs_v1'/key/'recovery.json').exists():seen.add(key);continue
                # Keep memory and disk bounded; discover later records on subsequent scans.
                if len(futures)>=4:break
                futures[pool.submit(repair_one,row,plan,root,archive)]=row;seen.add(key)
            for future in list(futures):
                if future.done():
                    future.result();del futures[future]
            repaired=collect_repairs(root)
            eligible=[r for r in rows if available(r)]
            pending=sum(repair_key(r) not in seen for r in eligible)
            status={'stage':f'{split}_judge_recovery','updated_at':now(),'source_queue':counts,
                'unscored_original':len(rows),'repair_done':len(repaired),'recovered_valid':sum(r['score'] is not None for r in repaired),
                'still_unscored_after_repair':sum(r['score'] is None for r in repaired),
                'repair_active':len(futures),'repair_pending':pending,
                'foreign_key_reruns_waiting':sum(r['kind']=='snapshot_foreign_keys' for r in rows),
                'previous_116_recovered':sum(r['in_previous_116'] and r['score'] is not None for r in repaired)}
            write_json(root/'status.json',status)
            if tracker:emit(tracker,{'recovery':{k:v for k,v in status.items() if isinstance(v,(int,float))},'repaired_by_candidate':repaired_stats(repaired)})
            step+=1
            all_done=not any(v for k,v in counts.items() if k!='done')
            if all_done and not futures and not pending:
                if wait_primary:
                    launch=read_json(source/'validation/launch.json')
                    if any(alive(pid) for pid in launch['worker_pids']):time.sleep(10);continue
                    with (source/'supervisor.lock').open('a') as lock:
                        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:time.sleep(10);continue
                return rows
            if any(counts.get(x,0) for x in ('error','held_disk','held_storage')):
                raise RuntimeError('source queue needs inspection; no automatic selection')
            time.sleep(15)


def write_manifest(stage,name,manifest):
    write_json(stage/name/'manifest.json',manifest)
    write_json(stage/name/'manifest.sha256.json',{'sha256':file_hash(stage/name/'manifest.json')})


def overlay_repairs(stage,repairs):
    by_source={r['source_episode']:r for r in repairs if r['score'] is not None}
    for name in read_json(stage/'candidates.json'):
        manifest=read_json(stage/name/'manifest.json')
        for item in manifest['candidates']:
            if item['path'] in by_source:
                r=by_source[item['path']]
                item.update(path=r['episode'],repaired_original_attempt=r['source_episode'],reused_original_attempt=r['source_episode'])
        write_manifest(stage,name,manifest)


def prepare_validation(plan,rows):
    stage=OUTPUT/'validation';stage.mkdir(exist_ok=False)
    candidates=read_json(SOURCE/'validation/candidates.json');write_json(stage/'candidates.json',candidates)
    fk={(r['candidate'],r['task_instance_id']):r for r in rows if r['kind']=='snapshot_foreign_keys'}
    jobs=[]
    for name in candidates:
        manifest=read_json(SOURCE/'validation'/name/'manifest.json')
        for index,item in enumerate(manifest['candidates']):
            old=item['path'];item['reused_original_attempt']=old
            if (name,item['task_instance_id']) in fk:
                output=stage/name/'jobs'/f't{index:05d}'
                item.update(path=str(output/'episodes/t00000/a0'),original_interrupted_attempt=old)
                item.pop('reused_original_attempt',None)
                jobs.append((name,item['task_instance_id'],str(output)))
        manifest.update(recovery_source=str(SOURCE),recovery_policy_sha256=file_hash(OUTPUT/'policy.json'))
        write_manifest(stage,name,manifest)
    overlay_repairs(stage,collect_repairs())
    jobs.sort(key=lambda j:(Path(j[2]).name,j[0]));selection.init_queue(stage/'queue.sqlite',jobs)
    write_json(OUTPUT/'foreign_key_reruns.json',list(fk.values()))
    return candidates


def run_gpu_stage(plan,split,tracker):
    stage=OUTPUT/split;processes=[];logs=[];archived=set();step=100000 if split=='validation' else 200000
    env=dict(os.environ,SWANLAB_MODE='disabled',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        PYTORCH_ALLOC_CONF='expandable_segments:True',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    try:
        with sqlite3.connect(stage/'queue.sqlite') as db:n=db.execute("SELECT count(*) FROM jobs WHERE state='pending'").fetchone()[0]
        for index in range(min(18,n)):
            log=(stage/f'worker_{index:02d}.log').open('xb');logs.append(log)
            processes.append(subprocess.Popen([sys.executable,__file__,'worker','--split',split,'--index',str(index)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT))
            time.sleep(1)
        write_json(stage/'launch.json',{'worker_pids':[p.pid for p in processes],'started_at':now()})
        while True:
            exits=[p.poll() for p in processes]
            with sqlite3.connect(stage/'queue.sqlite') as db:
                counts=dict(db.execute('SELECT state,count(*) FROM jobs GROUP BY state'))
                done=db.execute("SELECT output FROM jobs WHERE state='done'").fetchall()
            for (output,) in done:
                if output in archived:continue
                folder=Path(output)/'episodes/t00000/a0'
                archive_episode(folder,ARCHIVE/folder.relative_to(OUTPUT));archived.add(output)
            reports=selection.aggregate(stage,plan)
            status={'stage':split+'_model_runs','updated_at':now(),'queue':counts,'live_workers':sum(x is None for x in exits),'candidates':{k:selection.compact(v) for k,v in reports.items()}}
            write_json(OUTPUT/'status.json',status)
            emit(tracker,{split:status['candidates'],'queue':counts,'live_workers':status['live_workers']});step+=1
            if not status['live_workers']:
                if any(x!=0 for x in exits) or any(v for k,v in counts.items() if k!='done'):raise RuntimeError('recovery model queue incomplete')
                return
            if shutil.disk_usage(OUTPUT).free<1024**3:
                with sqlite3.connect(stage/'queue.sqlite') as db:db.execute("UPDATE jobs SET state='held_disk' WHERE state='pending'")
            time.sleep(20)
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill()
        for log in logs:log.close()


def emit(tracker,data):
    step=getattr(tracker,'recovery_step',0)
    tracker.log(data,step=step);tracker.recovery_step=step+1


def repaired_stats(rows):
    values={}
    for row in rows:
        group=values.setdefault(row['candidate'],{'attempts':0,'valid':0,'confirmed_successes':0,'eval_reward_sum':0})
        group['attempts']+=1;group['valid']+=row['score'] is not None
        group['confirmed_successes']+=row['score']==1
        group['eval_reward_sum']+=row['score'] or 0
    return values


def final_reports(split,plan):
    reports=selection.aggregate(OUTPUT/split,plan)
    for name,report in reports.items():
        additions=[]
        if split=='validation':additions.append(read_json(SOURCE/split/name/'metrics.json')['budget'])
        additions.extend(r['usage'] for r in collect_repairs() if r['candidate']==name and r['split']==split)
        for values in additions:
            for key,value in values.items():report['budget'][key]=report['budget'].get(key,0)+value
        write_json(OUTPUT/split/name/'metrics.json',report)
    write_json(OUTPUT/f'{split}_recovery_budget.json',{name:r['budget'] for name,r in reports.items()})
    return reports


def execute():
    OUTPUT.mkdir(exist_ok=True);ARCHIVE.mkdir(exist_ok=True)
    with (OUTPUT/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=selection.checked_plan(SOURCE)
        policy={'created_at':now(),'source':str(SOURCE),'query_rounds':12,'review_query_rounds':4,
            'max_judge_calls':32,'format_repairs':2,'concurrency':4,'pilot_results_excluded':True,
            'rule':'Keep every valid original result. One judge-only fallback per unscored trusted episode, regardless of outcome. One new rollout only for proven post-tool snapshot foreign-key interruption. Same policy for base/all SFT/test. Preserve rubric, weights, decoding and split. Never select by test.',
            'files':{str(Path(p)):file_hash(ROOT/p) for p in ['scripts/recovery_judge_runtime.py','scripts/sft_recovery_support.py','scripts/sft_judge_recovery.py']}}
        if (OUTPUT/'launch.json').exists():raise RuntimeError('already launched; inspect existing controller before restart')
        write_json(OUTPUT/'policy.json',policy)
        plan={**plan,'judge_recovery_policy_sha256':file_hash(OUTPUT/'policy.json'),'judge_recovery_source':str(SOURCE)}
        write_json(OUTPUT/'plan.json',plan);write_json(OUTPUT/'plan.sha256.json',{'sha256':file_hash(OUTPUT/'plan.json')})
        # Existing supervisor cannot launch a premature test: prepare_evaluation refuses this directory.
        gate=SOURCE/'test';gate.mkdir(exist_ok=False)
        write_json(gate/'JUDGE_RECOVERY_PENDING.json',{'controller':str(OUTPUT),'reason':'Selection/test handed over until all uniform repairs settle.','at':now()})
        write_json(OUTPUT/'launch.json',{'pid':os.getpid(),'at':now()})
        tracker=SwanLabTracker(OUTPUT,mode='online',config={'stage':'sft-judge-recovery','policy':policy})
        write_json(OUTPUT/'tracking.json',{'url':tracker.url});failed=True
        try:
            rows=monitor_repairs(SOURCE,'validation',plan,tracker,wait_primary=True)
            candidates=prepare_validation(plan,rows)
            run_gpu_stage(plan,'validation',tracker)
            monitor_repairs(OUTPUT,'validation',plan,tracker)
            overlay_repairs(OUTPUT/'validation',collect_repairs())
            reports=final_reports('validation',plan)
            emit(tracker,{'validation':{k:selection.compact(v) for k,v in reports.items()},'budget':{k:v['budget'] for k,v in reports.items()}})
            best=selection.choose_best({k:reports[k] for k in plan['sft_candidates']},plan['minimum_coverage'])
            bounds={k:{'lower':v['evaluation']['confirmed_successes'],'upper':v['evaluation']['confirmed_successes']+v['evaluation']['planned']-v['evaluation']['scored']} for k,v in reports.items() if k in plan['sft_candidates']}
            sensitive=[k for k,b in bounds.items() if k!=best and b['upper']>=bounds[best]['lower'] and b['upper']>b['lower']]
            write_json(OUTPUT/'selection_uncertainty.json',{'bounds':bounds,'provisional_best':best,'unresolved_could_change_winner':sensitive})
            if sensitive:raise RuntimeError('unresolved validation scores could change winner; test held for evidence review')
            selected={'selected':best,'checkpoint':candidates[best],'selected_at':now(),'validation':{k:selection.compact(v) for k,v in reports.items()},'test_used_for_selection':False,'policy_sha256':file_hash(OUTPUT/'policy.json'),'selection_rule':plan['selection_rule']}
            write_json(OUTPUT/'selection.json',selected);write_json(OUTPUT/'selection.sha256.json',{'sha256':file_hash(OUTPUT/'selection.json')})
            selection.prepare_evaluation(OUTPUT,plan,{'base':candidates['base'],best:candidates[best]},'test')
            run_gpu_stage(plan,'test',tracker)
            monitor_repairs(OUTPUT,'test',plan,tracker)
            overlay_repairs(OUTPUT/'test',collect_repairs())
            test=final_reports('test',plan)
            summary={'selected':best,'validation':selected['validation'],'test':{k:selection.compact(v) for k,v in test.items()},'test_used_for_selection':False,'completed_at':now(),'swanlab_url':tracker.url}
            write_json(OUTPUT/'summary.json',summary);emit(tracker,{'summary':summary})
            write_json(OUTPUT/'status.json',{'stage':'finished','selected':best,'updated_at':now()});failed=False
        except BaseException as exc:
            write_json(OUTPUT/'status.json',{'stage':'needs_attention','error':f'{type(exc).__name__}: {exc}','updated_at':now()});raise
        finally:tracker.finish(failed=failed)


def main():
    load_dotenv(ROOT/'.env',override=False)
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['pilot','run','worker']);p.add_argument('--split',default='validation');p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.mode=='worker':
        policy=read_json(OUTPUT/'policy.json')
        if any(file_hash(ROOT/k)!=v for k,v in policy['files'].items()):raise ValueError('recovery implementation changed')
        install_snapshot_fix();selection.worker(OUTPUT,a.split,a.index)
    elif a.mode=='run':execute()
    else:
        plan=selection.checked_plan(SOURCE);rows=read_json(OUTPUT/'initial_audit.json')['attempts'];picked=[]
        for kind in ['query_round_budget_exhausted','required_evidence_incomplete','other_judge_error','rubric_error']:
            picked.append(next(r for r in rows if r['kind']==kind and available(r)))
        with ThreadPoolExecutor(max_workers=4) as pool:
            for result in pool.map(lambda r:repair_one(r,plan),picked):print(result,flush=True)

if __name__=='__main__':main()
