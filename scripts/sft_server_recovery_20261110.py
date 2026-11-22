"""Resume held GPU jobs and append bounded, paced rescoring on the new server."""
import argparse,asyncio,fcntl,json,os,shutil,sqlite3,subprocess,sys,time
from pathlib import Path
ROOT=Path('/root/awm-agent-train');sys.path.insert(0,str(ROOT))
from harness_awm.io import read_json,write_json,file_hash,now,digest,verify_seal
from harness_awm.training import selection
from scripts import resume_sft_validation as old
from scripts import sft_judge_recovery as recovery
from scripts.sft_recovery_support import copy_sealed_episode,recovery_judge,install_snapshot_fix
from harness_awm.backend import Budget
from harness_awm.contracts import RunConfig,TaskSpec
from harness_awm.judge_inputs import input_from_manifest
from harness_awm.evaluation import apply_evaluation
from harness_awm.evaluation_metrics import evaluation_score
from harness_awm.reward_policy import settle_reward
from harness_awm.tracking import SwanLabTracker
RUN=ROOT/'runs/sft_server_recovery_20261110_v1';REPAIRS=recovery.OUTPUT

def status(mode,stage,**kw):write_json(RUN/(mode+'_status.json'),{'stage':stage,'at':now(),'pid':os.getpid(),**kw})

def backup_db(src,dst):
 with sqlite3.connect('file:'+str(src)+'?mode=ro',uri=True) as a,sqlite3.connect(dst) as b:a.backup(b)

def worker(args):
 # Keep frozen generation and snapshot rules. Judge is a separate paced phase.
 from harness_awm.runner import EpisodeRunner
 original_run=EpisodeRunner.run
 async def rollout(self,*a,**kw):
  kw['evaluate_after']=False
  return await original_run(self,*a,**kw)
 EpisodeRunner.run=rollout
 original_claim=selection.claim
 def claim(path,owner,preferred=None):
  if owner in read_json(args.control/'worker_control.json')['retire_indices']:return None
  return original_claim(path,owner,preferred)
 selection.claim=claim;install_snapshot_fix();selection.worker(args.root,args.split,args.index)

def gpu():
 with (REPAIRS/'supervisor.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  plan=selection.checked_plan(REPAIRS)
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'GPU already in use'
  q=REPAIRS/'validation/queue.sqlite'
  backup_db(q,RUN/('queue_before_'+str(int(time.time()))+'.sqlite'))
  with sqlite3.connect(q,timeout=60) as db:
   db.execute('BEGIN IMMEDIATE')
   assert not db.execute("SELECT 1 FROM jobs WHERE state IN ('claimed','error')").fetchone(),'ownership audit required'
   held=db.execute("SELECT id,output FROM jobs WHERE state='held_disk'").fetchall()
   assert all(not Path(out).exists() for _,out in held),'held job already has output'
   db.execute("UPDATE jobs SET state='pending',owner=NULL WHERE state='held_disk'")
  write_json(RUN/'gpu_policy.json',{'frozen_plan_sha256':file_hash(REPAIRS/'plan.json'),'judge_deferred':True,'gpu_target_gib':85,'initial_workers':18,'resumed_ids':[x[0] for x in held]})
  tracker=SwanLabTracker(RUN/'tracking',mode='offline',config={'stage':'server-recovery','gpu_target_gib':85})
  try:
   status('gpu','running',resumed=len(held));old.__file__=__file__
   old.run_stage(REPAIRS,plan,'validation',tracker,'new_server_'+str(int(time.time())))
   status('gpu','complete',judge_pending=True)
  finally:tracker.finish()

def rescore():
 with (RUN/'judge.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  plan=selection.checked_plan(REPAIRS)
  cfg=RunConfig.model_validate(plan['config']);cfg=cfg.model_copy(update={'judge_v2':cfg.judge_v2.model_copy(update={'query_rounds':12})})
  runtime=recovery_judge();api=runtime.api_request
  pace={'next':0.0}
  async def paced(*a,**kw):
   await asyncio.sleep(max(0,pace['next']-time.monotonic()))
   try:return await api(*a,**kw)
   finally:pace['next']=time.monotonic()+20
  runtime.api_request=paced
  historical={x['source_episode']:x for x in recovery.collect_repairs()}
  while True:
   records=[];pending=[];skipped=0
   for name in read_json(REPAIRS/'validation/candidates.json'):
    for item in read_json(REPAIRS/'validation'/name/'manifest.json')['candidates']:
     source=Path(item['path']);rp=source/'episode_result.json'
     if not rp.exists():continue
     result=read_json(rp)
     if evaluation_score(result) is not None:continue
     if result['execution_integrity']!='valid' or result['terminal_state_status']!='trusted':skipped+=1;continue
     key=digest([name,item['task_instance_id'],str(source)])[:24];job=RUN/'judge_jobs'/key
     if (job/'recovery.json').exists():records.append(read_json(job/'recovery.json'));continue
     pending.append((name,item,source,job))
   status('judge','rescoring',completed=len(records),remaining=len(pending),untrusted=skipped,concurrency=1,request_interval_seconds=20)
   if not pending:
    gpu_state=read_json(RUN/'gpu_status.json') if (RUN/'gpu_status.json').exists() else {}
    if gpu_state.get('stage')=='complete':break
    if gpu_state.get('stage')=='needs_attention':raise RuntimeError('GPU stage needs inspection')
    time.sleep(20);continue
   if min(shutil.disk_usage(RUN).free,shutil.disk_usage(old.ARCHIVE).free)<2*1024**3:
    status('judge','waiting_disk',reason='less than 2 GiB free',remaining=len(pending));time.sleep(30);continue
   name,item,source,job=pending[0]
   job.mkdir(parents=True,exist_ok=True);target=job/'episode'
   if not target.exists():
    provenance=copy_sealed_episode(source,target);write_json(job/'provenance.json',provenance)
   provenance=read_json(job/'provenance.json');prev=historical.get(str(source));budget_path=job/'budget.sqlite'
   if not budget_path.exists():
    baseline={}
    if prev:
     old_budget=REPAIRS/'judge_jobs_v1'/prev['key']/'budget.sqlite'
     backup_db(old_budget,budget_path);baseline=Budget(budget_path,32,None).summary()
    write_json(job/'baseline_budget.json',baseline)
   baseline=read_json(job/'baseline_budget.json');budget=Budget(budget_path,32,None)
   task=TaskSpec.model_validate(read_json(target/'task.json'));bundle=input_from_manifest(target,task,cfg.judge_inputs_manifest)
   tries=read_json(job/'attempts.json') if (job/'attempts.json').exists() else []
   if len(tries)>=3:
    status('judge','held_rate_limit',reason='Infrastructure retry limit reached',job=str(job),usage=budget.summary());return
   if tries and tries[-1].get('rate_limit'):
    until=tries[-1]['retry_at']
    while time.time()<until:
     status('judge','waiting_rate_limit',retry_at=until,job=str(job),usage=budget.summary());time.sleep(min(30,max(1,until-time.time())))
   async def assess():return await runtime.evaluate_input(bundle,cfg,budget,asyncio.Semaphore(1),target/'evaluations_v2')
   assessment=asyncio.run(assess());errors=' '.join(assessment.get('errors',[]));limited='429' in errors
   delay=300
   if limited:
    import re
    match=re.search(r'try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?',errors)
    if match:delay=max(delay,sum(float(v or 0)*f for v,f in zip(match.groups(),[3600,60,1])))
   tries.append({'at':time.time(),'rate_limit':limited,'retry_at':time.time()+delay,'errors':assessment.get('errors',[])})
   write_json(job/'attempts.json',tries)
   if limited:continue
   outcome=settle_reward(assessment,bundle,task,cfg);write_json(target/'rewards_v2'/(digest(outcome)+'.json'),outcome)
   result=apply_evaluation(target,outcome);verify_seal(target)
   assert file_hash(source/'episode_result.json')==provenance['original_result_sha256']
   cumulative=budget.summary();increment={k:v-baseline.get(k,0) for k,v in cumulative.items()}
   record={'candidate':name,'task_instance_id':task.task_instance_id,'source_episode':str(source),'episode':str(target),
    'score':evaluation_score(result),'errors':assessment['errors'],'usage':increment,'cumulative_usage':cumulative,'at':now(),**provenance}
   write_json(job/'recovery.json',record)
  # Overlay only valid new assessments; all old attempts remain immutable.
  with (REPAIRS/'supervisor.lock').open('a') as lock2:
   fcntl.flock(lock2,fcntl.LOCK_EX|fcntl.LOCK_NB)
   recovery.overlay_repairs(REPAIRS/'validation',records)
   reports=recovery.final_reports('validation',plan)
   for name,report in reports.items():
    for row in records:
     if row['candidate']==name:
      for k,v in row['usage'].items():report['budget'][k]=report['budget'].get(k,0)+v
    write_json(REPAIRS/'validation'/name/'metrics.json',report)
   best=selection.choose_best({k:reports[k] for k in plan['sft_candidates']},plan['minimum_coverage'])
   bounds={k:{'lower':v['evaluation']['confirmed_successes'],'upper':v['evaluation']['confirmed_successes']+v['evaluation']['planned']-v['evaluation']['scored']} for k,v in reports.items() if k in plan['sft_candidates']}
   sensitive=[k for k,b in bounds.items() if k!=best and b['upper']>=bounds[best]['lower'] and b['upper']>b['lower']]
   write_json(REPAIRS/'selection_uncertainty.json',{'bounds':bounds,'provisional_best':best,'unresolved_could_change_winner':sensitive})
   if sensitive:raise RuntimeError('unresolved scores could change winner')
   candidates=read_json(REPAIRS/'validation/candidates.json')
   selected={'selected':best,'checkpoint':candidates[best],'selected_at':now(),'validation':{k:selection.compact(v) for k,v in reports.items()},'test_used_for_selection':False,'selection_rule':plan['selection_rule'],'policy_sha256':file_hash(REPAIRS/'policy.json')}
   write_json(REPAIRS/'selection.json',selected);write_json(REPAIRS/'selection.sha256.json',{'sha256':file_hash(REPAIRS/'selection.json')})
   write_json(REPAIRS/'status.json',{'stage':'validation_selected','selected':best,'updated_at':now()})
   status('judge','complete',selected=best)
  subprocess.run([sys.executable,str(ROOT/'runs/grpo_pipeline_20261110_v1/resume.py'),'resume'],check=True)

if __name__=='__main__':
 from dotenv import load_dotenv
 load_dotenv(ROOT/'.env',override=False)
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['gpu','judge','worker']);p.add_argument('--root',type=Path);p.add_argument('--split',default='validation');p.add_argument('--index',type=int);p.add_argument('--control',type=Path);a=p.parse_args()
 try:
  if a.mode=='worker':worker(a)
  elif a.mode=='gpu':gpu()
  else:rescore()
 except Exception as exc:
  if a.mode!='worker':status(a.mode,'needs_attention',reason=f'{type(exc).__name__}: {exc}')
  raise
