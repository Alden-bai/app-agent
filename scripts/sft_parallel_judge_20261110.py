"""Four concurrent evidence assessments with one persistent provider rate gate."""
import asyncio,fcntl,json,math,os,re,shutil,signal,sqlite3,sys,time
from pathlib import Path
ROOT=Path('/root/awm-agent-train');sys.path.insert(0,str(ROOT))
from scripts import sft_server_recovery_20261110 as r
from harness_awm.protocol import estimate_tokens
RUN=r.RUN
DRAIN=False

def quota_tokens(messages,output_tokens):
 import tiktoken
 enc=tiktoken.get_encoding("o200k_base")
 prompt=sum(len(enc.encode(m.get("content", ""),disallowed_special=()))+8 for m in messages)+16
 return math.ceil(prompt*1.10)+output_tokens+512

class RateGate:
 def __init__(self,path,tpm=85000,rpm=12):
  self.path=Path(path);self.tpm=tpm;self.rpm=rpm;self.lock=asyncio.Lock()
  with sqlite3.connect(self.path) as db:
   db.execute('CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,started REAL,tokens INTEGER,status TEXT)')
   db.execute('CREATE TABLE IF NOT EXISTS control(id INTEGER PRIMARY KEY,until REAL)')
   db.execute('INSERT OR IGNORE INTO control VALUES(1,0)')
 def delay(self,now,upper):
  if upper>self.tpm:raise RuntimeError(f'request_reservation_exceeds_rate_gate:{upper}')
  with sqlite3.connect(self.path) as db:
   until=db.execute('SELECT until FROM control WHERE id=1').fetchone()[0]
   rows=db.execute('SELECT started,tokens FROM events WHERE started>? ORDER BY started',(now-60,)).fetchall()
  if until>now:return until-now
  if len(rows)>=self.rpm or sum(x[1] for x in rows)+upper>self.tpm:return max(.05,rows[0][0]+60.1-now)
  return 0
 async def acquire(self,upper):
  while True:
   async with self.lock:
    wait=self.delay(time.time(),upper)
    if not wait:
     with sqlite3.connect(self.path) as db:
      cur=db.execute("INSERT INTO events(started,tokens,status) VALUES(?,?,'reserved')",(time.time(),upper));return cur.lastrowid
   await asyncio.sleep(min(wait,20))
 def finish(self,event,usage=None):
  with sqlite3.connect(self.path) as db:
   if isinstance(usage,int) and usage>=0:db.execute("UPDATE events SET tokens=?,status='response' WHERE id=?",(usage,event))
   else:db.execute("UPDATE events SET status='unknown_reserved' WHERE id=?",(event,))
 def cooldown(self,until):
  with sqlite3.connect(self.path) as db:db.execute('UPDATE control SET until=max(until,?) WHERE id=1',(until,))
 def snapshot(self):
  now=time.time()
  with sqlite3.connect(self.path) as db:
   count,total=db.execute('SELECT count(*),coalesce(sum(tokens),0) FROM events WHERE started>?',(now-60,)).fetchone()
   until=db.execute('SELECT until FROM control WHERE id=1').fetchone()[0]
  return {'last_60s_requests':count,'last_60s_tokens':total,'cooldown_until':until,'tpm':self.tpm,'rpm':self.rpm}

def retry_delay(errors):
 m=re.search(r'try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?',str(errors))
 return max(300,sum(float(v or 0)*f for v,f in zip(m.groups(),[3600,60,1]))) if m else 300

def discover():
 pending=[];records=[];untrusted=0
 for name in r.read_json(r.REPAIRS/'validation/candidates.json'):
  for item in r.read_json(r.REPAIRS/'validation'/name/'manifest.json')['candidates']:
   source=Path(item['path']);rp=source/'episode_result.json'
   if not rp.exists():continue
   result=r.read_json(rp)
   if r.evaluation_score(result) is not None:continue
   if result['execution_integrity']!='valid' or result['terminal_state_status']!='trusted':untrusted+=1;continue
   key=r.digest([name,item['task_instance_id'],str(source)])[:24];job=RUN/'judge_jobs'/key
   if (job/'recovery.json').exists():records.append(r.read_json(job/'recovery.json'))
   else:pending.append((name,item,source,job))
 return pending,records,untrusted

async def assess_one(row,cfg,runtime,historical,gate):
 name,item,source,job=row;job.mkdir(parents=True,exist_ok=True);target=job/'episode'
 if not target.exists():r.write_json(job/'provenance.json',r.copy_sealed_episode(source,target))
 provenance=r.read_json(job/'provenance.json');prev=historical.get(str(source));bp=job/'budget.sqlite'
 if not bp.exists():
  baseline={}
  if prev:
   r.backup_db(r.REPAIRS/'judge_jobs_v1'/prev['key']/'budget.sqlite',bp);baseline=r.Budget(bp,32,None).summary()
  r.write_json(job/'baseline_budget.json',baseline)
 baseline=r.read_json(job/'baseline_budget.json');budget=r.Budget(bp,32,None)
 task=r.TaskSpec.model_validate(r.read_json(target/'task.json'));bundle=r.input_from_manifest(target,task,cfg.judge_inputs_manifest)
 tries=r.read_json(job/'attempts.json') if (job/'attempts.json').exists() else []
 while True:
  if len(tries)>=3:raise RuntimeError('infrastructure_retry_limit:'+str(job))
  if tries and tries[-1].get('rate_limit'):gate.cooldown(tries[-1]['retry_at'])
  assessment=await runtime.evaluate_input(bundle,cfg,budget,asyncio.Semaphore(1),target/'evaluations_v2')
  errors=assessment.get('errors',[]);limited='429' in str(errors)
  tries.append({'at':time.time(),'rate_limit':limited,'retry_at':time.time()+retry_delay(errors),'errors':errors})
  r.write_json(job/'attempts.json',tries)
  if limited:gate.cooldown(tries[-1]['retry_at']);continue
  outcome=r.settle_reward(assessment,bundle,task,cfg);r.write_json(target/'rewards_v2'/(r.digest(outcome)+'.json'),outcome)
  result=r.apply_evaluation(target,outcome);r.verify_seal(target)
  assert r.file_hash(source/'episode_result.json')==provenance['original_result_sha256']
  cumulative=budget.summary();increment={k:v-baseline.get(k,0) for k,v in cumulative.items()}
  record={'candidate':name,'task_instance_id':task.task_instance_id,'source_episode':str(source),'episode':str(target),
   'score':r.evaluation_score(result),'errors':errors,'usage':increment,'cumulative_usage':cumulative,'at':r.now(),**provenance}
  r.write_json(job/'recovery.json',record);return record

async def drive(plan):
 cfg=r.RunConfig.model_validate(plan['config']);cfg=cfg.model_copy(update={'judge_v2':cfg.judge_v2.model_copy(update={'query_rounds':12})})
 gate=RateGate(RUN/'provider_rate.sqlite');runtime=r.recovery_judge();api=runtime.api_request
 # A full window after the handoff accounts conservatively for old-process requests.
 gate.cooldown(r.read_json(RUN/'parallel_handoff.json')['at']+60)
 async def paced(config,kwargs,budget,key,stage,folder,**kw):
  upper=quota_tokens(kwargs['messages'],kwargs.get('max_completion_tokens',kwargs.get('max_tokens',0)))
  event=await gate.acquire(upper)
  try:
   response=await api(config,kwargs,budget,key,stage,folder,**kw)
   gate.finish(event,response.get('usage',{}).get('total_tokens'));return response
  except BaseException as exc:
   gate.finish(event)
   if getattr(exc,'status_code',None)==429:gate.cooldown(time.time()+retry_delay(str(exc)))
   raise
 runtime.api_request=paced
 historical={x['source_episode']:x for x in r.recovery.collect_repairs()}
 fatal=[];active={};records=[];pending=[];untrusted=0
 async def report():
  while True:
   r.status('judge','needs_attention' if fatal else 'rescoring',concurrency=4,completed=len(records),remaining=len(pending)+len(active),active=list(active.values()),untrusted=untrusted,rate=gate.snapshot(),errors=fatal)
   await asyncio.sleep(5)
 async def worker(index):
  while pending and not fatal and not DRAIN:
   if min(shutil.disk_usage(RUN/'judge_jobs').free,shutil.disk_usage(RUN).free)<2*1024**3:
    fatal.append('insufficient_local_disk_less_than_2GiB');return
   row=pending.pop(0);active[index]=str(row[3])
   try:records.append(await assess_one(row,cfg,runtime,historical,gate))
   except Exception as exc:fatal.append(f'{type(exc).__name__}:{exc}')
   finally:active.pop(index,None)
 reporter=asyncio.create_task(report())
 try:
  while True:
   pending,records,untrusted=await asyncio.to_thread(discover)
   await asyncio.gather(*(worker(i) for i in range(4)))
   if fatal:raise RuntimeError(';'.join(fatal))
   if DRAIN:return None
   state=r.read_json(RUN/'gpu_status.json')
   if state['stage']=='complete':
    # Recheck once for tasks completed during this batch.
    pending,records,untrusted=await asyncio.to_thread(discover)
    if not pending:return records
   elif state['stage']=='needs_attention':raise RuntimeError('GPU stage needs inspection')
   await asyncio.sleep(5)
 finally:
  reporter.cancel()
  try:await reporter
  except asyncio.CancelledError:pass

def finalize(plan,records):
 with (r.REPAIRS/'supervisor.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  r.recovery.overlay_repairs(r.REPAIRS/'validation',records)
  reports=r.recovery.final_reports('validation',plan)
  for name,report in reports.items():
   for row in records:
    if row['candidate']==name:
     for k,v in row['usage'].items():report['budget'][k]=report['budget'].get(k,0)+v
   r.write_json(r.REPAIRS/'validation'/name/'metrics.json',report)
  r.write_json(r.REPAIRS/'validation_recovery_budget.json',{k:v['budget'] for k,v in reports.items()})
  best=r.selection.choose_best({k:reports[k] for k in plan['sft_candidates']},plan['minimum_coverage'])
  bounds={k:{'lower':v['evaluation']['confirmed_successes'],'upper':v['evaluation']['confirmed_successes']+v['evaluation']['planned']-v['evaluation']['scored']} for k,v in reports.items() if k in plan['sft_candidates']}
  sensitive=[k for k,b in bounds.items() if k!=best and b['upper']>=bounds[best]['lower'] and b['upper']>b['lower']]
  r.write_json(r.REPAIRS/'selection_uncertainty.json',{'bounds':bounds,'provisional_best':best,'unresolved_could_change_winner':sensitive})
  if sensitive:raise RuntimeError('unresolved scores could change winner')
  candidates=r.read_json(r.REPAIRS/'validation/candidates.json')
  selected={'selected':best,'checkpoint':candidates[best],'selected_at':r.now(),'validation':{k:r.selection.compact(v) for k,v in reports.items()},'test_used_for_selection':False,'selection_rule':plan['selection_rule'],'policy_sha256':r.file_hash(r.REPAIRS/'policy.json')}
  r.write_json(r.REPAIRS/'selection.json',selected);r.write_json(r.REPAIRS/'selection.sha256.json',{'sha256':r.file_hash(r.REPAIRS/'selection.json')})
  r.write_json(r.REPAIRS/'status.json',{'stage':'validation_selected','selected':best,'updated_at':r.now()});r.status('judge','complete',selected=best)
 r.subprocess.run([sys.executable,str(ROOT/'runs/grpo_pipeline_20261110_v1/resume.py'),'resume'],check=True)

def main():
 from dotenv import load_dotenv
 load_dotenv(ROOT/'.env',override=False)
 def drain(signum,frame):
  global DRAIN
  DRAIN=True
 signal.signal(signal.SIGUSR1,drain)
 with (RUN/'judge.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  plan=r.selection.checked_plan(r.REPAIRS)
  r.write_json(RUN/'parallel_policy.json',{'concurrency':4,'tokens_per_minute':85000,'requests_per_minute':12,'provider_reported_tpm':100000,'at':r.now(),'script_sha256':r.file_hash(__file__),'frozen_plan_sha256':r.file_hash(r.REPAIRS/'plan.json'),'budget_preserved':True,'rate_estimator':'o200k_base prompt +10% +512 + max_output; request budget unchanged'})
  try:
   records=asyncio.run(drive(plan))
   if records is None:r.status('judge','paused_at_episode_boundary')
   else:finalize(plan,records)
  except BaseException as exc:
   r.status('judge','needs_attention',reason=f'{type(exc).__name__}: {exc}');raise
if __name__=='__main__':main()
