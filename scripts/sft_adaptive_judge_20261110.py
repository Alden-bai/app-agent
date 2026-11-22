"""Persistent 16-to-1 Judge concurrency with one-step 429 backoff."""
import asyncio,fcntl,json,os,re,signal,sqlite3,sys,time
from pathlib import Path
ROOT=Path('/root/awm-agent-train');sys.path.insert(0,str(ROOT))
from scripts import sft_parallel_judge_20261110 as base
r=base.r;RUN=base.RUN
DRAIN=False

def retry_delay(error):
 text=str(error);match=re.search(r'try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?',text,re.I)
 delay=sum(float(v or 0)*f for v,f in zip(match.groups(),[3600,60,1])) if match else 60
 response=getattr(error,'response',None)
 if response is not None:
  value=response.headers.get('retry-after')
  if value:
   try:delay=max(delay,float(value))
   except ValueError:
    from email.utils import parsedate_to_datetime
    try:delay=max(delay,parsedate_to_datetime(value).timestamp()-time.time())
    except (TypeError,ValueError):pass
 return max(30,delay)

class AdaptiveGate(base.RateGate):
 def __init__(self,path):
  super().__init__(path,tpm=95000,rpm=60)
  self.live=set()
  with sqlite3.connect(self.path) as db:
   db.execute('CREATE TABLE IF NOT EXISTS adaptive(id INTEGER PRIMARY KEY,target INTEGER,tpm INTEGER,wave_until REAL,last_limit REAL,started REAL)')
   db.execute('INSERT OR IGNORE INTO adaptive VALUES(1,16,95000,0,0,?)',(time.time(),))
   db.execute('CREATE TABLE IF NOT EXISTS backoffs(id INTEGER PRIMARY KEY,at REAL,old_target INTEGER,new_target INTEGER,tpm INTEGER,until REAL,reason TEXT)')
  self.refresh()
 def refresh(self):
  with sqlite3.connect(self.path) as db:self.target,self.tpm=self._state(db)[:2]
 def _state(self,db):return db.execute('SELECT target,tpm,wave_until,last_limit,started FROM adaptive WHERE id=1').fetchone()
 async def acquire(self,upper):
  while True:
   async with self.lock:
    self.refresh()
    if len(self.live)>=self.target:wait=.2
    else:
     # A large single request can use the provider's declared ceiling exclusively.
     if upper>100000:raise RuntimeError('request_exceeds_provider_TPM:'+str(upper))
     original=self.tpm;self.tpm=max(self.tpm,upper)
     try:wait=self.delay(time.time(),upper)
     finally:self.tpm=original
     if not wait:
      with sqlite3.connect(self.path) as db:
       cur=db.execute("INSERT INTO events(started,tokens,status) VALUES(?,?,'reserved')",(time.time(),upper));event=cur.lastrowid
      self.live.add(event);return event
   await asyncio.sleep(min(wait,2))
 def finish(self,event,usage=None):
  super().finish(event,usage);self.live.discard(event)
 def limited(self,reason,until,now=None):
  now=time.time() if now is None else now
  with sqlite3.connect(self.path,timeout=30) as db:
   db.execute('BEGIN IMMEDIATE');target,tpm,wave,last,started=self._state(db)
   if now>=wave:
    lowered=max(1,target-1);new_tpm=max(50000,int(tpm*.9))
    db.execute('UPDATE adaptive SET target=?,tpm=?,wave_until=?,last_limit=? WHERE id=1',(lowered,new_tpm,until,now))
    db.execute('INSERT INTO backoffs(at,old_target,new_target,tpm,until,reason) VALUES(?,?,?,?,?,?)',(now,target,lowered,new_tpm,until,reason))
   else:db.execute('UPDATE adaptive SET wave_until=max(wave_until,?),last_limit=? WHERE id=1',(until,now))
   db.execute('UPDATE control SET until=max(until,?) WHERE id=1',(until,))
  self.refresh()
 def snapshot(self):
  self.refresh();result=super().snapshot();now=time.time()
  with sqlite3.connect(self.path) as db:
   target,tpm,wave,last,started=self._state(db)
   successes=db.execute("SELECT count(*) FROM events WHERE status='response' AND started>=?",(max(last,started),)).fetchone()[0]
   transitions=db.execute('SELECT old_target,new_target,at FROM backoffs ORDER BY id').fetchall()
  result.update(target_concurrency=target,inflight_requests=len(self.live),stable=now-max(last,started)>=600 and successes>=20,successful_requests_since_backoff=successes,backoffs=transitions)
  return result

async def request_with_backoff(api,gate,config,kwargs,budget,key,stage,folder,**kw):
 upper=base.quota_tokens(kwargs['messages'],kwargs.get('max_completion_tokens',kwargs.get('max_tokens',0)))
 for attempt in range(32):
  event=await gate.acquire(upper)
  call_key=key if attempt==0 else key+f'/rate_retry_{attempt:02d}'
  call_folder=Path(folder) if attempt==0 else Path(folder)/f'rate_retry_{attempt:02d}'
  try:
   response=await api(config,kwargs,budget,call_key,stage,call_folder,**kw)
   gate.finish(event,response.get('usage',{}).get('total_tokens'))
   if attempt:r.write_json(Path(folder)/'successful_rate_retry.json',{'attempt':attempt,'request_key':call_key,'evidence':str(call_folder),'conversation_preserved':True})
   return response
  except BaseException as exc:
   gate.finish(event)
   if getattr(exc,'status_code',None)!=429:raise
   gate.limited('HTTP 429; evidence: '+str(call_folder/'error.json'),time.time()+retry_delay(exc))
 # Normally the unchanged 32-request per-episode budget stops attempts first.
 raise RuntimeError('rate_retry_request_budget_exhausted')

async def drive(plan):
 cfg=r.RunConfig.model_validate(plan['config']);cfg=cfg.model_copy(update={'judge_v2':cfg.judge_v2.model_copy(update={'query_rounds':12})})
 gate=AdaptiveGate(RUN/'provider_rate.sqlite');runtime=r.recovery_judge();api=runtime.api_request
 base.retry_delay=retry_delay
 async def paced(config,kwargs,budget,key,stage,folder,**kw):
  return await request_with_backoff(api,gate,config,kwargs,budget,key,stage,folder,**kw)
 runtime.api_request=paced
 historical={x['source_episode']:x for x in r.recovery.collect_repairs()}
 fatal=[];active={};records=[];pending=[];untrusted=0
 async def report():
  while True:
   rate=gate.snapshot();stage='needs_attention' if fatal else ('draining' if DRAIN else ('waiting_rate_limit' if rate['cooldown_until']>time.time() else 'rescoring'))
   r.status('judge',stage,concurrency=rate['target_concurrency'],initial_concurrency=16,completed=len(records),remaining=len(pending)+len(active),active=list(active.values()),untrusted=untrusted,rate=rate,errors=fatal)
   await asyncio.sleep(5)
 async def worker(index):
  while pending and not fatal and not DRAIN:
   gate.refresh()
   if index>=gate.target:return
   if min(base.shutil.disk_usage(RUN/'judge_jobs').free,base.shutil.disk_usage(RUN).free)<2*1024**3:
    fatal.append('insufficient_local_disk_less_than_2GiB');return
   row=pending.pop(0);active[index]=str(row[3])
   try:records.append(await base.assess_one(row,cfg,runtime,historical,gate))
   except Exception as exc:fatal.append(f'{type(exc).__name__}:{exc}')
   finally:active.pop(index,None)
 reporter=asyncio.create_task(report())
 try:
  while True:
   pending,records,untrusted=await asyncio.to_thread(base.discover)
   await asyncio.gather(*(worker(i) for i in range(gate.target)))
   if fatal:raise RuntimeError(';'.join(fatal))
   if DRAIN:return None
   state=r.read_json(RUN/'gpu_status.json')
   if state['stage']=='complete':
    pending,records,untrusted=await asyncio.to_thread(base.discover)
    if not pending:return records
   elif state['stage']=='needs_attention':raise RuntimeError('GPU stage needs inspection')
   await asyncio.sleep(5)
 finally:
  reporter.cancel()
  try:await reporter
  except asyncio.CancelledError:pass

def main():
 from dotenv import load_dotenv
 load_dotenv(ROOT/'.env',override=False)
 def drain(signum,frame):
  global DRAIN
  DRAIN=True
 signal.signal(signal.SIGUSR1,drain)
 r.write_json(RUN/'adaptive_launch.json',{'pid':os.getpid(),'at':r.now(),'stage':'waiting_for_previous_judge','requested_concurrency':16})
 with (RUN/'judge.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  plan=r.selection.checked_plan(r.REPAIRS)
  r.write_json(RUN/'adaptive_policy.json',{'initial_concurrency':16,'minimum_concurrency':1,'backoff':'one level per 429 wave, preserve in-flight work, no automatic increase','stable_definition':'10 minutes without 429 and at least 20 successful requests','initial_tpm':95000,'rpm':60,'provider_reported_tpm':100000,'at':r.now(),'script_sha256':r.file_hash(__file__),'frozen_plan_sha256':r.file_hash(r.REPAIRS/'plan.json'),'budget_preserved':True,'rate_estimator':'o200k_base prompt +10% +512 + max_output; request budget unchanged','physical_request_budget_unchanged':True,'rate_retry':'same dialogue; unique physical request keys; bounded by existing 32-request episode budget'})
  r.write_json(RUN/'adaptive_launch.json',{'pid':os.getpid(),'at':r.now(),'stage':'running','requested_concurrency':16})
  try:
   records=asyncio.run(drive(plan))
   if records is None:r.status('judge','paused_at_episode_boundary')
   else:base.finalize(plan,records)
  except BaseException as exc:
   r.status('judge','needs_attention',reason=f'{type(exc).__name__}: {exc}');raise
if __name__=='__main__':main()
