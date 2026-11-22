import asyncio,sqlite3,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts.sft_adaptive_judge_20261110 import AdaptiveGate,retry_delay,request_with_backoff

class AdaptiveJudgeTest(unittest.TestCase):
 def test_one_step_per_wave_persists_without_resetting_usage(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'rate.sqlite';gate=AdaptiveGate(path)
   event=asyncio.run(gate.acquire(100));gate.finish(event,60)
   gate.limited('429',160,now=100)
   self.assertEqual(gate.target,15)
   gate.limited('same wave',170,now=101)
   self.assertEqual(gate.target,15)
   resumed=AdaptiveGate(path);self.assertEqual(resumed.target,15)
   self.assertEqual(resumed.snapshot()['last_60s_tokens'],60)
   resumed.limited('next wave',250,now=171)
   self.assertEqual(resumed.target,14)
   for n in range(20):resumed.limited('later wave',400+n*100,now=300+n*100)
   self.assertEqual(resumed.target,1)
 def test_request_slots_enforce_lowered_concurrency(self):
  with tempfile.TemporaryDirectory() as d:
   gate=AdaptiveGate(Path(d)/'rate.sqlite')
   with sqlite3.connect(gate.path) as db:db.execute('UPDATE adaptive SET target=1')
   async def run():
    first=await gate.acquire(100)
    second=asyncio.create_task(gate.acquire(100));await asyncio.sleep(.02)
    self.assertFalse(second.done());gate.finish(first,50)
    next_id=await asyncio.wait_for(second,1);gate.finish(next_id,50)
   asyncio.run(run())
 def test_429_retries_same_dialogue_with_unique_accounting_keys(self):
  class Limited(Exception):status_code=429
  class Gate:
   def __init__(self):self.events=[];self.drops=[]
   async def acquire(self,n):self.events.append(n);return len(self.events)
   def finish(self,*a):pass
   def limited(self,*a):self.drops.append(a)
  with tempfile.TemporaryDirectory() as d:
   calls=[];gate=Gate();budget=object();kwargs={'messages':[{'role':'user','content':'test'}],'max_completion_tokens':10}
   async def api(config,wire,ledger,key,stage,folder,**kw):
    calls.append((wire,ledger,key,folder))
    if len(calls)==1:raise Limited('try again in 1s')
    return {'usage':{'total_tokens':20},'choices':[]}
   with patch('scripts.sft_adaptive_judge_20261110.base.quota_tokens',return_value=100):
    response=asyncio.run(request_with_backoff(api,gate,None,kwargs,budget,'key','judge',Path(d)))
   self.assertEqual(len(gate.drops),1);self.assertEqual(len(calls),2)
   self.assertIs(calls[0][0],calls[1][0]);self.assertIs(calls[1][1],budget)
   self.assertNotEqual(calls[0][2],calls[1][2]);self.assertNotEqual(calls[0][3],calls[1][3])
   self.assertEqual(response['usage']['total_tokens'],20)
 def test_service_retry_after_is_honored(self):
  error=RuntimeError('try again in 1m2s');error.response=SimpleNamespace(headers={'retry-after':'120'})
  self.assertEqual(retry_delay(error),120)
  self.assertEqual(retry_delay('try again in 2h'),7200)
  self.assertEqual(retry_delay('unknown'),60)
if __name__=='__main__':unittest.main()
