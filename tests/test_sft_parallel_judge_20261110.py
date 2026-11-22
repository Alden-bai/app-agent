import asyncio,sqlite3,tempfile,time,unittest
from pathlib import Path
from scripts.sft_parallel_judge_20261110 import RateGate,retry_delay

class RateGateTest(unittest.TestCase):
 def test_concurrent_reservations_share_one_window(self):
  with tempfile.TemporaryDirectory() as d:
   gate=RateGate(Path(d)/'rate.sqlite',tpm=1000,rpm=4)
   async def reserve():return await asyncio.gather(*(gate.acquire(200) for _ in range(4)))
   ids=asyncio.run(reserve())
   self.assertEqual(len(set(ids)),4)
   self.assertEqual(gate.snapshot()['last_60s_tokens'],800)
   self.assertGreater(gate.delay(time.time(),100),0)
   self.assertEqual(gate.delay(time.time()+61,100),0)
 def test_restart_preserves_tokens_unknown_usage_and_cooldown(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'rate.sqlite';gate=RateGate(path,tpm=1000)
   a=asyncio.run(gate.acquire(600));gate.finish(a,400)
   b=asyncio.run(gate.acquire(500));gate.finish(b)
   resumed=RateGate(path,tpm=1000)
   self.assertEqual(resumed.snapshot()['last_60s_tokens'],900)
   self.assertGreater(resumed.delay(time.time(),200),0)
   resumed.cooldown(time.time()+300)
   self.assertGreater(RateGate(path,tpm=1000).delay(time.time(),50),290)
 def test_oversized_request_and_retry_after(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(RuntimeError):RateGate(Path(d)/'r',1000).delay(time.time(),1001)
  self.assertEqual(retry_delay('429 try again in 2h3m4s'),7384)
  self.assertEqual(retry_delay('429'),300)
if __name__=='__main__':unittest.main()
